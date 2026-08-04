"""Adaptive Paper Trader — main orchestrator.

Runs multiple STRATEGIES concurrently against the same market data, each
with its own independent paper ledger (separate SQLite file), risk manager,
and learning state — so they can be fairly compared against each other.
Only the primary strategy (adaptive_ml) is eligible for real Bybit orders;
every other strategy is permanently paper-only, regardless of config.

Async tasks:
  * one analyzer loop per symbol (fetch -> run every strategy -> maybe open)
  * one monitor loop (SL/TP/adaptive exits, journal, learning updates)
  * dashboard web server

Paper trading is always simulated for every strategy except adaptive_ml.
adaptive_ml places real Bybit orders ONLY when ALL of these hold (see
app/trading/executor.py): config live.enabled=true, BYBIT_API_KEY/SECRET
present, and env LIVE_CONFIRM=YES. Otherwise the executor runs in dry-run:
it logs the order it would send, sends nothing. Real orders are
notional-capped by live.max_notional_usd.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass, field

import numpy as np

from app.analysis.engine import pack_entry_features, unpack_entry_features
from app.analysis.model import load_learner_state
from app.analysis.reasoning import analysis_reasoning
from app.analysis.walkforward import resolve_barrier
from app.config import load_config
from app.dashboard.feed import ActivityFeed
from app.dashboard.learning_phases import dots, phase_for_count, phase_info
from app.dashboard.server import start_dashboard
from app.data.collector import MarketDataCollector
from app.db.database import Database
from app.db.retrieval import blend_confidence, retrieve_similar
from app.features.engine import FeatureVector
from app.journal.review import write_review
from app.logging_setup import get_logger, setup_logging
from app.strategies.ml_adapter import MLStrategyAdapter
from app.strategies.rule_based import (
    BreakoutStrategy, MeanReversionStrategy, MomentumStrategy, OrderFlowStrategy,
    XSectionMomentumStrategy,
)
from app.telegram.notifier import TelegramNotifier
from app.trading.executor import BybitExecutor
from app.trading.monitor import TradeMonitor
from app.trading.paper_engine import PaperTradingEngine
from app.trading.risk import RiskManager

LEARNER_STATE_PATH = "data/learner_state.json"
log = get_logger("main")


def _short(sym: str) -> str:
    """BTC/USDT:USDT -> BTC (for human-readable feed messages)."""
    return (sym or "").split("/")[0]


@dataclass
class StrategyRuntime:
    """Everything one strategy needs to run independently: its own paper
    ledger, risk manager, monitor, and learning state. Strategies never
    share a database — that's what keeps their comparison honest (each
    literally trades its own book) and lets every existing Database/
    PaperTradingEngine/dashboard-metrics method work unchanged, just
    pointed at a different file."""
    id: str
    label: str
    strategy: object          # MLStrategyAdapter | RuleBasedStrategy subtype
    db: Database
    risk: RiskManager
    paper: PaperTradingEngine
    monitor: TradeMonitor
    executor: BybitExecutor | None = None
    supports_live: bool = False
    supports_notifications: bool = False
    last_shadow_regime: dict = field(default_factory=dict)
    shadow_since_save: int = 0
    analysis_count: dict = field(default_factory=dict)


class App:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        setup_logging(self.cfg.get("logging.level", "INFO"),
                      bool(self.cfg.get("logging.json", False)))
        self.collector = MarketDataCollector(
            self.cfg.get("exchange.id"),
            self.cfg.get("exchange.timeframe"),
            int(self.cfg.get("exchange.candles_lookback", 600)),
            market_type=str(self.cfg.get("exchange.market_type", "swap")),
        )
        self.tg = TelegramNotifier(self.cfg)
        self.feed = ActivityFeed()
        self._stop = asyncio.Event()
        # Single running counter of ALL engine "work units" across every
        # strategy -- analyze() cycles plus shadow-label resolutions -- for
        # the dashboard heartbeat.
        self._activity_count = 0
        self._shadow_enabled = bool(self.cfg.get("shadow.enabled", True))
        self._shadow_horizon_ms = float(
            self.cfg.get("shadow.horizon_hours", 48)) * 3600_000.0
        self._shadow_max_resolve = int(
            self.cfg.get("shadow.max_resolve_per_cycle", 80))
        self._analyses_keep_rows = int(self.cfg.get("retention.analyses_max_rows", 60000))
        excluded = set(self.cfg.get("exchange.excluded_symbols", []) or [])
        self._symbols = [s for s in self.cfg.get("exchange.symbols")
                          if s not in excluded]

        self.strategies: dict[str, StrategyRuntime] = {}
        self._build_adaptive_ml()
        if bool(self.cfg.get("strategies.momentum.enabled", True)):
            self._build_rule_strategy(
                MomentumStrategy, "data/paper_trader_momentum.db",
                "data/strategy_state_momentum.json")
        if bool(self.cfg.get("strategies.mean_reversion.enabled", True)):
            self._build_rule_strategy(
                MeanReversionStrategy, "data/paper_trader_mean_reversion.db",
                "data/strategy_state_mean_reversion.json")
        if bool(self.cfg.get("strategies.breakout.enabled", True)):
            self._build_rule_strategy(
                BreakoutStrategy, "data/paper_trader_breakout.db",
                "data/strategy_state_breakout.json")
        if bool(self.cfg.get("strategies.order_flow.enabled", True)):
            self._build_rule_strategy(
                OrderFlowStrategy, "data/paper_trader_order_flow.db",
                "data/strategy_state_order_flow.json")
        if bool(self.cfg.get("strategies.xsect_momentum.enabled", True)):
            self._build_rule_strategy(
                XSectionMomentumStrategy, "data/paper_trader_xsect_momentum.db",
                "data/strategy_state_xsect_momentum.json")

    # ------------------------------------------------------------------
    def _build_adaptive_ml(self) -> None:
        n_features = len(FeatureVector.names())
        model, calibration = load_learner_state(
            LEARNER_STATE_PATH, n_features,
            lr=self.cfg.get("learner.lr"), l2=self.cfg.get("learner.l2"),
            decay=self.cfg.get("learner.decay"))
        strategy = MLStrategyAdapter(self.cfg, model, calibration, LEARNER_STATE_PATH)
        db = Database(self.cfg.get("database.path"))
        executor = BybitExecutor(self.cfg)
        max_pos = self.cfg.get(f"strategies.{strategy.id}.max_open_positions")
        risk = RiskManager(self.cfg, max_open_positions=max_pos)
        paper = PaperTradingEngine(self.cfg, db, risk, min_confidence=strategy.min_conf)
        monitor = TradeMonitor(self.cfg, db, self.collector, paper)
        self.strategies[strategy.id] = StrategyRuntime(
            id=strategy.id, label=strategy.label, strategy=strategy, db=db,
            risk=risk, paper=paper, monitor=monitor, executor=executor,
            supports_live=True, supports_notifications=True)

    def _build_rule_strategy(self, cls, db_path: str, state_path: str) -> None:
        strategy = cls(self.cfg, state_path)
        db = Database(db_path)
        max_pos = self.cfg.get(f"strategies.{strategy.id}.max_open_positions")
        risk = RiskManager(self.cfg, max_open_positions=max_pos)
        paper = PaperTradingEngine(self.cfg, db, risk, min_confidence=strategy.min_conf)
        monitor = TradeMonitor(self.cfg, db, self.collector, paper)
        self.strategies[strategy.id] = StrategyRuntime(
            id=strategy.id, label=strategy.label, strategy=strategy, db=db,
            risk=risk, paper=paper, monitor=monitor, executor=None,
            supports_live=False, supports_notifications=False)

    @property
    def primary(self) -> StrategyRuntime:
        return self.strategies["adaptive_ml"]

    # ------------------------------------------------------------------
    async def analyzer_loop(self, symbol: str, initial_delay: float = 0.0) -> None:
        interval = float(self.cfg.get("engine.analysis_interval_seconds", 300))
        warmup = int(self.cfg.get("engine.warmup_candles", 250))

        # All symbol loops otherwise start at the exact same instant and each
        # sleeps the full `interval` after processing, so without this they'd
        # run in permanent lockstep. Staggering start times spreads analyses
        # evenly across the interval instead.
        if initial_delay:
            await _wait(self._stop, initial_delay)

        while not self._stop.is_set():
            try:
                snap = await self.collector.fetch_snapshot(symbol)
                if snap.n < warmup:
                    log.warning("warmup", symbol=symbol, have=snap.n, need=warmup)
                else:
                    for rt in self.strategies.values():
                        await self._run_cycle(rt, symbol, snap)
            except Exception as e:
                log.error("analyzer_error", symbol=symbol, error=str(e), exc_info=True)
            await _wait(self._stop, interval)

    async def _run_cycle(self, rt: StrategyRuntime, symbol: str, snap) -> None:
        k = int(self.cfg.get("retrieval.k", 15))
        min_hist = int(self.cfg.get("retrieval.min_history", 8))
        res = rt.strategy.analyze(snap)

        # episodic memory: similar setups + blended confidence, scoped to
        # THIS strategy's own trade history (separate db per strategy).
        retrieval = await retrieve_similar(
            rt.db, res.features, symbol, k=k, same_direction=res.direction)
        res.confidence = round(blend_confidence(
            res.confidence, retrieval, min_hist), 4)
        if res.trade_recommended and res.confidence < rt.strategy.min_conf:
            res.trade_recommended = False
            res.veto_reasons.append(
                f"blended confidence {res.confidence:.2f} "
                f"< {rt.strategy.min_conf:.2f} (similar-history drag)")
            self.feed.say("veto", (
                f"🧠 [{rt.label}] {_short(symbol)}: a {res.direction} setup "
                f"looked promising, but memory of similar past setups dragged "
                f"confidence down to {res.confidence:.0%} — standing aside."), symbol)
            res.direction = None

        reasoning = ""
        if rt.supports_notifications:
            reasoning = await analysis_reasoning(self.cfg, res, retrieval)
            drift_dir = ("long" if res.bias == "bullish"
                         else "short" if res.bias == "bearish" else None)
            rt.monitor.push_context(symbol, res.changepoint_prob, drift_dir, res.confidence)

        await rt.db.insert_analysis({
            "symbol": symbol, "price": res.price, "bias": res.bias,
            "confidence": res.confidence, "raw_prob": res.raw_prob,
            "regime_label": res.regime_label,
            "regime_posterior": res.regime_posterior,
            "changepoint_prob": res.changepoint_prob,
            "features": res.features.as_dict(),
            "trade_recommended": res.trade_recommended,
            "reasoning": reasoning,
        })

        if self._shadow_enabled:
            self._activity_count += await self._process_shadows(rt, snap, res)

        c = rt.analysis_count.get(symbol, 0) + 1
        rt.analysis_count[symbol] = c
        self._activity_count += 1
        if rt.supports_notifications and (res.trade_recommended or c % 12 == 1):
            await self.tg.analysis_update(res, reasoning)

        if res.trade_recommended:
            trade_id, why = await rt.paper.maybe_open(res, reasoning, retrieval)
            if trade_id:
                t = await rt.db.get_trade(trade_id)
                risk_amt = float(t.get("risk_amount") or 0)
                self.feed.say("open", (
                    f"🚀 [{rt.label}] Opened a paper {str(res.direction).upper()} "
                    f"on {_short(symbol)} at {res.price:g} — "
                    f"{res.confidence:.0%} confident. Risking "
                    f"${risk_amt:.2f} to make about "
                    f"${risk_amt * (res.rr or 0):.2f}."), symbol)
                if rt.supports_live:
                    await self._mirror_open(rt, t, res)
                    await self.tg.trade_opened(
                        trade_id, res, t["position_size"], t["risk_amount"], reasoning)
            else:
                self.feed.say("skip", (
                    f"⏸ [{rt.label}] Liked a {res.direction} on {_short(symbol)} "
                    f"but held back: {why}."), symbol)
                log.info("entry_skipped", strategy=rt.id, symbol=symbol, why=why)

    async def _process_shadows(self, rt: StrategyRuntime, snap, res) -> int:
        """Record the current setup as a shadow label, then resolve any matured
        shadow setups for this symbol against the forward price path (did TP hit
        before SL?) and train this strategy's learner on the clean outcome.

        Returns the number of shadow setups resolved this call, so the caller
        can fold it into the engine activity counter (dashboard heartbeat)."""
        sym = snap.symbol
        # Only take one shadow sample per regime-run per symbol: regime.sticky
        # implies a regime persists for ~1/(1-sticky) candles, so recording
        # every cycle treats dozens of highly correlated observations of the
        # same market state as independent training evidence (pseudo-
        # replication) and inflates apparent confidence beyond real sample size.
        # Rule-based strategies have no regime model (regime_label is a fixed
        # string), so also key on shadow_direction -- otherwise the "same
        # state" check never re-arms after the very first cycle per symbol
        # and they'd stop recording shadow setups forever.
        state_key = f"{res.regime_label}|{res.shadow_direction}"
        regime_changed = rt.last_shadow_regime.get(sym) != state_key
        rt.last_shadow_regime[sym] = state_key
        d, sl, tp = res.shadow_direction, res.shadow_stop, res.shadow_take_profit
        if (regime_changed and d in ("long", "short") and sl and tp
                and np.isfinite(sl) and np.isfinite(tp) and sl != res.price):
            await rt.db.record_shadow_setup({
                "symbol": sym, "direction": d, "entry_price": res.price,
                "entry_ms": int(snap.ts[-1]), "stop_loss": sl,
                "take_profit": tp,
                "features": pack_entry_features(res.features, res.norm_snapshot)})

        ts, hi, lo = snap.ts, snap.high, snap.low
        latest_ms = float(ts[-1])
        oldest_ms = float(ts[0])
        learned = 0
        tp_hits = 0
        for s in await rt.db.open_shadow_setups(sym, self._shadow_max_resolve):
            e_ms = float(s["entry_ms"])
            # forward-only barrier resolution -- strictly ts > entry_ms, so
            # no future info leaks into the label.
            outcome = resolve_barrier(e_ms, s["direction"],
                                      float(s["stop_loss"]), float(s["take_profit"]),
                                      ts, hi, lo)
            if outcome is not None:
                try:
                    feats, norm_snap = unpack_entry_features(json.loads(s["features"]))
                    rt.strategy.learn_from_shadow(
                        feats, s["direction"], bool(outcome),
                        symbol=sym, norm_snapshot=norm_snap)
                except Exception as e:
                    log.error("shadow_learn_failed", strategy=rt.id, id=s["id"], error=str(e))
                await rt.db.resolve_shadow_setup(s["id"], 1, outcome)
                learned += 1
                tp_hits += 1 if outcome else 0
            elif (latest_ms - e_ms) >= self._shadow_horizon_ms or e_ms < oldest_ms:
                await rt.db.resolve_shadow_setup(s["id"], 2, None)

        if learned:
            self.feed.say("learn", (
                f"📚 [{rt.label}] {_short(sym)}: checked {learned} practice setup"
                f"{'s' if learned != 1 else ''} against what price actually "
                f"did — {tp_hits} would have hit take-profit, "
                f"{learned - tp_hits} the stop. Lessons learned so far: "
                f"{rt.strategy.n_updates:,}."), sym)
            await self._check_learning_phase(rt, sym, learned)
            rt.shadow_since_save += learned
            if rt.shadow_since_save >= 20:   # persist state periodically
                rt.strategy.save()
                await rt.db.prune_shadow_setups()
                await rt.db.prune_analyses(self._analyses_keep_rows)
                rt.shadow_since_save = 0
                self.feed.say("save", (
                    f"💾 [{rt.label}] Progress saved — everything learned "
                    f"({rt.strategy.n_updates:,} lessons) is now safe on disk."))
            log.info("shadow_resolved", strategy=rt.id, symbol=sym, learned=learned)
        return learned

    async def _check_learning_phase(self, rt: StrategyRuntime, sym: str,
                                    resolved_this_call: int) -> None:
        """Advance and announce this symbol's learning phase for this
        strategy (see app/dashboard/learning_phases.py). One announcement
        per phase, ever — the counter is persisted in this strategy's own db
        so pruning shadow_setups can't re-trigger it."""
        row = await rt.db.bump_symbol_learning(sym, resolved_this_call)
        new_phase = phase_for_count(row["resolved_count"])
        if new_phase > row["phase"]:
            await rt.db.set_symbol_phase(sym, new_phase)
            emoji, label = phase_info(new_phase)
            self.feed.say("milestone", (
                f"{emoji} [{rt.label}] {_short(sym)} reached learning phase "
                f"{new_phase}/3 — {label} {dots(new_phase)} "
                f"({row['resolved_count']:,} outcomes studied so far)"), sym)

    async def _mirror_open(self, rt: StrategyRuntime, t: dict, res) -> None:
        """Mirror a paper open onto Bybit. Only ever called for the primary
        (adaptive_ml) strategy — see rt.supports_live. When live, persist the
        order state BEFORE sending (so a crash mid-send is recoverable) then
        link the exchange id — this is what lets reconcile_on_startup work."""
        executor = rt.executor
        if not executor.live:
            try:
                await executor.open_position(t)   # dry-run: logs only
            except Exception as e:
                log.error("live_open_failed", trade_id=t["id"], error=str(e))
            return
        oid = f"t{t['id']}"
        qty = executor.live_qty(t)
        await rt.db.record_live_order({
            "order_id": oid, "trade_id": t["id"], "symbol": t["symbol"],
            "direction": t["direction"], "entry_price": t["entry_price"],
            "quantity": qty if qty is not None else t["position_size"],
            "order_status": "pending",
            "regime_state_at_entry": {"label": res.regime_label,
                                      "posterior": res.regime_posterior,
                                      "changepoint_prob": res.changepoint_prob},
            "prediction_state_at_entry": {"raw_prob": res.raw_prob,
                                          "confidence": res.confidence,
                                          "rr": res.rr},
        })
        await rt.db.add_order_event(oid, "created", {"symbol": t["symbol"]})
        try:
            bybit_id = await executor.open_position(t)
            await rt.db.set_live_order(oid, status="open", bybit_order_id=bybit_id or "")
            await rt.db.add_order_event(oid, "entry_filled", {"bybit_order_id": bybit_id})
        except Exception as e:
            await rt.db.set_live_order(oid, status="unknown_state")
            await rt.db.add_order_event(oid, "open_error", {"error": str(e)})
            log.error("live_open_failed", trade_id=t["id"], symbol=t["symbol"],
                      error=str(e), hint="paper opened but live order failed")

    async def reconcile_on_startup(self) -> None:
        """3-way check on restart: persisted live orders vs Bybit positions.
        Only the primary strategy ever has live orders."""
        rt = self.primary
        open_orders = await rt.db.open_live_orders()
        if not open_orders:
            return
        log.info("reconcile_start", n=len(open_orders), live=rt.executor.live)
        if not rt.executor.live:
            log.warning("reconcile_skipped", reason="not live — leaving persisted "
                        "orders untouched", n=len(open_orders))
            return
        try:
            positions = await rt.executor.fetch_open_positions()
        except Exception as e:
            log.error("reconcile_bybit_error", error=str(e),
                      hint="cannot contact Bybit — manual review advised")
            return
        for o in open_orders:
            oid, sym = o["order_id"], o["symbol"]
            pos = positions.get(sym)
            if pos is None:
                await rt.db.set_live_order(oid, status="unknown_state")
                await rt.db.add_order_event(oid, "reconcile_mismatch",
                                            {"note": "position not on Bybit"})
                log.warning("reconcile_missing", order_id=oid, symbol=sym)
                continue
            pos_qty = abs(float(pos.get("contracts") or 0))
            side_ok = pos.get("side") == o["direction"]
            qty_ok = abs(pos_qty - float(o["quantity"])) <= max(
                0.02 * float(o["quantity"]), 1e-8)
            if side_ok and qty_ok:
                await rt.db.set_live_order(oid, status="open")
                await rt.db.add_order_event(oid, "reconcile_ok",
                                            {"qty": pos_qty, "side": pos.get("side")})
                log.info("reconcile_ok", order_id=oid, symbol=sym)
            else:
                await rt.db.add_order_event(oid, "reconcile_mismatch", {
                    "db_qty": o["quantity"], "bybit_qty": pos_qty,
                    "db_side": o["direction"], "bybit_side": pos.get("side")})
                log.warning("reconcile_qty_side_mismatch", order_id=oid, symbol=sym)
        log.info("reconcile_done")

    # ------------------------------------------------------------------
    async def monitor_loop(self) -> None:
        interval = float(self.cfg.get("engine.monitor_interval_seconds", 20))
        while not self._stop.is_set():
            try:
                for rt in self.strategies.values():
                    closed = await rt.monitor.check_all()
                    for trade in closed:
                        await self._handle_closed_trade(rt, trade)
            except Exception as e:
                log.error("monitor_loop_error", error=str(e), exc_info=True)
            await _wait(self._stop, interval)

    async def _handle_closed_trade(self, rt: StrategyRuntime, trade: dict) -> None:
        won = (trade.get("pnl_usd") or 0) > 0
        pnl = float(trade.get("pnl_usd") or 0)
        base = _short(trade.get("symbol", ""))
        self.feed.say("close", (
            f"✅ [{rt.label}] {base} {trade.get('direction')} closed "
            f"({trade.get('exit_reason')}): +${pnl:.2f}.") if won else (
            f"❌ [{rt.label}] {base} {trade.get('direction')} closed "
            f"({trade.get('exit_reason')}): −${abs(pnl):.2f}."), trade.get("symbol"))
        # 1) learning updates FIRST — the paper trade is already terminally
        #    closed, so if a fallible live close ran first and raised, this
        #    outcome would be lost from the model forever.
        try:
            feats, norm_snap = unpack_entry_features(json.loads(trade["features"]))
            rt.strategy.learn_from_trade(
                feats, trade["direction"], won, float(trade["confidence"]),
                exit_reason=trade.get("exit_reason"), symbol=trade.get("symbol"),
                norm_snapshot=norm_snap)
            rt.strategy.save()
        except Exception as e:
            log.error("learning_update_failed", strategy=rt.id,
                      trade_id=trade.get("id"), error=str(e))
        # 2) mirror the close on the exchange (primary strategy only).
        if rt.supports_live:
            try:
                await rt.executor.close_position(trade)
                if rt.executor.live:
                    oid = f"t{trade['id']}"
                    await rt.db.set_live_order(oid, status="closed")
                    await rt.db.add_order_event(oid, "closed", {
                        "exit_reason": trade.get("exit_reason"),
                        "pnl_usd": trade.get("pnl_usd")})
            except Exception as e:
                log.error("live_close_failed", trade_id=trade.get("id"),
                          symbol=trade.get("symbol"), error=str(e),
                          hint="real position may remain open — guarded by "
                               "exchange SL/TP; manual check advised")
        # 3) journal review + Telegram notification (primary strategy only —
        #    LLM journal review is an adaptive_ml-specific feature).
        if rt.supports_notifications:
            calibration_score = getattr(rt.strategy, "calibration", None)
            score = calibration_score.calibration_score() if calibration_score else 0.0
            review, lessons, _ = await write_review(self.cfg, rt.db, trade, score)
            equity = await rt.db.latest_equity(rt.risk.starting_equity)
            await self.tg.trade_closed(trade, lessons, equity)

    # ------------------------------------------------------------------
    async def run(self) -> None:
        for rt in self.strategies.values():
            await rt.db.connect()
            eq = await rt.db.latest_equity(rt.risk.starting_equity)
            if not (await rt.db.equity_curve()):
                await rt.db.record_equity(rt.risk.starting_equity, "start")

        primary = self.primary
        mode = "LIVE" if primary.executor.live else "dry-run (no real orders)"
        strategy_names = ", ".join(rt.label for rt in self.strategies.values())
        primary_eq = await primary.db.latest_equity(primary.risk.starting_equity)
        self.feed.say("sys", (
            f"🟢 Bot is up in {mode} mode — watching {len(self._symbols)} "
            f"markets across {len(self.strategies)} strategies ({strategy_names}), "
            f"each with its own ${primary_eq:,.2f} paper balance. I'll narrate "
            f"here every time any strategy opens, closes, skips a trade, or "
            f"learns from a practice setup."))
        log.info("startup", equity=primary_eq, symbols=self._symbols,
                 strategies=list(self.strategies.keys()),
                 exchange=self.cfg.get("exchange.id"), execution_mode=mode)
        await self.tg.system_event(
            f"Adaptive Paper Trader started — {len(self.strategies)} strategies "
            f"({strategy_names}), equity ${primary_eq:,.2f} each "
            f"({', '.join(self._symbols)}). Execution: {mode}.")

        # bound disk on boot: cap the analyses table to the retention row count
        for rt in self.strategies.values():
            try:
                total = await rt.db.analyses_count()
                dropped = await rt.db.prune_analyses(self._analyses_keep_rows)
                log.info("analyses_retention", strategy=rt.id, total_before=total,
                         dropped=dropped, keep_rows=self._analyses_keep_rows)
            except Exception as e:
                log.error("prune_analyses_error", strategy=rt.id, error=str(e))

        # reconcile persisted live orders against Bybit before trading resumes
        try:
            await self.reconcile_on_startup()
        except Exception as e:
            log.error("reconcile_error", error=str(e), exc_info=True)

        runner = None
        if self.cfg.get("dashboard.enabled"):
            # PORT env (set by Railway & friends) overrides config
            port = int(os.getenv("PORT", self.cfg.get("dashboard.port", 8787)))
            strategies_meta = [
                {"id": rt.id, "label": rt.label, "db": rt.db,
                 "starting_equity": rt.risk.starting_equity,
                 "min_confidence": rt.strategy.min_conf,
                 "max_open_positions": rt.risk.max_positions,
                 "learner_provider": (lambda rt=rt: rt.strategy.snapshot(FeatureVector.names()))}
                for rt in self.strategies.values()
            ]
            runner = await start_dashboard(
                primary.db, primary.risk.starting_equity,
                self.cfg.get("dashboard.host", "0.0.0.0"), port,
                learner_provider=lambda: primary.strategy.snapshot(FeatureVector.names()),
                cycle_count_provider=lambda: self._activity_count,
                cfg=self.cfg, feed=self.feed,
                info={
                    "mode": mode,
                    "live": primary.executor.live,
                    "exchange": self.cfg.get("exchange.id"),
                    "market_type": self.cfg.get("exchange.market_type", "swap"),
                    "symbols": self._symbols,
                    "timeframe": self.cfg.get("exchange.timeframe"),
                    "min_confidence": primary.strategy.min_conf,
                    "max_notional_usd": float(
                        self.cfg.get("live.max_notional_usd", 0)),
                    "strategies": [{"id": rt.id, "label": rt.label}
                                  for rt in self.strategies.values()],
                },
                strategies_meta=strategies_meta,
            )
            log.info("dashboard_up", port=port)

        stagger_interval = float(self.cfg.get("engine.analysis_interval_seconds", 300))
        n = max(1, len(self._symbols))
        tasks = [asyncio.create_task(
                    self.analyzer_loop(s, initial_delay=i * stagger_interval / n))
                 for i, s in enumerate(self._symbols)]
        tasks.append(asyncio.create_task(self.monitor_loop()))

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                pass  # windows

        await self._stop.wait()
        log.info("shutting_down")
        # Log open live orders for recovery — they persist on Bybit (SL/TP
        # active) and reconcile_on_startup will re-check them on next boot.
        try:
            open_live = await primary.db.open_live_orders()
            if open_live:
                log.info("shutdown_open_live_orders",
                         n=len(open_live),
                         orders=[o["order_id"] for o in open_live])
                for o in open_live:
                    await primary.db.add_order_event(
                        o["order_id"], "app_shutdown",
                        {"note": "persists on Bybit; reconcile on next boot"})
        except Exception as e:
            log.error("shutdown_order_log_failed", error=str(e))
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if runner:
            await runner.cleanup()
        await primary.executor.close()
        await self.collector.close()
        for rt in self.strategies.values():
            await rt.db.close()


async def _wait(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


def main() -> None:
    async def _amain() -> None:
        # App() must be constructed inside the running loop: on Python <=3.9
        # asyncio primitives bind to the loop that exists at creation time.
        await App().run()
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
