"""Adaptive Paper Trader — main orchestrator.

Async tasks:
  * one analyzer loop per symbol (fetch -> analyze -> maybe open paper trade)
  * one monitor loop (SL/TP/adaptive exits, journal, learning updates)
  * dashboard web server

Paper trading is always simulated. Real Bybit orders are placed ONLY when
ALL of these hold (see app/trading/executor.py): config live.enabled=true,
BYBIT_API_KEY/BYBIT_API_SECRET present, and env LIVE_CONFIRM=YES. Otherwise
the executor runs in dry-run: it logs the order it would send, sends nothing.
Real orders are notional-capped by live.max_notional_usd.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal

import numpy as np

from app.analysis.engine import AnalysisEngine
from app.analysis.model import load_learner_state, save_learner_state
from app.analysis.reasoning import analysis_reasoning
from app.analysis.walkforward import resolve_barrier
from app.config import load_config
from app.dashboard.feed import ActivityFeed
from app.dashboard.server import start_dashboard
from app.data.collector import MarketDataCollector
from app.db.database import Database
from app.db.retrieval import blend_confidence, retrieve_similar
from app.features.engine import FeatureVector
from app.journal.review import write_review
from app.logging_setup import get_logger, setup_logging
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


class App:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        setup_logging(self.cfg.get("logging.level", "INFO"),
                      bool(self.cfg.get("logging.json", False)))
        self.db = Database(self.cfg.get("database.path"))
        self.collector = MarketDataCollector(
            self.cfg.get("exchange.id"),
            self.cfg.get("exchange.timeframe"),
            int(self.cfg.get("exchange.candles_lookback", 600)),
            market_type=str(self.cfg.get("exchange.market_type", "swap")),
        )
        n_features = len(FeatureVector.names())
        self.model, self.calibration = load_learner_state(
            LEARNER_STATE_PATH, n_features,
            lr=self.cfg.get("learner.lr"), l2=self.cfg.get("learner.l2"),
            decay=self.cfg.get("learner.decay"))
        self.engine = AnalysisEngine(self.cfg, self.model, self.calibration)
        self.risk = RiskManager(self.cfg)
        self.paper = PaperTradingEngine(self.cfg, self.db, self.risk)
        self.executor = BybitExecutor(self.cfg)
        self.monitor = TradeMonitor(self.cfg, self.db, self.collector, self.paper)
        self.tg = TelegramNotifier(self.cfg)
        self.feed = ActivityFeed()
        self._stop = asyncio.Event()
        self._analysis_count: dict[str, int] = {}
        self._shadow_enabled = bool(self.cfg.get("shadow.enabled", True))
        self._shadow_horizon_ms = float(
            self.cfg.get("shadow.horizon_hours", 48)) * 3600_000.0
        self._shadow_max_resolve = int(
            self.cfg.get("shadow.max_resolve_per_cycle", 80))
        self._shadow_since_save = 0
        self._analyses_keep_rows = int(self.cfg.get("retention.analyses_max_rows", 60000))
        excluded = set(self.cfg.get("exchange.excluded_symbols", []) or [])
        self._symbols = [s for s in self.cfg.get("exchange.symbols")
                          if s not in excluded]
        self._last_shadow_regime: dict[str, str] = {}

    # ------------------------------------------------------------------
    async def analyzer_loop(self, symbol: str) -> None:
        interval = float(self.cfg.get("engine.analysis_interval_seconds", 300))
        warmup = int(self.cfg.get("engine.warmup_candles", 250))
        k = int(self.cfg.get("retrieval.k", 15))
        min_hist = int(self.cfg.get("retrieval.min_history", 8))

        while not self._stop.is_set():
            try:
                snap = await self.collector.fetch_snapshot(symbol)
                if snap.n < warmup:
                    log.warning("warmup", symbol=symbol, have=snap.n, need=warmup)
                else:
                    res = self.engine.analyze(snap)

                    # episodic memory: similar setups + blended confidence
                    retrieval = await retrieve_similar(
                        self.db, res.features, symbol, k=k,
                        same_direction=res.direction)
                    res.confidence = round(blend_confidence(
                        res.confidence, retrieval, min_hist), 4)
                    # re-check confidence veto after blending
                    if (res.trade_recommended
                            and res.confidence < self.engine.min_conf):
                        res.trade_recommended = False
                        res.veto_reasons.append(
                            f"blended confidence {res.confidence:.2f} "
                            f"< {self.engine.min_conf:.2f} (similar-history drag)")
                        self.feed.say("veto", (
                            f"🧠 {_short(symbol)}: a {res.direction} setup looked "
                            f"promising, but my memory of similar past setups "
                            f"dragged confidence down to {res.confidence:.0%} — "
                            f"standing aside."), symbol)
                        res.direction = None

                    reasoning = await analysis_reasoning(self.cfg, res, retrieval)

                    # give the monitor live regime context for adaptive exits
                    drift_dir = ("long" if res.bias == "bullish"
                                 else "short" if res.bias == "bearish" else None)
                    self.monitor.push_context(
                        symbol, res.changepoint_prob, drift_dir, res.confidence)

                    await self.db.insert_analysis({
                        "symbol": symbol, "price": res.price, "bias": res.bias,
                        "confidence": res.confidence, "raw_prob": res.raw_prob,
                        "regime_label": res.regime_label,
                        "regime_posterior": res.regime_posterior,
                        "changepoint_prob": res.changepoint_prob,
                        "features": res.features.as_dict(),
                        "trade_recommended": res.trade_recommended,
                        "reasoning": reasoning,
                    })

                    # shadow labels: record this setup + resolve matured ones
                    if self._shadow_enabled:
                        await self._process_shadows(snap, res)

                    # notify analysis (every Nth cycle to avoid spam, always on rec)
                    c = self._analysis_count.get(symbol, 0) + 1
                    self._analysis_count[symbol] = c
                    if res.trade_recommended or c % 12 == 1:
                        await self.tg.analysis_update(res, reasoning)

                    if res.trade_recommended:
                        trade_id, why = await self.paper.maybe_open(
                            res, reasoning, retrieval)
                        if trade_id:
                            t = await self.db.get_trade(trade_id)
                            risk = float(t.get("risk_amount") or 0)
                            self.feed.say("open", (
                                f"🚀 Opened a paper {str(res.direction).upper()} "
                                f"on {_short(symbol)} at {res.price:g} — "
                                f"{res.confidence:.0%} confident. Risking "
                                f"${risk:.2f} to make about "
                                f"${risk * (res.rr or 0):.2f}."), symbol)
                            await self._mirror_open(t, res)
                            await self.tg.trade_opened(
                                trade_id, res, t["position_size"],
                                t["risk_amount"], reasoning)
                        else:
                            self.feed.say("skip", (
                                f"⏸ Liked a {res.direction} on {_short(symbol)} "
                                f"but held back: {why}."), symbol)
                            log.info("entry_skipped", symbol=symbol, why=why)
            except Exception as e:
                log.error("analyzer_error", symbol=symbol, error=str(e), exc_info=True)
            await _wait(self._stop, interval)

    async def _process_shadows(self, snap, res) -> None:
        """Record the current setup as a shadow label, then resolve any matured
        shadow setups for this symbol against the forward price path (did TP hit
        before SL?) and train the learner on the clean barrier outcome."""
        sym = snap.symbol
        # 1) record the current setup (guard degenerate geometry). Only take one
        # shadow sample per regime-run per symbol: regime.sticky implies a regime
        # persists for ~1/(1-sticky) candles, so recording every cycle treats
        # dozens of highly correlated observations of the same market state as
        # independent training evidence (pseudo-replication) and inflates the
        # learner's apparent confidence beyond its real effective sample size.
        regime_changed = self._last_shadow_regime.get(sym) != res.regime_label
        self._last_shadow_regime[sym] = res.regime_label
        d, sl, tp = res.shadow_direction, res.shadow_stop, res.shadow_take_profit
        if (regime_changed and d in ("long", "short") and sl and tp
                and np.isfinite(sl) and np.isfinite(tp) and sl != res.price):
            await self.db.record_shadow_setup({
                "symbol": sym, "direction": d, "entry_price": res.price,
                "entry_ms": int(snap.ts[-1]), "stop_loss": sl,
                "take_profit": tp, "features": {**res.features.as_dict(),
                                                "_norm_snapshot": res.norm_snapshot}})

        # 2) resolve matured setups using this snapshot's OHLC path
        ts, hi, lo = snap.ts, snap.high, snap.low
        latest_ms = float(ts[-1])
        oldest_ms = float(ts[0])
        learned = 0
        tp_hits = 0
        for s in await self.db.open_shadow_setups(sym, self._shadow_max_resolve):
            e_ms = float(s["entry_ms"])
            # forward-only barrier resolution (see walkforward.resolve_barrier —
            # strictly ts > entry_ms, so no future info leaks into the label)
            outcome = resolve_barrier(e_ms, s["direction"],
                                      float(s["stop_loss"]), float(s["take_profit"]),
                                      ts, hi, lo)
            if outcome is not None:
                try:
                    feats = json.loads(s["features"])
                    norm_snap = feats.pop("_norm_snapshot", None)
                    self.engine.learn_from_shadow(
                        feats, s["direction"], bool(outcome),
                        symbol=sym, norm_snapshot=norm_snap)
                except Exception as e:
                    log.error("shadow_learn_failed", id=s["id"], error=str(e))
                await self.db.resolve_shadow_setup(s["id"], 1, outcome)
                learned += 1
                tp_hits += 1 if outcome else 0
            elif (latest_ms - e_ms) >= self._shadow_horizon_ms or e_ms < oldest_ms:
                # neither barrier hit within the horizon / data window -> expire
                await self.db.resolve_shadow_setup(s["id"], 2, None)

        if learned:
            self.feed.say("learn", (
                f"📚 {_short(sym)}: checked {learned} practice setup"
                f"{'s' if learned != 1 else ''} against what price actually "
                f"did — {tp_hits} would have hit take-profit, "
                f"{learned - tp_hits} the stop. Lessons learned so far: "
                f"{self.model.n_updates:,}."), sym)
            self._shadow_since_save += learned
            if self._shadow_since_save >= 20:   # persist learner periodically
                save_learner_state(LEARNER_STATE_PATH, self.model, self.calibration)
                await self.db.prune_shadow_setups()          # bound shadow table
                await self.db.prune_analyses(self._analyses_keep_rows)  # bound disk
                self._shadow_since_save = 0
                self.feed.say("save", (
                    f"💾 Progress saved — everything learned "
                    f"({self.model.n_updates:,} lessons) is now safe on disk."))
            log.info("shadow_resolved", symbol=sym, learned=learned)

    async def _mirror_open(self, t: dict, res) -> None:
        """Mirror a paper open onto Bybit. When live, persist the order state
        BEFORE sending (so a crash mid-send is recoverable) then link the
        exchange id — this is what lets reconcile_on_startup work."""
        if not self.executor.live:
            try:
                await self.executor.open_position(t)   # dry-run: logs only
            except Exception as e:
                log.error("live_open_failed", trade_id=t["id"], error=str(e))
            return
        oid = f"t{t['id']}"
        qty = self.executor.live_qty(t)
        await self.db.record_live_order({
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
        await self.db.add_order_event(oid, "created", {"symbol": t["symbol"]})
        try:
            bybit_id = await self.executor.open_position(t)
            await self.db.set_live_order(oid, status="open",
                                         bybit_order_id=bybit_id or "")
            await self.db.add_order_event(oid, "entry_filled",
                                          {"bybit_order_id": bybit_id})
        except Exception as e:
            await self.db.set_live_order(oid, status="unknown_state")
            await self.db.add_order_event(oid, "open_error", {"error": str(e)})
            log.error("live_open_failed", trade_id=t["id"], symbol=t["symbol"],
                      error=str(e), hint="paper opened but live order failed")

    async def reconcile_on_startup(self) -> None:
        """3-way check on restart: persisted live orders vs Bybit positions.
        Marks vanished positions unknown_state; confirms matches; logs events."""
        open_orders = await self.db.open_live_orders()
        if not open_orders:
            return
        log.info("reconcile_start", n=len(open_orders), live=self.executor.live)
        if not self.executor.live:
            log.warning("reconcile_skipped", reason="not live — leaving persisted "
                        "orders untouched", n=len(open_orders))
            return
        try:
            positions = await self.executor.fetch_open_positions()
        except Exception as e:
            log.error("reconcile_bybit_error", error=str(e),
                      hint="cannot contact Bybit — manual review advised")
            return
        for o in open_orders:
            oid, sym = o["order_id"], o["symbol"]
            pos = positions.get(sym)
            if pos is None:
                await self.db.set_live_order(oid, status="unknown_state")
                await self.db.add_order_event(oid, "reconcile_mismatch",
                                              {"note": "position not on Bybit"})
                log.warning("reconcile_missing", order_id=oid, symbol=sym)
                continue
            pos_qty = abs(float(pos.get("contracts") or 0))
            side_ok = pos.get("side") == o["direction"]
            qty_ok = abs(pos_qty - float(o["quantity"])) <= max(
                0.02 * float(o["quantity"]), 1e-8)
            if side_ok and qty_ok:
                await self.db.set_live_order(oid, status="open")
                await self.db.add_order_event(oid, "reconcile_ok",
                                              {"qty": pos_qty, "side": pos.get("side")})
                log.info("reconcile_ok", order_id=oid, symbol=sym)
            else:
                await self.db.add_order_event(oid, "reconcile_mismatch", {
                    "db_qty": o["quantity"], "bybit_qty": pos_qty,
                    "db_side": o["direction"], "bybit_side": pos.get("side")})
                log.warning("reconcile_qty_side_mismatch", order_id=oid, symbol=sym)
        log.info("reconcile_done")

    # ------------------------------------------------------------------
    async def monitor_loop(self) -> None:
        interval = float(self.cfg.get("engine.monitor_interval_seconds", 20))
        while not self._stop.is_set():
            try:
                closed = await self.monitor.check_all()
                for trade in closed:
                    await self._handle_closed_trade(trade)
            except Exception as e:
                log.error("monitor_loop_error", error=str(e), exc_info=True)
            await _wait(self._stop, interval)

    async def _handle_closed_trade(self, trade: dict) -> None:
        import json as _json
        won = (trade.get("pnl_usd") or 0) > 0
        pnl = float(trade.get("pnl_usd") or 0)
        base = _short(trade.get("symbol", ""))
        self.feed.say("close", (
            f"✅ {base} {trade.get('direction')} closed "
            f"({trade.get('exit_reason')}): +${pnl:.2f}. I'll trust setups "
            f"like this a little more.") if won else (
            f"❌ {base} {trade.get('direction')} closed "
            f"({trade.get('exit_reason')}): −${abs(pnl):.2f}. I'll be more "
            f"careful with setups like this."), trade.get("symbol"))
        # 1) learning updates FIRST — the paper trade is already terminally
        #    closed, so if we let the (fallible) live close run first and it
        #    raised, this outcome would be lost from the model forever.
        try:
            feats = _json.loads(trade["features"])
            norm_snap = feats.pop("_norm_snapshot", None)
            self.engine.learn_from_trade(
                feats, trade["direction"], won, float(trade["confidence"]),
                exit_reason=trade.get("exit_reason"), symbol=trade.get("symbol"),
                norm_snapshot=norm_snap)
            save_learner_state(LEARNER_STATE_PATH, self.model, self.calibration)
        except Exception as e:
            log.error("learning_update_failed", trade_id=trade.get("id"),
                      error=str(e))
        # 2) mirror the close on the exchange; isolate failures so one bad
        #    close can't abort the loop. A raised close still leaves the real
        #    position protected by its exchange-side SL/TP (swap).
        try:
            await self.executor.close_position(trade)
            if self.executor.live:
                oid = f"t{trade['id']}"
                await self.db.set_live_order(oid, status="closed")
                await self.db.add_order_event(oid, "closed", {
                    "exit_reason": trade.get("exit_reason"),
                    "pnl_usd": trade.get("pnl_usd")})
        except Exception as e:
            log.error("live_close_failed", trade_id=trade.get("id"),
                      symbol=trade.get("symbol"), error=str(e),
                      hint="real position may remain open — guarded by "
                           "exchange SL/TP; manual check advised")
        # 2) journal review
        review, lessons, _ = await write_review(
            self.cfg, self.db, trade, self.calibration.calibration_score())
        # 3) notify
        equity = await self.db.latest_equity(self.risk.starting_equity)
        await self.tg.trade_closed(trade, lessons, equity)

    # ------------------------------------------------------------------
    async def run(self) -> None:
        await self.db.connect()
        eq = await self.db.latest_equity(self.risk.starting_equity)
        if not (await self.db.equity_curve()):
            await self.db.record_equity(self.risk.starting_equity, "start")
        mode = "LIVE" if self.executor.live else "dry-run (no real orders)"
        self.feed.say("sys", (
            f"🟢 Bot is up in {mode} mode — watching "
            f"{len(self._symbols)} markets with a "
            f"${eq:,.2f} paper balance. I'll narrate here every time I open, "
            f"close, skip a trade, or learn from a practice setup."))
        log.info("startup", equity=eq,
                 symbols=self._symbols,
                 exchange=self.cfg.get("exchange.id"),
                 execution_mode=mode)
        await self.tg.system_event(
            f"Adaptive Paper Trader started — equity ${eq:,.2f} "
            f"({', '.join(self._symbols)}). "
            f"Execution: {mode}.")

        # bound disk on boot: cap the analyses table to the retention row count
        try:
            total = await self.db.analyses_count()
            dropped = await self.db.prune_analyses(self._analyses_keep_rows)
            log.info("analyses_retention", total_before=total, dropped=dropped,
                     keep_rows=self._analyses_keep_rows)
        except Exception as e:
            log.error("prune_analyses_error", error=str(e))

        # reconcile persisted live orders against Bybit before trading resumes
        try:
            await self.reconcile_on_startup()
        except Exception as e:
            log.error("reconcile_error", error=str(e), exc_info=True)

        runner = None
        if self.cfg.get("dashboard.enabled"):
            # PORT env (set by Railway & friends) overrides config
            port = int(os.getenv("PORT", self.cfg.get("dashboard.port", 8787)))
            runner = await start_dashboard(
                self.db, self.risk.starting_equity,
                self.cfg.get("dashboard.host", "0.0.0.0"), port,
                learner_provider=lambda: {
                    "model": self.model.snapshot(FeatureVector.names()),
                    "calibration": self.calibration.snapshot(),
                },
                cfg=self.cfg, feed=self.feed,
                info={
                    "mode": mode,
                    "live": self.executor.live,
                    "exchange": self.cfg.get("exchange.id"),
                    "market_type": self.cfg.get("exchange.market_type", "swap"),
                    "symbols": self._symbols,
                    "timeframe": self.cfg.get("exchange.timeframe"),
                    "min_confidence": float(
                        self.cfg.get("strategy.min_confidence", 0.6)),
                    "max_notional_usd": float(
                        self.cfg.get("live.max_notional_usd", 0)),
                })
            log.info("dashboard_up", port=port)

        tasks = [asyncio.create_task(self.analyzer_loop(s))
                 for s in self._symbols]
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
            open_live = await self.db.open_live_orders()
            if open_live:
                log.info("shutdown_open_live_orders",
                         n=len(open_live),
                         orders=[o["order_id"] for o in open_live])
                for o in open_live:
                    await self.db.add_order_event(
                        o["order_id"], "app_shutdown",
                        {"note": "persists on Bybit; reconcile on next boot"})
        except Exception as e:
            log.error("shutdown_order_log_failed", error=str(e))
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if runner:
            await runner.cleanup()
        await self.executor.close()
        await self.collector.close()
        await self.db.close()


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
