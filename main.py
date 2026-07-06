"""Adaptive Paper Trader — main orchestrator.

Async tasks:
  * one analyzer loop per symbol (fetch -> analyze -> maybe open paper trade)
  * one monitor loop (SL/TP/adaptive exits, journal, learning updates)
  * dashboard web server

PAPER TRADING ONLY. No exchange credentials are ever loaded; no order
endpoint is ever called.
"""
from __future__ import annotations

import asyncio
import os
import signal

from app.analysis.engine import AnalysisEngine
from app.analysis.model import load_learner_state, save_learner_state
from app.analysis.reasoning import analysis_reasoning
from app.config import load_config
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
        self.model, self.calibration = load_learner_state(LEARNER_STATE_PATH, n_features)
        self.engine = AnalysisEngine(self.cfg, self.model, self.calibration)
        self.risk = RiskManager(self.cfg)
        self.paper = PaperTradingEngine(self.cfg, self.db, self.risk)
        self.executor = BybitExecutor(self.cfg)
        self.monitor = TradeMonitor(self.cfg, self.db, self.collector, self.paper)
        self.tg = TelegramNotifier(self.cfg)
        self._stop = asyncio.Event()
        self._analysis_count: dict[str, int] = {}

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
                            await self.executor.open_position(t)
                            await self.tg.trade_opened(
                                trade_id, res, t["position_size"],
                                t["risk_amount"], reasoning)
                        else:
                            log.info("entry_skipped", symbol=symbol, why=why)
            except Exception as e:
                log.error("analyzer_error", symbol=symbol, error=str(e), exc_info=True)
            await _wait(self._stop, interval)

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
        await self.executor.close_position(trade)
        won = (trade.get("pnl_usd") or 0) > 0
        feats = _json.loads(trade["features"])
        # 1) learning updates (model + calibration), persisted
        self.engine.learn_from_trade(
            feats, trade["direction"], won, float(trade["confidence"]))
        save_learner_state(LEARNER_STATE_PATH, self.model, self.calibration)
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
        log.info("startup", equity=eq,
                 symbols=self.cfg.get("exchange.symbols"),
                 exchange=self.cfg.get("exchange.id"),
                 execution_mode=mode)
        await self.tg.system_event(
            f"Adaptive Paper Trader started — equity ${eq:,.2f} "
            f"({', '.join(self.cfg.get('exchange.symbols'))}). "
            f"Execution: {mode}.")

        runner = None
        if self.cfg.get("dashboard.enabled"):
            # PORT env (set by Railway & friends) overrides config
            port = int(os.getenv("PORT", self.cfg.get("dashboard.port", 8787)))
            runner = await start_dashboard(
                self.db, self.risk.starting_equity,
                self.cfg.get("dashboard.host", "0.0.0.0"), port,
                info={
                    "mode": mode,
                    "live": self.executor.live,
                    "exchange": self.cfg.get("exchange.id"),
                    "market_type": self.cfg.get("exchange.market_type", "swap"),
                    "symbols": self.cfg.get("exchange.symbols"),
                    "timeframe": self.cfg.get("exchange.timeframe"),
                    "min_confidence": float(
                        self.cfg.get("strategy.min_confidence", 0.6)),
                    "max_notional_usd": float(
                        self.cfg.get("live.max_notional_usd", 0)),
                })
            log.info("dashboard_up", port=port)

        tasks = [asyncio.create_task(self.analyzer_loop(s))
                 for s in self.cfg.get("exchange.symbols")]
        tasks.append(asyncio.create_task(self.monitor_loop()))

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                pass  # windows

        await self._stop.wait()
        log.info("shutting_down")
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
