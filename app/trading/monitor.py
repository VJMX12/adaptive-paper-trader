"""Open-trade monitoring loop.

Closes a paper trade when:
  - TP reached, or SL reached (checked against live price)
  - adaptive exit: changepoint alarm while in profit, or regime flips hard
    against the position
  - time-based safety exit (max_trade_hours)

Also tracks MAE/MFE (max adverse / favorable excursion) which the journal
uses to judge whether entries were early/late and SL/TP placement quality.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.data.collector import MarketDataCollector
from app.db.database import Database
from app.logging_setup import get_logger
from app.trading.paper_engine import PaperTradingEngine

log = get_logger("monitor")


class TradeMonitor:
    def __init__(self, cfg, db: Database, collector: MarketDataCollector,
                 paper: PaperTradingEngine):
        self.cfg = cfg
        self.db = db
        self.collector = collector
        self.paper = paper
        self.max_trade_hours = float(cfg.get("strategy.max_trade_hours", 48))
        self.cp_alert = float(cfg.get("changepoint.alert_threshold", 0.35))
        # latest analysis context per symbol, pushed by the analyzer loop
        self.live_context: dict[str, dict] = {}

    def push_context(self, symbol: str, cp_prob: float, drift_direction: str | None,
                     drift_strength: float) -> None:
        self.live_context[symbol] = {
            "cp_prob": cp_prob,
            "drift_direction": drift_direction,
            "drift_strength": drift_strength,
        }

    async def check_all(self) -> list[dict]:
        """Check every open trade once. Returns list of closed-trade dicts."""
        closed: list[dict] = []
        for trade in await self.db.get_open_trades():
            try:
                result = await self._check_one(trade)
                if result:
                    closed.append(result)
            except Exception as e:
                log.error("monitor_error", trade_id=trade["id"], error=str(e))
        return closed

    async def _check_one(self, trade: dict) -> dict | None:
        symbol = trade["symbol"]
        price = await self.collector.fetch_last_price(symbol)
        entry = float(trade["entry_price"])
        sl, tp = float(trade["stop_loss"]), float(trade["take_profit"])
        long = trade["direction"] == "long"

        # --- excursion tracking ---
        move_pct = (price - entry) / entry * 100.0 * (1 if long else -1)
        mae = min(float(trade.get("mae_pct") or 0.0), move_pct)
        mfe = max(float(trade.get("mfe_pct") or 0.0), move_pct)
        await self.db.update_excursions(trade["id"], mae, mfe)

        exit_reason = None
        exit_price = price
        if long:
            if price <= sl:
                exit_reason, exit_price = "sl", sl
            elif price >= tp:
                exit_reason, exit_price = "tp", tp
        else:
            if price >= sl:
                exit_reason, exit_price = "sl", sl
            elif price <= tp:
                exit_reason, exit_price = "tp", tp

        # --- adaptive exits (strategy-level, not fixed rules on price) ---
        ctx = self.live_context.get(symbol)
        if exit_reason is None and ctx:
            in_profit = move_pct > 0
            if ctx["cp_prob"] >= self.cp_alert and in_profit:
                exit_reason = "changepoint_exit"
            elif (ctx["drift_direction"] is not None
                  and ctx["drift_direction"] != trade["direction"]
                  and ctx["drift_strength"] >= 0.75):
                exit_reason = "regime_exit"

        # --- time safety net ---
        if exit_reason is None:
            try:
                t0 = datetime.fromisoformat(trade["entry_ts"])
                hours = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
                if hours >= self.max_trade_hours:
                    exit_reason = "time_exit"
            except ValueError:
                pass

        if exit_reason is None:
            return None

        close = self.paper.compute_close(trade, exit_price, exit_reason)
        await self.db.close_trade(trade["id"], close)
        equity = await self.db.latest_equity(
            float(self.cfg.get("risk.starting_equity")))
        new_equity = equity + close["pnl_usd"]
        await self.db.record_equity(new_equity, event=f"close#{trade['id']}:{exit_reason}")
        log.info("paper_trade_closed", trade_id=trade["id"], symbol=symbol,
                 reason=exit_reason, pnl_usd=close["pnl_usd"], r=close["r_multiple"])
        full = await self.db.get_trade(trade["id"])
        return full
