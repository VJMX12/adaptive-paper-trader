"""Paper trading engine.

Computes simulated entries and PnL. Real Bybit orders are mirrored by a
separate, gated layer (app/trading/executor.py) only when live is fully
armed; this engine itself never talks to an exchange.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.analysis.engine import AnalysisResult
from app.db.database import Database, utcnow
from app.logging_setup import get_logger
from app.trading.risk import RiskManager, SizingDecision

log = get_logger("paper")


class PaperTradingEngine:
    def __init__(self, cfg, db: Database, risk: RiskManager):
        self.cfg = cfg
        self.db = db
        self.risk = risk
        self.cooldown_minutes = float(cfg.get("strategy.cooldown_minutes", 45))
        self.min_conf = float(cfg.get("strategy.min_confidence", 0.6))
        self.costs = {
            "taker_fee_pct": float(cfg.get("costs.taker_fee_pct", 0.0)),
            "slippage_pct": float(cfg.get("costs.slippage_pct", 0.0)),
            "funding_pct_per_8h": float(cfg.get("costs.funding_pct_per_8h", 0.0)),
        }
        # Serialize the check-then-open critical section. With one analyzer
        # loop per symbol (100+), concurrent opens could each read
        # open_positions_count below the cap and all pass, overshooting
        # max_open_positions (TOCTOU). One lock across symbols is fine —
        # opens are rare relative to the analysis interval.
        self._open_lock = asyncio.Lock()

    async def _in_cooldown(self, symbol: str) -> bool:
        last = await self.db.last_exit_ts(symbol)
        if not last:
            return False
        try:
            last_dt = datetime.fromisoformat(last)
        except ValueError:
            return False
        age_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
        return age_min < self.cooldown_minutes

    async def maybe_open(self, res: AnalysisResult, reasoning: str,
                         retrieval: dict) -> tuple[int | None, str]:
        """Open a simulated position if every condition holds.

        Returns (trade_id | None, explanation).
        """
        if not res.trade_recommended or res.direction is None:
            return None, "analysis did not recommend a trade"

        # Everything from the checks through the insert must be atomic w.r.t.
        # other symbols' opens, or max_open_positions can be overshot (TOCTOU).
        async with self._open_lock:
            if await self.db.has_open_trade(res.symbol):
                return None, f"already an open paper trade on {res.symbol}"
            if await self._in_cooldown(res.symbol):
                return None, f"{res.symbol} in post-trade cooldown"

            equity = await self.db.latest_equity(self.risk.starting_equity)
            peak = await self.db.peak_equity(self.risk.starting_equity)
            pnl_today = await self.db.realized_pnl_today()
            open_count = await self.db.open_positions_count()

            sizing: SizingDecision = self.risk.size_position(
                equity=equity, peak_equity=peak, pnl_today=pnl_today,
                open_positions=open_count,
                entry=res.price, stop=res.stop_loss,
                confidence=res.confidence, min_confidence=self.min_conf,
                sigma_per_candle=res.sigma_per_candle,
                candles_per_year=res.candles_per_year,
                cp_prob=res.changepoint_prob,
            )
            if not sizing.allowed:
                log.info("entry_blocked", symbol=res.symbol, reason=sizing.reason)
                return None, f"sizing blocked: {sizing.reason}"

            trade_id = await self.db.open_trade({
                "symbol": res.symbol, "direction": res.direction,
                "entry_ts": utcnow(), "entry_price": res.price,
                "stop_loss": res.stop_loss, "take_profit": res.take_profit,
                "position_size": sizing.position_size,
                "risk_amount": sizing.risk_amount, "rr": res.rr,
                "confidence": res.confidence, "raw_prob": res.raw_prob,
                "regime_label": res.regime_label,
                "changepoint_prob": res.changepoint_prob,
                "features": {**res.features.as_dict(),
                            "_norm_snapshot": res.norm_snapshot},
                "reasoning": reasoning,
                "similar_trades": {
                    "n": len(retrieval.get("neighbors") or []),
                    "win_rate": retrieval.get("win_rate"),
                    "avg_r": retrieval.get("avg_r"),
                    "neighbor_ids": [nb["trade_id"] for nb in
                                     (retrieval.get("neighbors") or [])[:10]],
                },
            })
        log.info("paper_trade_opened", trade_id=trade_id, symbol=res.symbol,
                 direction=res.direction, entry=res.price, sl=res.stop_loss,
                 tp=res.take_profit, size=sizing.position_size,
                 multipliers=sizing.multipliers)
        return trade_id, "opened"

    @staticmethod
    def compute_close(trade: dict, exit_price: float, exit_reason: str,
                      costs: dict | None = None) -> dict:
        entry = float(trade["entry_price"])
        size = float(trade["position_size"])
        sign = 1.0 if trade["direction"] == "long" else -1.0
        gross = sign * (exit_price - entry) * size
        try:
            t0 = datetime.fromisoformat(trade["entry_ts"])
            dur = (datetime.now(timezone.utc) - t0).total_seconds() / 60.0
        except ValueError:
            dur = 0.0

        # Realistic frictions so the learner/calibrator train on tradeable PnL,
        # not a cost-free market. Fees+slippage on both fills; funding prorated
        # over the hold. Defaults to zero when no costs are supplied.
        costs = costs or {}
        fee = float(costs.get("taker_fee_pct", 0.0))
        slip = float(costs.get("slippage_pct", 0.0))
        fund_8h = float(costs.get("funding_pct_per_8h", 0.0))
        entry_notional, exit_notional = entry * size, exit_price * size
        cost = ((fee + slip) * (entry_notional + exit_notional)
                + fund_8h * exit_notional * (dur / 60.0 / 8.0))
        pnl_usd = gross - cost
        pnl_pct = (pnl_usd / max(entry_notional, 1e-9)) * 100.0
        risk = float(trade["risk_amount"]) or 1e-9
        r_multiple = pnl_usd / risk
        return {
            "exit_ts": utcnow(), "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_usd": round(pnl_usd, 6), "pnl_pct": round(pnl_pct, 6),
            "r_multiple": round(r_multiple, 4),
            "duration_minutes": round(dur, 2),
            "costs_usd": round(cost, 6),
        }
