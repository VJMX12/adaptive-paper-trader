"""Adaptive position sizing + deterministic circuit breakers.

Size = fractional-Kelly base risk, multiplied down by four continuous factors:
  1. volatility targeting        (calmer markets -> closer to full size)
  2. calibrated confidence       (edge must be statistically earned)
  3. drawdown decay              (exposure shrinks smoothly in drawdown)
  4. changepoint collapse        (uncertainty alarm -> exposure toward zero)

Circuit breakers are DELIBERATELY dumb and fixed:
  - hard drawdown stop, max daily loss, max open positions.
An adaptive system must never decide its own hard limits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SizingDecision:
    allowed: bool
    reason: str
    position_size: float = 0.0   # base asset units
    risk_amount: float = 0.0     # $ lost if SL hit
    multipliers: dict | None = None


class RiskManager:
    def __init__(self, cfg):
        self.starting_equity = float(cfg.get("risk.starting_equity"))
        self.base_risk_pct = float(cfg.get("risk.base_risk_pct", 0.01))
        self.kelly_fraction = float(cfg.get("risk.kelly_fraction", 0.25))
        self.max_positions = int(cfg.get("risk.max_open_positions", 3))
        self.vol_target = float(cfg.get("risk.vol_target_annual", 0.35))
        self.dd_soft = float(cfg.get("risk.drawdown_soft_pct", 0.05))
        self.dd_hard = float(cfg.get("risk.drawdown_hard_pct", 0.15))
        self.max_daily_loss_pct = float(cfg.get("risk.max_daily_loss_pct", 0.04))
        self.cp_alert = float(cfg.get("changepoint.alert_threshold", 0.35))

    # ---------- circuit breakers (fixed, non-adaptive) ----------
    def circuit_breakers(self, equity: float, peak_equity: float,
                         pnl_today: float, open_positions: int) -> str | None:
        if open_positions >= self.max_positions:
            return f"max open positions ({self.max_positions}) reached"
        dd = 0.0 if peak_equity <= 0 else max(0.0, 1.0 - equity / peak_equity)
        if dd >= self.dd_hard:
            return f"hard drawdown breaker: {dd:.1%} >= {self.dd_hard:.0%}"
        if pnl_today < 0 and abs(pnl_today) >= self.max_daily_loss_pct * equity:
            return (f"daily loss breaker: {abs(pnl_today):.2f} >= "
                    f"{self.max_daily_loss_pct:.0%} of equity")
        return None

    # ---------- adaptive multipliers ----------
    def _vol_multiplier(self, sigma_per_candle: float, candles_per_year: float) -> float:
        realized_annual = sigma_per_candle * np.sqrt(max(candles_per_year, 1.0))
        if realized_annual <= 1e-9:
            return 1.0
        return float(np.clip(self.vol_target / realized_annual, 0.2, 1.5))

    def _confidence_multiplier(self, confidence: float, min_conf: float) -> float:
        # 0 at min_conf, 1 at 0.85+ — confidence must be *earned* to add size
        return float(np.clip((confidence - min_conf) / max(0.85 - min_conf, 1e-6), 0.05, 1.0))

    def _drawdown_multiplier(self, equity: float, peak_equity: float) -> float:
        dd = 0.0 if peak_equity <= 0 else max(0.0, 1.0 - equity / peak_equity)
        if dd <= self.dd_soft:
            return 1.0
        span = max(self.dd_hard - self.dd_soft, 1e-6)
        return float(np.clip(1.0 - (dd - self.dd_soft) / span, 0.0, 1.0))

    def _changepoint_multiplier(self, cp_prob: float) -> float:
        if cp_prob <= 0.5 * self.cp_alert:
            return 1.0
        return float(np.clip(1.0 - cp_prob / max(self.cp_alert, 1e-6), 0.0, 1.0))

    def size_position(
        self, *, equity: float, peak_equity: float, pnl_today: float,
        open_positions: int, entry: float, stop: float,
        confidence: float, min_confidence: float,
        sigma_per_candle: float, candles_per_year: float, cp_prob: float,
    ) -> SizingDecision:
        breaker = self.circuit_breakers(equity, peak_equity, pnl_today, open_positions)
        if breaker:
            return SizingDecision(False, breaker)

        stop_dist = abs(entry - stop)
        if stop_dist <= 0 or entry <= 0:
            return SizingDecision(False, "invalid entry/stop distance")

        mult = {
            "vol": self._vol_multiplier(sigma_per_candle, candles_per_year),
            "confidence": self._confidence_multiplier(confidence, min_confidence),
            "drawdown": self._drawdown_multiplier(equity, peak_equity),
            "changepoint": self._changepoint_multiplier(cp_prob),
        }
        combined = float(np.prod(list(mult.values())))
        risk_amount = equity * self.base_risk_pct * self.kelly_fraction * 4.0 * combined
        # (kelly_fraction*4 keeps risk == base_risk_pct at kelly_fraction=0.25, full multipliers)
        risk_amount = min(risk_amount, equity * 0.02)  # absolute per-trade cap

        if risk_amount < equity * 0.0005:
            return SizingDecision(False, f"risk too small after multipliers {mult}",
                                  multipliers=mult)
        size = risk_amount / stop_dist
        return SizingDecision(True, "ok", position_size=size,
                              risk_amount=risk_amount, multipliers=mult)
