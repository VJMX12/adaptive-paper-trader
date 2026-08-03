"""Rule-based strategies: fixed entry logic (no HMM/online-learner), but
still genuinely adaptive — confidence for each is a Beta-Bernoulli win-rate
learned from that strategy's OWN resolved outcomes (shadow setups + real
trades), persisted to disk like the ML learner's state.

Each implements the same interface as MLStrategyAdapter (app/strategies/ml_adapter.py)
so main.py can drive every strategy through one uniform code path:
  .analyze(snap) -> AnalysisResult
  .learn_from_shadow(feats, direction, won, *, symbol, norm_snapshot)
  .learn_from_trade(feats, direction, won, confidence, *, exit_reason, symbol, norm_snapshot)
  .n_updates -> int
  .save() -> None
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.analysis.engine import AnalysisResult
from app.features.engine import FeatureVector, compute_features

TIMEFRAME_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                     "1h": 60, "4h": 240, "1d": 1440}


class BetaTracker:
    """Persisted Beta-Bernoulli win-rate estimate per direction bucket.
    Weak prior (alpha=beta=3, mean 0.5) so a strategy starts cautious and its
    confidence is *earned* from real outcomes, same philosophy as the ML
    learner's cold-start handover."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.counts = {"long": [3.0, 3.0], "short": [3.0, 3.0]}  # [alpha, beta]
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                for k in ("long", "short"):
                    if k in data:
                        self.counts[k] = [float(data[k][0]), float(data[k][1])]
            except Exception:
                pass

    def mean(self, direction: str) -> float:
        a, b = self.counts.get(direction, [3.0, 3.0])
        return a / (a + b)

    def update(self, direction: str, won: bool) -> None:
        a, b = self.counts.get(direction, [3.0, 3.0])
        if won:
            a += 1.0
        else:
            b += 1.0
        self.counts[direction] = [a, b]

    @property
    def n_updates(self) -> int:
        return int(sum(a + b for a, b in self.counts.values()) - 12.0)  # minus the two priors' 6 pseudo-counts

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.counts))


class RuleBasedStrategy:
    """Shared scaffolding: geometry, veto checks, shadow direction — same
    shape as AnalysisEngine.analyze, minus the HMM/changepoint/online-model
    machinery. Subclasses only implement _signal(fv) -> direction | None."""

    id = "rule"
    label = "Rule-based"

    def __init__(self, cfg, state_path: str):
        self.cfg = cfg
        self.min_conf = float(cfg.get(f"strategies.{self.id}.min_confidence", 0.55))
        self.min_rr = float(cfg.get("strategy.min_rr", 1.5))
        self.sl_mult = float(cfg.get("strategy.sl_sigma_mult", 1.6))
        self.tp_rr = float(cfg.get("strategy.tp_rr", 2.0))
        self.min_stop_pct = float(cfg.get("strategy.min_stop_pct", 0.004))
        tf = cfg.get("exchange.timeframe", "5m")
        minutes = TIMEFRAME_MINUTES.get(tf, 5)
        self.candles_per_year = 365 * 24 * 60 / minutes
        self.tracker = BetaTracker(state_path)

    def _signal(self, fv: FeatureVector) -> str | None:
        raise NotImplementedError

    def analyze(self, snap) -> AnalysisResult:
        fv = compute_features(snap)
        price = snap.last_price
        direction = self._signal(fv)

        sigma_px = fv.sigma * price
        # Floor stop at min_stop_pct of price: fee+slippage cost is fixed per
        # notional regardless of stop distance, so an unfloored sigma-based
        # stop on low-vol symbols lets cost_in_R blow up (see engine.py).
        stop_dist = max(self.sl_mult * sigma_px, self.min_stop_pct * price)
        tp_dist = self.tp_rr * stop_dist
        rr = tp_dist / stop_dist if stop_dist > 0 else 0.0

        if direction is None:
            confidence = 0.5
            stop = tp = None
        else:
            confidence = self.tracker.mean(direction)
            if direction == "long":
                stop, tp = price - stop_dist, price + tp_dist
            else:
                stop, tp = price + stop_dist, price - tp_dist

        veto: list[str] = []
        if direction is None:
            veto.append("no signal")
        elif not np.isfinite(confidence):
            veto.append("non-finite confidence")
        elif confidence < self.min_conf:
            veto.append(f"confidence {confidence:.2f} < {self.min_conf:.2f}")
        if fv.sigma <= 0:
            veto.append("degenerate volatility")
        if direction is not None and rr < self.min_rr:
            veto.append(f"RR {rr:.2f} < {self.min_rr:.2f}")

        trade_recommended = direction is not None and not veto
        bias = ("bullish" if direction == "long" else
                "bearish" if direction == "short" else "neutral")

        return AnalysisResult(
            symbol=snap.symbol, price=price, bias=bias,
            confidence=round(float(confidence), 4), raw_prob=float(confidence),
            direction=direction if direction is not None else None,
            entry_zone=(price, price), stop_loss=stop, take_profit=tp, rr=rr,
            regime_label=f"{self.label} (rule-based)", regime_posterior=[],
            changepoint_prob=0.0, features=fv, trade_recommended=trade_recommended,
            veto_reasons=veto, key_factors=[], invalidation="",
            sigma_per_candle=fv.sigma, candles_per_year=self.candles_per_year,
            shadow_direction=direction, shadow_stop=stop, shadow_take_profit=tp,
            norm_snapshot={},
        )

    def learn_from_shadow(self, feats, direction, won, *, symbol=None, norm_snapshot=None) -> None:
        self.tracker.update(direction, won)

    def learn_from_trade(self, feats, direction, won, confidence, *,
                         exit_reason=None, symbol=None, norm_snapshot=None) -> None:
        self.tracker.update(direction, won)

    @property
    def n_updates(self) -> int:
        return self.tracker.n_updates

    def save(self) -> None:
        self.tracker.save()

    def snapshot(self, names) -> dict:
        return {"kind": "rule", "id": self.id,
                "win_rate_long": round(self.tracker.mean("long"), 4),
                "win_rate_short": round(self.tracker.mean("short"), 4),
                "n_updates": self.n_updates}


class MomentumStrategy(RuleBasedStrategy):
    """Trend-following: trades in the direction of aligned short- and
    long-horizon price slope. No mean-reversion, no regime model — pure
    'the trend is your friend', for a genuinely different risk/return shape
    to compare against the adaptive ML strategy."""

    id = "momentum"
    label = "Momentum"

    def __init__(self, cfg, state_path: str):
        super().__init__(cfg, state_path)
        self.slope_threshold = float(cfg.get(f"strategies.{self.id}.slope_threshold", 0.15))

    def _signal(self, fv: FeatureVector) -> str | None:
        if fv.slope_24 > self.slope_threshold and fv.slope_96 > self.slope_threshold:
            return "long"
        if fv.slope_24 < -self.slope_threshold and fv.slope_96 < -self.slope_threshold:
            return "short"
        return None


class BreakoutStrategy(RuleBasedStrategy):
    """Trade FRESH range breaks confirmed by volatility expansion: price at
    the edge of its 96-candle range AND short-term vol running hot versus
    long-term vol (sigma_ratio > 1 means something is actually happening,
    not just noise sitting at an old extreme). This is the mirror image of
    MeanReversionStrategy's trigger (same close_position read) but the
    OPPOSITE direction and gated on vol expansion instead of vol level --
    deliberately so the two disagree on what a range extreme means and can
    be compared to see which regime the market is actually rewarding."""

    id = "breakout"
    label = "Breakout"

    def __init__(self, cfg, state_path: str):
        super().__init__(cfg, state_path)
        self.breakout_threshold = float(
            cfg.get(f"strategies.{self.id}.breakout_threshold", 0.85))
        self.vol_expansion_min = float(
            cfg.get(f"strategies.{self.id}.vol_expansion_min", 1.2))

    def _signal(self, fv: FeatureVector) -> str | None:
        if fv.sigma_ratio < self.vol_expansion_min:
            return None
        lo = 1.0 - self.breakout_threshold
        if fv.close_position >= self.breakout_threshold:
            return "long"    # pushing out the top of the range, vol confirms
        if fv.close_position <= lo:
            return "short"   # pushing out the bottom of the range, vol confirms
        return None


class OrderFlowStrategy(RuleBasedStrategy):
    """Follow the tape: trade in the direction of signed volume flow (which
    side is actually being bought/sold) when participation itself is
    elevated (vol_zscore high). Unlike Momentum (price slope) or Breakout
    (range position + vol expansion), this reacts to WHO is trading, not
    WHERE price is -- a distinct information source that can lead price
    structure instead of confirming it."""

    id = "order_flow"
    label = "Order Flow"

    def __init__(self, cfg, state_path: str):
        super().__init__(cfg, state_path)
        self.flow_threshold = float(
            cfg.get(f"strategies.{self.id}.flow_threshold", 0.15))
        self.vol_z_min = float(
            cfg.get(f"strategies.{self.id}.vol_z_min", 0.5))

    def _signal(self, fv: FeatureVector) -> str | None:
        if fv.vol_zscore < self.vol_z_min:
            return None
        if fv.signed_flow > self.flow_threshold:
            return "long"
        if fv.signed_flow < -self.flow_threshold:
            return "short"
        return None


class MeanReversionStrategy(RuleBasedStrategy):
    """Fade-the-extreme: trades AGAINST price sitting near the edge of its
    recent range, betting on reversion toward the middle. The structural
    opposite of MomentumStrategy — deliberately so, for a meaningful
    comparison of which regime the market is actually rewarding."""

    id = "mean_reversion"
    label = "Mean Reversion"

    def __init__(self, cfg, state_path: str):
        super().__init__(cfg, state_path)
        self.extreme_threshold = float(cfg.get(f"strategies.{self.id}.extreme_threshold", 0.82))

    def _signal(self, fv: FeatureVector) -> str | None:
        lo = 1.0 - self.extreme_threshold
        if fv.close_position >= self.extreme_threshold:
            return "short"   # near range top -> expect reversion down
        if fv.close_position <= lo:
            return "long"    # near range bottom -> expect reversion up
        return None
