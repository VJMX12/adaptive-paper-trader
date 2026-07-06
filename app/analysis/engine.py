"""Analysis engine.

Per cycle, per symbol:
  features -> regime posterior (HMM) -> changepoint prob (BOCPD)
  -> directional probability (online learner, calibration-shrunk)
  -> blended with similar-trade retrieval -> bias / confidence / zones.

Bias direction comes from the *learned* probability plus regime drift —
there is no fixed "X crosses Y" rule anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.analysis.model import CalibrationTracker, OnlineLogistic
from app.data.collector import MarketSnapshot
from app.features.engine import FeatureVector, compute_features, regime_observations
from app.regime.bocpd import BOCPD
from app.regime.hmm import StickyGaussianHMM

TIMEFRAME_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}


@dataclass
class AnalysisResult:
    symbol: str
    price: float
    bias: str                 # bullish / bearish / neutral
    confidence: float         # calibrated, retrieval-blended
    raw_prob: float           # raw learner P(direction wins)
    direction: str | None     # long / short / None
    entry_zone: tuple[float, float] | None
    stop_loss: float | None
    take_profit: float | None
    rr: float | None
    regime_label: str
    regime_posterior: list[float]
    changepoint_prob: float
    features: FeatureVector
    trade_recommended: bool
    veto_reasons: list[str] = field(default_factory=list)
    key_factors: list[tuple[str, float]] = field(default_factory=list)
    invalidation: str = ""
    sigma_per_candle: float = 0.0
    candles_per_year: float = 0.0


class AnalysisEngine:
    def __init__(self, cfg, model: OnlineLogistic, calibration: CalibrationTracker):
        self.cfg = cfg
        self.model = model
        self.calibration = calibration
        self.hmm = StickyGaussianHMM(
            n_states=int(cfg.get("regime.n_states", 4)),
            sticky=float(cfg.get("regime.sticky", 0.97)),
        )
        self.bocpd: dict[str, BOCPD] = {}
        self._cycles_since_fit = 10**9  # force fit on first cycle
        self.refit_every = int(cfg.get("regime.refit_every", 288))
        self.fit_window = int(cfg.get("regime.fit_window", 1500))
        tf = cfg.get("exchange.timeframe", "5m")
        minutes = TIMEFRAME_MINUTES.get(tf, 5)
        self.candles_per_year = 365 * 24 * 60 / minutes
        self.min_conf = float(cfg.get("strategy.min_confidence", 0.6))
        self.min_rr = float(cfg.get("strategy.min_rr", 1.5))
        self.sl_mult = float(cfg.get("strategy.sl_sigma_mult", 1.6))
        self.tp_rr = float(cfg.get("strategy.tp_rr", 2.0))
        self.cp_alert = float(cfg.get("changepoint.alert_threshold", 0.35))

    # ---------- regime machinery ----------
    def _ensure_regime_models(self, snap: MarketSnapshot) -> tuple[np.ndarray, str, float]:
        obs = regime_observations(snap)
        if self._cycles_since_fit >= self.refit_every or not self.hmm.fitted:
            self.hmm.fit(obs[-self.fit_window:])
            self._cycles_since_fit = 0
        else:
            self._cycles_since_fit += 1

        posterior = self.hmm.filter_posterior(obs[-300:])
        labels = self.hmm.describe_states()
        top = int(np.argmax(posterior))
        regime_label = f"{labels[top]} (state {top})"

        det = self.bocpd.get(snap.symbol)
        if det is None:
            det = BOCPD(hazard=float(self.cfg.get("changepoint.hazard", 250.0)))
            det.update_many(obs[-300:, 0])  # warm start on history
            self.bocpd[snap.symbol] = det
        else:
            det.update(float(obs[-1, 0]))   # one new candle per cycle
        cp_prob = det.p_recent_changepoint(within=6)
        return posterior, regime_label, cp_prob

    # ---------- direction & probability ----------
    def _directional_view(self, fv: FeatureVector, posterior: np.ndarray) -> tuple[str, float]:
        """Candidate direction + learner probability that it reaches TP first.

        Direction candidate = sign of a drift score built from learned regime
        drift and recent multi-horizon returns (a *belief*, refined by the
        learner's probability — not a crossover rule).
        """
        drift = 0.0
        if self.hmm.fitted:
            drift = float(np.dot(posterior, self.hmm.means[:, 0]))  # E[return | regimes]
        score = np.tanh(drift / 1e-4) + 0.6 * np.tanh(fv.slope_96 / 2.0) \
            + 0.3 * np.tanh(fv.ret_24 / max(fv.sigma * 5, 1e-9)) \
            + 0.2 * np.tanh(fv.signed_flow * 3) + 0.1 * np.tanh(fv.ob_imbalance * 3)
        direction = "long" if score >= 0 else "short"
        raw_prob = self.model.predict_proba(fv.as_array())
        # learner predicts P(long-favorable); mirror for shorts
        p_model = raw_prob if direction == "long" else 1.0 - raw_prob

        # Cold-start bootstrap: an untrained learner always says 0.50, which
        # would mean "never trade, never learn". Blend a structural prior
        # (bounded by the drift-score strength) that hands over to the
        # learner as real trade outcomes accumulate.
        p_prior = 0.5 + 0.22 * float(np.tanh(abs(score)))
        w = min(1.0, self.model.n_updates / 40.0)
        p_dir = w * p_model + (1.0 - w) * p_prior
        return direction, float(p_dir)

    # ---------- main entry point ----------
    def analyze(self, snap: MarketSnapshot) -> AnalysisResult:
        fv = compute_features(snap)
        posterior, regime_label, cp_prob = self._ensure_regime_models(snap)
        direction, p_dir = self._directional_view(fv, posterior)
        shrunk = self.calibration.shrink(p_dir)

        price = snap.last_price
        sigma_px = fv.sigma * price                 # per-candle sigma in price units
        stop_dist = self.sl_mult * sigma_px
        tp_dist = self.tp_rr * stop_dist

        if direction == "long":
            stop, tp = price - stop_dist, price + tp_dist
            entry_zone = (price - 0.25 * sigma_px, price + 0.25 * sigma_px)
        else:
            stop, tp = price + stop_dist, price - tp_dist
            entry_zone = (price - 0.25 * sigma_px, price + 0.25 * sigma_px)
        rr = tp_dist / stop_dist if stop_dist > 0 else 0.0

        veto: list[str] = []
        if not np.isfinite(shrunk):
            veto.append("non-finite confidence")
        elif shrunk < self.min_conf:
            veto.append(f"confidence {shrunk:.2f} < {self.min_conf:.2f}")
        if cp_prob >= self.cp_alert:
            veto.append(f"changepoint alarm {cp_prob:.2f} >= {self.cp_alert:.2f}")
        if rr < self.min_rr:
            veto.append(f"RR {rr:.2f} < {self.min_rr:.2f}")
        if fv.sigma <= 0:
            veto.append("degenerate volatility")

        if shrunk >= self.min_conf:
            bias = "bullish" if direction == "long" else "bearish"
        else:
            bias = "neutral"

        recommended = len(veto) == 0
        invalidation = (
            f"Invalid if: changepoint prob > {self.cp_alert:.2f}, "
            f"regime posterior flips against the {direction} drift, "
            f"or price closes beyond {stop:.6g}."
        )
        return AnalysisResult(
            symbol=snap.symbol, price=price, bias=bias,
            confidence=round(shrunk, 4), raw_prob=round(p_dir, 4),
            direction=direction if recommended else None,
            entry_zone=entry_zone if recommended else None,
            stop_loss=stop if recommended else None,
            take_profit=tp if recommended else None,
            rr=round(rr, 2) if recommended else None,
            regime_label=regime_label,
            regime_posterior=[round(float(p), 4) for p in posterior],
            changepoint_prob=round(cp_prob, 4),
            features=fv, trade_recommended=recommended, veto_reasons=veto,
            key_factors=self.model.contributions(fv.as_array(), FeatureVector.names()),
            invalidation=invalidation,
            sigma_per_candle=fv.sigma, candles_per_year=self.candles_per_year,
        )

    # ---------- learning ----------
    def learn_from_trade(self, entry_features: dict, direction: str, won: bool,
                         stated_confidence: float) -> None:
        """Update learner + calibration after a closed trade.

        Learner target is 'long-favorable': a long win or a short loss => y=1.
        """
        x = np.array([entry_features[n] for n in FeatureVector.names()], dtype=float)
        y_long = int(won) if direction == "long" else int(not won)
        self.model.update(x, y_long)
        self.calibration.record(stated_confidence, int(won))
