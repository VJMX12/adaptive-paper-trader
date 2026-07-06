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
        # Per-symbol HMM + refit counter: 5m return scale differs ~10-20x
        # across the universe, so one shared HMM fit on one symbol produces a
        # degenerate posterior for every other. Each symbol gets its own
        # (BOCPD already is per-symbol).
        self._n_states = int(cfg.get("regime.n_states", 4))
        self._sticky = float(cfg.get("regime.sticky", 0.97))
        self.hmms: dict[str, StickyGaussianHMM] = {}
        self._cycles_since_fit: dict[str, int] = {}
        self.bocpd: dict[str, BOCPD] = {}
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
    def _ensure_regime_models(self, snap: MarketSnapshot):
        sym = snap.symbol
        obs = regime_observations(snap)
        hmm = self.hmms.get(sym)
        if hmm is None:
            hmm = StickyGaussianHMM(n_states=self._n_states, sticky=self._sticky)
            self.hmms[sym] = hmm
            self._cycles_since_fit[sym] = 10**9  # force fit on first cycle
        if self._cycles_since_fit[sym] >= self.refit_every or not hmm.fitted:
            hmm.fit(obs[-self.fit_window:])
            self._cycles_since_fit[sym] = 0
        else:
            self._cycles_since_fit[sym] += 1

        posterior = hmm.filter_posterior(obs[-300:])
        labels = hmm.describe_states()
        top = int(np.argmax(posterior))
        regime_label = f"{labels[top]} (state {top})"

        det = self.bocpd.get(sym)
        if det is None:
            det = BOCPD(hazard=float(self.cfg.get("changepoint.hazard", 250.0)))
            det.update_many(obs[-300:, 0])  # warm start on history
            self.bocpd[sym] = det
        else:
            det.update(float(obs[-1, 0]))   # one new candle per cycle
        cp_prob = det.p_recent_changepoint(within=6)
        return posterior, regime_label, cp_prob, hmm

    # ---------- direction & probability ----------
    def _directional_view(self, fv: FeatureVector, posterior: np.ndarray,
                          hmm: StickyGaussianHMM) -> tuple[str, float]:
        """Pick the direction and its P(win) that it reaches TP before SL.

        The model predicts P(win) on DIRECTION-ORIENTED features (a short is
        the mirror of a long), so P(short wins) is computed DIRECTLY from the
        short-oriented features — never as 1 - P(long). Direction is chosen by
        whichever side has the higher P(win), with a drift prior for cold start.
        """
        # feed the normalizer on every analysis (unbiased feature distribution)
        self.model.observe(fv.as_array())

        drift = 0.0
        if hmm.fitted:
            drift = float(np.dot(posterior, hmm.means[:, 0]))  # E[return | regimes]
        score = np.tanh(drift / 1e-4) + 0.6 * np.tanh(fv.slope_96 / 2.0) \
            + 0.3 * np.tanh(fv.ret_24 / max(fv.sigma * 5, 1e-9)) \
            + 0.2 * np.tanh(fv.signed_flow * 3) + 0.1 * np.tanh(fv.ob_imbalance * 3)

        # learner's P(win) for each side from its own oriented features
        p_long = self.model.predict_proba(fv.oriented("long"))
        p_short = self.model.predict_proba(fv.oriented("short"))

        # Cold-start: an untrained learner says ~0.50 for both, so hand the
        # direction to the drift belief and keep confidence modest (a bounded
        # prior that fades out as real outcomes accumulate).
        w = min(1.0, self.model.n_updates / 40.0)
        prior_mag = 0.20 * float(np.tanh(abs(score)))     # <=0.20 -> caps prior at 0.70
        favored = "long" if score >= 0 else "short"
        p_long_b = w * p_long + (1 - w) * (0.5 + (prior_mag if favored == "long" else -prior_mag))
        p_short_b = w * p_short + (1 - w) * (0.5 + (prior_mag if favored == "short" else -prior_mag))

        if p_long_b >= p_short_b:
            return "long", float(p_long_b)
        return "short", float(p_short_b)

    # ---------- main entry point ----------
    def analyze(self, snap: MarketSnapshot) -> AnalysisResult:
        fv = compute_features(snap)
        posterior, regime_label, cp_prob, hmm = self._ensure_regime_models(snap)
        direction, p_dir = self._directional_view(fv, posterior, hmm)
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

        Single coherent target: on features ORIENTED to the trade's direction,
        y=1 iff the trade won. A long and a short that both won are the same
        'favorable setup -> win' example — no long/short target mixing.
        """
        fv = FeatureVector(**{n: float(entry_features[n]) for n in FeatureVector.names()})
        self.model.update(fv.oriented(direction), int(won))
        self.calibration.record(stated_confidence, int(won))
