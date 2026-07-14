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

# Key under which the entry-time normalizer snapshot rides alongside a
# FeatureVector in persisted JSON (trades.features / shadow_setups.features).
# pack/unpack are the single place that format is defined, so callers never
# hand-roll the dict shape (and risk the two copies drifting apart).
_NORM_SNAPSHOT_KEY = "_norm_snapshot"


def pack_entry_features(fv: FeatureVector, norm_snapshot: dict) -> dict:
    """Serialize entry features + normalizer snapshot for DB persistence."""
    return {**fv.as_dict(), _NORM_SNAPSHOT_KEY: norm_snapshot}


def unpack_entry_features(stored: dict) -> tuple[dict, dict | None]:
    """Inverse of pack_entry_features. Returns (raw FeatureVector fields,
    norm snapshot) — snapshot is None for legacy records predating this
    fix, which callers treat as "use the model's live normalizer"."""
    stored = dict(stored)
    return stored, stored.pop(_NORM_SNAPSHOT_KEY, None)


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
    # shadow setup: the model's preferred direction + barrier geometry, ALWAYS
    # populated (even when no trade is recommended) so it can be resolved
    # against the forward price path for extra training labels.
    shadow_direction: str | None = None
    shadow_stop: float | None = None
    shadow_take_profit: float | None = None
    # normalizer state at the moment confidence was scored (see
    # OnlineLogistic.snapshot_norm) — persisted alongside entry features so
    # learn_from_trade/learn_from_shadow can re-score on the same scale later.
    norm_snapshot: dict = field(default_factory=dict)


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
        # Concept-drift adaptation: temporarily speed up learning for a symbol
        # right after a changepoint (regime shift), then relax back to the base
        # rate once things settle -- lets the model re-anchor to a NEW market
        # regime fast instead of being dragged down by stale pre-shift weights,
        # without noisily over-reacting during normal (stable-regime) trading.
        self.cp_lr_boost = float(cfg.get("learner.changepoint_lr_boost", 0.0))
        self.cold_start_handover = float(cfg.get("learner.cold_start_handover", 150))
        self._latest_cp: dict[str, float] = {}

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
        self._latest_cp[sym] = cp_prob
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
        # feed BOTH orientations to the normalizer so the long/short mirror is
        # exact: directional-feature means cancel to ~0, so a short is scored as
        # the true mirror of a long (not offset by 2*mean/std).
        self.model.observe(fv.oriented("long"))
        self.model.observe(fv.oriented("short"))

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
        # prior that fades out as real outcomes accumulate). Handover at 150
        # updates (~9 samples/feature) — 40 trusted a barely-seen model too soon.
        w = min(1.0, self.model.n_updates / self.cold_start_handover)
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
        # Freeze the normalizer NOW, at the moment p_dir/confidence was scored —
        # the model's mean/var keep advancing on every symbol's every cycle, so
        # by the time this trade/shadow-setup resolves and calls learn_from_*,
        # self.model's live normalizer has drifted well past this. Persisting
        # this snapshot alongside the entry features lets training happen on
        # the SAME feature scale that produced the reported confidence.
        norm_snapshot = self.model.snapshot_norm()

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
            shadow_direction=direction, shadow_stop=stop, shadow_take_profit=tp,
            norm_snapshot=norm_snapshot,
        )

    # ---------- learning ----------
    def _lr_mult(self, symbol: str | None) -> float:
        cp = self._latest_cp.get(symbol, 0.0) if symbol else 0.0
        return 1.0 + self.cp_lr_boost * cp

    def learn_from_trade(self, entry_features: dict, direction: str, won: bool,
                         stated_confidence: float, exit_reason: str | None = None,
                         symbol: str | None = None, norm_snapshot: dict | None = None) -> None:
        """Update learner + calibration after a closed trade.

        Target = P(TP before SL): only tp/sl exits are a clean barrier outcome.
        time_exit is near-flat label-noise and changepoint/regime exits fire on
        a different condition, so they DON'T train the P(win) head (they'd teach
        the model the wrong event). Calibration only records clean outcomes too,
        so its Brier measures the same event the confidence predicts.

        norm_snapshot (from AnalysisResult.norm_snapshot, persisted on the
        trade) re-scores the entry features on the SAME normalizer state that
        produced stated_confidence, instead of today's drifted one — without
        this, entry-time confidence and this update silently disagree about
        what the raw feature values mean.
        """
        if exit_reason is not None and exit_reason not in ("tp", "sl"):
            return
        fv = FeatureVector(**{n: float(entry_features[n]) for n in FeatureVector.names()})
        self.model.update(fv.oriented(direction), int(won),
                          lr_mult=self._lr_mult(symbol), norm=norm_snapshot)
        self.calibration.record(stated_confidence, int(won))

    def learn_from_shadow(self, entry_features: dict, direction: str,
                          won: bool, symbol: str | None = None,
                          norm_snapshot: dict | None = None) -> None:
        """Train on a SHADOW label: a recommended setup resolved against the
        forward price path (TP-before-SL) even though no live trade opened.
        Model weights only (not calibration — no stated confidence was acted
        on). This multiplies training signal far beyond actual fills."""
        fv = FeatureVector(**{n: float(entry_features[n]) for n in FeatureVector.names()})
        self.model.update(fv.oriented(direction), int(won),
                          lr_mult=self._lr_mult(symbol), norm=norm_snapshot)
