"""Feature engine.

Computes *measurements* of the market (returns, volatility, flow, structure).
Deliberately contains no trading rules: nothing here says buy or sell.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from app.data.collector import MarketSnapshot

EPS = 1e-12


@dataclass
class FeatureVector:
    # returns at multiple horizons (log returns)
    ret_1: float
    ret_6: float
    ret_24: float
    ret_72: float
    # volatility
    sigma: float              # realized vol of last 48 candles (per-candle stdev of log returns)
    sigma_ratio: float        # short vol / long vol (vol expansion > 1)
    # range / structure
    hl_range_norm: float      # mean high-low range / close, last 24
    close_position: float     # where last close sits in the last-96-candle range [0..1]
    upper_shadow: float       # mean upper wick fraction, last 24
    lower_shadow: float       # mean lower wick fraction, last 24
    # volume / flow
    vol_zscore: float         # z-score of last-6 volume vs last-96
    signed_flow: float        # sum(sign(ret)*volume) normalized, last 24  (order-flow proxy)
    ob_imbalance: float       # live order book imbalance, 0 if unavailable
    # trend structure without indicators: regression slope of log price
    slope_24: float           # per-candle log-price slope * 1e4
    slope_96: float
    r2_96: float              # how "line-like" the last 96 candles are

    def as_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=float)

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def names() -> list[str]:
        return list(FeatureVector.__dataclass_fields__.keys())

    # Directional features whose sign flips when a setup is viewed as a SHORT
    # (positive = bullish). Non-directional features (volatility, range, R²)
    # are left unchanged. close_position is mirrored; the two shadows swap.
    _NEGATE = ("ret_1", "ret_6", "ret_24", "ret_72",
               "signed_flow", "ob_imbalance", "slope_24", "slope_96")

    def oriented(self, direction: str) -> np.ndarray:
        """Feature vector re-expressed so that 'positive = favorable for THIS
        trade direction'. A short is the mirror image of a long, so one model
        can learn a single P(win) map for both directions (no 1-p complement)."""
        d = asdict(self)
        if direction == "short":
            for k in self._NEGATE:
                d[k] = -d[k]
            d["close_position"] = 1.0 - d["close_position"]
            d["upper_shadow"], d["lower_shadow"] = d["lower_shadow"], d["upper_shadow"]
        return np.array(list(d.values()), dtype=float)


def _log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.maximum(close, EPS)))


def _slope_r2(y: np.ndarray) -> tuple[float, float]:
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    yv = y - y.mean()
    denom = float((x**2).sum())
    if denom < EPS:
        return 0.0, 0.0
    slope = float((x * yv).sum() / denom)
    yhat = slope * x
    ss_res = float(((yv - yhat) ** 2).sum())
    ss_tot = float((yv**2).sum())
    r2 = 0.0 if ss_tot < EPS else max(0.0, 1.0 - ss_res / ss_tot)
    return slope, r2


def compute_features(snap: MarketSnapshot) -> FeatureVector:
    c, h, l, o, v = snap.close, snap.high, snap.low, snap.open, snap.volume
    n = len(c)
    if n < 100:
        raise ValueError(f"Need >=100 candles, got {n}")

    logc = np.log(np.maximum(c, EPS))
    r = _log_returns(c)

    def ret_h(k: int) -> float:
        return float(logc[-1] - logc[-1 - k]) if n > k else 0.0

    sigma_short = float(np.std(r[-48:], ddof=1))
    sigma_long = float(np.std(r[-192:], ddof=1)) if len(r) >= 192 else sigma_short
    # clip: a near-flat long window can send this to 1e6 and dominate z-scoring
    sigma_ratio = float(np.clip(sigma_short / max(sigma_long, EPS), 0.0, 10.0))

    rng = (h[-24:] - l[-24:]) / np.maximum(c[-24:], EPS)
    hi96, lo96 = float(h[-96:].max()), float(l[-96:].min())
    close_pos = 0.5 if hi96 - lo96 < EPS else (float(c[-1]) - lo96) / (hi96 - lo96)

    body_hi = np.maximum(o[-24:], c[-24:])
    body_lo = np.minimum(o[-24:], c[-24:])
    full = np.maximum(h[-24:] - l[-24:], EPS)
    upper = float(np.mean((h[-24:] - body_hi) / full))
    lower = float(np.mean((body_lo - l[-24:]) / full))

    v96_mu, v96_sd = float(v[-96:].mean()), float(v[-96:].std(ddof=1))
    vol_z = 0.0 if v96_sd < EPS else (float(v[-6:].mean()) - v96_mu) / v96_sd

    sr = np.sign(r[-24:]) * v[-24:][-len(r[-24:]):]
    denom = float(np.abs(v[-24:]).sum())
    signed_flow = 0.0 if denom < EPS else float(sr.sum()) / denom

    slope24, _ = _slope_r2(logc[-24:])
    slope96, r2_96 = _slope_r2(logc[-96:])

    return FeatureVector(
        ret_1=ret_h(1), ret_6=ret_h(6), ret_24=ret_h(24), ret_72=ret_h(72),
        sigma=sigma_short, sigma_ratio=sigma_ratio,
        hl_range_norm=float(rng.mean()), close_position=float(np.clip(close_pos, 0, 1)),
        upper_shadow=upper, lower_shadow=lower,
        vol_zscore=float(np.clip(vol_z, -5, 5)), signed_flow=signed_flow,
        ob_imbalance=float(snap.ob_imbalance or 0.0),
        slope_24=slope24 * 1e4, slope_96=slope96 * 1e4, r2_96=r2_96,
    )


def regime_observations(snap: MarketSnapshot) -> np.ndarray:
    """Per-candle 2D observations (return, |return|) for the regime HMM."""
    r = _log_returns(snap.close)
    return np.column_stack([r, np.abs(r)])
