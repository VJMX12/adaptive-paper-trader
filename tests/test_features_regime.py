import numpy as np
import pytest

from app.data.collector import MarketSnapshot
from app.features.engine import FeatureVector, compute_features, regime_observations
from app.regime.bocpd import BOCPD
from app.regime.hmm import StickyGaussianHMM


def make_snapshot(n=400, drift=0.0005, vol=0.005, seed=1, symbol="BTC/USDT"):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    high = close * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = close * (1 - np.abs(rng.normal(0, vol / 2, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    volu = np.abs(rng.normal(100, 20, n))
    ts = np.arange(n) * 300000.0
    return MarketSnapshot(symbol=symbol, timeframe="5m", ts=ts, open=open_,
                          high=high, low=low, close=close, volume=volu,
                          last_price=float(close[-1]), ob_imbalance=0.1)


def test_feature_vector_shape_and_finiteness():
    fv = compute_features(make_snapshot())
    arr = fv.as_array()
    assert len(arr) == len(FeatureVector.names())
    assert np.all(np.isfinite(arr))
    assert 0.0 <= fv.close_position <= 1.0


def test_features_detect_uptrend():
    fv = compute_features(make_snapshot(drift=0.002, vol=0.002))
    assert fv.slope_96 > 0
    assert fv.ret_72 > 0


def test_hmm_separates_volatility_regimes():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.001, 600)
    wild = rng.normal(0, 0.02, 600)
    r = np.concatenate([calm, wild])
    X = np.column_stack([r, np.abs(r)])
    hmm = StickyGaussianHMM(n_states=2, sticky=0.95)
    hmm.fit(X)
    p_calm = hmm.filter_posterior(X[:600][-200:])
    p_wild = hmm.filter_posterior(X[-200:])
    # the dominant state must differ between calm and wild segments
    assert int(np.argmax(p_calm)) != int(np.argmax(p_wild))
    assert max(p_calm) > 0.6 and max(p_wild) > 0.6
    labels = hmm.describe_states()
    assert len(labels) == 2 and labels[0] != labels[1]


def test_hmm_posterior_is_distribution():
    snap = make_snapshot()
    X = regime_observations(snap)
    hmm = StickyGaussianHMM(n_states=3)
    hmm.fit(X)
    p = hmm.filter_posterior(X[-100:])
    assert p.shape == (3,)
    assert abs(p.sum() - 1.0) < 1e-6
    assert np.all(p >= 0)


def test_bocpd_fires_on_regime_break():
    rng = np.random.default_rng(3)
    det = BOCPD(hazard=200.0)
    det.update_many(rng.normal(0, 0.002, 300))
    p_before = det.p_recent_changepoint(within=6)
    det.update_many(rng.normal(0.03, 0.03, 6))  # violent break
    p_after = det.p_recent_changepoint(within=8)
    assert p_after > p_before
    assert p_after > 0.3


def test_bocpd_quiet_on_stationary_data():
    rng = np.random.default_rng(4)
    det = BOCPD(hazard=200.0)
    det.update_many(rng.normal(0, 0.002, 400))
    assert det.expected_run_length() > 50
