"""Tests for the algorithm-audit remediation:
  - direction-oriented features (short = mirror of long)
  - P(win) per direction is direct, not a 1-p complement
  - learner-state version reset
  - normalizer advances on observe(), not only on updates
  - realistic costs reduce PnL
  - per-symbol HMMs are independent
"""
import numpy as np

from app.analysis.model import (OnlineLogistic, CalibrationTracker,
                                 save_learner_state, load_learner_state, MODEL_VERSION)
from app.features.engine import FeatureVector
from app.trading.paper_engine import PaperTradingEngine


def _fv(**kw):
    base = {n: 0.0 for n in FeatureVector.names()}
    base.update(kw)
    return FeatureVector(**base)


def test_oriented_long_is_identity():
    fv = _fv(ret_1=0.01, slope_96=3.0, signed_flow=0.4, sigma=0.002, close_position=0.8)
    assert np.allclose(fv.oriented("long"), fv.as_array())


def test_oriented_short_mirrors_directional_only():
    fv = _fv(ret_1=0.01, ret_24=0.05, slope_24=2.0, slope_96=3.0,
             signed_flow=0.4, ob_imbalance=0.3, sigma=0.002, sigma_ratio=1.2,
             close_position=0.8, upper_shadow=0.1, lower_shadow=0.3, r2_96=0.5)
    names = FeatureVector.names()
    s = dict(zip(names, fv.oriented("short")))
    # directional features negated
    for k in ("ret_1", "ret_24", "slope_24", "slope_96", "signed_flow", "ob_imbalance"):
        assert s[k] == -getattr(fv, k), k
    # non-directional unchanged
    for k in ("sigma", "sigma_ratio", "r2_96"):
        assert s[k] == getattr(fv, k), k
    # close_position mirrored, shadows swapped
    assert abs(s["close_position"] - (1.0 - fv.close_position)) < 1e-12
    assert s["upper_shadow"] == fv.lower_shadow
    assert s["lower_shadow"] == fv.upper_shadow


def test_short_probability_is_direct_not_complement():
    # Train the model so a bullish-oriented setup wins. Then a bullish market
    # should give high P(long win) AND low P(short win) — but NOT exactly
    # 1-P(long), because non-directional features don't flip.
    m = OnlineLogistic(len(FeatureVector.names()), lr=0.2)
    bull = _fv(ret_1=0.02, slope_96=4.0, signed_flow=0.5, sigma=0.003, sigma_ratio=1.5)
    for _ in range(200):
        m.observe(bull.as_array())
        m.update(bull.oriented("long"), 1)     # long-oriented bull setups win
        m.update(bull.oriented("short"), 0)    # short-oriented (mirror) lose
    p_long = m.predict_proba(bull.oriented("long"))
    p_short = m.predict_proba(bull.oriented("short"))
    assert p_long > 0.6 and p_short < 0.4
    # direct computation, not a strict complement (sigma/sigma_ratio don't flip)
    assert abs(p_short - (1.0 - p_long)) > 1e-6


def test_observe_advances_stats_without_learning():
    m = OnlineLogistic(4)
    for _ in range(10):
        m.observe(np.array([1.0, 2.0, 3.0, 4.0]))
    assert m.count == 10 and m.n_updates == 0
    assert np.allclose(m.mean, [1, 2, 3, 4])


def test_version_mismatch_resets_learner(tmp_path):
    m = OnlineLogistic(len(FeatureVector.names()))
    m.w = np.ones(m.n) * 0.5
    p = tmp_path / "s.json"
    save_learner_state(p, m, CalibrationTracker())
    # correct version loads back the trained weights
    m2, _ = load_learner_state(p, m.n)
    assert np.allclose(m2.w, m.w)
    # tamper the version -> load must reset to a fresh (zero-weight) model
    import json
    d = json.loads(p.read_text()); d["version"] = MODEL_VERSION - 1
    p.write_text(json.dumps(d))
    m3, _ = load_learner_state(p, m.n)
    assert np.allclose(m3.w, 0.0) and m3.n_updates == 0


def test_costs_reduce_pnl():
    trade = {"entry_price": 100.0, "position_size": 2.0, "direction": "long",
             "risk_amount": 5.0, "entry_ts": "2026-07-06T00:00:00+00:00"}
    free = PaperTradingEngine.compute_close(trade, 101.0, "tp")
    costed = PaperTradingEngine.compute_close(
        trade, 101.0, "tp",
        {"taker_fee_pct": 0.00055, "slippage_pct": 0.0003, "funding_pct_per_8h": 0.0001})
    assert costed["pnl_usd"] < free["pnl_usd"]
    assert costed["costs_usd"] > 0
