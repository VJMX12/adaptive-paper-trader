import numpy as np

from app.analysis.model import CalibrationTracker, OnlineLogistic
from app.config import AppConfig
from app.trading.risk import RiskManager


def make_cfg():
    return AppConfig(raw={
        "risk": {"starting_equity": 10000, "base_risk_pct": 0.01,
                 "kelly_fraction": 0.25, "max_open_positions": 3,
                 "vol_target_annual": 0.35, "drawdown_soft_pct": 0.05,
                 "drawdown_hard_pct": 0.15, "max_daily_loss_pct": 0.04},
        "changepoint": {"alert_threshold": 0.35},
    })


def test_online_logistic_learns_separable_pattern():
    rng = np.random.default_rng(0)
    model = OnlineLogistic(n_features=4, lr=0.1)
    for _ in range(600):
        x = rng.normal(0, 1, 4)
        y = int(x[0] + 0.5 * x[1] > 0)
        model.update(x, y)
    p_pos = model.predict_proba(np.array([2.0, 1.0, 0.0, 0.0]))
    p_neg = model.predict_proba(np.array([-2.0, -1.0, 0.0, 0.0]))
    assert p_pos > 0.8 and p_neg < 0.2


def test_calibration_shrinks_when_dishonest():
    cal = CalibrationTracker()
    for _ in range(30):          # always claims 90%, always loses
        cal.record(0.9, 0)
    assert cal.calibration_score() < 0.2
    assert cal.shrink(0.9) < 0.7  # confidence pulled hard toward 0.5


def test_calibration_keeps_honest_confidence():
    cal = CalibrationTracker()
    rng = np.random.default_rng(1)
    for _ in range(60):          # claims 70%, wins ~70%
        cal.record(0.7, int(rng.random() < 0.7))
    assert cal.shrink(0.7) > 0.62


def test_risk_multipliers_reduce_size_under_uncertainty():
    rm = RiskManager(make_cfg())
    base = dict(equity=10000, peak_equity=10000, pnl_today=0, open_positions=0,
                entry=100.0, stop=98.0, min_confidence=0.6,
                sigma_per_candle=0.002, candles_per_year=105120)
    confident = rm.size_position(**base, confidence=0.85, cp_prob=0.0)
    nervous = rm.size_position(**base, confidence=0.62, cp_prob=0.30)
    assert confident.allowed
    assert confident.risk_amount > nervous.risk_amount or not nervous.allowed


def test_circuit_breaker_hard_drawdown():
    rm = RiskManager(make_cfg())
    d = rm.size_position(
        equity=8000, peak_equity=10000, pnl_today=0, open_positions=0,
        entry=100.0, stop=98.0, confidence=0.9, min_confidence=0.6,
        sigma_per_candle=0.002, candles_per_year=105120, cp_prob=0.0)
    assert not d.allowed
    assert "drawdown" in d.reason


def test_circuit_breaker_daily_loss_and_max_positions():
    rm = RiskManager(make_cfg())
    d = rm.size_position(
        equity=10000, peak_equity=10000, pnl_today=-500, open_positions=0,
        entry=100.0, stop=98.0, confidence=0.9, min_confidence=0.6,
        sigma_per_candle=0.002, candles_per_year=105120, cp_prob=0.0)
    assert not d.allowed and "daily loss" in d.reason
    d2 = rm.size_position(
        equity=10000, peak_equity=10000, pnl_today=0, open_positions=3,
        entry=100.0, stop=98.0, confidence=0.9, min_confidence=0.6,
        sigma_per_candle=0.002, candles_per_year=105120, cp_prob=0.0)
    assert not d2.allowed and "max open positions" in d2.reason


def test_changepoint_collapses_exposure():
    rm = RiskManager(make_cfg())
    base = dict(equity=10000, peak_equity=10000, pnl_today=0, open_positions=0,
                entry=100.0, stop=98.0, confidence=0.85, min_confidence=0.6,
                sigma_per_candle=0.002, candles_per_year=105120)
    calm = rm.size_position(**base, cp_prob=0.0)
    alarm = rm.size_position(**base, cp_prob=0.40)
    assert calm.allowed
    assert (not alarm.allowed) or alarm.risk_amount < 0.2 * calm.risk_amount
