import json

import numpy as np
import pytest

from app.analysis.model import CalibrationTracker
from app.config import AppConfig
from app.dashboard.metrics import compute_metrics
from app.db.database import Database
from app.db.retrieval import blend_confidence, retrieve_similar
from app.features.engine import FeatureVector
from app.journal.review import build_review, write_review
from app.trading.paper_engine import PaperTradingEngine
from app.trading.risk import RiskManager

pytestmark = pytest.mark.asyncio


def cfg():
    return AppConfig(raw={
        "strategy": {"cooldown_minutes": 0, "min_confidence": 0.6},
        "risk": {"starting_equity": 10000, "base_risk_pct": 0.01,
                 "kelly_fraction": 0.25, "max_open_positions": 3,
                 "vol_target_annual": 0.35, "drawdown_soft_pct": 0.05,
                 "drawdown_hard_pct": 0.15, "max_daily_loss_pct": 0.04},
        "changepoint": {"alert_threshold": 0.35},
    })


def fake_features(seed=0) -> dict:
    rng = np.random.default_rng(seed)
    return {n: float(rng.normal()) for n in FeatureVector.names()}


async def make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    return db


async def open_and_close(db, symbol="BTC/USDT", direction="long",
                         entry=100.0, exit_price=104.0, reason="tp", seed=0):
    tid = await db.open_trade({
        "symbol": symbol, "direction": direction, "entry_price": entry,
        "stop_loss": entry - 2 if direction == "long" else entry + 2,
        "take_profit": entry + 4 if direction == "long" else entry - 4,
        "position_size": 10.0, "risk_amount": 20.0, "rr": 2.0,
        "confidence": 0.7, "raw_prob": 0.7, "regime_label": "low-vol up-drift",
        "changepoint_prob": 0.05, "features": fake_features(seed),
        "reasoning": "test", "similar_trades": {},
    })
    trade = await db.get_trade(tid)
    close = PaperTradingEngine.compute_close(trade, exit_price, reason)
    await db.close_trade(tid, close)
    return await db.get_trade(tid)


async def test_trade_lifecycle_and_pnl_math(tmp_path):
    db = await make_db(tmp_path)
    t = await open_and_close(db, exit_price=104.0, reason="tp")
    assert t["status"] == "closed"
    assert t["pnl_usd"] == pytest.approx(40.0)      # (104-100)*10
    assert t["pnl_pct"] == pytest.approx(4.0)
    assert t["r_multiple"] == pytest.approx(2.0)    # 40 / 20 risk
    # short loss math
    t2 = await open_and_close(db, direction="short", entry=100.0,
                              exit_price=102.0, reason="sl", seed=1)
    assert t2["pnl_usd"] == pytest.approx(-20.0)
    assert t2["r_multiple"] == pytest.approx(-1.0)
    await db.close()


async def test_retrieval_finds_similar_and_blends(tmp_path):
    db = await make_db(tmp_path)
    for i in range(12):  # winners with similar features
        await open_and_close(db, exit_price=104.0, reason="tp", seed=i % 3)
    fv = FeatureVector(**fake_features(0))
    r = await retrieve_similar(db, fv, "BTC/USDT", k=5, same_direction="long")
    assert r["n_history"] == 12
    assert len(r["neighbors"]) == 5
    assert r["win_rate"] == 1.0
    blended = blend_confidence(0.6, r, min_history=8)
    assert blended > 0.6  # perfect history pulls confidence up
    await db.close()


async def test_journal_review_written(tmp_path):
    db = await make_db(tmp_path)
    t = await open_and_close(db, exit_price=98.0, reason="sl")
    review, lessons = build_review(t, calibration_score=0.5)
    assert review["outcome"] == "loss"
    assert "entry_timing" in review and "sl_tp_assessment" in review
    c = AppConfig(raw={"llm": {"enabled": False}})
    review2, lessons2, prose = await write_review(c, db, t, 0.5)
    assert prose is None
    assert (await db.recent_lessons())[0] == lessons2
    await db.close()


async def test_metrics_computation(tmp_path):
    db = await make_db(tmp_path)
    await db.record_equity(10000, "start")
    t1 = await open_and_close(db, exit_price=104.0, reason="tp")
    await db.record_equity(10000 + t1["pnl_usd"])
    t2 = await open_and_close(db, exit_price=98.0, reason="sl", seed=2)
    await db.record_equity(10000 + t1["pnl_usd"] + t2["pnl_usd"])
    m = await compute_metrics(db, 10000)
    assert m["n_trades"] == 2
    assert m["win_rate"] == 0.5
    assert m["profit_factor"] == pytest.approx(2.0)   # +40 / -20
    assert m["expectancy_usd"] == pytest.approx(10.0)
    assert "by_regime" in m and "confidence_calibration" in m
    assert m["max_drawdown_pct"] >= 0
    await db.close()


async def test_paper_engine_blocks_duplicate_symbol(tmp_path):
    db = await make_db(tmp_path)
    await db.open_trade({
        "symbol": "BTC/USDT", "direction": "long", "entry_price": 100,
        "stop_loss": 98, "take_profit": 104, "position_size": 1,
        "risk_amount": 2, "rr": 2, "confidence": 0.7, "raw_prob": 0.7,
        "regime_label": "x", "changepoint_prob": 0.0,
        "features": fake_features(), "reasoning": "", "similar_trades": {},
    })
    assert await db.has_open_trade("BTC/USDT")
    assert await db.open_positions_count() == 1
    await db.close()
