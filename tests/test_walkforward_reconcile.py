"""Walk-forward validation windowing/metrics + order reconciliation."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.analysis.walkforward import assign_windows, _metrics, run_walk_forward
from app.db.database import Database


def _trade(day, won, prob, symbol="BTC/USDT:USDT"):
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    ts = (base + timedelta(days=day)).isoformat()
    pnl = 20.0 if won else -10.0
    return {"symbol": symbol, "direction": "long", "exit_ts": ts,
            "pnl_usd": pnl, "pnl_pct": pnl / 1000.0, "raw_prob": prob,
            "confidence": prob, "r_multiple": 2.0 if won else -1.0,
            "similar_trades": json.dumps({"win_rate": 0.6 if won else 0.4, "n": 10})}


def test_metrics_basic():
    trades = [_trade(0, True, 0.6), _trade(0, False, 0.55), _trade(0, True, 0.7)]
    m = _metrics(trades)
    assert m["n"] == 3
    assert abs(m["win_rate"] - 2 / 3) < 1e-3   # win_rate is rounded to 4dp
    assert 0.0 <= m["brier"] <= 1.0
    assert m["payoff_ratio"] is not None


def test_assign_windows_rolls():
    trades = [_trade(d, d % 2 == 0, 0.6) for d in range(0, 40)]
    w = assign_windows(trades, train_days=14, test_days=7, step_days=3)
    assert len(w) >= 3
    for win in w:
        assert "train" in win and "test" in win and win["window_id"]


def test_run_walk_forward_insufficient_data_is_honest():
    # only a handful of trades -> must report insufficient, not a false pass
    trades = [_trade(d, True, 0.6) for d in range(5)]
    s = run_walk_forward(trades, [], cfg=None)
    assert s["edge_validation"]["insufficient_data"] is True
    assert s["edge_validation"]["overall_pass"] is False
    assert "INSUFFICIENT" in s["edge_validation"]["status"]


def test_run_walk_forward_structure():
    trades = [_trade(d, d % 3 != 0, 0.6) for d in range(0, 60)]
    analyses = [{"symbol": "BTC/USDT:USDT",
                 "ts": (datetime(2026, 7, 1, tzinfo=timezone.utc)
                        + timedelta(hours=8 * i)).isoformat(),
                 "regime_label": f"r{i // 3}", "changepoint_prob": 0.01}
                for i in range(40)]
    s = run_walk_forward(trades, analyses, cfg=None)
    assert "overall_metrics" in s and "edge_validation" in s
    assert s["overall_metrics"]["n_test_trades"] >= 0
    assert "avg_state_duration_hours" in s["regime_analysis"]
    assert "knn_hit_rate" in s["memory_analysis"]


async def test_order_reconciliation_persistence(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    # persist a pending live order, then link + open
    await db.record_live_order({
        "order_id": "t1", "trade_id": 1, "symbol": "BTC/USDT:USDT",
        "direction": "long", "entry_price": 50000.0, "quantity": 0.004,
        "order_status": "pending",
        "regime_state_at_entry": {"label": "x"},
        "prediction_state_at_entry": {"raw_prob": 0.6}})
    await db.add_order_event("t1", "created", {"symbol": "BTC/USDT:USDT"})
    open_orders = await db.open_live_orders()
    assert len(open_orders) == 1 and open_orders[0]["order_status"] == "pending"

    await db.set_live_order("t1", status="open", bybit_order_id="bb-123")
    rows = await db.open_live_orders()
    assert rows[0]["order_status"] == "open" and rows[0]["bybit_order_id"] == "bb-123"

    # closing removes it from the open set
    await db.set_live_order("t1", status="closed")
    assert await db.open_live_orders() == []
    await db.close()


def test_dedupe_prevents_double_count():
    from app.analysis.walkforward import _dedupe
    t = [_trade(0, True, 0.6) for _ in range(3)]
    for i, x in enumerate(t):
        x["id"] = i
    dup = t + t  # same ids twice (as overlapping windows would produce)
    assert len(_dedupe(dup)) == 3


def test_sharpe_is_per_trade_not_sqrt_n_inflated():
    # The reported sharpe must be a per-trade effect size (mean/std), invariant
    # to sample size — NOT the old sqrt(n)-scaled statistic. Same per-trade
    # distribution at 20 vs 500 trades -> ~same sharpe, but t_stat grows ~5x.
    from app.analysis.walkforward import _metrics
    small = _metrics([_trade(i, i % 20 < 9, 0.55) for i in range(20)])
    big = _metrics([_trade(i, i % 20 < 9, 0.55) for i in range(500)])
    assert abs(small["sharpe"] - big["sharpe"]) < 0.15   # sharpe ~ invariant to n
    assert big["t_stat"] > 3 * small["t_stat"]           # t_stat scales with sqrt(n)


def test_stable_regime_passes_duration():
    # one unchanging regime over 24h must yield a large duration, not None
    from app.analysis.walkforward import regime_stability
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = [{"symbol": "BTC/USDT:USDT", "ts": (base + timedelta(hours=h)).isoformat(),
             "regime_label": "calm", "changepoint_prob": 0.01} for h in range(0, 25, 2)]
    r = regime_stability(rows)
    assert r["avg_state_duration_hours"] is not None
    assert r["avg_state_duration_hours"] > 6.0


async def test_shadow_setup_lifecycle(tmp_path):
    db = Database(str(tmp_path / "s.db")); await db.connect()
    await db.record_shadow_setup({
        "symbol": "BTC/USDT:USDT", "direction": "long", "entry_price": 100.0,
        "entry_ms": 1000, "stop_loss": 98.0, "take_profit": 104.0,
        "features": {"ret_1": 0.1}})
    opens = await db.open_shadow_setups("BTC/USDT:USDT")
    assert len(opens) == 1 and opens[0]["resolved"] == 0
    await db.resolve_shadow_setup(opens[0]["id"], 1, 1)   # resolved tp-first
    assert await db.open_shadow_setups("BTC/USDT:USDT") == []
    counts = await db.shadow_counts()
    assert counts["resolved"] == 1 and counts["resolved_win_rate"] == 1.0
    await db.close()


def test_ewma_normalizer_forgets():
    from app.analysis.model import OnlineLogistic
    m = OnlineLogistic(2, decay=0.9)
    for _ in range(200):
        m.observe(np.array([0.0, 0.0]))
    # then the distribution shifts; EWMA mean must move toward the new level
    for _ in range(50):
        m.observe(np.array([10.0, 10.0]))
    assert m.mean[0] > 5.0   # a frozen Welford mean would still be ~0


async def test_shadow_prune_keeps_recent(tmp_path):
    db = Database(str(tmp_path / "p.db")); await db.connect()
    for i in range(30):
        await db.record_shadow_setup({
            "symbol": "BTC/USDT:USDT", "direction": "long", "entry_price": 100.0,
            "entry_ms": 1000 + i, "stop_loss": 98.0, "take_profit": 104.0,
            "features": {"ret_1": 0.0}})
    # resolve all, then prune to keep 10
    for r in await db.open_shadow_setups("BTC/USDT:USDT", 100):
        await db.resolve_shadow_setup(r["id"], 1, 1)
    deleted = await db.prune_shadow_setups(keep=10)
    assert deleted == 20
    assert (await db.shadow_counts())["resolved"] == 10
    await db.close()


async def test_prune_analyses_by_rowcount(tmp_path):
    db = Database(str(tmp_path / "a.db")); await db.connect()
    for i in range(25):
        await db.insert_analysis({"symbol": "BTC/USDT:USDT", "price": 1.0,
            "bias": "neutral", "confidence": 0.5, "regime_posterior": [1.0],
            "changepoint_prob": 0.0, "features": {}, "trade_recommended": False})
    assert await db.analyses_count() == 25
    deleted = await db.prune_analyses(keep_rows=10)
    assert deleted == 15
    assert await db.analyses_count() == 10
    await db.close()
