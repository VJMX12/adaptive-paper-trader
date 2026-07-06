"""Episodic memory: retrieve historical trades most similar to the current setup.

kNN over z-scored entry feature vectors. Returns the analogues plus summary
statistics (win rate, avg R, common exit reasons) that feed both the
confidence blend and the reasoning narrative.
"""
from __future__ import annotations

import json

import numpy as np

from app.db.database import Database
from app.features.engine import FeatureVector

EPS = 1e-9


async def retrieve_similar(
    db: Database,
    current_features: FeatureVector,
    symbol: str | None,
    k: int = 15,
    same_direction: str | None = None,
) -> dict:
    """Return {'n_history', 'neighbors', 'win_rate', 'avg_r', 'common_mistakes'}."""
    trades = await db.get_closed_trades(symbol=None)  # learn across all symbols
    usable = []
    for t in trades:
        try:
            f = json.loads(t["features"])
            vec = np.array([f[name] for name in FeatureVector.names()], dtype=float)
        except Exception:
            continue
        if same_direction and t["direction"] != same_direction:
            continue
        usable.append((t, vec))

    result = {
        "n_history": len(usable), "neighbors": [], "win_rate": None,
        "avg_r": None, "common_exit_reasons": {}, "lessons": [],
    }
    if not usable:
        return result

    X = np.stack([v for _, v in usable])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    q = (current_features.as_array() - mu) / np.maximum(sd, EPS)
    Z = (X - mu) / np.maximum(sd, EPS)
    dists = np.linalg.norm(Z - q, axis=1)
    order = np.argsort(dists)[: min(k, len(usable))]

    wins, rs, reasons = 0, [], {}
    for i in order:
        t, _ = usable[i]
        win = 1 if (t["pnl_usd"] or 0) > 0 else 0
        wins += win
        if t["r_multiple"] is not None:
            rs.append(float(t["r_multiple"]))
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
        result["neighbors"].append({
            "trade_id": t["id"], "symbol": t["symbol"], "direction": t["direction"],
            "distance": round(float(dists[i]), 3),
            "regime": t["regime_label"],
            "r_multiple": t["r_multiple"], "exit_reason": t["exit_reason"],
            "pnl_pct": t["pnl_pct"],
        })
    n = len(order)
    result["win_rate"] = wins / n
    result["avg_r"] = float(np.mean(rs)) if rs else None
    result["common_exit_reasons"] = reasons
    result["lessons"] = await db.recent_lessons(symbol=symbol, limit=5)
    return result


def blend_confidence(model_prob: float, retrieval: dict, min_history: int) -> float:
    """Blend model probability with the empirical win rate of similar setups.

    Retrieval weight grows with the amount of history (capped at 0.35).
    """
    n = len(retrieval.get("neighbors") or [])
    wr = retrieval.get("win_rate")
    if wr is None or retrieval.get("n_history", 0) < min_history or n == 0:
        return model_prob
    w = min(0.35, n / 40.0)
    return float((1 - w) * model_prob + w * wr)
