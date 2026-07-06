"""Performance analytics over the learning database."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np

from app.db.database import Database


async def compute_metrics(db: Database, starting_equity: float) -> dict:
    trades = await db.get_closed_trades()
    curve = await db.equity_curve()
    out: dict = {
        "n_trades": len(trades),
        "open_positions": await db.open_positions_count(),
        "equity": await db.latest_equity(starting_equity),
        "starting_equity": starting_equity,
    }
    if not trades:
        return out

    pnl = np.array([t["pnl_usd"] or 0.0 for t in trades])
    rs = np.array([t["r_multiple"] or 0.0 for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]

    out["win_rate"] = round(float((pnl > 0).mean()), 4)
    out["avg_r"] = round(float(rs.mean()), 4)
    out["profit_factor"] = (
        round(float(wins.sum() / abs(losses.sum())), 4)
        if losses.sum() != 0 else None
    )
    out["expectancy_usd"] = round(float(pnl.mean()), 4)
    out["expectancy_r"] = round(float(rs.mean()), 4)
    out["total_pnl_usd"] = round(float(pnl.sum()), 2)

    # --- drawdown & Sharpe from equity curve ---
    if curve:
        eq = np.array([c["equity"] for c in curve])
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(peak, 1e-9)
        out["max_drawdown_pct"] = round(float(dd.max()) * 100, 3)
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
        if len(rets) > 2 and rets.std() > 0:
            out["sharpe_per_trade"] = round(float(rets.mean() / rets.std()
                                                  * np.sqrt(len(rets))), 3)

    # --- monthly returns ---
    monthly: dict[str, float] = defaultdict(float)
    for t in trades:
        try:
            m = datetime.fromisoformat(t["exit_ts"]).strftime("%Y-%m")
            monthly[m] += t["pnl_usd"] or 0.0
        except (ValueError, TypeError):
            continue
    out["monthly_pnl_usd"] = {k: round(v, 2) for k, v in sorted(monthly.items())}

    # --- performance by regime / by symbol+direction ---
    by_regime: dict[str, list[float]] = defaultdict(list)
    by_setup: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_regime[t["regime_label"] or "unknown"].append(t["r_multiple"] or 0.0)
        by_setup[f"{t['symbol']} {t['direction']}"].append(t["r_multiple"] or 0.0)

    def summarize(groups: dict[str, list[float]]) -> dict:
        return {
            k: {"n": len(v), "avg_r": round(float(np.mean(v)), 3),
                "win_rate": round(float(np.mean([x > 0 for x in v])), 3)}
            for k, v in groups.items()
        }

    out["by_regime"] = summarize(by_regime)
    setups = summarize(by_setup)
    out["by_setup"] = setups
    ranked = sorted(setups.items(), key=lambda kv: kv[1]["avg_r"])
    if ranked:
        out["worst_setup"] = {ranked[0][0]: ranked[0][1]}
        out["best_setup"] = {ranked[-1][0]: ranked[-1][1]}

    # --- confidence accuracy (were stated probabilities honest?) ---
    conf = np.array([t["confidence"] or 0.5 for t in trades])
    won = (pnl > 0).astype(float)
    out["brier_score"] = round(float(np.mean((conf - won) ** 2)), 4)
    buckets = []
    edges = np.linspace(0.5, 1.0, 6)
    for i in range(5):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] + (i == 4))
        if mask.sum():
            buckets.append({
                "confidence_bucket": f"{edges[i]:.2f}-{edges[i+1]:.2f}",
                "n": int(mask.sum()),
                "stated": round(float(conf[mask].mean()), 3),
                "actual_win_rate": round(float(won[mask].mean()), 3),
            })
    out["confidence_calibration"] = buckets
    return out
