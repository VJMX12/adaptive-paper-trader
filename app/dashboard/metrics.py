"""Performance analytics over the learning database."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from app.db.database import Database


async def compute_metrics(db: Database, starting_equity: float,
                          universe: set[str] | None = None) -> dict:
    """`universe`, when given, is the set of symbols currently traded (e.g.
    the live config's symbol list). Financial totals (equity, total PnL,
    period buckets) always cover ALL closed trades — that's real account
    history and shouldn't be hidden. But *strategy-quality* diagnostics
    (win rate, profit factor, expectancy, regime/setup breakdown,
    calibration) are computed only over `universe` trades: legacy trades on
    symbols no longer traded (e.g. thin synthetic/commodity pairs dropped
    from an earlier, wider universe) skew these with execution artifacts —
    extreme slippage on illiquid instruments — that have nothing to do with
    how the current strategy performs on its current symbol list.
    """
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

    pnl_all = np.array([t["pnl_usd"] or 0.0 for t in trades])
    out["total_pnl_usd"] = round(float(pnl_all.sum()), 2)

    strategy_trades = ([t for t in trades if t["symbol"] in universe]
                       if universe else trades)
    out["legacy_trades_excluded_from_stats"] = len(trades) - len(strategy_trades)
    if not strategy_trades:
        return out

    pnl = np.array([t["pnl_usd"] or 0.0 for t in strategy_trades])
    rs = np.array([t["r_multiple"] or 0.0 for t in strategy_trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]

    out["win_rate"] = round(float((pnl > 0).mean()), 4)
    out["avg_r"] = round(float(rs.mean()), 4)
    out["profit_factor"] = (
        round(float(wins.sum() / abs(losses.sum())), 4)
        if losses.sum() != 0 else None
    )
    out["expectancy_usd"] = round(float(pnl.mean()), 4)
    out["expectancy_r"] = round(float(rs.mean()), 4)

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

    # --- daily / weekly / monthly PnL buckets (for time-range views) ---
    daily: dict[str, float] = defaultdict(float)
    weekly: dict[str, float] = defaultdict(float)
    monthly: dict[str, float] = defaultdict(float)
    for t in trades:
        try:
            dt = datetime.fromisoformat(t["exit_ts"])
        except (ValueError, TypeError):
            continue
        v = t["pnl_usd"] or 0.0
        daily[dt.strftime("%Y-%m-%d")] += v
        iso_year, iso_week, _ = dt.isocalendar()
        weekly[f"{iso_year}-W{iso_week:02d}"] += v
        monthly[dt.strftime("%Y-%m")] += v
    out["daily_pnl_usd"] = {k: round(v, 2) for k, v in sorted(daily.items())}
    out["weekly_pnl_usd"] = {k: round(v, 2) for k, v in sorted(weekly.items())}
    out["monthly_pnl_usd"] = {k: round(v, 2) for k, v in sorted(monthly.items())}

    # --- convenience rollups for "this/last week/month" dashboard views ---
    now = datetime.now()
    today = now.date()
    week_start = today - timedelta(days=today.weekday())
    last_week_start = week_start - timedelta(days=7)
    month_start = today.replace(day=1)
    last_month_end = month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)

    def _sum_daily(d0, d1_exclusive) -> tuple[float, int]:
        total, n = 0.0, 0
        for k, v in daily.items():
            d = datetime.strptime(k, "%Y-%m-%d").date()
            if d0 <= d < d1_exclusive:
                total += v
                n += 1
        return round(total, 2), n

    tomorrow = today + timedelta(days=1)
    pnl_today, n_today = _sum_daily(today, tomorrow)
    pnl_this_week, n_this_week = _sum_daily(week_start, tomorrow)
    pnl_last_week, n_last_week = _sum_daily(last_week_start, week_start)
    pnl_this_month, n_this_month = _sum_daily(month_start, tomorrow)
    pnl_last_month, n_last_month = _sum_daily(last_month_start, month_start)
    out["pnl_ranges"] = {
        "today": {"pnl_usd": pnl_today, "days_with_trades": n_today},
        "this_week": {"pnl_usd": pnl_this_week, "days_with_trades": n_this_week},
        "last_week": {"pnl_usd": pnl_last_week, "days_with_trades": n_last_week},
        "this_month": {"pnl_usd": pnl_this_month, "days_with_trades": n_this_month},
        "last_month": {"pnl_usd": pnl_last_month, "days_with_trades": n_last_month},
    }

    # --- performance by regime / by symbol+direction ---
    by_regime: dict[str, list[float]] = defaultdict(list)
    by_setup: dict[str, list[float]] = defaultdict(list)
    for t in strategy_trades:
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
    conf = np.array([t["confidence"] or 0.5 for t in strategy_trades])
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
