"""Walk-forward / prequential validation over the live-accumulated trade log.

Why prequential (not a separate frozen backtest): the online learner predicts
every trade's outcome BEFORE it resolves and only updates afterward, so each
stored prediction (raw_prob) is already out-of-sample by construction. This
module partitions the closed-trade history into rolling train/test windows for
REPORTING and computes, per test window: profitability, calibration
(Brier/ECE), regime stability and train-vs-test degradation — scored against
the audit's edge-validation criteria. It answers "is the edge real out of
sample?" from real forward data as it accumulates.

Pure functions over already-fetched rows so they are trivially testable.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

# audit pass criteria
PASS = {
    "min_test_win_rate": 0.35,       # 2:1 payoff break-even (with costs)
    "min_sharpe_per_trade": 0.10,    # effect size (mean/std of per-trade PnL)
    "min_tstat": 2.0,                # statistical significance (mean/std*sqrt(n))
    "max_test_brier": 0.20,
    "min_degradation": -0.12,        # test_wr - train_wr must exceed this
    "max_consistency_std": 0.10,
    "min_regime_hours": 6.0,
    "min_knn_hit_rate": 0.70,
    "min_trades_per_window": 8,      # a window below this is "insufficient"
    "min_test_windows": 4,
}


def resolve_barrier(entry_ms, direction, stop_loss, take_profit,
                    ts, high, low, tie_to_sl: bool = True):
    """Resolve a shadow setup against the forward price path: did TP hit before
    SL? Returns 1 (TP-first), 0 (SL-first), or None (neither yet).

    NO LOOKAHEAD: only candles STRICTLY AFTER the entry candle (ts > entry_ms)
    are considered, so the label never sees information from at/before entry.
    Intrabar ambiguity (both barriers first touched in the SAME candle — order
    unknowable from OHLC) resolves to SL by default (conservative), since the
    nearer barrier is the more likely first touch; assuming TP would inflate the
    win rate and the training labels.
    """
    ts = np.asarray(ts, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    fwd = ts > entry_ms                      # strictly forward — the leak guard
    if not bool(fwd.any()):
        return None
    fh, fl = high[fwd], low[fwd]
    if direction == "long":
        tp_hit, sl_hit = fh >= take_profit, fl <= stop_loss
    else:
        tp_hit, sl_hit = fl <= take_profit, fh >= stop_loss
    tp_idx = int(np.argmax(tp_hit)) if tp_hit.any() else None
    sl_idx = int(np.argmax(sl_hit)) if sl_hit.any() else None
    if tp_idx is None and sl_idx is None:
        return None
    if sl_idx is None:
        return 1
    if tp_idx is None:
        return 0
    if tp_idx < sl_idx:
        return 1
    if sl_idx < tp_idx:
        return 0
    return 0 if tie_to_sl else 1             # same candle: conservative -> SL


def _dedupe(trades: list[dict]) -> list[dict]:
    """Overlapping step windows share trades; dedupe before any overall
    aggregate so a trade is never counted 2-3x (which biases the verdict)."""
    seen, out = set(), []
    for t in trades:
        key = t.get("id")
        if key is None:
            key = (t.get("exit_ts"), t.get("symbol"), t.get("entry_price"))
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _ts(s):
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _won(t) -> int:
    return int((t.get("pnl_usd") or 0.0) > 0)


def _metrics(trades: list[dict]) -> dict:
    """Profitability + calibration for one set of trades."""
    n = len(trades)
    if n == 0:
        return {"n": 0}
    won = np.array([_won(t) for t in trades], dtype=float)
    pnl_pct = np.array([float(t.get("pnl_pct") or 0.0) for t in trades])
    pnl_usd = np.array([float(t.get("pnl_usd") or 0.0) for t in trades])
    # predicted P(win): raw_prob is now a direct per-direction P(win)
    p = np.array([float(t.get("raw_prob") if t.get("raw_prob") is not None
                        else t.get("confidence") or 0.5) for t in trades])

    win_rate = float(won.mean())
    wins, losses = pnl_pct[pnl_pct > 0], pnl_pct[pnl_pct <= 0]
    payoff = (float(wins.mean() / abs(losses.mean()))
              if len(wins) and len(losses) and losses.mean() != 0 else None)
    ret_std = float(pnl_pct.std(ddof=1)) if n > 1 else 0.0
    # per-trade Sharpe = effect size; t_stat = significance (scales with sqrt n).
    # Reporting both stops a weak per-trade edge from "passing" just by n.
    # Guard tiny std (e.g. all-identical outcomes) to avoid a blown-up ratio.
    sharpe = float(pnl_pct.mean() / ret_std) if ret_std > 1e-9 else 0.0
    t_stat = float(sharpe * np.sqrt(n))
    # max drawdown on cumulative USD pnl
    cum = np.cumsum(pnl_usd)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    dd = float(np.max(peak - np.concatenate([[0.0], cum]))) if n else 0.0

    brier = float(np.mean((p - won) ** 2))
    # ECE: 5 equal bins over [0,1]
    ece, edges = 0.0, np.linspace(0, 1, 6)
    for i in range(5):
        m = (p >= edges[i]) & (p < edges[i + 1] + (i == 4))
        if m.sum():
            ece += (m.sum() / n) * abs(float(p[m].mean()) - float(won[m].mean()))
    return {
        "n": n, "win_rate": round(win_rate, 4),
        "mean_pnl_pct": round(float(pnl_pct.mean()), 4),
        "total_pnl_usd": round(float(pnl_usd.sum()), 2),
        "sharpe": round(sharpe, 3), "t_stat": round(t_stat, 3),
        "max_drawdown_usd": round(dd, 2),
        "payoff_ratio": round(payoff, 3) if payoff is not None else None,
        "brier": round(brier, 4), "ece": round(ece, 4),
    }


def _knn_analysis(trades: list[dict]) -> dict:
    """Was the episodic-memory win-rate predictive of the actual outcome?"""
    import json
    hits, preds, outs = 0, [], []
    n = 0
    for t in trades:
        st = t.get("similar_trades")
        if isinstance(st, str):
            try:
                st = json.loads(st)
            except ValueError:
                st = None
        if not st or st.get("win_rate") is None:
            continue
        wr = float(st["win_rate"])
        won = _won(t)
        n += 1
        hits += int((wr >= 0.5) == bool(won))
        preds.append(wr)
        outs.append(won)
    hit_rate = round(hits / n, 4) if n else None
    corr = None
    if n >= 3 and np.std(preds) > 0 and np.std(outs) > 0:
        corr = round(float(np.corrcoef(preds, outs)[0, 1]), 4)
    return {"n": n, "knn_hit_rate": hit_rate, "win_rate_correlation": corr}


def regime_stability(analyses: list[dict]) -> dict:
    """Average regime-state duration and changepoint frequency, per symbol,
    from the analysis feed (analyses are spaced by the analysis interval)."""
    by_sym: dict[str, list[dict]] = {}
    for a in analyses:
        by_sym.setdefault(a["symbol"], []).append(a)
    durations_h, cp_events, total = [], 0, 0
    span_hours = 0.0
    for sym, rows in by_sym.items():
        rows = sorted(rows, key=lambda r: r.get("ts") or "")
        if len(rows) < 2:
            continue
        t0, t1 = _ts(rows[0]["ts"]), _ts(rows[-1]["ts"])
        if t0 and t1:
            span_hours += (t1 - t0).total_seconds() / 3600.0
        run_start = _ts(rows[0]["ts"])
        prev_label = rows[0].get("regime_label")
        for r in rows[1:]:
            total += 1
            if (r.get("changepoint_prob") or 0) >= 0.35:
                cp_events += 1
            if r.get("regime_label") != prev_label:
                t_now, ts_run = _ts(r["ts"]), run_start
                if t_now and ts_run:
                    durations_h.append((t_now - ts_run).total_seconds() / 3600.0)
                run_start = _ts(r["ts"])
                prev_label = r.get("regime_label")
        # count the trailing (still-open) run — otherwise a very STABLE regime
        # with few/no changes yields no durations and wrongly fails the gate.
        t_end, ts_run = _ts(rows[-1]["ts"]), run_start
        if t_end and ts_run and t_end > ts_run:
            durations_h.append((t_end - ts_run).total_seconds() / 3600.0)
    avg_dur = round(float(np.mean(durations_h)), 2) if durations_h else None
    cp_per_day = round(cp_events / span_hours * 24.0, 3) if span_hours > 0 else None
    return {"avg_state_duration_hours": avg_dur,
            "changepoint_signals_per_day": cp_per_day,
            "n_analyses": len(analyses)}


def assign_windows(trades: list[dict], train_days: int, test_days: int,
                   step_days: int) -> list[dict]:
    """Rolling train/test partition keyed on exit time. Each window's TEST set
    is the trades that closed in [train_end, train_end+test)."""
    dated = [(t, _ts(t.get("exit_ts"))) for t in trades]
    dated = [(t, d) for t, d in dated if d is not None]
    if not dated:
        return []
    dated.sort(key=lambda x: x[1])
    start, end = dated[0][1], dated[-1][1]
    windows, cur = [], start
    while cur + timedelta(days=train_days) <= end + timedelta(days=step_days):
        tr0, tr1 = cur, cur + timedelta(days=train_days)
        te0, te1 = tr1, tr1 + timedelta(days=test_days)
        train = [t for t, d in dated if tr0 <= d < tr1]
        test = [t for t, d in dated if te0 <= d < te1]
        windows.append({
            "window_id": f"{tr0.date()}_to_{te1.date()}",
            "train_start": tr0.isoformat(), "test_start": te0.isoformat(),
            "test_end": te1.isoformat(),
            "train": train, "test": test,
        })
        cur += timedelta(days=step_days)
    return windows


def run_walk_forward(closed_trades: list[dict], analyses: list[dict],
                     cfg=None) -> dict:
    g = (lambda k, d: cfg.get(k, d)) if cfg else (lambda k, d: d)
    train_days = int(g("validation.train_days", 14))
    test_days = int(g("validation.test_days", 7))
    step_days = int(g("validation.step_days", 3))

    windows = assign_windows(closed_trades, train_days, test_days, step_days)
    per_window, test_wrs, test_briers, test_sharpes = [], [], [], []
    for w in windows:
        tr_m, te_m = _metrics(w["train"]), _metrics(w["test"])
        if te_m.get("n", 0) >= PASS["min_trades_per_window"]:
            test_wrs.append(te_m["win_rate"])
            test_briers.append(te_m["brier"])
            test_sharpes.append(te_m["sharpe"])
        per_window.append({
            "window_id": w["window_id"], "test_start": w["test_start"],
            "test_end": w["test_end"], "train": tr_m, "test": te_m,
            "degradation": (round(te_m["win_rate"] - tr_m["win_rate"], 4)
                            if te_m.get("n") and tr_m.get("n") else None),
        })

    # dedupe across overlapping windows before the overall aggregate
    overall_test = _metrics(_dedupe([t for w in windows for t in w["test"]]))
    overall_train = _metrics(_dedupe([t for w in windows for t in w["train"]]))
    consistency = round(float(np.std(test_wrs)), 4) if len(test_wrs) >= 2 else None
    degradation = (round(overall_test.get("win_rate", 0) - overall_train.get("win_rate", 0), 4)
                   if overall_test.get("n") and overall_train.get("n") else None)
    regime = regime_stability(analyses)
    mem = _knn_analysis(closed_trades)

    n_valid_windows = len(test_wrs)
    insufficient = (n_valid_windows < PASS["min_test_windows"]
                    or overall_test.get("n", 0) < PASS["min_trades_per_window"])

    def ok(cond):  # None-safe gate
        return bool(cond)

    checks = {
        "pass_win_rate": ok(overall_test.get("win_rate", 0) > PASS["min_test_win_rate"]),
        "pass_sharpe": ok(overall_test.get("sharpe", 0) > PASS["min_sharpe_per_trade"]
                          and overall_test.get("t_stat", 0) > PASS["min_tstat"]),
        "pass_calibration": ok(overall_test.get("brier", 1) < PASS["max_test_brier"]),
        "pass_degradation": ok(degradation is not None and degradation > PASS["min_degradation"]),
        "pass_consistency": ok(consistency is not None and consistency < PASS["max_consistency_std"]),
        "pass_regime": ok(regime.get("avg_state_duration_hours") is not None
                          and regime["avg_state_duration_hours"] > PASS["min_regime_hours"]),
        "pass_memory": ok(mem.get("knn_hit_rate") is not None
                          and mem["knn_hit_rate"] > PASS["min_knn_hit_rate"]),
    }
    overall_pass = (not insufficient) and all(checks.values())
    if insufficient:
        status = (f"INSUFFICIENT DATA — {overall_test.get('n', 0)} test trades in "
                  f"{n_valid_windows} valid windows; need "
                  f">={PASS['min_test_windows']} windows of "
                  f">={PASS['min_trades_per_window']} trades. Keep collecting.")
    elif overall_pass:
        status = "EDGE VALIDATED out-of-sample (all criteria met — monitor for drift)."
    else:
        failed = [k for k, v in checks.items() if not v]
        status = "EDGE NOT VALIDATED — failing: " + ", ".join(failed)

    return {
        "config": {"train_days": train_days, "test_days": test_days,
                   "step_days": step_days},
        "total_windows": len(windows), "valid_test_windows": n_valid_windows,
        "overall_metrics": {
            "test_win_rate": overall_test.get("win_rate"),
            "test_sharpe": overall_test.get("sharpe"),      # per-trade (effect size)
            "test_t_stat": overall_test.get("t_stat"),      # significance
            "test_brier": overall_test.get("brier"),
            "test_ece": overall_test.get("ece"),
            "train_win_rate": overall_train.get("win_rate"),
            "n_test_trades": overall_test.get("n", 0),
            "n_train_trades": overall_train.get("n", 0),
            "degradation": degradation, "consistency_std": consistency,
        },
        "regime_analysis": regime,
        "memory_analysis": mem,
        "per_window": per_window,
        "edge_validation": {**checks, "overall_pass": overall_pass,
                            "insufficient_data": insufficient, "status": status},
    }
