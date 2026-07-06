"""Self-learning journal: structured post-trade review after every close.

Deterministic diagnostics from MAE/MFE + outcome, with optional LLM prose
on top. The structured review is what the learning database stores; the
`lessons` string is fed back into future reasoning via retrieval.
"""
from __future__ import annotations

import json

from app.analysis.reasoning import REVIEW_SYSTEM, llm_narrative
from app.db.database import Database, utcnow
from app.logging_setup import get_logger

log = get_logger("journal")


def _entry_timing(mae_pct: float, mfe_pct: float, won: bool) -> str:
    """Judge entry timing from excursions.

    Deep MAE before resolution => entered too early (price went against first).
    Tiny MAE + straight to TP => timing was good. Big MFE that reversed to a
    loss => exit/TP management problem more than entry.
    """
    if won:
        if mae_pct <= -0.6 * abs(mfe_pct or 1e-9):
            return "too early — trade survived a deep drawdown before working"
        if abs(mae_pct) < 0.15 * max(abs(mfe_pct), 1e-9):
            return "well-timed — minimal adverse excursion"
        return "acceptable — moderate adverse excursion before target"
    if mfe_pct > 0.5:
        return "entry fine, exit management failed — trade was profitable and reversed"
    if abs(mae_pct) > 0 and mfe_pct <= 0.1:
        return "likely wrong thesis — price moved against almost immediately"
    return "inconclusive"


def _sl_tp_assessment(trade: dict, won: bool) -> str:
    mae, mfe = float(trade.get("mae_pct") or 0), float(trade.get("mfe_pct") or 0)
    entry, sl = float(trade["entry_price"]), float(trade["stop_loss"])
    sl_pct = abs(entry - sl) / entry * 100
    parts = []
    if not won and trade["exit_reason"] == "sl" and mfe > 0.6 * float(trade["rr"]) * sl_pct:
        parts.append("TP may be too ambitious: price covered most of the distance then reversed")
    if won and abs(mae) > 0.8 * sl_pct:
        parts.append("SL nearly hit before winning: stop may be too tight for this regime")
    if not won and abs(mae) < 1.05 * sl_pct and mfe < 0.2 * sl_pct:
        parts.append("clean stop-out with little favorable movement: setup filter should have vetoed")
    return "; ".join(parts) or "SL/TP placement reasonable for the volatility at entry"


def build_review(trade: dict, calibration_score: float) -> tuple[dict, str]:
    won = (trade.get("pnl_usd") or 0) > 0
    mae, mfe = float(trade.get("mae_pct") or 0), float(trade.get("mfe_pct") or 0)
    feats = json.loads(trade["features"]) if isinstance(trade["features"], str) else trade["features"]

    # which learned factors pointed the right / wrong way
    conf = float(trade["confidence"])
    conf_gap = conf - (1.0 if won else 0.0)
    review = {
        "trade_id": trade["id"],
        "outcome": "win" if won else "loss",
        "exit_reason": trade["exit_reason"],
        "r_multiple": trade["r_multiple"],
        "pnl_pct": trade["pnl_pct"],
        "duration_minutes": trade["duration_minutes"],
        "regime_at_entry": trade["regime_label"],
        "entry_timing": _entry_timing(mae, mfe, won),
        "sl_tp_assessment": _sl_tp_assessment(trade, won),
        "mae_pct": mae, "mfe_pct": mfe,
        "stated_confidence": conf,
        "confidence_error": round(conf_gap, 3),
        "calibration_score_after": round(calibration_score, 3),
        "notable_features_at_entry": {
            k: round(v, 5) for k, v in sorted(
                feats.items(), key=lambda kv: -abs(kv[1]))[:5]
        },
    }
    lesson_bits = [f"{trade['symbol']} {trade['direction']} in '{trade['regime_label']}'"]
    if won:
        lesson_bits.append(f"won {trade['r_multiple']:+.2f}R ({trade['exit_reason']})")
    else:
        lesson_bits.append(f"lost {trade['r_multiple']:+.2f}R ({trade['exit_reason']})")
    lesson_bits.append(review["entry_timing"])
    if abs(conf_gap) > 0.45:
        lesson_bits.append(
            f"confidence {conf:.2f} was badly calibrated for this setup")
    lessons = "; ".join(lesson_bits)
    return review, lessons


async def write_review(cfg, db: Database, trade: dict,
                       calibration_score: float) -> tuple[dict, str, str | None]:
    """Build + persist review. Returns (review, lessons, llm_prose|None)."""
    review, lessons = build_review(trade, calibration_score)
    prose = await llm_narrative(cfg, REVIEW_SYSTEM, {
        "trade": {k: trade[k] for k in (
            "symbol", "direction", "entry_price", "exit_price", "stop_loss",
            "take_profit", "exit_reason", "pnl_pct", "r_multiple",
            "duration_minutes", "mae_pct", "mfe_pct", "confidence",
            "regime_label")},
        "structured_review": review,
    })
    if prose:
        review["llm_review"] = prose
    await db.insert_review(trade["id"], review, lessons)
    log.info("review_written", trade_id=trade["id"], lessons=lessons)
    return review, lessons, prose
