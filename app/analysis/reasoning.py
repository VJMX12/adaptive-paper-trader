"""Human-readable reasoning for analyses and post-trade reviews.

Two modes:
- Template mode (default): deterministic narrative built from the numbers.
- LLM mode (llm.enabled: true + ANTHROPIC_API_KEY): Claude writes the
  narrative from the same structured facts. Numbers and decisions always
  come from the quantitative engine — the LLM only explains, never decides.
"""
from __future__ import annotations

import json

import aiohttp

from app.analysis.engine import AnalysisResult
from app.logging_setup import get_logger

log = get_logger("reasoning")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def template_analysis_reasoning(res: AnalysisResult, retrieval: dict) -> str:
    lines = [
        f"Regime: {res.regime_label}; posterior {res.regime_posterior}.",
        f"Changepoint probability {res.changepoint_prob:.2f} "
        f"({'ALARM' if res.changepoint_prob >= 0.35 else 'stable'}).",
        f"Learner P(direction favorable) raw={res.raw_prob:.2f}, "
        f"calibration-adjusted confidence={res.confidence:.2f}.",
    ]
    if res.key_factors:
        top = ", ".join(f"{n} ({v:+.2f})" for n, v in res.key_factors)
        lines.append(f"Top learned factors: {top}.")
    n = len(retrieval.get("neighbors") or [])
    if n:
        wr = retrieval.get("win_rate")
        avg_r = retrieval.get("avg_r")
        lines.append(
            f"Memory: {n} similar past setups, win rate {wr:.0%}"
            + (f", avg R {avg_r:+.2f}" if avg_r is not None else "") + "."
        )
    if retrieval.get("lessons"):
        lines.append("Recent lessons applied: " + " | ".join(retrieval["lessons"][:2]))
    if res.trade_recommended:
        lines.append(
            f"Decision: {res.direction} — edge is statistically justified. {res.invalidation}"
        )
    else:
        lines.append("Decision: NO TRADE — " + "; ".join(res.veto_reasons) + ".")
    return " ".join(lines)


async def llm_narrative(cfg, system: str, user_payload: dict) -> str | None:
    """Optional Claude call. Returns None on any failure (caller falls back)."""
    key = cfg.secrets.anthropic_api_key
    if not (cfg.get("llm.enabled") and key):
        return None
    body = {
        "model": cfg.get("llm.model", "claude-sonnet-4-6"),
        "max_tokens": int(cfg.get("llm.max_tokens", 800)),
        "system": system,
        "messages": [{"role": "user", "content": json.dumps(user_payload, default=str)}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(ANTHROPIC_URL, json=body, headers=headers,
                              timeout=aiohttp.ClientTimeout(total=45)) as r:
                if r.status != 200:
                    log.warning("llm_http_error", status=r.status)
                    return None
                data = await r.json()
                parts = [b.get("text", "") for b in data.get("content", [])
                         if b.get("type") == "text"]
                text = "\n".join(p for p in parts if p).strip()
                return text or None
    except Exception as e:
        log.warning("llm_error", error=str(e))
        return None


ANALYSIS_SYSTEM = (
    "You are the explanation layer of a paper-trading research system. "
    "You receive structured facts (regime posterior, changepoint probability, "
    "learner confidence, retrieved similar historical trades, veto reasons). "
    "Write a concise trading-desk style rationale (<=120 words). Never invent "
    "numbers; never change the decision; state the invalidation conditions."
)

REVIEW_SYSTEM = (
    "You are the post-trade review layer of a paper-trading research system. "
    "You receive structured facts about one closed simulated trade. Write a "
    "concise honest review (<=150 words): why it won/lost, whether entry was "
    "early/late given MAE/MFE, whether SL/TP placement was appropriate, and "
    "one concrete lesson. Never invent numbers."
)


async def analysis_reasoning(cfg, res: AnalysisResult, retrieval: dict) -> str:
    fallback = template_analysis_reasoning(res, retrieval)
    payload = {
        "symbol": res.symbol, "price": res.price, "bias": res.bias,
        "direction": res.direction, "confidence": res.confidence,
        "raw_prob": res.raw_prob, "regime": res.regime_label,
        "regime_posterior": res.regime_posterior,
        "changepoint_prob": res.changepoint_prob,
        "veto_reasons": res.veto_reasons, "key_factors": res.key_factors,
        "rr": res.rr, "invalidation": res.invalidation,
        "similar_history": {
            "n": len(retrieval.get("neighbors") or []),
            "win_rate": retrieval.get("win_rate"),
            "avg_r": retrieval.get("avg_r"),
            "lessons": retrieval.get("lessons", [])[:3],
        },
    }
    text = await llm_narrative(cfg, ANALYSIS_SYSTEM, payload)
    return text or fallback
