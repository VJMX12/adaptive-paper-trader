"""Per-symbol learning progress, shown as 3 phase-dots (e.g. BTC ●●○).

Phase is driven by how many shadow-labeled outcomes (resolve_barrier results,
fills or not) have accumulated for that symbol — the same signal that trains
the model, just counted per-symbol instead of globally. Thresholds are
deliberately round numbers, not a statistical claim: they mark "the model has
seen enough of this symbol's behavior to say something," not "this symbol has
edge."
"""
from __future__ import annotations

PHASES = [
    (1, 1, "\U0001F440", "First look"),
    (2, 25, "\U0001F3AF", "Finding patterns"),
    (3, 100, "\U0001F9E0", "Fluent"),
]
TOTAL_PHASES = len(PHASES)


def phase_for_count(resolved_count: int) -> int:
    phase = 0
    for p, threshold, _emoji, _label in PHASES:
        if resolved_count >= threshold:
            phase = p
    return phase


def phase_info(phase: int) -> tuple[str, str]:
    """-> (emoji, label) for a phase, or ("", "Not started") for 0."""
    for p, _threshold, emoji, label in PHASES:
        if p == phase:
            return emoji, label
    return "", "Not started"


def dots(phase: int) -> str:
    return "●" * phase + "○" * (TOTAL_PHASES - phase)
