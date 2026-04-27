from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import List, Tuple


@dataclass(frozen=True)
class PatternMetrics:
    support_count: int
    total_outcome_events: int
    effect_ratio: float
    reverse_ratio: float
    gap_days: List[int]
    contradiction_count: int = 0


def _bounded(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _consistency_from_gaps(gaps: List[int]) -> float:
    if not gaps:
        return 0.35
    if len(gaps) == 1:
        return 0.70
    avg = mean(gaps)
    spread = pstdev(gaps)
    if avg <= 0:
        return 0.45
    # Low spread relative to mean implies temporal consistency.
    return _bounded(1.0 - (spread / (avg + 1.0)))


def compute_confidence_score(metrics: PatternMetrics) -> Tuple[float, str]:
    coverage = _bounded(metrics.support_count / max(metrics.total_outcome_events, 1))
    volume = _bounded(metrics.support_count / 3.0)
    strength = _bounded(metrics.effect_ratio)
    direction = _bounded(1.0 - metrics.reverse_ratio)
    consistency = _consistency_from_gaps(metrics.gap_days)
    contradiction_penalty = min(metrics.contradiction_count * 0.08, 0.32)

    score = (
        0.24 * coverage
        + 0.22 * strength
        + 0.18 * direction
        + 0.14 * consistency
        + 0.22 * volume
        - contradiction_penalty
    )
    score = _bounded(score)

    gap_note = "variable gaps"
    if metrics.gap_days:
        gap_note = f"median lag {sorted(metrics.gap_days)[len(metrics.gap_days)//2]}d"

    justification = (
        f"support={metrics.support_count}/{metrics.total_outcome_events}, "
        f"effect={metrics.effect_ratio:.2f}, direction={direction:.2f}, {gap_note}."
    )
    return round(score, 2), justification
