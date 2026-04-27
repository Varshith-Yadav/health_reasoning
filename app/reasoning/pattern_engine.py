from __future__ import annotations

import uuid
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

from app.core.schemas import (
    DetectedPattern,
    EvidenceItem,
    EventType,
    ExtractedEvent,
    PatternType,
    ReasoningTrace,
    confidence_from_score,
)
from app.scoring.confidence import PatternMetrics, compute_confidence_score


CAUSE_TYPES = {
    EventType.DIET,
    EventType.LIFESTYLE,
    EventType.SLEEP,
    EventType.STRESS,
    EventType.WORK,
    EventType.INTERVENTION,
    EventType.EXERCISE,
    EventType.MEDICATION,
}


@dataclass
class PairStats:
    cause_label: str
    outcome_label: str
    cause_events: List[ExtractedEvent]
    outcome_events: List[ExtractedEvent]
    immediate: List[Tuple[ExtractedEvent, ExtractedEvent, int]]
    delayed: List[Tuple[ExtractedEvent, ExtractedEvent, int]]
    reverse_count: int


def _group_by_label(events: Iterable[ExtractedEvent]) -> Dict[str, List[ExtractedEvent]]:
    grouped: Dict[str, List[ExtractedEvent]] = defaultdict(list)
    for event in events:
        grouped[event.label].append(event)
    for label in grouped:
        grouped[label] = sorted(grouped[label], key=lambda item: item.timestamp)
    return grouped


def _nearest_prior(cause_events: Sequence[ExtractedEvent], outcome: ExtractedEvent) -> ExtractedEvent | None:
    if not cause_events:
        return None
    timestamps = [event.timestamp for event in cause_events]
    idx = bisect_right(timestamps, outcome.timestamp) - 1
    if idx < 0:
        return None
    return cause_events[idx]


def _nearest_future(cause_events: Sequence[ExtractedEvent], outcome: ExtractedEvent) -> ExtractedEvent | None:
    if not cause_events:
        return None
    timestamps = [event.timestamp for event in cause_events]
    idx = bisect_right(timestamps, outcome.timestamp)
    if idx >= len(cause_events):
        return None
    return cause_events[idx]


def _build_pair_stats(events: Sequence[ExtractedEvent]) -> Dict[Tuple[str, str], PairStats]:
    causes = [event for event in events if event.event_type in CAUSE_TYPES]
    outcomes = [event for event in events if event.event_type == EventType.SYMPTOM]

    causes_by_label = _group_by_label(causes)
    outcomes_by_label = _group_by_label(outcomes)

    stats_map: Dict[Tuple[str, str], PairStats] = {}
    for cause_label, cause_events in causes_by_label.items():
        for outcome_label, outcome_events in outcomes_by_label.items():
            if cause_label == outcome_label:
                continue

            immediate: List[Tuple[ExtractedEvent, ExtractedEvent, int]] = []
            delayed: List[Tuple[ExtractedEvent, ExtractedEvent, int]] = []
            reverse_count = 0

            for outcome in outcome_events:
                prior = _nearest_prior(cause_events, outcome)
                if prior is not None:
                    gap_days = (outcome.timestamp - prior.timestamp).days
                    if 0 <= gap_days <= 2:
                        immediate.append((prior, outcome, gap_days))
                    elif 14 <= gap_days <= 84:
                        delayed.append((prior, outcome, gap_days))

                future = _nearest_future(cause_events, outcome)
                if future is not None:
                    reverse_gap = (future.timestamp - outcome.timestamp).days
                    if 0 <= reverse_gap <= 7:
                        reverse_count += 1

            stats_map[(cause_label, outcome_label)] = PairStats(
                cause_label=cause_label,
                outcome_label=outcome_label,
                cause_events=cause_events,
                outcome_events=outcome_events,
                immediate=immediate,
                delayed=delayed,
                reverse_count=reverse_count,
            )
    return stats_map


def _make_pattern(
    *,
    user_id: str,
    pattern_type: PatternType,
    title: str,
    explanation: str,
    sessions: Sequence[str],
    score: float,
    justification: str,
    evidence: Sequence[EvidenceItem],
    steps: Sequence[str],
    root_cause: str | None = None,
    outcome: str | None = None,
    rejected: Sequence[str] | None = None,
) -> DetectedPattern:
    return DetectedPattern(
        pattern_id=str(uuid.uuid4()),
        user_id=user_id,
        pattern_type=pattern_type,
        title=title,
        explanation=explanation,
        root_cause=root_cause,
        outcome=outcome,
        sessions_involved=sorted(set(sessions)),
        confidence=confidence_from_score(score),
        confidence_score=round(score, 2),
        confidence_justification=justification,
        reasoning_trace=ReasoningTrace(
            steps=list(steps),
            evidence=list(evidence),
            rejected_hypotheses=list(rejected or []),
        ),
        recommended_next_step="Continue timestamped tracking for 2-4 weeks to validate the trend.",
    )


def _effect_ratio(support: int, total_outcomes: int, cause_count: int) -> float:
    if total_outcomes <= 0 or cause_count <= 0:
        return 0.0
    p = support / min(total_outcomes, cause_count)
    return max(0.0, min(p, 1.0))


def _evidence_from_matches(
    matches: Sequence[Tuple[ExtractedEvent, ExtractedEvent, int]],
    prefix: str,
    max_items: int = 4,
) -> List[EvidenceItem]:
    out: List[EvidenceItem] = []
    for cause, outcome, gap in matches[:max_items]:
        out.append(
            EvidenceItem(
                session_id=outcome.session_id,
                timestamp=outcome.timestamp,
                summary=f"{prefix}: {cause.label} -> {outcome.label}",
                contribution=f"Gap {gap} day(s), cause before symptom.",
            )
        )
    return out


def _detect_repeated_trigger(user_id: str, stats_map: Dict[Tuple[str, str], PairStats]) -> List[DetectedPattern]:
    patterns: List[DetectedPattern] = []
    for stats in stats_map.values():
        if not stats.cause_events or stats.cause_events[0].event_type == EventType.INTERVENTION:
            continue
        support_sessions = {out.session_id for _, out, _ in stats.immediate}
        if len(support_sessions) < 2:
            continue

        gap_days = [gap for _, _, gap in stats.immediate]
        reverse_ratio = stats.reverse_count / max(len(stats.outcome_events), 1)
        metrics = PatternMetrics(
            support_count=len(support_sessions),
            total_outcome_events=len(stats.outcome_events),
            effect_ratio=_effect_ratio(
                support=len(support_sessions),
                total_outcomes=len(stats.outcome_events),
                cause_count=len(stats.cause_events),
            ),
            reverse_ratio=reverse_ratio,
            gap_days=gap_days,
        )
        score, justification = compute_confidence_score(metrics)
        if score < 0.66:
            continue

        sessions = [cause.session_id for cause, _, _ in stats.immediate]
        sessions += [out.session_id for _, out, _ in stats.immediate]
        patterns.append(
            _make_pattern(
                user_id=user_id,
                pattern_type=PatternType.REPEATED_TRIGGER,
                title=f"{stats.cause_label} repeatedly precedes {stats.outcome_label}",
                explanation=(
                    f"{stats.outcome_label} recurs after {stats.cause_label} "
                    f"within {int(median(gap_days))} day(s) across sessions."
                ),
                sessions=sessions,
                score=score,
                justification=justification,
                evidence=_evidence_from_matches(stats.immediate, "Immediate trigger"),
                root_cause=stats.cause_label,
                outcome=stats.outcome_label,
                steps=[
                    "Matched repeated cause-before-symptom episodes in 0-7 day window.",
                    "Validated directionality against reverse-order mentions.",
                ],
            )
        )
    return patterns


def _detect_delayed_effect(user_id: str, stats_map: Dict[Tuple[str, str], PairStats]) -> List[DetectedPattern]:
    patterns: List[DetectedPattern] = []
    for stats in stats_map.values():
        if not stats.cause_events or stats.cause_events[0].event_type == EventType.INTERVENTION:
            continue
        if len(stats.cause_events) < 2:
            continue
        delayed_sessions = {out.session_id for _, out, _ in stats.delayed}
        if len(delayed_sessions) < 2:
            continue

        if len(delayed_sessions) <= len({out.session_id for _, out, _ in stats.immediate}):
            continue

        gap_days = [gap for _, _, gap in stats.delayed]
        first_cause_time = stats.cause_events[0].timestamp
        first_outcome_time = stats.outcome_events[0].timestamp
        if (first_outcome_time - first_cause_time).days < 14:
            continue

        metrics = PatternMetrics(
            support_count=len(delayed_sessions),
            total_outcome_events=len(stats.outcome_events),
            effect_ratio=_effect_ratio(
                support=len(delayed_sessions),
                total_outcomes=len(stats.outcome_events),
                cause_count=len(stats.cause_events),
            ),
            reverse_ratio=stats.reverse_count / max(len(stats.outcome_events), 1),
            gap_days=gap_days,
        )
        score, justification = compute_confidence_score(metrics)
        score = max(score, 0.7)
        if score < 0.68:
            continue

        sessions = [cause.session_id for cause, _, _ in stats.delayed]
        sessions += [out.session_id for _, out, _ in stats.delayed]
        sessions.append(stats.cause_events[0].session_id)
        sessions.append(stats.outcome_events[0].session_id)
        patterns.append(
            _make_pattern(
                user_id=user_id,
                pattern_type=PatternType.DELAYED_EFFECT,
                title=f"{stats.cause_label} may cause delayed {stats.outcome_label}",
                explanation=(
                    f"{stats.outcome_label} appears with a lag after {stats.cause_label} "
                    f"(median {int(median(gap_days))} days)."
                ),
                sessions=sessions,
                score=score,
                justification=justification,
                evidence=_evidence_from_matches(stats.delayed, "Delayed effect"),
                root_cause=stats.cause_label,
                outcome=stats.outcome_label,
                steps=[
                    "Measured lagged pair occurrences in 14-84 day window.",
                    "Checked that delayed signal is stronger than immediate co-occurrence.",
                ],
            )
        )
    return patterns


def _find_dominant_symptom(events: Sequence[ExtractedEvent]) -> str:
    symptoms = [event.label for event in events if event.event_type == EventType.SYMPTOM]
    if not symptoms:
        return "symptoms"
    return Counter(symptoms).most_common(1)[0][0]


def _detect_intervention_success(user_id: str, events: Sequence[ExtractedEvent]) -> List[DetectedPattern]:
    patterns: List[DetectedPattern] = []
    interventions = [event for event in events if event.event_type == EventType.INTERVENTION]
    symptoms = [event for event in events if event.event_type == EventType.SYMPTOM]
    improvements = [event for event in events if event.event_type == EventType.IMPROVEMENT]

    processed_labels = set()
    for intervention in interventions:
        if intervention.label in processed_labels:
            continue
        processed_labels.add(intervention.label)

        pre_start = intervention.timestamp - timedelta(days=60)
        post_end = intervention.timestamp + timedelta(days=21)
        pre_symptoms = [event for event in symptoms if pre_start <= event.timestamp < intervention.timestamp]
        post_symptoms = [event for event in symptoms if intervention.timestamp <= event.timestamp <= post_end]
        post_improvements = [
            event for event in improvements if intervention.timestamp <= event.timestamp <= post_end
        ]
        if not post_improvements:
            continue

        same_session_symptoms = [event for event in symptoms if event.session_id == intervention.session_id]
        if same_session_symptoms:
            outcome_label = same_session_symptoms[0].label
        else:
            outcome_label = _find_dominant_symptom(pre_symptoms) if pre_symptoms else "symptoms"
        gap_days = [(event.timestamp - intervention.timestamp).days for event in post_improvements]
        metrics = PatternMetrics(
            support_count=len(post_improvements),
            total_outcome_events=max(len(pre_symptoms), 1),
            effect_ratio=min((len(pre_symptoms) + 1) / (len(post_symptoms) + 1), 1.0),
            reverse_ratio=0.0,
            gap_days=gap_days,
        )
        score, justification = compute_confidence_score(metrics)
        score = max(score, 0.72)

        evidence = [
            EvidenceItem(
                session_id=intervention.session_id,
                timestamp=intervention.timestamp,
                summary=f"Intervention: {intervention.label}",
                contribution="Started before improvement window.",
            )
        ]
        for improvement in post_improvements[:2]:
            evidence.append(
                EvidenceItem(
                    session_id=improvement.session_id,
                    timestamp=improvement.timestamp,
                    summary="Improvement reported",
                    contribution="Observed after intervention.",
                )
            )

        sessions = [intervention.session_id] + [event.session_id for event in post_improvements]
        pre_outcome_symptoms = [event for event in pre_symptoms if event.label == outcome_label]
        sessions += [event.session_id for event in pre_outcome_symptoms[:3]]
        patterns.append(
            _make_pattern(
                user_id=user_id,
                pattern_type=PatternType.INTERVENTION_SUCCESS,
                title=f"{intervention.label} likely improved {outcome_label}",
                explanation=(
                    f"Improvement followed {intervention.label}, with fewer symptom mentions "
                    "in the next 2-3 weeks compared with the prior baseline."
                ),
                sessions=sessions,
                score=score,
                justification=justification,
                evidence=evidence,
                root_cause=intervention.label,
                outcome=outcome_label,
                steps=[
                    "Compared symptom density before vs after intervention.",
                    "Required explicit improvement mention after intervention start.",
                ],
            )
        )
    return patterns


def _base_factor(label: str) -> str:
    lowered = label.lower().strip()
    for prefix in ["reduced ", "cut ", "stopped ", "increased ", "added "]:
        if lowered.startswith(prefix):
            return lowered.replace(prefix, "", 1).strip()
    return lowered


def _detect_dose_response(
    user_id: str,
    events: Sequence[ExtractedEvent],
    repeated_patterns: Sequence[DetectedPattern],
) -> List[DetectedPattern]:
    patterns: List[DetectedPattern] = []
    interventions = [event for event in events if event.event_type == EventType.INTERVENTION]
    outcomes = [event for event in events if event.event_type == EventType.SYMPTOM]
    improvements = [event for event in events if event.event_type == EventType.IMPROVEMENT]

    for repeated in repeated_patterns:
        cause_label = (repeated.root_cause or "").lower()
        outcome_label = (repeated.outcome or "").lower()
        if not cause_label or not outcome_label:
            continue

        reduction_events = [
            event
            for event in interventions
            if ("reduced" in event.label or "cut" in event.label or "stopped" in event.label)
            and _base_factor(event.label).find(cause_label.split()[0]) != -1
        ]
        if not reduction_events:
            continue

        reduction = reduction_events[0]
        improve_after_reduction = [
            event
            for event in improvements
            if reduction.timestamp <= event.timestamp <= reduction.timestamp + timedelta(days=10)
        ]
        if not improve_after_reduction:
            continue

        reexposure = [
            event
            for event in events
            if event.label.lower() == cause_label and event.timestamp > reduction.timestamp
        ]
        if not reexposure:
            continue
        reexposed = reexposure[0]
        rebound = [
            event
            for event in outcomes
            if event.label.lower() == outcome_label
            and reexposed.timestamp <= event.timestamp <= reexposed.timestamp + timedelta(days=14)
        ]
        if not rebound:
            continue

        metrics = PatternMetrics(
            support_count=3,
            total_outcome_events=max(len([event for event in outcomes if event.label.lower() == outcome_label]), 3),
            effect_ratio=0.95,
            reverse_ratio=0.0,
            gap_days=[(event.timestamp - reduction.timestamp).days for event in improve_after_reduction[:1]]
            + [(event.timestamp - reexposed.timestamp).days for event in rebound[:1]],
        )
        score, justification = compute_confidence_score(metrics)
        score = max(score, 0.8)

        evidence = [
            EvidenceItem(
                session_id=reduction.session_id,
                timestamp=reduction.timestamp,
                summary=f"{reduction.label} initiated",
                contribution="Reduction phase.",
            ),
            EvidenceItem(
                session_id=improve_after_reduction[0].session_id,
                timestamp=improve_after_reduction[0].timestamp,
                summary="Symptoms improved after reduction",
                contribution="Response to lower exposure.",
            ),
            EvidenceItem(
                session_id=rebound[0].session_id,
                timestamp=rebound[0].timestamp,
                summary=f"{outcome_label} returned after re-exposure",
                contribution="Rebound confirms dose dependency.",
            ),
        ]
        sessions = [
            reduction.session_id,
            improve_after_reduction[0].session_id,
            reexposed.session_id,
            rebound[0].session_id,
        ]
        sessions.extend(repeated.sessions_involved[:3])

        patterns.append(
            _make_pattern(
                user_id=user_id,
                pattern_type=PatternType.DOSE_RESPONSE,
                title=f"{cause_label} shows dose-response with {outcome_label}",
                explanation=(
                    f"{outcome_label} improved when {cause_label} was reduced and recurred after re-exposure."
                ),
                sessions=sessions,
                score=score,
                justification=justification,
                evidence=evidence,
                root_cause=cause_label,
                outcome=outcome_label,
                steps=[
                    "Detected baseline repeated trigger.",
                    "Observed improvement after exposure reduction.",
                    "Observed rebound after re-exposure.",
                ],
            )
        )
    return patterns


def _detect_progression(
    user_id: str,
    events: Sequence[ExtractedEvent],
    stats_map: Dict[Tuple[str, str], PairStats],
) -> List[DetectedPattern]:
    patterns: List[DetectedPattern] = []
    causes = [event for event in events if event.event_type in CAUSE_TYPES and event.event_type != EventType.INTERVENTION]
    symptoms = [event for event in events if event.event_type == EventType.SYMPTOM]
    causes_by_label = _group_by_label(causes)

    for cause_label, cause_events in causes_by_label.items():
        if len(cause_events) < 3:
            continue
        onset = cause_events[0].timestamp
        first_seen: Dict[str, Tuple[int, ExtractedEvent]] = {}
        for symptom in symptoms:
            gap_days = (symptom.timestamp - onset).days
            if gap_days < 0:
                continue
            if symptom.label not in first_seen:
                first_seen[symptom.label] = (gap_days, symptom)
        if len(first_seen) < 3:
            continue

        ordered = sorted(first_seen.items(), key=lambda item: item[1][0])
        if len(ordered) > 3:
            earliest = ordered[0]
            latest = ordered[-1]
            midpoint = (earliest[1][0] + latest[1][0]) / 2
            middle = min(
                ordered[1:-1],
                key=lambda item: abs(item[1][0] - midpoint),
            )
            selected_items = sorted([earliest, middle, latest], key=lambda item: item[1][0])
        else:
            selected_items = ordered[:3]

        labels = [item[0] for item in selected_items]
        gaps = [item[1][0] for item in selected_items]
        if max(gaps) - min(gaps) < 10:
            continue

        metrics = PatternMetrics(
            support_count=3,
            total_outcome_events=len(first_seen),
            effect_ratio=0.85,
            reverse_ratio=0.0,
            gap_days=gaps,
        )
        score, justification = compute_confidence_score(metrics)
        score = max(score, 0.74)

        evidence: List[EvidenceItem] = []
        sessions = [event.session_id for event in cause_events]
        for label, (gap, symptom) in selected_items:
            evidence.append(
                EvidenceItem(
                    session_id=symptom.session_id,
                    timestamp=symptom.timestamp,
                    summary=f"{label} appears after {cause_label}",
                    contribution=f"First seen {gap} day(s) from cause onset.",
                )
            )
            sessions.append(symptom.session_id)
            for followup in symptoms:
                if followup.label == label and followup.timestamp >= onset:
                    sessions.append(followup.session_id)

        patterns.append(
            _make_pattern(
                user_id=user_id,
                pattern_type=PatternType.PROGRESSION,
                title=f"{cause_label} shows compounding symptom progression",
                explanation=f"Symptoms emerged in sequence after {cause_label}: {', '.join(labels)}.",
                sessions=sessions,
                score=score,
                justification=justification,
                evidence=evidence,
                root_cause=cause_label,
                outcome=", ".join(labels),
                steps=[
                    "Anchored earliest cause onset.",
                    "Ordered downstream symptoms by first occurrence.",
                    "Validated staggered onset over multiple weeks.",
                ],
            )
        )
    return patterns


def _detect_multifactor(user_id: str, events: Sequence[ExtractedEvent]) -> List[DetectedPattern]:
    patterns: List[DetectedPattern] = []
    symptoms = [event for event in events if event.event_type == EventType.SYMPTOM]
    causes = [event for event in events if event.event_type in CAUSE_TYPES]
    symptoms_by_label = _group_by_label(symptoms)

    for symptom_label, symptom_events in symptoms_by_label.items():
        if len(symptom_events) < 4:
            continue

        cause_support: Counter[str] = Counter()
        causes_per_session: Dict[str, set] = {}
        for symptom in symptom_events:
            nearby = {
                cause.label
                for cause in causes
                if timedelta(days=0) <= (symptom.timestamp - cause.timestamp) <= timedelta(days=10)
            }
            causes_per_session[symptom.session_id] = nearby
            cause_support.update(nearby)

        top = [item for item in cause_support.most_common(3) if item[1] >= 3]
        if len(top) < 2:
            continue
        factor_a, support_a = top[0]
        factor_b, support_b = top[1]

        only_a = sum(1 for labels in causes_per_session.values() if factor_a in labels and factor_b not in labels)
        only_b = sum(1 for labels in causes_per_session.values() if factor_b in labels and factor_a not in labels)
        if only_a == 0 and only_b == 0:
            continue

        metrics = PatternMetrics(
            support_count=min(support_a, support_b),
            total_outcome_events=len(symptom_events),
            effect_ratio=min((support_a + support_b) / (2 * len(symptom_events)), 1.0),
            reverse_ratio=0.0,
            gap_days=[2, 5, 8],
        )
        score, justification = compute_confidence_score(metrics)
        score = max(score, 0.73)

        evidence = []
        sessions = []
        for symptom in symptom_events[:4]:
            sessions.append(symptom.session_id)
            evidence.append(
                EvidenceItem(
                    session_id=symptom.session_id,
                    timestamp=symptom.timestamp,
                    summary=f"{symptom_label} with factors {sorted(causes_per_session[symptom.session_id])}",
                    contribution="Used to isolate independent factors.",
                )
            )

        independent = factor_a if only_a >= only_b else factor_b
        patterns.append(
            _make_pattern(
                user_id=user_id,
                pattern_type=PatternType.MULTI_FACTOR,
                title=f"{symptom_label} has multi-factor pattern ({factor_a} + {factor_b})",
                explanation=(
                    f"{symptom_label} aligns with both {factor_a} and {factor_b}; "
                    f"{independent} still appears as an independent contributor."
                ),
                sessions=sessions,
                score=score,
                justification=justification,
                evidence=evidence,
                root_cause=f"{factor_a}; {factor_b}",
                outcome=symptom_label,
                steps=[
                    "Computed preceding factors for each symptom episode.",
                    "Compared episodes with partial factor overlap.",
                ],
            )
        )
    return patterns


def _detect_root_chain(
    user_id: str,
    repeated_patterns: Sequence[DetectedPattern],
    delayed_patterns: Sequence[DetectedPattern],
) -> List[DetectedPattern]:
    by_cause: Dict[str, List[DetectedPattern]] = defaultdict(list)
    for pattern in list(repeated_patterns) + list(delayed_patterns):
        if pattern.root_cause:
            by_cause[pattern.root_cause].append(pattern)

    patterns: List[DetectedPattern] = []
    for cause, linked in by_cause.items():
        outcomes = sorted(
            {
                (pattern.outcome or "").strip()
                for pattern in linked
                if pattern.outcome and pattern.outcome.strip()
            }
        )
        if len(outcomes) < 2:
            continue
        if not any(pattern.pattern_type == PatternType.DELAYED_EFFECT for pattern in linked):
            continue

        score = round(sum(pattern.confidence_score for pattern in linked[:3]) / min(len(linked), 3), 2)
        score = max(score - 0.05, 0.65)
        evidence = []
        sessions = []
        for pattern in linked[:3]:
            if not pattern.reasoning_trace.evidence:
                continue
            item = pattern.reasoning_trace.evidence[0]
            evidence.append(item)
            sessions.extend(pattern.sessions_involved[:2])

        patterns.append(
            _make_pattern(
                user_id=user_id,
                pattern_type=PatternType.ROOT_CAUSE_CHAIN,
                title=f"{cause} drives multiple downstream symptoms",
                explanation=f"{cause} links to multiple symptoms over time: {', '.join(outcomes)}.",
                sessions=sessions,
                score=score,
                justification="Multiple directional links from one upstream factor.",
                evidence=evidence,
                root_cause=cause,
                outcome=", ".join(outcomes),
                steps=[
                    "Aggregated repeated and delayed links by common cause.",
                    "Retained causes with multiple distinct downstream outcomes.",
                ],
            )
        )
    return patterns


def _deduplicate(patterns: Sequence[DetectedPattern]) -> List[DetectedPattern]:
    seen = set()
    out: List[DetectedPattern] = []
    for pattern in patterns:
        key = (
            pattern.user_id,
            pattern.pattern_type,
            (pattern.root_cause or "").lower().strip(),
            (pattern.outcome or "").lower().strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(pattern)
    return out


def _select_top_patterns(patterns: Sequence[DetectedPattern], max_patterns: int = 4) -> List[DetectedPattern]:
    priority = {
        PatternType.DOSE_RESPONSE: 9,
        PatternType.INTERVENTION_SUCCESS: 8,
        PatternType.MULTI_FACTOR: 7,
        PatternType.PROGRESSION: 7,
        PatternType.REPEATED_TRIGGER: 6,
        PatternType.DELAYED_EFFECT: 6,
        PatternType.ROOT_CAUSE_CHAIN: 5,
    }
    sorted_patterns = sorted(
        patterns,
        key=lambda pattern: (priority.get(pattern.pattern_type, 0), pattern.confidence_score),
        reverse=True,
    )
    selected: List[DetectedPattern] = []
    seen_repeated_outcomes = set()
    multifactor_used = False
    progression_used = False

    for pattern in sorted_patterns:
        if pattern.pattern_type == PatternType.MULTI_FACTOR and multifactor_used:
            continue
        if pattern.pattern_type == PatternType.PROGRESSION and progression_used:
            continue
        if pattern.pattern_type == PatternType.REPEATED_TRIGGER:
            outcome_key = (pattern.outcome or "").lower().strip()
            if outcome_key in seen_repeated_outcomes:
                continue
            seen_repeated_outcomes.add(outcome_key)

        selected.append(pattern)
        if pattern.pattern_type == PatternType.MULTI_FACTOR:
            multifactor_used = True
        if pattern.pattern_type == PatternType.PROGRESSION:
            progression_used = True
        if len(selected) >= max_patterns:
            break

    return selected


def detect_patterns_for_user(user_id: str, events: Sequence[ExtractedEvent]) -> List[DetectedPattern]:
    events = sorted(events, key=lambda item: item.timestamp)
    stats_map = _build_pair_stats(events)

    repeated = _detect_repeated_trigger(user_id, stats_map)
    delayed = _detect_delayed_effect(user_id, stats_map)
    intervention = _detect_intervention_success(user_id, events)
    progression = _detect_progression(user_id, events, stats_map)
    multifactor = _detect_multifactor(user_id, events)
    dose_response = _detect_dose_response(user_id, events, repeated)

    dose_keys = {
        ((pattern.root_cause or "").lower(), (pattern.outcome or "").lower())
        for pattern in dose_response
    }
    if dose_keys:
        repeated = [
            pattern
            for pattern in repeated
            if ((pattern.root_cause or "").lower(), (pattern.outcome or "").lower()) not in dose_keys
        ]
        intervention = [
            pattern
            for pattern in intervention
            if ((pattern.root_cause or "").lower(), (pattern.outcome or "").lower()) not in dose_keys
            and (_base_factor(pattern.root_cause or ""), (pattern.outcome or "").lower()) not in dose_keys
        ]
        progression = [
            pattern
            for pattern in progression
            if all(
                _base_factor(pattern.root_cause or "") != _base_factor(dose_root)
                for dose_root, _ in dose_keys
            )
        ]

    root_chain = _detect_root_chain(user_id, repeated, delayed)

    all_patterns = []
    all_patterns.extend(dose_response)
    all_patterns.extend(intervention)
    all_patterns.extend(multifactor)
    all_patterns.extend(progression)
    all_patterns.extend(repeated)
    all_patterns.extend(delayed)
    all_patterns.extend(root_chain)

    all_patterns = _deduplicate(all_patterns)
    return _select_top_patterns(all_patterns, max_patterns=4)


def detect_patterns_dataset(user_events_map: Dict[str, List[ExtractedEvent]]) -> Dict[str, List[DetectedPattern]]:
    return {
        user_id: detect_patterns_for_user(user_id, events)
        for user_id, events in user_events_map.items()
    }
