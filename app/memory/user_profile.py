from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

from app.core.schemas import ExtractedEvent, UserProfile


@dataclass(frozen=True)
class UserReasoningProfile:
    user_id: str
    name: str
    baseline_notes: str
    top_symptoms: List[str]
    top_lifestyle_factors: List[str]
    total_events: int


def build_user_reasoning_profile(
    user: UserProfile,
    events: Sequence[ExtractedEvent],
) -> UserReasoningProfile:
    symptom_counter = Counter(
        event.label for event in events if event.event_type.value == "symptom"
    )
    factor_counter = Counter(
        event.label
        for event in events
        if event.event_type.value in {"diet", "lifestyle", "sleep", "stress", "intervention"}
    )
    return UserReasoningProfile(
        user_id=user.user_id,
        name=user.name,
        baseline_notes=user.onboarding_notes or "",
        top_symptoms=[item[0] for item in symptom_counter.most_common(5)],
        top_lifestyle_factors=[item[0] for item in factor_counter.most_common(5)],
        total_events=len(events),
    )


def profile_as_dict(profile: UserReasoningProfile) -> Dict[str, object]:
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "baseline_notes": profile.baseline_notes,
        "top_symptoms": profile.top_symptoms,
        "top_lifestyle_factors": profile.top_lifestyle_factors,
        "total_events": profile.total_events,
    }
