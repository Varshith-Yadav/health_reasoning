from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from app.core.schemas import Conversation, EventType, ExtractedEvent, SeverityLevel, UserProfile
from app.core.timeline import compute_week_index, get_first_session_date
from app.extraction.symptom_mapper import RuleCandidate, extract_rule_candidates
from app.llm.client import LLMClient, NoOpLLMClient


@dataclass(frozen=True)
class ExtractionConfig:
    mode: str = "hybrid"  # rule | llm | hybrid
    include_clary_response: bool = False


def _infer_severity(text: str) -> Optional[SeverityLevel]:
    normalized = text.lower()
    if any(word in normalized for word in ["severe", "really bad", "very bad", "everywhere"]):
        return SeverityLevel.SEVERE
    if any(word in normalized for word in ["worse", "bad", "hurting", "pain"]):
        return SeverityLevel.MODERATE
    if any(word in normalized for word in ["mild", "low level"]):
        return SeverityLevel.MILD
    if any(word in normalized for word in ["better", "clearer", "resolved", "stopped", "gone"]):
        return SeverityLevel.NONE
    return None


def _extract_numeric_metadata(text: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    normalized = text.lower()

    calories = re.search(r"\b(\d{3,4})\s*calories\b", normalized)
    if calories:
        metadata["calories_per_day"] = calories.group(1)

    water = re.search(r"\b(\d{1,2})\s+glasses?\s+of\s+water\b", normalized)
    if water:
        metadata["water_glasses"] = water.group(1)

    coffee = re.search(r"\b(\d{1,2})\s+or\s+(\d{1,2})\s+cups?\b", normalized)
    if coffee:
        metadata["coffee_cups_range"] = f"{coffee.group(1)}-{coffee.group(2)}"

    bedtime = re.search(r"\b(1|2|11|12)(:\d\d)?\s?(am|pm)\b", normalized)
    if bedtime:
        metadata["time_mention"] = bedtime.group(0).replace(" ", "")

    cycle_day = re.search(r"\bday\s+(\d{1,2})\b", normalized)
    if cycle_day:
        metadata["cycle_day"] = cycle_day.group(1)

    return metadata


def _conversation_text(convo: Conversation, include_clary_response: bool) -> str:
    fields = [convo.user_message or "", convo.user_followup or ""]
    if include_clary_response:
        fields.append(convo.clary_response or "")
    return " ".join(part.strip() for part in fields if part).strip()


def _make_event(
    user_id: str,
    session_id: str,
    timestamp,
    week_index: int,
    label: str,
    event_type: EventType,
    raw_text: str,
    source: str,
    metadata: Optional[Dict[str, str]] = None,
) -> ExtractedEvent:
    metadata = metadata or {}
    metadata.update(_extract_numeric_metadata(raw_text))
    return ExtractedEvent(
        event_id=str(uuid.uuid4()),
        user_id=user_id,
        session_id=session_id,
        timestamp=timestamp,
        week_index=week_index,
        event_type=event_type,
        label=label,
        raw_text=raw_text,
        severity=_infer_severity(raw_text),
        source=source,
        metadata=metadata,
    )


def _rule_extract(
    user_id: str,
    convo: Conversation,
    week_index: int,
    text: str,
) -> List[ExtractedEvent]:
    candidates: List[RuleCandidate] = extract_rule_candidates(text)
    events = [
        _make_event(
            user_id=user_id,
            session_id=convo.session_id,
            timestamp=convo.timestamp,
            week_index=week_index,
            label=candidate.label,
            event_type=candidate.event_type,
            raw_text=text,
            source="rule_based",
            metadata=candidate.metadata,
        )
        for candidate in candidates
    ]
    return events


def _llm_extract(
    llm: LLMClient,
    user_id: str,
    convo: Conversation,
    week_index: int,
    text: str,
) -> List[ExtractedEvent]:
    response = llm.extract_events(
        user_id=user_id,
        session_id=convo.session_id,
        timestamp=convo.timestamp.isoformat(),
        week_index=week_index,
        text=text,
    )
    events: List[ExtractedEvent] = []
    for item in response:
        label = str(item.get("label", "")).strip().lower()
        event_type_raw = str(item.get("event_type", "unknown")).strip().lower()
        if not label:
            continue
        try:
            event_type = EventType(event_type_raw)
        except ValueError:
            event_type = EventType.UNKNOWN
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        events.append(
            _make_event(
                user_id=user_id,
                session_id=convo.session_id,
                timestamp=convo.timestamp,
                week_index=week_index,
                label=label,
                event_type=event_type,
                raw_text=text,
                source="llm_extraction",
                metadata={k: str(v) for k, v in metadata.items()},
            )
        )
    return events


def _dedup_events(events: Sequence[ExtractedEvent]) -> List[ExtractedEvent]:
    seen: Dict[tuple, ExtractedEvent] = {}
    for event in events:
        key = (event.session_id, event.label, event.event_type)
        seen[key] = event
    return sorted(seen.values(), key=lambda value: value.timestamp)


def extract_events_from_conversation(
    user: UserProfile,
    convo: Conversation,
    config: ExtractionConfig,
    llm: Optional[LLMClient] = None,
) -> List[ExtractedEvent]:
    llm = llm or NoOpLLMClient()
    baseline = get_first_session_date(user)
    week_index = compute_week_index(convo.timestamp, baseline)
    text = _conversation_text(convo, include_clary_response=config.include_clary_response)

    events: List[ExtractedEvent] = []
    if config.mode in {"rule", "hybrid"}:
        events.extend(_rule_extract(user.user_id, convo, week_index, text))
    if config.mode in {"llm", "hybrid"} and llm.is_available():
        events.extend(_llm_extract(llm, user.user_id, convo, week_index, text))

    return _dedup_events(events)


def extract_events_for_user(
    user: UserProfile,
    config: Optional[ExtractionConfig] = None,
    llm: Optional[LLMClient] = None,
) -> List[ExtractedEvent]:
    config = config or ExtractionConfig()
    all_events: List[ExtractedEvent] = []
    for conversation in user.conversations:
        all_events.extend(extract_events_from_conversation(user, conversation, config, llm))
    return _dedup_events(all_events)


def extract_events_for_dataset(
    dataset,
    config: Optional[ExtractionConfig] = None,
    llm: Optional[LLMClient] = None,
) -> Dict[str, List[ExtractedEvent]]:
    config = config or ExtractionConfig()
    return {
        user.user_id: extract_events_for_user(user, config=config, llm=llm)
        for user in dataset.users
    }
