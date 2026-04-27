from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from app.core.schemas import ExtractedEvent


@dataclass(frozen=True)
class TimelineChunk:
    chunk_id: str
    start_week: int
    end_week: int
    event_count: int
    events: List[ExtractedEvent]


def build_timeline(events: Sequence[ExtractedEvent]) -> List[ExtractedEvent]:
    return sorted(events, key=lambda event: event.timestamp)


def chunk_timeline(
    events: Sequence[ExtractedEvent],
    max_events_per_chunk: int = 8,
    overlap_events: int = 2,
) -> List[TimelineChunk]:
    ordered = build_timeline(events)
    if not ordered:
        return []

    chunks: List[TimelineChunk] = []
    index = 0
    chunk_number = 1
    step = max(max_events_per_chunk - overlap_events, 1)

    while index < len(ordered):
        slice_events = ordered[index : index + max_events_per_chunk]
        chunk = TimelineChunk(
            chunk_id=f"chunk_{chunk_number:02d}",
            start_week=min(event.week_index for event in slice_events),
            end_week=max(event.week_index for event in slice_events),
            event_count=len(slice_events),
            events=slice_events,
        )
        chunks.append(chunk)
        index += step
        chunk_number += 1

    return chunks


def build_chunk_summary(chunk: TimelineChunk) -> Dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "week_span": [chunk.start_week, chunk.end_week],
        "event_count": chunk.event_count,
        "labels": [event.label for event in chunk.events],
        "sessions": [event.session_id for event in chunk.events],
    }
