from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Generator, Iterable, List

from app.core.schemas import DetectedPattern


def _dumps(payload: Dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def stream_start(dataset_path: str) -> str:
    return _dumps(
        {
            "type": "start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": dataset_path,
        }
    )


def stream_progress(stage: str, detail: str) -> str:
    return _dumps({"type": "progress", "stage": stage, "detail": detail})


def stream_pattern(user_id: str, pattern: DetectedPattern) -> str:
    return _dumps(
        {
            "type": "pattern",
            "user_id": user_id,
            "payload": pattern.model_dump(mode="json"),
        }
    )


def stream_patterns_map(
    patterns_map: Dict[str, List[DetectedPattern]],
) -> Generator[str, None, None]:
    for user_id, patterns in patterns_map.items():
        yield stream_progress("pattern_emission", f"{user_id}: {len(patterns)} patterns")
        for pattern in patterns:
            yield stream_pattern(user_id, pattern)


def stream_end(total_users: int, total_patterns: int) -> str:
    return _dumps(
        {
            "type": "end",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_users": total_users,
            "total_patterns": total_patterns,
            "status": "complete",
        }
    )
