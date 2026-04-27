from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Generator, Any

from app.core.schemas import (
    DetectedPattern,
    UserAnalysisResult,
    FullAnalysisResponse,
)




def _serialize(obj: Any):
    """
    Handles datetime / enums / pydantic objects.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()

    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")

    if hasattr(obj, "value"):
        return obj.value

    return str(obj)




def build_user_result(
    user_id: str,
    user_name: str,
    patterns: List[DetectedPattern],
) -> UserAnalysisResult:

    return UserAnalysisResult(
        user_id=user_id,
        user_name=user_name,
        total_patterns_found=len(patterns),
        patterns=patterns,
    )




def build_full_response(
    user_results: List[UserAnalysisResult],
    version: str = "1.0.0",
) -> FullAnalysisResponse:

    return FullAnalysisResponse(
        generated_at=datetime.now(timezone.utc),
        system_version=version,
        results=user_results,
    )



def to_json(
    response: FullAnalysisResponse,
    indent: int = 2,
) -> str:

    return json.dumps(
        response.model_dump(mode="json"),
        indent=indent,
        ensure_ascii=False,
        default=_serialize,
    )




def save_json(
    response: FullAnalysisResponse,
    output_path: str | Path,
) -> Path:

    path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(to_json(response))

    return path




def stream_patterns(
    user_id: str,
    patterns: List[DetectedPattern],
) -> Generator[str, None, None]:
    """
    Yields newline-delimited JSON chunks.
    Great for Streamlit live rendering / APIs.
    """

    for pattern in patterns:
        payload = {
            "type": "pattern",
            "user_id": user_id,
            "data": pattern.model_dump(mode="json"),
        }

        yield json.dumps(
            payload,
            ensure_ascii=False,
            default=_serialize
        )


def stream_full_results(
    results: Dict[str, List[DetectedPattern]],
) -> Generator[str, None, None]:
    """
    Streams all users sequentially.
    """

    yield json.dumps(
        {
            "type": "start",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    for user_id, patterns in results.items():
        for chunk in stream_patterns(user_id, patterns):
            yield chunk

    yield json.dumps(
        {
            "type": "end",
            "status": "complete",
        }
    )



def api_response(
    user_id: str,
    user_name: str,
    patterns: List[DetectedPattern],
) -> Dict:

    result = build_user_result(
        user_id=user_id,
        user_name=user_name,
        patterns=patterns,
    )

    return result.model_dump(mode="json")




def response_summary(
    response: FullAnalysisResponse,
) -> Dict:

    total_users = len(response.results)
    total_patterns = sum(
        x.total_patterns_found for x in response.results
    )

    return {
        "generated_at": response.generated_at.isoformat(),
        "users_processed": total_users,
        "patterns_found": total_patterns,
        "system_version": response.system_version,
    }




def print_json(
    response: FullAnalysisResponse
):
    print(to_json(response))



if __name__ == "__main__":
    print("Formatter ready.")
