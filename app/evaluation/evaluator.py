from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.core.schemas import DetectedPattern


@dataclass(frozen=True)
class GoldPattern:
    pattern_id: str
    user_id: str
    title: str
    temporal_logic: str
    sessions_involved: List[str]


@dataclass(frozen=True)
class PredictionPattern:
    pattern_id: str
    user_id: str
    title: str
    explanation: str
    sessions_involved: List[str]


def _normalize_session(session: str) -> str:
    match = re.search(r"(S\d+)$", session.upper())
    if match:
        return match.group(1)
    return session.upper()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _pattern_similarity(gold: GoldPattern, prediction: PredictionPattern) -> float:
    if gold.user_id != prediction.user_id:
        return 0.0

    gold_sessions = [_normalize_session(session) for session in gold.sessions_involved]
    pred_sessions = [_normalize_session(session) for session in prediction.sessions_involved]
    session_score = _jaccard(gold_sessions, pred_sessions)

    title_score = _text_similarity(gold.title, prediction.title)
    logic_score = _text_similarity(gold.temporal_logic, prediction.explanation)
    keyword_score = _jaccard(_tokenize(gold.title + " " + gold.temporal_logic), _tokenize(prediction.title))

    return 0.45 * session_score + 0.30 * title_score + 0.15 * logic_score + 0.10 * keyword_score


def load_gold_patterns(dataset_path: str | Path) -> List[GoldPattern]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    hidden = raw.get("hidden_patterns_reference", {})
    entries = hidden.get("patterns", [])

    gold: List[GoldPattern] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        gold.append(
            GoldPattern(
                pattern_id=str(item.get("pattern_id", "")),
                user_id=str(item.get("user", "")),
                title=str(item.get("title", "")),
                temporal_logic=str(item.get("temporal_logic", "")),
                sessions_involved=[str(value) for value in item.get("sessions_involved", [])],
            )
        )
    return gold


def predictions_from_map(pattern_map: Dict[str, List[DetectedPattern]]) -> List[PredictionPattern]:
    predictions: List[PredictionPattern] = []
    for user_id, patterns in pattern_map.items():
        for pattern in patterns:
            predictions.append(
                PredictionPattern(
                    pattern_id=pattern.pattern_id,
                    user_id=user_id,
                    title=pattern.title,
                    explanation=pattern.explanation,
                    sessions_involved=pattern.sessions_involved,
                )
            )
    return predictions


def evaluate_predictions(
    gold_patterns: Sequence[GoldPattern],
    predicted_patterns: Sequence[PredictionPattern],
    match_threshold: float = 0.46,
) -> Dict[str, object]:
    available = list(predicted_patterns)
    matches: List[Tuple[GoldPattern, PredictionPattern, float]] = []
    missed: List[GoldPattern] = []

    for gold in gold_patterns:
        best_idx: Optional[int] = None
        best_score = 0.0
        for idx, prediction in enumerate(available):
            score = _pattern_similarity(gold, prediction)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None or best_score < match_threshold:
            missed.append(gold)
            continue
        prediction = available.pop(best_idx)
        matches.append((gold, prediction, round(best_score, 3)))

    precision = len(matches) / max(len(predicted_patterns), 1)
    recall = len(matches) / max(len(gold_patterns), 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "summary": {
            "gold_patterns": len(gold_patterns),
            "predicted_patterns": len(predicted_patterns),
            "matched_patterns": len(matches),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "match_threshold": match_threshold,
        },
        "matches": [
            {
                "gold_pattern_id": gold.pattern_id,
                "gold_title": gold.title,
                "prediction_id": prediction.pattern_id,
                "prediction_title": prediction.title,
                "similarity_score": score,
            }
            for gold, prediction, score in matches
        ],
        "missed_gold_patterns": [
            {
                "gold_pattern_id": gold.pattern_id,
                "gold_title": gold.title,
                "user_id": gold.user_id,
            }
            for gold in missed
        ],
        "unmatched_predictions": [
            {
                "prediction_id": prediction.pattern_id,
                "prediction_title": prediction.title,
                "user_id": prediction.user_id,
            }
            for prediction in available
        ],
    }


def save_evaluation(report: Dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return path
