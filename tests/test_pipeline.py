from __future__ import annotations

import json

from app.core.loader import load_dataset
from app.evaluation.evaluator import evaluate_predictions, load_gold_patterns, predictions_from_map
from app.extraction.event_extractor import ExtractionConfig, extract_events_for_dataset
from app.llm.client import NoOpLLMClient
from app.output.json_streamer import stream_end, stream_progress, stream_start
from app.reasoning.pattern_engine import detect_patterns_dataset


DATASET_PATH = "data/askfirst_synthetic_dataset.json"


def test_end_to_end_pipeline_detects_patterns():
    dataset = load_dataset(DATASET_PATH)
    events = extract_events_for_dataset(
        dataset,
        config=ExtractionConfig(mode="rule", include_clary_response=False),
        llm=NoOpLLMClient(),
    )
    pattern_map = detect_patterns_dataset(events)

    assert len(pattern_map) == 3
    total_patterns = sum(len(patterns) for patterns in pattern_map.values())
    assert total_patterns >= 7


def test_evaluation_report_scores_reasonably():
    dataset = load_dataset(DATASET_PATH)
    events = extract_events_for_dataset(
        dataset,
        config=ExtractionConfig(mode="rule", include_clary_response=False),
        llm=NoOpLLMClient(),
    )
    pattern_map = detect_patterns_dataset(events)
    report = evaluate_predictions(
        load_gold_patterns(DATASET_PATH),
        predictions_from_map(pattern_map),
    )

    assert report["summary"]["recall"] >= 0.6
    assert report["summary"]["precision"] >= 0.5


def test_stream_messages_are_valid_json():
    payloads = [
        stream_start(DATASET_PATH),
        stream_progress("extract", "ok"),
        stream_end(total_users=3, total_patterns=8),
    ]
    for payload in payloads:
        loaded = json.loads(payload)
        assert "type" in loaded
