from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from app.core.loader import load_dataset
from app.evaluation.evaluator import (
    evaluate_predictions,
    load_gold_patterns,
    predictions_from_map,
    save_evaluation,
)
from app.extraction.event_extractor import ExtractionConfig, extract_events_for_dataset
from app.llm.client import build_llm_client
from app.output.formatter import build_full_response, build_user_result, save_json, to_json
from app.output.json_streamer import (
    stream_end,
    stream_patterns_map,
    stream_progress,
    stream_start,
)
from app.reasoning.pattern_engine import detect_patterns_dataset


def _build_analysis_json(dataset, pattern_map: Dict[str, List]) -> str:
    user_results = []
    for user in dataset.users:
        patterns = pattern_map.get(user.user_id, [])
        user_results.append(
            build_user_result(
                user_id=user.user_id,
                user_name=user.name,
                patterns=patterns,
            )
        )
    response = build_full_response(user_results, version="2.0.0")
    return to_json(response, indent=2)


def run_analysis(args) -> Dict[str, object]:
    dataset = load_dataset(args.dataset)
    llm = build_llm_client(
        provider=args.llm_provider,
        model=args.model,
        api_base=args.api_base,
    )
    extraction_config = ExtractionConfig(
        mode=args.extraction_mode,
        include_clary_response=args.include_clary_response,
    )

    if args.stream:
        print(stream_start(args.dataset))
        print(stream_progress("load_dataset", "done"))

    event_map = extract_events_for_dataset(dataset, config=extraction_config, llm=llm)
    if args.stream:
        total_events = sum(len(events) for events in event_map.values())
        print(stream_progress("extract_events", f"{total_events} events"))

    pattern_map = detect_patterns_dataset(event_map)
    if args.stream:
        total_patterns = sum(len(patterns) for patterns in pattern_map.values())
        print(stream_progress("detect_patterns", f"{total_patterns} patterns"))
        for line in stream_patterns_map(pattern_map):
            print(line)
        print(stream_end(len(dataset.users), total_patterns))

    output_json = _build_analysis_json(dataset, pattern_map)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_json, encoding="utf-8")

    return {
        "dataset": dataset,
        "event_map": event_map,
        "pattern_map": pattern_map,
        "analysis_output_path": str(output_path),
    }


def run_evaluation(args, pattern_map: Dict[str, List]) -> Dict[str, object]:
    gold_patterns = load_gold_patterns(args.dataset)
    if not gold_patterns:
        return {"skipped": True, "reason": "No hidden_patterns_reference found in dataset."}

    predictions = predictions_from_map(pattern_map)
    report = evaluate_predictions(gold_patterns, predictions, match_threshold=args.match_threshold)
    save_path = save_evaluation(report, args.eval_output)
    report["saved_to"] = str(save_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask First temporal reasoning engine with optional evaluation."
    )
    parser.add_argument(
        "--dataset",
        default="data/askfirst_synthetic_dataset.json",
        help="Path to dataset JSON.",
    )
    parser.add_argument(
        "--output",
        default="outputs/analysis_output.json",
        help="Where final analysis JSON should be saved.",
    )
    parser.add_argument(
        "--eval-output",
        default="outputs/evaluation_report.json",
        help="Where evaluation report should be saved.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip matching predictions against hidden reference.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.46,
        help="Similarity threshold used for evaluation matching.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream NDJSON progress and pattern chunks to stdout.",
    )
    parser.add_argument(
        "--extraction-mode",
        choices=["rule", "llm", "hybrid"],
        default="hybrid",
        help="Event extraction strategy.",
    )
    parser.add_argument(
        "--include-clary-response",
        action="store_true",
        help="Include Clary response text in extraction context.",
    )
    parser.add_argument(
        "--llm-provider",
        default="none",
        help="none | litellm | openai | anthropic | gemini | openrouter",
    )
    parser.add_argument("--model", default=None, help="LLM model identifier.")
    parser.add_argument("--api-base", default=None, help="Optional provider base URL.")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print final analysis JSON to stdout (strict JSON).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    analysis = run_analysis(args)
    pattern_map = analysis["pattern_map"]

    evaluation_report = None
    if not args.skip_eval:
        evaluation_report = run_evaluation(args, pattern_map)

    if args.print_json:
        json_text = Path(analysis["analysis_output_path"]).read_text(encoding="utf-8")
        print(json_text)
    else:
        print(f"Analysis saved to: {analysis['analysis_output_path']}")
        if evaluation_report:
            if evaluation_report.get("skipped"):
                print(f"Evaluation skipped: {evaluation_report['reason']}")
            else:
                summary = evaluation_report["summary"]
                print(
                    "Evaluation: "
                    f"recall={summary['recall']}, precision={summary['precision']}, f1={summary['f1']}"
                )
                print(f"Report saved to: {evaluation_report['saved_to']}")


if __name__ == "__main__":
    main()
