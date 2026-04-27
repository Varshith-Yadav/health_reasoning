from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure package imports work no matter where Streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# When this file is loaded as module name "app", it shadows the app package.
loaded_app_module = sys.modules.get("app")
if loaded_app_module is not None and Path(getattr(loaded_app_module, "__file__", "")).resolve() == Path(__file__).resolve():
    del sys.modules["app"]

from app.core.loader import load_dataset
from app.evaluation.evaluator import evaluate_predictions, load_gold_patterns, predictions_from_map
from app.extraction.event_extractor import ExtractionConfig, extract_events_for_dataset
from app.llm.client import build_llm_client
from app.output.formatter import build_full_response, build_user_result, to_json
from app.reasoning.pattern_engine import detect_patterns_dataset


st.set_page_config(page_title="Ask First Reasoning Engine", layout="wide")
st.title("Ask First - Temporal Health Pattern Reasoner")
st.caption("Dynamic cross-session pattern detection with confidence scoring and reasoning trace.")

st.sidebar.header("Run Settings")
dataset_path = st.sidebar.text_input(
    "Dataset path",
    value="data/askfirst_synthetic_dataset.json",
)
extraction_mode = st.sidebar.selectbox("Extraction mode", options=["hybrid", "rule", "llm"])
provider = st.sidebar.text_input("LLM provider", value="none")
model = st.sidebar.text_input("Model (if provider enabled)", value="")
api_base = st.sidebar.text_input("API base (optional)", value="")
include_clary_response = st.sidebar.checkbox("Include Clary response text", value=False)
run_button = st.sidebar.button("Run Analysis")


def render_pattern(pattern) -> None:
    with st.container(border=True):
        st.subheader(pattern.title)
        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", pattern.confidence)
        col2.metric("Score", pattern.confidence_score)
        col3.metric("Sessions", len(pattern.sessions_involved))

        st.write(pattern.explanation)
        st.caption(pattern.confidence_justification)

        if pattern.root_cause:
            st.write(f"Root cause: {pattern.root_cause}")
        if pattern.outcome:
            st.write(f"Outcome: {pattern.outcome}")

        if pattern.reasoning_trace.steps:
            st.write("Reasoning steps:")
            for step in pattern.reasoning_trace.steps:
                st.write(f"- {step}")

        if pattern.reasoning_trace.evidence:
            st.write("Evidence:")
            for item in pattern.reasoning_trace.evidence:
                st.write(f"- {item.session_id}: {item.summary} ({item.contribution})")

        if pattern.reasoning_trace.rejected_hypotheses:
            st.write("Rejected hypotheses:")
            for item in pattern.reasoning_trace.rejected_hypotheses:
                st.write(f"- {item}")


if run_button:
    try:
        dataset = load_dataset(dataset_path)
        llm = build_llm_client(
            provider=provider,
            model=model or None,
            api_base=api_base or None,
        )
        event_map = extract_events_for_dataset(
            dataset,
            config=ExtractionConfig(
                mode=extraction_mode,
                include_clary_response=include_clary_response,
            ),
            llm=llm,
        )
        pattern_map = detect_patterns_dataset(event_map)

        tabs = st.tabs(["Summary", "Per User", "JSON", "Evaluation"])

        with tabs[0]:
            total_patterns = sum(len(items) for items in pattern_map.values())
            st.metric("Users", len(dataset.users))
            st.metric("Patterns detected", total_patterns)

            for user in dataset.users:
                st.write(f"{user.user_id} - {user.name}: {len(pattern_map.get(user.user_id, []))} patterns")

        with tabs[1]:
            user_ids = [user.user_id for user in dataset.users]
            selected = st.selectbox("User", options=user_ids)
            patterns = pattern_map.get(selected, [])
            if not patterns:
                st.info("No patterns detected.")
            else:
                for pattern in patterns:
                    render_pattern(pattern)

        with tabs[2]:
            user_results = []
            for user in dataset.users:
                user_results.append(
                    build_user_result(
                        user_id=user.user_id,
                        user_name=user.name,
                        patterns=pattern_map.get(user.user_id, []),
                    )
                )
            response = build_full_response(user_results, version="2.0.0")
            payload = to_json(response, indent=2)
            st.code(payload, language="json")
            st.download_button(
                "Download JSON",
                data=payload,
                file_name="analysis_output.json",
                mime="application/json",
            )

        with tabs[3]:
            gold = load_gold_patterns(dataset_path)
            if not gold:
                st.info("No hidden reference found in dataset.")
            else:
                report = evaluate_predictions(gold, predictions_from_map(pattern_map))
                st.json(report["summary"])
                st.write("Matched patterns:")
                for match in report["matches"]:
                    st.write(
                        f"- {match['gold_pattern_id']} -> {match['prediction_title']} "
                        f"(score {match['similarity_score']})"
                    )

    except Exception as exc:  # pragma: no cover
        st.error(str(exc))
