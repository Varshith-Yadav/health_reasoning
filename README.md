# Ask First - Temporal Health Reasoning Engine

This project detects health patterns across multiple conversations using timeline-aware reasoning and confidence scoring.

## What is included

- Dynamic pattern mining (no fixed hidden answer list)
- Temporal reasoning across sessions (immediate, delayed, progression, multi-factor)
- Confidence scoring with explicit justification text
- JSON + NDJSON streaming output
- LLM-optional extraction (fully runnable with rules only)
- Built-in evaluator (precision/recall/F1)
- CLI and Streamlit UI

## LLM choice and rationale

- Primary default: rule-first extraction (`--llm-provider none`) for deterministic, reproducible grading without API keys.
- Optional model used for enrichment: `openai/gpt-4.1-mini` via `litellm`.
- Why this choice: good cost/quality tradeoff for noisy symptom language, while keeping the core reasoning deterministic and auditable.

## Project structure

- `run.py`: CLI entry point (`analyze + evaluate`)
- `app/extraction/event_extractor.py`: hybrid event extraction (rules + optional LLM)
- `app/extraction/symptom_mapper.py`: symptom/event normalization rules
- `app/reasoning/pattern_engine.py`: temporal pattern mining
- `app/scoring/confidence.py`: confidence scoring
- `app/evaluation/evaluator.py`: prediction vs gold evaluation
- `app/output/json_streamer.py`: NDJSON stream helpers
- `app/ui/app.py`: Streamlit app
- `tests/test_pipeline.py`: end-to-end tests

## Local setup

### Prerequisites

- Python 3.10+

### Install

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the pipeline (CLI)

### 1) Analyze + evaluate

```bash
python run.py --dataset data/askfirst_synthetic_dataset.json
```

Expected outputs:

- `outputs/analysis_output.json`
- `outputs/evaluation_report.json`

Latest verified run (April 27, 2026):

- `predicted_patterns=11`
- `matched_patterns=8`
- `precision=0.727`
- `recall=1.0`
- `f1=0.842`

### 2) Strict JSON to stdout

```bash
python run.py --print-json --skip-eval
```

### 3) NDJSON streaming mode

```bash
python run.py --stream --skip-eval
```

### 4) Enable LLM extraction (optional)

```bash
python run.py --llm-provider litellm --model openai/gpt-4.1-mini
```

Provider/model is CLI-swappable (OpenAI, Anthropic, Gemini, OpenRouter through `litellm`).

## Streamlit UI

```bash
streamlit run app/ui/app.py
```

## Test

```bash
pytest -q
```

Latest verified test run (April 27, 2026): `3 passed`.

## Reasoning summary

1. Convert conversation text into structured events.
2. Anchor events on per-user timelines (`week_index` + timestamps).
3. Build dynamic cause->outcome candidates from event types.
4. Score evidence using directionality, support, lag fit, and effect size.
5. Emit pattern families (repeated trigger, delayed effect, intervention success, progression, multi-factor, dose-response).
6. Attach confidence score + explanation + evidence trace.

## Chunking and context management

- Timeline chunking is implemented in `app/memory/timelineBuilder.py`.
- `max_events_per_chunk=8` keeps reasoning windows focused and token-efficient.
- `overlap_events=2` preserves continuity at chunk boundaries.
- Each chunk carries week-span metadata, so temporal order is preserved across chunked reasoning calls.
- Cross-session pattern detection aggregates chunk-level evidence into one user-level reasoning trace.

## Assignment requirement mapping

- Cross-conversation temporal pattern reasoning: yes (`app/reasoning/pattern_engine.py`).
- Confidence score + one-line justification per pattern: yes (`confidence` + `confidence_justification` fields in output JSON).
- Strict JSON output: yes (`--print-json` and `outputs/analysis_output.json`).
- Streaming support: yes (`--stream`, NDJSON from `app/output/json_streamer.py`).
- No hardcoded hidden patterns: yes (dynamic candidate generation from extracted events).
- Reasoning trace included: yes (`reasoning_trace.steps/evidence/rejected_hypotheses`).

## Notes

- No API key is required for rule-only mode.
- The system is designed to be auditable (trace + confidence rationale per pattern).
