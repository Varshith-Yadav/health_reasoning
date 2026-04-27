# Ask First - Temporal Health Reasoning Engine

This project builds a cross-conversation reasoning layer for health history analysis.
It detects latent patterns across multiple sessions with explicit temporal logic and confidence scoring.

## What this implementation includes

- Dynamic pattern mining (no hardcoded cause->effect pattern list)
- Time-aware reasoning across session history (immediate, delayed, progression, multi-factor)
- Confidence scoring with metric-based justification
- Strict JSON output and NDJSON streaming support
- LLM-agnostic extraction layer (switch providers/models without changing core code)
- Built-in evaluator against hidden reference patterns (precision/recall/F1)
- Streamlit UI + CLI

## Project structure

- `run.py` - CLI entry point (`analyze + evaluate`)
- `app/extraction/event_extractor.py` - hybrid event extraction (rules + optional LLM)
- `app/extraction/symptom_mapper.py` - canonical event ontology and normalization rules
- `app/reasoning/pattern_engine.py` - dynamic temporal pattern detection
- `app/scoring/confidence.py` - confidence math and one-line rationale generation
- `app/evaluation/evaluator.py` - prediction-vs-gold matching and metrics
- `app/output/json_streamer.py` - NDJSON streaming helpers
- `app/ui/app.py` - Streamlit app
- `tests/test_pipeline.py` - end-to-end tests

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run (CLI)

### 1) Analyze + evaluate

```bash
python run.py --dataset data/askfirst_synthetic_dataset.json
```

Outputs:
- `outputs/analysis_output.json`
- `outputs/evaluation_report.json`

### 2) Strict JSON output to stdout

```bash
python run.py --print-json --skip-eval
```

### 3) Stream NDJSON progress/pattern chunks

```bash
python run.py --stream --skip-eval
```

### 4) Enable any LLM provider/model (LLM-agnostic)

```bash
python run.py --llm-provider litellm --model openai/gpt-4.1-mini
```

You can swap model/provider by changing only CLI args:
- OpenAI-style: `openai/gpt-4.1-mini`
- Anthropic-style: `anthropic/claude-3-7-sonnet-latest`
- Gemini-style: `gemini/gemini-2.5-pro`
- OpenRouter-style: `openrouter/<model>`

`litellm` handles provider routing through env keys.

## Streamlit UI

```bash
streamlit run app/ui/app.py
```

## How pattern reasoning works

1. Convert each conversation into structured events (symptoms, behaviors, interventions, improvements).
2. Place events on a user timeline (`week_index` + actual timestamps).
3. Build candidate cause/outcome pairs from event types (dynamic, no fixed pair list).
4. Score pair-level temporal evidence:
   - Directionality (cause before symptom)
   - Repetition across sessions
   - Delay window fit (lagged effects)
   - Effect size vs baseline co-occurrence
5. Emit pattern families:
   - Repeated Trigger
   - Delayed Effect
   - Intervention Success
   - Progression
   - Multi-factor
   - Root Cause Chain
6. Attach reasoning trace and confidence rationale per pattern.

## Chunking and context management

Timeline context is chunked in `app/memory/timelineBuilder.py`:

- `max_events_per_chunk` controls chunk size (default `8`)
- `overlap_events` keeps continuity between chunks (default `2`)
- each chunk has explicit week-span metadata

This strategy avoids context loss on longer histories while preserving temporal ordering.

## Testing

```bash
pytest -q
```

The tests validate:
- end-to-end pipeline detection
- evaluation report quality thresholds
- streaming output format validity

## Notes

- The engine intentionally does not hardcode specific hidden assignment answers.
- Extraction can run fully rule-based when no API keys are available.
- With an LLM enabled, extraction can capture additional nuance from messy language.
