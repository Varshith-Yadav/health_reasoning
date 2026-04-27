# Ask First Submission Writeup

## 1) Problem framing and approach

I framed this assignment as a temporal causal reasoning task across sessions, not a single-chat classification task.

Pipeline:

1. Event extraction from each conversation into structured events (symptom, trigger, intervention, improvement, worsening).
2. Timeline indexing (`week_index` + timestamp) per user.
3. Dynamic candidate generation (cause->outcome pairs from extracted events, not a hardcoded answer key).
4. Pattern detection + confidence scoring + evaluator against hidden references.

This makes the system runnable in fully rule-based mode while still supporting optional LLM enrichment.

LLM choice:

- I used a hybrid strategy: deterministic rule extraction by default, with optional `openai/gpt-4.1-mini` via `litellm` for semantic enrichment.
- Reason: internship evaluation needs reproducibility; rule-first guarantees stable outputs, while the LLM path improves recall on messy phrasing when enabled.

## 2) Pattern recognition rules used and why

The engine uses explicit temporal rules so output is auditable.

### Rule A: Directionality gate

- Keep pairs where cause occurs before outcome.
- Penalize/reject reverse-order evidence.

Why: causal claims without order are unreliable.

### Rule B: Support threshold across sessions

- Require repeated evidence, not one-off co-occurrence.
- Measure support as matched episodes / eligible episodes.

Why: reduces chance correlations from sparse notes.

### Rule C: Lag-window compatibility

- Match by clinically plausible windows:
- immediate (0-7 days), delayed (multi-week), and staged progression windows.

Why: same pair can mean different things if lag changes.

### Rule D: Effect size vs baseline

- Compare outcome rate after suspected cause vs baseline outcome rate.
- Stronger lift increases confidence.

Why: separates true trigger-like behavior from frequent background symptoms.

### Rule E: Contradiction penalty

- Downweight when outcome repeatedly appears without the cause or with opposite ordering.

Why: prevents overconfident false positives.

### Rule F: Pattern family-specific checks

- `repeated_trigger`: repeated cause-before-outcome episodes.
- `delayed_effect`: stable multi-week lag before onset.
- `intervention_success`: intervention precedes outcome reduction.
- `progression`: ordered downstream symptom sequence.
- `multi_factor`: outcome linked to co-occurring drivers.
- `dose_response`: stronger exposure links to stronger symptom frequency.

Why: each family has a different causal signature, so one generic rule is not enough.

## 3) Chunking and context strategy

- I chunk long per-user timelines into overlapping windows (`max_events_per_chunk=8`, `overlap_events=2`).
- Each chunk includes explicit temporal anchors (`week_index`, timestamps, week span).
- Overlap avoids losing causal chains that cross chunk boundaries.
- Final pattern inference merges evidence across chunks, so output remains user-level, not chunk-level.
## 4) Which rules/families worked best (based on actual run)

I re-ran the project on April 27, 2026:

```bash
python run.py --dataset data/askfirst_synthetic_dataset.json
```

Observed metrics:

- Gold patterns: 8
- Predicted patterns: 11
- Matched patterns: 8
- Precision: 0.727
- Recall: 1.0
- F1: 0.842

Matched by pattern family:

- `repeated_trigger`: 2 matched
- `progression`: 2 matched
- `delayed_effect`: 1 matched
- `dose_response`: 1 matched
- `intervention_success`: 1 matched
- `multi_factor`: 1 matched

Unmatched predictions:

- `progression`: 1 false positive
- `intervention_success`: 1 false positive
- `repeated_trigger`: 1 false positive

Interpretation:

- Best precision on this run came from `delayed_effect`, `dose_response`, and `multi_factor` (1/1 each, though low sample size).
- Most robust/high-coverage family is `repeated_trigger` because it matched multiple gold patterns with strong directionality evidence.
- `progression` is useful but easier to over-predict, so it benefits most from stronger contradiction filtering.

## 5) Failure modes

- Sparse histories can create accidental co-occurrence patterns.
- Multi-factor outcomes can over-attribute one cause when causes travel together.
- Rule-only extraction can miss implicit language.
- Evaluator similarity matching can under-credit semantically-correct but differently worded predictions.

## 6) What I would improve next

1. Add counterfactual checks (symptom absence when cause is absent).
2. Add uncertainty bounds per pattern (instead of single-point confidence).
3. Add stricter progression constraints to cut false positives.
4. Add evaluator rubric for temporal correctness beyond string similarity.
5. Add calibration tracking over time as new sessions arrive.
