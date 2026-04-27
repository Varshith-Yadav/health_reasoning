# Ask First Submission Writeup (1 page)

## 1) Approach to the reasoning problem

I treated the problem as a temporal causal mining task, not a keyword retrieval task.
The pipeline has four stages:

1. **Event extraction**  
   Every session is converted to atomic events (symptoms, diet/lifestyle factors, interventions, improvements, worsening).  
   I used a hybrid extractor:
   - deterministic rules for high precision baseline
   - optional LLM enrichment for messy language variants  
   This keeps the system runnable without API keys while still allowing stronger semantic extraction when an LLM is available.

2. **Timeline memory with temporal indexing**  
   Events are attached to `week_index` and absolute timestamps.  
   I preserve ordering and compute lag windows (0-7 days, 14-84 days, etc.) so the same event pair can mean different things depending on order and delay.

3. **Dynamic pattern mining (no fixed pair list)**  
   I generate candidate cause->outcome pairs directly from extracted events using event types, then detect:
   - repeated triggers
   - delayed effects
   - intervention success
   - progression chains
   - multi-factor drivers
   - root-cause chains  
   Patterns are emitted only when directional and repeated temporal evidence is strong enough.

4. **Confidence scoring with explicit rationale**  
   Confidence score uses measurable factors:
   - support/coverage
   - effect size vs baseline
   - directionality (reverse-order penalty)
   - temporal consistency of lag
   - contradiction penalty  
   Each pattern stores score + one-line justification + evidence items + rejected hypotheses.

I also added an **evaluation module** that compares predicted patterns against hidden references and reports precision/recall/F1 with match trace.

---

## 2) Failure modes / where it can hallucinate confidently

The main failure risk is **over-association from sparse data**:
- If a symptom appears only a few times, co-occurrence can look causal by chance.
- Multi-factor logic can incorrectly prioritize one driver when two factors usually co-occur.
- Rule extraction can miss nuanced language when users describe causes indirectly.

LLM mode adds another risk:
- extractor may return plausible but unsupported labels if prompts are too permissive.
- temporal details can be compressed or normalized incorrectly if timestamps are not enforced in prompt constraints.

The evaluator itself can over-credit or under-credit due to fuzzy text matching:
- semantically correct patterns with different wording might score low.
- surface-level wording overlap can score higher than true causal equivalence.

---

## 3) What I would build with more time

With more time, I would add:

1. **Counterfactual checks**  
   Explicitly test whether symptoms *do not* appear when a suspected trigger is absent (causal robustness).

2. **Per-pattern Bayesian uncertainty**  
   Replace single deterministic score with uncertainty intervals and posterior updates as new sessions arrive.

3. **LLM reasoner second-pass**  
   Keep deterministic candidate generation, then run a constrained LLM verifier that only chooses among evidence-backed hypotheses.

4. **Better evaluator**  
   Add an LLM-judge mode with strict rubric (temporal direction, lag plausibility, intervention response) plus human-auditable scoring breakdown.

5. **Production observability**  
   Track pattern drift, false-positive hotspots, and confidence calibration over time.

Overall, the current system is designed to be transparent, measurable, and easy to improve, while already meeting the assignment goals of temporal reasoning + confidence-backed pattern detection.
