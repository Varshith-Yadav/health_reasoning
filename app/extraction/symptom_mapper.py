from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from app.core.schemas import EventType


@dataclass(frozen=True)
class RuleCandidate:
    label: str
    event_type: EventType
    match_text: str
    metadata: Dict[str, str]


def _compile(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


# Generic health ontology. This is event normalization (not hardcoded patterns).
CANONICAL_PATTERNS: Dict[str, Dict[str, object]] = {
    "stomach pain": {
        "event_type": EventType.SYMPTOM,
        "patterns": _compile(
            [
                r"\bstomach\b.*\b(hurt|pain|burn)",
                r"\bacidity\b",
                r"\bburning\b.*\bstomach\b",
                r"\bgastric\b",
                r"\bstomach again\b",
                r"\bstomach is acting up\b",
            ]
        ),
    },
    "headache": {
        "event_type": EventType.SYMPTOM,
        "patterns": _compile(
            [
                r"\bheadache(s)?\b",
                r"\bpressure behind (my )?eyes\b",
                r"\bhead pain\b",
            ]
        ),
    },
    "fatigue": {
        "event_type": EventType.SYMPTOM,
        "patterns": _compile(
            [
                r"\btired\b",
                r"\bexhausted\b",
                r"\blow energy\b",
                r"\bbrain fog\b",
                r"\benergy crash\b",
                r"\bcrash\b",
                r"\bcan.?t focus\b",
            ]
        ),
    },
    "dizziness": {
        "event_type": EventType.SYMPTOM,
        "patterns": _compile(
            [
                r"\bdizzy\b",
                r"\bdizziness\b",
                r"\blight[- ]?headed\b",
            ]
        ),
    },
    "hair fall": {
        "event_type": EventType.SYMPTOM,
        "patterns": _compile(
            [
                r"\bhair\b.*\b(fall|falling|loss|losing)\b",
                r"\bfinding it everywhere\b",
            ]
        ),
    },
    "acne": {
        "event_type": EventType.SYMPTOM,
        "patterns": _compile(
            [
                r"\bacne\b",
                r"\bbreaking out\b",
                r"\bpimple(s)?\b",
                r"\bskin\b.*\bbreakout\b",
            ]
        ),
    },
    "cramps": {
        "event_type": EventType.SYMPTOM,
        "patterns": _compile(
            [
                r"\bcramps?\b",
                r"\bperiod cramps?\b",
                r"\bmenstrual pain\b",
            ]
        ),
    },
    "anxiety": {
        "event_type": EventType.SYMPTOM,
        "patterns": _compile(
            [
                r"\banxious\b",
                r"\banxiety\b",
                r"\bgeneral unease\b",
                r"\blow level but constant\b",
            ]
        ),
    },
    "late eating": {
        "event_type": EventType.DIET,
        "patterns": _compile(
            [
                r"\blate dinner\b",
                r"\b(midnight|11(:\d\d)?\s?pm|12(:\d\d)?\s?am)\b",
                r"\bate around 11\b",
                r"\bdinner at midnight\b",
            ]
        ),
    },
    "low water intake": {
        "event_type": EventType.LIFESTYLE,
        "patterns": _compile(
            [
                r"\b(2|3)\s+glasses of water\b",
                r"\bbarely any water\b",
                r"\bnot enough water\b",
                r"\blow water intake\b",
            ]
        ),
    },
    "high caffeine": {
        "event_type": EventType.DIET,
        "patterns": _compile(
            [
                r"\b(3|4|5)\s+cups\b.*\bcoffee\b",
                r"\bmore coffee\b",
                r"\bhigh caffeine\b",
            ]
        ),
    },
    "calorie restriction": {
        "event_type": EventType.DIET,
        "patterns": _compile(
            [
                r"\b(under\s+)?\d{3,4}\s*calories\b",
                r"\blow calories\b",
                r"\bcalorie restriction\b",
                r"\bcutting down.*calories\b",
                r"\bintermittent fasting\b",
            ]
        ),
    },
    "dairy intake": {
        "event_type": EventType.DIET,
        "patterns": _compile(
            [
                r"\bdairy\b",
                r"\byogurt\b",
                r"\bgreek yogurt\b",
                r"\bpaneer\b",
            ]
        ),
    },
    "high stress": {
        "event_type": EventType.STRESS,
        "patterns": _compile(
            [
                r"\bstress(ful)?\b",
                r"\bdeadline\b",
                r"\bbig launch\b",
                r"\bwork pressure\b",
                r"\bmassive release\b",
            ]
        ),
    },
    "late night screens": {
        "event_type": EventType.SLEEP,
        "patterns": _compile(
            [
                r"\breels\b",
                r"\bseries\b.*\b(1|2)\s?am\b",
                r"\bphone\b.*\bagain\b",
                r"\blate night screen\b",
                r"\bscreens?\b.*\bnight\b",
                r"\bbecause of screens\b",
            ]
        ),
    },
    "sleep deprivation": {
        "event_type": EventType.SLEEP,
        "patterns": _compile(
            [
                r"\bsleep is bad\b",
                r"\bsleep has been broken\b",
                r"\bwake up tired\b",
                r"\bsleep deprivation\b",
                r"\bstaying up late\b",
                r"\bsleep is still bad\b",
            ]
        ),
    },
    "high carb low protein lunch": {
        "event_type": EventType.DIET,
        "patterns": _compile(
            [
                r"\brace\b.*\bdal\b",
                r"\bno extra protein\b",
                r"\bbiscuits\b",
                r"\bbig meal\b",
            ]
        ),
    },
    "added protein lunch": {
        "event_type": EventType.INTERVENTION,
        "patterns": _compile(
            [
                r"\badded\b.*\b(chicken|egg|protein)\b.*\blunch\b",
                r"\bprotein at lunch\b",
            ]
        ),
    },
    "increased calories": {
        "event_type": EventType.INTERVENTION,
        "patterns": _compile(
            [
                r"\bstarted eating more\b",
                r"\bsince eating more\b",
                r"\bincreased calories\b",
            ]
        ),
    },
    "reduced dairy intake": {
        "event_type": EventType.INTERVENTION,
        "patterns": _compile(
            [
                r"\bcut dairy\b",
                r"\breduced dairy\b",
                r"\bstopped dairy\b",
            ]
        ),
    },
}


IMPROVEMENT_PATTERNS = _compile(
    [
        r"\bbetter\b",
        r"\bclear(er)?\b",
        r"\bgone\b",
        r"\bresolved\b",
        r"\bimproved\b",
        r"\bstopped\b",
    ]
)


WORSENING_PATTERNS = _compile(
    [
        r"\bworse\b",
        r"\bacting up again\b",
        r"\bagain\b",
        r"\bmore than usual\b",
        r"\breally bad\b",
    ]
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_rule_candidates(text: str) -> List[RuleCandidate]:
    normalized = normalize_text(text)
    out: List[RuleCandidate] = []

    for label, cfg in CANONICAL_PATTERNS.items():
        patterns = cfg["patterns"]
        event_type = cfg["event_type"]
        for pattern in patterns:
            matched = pattern.search(normalized)
            if matched:
                if label == "high stress" and re.search(
                    r"\bstress\s+is\s+(actually\s+)?low\b|\bstress low\b", normalized
                ):
                    break
                out.append(
                    RuleCandidate(
                        label=label,
                        event_type=event_type,
                        match_text=matched.group(0),
                        metadata={"rule": "regex_ontology"},
                    )
                )
                break

    if any(pattern.search(normalized) for pattern in IMPROVEMENT_PATTERNS):
        out.append(
            RuleCandidate(
                label="health improvement",
                event_type=EventType.IMPROVEMENT,
                match_text="improvement cue",
                metadata={"rule": "improvement_signal"},
            )
        )

    if any(pattern.search(normalized) for pattern in WORSENING_PATTERNS):
        out.append(
            RuleCandidate(
                label="symptom worsening",
                event_type=EventType.WORSENING,
                match_text="worsening cue",
                metadata={"rule": "worsening_signal"},
            )
        )

    dedup: Dict[Tuple[str, EventType], RuleCandidate] = {}
    for item in out:
        dedup[(item.label, item.event_type)] = item
    return list(dedup.values())
