from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict



class SeverityLevel(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class EventType(str, Enum):
    SYMPTOM = "symptom"
    LIFESTYLE = "lifestyle"
    DIET = "diet"
    MEDICATION = "medication"
    SLEEP = "sleep"
    STRESS = "stress"
    WORK = "work"
    IMPROVEMENT = "improvement"
    WORSENING = "worsening"
    INTERVENTION = "intervention"
    EXERCISE = "exercise"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PatternType(str, Enum):
    REPEATED_TRIGGER = "repeated_trigger"
    DELAYED_EFFECT = "delayed_effect"
    DOSE_RESPONSE = "dose_response"
    TEMPORAL_CORRELATION = "temporal_correlation"
    INTERVENTION_SUCCESS = "intervention_success"
    MULTI_FACTOR = "multi_factor"
    PROGRESSION = "progression"
    ROOT_CAUSE_CHAIN = "root_cause_chain"
    UNKNOWN = "unknown"




class Conversation(BaseModel):
    session_id: str
    timestamp: datetime
    user_message: str
    clary_questions: List[str] = Field(default_factory=list)
    user_followup: Optional[str] = None
    clary_response: Optional[str] = None
    severity: SeverityLevel = SeverityLevel.NONE
    tags: List[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    user_id: str
    name: str
    age: int
    gender: str
    location: str
    occupation: str
    onboarding_notes: Optional[str] = None
    conversations: List[Conversation]


class DatasetInfo(BaseModel):
    version: str
    created_for: str
    total_users: int
    total_conversations: int
    date_range: str
    note: Optional[str] = None


class InputDataset(BaseModel):
    dataset_info: DatasetInfo
    users: List[UserProfile]



class ExtractedEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    event_id: str
    user_id: str
    session_id: str

    timestamp: datetime
    week_index: int = Field(..., description="Week relative to user's first session")

    event_type: EventType
    label: str = Field(..., description="Short canonical label")
    raw_text: str = Field(..., description="Original extracted sentence")

    normalized_value: Optional[str] = None
    severity: Optional[SeverityLevel] = None

    duration_days: Optional[int] = None
    quantity: Optional[str] = None

    source: str = Field(
        default="llm_extraction",
        description="llm_extraction / rule_based / hybrid"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)




class EvidenceItem(BaseModel):
    session_id: str
    timestamp: datetime
    summary: str
    contribution: str


class ReasoningTrace(BaseModel):
    steps: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    rejected_hypotheses: List[str] = Field(default_factory=list)




class DetectedPattern(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    pattern_id: str
    user_id: str

    pattern_type: PatternType
    title: str
    explanation: str

    root_cause: Optional[str] = None
    outcome: Optional[str] = None

    sessions_involved: List[str] = Field(default_factory=list)

    confidence: ConfidenceLevel
    confidence_score: float = Field(..., ge=0.0, le=1.0)

    confidence_justification: str

    reasoning_trace: ReasoningTrace

    recommended_next_step: Optional[str] = None




class UserAnalysisResult(BaseModel):
    user_id: str
    user_name: str
    total_patterns_found: int
    patterns: List[DetectedPattern]


class FullAnalysisResponse(BaseModel):
    generated_at: datetime
    system_version: str = "1.0.0"
    results: List[UserAnalysisResult]




def confidence_from_score(score: float) -> ConfidenceLevel:
    if score >= 0.90:
        return ConfidenceLevel.VERY_HIGH
    elif score >= 0.75:
        return ConfidenceLevel.HIGH
    elif score >= 0.50:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW