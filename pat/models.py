"""Pydantic models for PAT requests, workflow state, predictions and agent output.

Stored and internal data use snake_case; API JSON uses camelCase aliases,
matching the existing chat API.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

GateCode = Literal["RM", "SA", "CG", "CC"]
EventType = Literal["media", "conference", "client", "internal"]
Region = Literal["AMER", "EMEA", "APAC"]
SpeakerRole = Literal["analyst", "strategist", "economist"]
LoadBand = Literal["low", "medium", "high"]
Health = Literal["on_track", "at_risk", "critical"]
PatStatus = Literal["in_progress", "approved", "terminated"]
GateStatus = Literal["waiting", "pending", "approved"]
Confidence = Literal["high", "medium", "low"]
Impact = Literal["high", "medium", "low"]
Urgency = Literal["low", "medium", "high"]


def ensure_utc(value: datetime) -> datetime:
    """Treat naive datetimes as UTC and normalise aware ones to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# --------------------------------------------------------------------------
# PAT and workflow
# --------------------------------------------------------------------------


class GateStep(CamelModel):
    gate: GateCode
    status: GateStatus = "waiting"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    queue_load: LoadBand | None = None
    reminders: list[datetime] = Field(default_factory=list)

    @field_validator("started_at", "completed_at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value else None

    @field_validator("reminders")
    @classmethod
    def _utc_list(cls, value: list[datetime]) -> list[datetime]:
        return [ensure_utc(item) for item in value]


class PatRequest(CamelModel):
    """What a submitter fills in. Used by the Submission Advisor and to submit a PAT."""

    title: str
    description: str
    event_type: EventType
    region: Region
    city: str
    event_time: datetime
    speaker_role: SpeakerRole = "analyst"
    submitter: str = "submitter"
    securities: list[str] = Field(default_factory=list)
    sovereign_entities: list[str] = Field(default_factory=list)
    cross_region: bool = False

    @field_validator("event_time")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class Pat(PatRequest):
    id: str
    submitted_at: datetime
    status: PatStatus = "in_progress"
    gates: list[GateStep]
    completed_at: datetime | None = None

    @field_validator("submitted_at", "completed_at")
    @classmethod
    def _utc_times(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value else None

    def pending_gate(self) -> GateStep | None:
        return next((step for step in self.gates if step.status == "pending"), None)

    def remaining_gates(self) -> list[GateStep]:
        return [step for step in self.gates if step.status != "approved"]


class GateEvent(CamelModel):
    pat_id: str
    type: Literal["submitted", "gate_approved", "reminder_sent", "terminated", "approved"]
    gate: GateCode | None = None
    timestamp: datetime
    note: str | None = None


class GateEventRequest(CamelModel):
    type: Literal["gate_approved", "reminder_sent"]
    gate: GateCode | None = Field(
        default=None,
        description="Defaults to the PAT's pending gate.",
    )
    timestamp: datetime | None = None


# --------------------------------------------------------------------------
# Predictions (computed in code)
# --------------------------------------------------------------------------


class JourneyStep(CamelModel):
    gate: GateCode
    display_name: str
    status: GateStatus
    expected_hours_p50: float | None = None
    expected_hours_p90: float | None = None


class Outlook(CamelModel):
    probability: float
    probability_pct: int
    p50_eta: datetime
    p90_eta: datetime
    hours_to_complete_p50: float
    hours_to_complete_p90: float
    hours_to_event: float
    confidence: Confidence
    samples_used: int
    journey: list[JourneyStep]


class SimilarPat(CamelModel):
    id: str
    title: str
    event_type: EventType
    region: Region
    lead_hours: float
    status: PatStatus
    approval_hours: float | None


class SimilarSummary(CamelModel):
    count: int
    on_time_rate_pct: int
    last_minute_rate_pct: int
    median_approval_hours: float | None
    slowest_gate: GateCode | None
    examples: list[SimilarPat]


class RiskDriver(CamelModel):
    id: str
    label: str
    detail: str
    impact: Impact
    why: str | None = None


# --------------------------------------------------------------------------
# Submission Advisor
# --------------------------------------------------------------------------


class ExtractedSignals(CamelModel):
    securities: list[str] = Field(default_factory=list)
    sovereign_entities: list[str] = Field(default_factory=list)
    sensitive_topics: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(
        default_factory=list,
        description="Entities the LLM proposed that were not found in the request text.",
    )


class RecommendedAction(CamelModel):
    id: str
    phase: Literal["before_submit", "after_submit"]
    label: str
    reason: str


class AdviseRequest(PatRequest):
    as_of: datetime | None = None

    @field_validator("as_of")
    @classmethod
    def _utc_as_of(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value else None


class AdvisorResponse(CamelModel):
    as_of: datetime
    outlook: Outlook
    similar: SimilarSummary
    signals: ExtractedSignals
    risk_drivers: list[RiskDriver]
    recommended_actions: list[RecommendedAction]
    summary: str
    summary_source: Literal["llm", "template"]
    suggested_monitoring: list[str]


# --------------------------------------------------------------------------
# Approval Intelligence Agent
# --------------------------------------------------------------------------


class HealthAssessment(CamelModel):
    health: Health
    computed_health: Health
    llm_health: Health | None = None
    overridden: bool = False
    termination_risk_pct: int
    blocking_gate: GateCode | None
    time_remaining_hours: float
    explanation: str
    explanation_source: Literal["llm", "template"]


class Notification(CamelModel):
    pat_id: str
    audience: Literal["submitter", "approval_group", "backup"]
    recipient: str
    urgency: Urgency
    title: str
    body: str
    approval_group_note: str | None = None
    health: Health
    termination_risk_pct: int
    time_remaining_hours: float
    blocking_gate: GateCode | None
    recommended_steps: list[str] = Field(default_factory=list)
    reminder_count: int
    links: dict[str, str] = Field(default_factory=dict)
    source: Literal["llm", "template"]


class InterventionRecord(CamelModel):
    id: str
    pat_id: str
    created_at: datetime
    as_of: datetime
    reasoning_mode: Literal["tools", "fallback"]
    tool_calls: list[str] = Field(default_factory=list)
    agent_notes: str = ""
    assessment: HealthAssessment
    proposed_action: str
    action: str
    urgency: Urgency
    reason: str
    policy_notes: list[str] = Field(default_factory=list)
    outcome: Literal["sent", "no_action", "suppressed", "cancelled"]
    notification: Notification | None = None


class TickRequest(CamelModel):
    as_of: datetime | None = None

    @field_validator("as_of")
    @classmethod
    def _utc_as_of(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value else None


class ResumeRequest(CamelModel):
    message: str


class TickResponse(CamelModel):
    pat_id: str
    status: Literal["completed", "awaiting_confirmation", "closed"]
    message: str
    assessment: HealthAssessment | None = None
    intervention: InterventionRecord | None = None
    pending: dict[str, Any] | None = None


class PatHealthResponse(CamelModel):
    pat: Pat
    as_of: datetime
    time_remaining_hours: float
    blocking_gate: GateCode | None
    outlook: Outlook | None
    latest: InterventionRecord | None
    history: list[InterventionRecord]
    awaiting_confirmation: dict[str, Any] | None = None
