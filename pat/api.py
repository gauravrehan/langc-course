"""REST endpoints for the PAT AI Intelligence prototype, mounted at /pat."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from pat.advisor_graph import builder as advisor_builder
from pat.advisor_graph import run_advisor
from pat.config import gate_path, load_config
from pat.data import new_in_flight_id
from pat.models import (
    AdviseRequest,
    AdvisorResponse,
    GateEvent,
    GateEventRequest,
    GateStep,
    InterventionRecord,
    Notification,
    Pat,
    PatHealthResponse,
    PatRequest,
    ResumeRequest,
    TickRequest,
    TickResponse,
)
from pat.monitor_graph import builder as monitor_builder
from pat.monitor_graph import pending_interrupt, resume_tick, start_tick
from pat.notify import LogNotifier
from pat.predict import PredictionModel, coverage_info, current_loads, time_remaining_hours
from pat.repository import InMemoryPatRepository, build_repository
from pat.runtime import PatContext
from pat.seed import seed

logger = logging.getLogger("pat")

router = APIRouter(prefix="/pat", tags=["PAT AI Intelligence"])


@dataclass
class PatServices:
    context: PatContext
    advisor: Any
    monitor: Any


_services: PatServices | None = None


def build_services(checkpointer: Any) -> PatServices:
    repo = build_repository()
    history = repo.load_history()
    if not history and isinstance(repo, InMemoryPatRepository):
        logger.info("Seeding the in-memory PAT store with synthetic data")
        seed(repo)
        history = repo.load_history()
    if not history:
        raise HTTPException(
            status_code=503,
            detail="No PAT history found. Run `uv run python -m pat.seed` to load synthetic data.",
        )
    context = PatContext(repo=repo, model=PredictionModel(history, load_config()), notifier=LogNotifier())
    return PatServices(
        context=context,
        advisor=advisor_builder.compile(),
        monitor=monitor_builder.compile(checkpointer=checkpointer),
    )


def set_services(services: PatServices | None) -> None:
    """Replace the services (used by tests)."""
    global _services
    _services = services


def get_services() -> PatServices:
    global _services
    if _services is None:
        from chat_graph import checkpointer  # share the chat API's checkpointer

        _services = build_services(checkpointer)
    return _services


def _get_pat(services: PatServices, pat_id: str) -> Pat:
    pat = services.context.repo.get_pat(pat_id)
    if pat is None:
        raise HTTPException(status_code=404, detail=f"PAT {pat_id} not found")
    return pat


def _tick_response(pat_id: str, result: dict[str, Any]) -> TickResponse:
    interrupts = result.get("__interrupt__") or ()
    assessment = result.get("assessment") or None
    if interrupts:
        payload = interrupts[0].value
        return TickResponse(
            pat_id=pat_id,
            status="awaiting_confirmation",
            message=payload["message"],
            assessment=assessment,
            pending=payload,
        )
    if result.get("closed"):
        return TickResponse(pat_id=pat_id, status="closed", message=result.get("message", ""))
    return TickResponse(
        pat_id=pat_id,
        status="completed",
        message=result.get("message", ""),
        assessment=assessment,
        intervention=result.get("intervention"),
    )


# Sample execution:
#   POST /pat/advise
#   {"title": "Rates outlook - TV interview",
#    "description": "Speaking at a television interview about the interest rate outlook. Will reference Contoso Bank.",
#    "eventType": "media", "region": "AMER", "city": "New York",
#    "eventTime": "2026-11-26T15:00:00-05:00"}
# Returns the outlook, similar PATs, risk drivers, recommended actions and an AI summary.
@router.post("/advise", response_model=AdvisorResponse)
def advise(request: AdviseRequest) -> AdvisorResponse:
    services = get_services()
    as_of = request.as_of or datetime.now(UTC)
    if request.event_time <= as_of:
        raise HTTPException(status_code=422, detail="eventTime must be in the future")
    pat_request = PatRequest.model_validate(request.model_dump(exclude={"as_of"}))
    return run_advisor(services.advisor, services.context, pat_request, as_of)


@router.get("", response_model=list[dict[str, Any]])
def list_pats(status: str | None = "in_progress") -> list[dict[str, Any]]:
    services = get_services()
    now = datetime.now(UTC)
    pats = services.context.repo.list_pats(status if status != "all" else None)
    rows = []
    for pat in sorted(pats, key=lambda item: item.event_time):
        pending = pat.pending_gate()
        rows.append({
            "id": pat.id,
            "title": pat.title,
            "status": pat.status,
            "eventType": pat.event_type,
            "region": pat.region,
            "eventTime": pat.event_time,
            "hoursToEvent": time_remaining_hours(pat, now),
            "pendingGate": pending.gate if pending else None,
            "awaitingConfirmation": pending_interrupt(services.monitor, pat.id) is not None,
        })
    return rows


# Sample execution:
#   POST /pat   (same body as /pat/advise)
# Creates an in-flight PAT with its first gate pending.
@router.post("", response_model=Pat, status_code=201)
def submit_pat(request: PatRequest) -> Pat:
    services = get_services()
    now = datetime.now(UTC)
    if request.event_time <= now:
        raise HTTPException(status_code=422, detail="eventTime must be in the future")
    path = gate_path(request.event_type, load_config())
    gates = [GateStep(gate=gate) for gate in path]
    gates[0].status = "pending"
    gates[0].started_at = now
    pat = Pat(**request.model_dump(), id=new_in_flight_id(), submitted_at=now, gates=gates)
    services.context.repo.save_pat(pat)
    services.context.repo.add_event(GateEvent(pat_id=pat.id, type="submitted", gate=path[0], timestamp=now))
    return pat


# Sample execution:
#   POST /pat/PAT-1A2B3C/events
#   {"type": "gate_approved"}            # approves the pending gate
#   {"type": "reminder_sent"}            # records a workflow reminder
@router.post("/{pat_id}/events", response_model=Pat)
def add_event(pat_id: str, request: GateEventRequest) -> Pat:
    services = get_services()
    repo = services.context.repo
    pat = _get_pat(services, pat_id)
    if pat.status != "in_progress":
        raise HTTPException(status_code=409, detail=f"PAT {pat_id} is {pat.status}")
    timestamp = request.timestamp or datetime.now(UTC)
    pending = pat.pending_gate()
    gate = request.gate or (pending.gate if pending else None)
    if pending is None or gate != pending.gate:
        raise HTTPException(status_code=409, detail=f"Gate {gate} is not the pending gate")
    if request.type == "reminder_sent":
        pending.reminders.append(timestamp)
    else:
        pending.status = "approved"
        pending.completed_at = timestamp
        following = next((step for step in pat.gates if step.status == "waiting"), None)
        if following:
            following.status = "pending"
            following.started_at = timestamp
        else:
            pat.status = "approved"
            pat.completed_at = timestamp
    repo.save_pat(pat)
    repo.add_event(GateEvent(pat_id=pat.id, type=request.type, gate=gate, timestamp=timestamp))
    if pat.status == "approved":
        repo.add_event(GateEvent(pat_id=pat.id, type="approved", timestamp=timestamp))
    return pat


# Sample execution:
#   POST /pat/PAT-1A2B3C/tick
#   {}                                   # or {"asOf": "2026-11-26T18:00:00Z"} to simulate a later time
# Runs one Approval Intelligence Agent pass. May return status "awaiting_confirmation"
# when the agent wants to escalate; answer with /pat/{id}/resume.
@router.post("/{pat_id}/tick", response_model=TickResponse)
def tick(pat_id: str, request: TickRequest = Body(default_factory=TickRequest)) -> TickResponse:
    services = get_services()
    _get_pat(services, pat_id)
    if pending_interrupt(services.monitor, pat_id):
        raise HTTPException(status_code=409, detail="An escalation is awaiting confirmation; use /resume first")
    result = start_tick(services.monitor, services.context, pat_id, request.as_of)
    return _tick_response(pat_id, result)


# Sample execution:
#   POST /pat/PAT-1A2B3C/resume
#   {"message": "Yes, escalate it"}
@router.post("/{pat_id}/resume", response_model=TickResponse)
def resume(pat_id: str, request: ResumeRequest) -> TickResponse:
    services = get_services()
    _get_pat(services, pat_id)
    if not pending_interrupt(services.monitor, pat_id):
        raise HTTPException(status_code=409, detail="Nothing is awaiting confirmation for this PAT")
    result = resume_tick(services.monitor, services.context, pat_id, request.message)
    return _tick_response(pat_id, result)


# Sample execution:
#   POST /pat/monitor/run
#   {}
# Runs the agent for every in-flight PAT that isn't waiting for a confirmation.
@router.post("/monitor/run", response_model=list[TickResponse])
def monitor_run(request: TickRequest = Body(default_factory=TickRequest)) -> list[TickResponse]:
    services = get_services()
    responses = []
    for pat in services.context.repo.list_pats("in_progress"):
        if pending_interrupt(services.monitor, pat.id):
            continue
        result = start_tick(services.monitor, services.context, pat.id, request.as_of)
        responses.append(_tick_response(pat.id, result))
    return responses


@router.get("/{pat_id}/health", response_model=PatHealthResponse)
def health(pat_id: str) -> PatHealthResponse:
    services = get_services()
    context = services.context
    pat = _get_pat(services, pat_id)
    now = datetime.now(UTC)
    history: list[InterventionRecord] = context.repo.list_interventions(pat_id)
    outlook = None
    if pat.status == "in_progress" and pat.event_time > now:
        loads = current_loads(context.repo.list_pats("in_progress"), load_config())
        outlook = context.model.outlook(pat, now, loads)
    pending = pat.pending_gate()
    return PatHealthResponse(
        pat=pat,
        as_of=now,
        time_remaining_hours=time_remaining_hours(pat, now),
        blocking_gate=pending.gate if pending else None,
        outlook=outlook,
        latest=history[-1] if history else None,
        history=history,
        awaiting_confirmation=pending_interrupt(services.monitor, pat_id),
    )


@router.get("/{pat_id}/notifications", response_model=list[Notification])
def notifications(pat_id: str) -> list[Notification]:
    services = get_services()
    _get_pat(services, pat_id)
    return [
        record.notification
        for record in services.context.repo.list_interventions(pat_id)
        if record.notification is not None
    ]


@router.get("/{pat_id}/coverage")
def coverage(pat_id: str) -> dict[str, Any]:
    services = get_services()
    pat = _get_pat(services, pat_id)
    pending = pat.pending_gate()
    if not pending:
        return {"blockingGate": None}
    return coverage_info(pending.gate, pat, datetime.now(UTC), load_config())
