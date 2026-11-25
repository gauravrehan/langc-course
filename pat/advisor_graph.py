"""Submission Advisor: predicts and explains a PAT's approval outlook before it is submitted.

    extract_signals (LLM) -> validate_signals -> compute_outlook
      -> assess_risk (LLM) -> recommend_actions (LLM) -> summarise (LLM)
      -> validate_output -> (summarise again once | respond)

LLM nodes read the request text and the computed facts; code computes every
number and checks every LLM output.
"""

import json
from datetime import UTC, datetime
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, create_model
from typing_extensions import TypedDict

from pat import llm as pat_llm
from pat.clock import format_local
from pat.config import gate_display_name, load_config
from pat.guards import allowed_numbers, filter_entities, grounding_problems
from pat.models import (
    AdvisorResponse,
    ExtractedSignals,
    Outlook,
    PatRequest,
    RecommendedAction,
    RiskDriver,
    SimilarSummary,
)
from pat.predict import candidate_risk_drivers, current_loads
from pat.runtime import PatContext

MAX_SUMMARY_RETRIES = 1
MAX_RECOMMENDED_ACTIONS = 6


class AdvisorState(TypedDict, total=False):
    request: dict[str, Any]
    as_of: str
    signals: dict[str, Any]
    outlook: dict[str, Any]
    similar: dict[str, Any]
    candidates: list[dict[str, Any]]
    facts: dict[str, Any]
    risk_drivers: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    summary: str
    summary_source: str
    summary_attempts: int
    summary_feedback: str | None
    response: dict[str, Any]


# --------------------------------------------------------------------------
# LLM output schemas
# --------------------------------------------------------------------------


class SignalExtraction(BaseModel):
    securities: list[str] = Field(
        default_factory=list,
        description="Companies, issuers or securities named in the text, exactly as written.",
    )
    sovereign_entities: list[str] = Field(
        default_factory=list,
        description="Countries or sovereign issuers named in the text, exactly as written.",
    )
    sensitive_topics: list[str] = Field(
        default_factory=list,
        description="Short phrases from the text that a compliance reviewer would look at closely.",
    )


class AdvisorSummary(BaseModel):
    summary: str = Field(description="Three or four sentences for the submitter.")


def _risk_ranking_schema(driver_ids: list[str]) -> type[BaseModel]:
    ranked = create_model(
        "RankedDriver",
        id=(Literal[tuple(driver_ids)], Field(description="A driver id from the list.")),
        why=(str, Field(description="One sentence on why it matters for this PAT, using only the given facts.")),
    )
    return create_model("RiskRanking", drivers=(list[ranked], Field(description="Most important first.")))


def _action_choice_schema(action_ids: list[str]) -> type[BaseModel]:
    choice = create_model(
        "ActionChoice",
        id=(Literal[tuple(action_ids)], Field(description="An action id from the catalogue.")),
        reason=(str, Field(description="One sentence tying the action to this PAT's facts.")),
    )
    return create_model("ActionChoices", actions=(list[choice], Field(description="Two to five actions.")))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _request(state: AdvisorState) -> PatRequest:
    return PatRequest.model_validate(state["request"])


def _as_of(state: AdvisorState) -> datetime:
    return datetime.fromisoformat(state["as_of"])


def _request_text(request: PatRequest) -> str:
    return f"{request.title}\n{request.description}"


def _allowed_gates(state: AdvisorState) -> set[str]:
    return {step["gate"] for step in state["outlook"]["journey"]}


def _facts_json(state: AdvisorState) -> str:
    return json.dumps(state["facts"], indent=1, default=str)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def extract_signals(state: AdvisorState) -> dict[str, Any]:
    request = _request(state)
    result = pat_llm.invoke_structured(
        SignalExtraction,
        [
            (
                "system",
                "You read public appearance requests for a compliance approval workflow. "
                "Extract three lists, copying names exactly as written in the text:\n"
                "- securities: companies, banks or issuers (for example 'Acme Corp').\n"
                "- sovereign_entities: countries or governments (for example 'France').\n"
                "- sensitive_topics: subject phrases a reviewer would check, not company or country names.\n"
                "Never put a company in sensitive_topics. Do not infer names that are not written. "
                "Return empty lists when nothing applies.",
            ),
            ("human", _request_text(request)),
        ],
        "advisor.extract_signals",
    )
    return {"signals": result.model_dump() if result else {}}


def validate_signals(state: AdvisorState) -> dict[str, Any]:
    """Keep only extracted entities that appear in the user's own text."""
    request = _request(state)
    text = _request_text(request)
    raw = state.get("signals") or {}
    securities, rejected_securities = filter_entities(raw.get("securities", []), text)
    sovereigns, rejected_sovereigns = filter_entities(raw.get("sovereign_entities", []), text)
    topics, rejected_topics = filter_entities(raw.get("sensitive_topics", []), text)
    signals = ExtractedSignals(
        securities=securities,
        sovereign_entities=sovereigns,
        sensitive_topics=topics,
        rejected=rejected_securities + rejected_sovereigns + rejected_topics,
    )
    return {"signals": signals.model_dump(mode="json")}


def compute_outlook(state: AdvisorState, runtime: Runtime[PatContext]) -> dict[str, Any]:
    config = load_config()
    request = _request(state)
    as_of = _as_of(state)
    signals = ExtractedSignals.model_validate(state["signals"])
    # Entities the submitter entered plus validated ones found in the text.
    enriched = request.model_copy(update={
        "securities": sorted(set(request.securities) | set(signals.securities)),
        "sovereign_entities": sorted(set(request.sovereign_entities) | set(signals.sovereign_entities)),
    })
    ctx = runtime.context
    loads = current_loads(ctx.repo.list_pats("in_progress"), config)
    outlook = ctx.model.outlook(enriched, as_of, loads)
    similar = ctx.model.similar(enriched, as_of)
    candidates = candidate_risk_drivers(enriched, as_of, outlook, loads, similar, signals, config)
    facts = {
        "event": {
            "type": request.event_type,
            "region": request.region,
            "city": request.city,
            "localTime": format_local(request.event_time, request.region, config),
            "hoursToEvent": outlook.hours_to_event,
        },
        "outlook": {
            "probabilityOnTimePct": outlook.probability_pct,
            "hoursToCompleteP50": outlook.hours_to_complete_p50,
            "hoursToCompleteP90": outlook.hours_to_complete_p90,
            "estimatedCompletion": format_local(outlook.p50_eta, request.region, config),
            "confidence": outlook.confidence,
        },
        "journey": [
            {"gate": step.gate, "name": step.display_name, "expectedHoursP50": step.expected_hours_p50}
            for step in outlook.journey
        ],
        "similarPats": similar.model_dump(mode="json", by_alias=True, exclude={"examples"}),
        "riskDrivers": [driver.model_dump(include={"id", "label", "detail", "impact"}) for driver in candidates],
    }
    return {
        "outlook": outlook.model_dump(mode="json"),
        "similar": similar.model_dump(mode="json"),
        "candidates": [driver.model_dump(mode="json") for driver in candidates],
        "facts": facts,
    }


def assess_risk(state: AdvisorState) -> dict[str, Any]:
    """LLM ranks the code-computed risk drivers and explains each one."""
    candidates = [RiskDriver.model_validate(item) for item in state["candidates"]]
    if not candidates:
        return {"risk_drivers": []}
    impact_order = {"high": 0, "medium": 1, "low": 2}
    by_id = {driver.id: driver for driver in candidates}
    result = pat_llm.invoke_structured(
        _risk_ranking_schema(list(by_id)),
        [
            (
                "system",
                "You are a compliance workflow analyst. Rank the risk drivers by how much they "
                "threaten approval before the event. For each, write one sentence explaining why. "
                "Use only numbers and facts from the JSON. Do not invent numbers.",
            ),
            ("human", _facts_json(state)),
        ],
        "advisor.assess_risk",
    )
    allowed = allowed_numbers(state["facts"])
    gates = _allowed_gates(state)
    ranked: list[RiskDriver] = []
    for item in (result.drivers if result else []):
        driver = by_id.get(item.id)
        if driver is None or item.id in {seen.id for seen in ranked}:
            continue
        why = item.why.strip()
        driver = driver.model_copy(update={"why": why if why and not grounding_problems(why, allowed, gates) else None})
        ranked.append(driver)
    # Never hide a driver the LLM skipped; append them by impact.
    seen = {driver.id for driver in ranked}
    ranked += sorted(
        (driver for driver in candidates if driver.id not in seen),
        key=lambda driver: impact_order[driver.impact],
    )
    return {"risk_drivers": [driver.model_dump(mode="json") for driver in ranked]}


def _default_actions(state: AdvisorState, catalogue: dict[str, dict[str, Any]]) -> list[str]:
    request = _request(state)
    signals = state["signals"]
    ids = []
    if state["outlook"]["hours_to_event"] < load_config()["lateSubmissionHours"]:
        ids.append("submit_now")
    ids.append("verify_event_details")
    if request.securities or signals.get("securities") or signals.get("sovereign_entities"):
        ids.append("review_securities")
    ids.append("enable_monitoring")
    if any(step["gate"] == "CG" for step in state["outlook"]["journey"]) and state["outlook"]["probability"] < 0.9:
        ids.append("contact_cg_early")
    return [action_id for action_id in ids if action_id in catalogue]


def recommend_actions(state: AdvisorState) -> dict[str, Any]:
    """LLM picks actions from the catalogue; code checks and fills in reasons."""
    catalogue = {action["id"]: action for action in load_config()["advisorActions"]}
    facts = dict(state["facts"])
    facts["rankedDrivers"] = [driver["id"] for driver in state["risk_drivers"]]
    result = pat_llm.invoke_structured(
        _action_choice_schema(list(catalogue)),
        [
            (
                "system",
                "You advise a research analyst submitting a public appearance request. Choose the "
                "actions from the catalogue that best reduce this PAT's approval risk, most useful "
                "first. Give a one-sentence reason for each based on the facts. Do not invent numbers.\n\n"
                "Catalogue:\n" + json.dumps(list(catalogue.values()), indent=1),
            ),
            ("human", json.dumps(facts, indent=1, default=str)),
        ],
        "advisor.recommend_actions",
    )
    allowed = allowed_numbers(state["facts"])
    gates = _allowed_gates(state)
    chosen: list[RecommendedAction] = []
    picks = [(item.id, item.reason.strip()) for item in result.actions] if result else []
    if not picks:
        picks = [(action_id, "") for action_id in _default_actions(state, catalogue)]
    for action_id, reason in picks:
        if action_id in {action.id for action in chosen}:
            continue
        action = catalogue[action_id]
        if not reason or grounding_problems(reason, allowed, gates):
            reason = action["description"]
        chosen.append(RecommendedAction(id=action_id, phase=action["phase"], label=action["label"], reason=reason))
    return {"actions": [action.model_dump(mode="json") for action in chosen[:MAX_RECOMMENDED_ACTIONS]]}


def summarise(state: AdvisorState) -> dict[str, Any]:
    facts = dict(state["facts"])
    facts["topRiskDrivers"] = [
        {"label": driver["label"], "detail": driver["detail"]} for driver in state["risk_drivers"][:3]
    ]
    instructions = (
        "Write a short approval outlook for the analyst who is about to submit this public "
        "appearance request: how likely approval before the event is, the expected timing, which "
        "review is likely to take longest, and the main risk. Three or four plain sentences. "
        "Use only numbers that appear in the JSON, written the same way. Do not invent numbers or reviews."
    )
    feedback = state.get("summary_feedback")
    if feedback:
        instructions += f"\n\nYour previous summary was rejected: {feedback}. Fix this."
    result = pat_llm.invoke_structured(
        AdvisorSummary,
        [("system", instructions), ("human", json.dumps(facts, indent=1, default=str))],
        "advisor.summarise",
    )
    return {
        "summary": result.summary.strip() if result else "",
        "summary_attempts": state.get("summary_attempts", 0) + 1,
    }


def _template_summary(state: AdvisorState) -> str:
    config = load_config()
    outlook = Outlook.model_validate(state["outlook"])
    slowest = max(
        (step for step in outlook.journey if step.expected_hours_p50 is not None),
        key=lambda step: step.expected_hours_p50,
        default=None,
    )
    likelihood = (
        "likely" if outlook.probability >= config["healthBands"]["onTrack"]
        else "at risk of not being" if outlook.probability >= config["healthBands"]["atRisk"]
        else "unlikely to be"
    )
    parts = [
        f"This PAT is {likelihood} approved before the event: {outlook.probability_pct}% of simulated "
        f"approval journeys finish in time ({outlook.confidence} confidence).",
        f"Approval is expected in about {outlook.hours_to_complete_p50} hours; the event starts in "
        f"{outlook.hours_to_event} hours.",
    ]
    if slowest:
        parts.append(f"{slowest.display_name} review is expected to take longest.")
    if state["risk_drivers"]:
        parts.append(f"Main risk: {state['risk_drivers'][0]['detail']}")
    return " ".join(parts)


def validate_output(state: AdvisorState) -> dict[str, Any]:
    summary = state.get("summary", "")
    problems = (
        grounding_problems(summary, allowed_numbers(state["facts"]), _allowed_gates(state))
        if summary else ["empty summary"]
    )
    if not problems:
        return {"summary_source": "llm", "summary_feedback": None}
    if state.get("summary_attempts", 0) <= MAX_SUMMARY_RETRIES:
        return {"summary_feedback": "; ".join(problems)}
    return {"summary": _template_summary(state), "summary_source": "template", "summary_feedback": None}


def route_after_validation(state: AdvisorState) -> Literal["summarise", "respond"]:
    return "summarise" if state.get("summary_feedback") else "respond"


def _suggested_monitoring(state: AdvisorState) -> list[str]:
    config = load_config()
    outlook = Outlook.model_validate(state["outlook"])
    slowest = max(
        (step for step in outlook.journey if step.expected_hours_p50 is not None),
        key=lambda step: step.expected_hours_p50,
        default=None,
    )
    suggestions = []
    if slowest:
        suggestions.append(f"Notify me if {gate_display_name(slowest.gate, config)} review is delayed")
    suggestions.append("Notify me if the PAT becomes high risk")
    suggestions.append("Notify me if the PAT is still not approved 5 hours before the event")
    if outlook.probability < config["healthBands"]["onTrack"]:
        suggestions.append("Notify me if termination risk increases")
    return suggestions


def respond(state: AdvisorState) -> dict[str, Any]:
    response = AdvisorResponse(
        as_of=_as_of(state),
        outlook=Outlook.model_validate(state["outlook"]),
        similar=SimilarSummary.model_validate(state["similar"]),
        signals=ExtractedSignals.model_validate(state["signals"]),
        risk_drivers=[RiskDriver.model_validate(item) for item in state["risk_drivers"]],
        recommended_actions=[RecommendedAction.model_validate(item) for item in state["actions"]],
        summary=state["summary"],
        summary_source=state.get("summary_source", "template"),
        suggested_monitoring=_suggested_monitoring(state),
    )
    return {"response": response.model_dump(mode="json")}


builder = StateGraph(AdvisorState, context_schema=PatContext)
builder.add_node("extract_signals", extract_signals)
builder.add_node("validate_signals", validate_signals)
builder.add_node("compute_outlook", compute_outlook)
builder.add_node("assess_risk", assess_risk)
builder.add_node("recommend_actions", recommend_actions)
builder.add_node("summarise", summarise)
builder.add_node("validate_output", validate_output)
builder.add_node("respond", respond)
builder.add_edge(START, "extract_signals")
builder.add_edge("extract_signals", "validate_signals")
builder.add_edge("validate_signals", "compute_outlook")
builder.add_edge("compute_outlook", "assess_risk")
builder.add_edge("assess_risk", "recommend_actions")
builder.add_edge("recommend_actions", "summarise")
builder.add_edge("summarise", "validate_output")
builder.add_conditional_edges(
    "validate_output", route_after_validation, {"summarise": "summarise", "respond": "respond"}
)
builder.add_edge("respond", END)


def run_advisor(graph: Any, context: PatContext, request: PatRequest, as_of: datetime | None = None) -> AdvisorResponse:
    as_of = as_of or datetime.now(UTC)
    result = graph.invoke(
        {"request": request.model_dump(mode="json"), "as_of": as_of.isoformat()},
        context=context,
    )
    return AdvisorResponse.model_validate(result["response"])

