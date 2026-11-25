"""Approval Intelligence Agent: monitors one in-flight PAT and decides whether to intervene.

    load_state -> agent_reason (LLM + tools) -> assess_health (LLM)
      -> choose_intervention (LLM) -> policy_check
         -> record                                  (no action / suppressed)
         -> confirm_escalation (interrupt) -> classify_escalation_reply (LLM)
         -> draft_message (LLM) -> emit_notification -> record

One LangGraph thread per PAT (thread_id "pat:<id>"); each monitoring pass is
a new run on that thread. LLM nodes reason and write; code computes the facts,
applies policy and can override the LLM.
"""

import json
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, create_model
from typing_extensions import TypedDict

from pat import llm as pat_llm
from pat.clock import format_local, hours_between
from pat.config import gate_display_name, load_config, monitor_action
from pat.guards import allowed_numbers, grounding_problems, time_left_problems
from pat.models import (
    GateEvent,
    HealthAssessment,
    InterventionRecord,
    Notification,
    Pat,
)
from pat.predict import current_loads
from pat.runtime import PatContext
from pat.tools import TOOL_FUNCTIONS, ToolContext, build_tools, run_all_tools, run_tool

logger = pat_llm.logger

SEVERITY = {"on_track": 0, "at_risk": 1, "critical": 2}


class MonitorState(TypedDict, total=False):
    pat_id: str
    as_of: str
    closed: bool
    message: str
    facts: dict[str, Any]
    tool_results: dict[str, Any]
    tool_calls: list[str]
    agent_notes: str
    reasoning_mode: str
    assessment: dict[str, Any]
    proposed: dict[str, Any]
    decision: dict[str, Any]
    confirmation_reply: str | None
    unclear_count: int
    notification: dict[str, Any] | None
    intervention: dict[str, Any] | None


# --------------------------------------------------------------------------
# LLM output schemas
# --------------------------------------------------------------------------


class HealthJudgement(BaseModel):
    health: Literal["on_track", "at_risk", "critical"]
    explanation: str = Field(description="Two sentences: why the PAT is in this state and what is blocking it.")


class EscalationReply(BaseModel):
    decision: Literal["confirmed", "rejected", "unclear"]


class DraftMessage(BaseModel):
    title: str = Field(description="Short headline, under 10 words.")
    body: str = Field(description="Two or three sentences for the recipient.")
    next_steps: list[str] = Field(default_factory=list, description="Up to three concrete next steps.")
    approval_group_note: str | None = Field(
        default=None, description="One sentence the recipient can forward to the approval group, or null."
    )


def _intervention_schema(action_ids: list[str]) -> type[BaseModel]:
    return create_model(
        "InterventionChoice",
        action=(Literal[tuple(action_ids)], Field(description="An action id from the catalogue.")),
        urgency=(Literal["low", "medium", "high"], ...),
        reason=(str, Field(description="One sentence explaining the choice from the facts.")),
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _as_of(state: MonitorState) -> datetime:
    return datetime.fromisoformat(state["as_of"])


def _tool_context(state: MonitorState, runtime: Runtime[PatContext], pat: Pat | None = None) -> ToolContext:
    ctx = runtime.context
    config = load_config()
    in_flight = ctx.repo.list_pats("in_progress")
    pat = pat or ctx.repo.get_pat(state["pat_id"])
    return ToolContext(
        pat=pat,
        as_of=_as_of(state),
        repo=ctx.repo,
        model=ctx.model,
        loads=current_loads(in_flight, config),
        config=config,
        in_flight=in_flight,
    )


def _pat(state: MonitorState, runtime: Runtime[PatContext]) -> Pat:
    pat = runtime.context.repo.get_pat(state["pat_id"])
    if pat is None:
        raise ValueError(f"PAT {state['pat_id']} not found")
    return pat


def _evidence(state: MonitorState) -> dict[str, Any]:
    """Everything the LLM may cite: core facts plus tool results."""
    return {"facts": state["facts"], "toolResults": state.get("tool_results", {})}


def _allowed_gates(state: MonitorState) -> set[str]:
    return {gate["gate"] for gate in state["facts"]["gateStatus"]["gates"]}


def _grounded(text: str, state: MonitorState, *extra: Any) -> bool:
    allowed = allowed_numbers(_evidence(state), *extra)
    problems = grounding_problems(text, allowed, _allowed_gates(state))
    problems += time_left_problems(text, state["facts"]["prediction"]["hoursToEvent"])
    if problems:
        logger.info("Rejected LLM text for %s: %s", state["pat_id"], "; ".join(problems))
    return not problems


def _computed_health(facts: dict[str, Any], config: dict[str, Any]) -> str:
    probability = facts["prediction"]["probabilityOnTimePct"] / 100
    coverage = facts.get("coverage") or {}
    if coverage.get("blockingGate", True) and coverage.get("coverageBeforeDeadline") is False:
        return "critical"
    if probability >= config["healthBands"]["onTrack"]:
        return "on_track"
    if probability >= config["healthBands"]["atRisk"]:
        return "at_risk"
    return "critical"


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def load_state(state: MonitorState, runtime: Runtime[PatContext]) -> dict[str, Any]:
    repo = runtime.context.repo
    config = load_config()
    as_of = _as_of(state)
    pat = repo.get_pat(state["pat_id"])
    if pat is None:
        return {"closed": True, "message": f"PAT {state['pat_id']} was not found."}
    if pat.status != "in_progress":
        return {"closed": True, "message": f"PAT {pat.id} is already {pat.status}; monitoring has stopped."}
    if as_of >= pat.event_time:
        pending = pat.pending_gate()
        pat.status = "terminated"
        pat.completed_at = pat.event_time
        repo.save_pat(pat)
        repo.add_event(GateEvent(
            pat_id=pat.id, type="terminated", gate=pending.gate if pending else None,
            timestamp=pat.event_time, note="Approval not received before the event.",
        ))
        return {
            "closed": True,
            "message": f"PAT {pat.id} was terminated: approval was not received before the event.",
        }

    ctx = _tool_context(state, runtime, pat)
    gate_status = run_tool("get_gate_status", ctx)
    prediction = run_tool("predict_remaining_time", ctx)
    coverage = run_tool("get_group_coverage", ctx)
    reminders = run_tool("get_reminder_history", ctx)
    facts = {
        "keyFacts": _key_facts(gate_status, prediction, coverage, reminders, config),
        "pat": {
            "id": pat.id,
            "title": pat.title,
            "eventType": pat.event_type,
            "region": pat.region,
            "city": pat.city,
            "eventLocalTime": format_local(pat.event_time, pat.region, config),
            "nowLocalTime": format_local(as_of, pat.region, config),
            "submitter": pat.submitter,
        },
        "gateStatus": gate_status,
        "prediction": prediction,
        "coverage": coverage,
        "reminders": reminders,
    }
    return {"closed": False, "facts": facts}


def _key_facts(
    gate_status: dict[str, Any],
    prediction: dict[str, Any],
    coverage: dict[str, Any],
    reminders: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    """Plain sentences so small models don't confuse 'hours pending' with 'hours left'."""
    lines = [
        f"Time left before the event (the approval deadline): {prediction['hoursToEvent']} hours.",
        f"Chance all remaining gates finish before the event: {prediction['probabilityOnTimePct']}%.",
    ]
    blocking = gate_status["blockingGate"]
    if blocking:
        name = gate_display_name(blocking, config)
        pending = next((gate.get("hoursPending") for gate in gate_status["gates"] if gate["gate"] == blocking), None)
        if pending is not None:
            lines.append(f"{name} ({blocking}) has already been pending for {pending} hours.")
        if coverage.get("coveredNow"):
            lines.append(f"The {name} group is working now.")
        else:
            lines.append(f"The {name} group is off shift until {coverage.get('nextCoverageStart')}.")
    lines.append(f"Workflow reminders already sent: {reminders['reminderCount']}.")
    return lines


def route_after_load(state: MonitorState) -> Literal["agent_reason", "__end__"]:
    return END if state.get("closed") else "agent_reason"


AGENT_SYSTEM_PROMPT = (
    "You are the PAT Approval Intelligence Agent. A PAT (public appearance request) must pass "
    "every approval gate before its event or it is terminated. Investigate this PAT using the "
    "tools: check which gate is blocking it, how likely it is to finish in time, whether the "
    "blocking group is working before the deadline, and what has already been done. Call only "
    "the tools you need. When you have enough, reply with three or four short notes on what "
    "you found. Do not invent numbers."
)


def agent_reason(state: MonitorState, runtime: Runtime[PatContext]) -> dict[str, Any]:
    """ReAct loop: the LLM chooses which tools to call. Falls back to running all tools."""
    ctx = _tool_context(state, runtime)
    max_steps = load_config()["agent"]["maxToolSteps"]
    facts = state["facts"]
    messages: list[Any] = [
        SystemMessage(AGENT_SYSTEM_PROMPT),
        HumanMessage(
            f"PAT {facts['pat']['id']}: {facts['pat']['title']}. Event at "
            f"{facts['pat']['eventLocalTime']}; it is now {facts['pat']['nowLocalTime']}. "
            f"Blocking gate: {facts['gateStatus']['blockingGate']}."
        ),
    ]
    tool_results: dict[str, Any] = {}
    calls: list[str] = []
    notes = ""
    try:
        model = pat_llm.bind_tools(build_tools(ctx))
        for _ in range(max_steps):
            reply = model.invoke(messages)
            messages.append(reply)
            tool_calls = getattr(reply, "tool_calls", None) or []
            if not tool_calls:
                notes = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
                break
            for call in tool_calls:
                name = call.get("name")
                if name in TOOL_FUNCTIONS:
                    result = run_tool(name, ctx)
                    tool_results[name] = result
                    calls.append(name)
                    content = json.dumps(result, default=str)
                else:
                    content = json.dumps({"error": f"Unknown tool {name}"})
                messages.append(ToolMessage(content=content, tool_call_id=call.get("id") or name))
    except Exception as error:  # noqa: BLE001 - any tool-calling failure means fall back
        logger.warning("Agent tool loop failed for %s: %s", state["pat_id"], error)
        calls = []

    if not calls:
        logger.info("Agent for %s fell back to running all tools", state["pat_id"])
        return {
            "tool_results": run_all_tools(ctx),
            "tool_calls": [],
            "agent_notes": "",
            "reasoning_mode": "fallback",
        }
    logger.info("Agent for %s called tools: %s", state["pat_id"], calls)
    return {
        "tool_results": tool_results,
        "tool_calls": calls,
        "agent_notes": notes[:1500],
        "reasoning_mode": "tools",
    }


def _template_explanation(state: MonitorState, health: str) -> str:
    config = load_config()
    facts = state["facts"]
    blocking = facts["gateStatus"]["blockingGate"]
    prediction = facts["prediction"]
    parts = [
        f"The event starts in {prediction['hoursToEvent']} hours and {prediction['probabilityOnTimePct']}% "
        f"of simulated journeys finish in time."
    ]
    if blocking:
        name = gate_display_name(blocking, config)
        pending = next((gate.get("hoursPending") for gate in facts["gateStatus"]["gates"] if gate["gate"] == blocking), None)
        parts.append(f"{name} review is pending" + (f" for {pending} hours." if pending is not None else "."))
        coverage = facts.get("coverage") or {}
        if coverage and not coverage.get("coveredNow"):
            parts.append(f"The {name} group is outside working hours until {coverage['nextCoverageStart']}.")
    if health == "on_track":
        parts.append("The PAT is progressing normally.")
    return " ".join(parts)


def assess_health(state: MonitorState) -> dict[str, Any]:
    """LLM judges health and explains it; code may override to keep it consistent with the numbers."""
    config = load_config()
    facts = state["facts"]
    computed = _computed_health(facts, config)
    result = pat_llm.invoke_structured(
        HealthJudgement,
        [
            (
                "system",
                "You assess the health of a PAT in an approval workflow. on_track: likely to be "
                "approved in time. at_risk: may miss the deadline without follow-up. critical: "
                "likely to be terminated unless someone acts now. Explain in two sentences using "
                "only the facts given, including which gate is blocking. Do not invent numbers.",
            ),
            (
                "human",
                json.dumps(
                    {**_evidence(state), "agentNotes": state.get("agent_notes", "")},
                    indent=1, default=str,
                ),
            ),
        ],
        "monitor.assess_health",
    )
    llm_health = result.health if result else None
    # The LLM may be one level more cautious than the statistics, never less.
    if llm_health is not None and 0 <= SEVERITY[llm_health] - SEVERITY[computed] <= 1:
        health = llm_health
    else:
        health = computed
    explanation = result.explanation.strip() if result else ""
    source = "llm"
    if not explanation or not _grounded(explanation, state):
        explanation = _template_explanation(state, health)
        source = "template"
    assessment = HealthAssessment(
        health=health,
        computed_health=computed,
        llm_health=llm_health,
        overridden=llm_health is not None and llm_health != health,
        termination_risk_pct=facts["prediction"]["terminationRiskPct"],
        blocking_gate=facts["gateStatus"]["blockingGate"],
        time_remaining_hours=facts["prediction"]["hoursToEvent"],
        explanation=explanation,
        explanation_source=source,
    )
    return {"assessment": assessment.model_dump(mode="json")}


def _default_intervention(health: str, interventions: dict[str, Any]) -> tuple[str, str]:
    escalated = any(item["action"] == "escalate_to_backup" for item in interventions.get("recent", []))
    if health == "on_track":
        return "none", "low"
    if health == "at_risk":
        return "notify_submitter", "medium"
    return ("nudge_approval_group" if escalated else "escalate_to_backup"), "high"


def choose_intervention(state: MonitorState, runtime: Runtime[PatContext]) -> dict[str, Any]:
    config = load_config()
    actions = config["monitorActions"]
    assessment = state["assessment"]
    history = state.get("tool_results", {}).get("get_intervention_history") or run_tool(
        "get_intervention_history", _tool_context(state, runtime)
    )
    result = pat_llm.invoke_structured(
        _intervention_schema([action["id"] for action in actions]),
        [
            (
                "system",
                "You decide the next intervention for a PAT in an approval workflow. Choose one "
                "action from the catalogue. Prefer 'none' when the PAT is on track or a recent "
                "intervention already covers it, to avoid alert fatigue. Escalation is only for "
                "critical PATs. Explain your choice in one sentence using the facts.\n\nCatalogue:\n"
                + json.dumps(actions, indent=1),
            ),
            (
                "human",
                json.dumps(
                    {"assessment": assessment, **_evidence(state), "interventionHistory": history},
                    indent=1, default=str,
                ),
            ),
        ],
        "monitor.choose_intervention",
    )
    if result:
        action, urgency, reason = result.action, result.urgency, result.reason.strip()
        if not reason or not _grounded(reason, state, assessment, history):
            reason = f"Chosen for a PAT that is {assessment['health'].replace('_', ' ')}."
    else:
        action, urgency = _default_intervention(assessment["health"], history)
        reason = f"Default policy for a PAT that is {assessment['health'].replace('_', ' ')}."
    return {"proposed": {"action": action, "urgency": urgency, "reason": reason}}


def policy_check(state: MonitorState, runtime: Runtime[PatContext]) -> dict[str, Any]:
    """Code rules that every proposed action must pass."""
    config = load_config()
    as_of = _as_of(state)
    health = state["assessment"]["health"]
    coverage = state["facts"].get("coverage") or {}
    proposed = state["proposed"]
    action = proposed["action"]
    notes: list[str] = []
    history = runtime.context.repo.list_interventions(state["pat_id"])
    sent = [record for record in history if record.outcome == "sent"]

    if health == "on_track" and action != "none":
        notes.append(f"{action} skipped: the PAT is on track, so no alert is sent.")
        action = "none"
    if health == "critical" and action == "none":
        notes.append("Critical PATs always get at least a submitter notification.")
        action = "notify_submitter"
    if action == "escalate_to_backup":
        rule = monitor_action(action, config)
        escalations = [record for record in sent if record.action == action]
        if health != rule.get("requiresHealth", "critical"):
            notes.append("Escalation needs a critical PAT; nudging the approval group instead.")
            action = "nudge_approval_group"
        elif len(escalations) >= rule.get("maxPerPat", 99):
            notes.append("Escalation limit reached for this PAT; nudging the approval group instead.")
            action = "nudge_approval_group"

    floor = {"critical": "high", "at_risk": "medium"}.get(health, "low")
    urgency = max(proposed["urgency"], floor, key=["low", "medium", "high"].index)

    outcome = "pending"
    if action == "none":
        outcome = "no_action"
    else:
        rule = monitor_action(action, config)
        if rule.get("respectsCoverage") and not coverage.get("coveredNow", True) and health != "critical":
            notes.append(f"{action} deferred: the approval group is outside working hours.")
            action, outcome = "none", "suppressed"
        else:
            cooldown = rule.get("cooldownHours", 0)
            recent = [
                record for record in sent
                if record.action == action and hours_between(record.as_of, as_of) < cooldown
            ]
            if recent:
                notes.append(f"{action} suppressed: already sent within the last {cooldown} hours.")
                action, outcome = "none", "suppressed"

    needs_confirmation = outcome == "pending" and bool(monitor_action(action, config).get("requiresConfirmation"))
    return {
        "decision": {
            "action": action,
            "urgency": urgency,
            "reason": proposed["reason"],
            "policy_notes": notes,
            "outcome": outcome,
            "needs_confirmation": needs_confirmation,
        },
        "unclear_count": state.get("unclear_count", 0),
    }


def route_after_policy(state: MonitorState) -> Literal["record", "confirm_escalation", "draft_message"]:
    decision = state["decision"]
    if decision["outcome"] != "pending":
        return "record"
    if decision["needs_confirmation"]:
        return "confirm_escalation"
    return "draft_message"


def confirm_escalation(state: MonitorState) -> dict[str, Any]:
    config = load_config()
    assessment = state["assessment"]
    gate = assessment["blocking_gate"]
    backup = config["gates"][gate]["backup"] if gate else "the backup contacts"
    prefix = "I didn't understand that. " if state.get("unclear_count") else ""
    reply = interrupt({
        "kind": "confirm_escalation",
        "message": (
            f"{prefix}PAT {state['pat_id']} is critical: {assessment['explanation']} "
            f"Escalate to {backup}? Reply yes or no."
        ),
        "action": "escalate_to_backup",
        "parameters": {"recipient": backup, "blockingGate": gate, "reason": state["decision"]["reason"]},
    })
    message = reply.get("message", "") if isinstance(reply, dict) else str(reply)
    return {"confirmation_reply": message}


def classify_escalation_reply(state: MonitorState) -> dict[str, Any]:
    result = pat_llm.invoke_structured(
        EscalationReply,
        [
            (
                "system",
                "A person was asked whether to escalate a PAT to backup approvers. Classify their "
                "reply: confirmed (yes, go ahead), rejected (no, don't), or unclear.",
            ),
            ("human", state.get("confirmation_reply") or ""),
        ],
        "monitor.classify_escalation_reply",
    )
    decision = result.decision if result else "unclear"
    update: dict[str, Any] = {}
    if decision == "unclear":
        update["unclear_count"] = state.get("unclear_count", 0) + 1
        if update["unclear_count"] > load_config()["agent"]["maxUnclearConfirmations"]:
            decision = "rejected"
    if decision == "rejected":
        update["decision"] = {
            **state["decision"],
            "outcome": "cancelled",
            "policy_notes": state["decision"]["policy_notes"] + ["Escalation was not confirmed."],
        }
    update["confirmation_reply"] = decision
    return update


def route_after_reply(state: MonitorState) -> Literal["draft_message", "record", "confirm_escalation"]:
    decision = state.get("confirmation_reply")
    if decision == "confirmed":
        return "draft_message"
    if decision == "rejected":
        return "record"
    return "confirm_escalation"


def _recipient(action: str, gate: str | None, submitter: str, config: dict[str, Any]) -> tuple[str, str]:
    audience = monitor_action(action, config).get("audience", "submitter")
    if audience == "approval_group" and gate:
        return audience, config["gates"][gate]["group"]
    if audience == "backup" and gate:
        return audience, config["gates"][gate]["backup"]
    return "submitter", submitter


def _template_message(state: MonitorState) -> DraftMessage:
    config = load_config()
    assessment = state["assessment"]
    gate = assessment["blocking_gate"]
    name = gate_display_name(gate, config) if gate else "approval"
    title = {
        "critical": "Action required: PAT at risk of termination",
        "at_risk": "PAT at risk: approval may be late",
    }.get(assessment["health"], "PAT update")
    steps = []
    if gate:
        steps.append(f"Contact {config['gates'][gate]['group']} for approval now.")
        steps.append(f"If not approved soon, escalate via {config['gates'][gate]['backup']}.")
    steps.append("Monitor the PAT and follow up within the next hour.")
    return DraftMessage(
        title=title,
        body=f"{assessment['explanation']} {name} review is the blocker.",
        next_steps=steps,
        approval_group_note=(
            f"PAT {state['pat_id']} is waiting on {name} review and the event is close; please review it as a priority."
            if gate else None
        ),
    )


def draft_message(state: MonitorState) -> dict[str, Any]:
    config = load_config()
    decision = state["decision"]
    assessment = state["assessment"]
    audience, recipient = _recipient(
        decision["action"], assessment["blocking_gate"], state["facts"]["pat"]["submitter"], config
    )
    result = pat_llm.invoke_structured(
        DraftMessage,
        [
            (
                "system",
                f"Write a short, actionable notification for {audience.replace('_', ' ')} '{recipient}' "
                "about a PAT (public appearance request) in an approval workflow. Be specific about "
                "which gate is blocking, how much time is left and what to do. Use only numbers from "
                "the facts. Do not invent numbers, people or approvals.",
            ),
            (
                "human",
                json.dumps(
                    {"action": decision, "assessment": assessment, **_evidence(state)},
                    indent=1, default=str,
                ),
            ),
        ],
        "monitor.draft_message",
    )
    source = "llm"
    if result is None:
        draft, source = _template_message(state), "template"
    else:
        text = " ".join([result.title, result.body, *result.next_steps, result.approval_group_note or ""])
        if not result.title.strip() or not result.body.strip() or not _grounded(text, state, assessment):
            draft, source = _template_message(state), "template"
        else:
            draft = result
    notification = Notification(
        pat_id=state["pat_id"],
        audience=audience,
        recipient=recipient,
        urgency=decision["urgency"],
        title=draft.title.strip(),
        body=draft.body.strip(),
        approval_group_note=draft.approval_group_note,
        health=assessment["health"],
        termination_risk_pct=assessment["termination_risk_pct"],
        time_remaining_hours=assessment["time_remaining_hours"],
        blocking_gate=assessment["blocking_gate"],
        recommended_steps=draft.next_steps[:3],
        reminder_count=state["facts"]["reminders"]["reminderCount"],
        links={"pat": f"/pat/{state['pat_id']}/health"},
        source=source,
    )
    return {"notification": notification.model_dump(mode="json")}


def emit_notification(state: MonitorState, runtime: Runtime[PatContext]) -> dict[str, Any]:
    notification = Notification.model_validate(state["notification"])
    runtime.context.notifier.send(notification)
    return {"decision": {**state["decision"], "outcome": "sent"}}


def record(state: MonitorState, runtime: Runtime[PatContext]) -> dict[str, Any]:
    decision = state["decision"]
    as_of = _as_of(state)
    record_ = InterventionRecord(
        id=str(uuid4()),
        pat_id=state["pat_id"],
        created_at=datetime.now(UTC),
        as_of=as_of,
        reasoning_mode=state.get("reasoning_mode", "fallback"),
        tool_calls=state.get("tool_calls", []),
        agent_notes=state.get("agent_notes", ""),
        assessment=HealthAssessment.model_validate(state["assessment"]),
        proposed_action=state["proposed"]["action"],
        action=decision["action"],
        urgency=decision["urgency"],
        reason=decision["reason"],
        policy_notes=decision["policy_notes"],
        outcome=decision["outcome"],
        notification=(
            Notification.model_validate(state["notification"])
            if decision["outcome"] == "sent" and state.get("notification") else None
        ),
    )
    runtime.context.repo.add_intervention(record_)
    health = record_.assessment.health.replace("_", " ")
    if record_.outcome == "sent":
        message = f"PAT {record_.pat_id} is {health}; sent {record_.action} to {record_.notification.recipient}."
    elif record_.outcome == "cancelled":
        message = f"PAT {record_.pat_id} is {health}; escalation was not confirmed, so nothing was sent."
    elif record_.outcome == "suppressed":
        message = f"PAT {record_.pat_id} is {health}; {'; '.join(record_.policy_notes)}"
    else:
        message = f"PAT {record_.pat_id} is {health}; no action needed."
    return {"intervention": record_.model_dump(mode="json"), "message": message}


builder = StateGraph(MonitorState, context_schema=PatContext)
builder.add_node("load_state", load_state)
builder.add_node("agent_reason", agent_reason)
builder.add_node("assess_health", assess_health)
builder.add_node("choose_intervention", choose_intervention)
builder.add_node("policy_check", policy_check)
builder.add_node("confirm_escalation", confirm_escalation)
builder.add_node("classify_escalation_reply", classify_escalation_reply)
builder.add_node("draft_message", draft_message)
builder.add_node("emit_notification", emit_notification)
builder.add_node("record", record)
builder.add_edge(START, "load_state")
builder.add_conditional_edges("load_state", route_after_load, {"agent_reason": "agent_reason", END: END})
builder.add_edge("agent_reason", "assess_health")
builder.add_edge("assess_health", "choose_intervention")
builder.add_edge("choose_intervention", "policy_check")
builder.add_conditional_edges(
    "policy_check",
    route_after_policy,
    {"record": "record", "confirm_escalation": "confirm_escalation", "draft_message": "draft_message"},
)
builder.add_edge("confirm_escalation", "classify_escalation_reply")
builder.add_conditional_edges(
    "classify_escalation_reply",
    route_after_reply,
    {"draft_message": "draft_message", "record": "record", "confirm_escalation": "confirm_escalation"},
)
builder.add_edge("draft_message", "emit_notification")
builder.add_edge("emit_notification", "record")
builder.add_edge("record", END)


def thread_config(pat_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"pat:{pat_id}"}}


def start_tick(graph: Any, context: PatContext, pat_id: str, as_of: datetime | None = None) -> dict[str, Any]:
    """Run one monitoring pass. Per-run fields are reset because the thread is reused."""
    as_of = as_of or datetime.now(UTC)
    return graph.invoke(
        {
            "pat_id": pat_id,
            "as_of": as_of.isoformat(),
            "closed": False,
            "message": "",
            "facts": {},
            "tool_results": {},
            "tool_calls": [],
            "agent_notes": "",
            "reasoning_mode": "fallback",
            "assessment": {},
            "proposed": {},
            "decision": {},
            "confirmation_reply": None,
            "unclear_count": 0,
            "notification": None,
            "intervention": None,
        },
        thread_config(pat_id),
        context=context,
    )


def resume_tick(graph: Any, context: PatContext, pat_id: str, message: str) -> dict[str, Any]:
    return graph.invoke(Command(resume={"message": message}), thread_config(pat_id), context=context)


def pending_interrupt(graph: Any, pat_id: str) -> dict[str, Any] | None:
    snapshot = graph.get_state(thread_config(pat_id))
    for task in snapshot.tasks or ():
        if task.interrupts:
            return task.interrupts[0].value
    return None
