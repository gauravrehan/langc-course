"""Tools the Approval Intelligence Agent can call.

Each tool is bound to one PAT and one point in time, so the LLM calls them
without arguments and cannot look up other PATs. Results are plain dicts
computed in code.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from pat.clock import format_local, hours_between
from pat.config import gate_display_name
from pat.models import LoadBand, Outlook, Pat
from pat.predict import PredictionModel, coverage_info, queue_counts
from pat.repository import PatRepository


@dataclass
class ToolContext:
    pat: Pat
    as_of: datetime
    repo: PatRepository
    model: PredictionModel
    loads: dict[str, LoadBand]
    config: dict[str, Any]
    in_flight: list[Pat] = field(default_factory=list)
    _outlook: Outlook | None = None

    def outlook(self) -> Outlook:
        if self._outlook is None:
            self._outlook = self.model.outlook(self.pat, self.as_of, self.loads)
        return self._outlook


def get_gate_status(ctx: ToolContext) -> dict[str, Any]:
    gates = []
    for step in ctx.pat.gates:
        entry: dict[str, Any] = {
            "gate": step.gate,
            "name": gate_display_name(step.gate, ctx.config),
            "status": step.status,
        }
        if step.status == "pending" and step.started_at:
            entry["hoursPending"] = round(hours_between(step.started_at, ctx.as_of), 1)
        gates.append(entry)
    pending = ctx.pat.pending_gate()
    return {
        "blockingGate": pending.gate if pending else None,
        "gates": gates,
        "hoursToEvent": round(hours_between(ctx.as_of, ctx.pat.event_time), 1),
    }


def predict_remaining_time(ctx: ToolContext) -> dict[str, Any]:
    outlook = ctx.outlook()
    return {
        "probabilityOnTimePct": outlook.probability_pct,
        "terminationRiskPct": 100 - outlook.probability_pct,
        "hoursToCompleteP50": outlook.hours_to_complete_p50,
        "hoursToCompleteP90": outlook.hours_to_complete_p90,
        "hoursToEvent": outlook.hours_to_event,
        "confidence": outlook.confidence,
        "remainingGates": [
            {"gate": step.gate, "expectedHoursP50": step.expected_hours_p50}
            for step in outlook.journey
            if step.status != "approved"
        ],
    }


def get_reminder_history(ctx: ToolContext) -> dict[str, Any]:
    reminders = [
        {"gate": step.gate, "sentAt": format_local(moment, ctx.pat.region, ctx.config)}
        for step in ctx.pat.gates
        for moment in step.reminders
    ]
    return {"reminderCount": len(reminders), "reminders": reminders}


def get_similar_cases(ctx: ToolContext) -> dict[str, Any]:
    summary = ctx.model.similar(ctx.pat, ctx.as_of)
    return summary.model_dump(mode="json", by_alias=True, exclude={"examples"})


def get_group_coverage(ctx: ToolContext) -> dict[str, Any]:
    pending = ctx.pat.pending_gate()
    if not pending:
        return {"blockingGate": None}
    return coverage_info(pending.gate, ctx.pat, ctx.as_of, ctx.config)


def get_queue_status(ctx: ToolContext) -> dict[str, Any]:
    return {"pendingCounts": queue_counts(ctx.in_flight), "loads": ctx.loads}


def get_intervention_history(ctx: ToolContext) -> dict[str, Any]:
    records = ctx.repo.list_interventions(ctx.pat.id)[-5:]
    return {
        "count": len(records),
        "recent": [
            {
                "hoursAgo": round(hours_between(record.as_of, ctx.as_of), 1),
                "action": record.action,
                "outcome": record.outcome,
                "health": record.assessment.health,
            }
            for record in records
        ],
    }


TOOL_FUNCTIONS: dict[str, tuple[Callable[[ToolContext], dict[str, Any]], str]] = {
    "get_gate_status": (get_gate_status, "Status of each approval gate, the blocking gate and hours to the event."),
    "predict_remaining_time": (predict_remaining_time, "Predicted probability of finishing before the event and expected remaining time per gate."),
    "get_reminder_history": (get_reminder_history, "Workflow reminders already sent for this PAT."),
    "get_similar_cases": (get_similar_cases, "How similar historical PATs turned out."),
    "get_group_coverage": (get_group_coverage, "Whether the blocking approval group is working now and before the deadline, and its backup contact."),
    "get_queue_status": (get_queue_status, "How many PATs are waiting at each gate right now."),
    "get_intervention_history": (get_intervention_history, "Interventions the agent already made for this PAT."),
}


def build_tools(ctx: ToolContext) -> list[StructuredTool]:
    tools = []
    for name, (function, description) in TOOL_FUNCTIONS.items():
        def run(_function: Callable = function) -> str:
            return json.dumps(_function(ctx), default=str)

        tools.append(StructuredTool.from_function(func=run, name=name, description=description))
    return tools


def run_tool(name: str, ctx: ToolContext) -> dict[str, Any]:
    function, _ = TOOL_FUNCTIONS[name]
    return function(ctx)


def run_all_tools(ctx: ToolContext) -> dict[str, Any]:
    return {name: run_tool(name, ctx) for name in TOOL_FUNCTIONS}
