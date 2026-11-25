"""Shared fakes and fixtures for the PAT tests. No LLM, Ollama or MongoDB needed."""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Callable

# Keep any configured MongoDB out of the tests.
os.environ["MONGODB_URI"] = ""

from langchain_core.messages import AIMessage  # noqa: E402

from pat import llm as pat_llm  # noqa: E402
from pat.config import load_config  # noqa: E402
from pat.data import demo_scenarios, generate_history, generate_in_flight  # noqa: E402
from pat.notify import RecordingNotifier  # noqa: E402
from pat.predict import PredictionModel  # noqa: E402
from pat.repository import InMemoryPatRepository  # noqa: E402
from pat.runtime import PatContext  # noqa: E402

pat_llm.logger.setLevel(logging.WARNING)

# Tuesday 10:00 in New York, inside approver coverage for AMER.
NOW = datetime(2026, 11, 24, 15, 0, tzinfo=UTC)

_HISTORY_CACHE: dict[int, list] = {}


def history(count: int = 3000) -> list:
    if count not in _HISTORY_CACHE:
        _HISTORY_CACHE[count] = generate_history(count, now=NOW)
    return _HISTORY_CACHE[count]


def make_context(extra_in_flight: int = 9) -> tuple[PatContext, dict[str, Any]]:
    """In-memory context with the three demo scenarios (critical, at risk, on track) first."""
    repo = InMemoryPatRepository()
    repo.replace_history(history())
    in_flight = generate_in_flight(3 + extra_in_flight, now=NOW, seed=7)
    repo.replace_in_flight(in_flight)
    context = PatContext(
        repo=repo,
        model=PredictionModel(repo.load_history(), load_config()),
        notifier=RecordingNotifier(),
    )
    names = {"critical": in_flight[0], "at_risk": in_flight[1], "on_track": in_flight[2]}
    return context, names


def human_json(messages: list[tuple[str, str]]) -> dict[str, Any]:
    return json.loads(messages[-1][1])


class _Structured:
    def __init__(self, fake: "FakeLLM", schema: type) -> None:
        self.fake = fake
        self.schema = schema

    def invoke(self, messages: list[tuple[str, str]]) -> Any:
        name = self.schema.__name__
        self.fake.calls.append(name)
        handler = self.fake.handlers.get(name)
        if handler is None:
            raise RuntimeError(f"No fake response for {name}")
        result = handler(self.schema, messages)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, dict):
            return self.schema.model_validate(result)
        return result


class _ToolModel:
    def __init__(self, fake: "FakeLLM") -> None:
        self.fake = fake

    def invoke(self, _messages: list[Any]) -> AIMessage:
        if not self.fake.tool_script:
            return AIMessage(content="No further notes.")
        step = self.fake.tool_script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class FakeLLM:
    """Answers structured calls by schema name; plays a scripted tool-calling conversation."""

    def __init__(self, handlers: dict[str, Callable] | None = None, tool_script: list | None = None) -> None:
        self.handlers = {**default_handlers(), **(handlers or {})}
        self.tool_script = list(tool_script or [])
        self.calls: list[str] = []

    def with_structured_output(self, schema: type, **_kwargs: Any) -> _Structured:
        return _Structured(self, schema)

    def bind_tools(self, _tools: list[Any]) -> _ToolModel:
        return _ToolModel(self)


def tool_calls(*names: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {}, "id": f"call-{index}"} for index, name in enumerate(names)],
    )


def default_tool_script() -> list:
    return [
        tool_calls("get_gate_status", "predict_remaining_time"),
        tool_calls("get_group_coverage", "get_intervention_history"),
        AIMessage(content="Control Group is the blocker and the event is close."),
    ]


# --------------------------------------------------------------------------
# Default structured responses
# --------------------------------------------------------------------------


def _signals(_schema: type, messages: list[tuple[str, str]]) -> dict[str, Any]:
    text = messages[-1][1]
    securities = [name for name in ("Contoso Bank", "Northwind Energy") if name in text]
    # "Initech" never appears in the request text, so validation must drop it.
    return {"securities": securities + ["Initech"], "sovereign_entities": [], "sensitive_topics": []}


def _risk_ranking(_schema: type, messages: list[tuple[str, str]]) -> dict[str, Any]:
    drivers = human_json(messages)["riskDrivers"]
    return {"drivers": [{"id": driver["id"], "why": "This matters for timing."} for driver in reversed(drivers)]}


def _actions(_schema: type, _messages: list[tuple[str, str]]) -> dict[str, Any]:
    return {"actions": [
        {"id": "submit_now", "reason": "Submitting now gives reviewers more time."},
        {"id": "enable_monitoring", "reason": "Monitoring flags delays early."},
    ]}


def _summary(_schema: type, messages: list[tuple[str, str]]) -> dict[str, Any]:
    facts = human_json(messages)
    pct = facts["outlook"]["probabilityOnTimePct"]
    return {"summary": f"This PAT has a {pct}% chance of approval before the event."}


def _health(schema: type, messages: list[tuple[str, str]]) -> dict[str, Any]:
    data = human_json(messages)
    pct = data["facts"]["prediction"]["probabilityOnTimePct"]
    health = "on_track" if pct >= 80 else "at_risk" if pct >= 50 else "critical"
    return {"health": health, "explanation": f"{pct}% of simulated journeys finish in time."}


def _intervention(_schema: type, messages: list[tuple[str, str]]) -> dict[str, Any]:
    health = human_json(messages)["assessment"]["health"]
    action = {"on_track": "none", "at_risk": "notify_submitter", "critical": "escalate_to_backup"}[health]
    return {"action": action, "urgency": "high", "reason": "Based on the current health."}


def _draft(_schema: type, messages: list[tuple[str, str]]) -> dict[str, Any]:
    data = human_json(messages)
    gate = data["assessment"]["blocking_gate"]
    return {
        "title": "PAT needs attention",
        "body": f"{gate} review is still pending.",
        "next_steps": ["Contact the approval group now."],
        "approval_group_note": None,
    }


def _escalation_reply(_schema: type, messages: list[tuple[str, str]]) -> dict[str, Any]:
    text = messages[-1][1].casefold()
    if text.startswith("yes"):
        return {"decision": "confirmed"}
    if text.startswith("no"):
        return {"decision": "rejected"}
    return {"decision": "unclear"}


def default_handlers() -> dict[str, Callable]:
    return {
        "SignalExtraction": _signals,
        "RiskRanking": _risk_ranking,
        "ActionChoices": _actions,
        "AdvisorSummary": _summary,
        "HealthJudgement": _health,
        "InterventionChoice": _intervention,
        "DraftMessage": _draft,
        "EscalationReply": _escalation_reply,
    }


def demo_names(now: datetime = NOW) -> list[str]:
    return [pat.title for pat in demo_scenarios(now)]
