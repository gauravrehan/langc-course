"""Checks applied to LLM output before it is used.

The LLM may only repeat numbers, gates and entities that appear in the facts
it was given (or, for extraction, in the user's own text).
"""

import json
import re
from typing import Any

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_GATE = re.compile(r"\b(RM|SA|CG|CC)\b")
_WORD = re.compile(r"[a-z0-9]+")
# Numbers the LLM may always use, e.g. "1 gate left".
_ALWAYS_ALLOWED = {0.0, 1.0}


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(text.replace(",", ""))


def allowed_numbers(*facts: Any) -> set[float]:
    """Every number in the facts, plus its rounded forms."""
    allowed = set(_ALWAYS_ALLOWED)
    for fact in facts:
        text = fact if isinstance(fact, str) else json.dumps(fact, default=str)
        for raw in _numbers(text):
            value = float(raw)
            allowed.update({value, float(round(value)), round(value, 1)})
    return allowed


def ungrounded_numbers(text: str, allowed: set[float]) -> list[str]:
    bad = []
    for raw in _numbers(text):
        value = float(raw)
        if not any(abs(value - known) <= 0.051 for known in allowed):
            bad.append(raw)
    return bad


def ungrounded_gates(text: str, allowed_gates: set[str]) -> list[str]:
    return sorted({gate for gate in _GATE.findall(text) if gate not in allowed_gates})


def grounding_problems(text: str, allowed: set[float], allowed_gates: set[str]) -> list[str]:
    problems = []
    numbers = ungrounded_numbers(text, allowed)
    if numbers:
        problems.append("numbers not in the facts: " + ", ".join(numbers))
    gates = ungrounded_gates(text, allowed_gates)
    if gates:
        problems.append("gates not in this PAT's journey: " + ", ".join(gates))
    return problems


_TIME_LEFT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:more\s+)?hours?\s+(?:left|remaining|remain|until|before|to go)",
    re.IGNORECASE,
)


def time_left_problems(text: str, hours_to_event: float) -> list[str]:
    """Any 'N hours left/remaining/until ...' phrase must state the real time left."""
    bad = []
    for raw in _TIME_LEFT.findall(text.replace(",", "")):
        if abs(float(raw) - hours_to_event) > 0.6:
            bad.append(f"says {raw} hours left but {hours_to_event} hours remain")
    return bad


def entity_in_text(entity: str, text: str) -> bool:
    """True when the entity's words appear in order in the text (case-insensitive)."""
    wanted = _WORD.findall(entity.casefold())
    if not wanted:
        return False
    words = _WORD.findall(text.casefold())
    for start in range(len(words) - len(wanted) + 1):
        if words[start:start + len(wanted)] == wanted:
            return True
    return False


def filter_entities(entities: list[str], text: str) -> tuple[list[str], list[str]]:
    kept, rejected = [], []
    for entity in entities:
        entity = entity.strip()
        if not entity:
            continue
        target = kept if entity_in_text(entity, text) else rejected
        if entity not in target:
            target.append(entity)
    return kept, rejected
