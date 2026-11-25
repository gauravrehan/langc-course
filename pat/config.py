"""Loads pat_config.json. Re-read on each call so config edits apply without a restart."""

import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

CONFIG_PATH = Path(__file__).with_name("pat_config.json")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def gate_path(event_type: str, config: dict[str, Any]) -> list[str]:
    paths = config["gatePaths"]
    return list(paths.get(event_type, paths["default"]))


def region_timezone(region: str, config: dict[str, Any]) -> ZoneInfo:
    return ZoneInfo(config["regions"][region]["timezone"])


def gate_display_name(gate: str, config: dict[str, Any]) -> str:
    return config["gates"][gate]["displayName"]


def monitor_action(action_id: str, config: dict[str, Any]) -> dict[str, Any]:
    for action in config["monitorActions"]:
        if action["id"] == action_id:
            return action
    raise KeyError(action_id)
