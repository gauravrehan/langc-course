"""Calendar helpers: approver coverage hours, time buckets and business-hour arithmetic.

Approval groups work in the PAT's region during the configured coverage window.
"""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pat.config import region_timezone

TimeBucket = Literal["business", "after_hours", "weekend"]
UrgencyBucket = Literal["<12h", "12-36h", ">36h"]


def _local(moment: datetime, tz: ZoneInfo) -> datetime:
    return moment.astimezone(tz)


def in_coverage(moment: datetime, region: str, config: dict[str, Any]) -> bool:
    coverage = config["coverage"]
    local = _local(moment, region_timezone(region, config))
    return (
        local.weekday() in coverage["workdays"]
        and coverage["startHour"] <= local.hour < coverage["endHour"]
    )


def time_bucket(moment: datetime, region: str, config: dict[str, Any]) -> TimeBucket:
    local = _local(moment, region_timezone(region, config))
    if local.weekday() not in config["coverage"]["workdays"]:
        return "weekend"
    if in_coverage(moment, region, config):
        return "business"
    return "after_hours"


def urgency_bucket(hours_to_event: float) -> UrgencyBucket:
    if hours_to_event < 12:
        return "<12h"
    if hours_to_event < 36:
        return "12-36h"
    return ">36h"


def next_coverage_boundary(moment: datetime, region: str, config: dict[str, Any]) -> datetime:
    """Next moment the coverage state flips (start or end of the working window)."""
    coverage = config["coverage"]
    tz = region_timezone(region, config)
    local = _local(moment, tz)
    if in_coverage(moment, region, config):
        end = local.replace(hour=coverage["endHour"], minute=0, second=0, microsecond=0)
        return end.astimezone(UTC)
    day = local
    if local.weekday() in coverage["workdays"] and local.hour < coverage["startHour"]:
        start = local.replace(hour=coverage["startHour"], minute=0, second=0, microsecond=0)
        return start.astimezone(UTC)
    for _ in range(8):
        day = (day + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if day.weekday() in coverage["workdays"]:
            start = day.replace(hour=coverage["startHour"])
            # Rebuild through the zone so DST offsets are correct.
            start = datetime(start.year, start.month, start.day, start.hour, tzinfo=tz)
            return start.astimezone(UTC)
    raise ValueError("Coverage config has no workdays")


def next_coverage_start(moment: datetime, region: str, config: dict[str, Any]) -> datetime:
    if in_coverage(moment, region, config):
        return moment
    return next_coverage_boundary(moment, region, config)


def advance_work(
    start: datetime,
    effort_hours: float,
    region: str,
    config: dict[str, Any],
    off_hours_rate: float,
) -> datetime:
    """Finish time for `effort_hours` of work starting at `start`.

    Work proceeds at full speed during coverage and at `off_hours_rate` outside it.
    """
    moment = start
    remaining = effort_hours
    for _ in range(1000):
        if remaining <= 1e-9:
            return moment
        covered = in_coverage(moment, region, config)
        boundary = next_coverage_boundary(moment, region, config)
        rate = 1.0 if covered else off_hours_rate
        span = (boundary - moment).total_seconds() / 3600
        if rate > 0 and span * rate >= remaining:
            return moment + timedelta(hours=remaining / rate)
        remaining -= span * rate
        moment = boundary
    raise RuntimeError("advance_work did not converge")


def hours_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 3600


def format_local(moment: datetime, region: str, config: dict[str, Any]) -> str:
    local = _local(moment, region_timezone(region, config))
    return local.strftime("%a %d %b %Y %I:%M %p %Z")
