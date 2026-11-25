"""Prediction engine. Code only: every number the LLM sees comes from here.

Gate durations are learned from history and grouped by gate, event type,
time bucket and queue load. Within a group, samples are matched on how many
hours before the event the gate started, because reviewers behave very
differently when an event is close. A Monte Carlo run over the remaining
gates gives the probability of finishing before the event and the ETA.
"""

import bisect
import math
import random
import statistics
import zlib
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from pat.clock import (
    format_local,
    hours_between,
    in_coverage,
    next_coverage_start,
    time_bucket,
)
from pat.config import gate_display_name, gate_path
from pat.models import (
    ExtractedSignals,
    GateStep,
    JourneyStep,
    LoadBand,
    Outlook,
    Pat,
    PatRequest,
    RiskDriver,
    SimilarPat,
    SimilarSummary,
)

# Fewest matching samples before a lookup widens its window or falls back to a coarser group.
MIN_WINDOW_SAMPLES = 20
WINDOW_RATIO = 1.35
MAX_WIDENINGS = 4


def _rng_for(*parts: Any) -> random.Random:
    return random.Random(zlib.crc32("|".join(str(part) for part in parts).encode()))


class _Segment:
    """Gate durations sorted by hours-to-event at gate start."""

    def __init__(self) -> None:
        self.hours_to_event: list[float] = []
        self.durations: list[float] = []
        # True when the gate never finished because the PAT was terminated.
        self.censored: list[bool] = []

    def finalise(self) -> None:
        rows = sorted(zip(self.hours_to_event, self.durations, self.censored))
        self.hours_to_event = [row[0] for row in rows]
        self.durations = [row[1] for row in rows]
        self.censored = [row[2] for row in rows]

    def window(self, hours_to_event: float) -> tuple[int, int] | None:
        ratio = WINDOW_RATIO
        for _ in range(MAX_WIDENINGS):
            low = bisect.bisect_left(self.hours_to_event, hours_to_event / ratio)
            high = bisect.bisect_right(self.hours_to_event, hours_to_event * ratio)
            if high - low >= MIN_WINDOW_SAMPLES:
                return low, high
            ratio *= 1.5
        return None


class PredictionModel:
    def __init__(self, history: list[Pat], config: dict[str, Any]) -> None:
        self.config = config
        self.history = history
        self.segments: dict[tuple, _Segment] = defaultdict(_Segment)
        for pat in history:
            for step in pat.gates:
                if not step.started_at:
                    continue
                hours_to_event = max(hours_between(step.started_at, pat.event_time), 0.01)
                if step.status == "approved" and step.completed_at:
                    duration = hours_between(step.started_at, step.completed_at)
                    censored = False
                elif pat.status == "terminated" and step.status == "pending":
                    # Ran until the event without finishing.
                    duration = hours_to_event
                    censored = True
                else:
                    continue
                bucket = time_bucket(step.started_at, pat.region, config)
                keys = self._keys(step.gate, pat.event_type, bucket, step.queue_load or "medium", _complexity(pat))
                for key in keys:
                    segment = self.segments[key]
                    segment.hours_to_event.append(hours_to_event)
                    segment.durations.append(duration)
                    segment.censored.append(censored)
        for segment in self.segments.values():
            segment.finalise()

    @staticmethod
    def _keys(gate: str, event_type: str, bucket: str, load: str, complexity: str) -> list[tuple]:
        """Most to least specific grouping."""
        return [
            (gate, event_type, bucket, load, complexity),
            (gate, bucket, load, complexity),
            (gate, load, complexity),
            (gate, load),
            (gate,),
        ]

    def _lookup(
        self, gate: str, event_type: str, bucket: str, load: str, complexity: str, hours_to_event: float
    ) -> tuple[_Segment, int, int]:
        for key in self._keys(gate, event_type, bucket, load, complexity):
            segment = self.segments.get(key)
            if segment is None:
                continue
            window = segment.window(max(hours_to_event, 0.01))
            if window:
                return segment, window[0], window[1]
        segment = self.segments.get((gate,))
        if segment is None or not segment.durations:
            raise ValueError(f"No history for gate {gate}")
        return segment, 0, len(segment.durations)

    # ------------------------------------------------------------------
    # Monte Carlo outlook
    # ------------------------------------------------------------------

    def outlook(
        self,
        pat: Pat | PatRequest,
        as_of: datetime,
        loads: dict[str, LoadBand],
        gates: list[GateStep] | None = None,
        samples: int | None = None,
    ) -> Outlook:
        """Probability that all remaining gates finish before the event."""
        samples = samples or self.config["monteCarloSamples"]
        if gates is None:
            gates = getattr(pat, "gates", None) or [
                GateStep(gate=gate) for gate in gate_path(pat.event_type, self.config)
            ]
        remaining = [step for step in gates if step.status != "approved"]
        rng = _rng_for(getattr(pat, "id", pat.title), as_of.isoformat())
        finish_hours: list[float] = []
        per_gate: dict[str, list[float]] = defaultdict(list)
        supports: list[int] = []
        weak_pending_estimate = False
        complexity = _complexity(pat)

        for _ in range(samples):
            moment = as_of
            for step in remaining:
                load = loads.get(step.gate, "medium")
                if step.status == "pending" and step.started_at:
                    start = step.started_at
                    elapsed = hours_between(start, as_of)
                else:
                    start = moment
                    elapsed = 0.0
                hours_to_event = hours_between(start, pat.event_time)
                bucket = time_bucket(start, pat.region, self.config)
                segment, low, high = self._lookup(step.gate, pat.event_type, bucket, load, complexity, hours_to_event)
                supports.append(high - low)
                duration = None
                for _attempt in range(30):
                    candidate = _draw(segment, rng.randrange(low, high), hours_to_event)
                    if candidate > elapsed:
                        duration = candidate - elapsed
                        break
                if duration is None:
                    # Stalled: the gate has run longer than almost all similar
                    # cases. Estimate a fresh review at today's urgency plus a
                    # penalty of half the overrun.
                    weak_pending_estimate = True
                    now_to_event = hours_between(as_of, pat.event_time)
                    now_bucket = time_bucket(as_of, pat.region, self.config)
                    fresh, f_low, f_high = self._lookup(
                        step.gate, pat.event_type, now_bucket, load, complexity, now_to_event
                    )
                    overrun = elapsed - _window_quantile(segment, low, high, 0.95)
                    duration = _draw(fresh, rng.randrange(f_low, f_high), now_to_event) + 0.5 * max(overrun, 0)
                per_gate[step.gate].append(duration)
                moment = moment + timedelta(hours=duration)
            finish_hours.append(hours_between(as_of, moment))

        hours_to_event = hours_between(as_of, pat.event_time)
        on_time = sum(1 for hours in finish_hours if hours <= hours_to_event)
        probability = on_time / samples
        p50 = _quantile(finish_hours, 0.5)
        p90 = _quantile(finish_hours, 0.9)
        support = int(statistics.mean(supports)) if supports else 0
        if weak_pending_estimate or support < 25:
            confidence = "low"
        elif support < 60:
            confidence = "medium"
        else:
            confidence = "high"

        journey = []
        for step in gates:
            durations = per_gate.get(step.gate)
            journey.append(
                JourneyStep(
                    gate=step.gate,
                    display_name=gate_display_name(step.gate, self.config),
                    status=step.status,
                    expected_hours_p50=round(_quantile(durations, 0.5), 1) if durations else None,
                    expected_hours_p90=round(_quantile(durations, 0.9), 1) if durations else None,
                )
            )
        return Outlook(
            probability=round(probability, 3),
            probability_pct=round(probability * 100),
            p50_eta=as_of + timedelta(hours=p50),
            p90_eta=as_of + timedelta(hours=p90),
            hours_to_complete_p50=round(p50, 1),
            hours_to_complete_p90=round(p90, 1),
            hours_to_event=round(hours_to_event, 1),
            confidence=confidence,
            samples_used=support,
            journey=journey,
        )

    # ------------------------------------------------------------------
    # Similar historical PATs
    # ------------------------------------------------------------------

    def similar(self, pat: Pat | PatRequest, as_of: datetime, k: int | None = None) -> SimilarSummary:
        k = k or self.config["similarCases"]
        submitted = getattr(pat, "submitted_at", None) or as_of
        target = _features(pat, submitted, self.config)
        scored = sorted(
            self.history,
            key=lambda other: (_distance(target, _features(other, other.submitted_at, self.config)), other.id),
        )[:k]
        last_minute_hours = self.config["lastMinuteHours"]
        approved = [other for other in scored if other.status == "approved"]
        approval_hours = [hours_between(other.submitted_at, other.completed_at) for other in approved]
        last_minute = [
            other for other in approved
            if hours_between(other.completed_at, other.event_time) <= last_minute_hours
        ]
        gate_totals: dict[str, list[float]] = defaultdict(list)
        for other in scored:
            for step in other.gates:
                if step.started_at and step.completed_at:
                    gate_totals[step.gate].append(hours_between(step.started_at, step.completed_at))
        slowest = max(gate_totals, key=lambda gate: statistics.median(gate_totals[gate])) if gate_totals else None
        examples = [
            SimilarPat(
                id=other.id,
                title=other.title,
                event_type=other.event_type,
                region=other.region,
                lead_hours=round(hours_between(other.submitted_at, other.event_time), 1),
                status=other.status,
                approval_hours=round(hours_between(other.submitted_at, other.completed_at), 1)
                if other.status == "approved" else None,
            )
            for other in scored[:5]
        ]
        return SimilarSummary(
            count=len(scored),
            on_time_rate_pct=round(100 * len(approved) / max(len(scored), 1)),
            last_minute_rate_pct=round(100 * len(last_minute) / max(len(approved), 1)),
            median_approval_hours=round(statistics.median(approval_hours), 1) if approval_hours else None,
            slowest_gate=slowest,
            examples=examples,
        )


def _complexity(pat: Pat | PatRequest) -> str:
    """Sovereign entities or many securities make reviews longer."""
    if pat.sovereign_entities or len(pat.securities) >= 3:
        return "complex"
    return "standard"


def _draw(segment: _Segment, index: int, hours_to_event: float) -> float:
    """Duration for one historical sample, adapted to this PAT's hours-to-event.

    A sample that finished close to its event (inside the last quarter of the
    time it had) was deadline-driven: the reviewer picked it up near the event.
    For those, keep the slack before the event rather than the raw duration.
    """
    sample_hours_to_event = segment.hours_to_event[index]
    duration = segment.durations[index]
    if segment.censored[index]:
        # The historical gate missed its event; this one misses too.
        return hours_to_event + 1.0
    slack = sample_hours_to_event - duration
    if 0 <= slack < 0.25 * sample_hours_to_event:
        return max(hours_to_event - slack, 0.1)
    return duration


_WINDOW_QUANTILES: dict[tuple, float] = {}


def _window_quantile(segment: _Segment, low: int, high: int, q: float) -> float:
    key = (id(segment), low, high, q)
    if key not in _WINDOW_QUANTILES:
        _WINDOW_QUANTILES[key] = _quantile(segment.durations[low:high], q)
    return _WINDOW_QUANTILES[key]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _features(pat: Pat | PatRequest, submitted: datetime, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": pat.event_type,
        "region": pat.region,
        "log_lead": math.log1p(max(hours_between(submitted, pat.event_time), 0)),
        "sovereign": bool(pat.sovereign_entities),
        "securities": min(len(pat.securities), 4),
        "bucket": time_bucket(pat.event_time, pat.region, config),
    }


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return (
        2.0 * (a["event_type"] != b["event_type"])
        + 1.0 * (a["region"] != b["region"])
        + 1.5 * abs(a["log_lead"] - b["log_lead"])
        + 1.0 * (a["sovereign"] != b["sovereign"])
        + 0.25 * abs(a["securities"] - b["securities"])
        + 0.5 * (a["bucket"] != b["bucket"])
    )


# --------------------------------------------------------------------------
# Live workflow facts
# --------------------------------------------------------------------------


def current_loads(in_flight: list[Pat], config: dict[str, Any]) -> dict[str, LoadBand]:
    """Queue load per gate from the number of PATs currently pending there."""
    counts: dict[str, int] = defaultdict(int)
    for pat in in_flight:
        step = pat.pending_gate()
        if step:
            counts[step.gate] += 1
    thresholds = config["queueLoad"]
    loads: dict[str, LoadBand] = {}
    for gate in config["gates"]:
        count = counts.get(gate, 0)
        if count >= thresholds["highAt"]:
            loads[gate] = "high"
        elif count >= thresholds["mediumAt"]:
            loads[gate] = "medium"
        else:
            loads[gate] = "low"
    return loads


def queue_counts(in_flight: list[Pat]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for pat in in_flight:
        step = pat.pending_gate()
        if step:
            counts[step.gate] += 1
    return dict(counts)


def coverage_info(gate: str, pat: Pat | PatRequest, as_of: datetime, config: dict[str, Any]) -> dict[str, Any]:
    """Whether the approval group is working now and before the deadline."""
    gate_config = config["gates"][gate]
    covered_now = in_coverage(as_of, pat.region, config)
    next_start = next_coverage_start(as_of, pat.region, config)
    return {
        "gate": gate,
        "group": gate_config["group"],
        "backup": gate_config["backup"],
        "coveredNow": covered_now,
        "nextCoverageStart": format_local(next_start, pat.region, config),
        "hoursUntilCoverage": round(max(hours_between(as_of, next_start), 0), 1),
        "coverageBeforeDeadline": covered_now or next_start < pat.event_time,
    }


def time_remaining_hours(pat: Pat | PatRequest, as_of: datetime) -> float:
    return round(hours_between(as_of, pat.event_time), 2)


# --------------------------------------------------------------------------
# Risk driver candidates (facts the LLM may rank and explain)
# --------------------------------------------------------------------------


def candidate_risk_drivers(
    pat: Pat | PatRequest,
    as_of: datetime,
    outlook: Outlook,
    loads: dict[str, LoadBand],
    similar: SimilarSummary,
    signals: ExtractedSignals | None,
    config: dict[str, Any],
) -> list[RiskDriver]:
    drivers: list[RiskDriver] = []
    hours_to_event = hours_between(as_of, pat.event_time)
    if hours_to_event < config["lateSubmissionHours"]:
        drivers.append(RiskDriver(
            id="short_lead_time",
            label="Short lead time",
            detail=f"The event starts in {hours_to_event:.1f} hours.",
            impact="high" if hours_to_event < 24 else "medium",
        ))
    path = [step.gate for step in getattr(pat, "gates", [])] or gate_path(pat.event_type, config)
    if "CG" in path and loads.get("CG") in {"medium", "high"}:
        drivers.append(RiskDriver(
            id="control_group_queue",
            label="Control Group queue volume",
            detail=f"Control Group queue load is currently {loads['CG']}.",
            impact="high" if loads["CG"] == "high" else "medium",
        ))
    event_bucket = time_bucket(pat.event_time, pat.region, config)
    if event_bucket != "business":
        drivers.append(RiskDriver(
            id="outside_coverage_event",
            label="Event outside approver hours",
            detail=f"The event is at {format_local(pat.event_time, pat.region, config)}, outside approval group coverage ({event_bucket.replace('_', ' ')}).",
            impact="medium",
        ))
    if pat.event_type == "media":
        drivers.append(RiskDriver(
            id="media_event",
            label="Media event",
            detail="Media events need an additional Corporate Communications review.",
            impact="medium",
        ))
    sovereigns = set(pat.sovereign_entities) | set(signals.sovereign_entities if signals else [])
    if sovereigns:
        drivers.append(RiskDriver(
            id="sovereign_entities",
            label="Sovereign entities",
            detail=f"{len(sovereigns)} sovereign entities mentioned: {', '.join(sorted(sovereigns))}.",
            impact="medium",
        ))
    securities = set(pat.securities) | set(signals.securities if signals else [])
    if len(securities) >= 3:
        drivers.append(RiskDriver(
            id="many_securities",
            label="Many securities or issuers",
            detail=f"{len(securities)} securities or issuers referenced.",
            impact="medium",
        ))
    if signals and signals.sensitive_topics:
        drivers.append(RiskDriver(
            id="sensitive_topics",
            label="Sensitive topics",
            detail="Topics that may need closer review: " + ", ".join(signals.sensitive_topics) + ".",
            impact="low",
        ))
    if pat.cross_region:
        drivers.append(RiskDriver(
            id="cross_region",
            label="Cross-region event",
            detail="Approvers and the event are in different regions.",
            impact="low",
        ))
    if similar.on_time_rate_pct < 95:
        drivers.append(RiskDriver(
            id="similar_cases_late",
            label="Similar PATs often struggled",
            detail=f"{similar.on_time_rate_pct}% of {similar.count} similar PATs were approved in time; {similar.last_minute_rate_pct}% of those approvals came in the last {config['lastMinuteHours']} hours.",
            impact="medium",
        ))
    elif similar.last_minute_rate_pct >= 25:
        drivers.append(RiskDriver(
            id="similar_cases_last_minute",
            label="Similar PATs approved late",
            detail=f"{similar.last_minute_rate_pct}% of similar approved PATs were approved in the last {config['lastMinuteHours']} hours.",
            impact="low",
        ))
    if outlook.confidence == "low":
        drivers.append(RiskDriver(
            id="low_confidence",
            label="Limited comparable history",
            detail="Few comparable historical cases, so the estimate is less certain.",
            impact="low",
        ))
    return drivers
