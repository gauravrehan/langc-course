"""Synthetic PAT data. No real PAT records are used anywhere in the prototype.

The generator bakes in the patterns from the proposal so the prediction engine
and the agent have something realistic to work with:

- Control Group (CG) is the slowest gate and slows down further under load.
- Weekends and after-hours slow approvals, because work mostly happens in
  coverage hours.
- Media events need an extra gate (CC); sovereign entities and many
  securities make CG and SA reviews longer.
- About a fifth of PATs are submitted late, and reviewers speed up as the
  event gets close, which produces last-minute approvals and a small number
  of terminations.
"""

import math
import random
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from pat.clock import advance_work, hours_between, urgency_bucket
from pat.config import gate_path, load_config, region_timezone
from pat.models import GateStep, Pat

SEED = 42

# Median effort per gate in coverage hours, before multipliers.
BASE_EFFORT_HOURS = {"RM": 1.5, "SA": 2.0, "CG": 4.0, "CC": 1.5}
EFFORT_SIGMA = 0.6
LOAD_MULTIPLIER = {"low": 1.0, "medium": 1.4, "high": 1.9}
LOAD_WEIGHTS = {
    "RM": [0.55, 0.35, 0.10],
    "SA": [0.55, 0.35, 0.10],
    "CG": [0.35, 0.40, 0.25],
    "CC": [0.60, 0.30, 0.10],
}
# Reviewers expedite as the event approaches: (effort multiplier, off-hours work rate).
URGENCY_BEHAVIOUR = {"<12h": (0.1, 1.0), "12-36h": (0.35, 0.6), ">36h": (1.0, 0.06)}
# Reviewers often defer PATs whose event is far away and pick them up close to
# the event. Probability of deferring, by gate and queue load.
DEFER_PROBABILITY = {
    "RM": {"low": 0.02, "medium": 0.04, "high": 0.08},
    "SA": {"low": 0.03, "medium": 0.06, "high": 0.10},
    "CG": {"low": 0.20, "medium": 0.35, "high": 0.55},
    "CC": {"low": 0.10, "medium": 0.20, "high": 0.30},
}
# A deferred gate is picked up this many hours before the event.
DEFER_PICKUP_HOURS = (2.0, 5.5)
# Sometimes a PAT is missed in the queue: no last-minute rush, slow progress.
# This is the main cause of terminations in the proposal's analysis.
STALL_PROBABILITY = {"RM": 0.0015, "SA": 0.0015, "CG": 0.006, "CC": 0.003}
STALL_EFFORT_MULTIPLIER = 6.0
STALL_OFF_HOURS_RATE = 0.02

EVENT_TYPES = ["media", "conference", "client", "internal"]
EVENT_TYPE_WEIGHTS = [0.25, 0.30, 0.30, 0.15]
REGIONS = ["AMER", "EMEA", "APAC"]
REGION_WEIGHTS = [0.5, 0.3, 0.2]
SPEAKER_ROLES = ["analyst", "strategist", "economist"]

TOPICS = [
    "energy transition", "interest rate outlook", "emerging market debt",
    "semiconductor supply chains", "consumer credit", "bank capital rules",
    "commodity prices", "inflation expectations", "AI infrastructure spending",
    "real estate valuations", "sovereign credit ratings", "currency volatility",
]
# Fictional issuers so no real company appears in generated data.
ISSUERS = [
    "Northwind Energy", "Contoso Bank", "Fabrikam Semiconductors",
    "Tailspin Airlines", "Litware Retail", "Adatum Pharma", "Woodgrove Financial",
    "Proseware Mining", "Wingtip Logistics", "Lucerne Utilities",
    "Coho Shipping", "Alpine Ski Resorts", "Blue Yonder Telecom",
]
SOVEREIGNS = [
    "Brazil", "Mexico", "India", "Indonesia", "Turkey", "South Africa",
    "Argentina", "Egypt", "Japan", "Germany", "Italy", "Saudi Arabia",
]
VENUES = {
    "media": ["a television interview", "a podcast recording", "a financial news panel"],
    "conference": ["an industry conference", "an investor summit", "a policy forum"],
    "client": ["a client lunch", "a client webinar", "an institutional client meeting"],
    "internal": ["an internal sales call", "a desk briefing", "an internal town hall"],
}


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return median * math.exp(rng.gauss(0, sigma))


def _event_content(rng: random.Random, event_type: str) -> dict[str, Any]:
    topic = rng.choice(TOPICS)
    securities = rng.sample(ISSUERS, k=rng.choices([0, 1, 2, 3, 5], [0.3, 0.25, 0.2, 0.15, 0.1])[0])
    sovereigns = rng.sample(SOVEREIGNS, k=rng.choices([0, 1, 2], [0.75, 0.18, 0.07])[0])
    venue = rng.choice(VENUES[event_type])
    parts = [f"Speaking at {venue} about {topic}."]
    if securities:
        parts.append("Will reference " + ", ".join(securities) + ".")
    if sovereigns:
        parts.append("Discussion includes the outlook for " + " and ".join(sovereigns) + ".")
    return {
        "title": f"{topic.capitalize()} - {venue.split(' ', 1)[1]}",
        "description": " ".join(parts),
        "securities": securities,
        "sovereign_entities": sovereigns,
    }


def _effort_hours(rng: random.Random, gate: str, load: str, pat: dict[str, Any]) -> float:
    median = BASE_EFFORT_HOURS[gate] * LOAD_MULTIPLIER[load]
    if gate == "CG":
        if pat["event_type"] == "media":
            median *= 1.3
        median *= 1 + 0.3 * len(pat["sovereign_entities"])
    if gate in {"SA", "CG"} and len(pat["securities"]) >= 3:
        median *= 1.3
    if pat["cross_region"]:
        median *= 1.2
    return _lognormal(rng, median, EFFORT_SIGMA)


def simulate_gates(
    rng: random.Random,
    pat: dict[str, Any],
    start: datetime,
    config: dict[str, Any],
    stop_at: datetime | None = None,
) -> tuple[list[GateStep], str, datetime | None]:
    """Run a PAT through its gates from `start`.

    Returns the gate steps, the final status and the completion time. With
    `stop_at`, the state is cut off at that moment (for in-flight PATs).
    """
    steps: list[GateStep] = []
    moment = start
    event_time = pat["event_time"]
    path = gate_path(pat["event_type"], config)
    for index, gate in enumerate(path):
        load = rng.choices(["low", "medium", "high"], LOAD_WEIGHTS[gate])[0]
        work_start = moment
        if (
            hours_between(moment, event_time) > 36
            and rng.random() < DEFER_PROBABILITY[gate][load]
        ):
            work_start = event_time - timedelta(hours=rng.uniform(*DEFER_PICKUP_HOURS))
        effort_factor, off_rate = URGENCY_BEHAVIOUR[
            urgency_bucket(hours_between(work_start, event_time))
        ]
        if rng.random() < STALL_PROBABILITY[gate]:
            work_start = moment
            effort_factor, off_rate = STALL_EFFORT_MULTIPLIER, STALL_OFF_HOURS_RATE
        effort = _effort_hours(rng, gate, load, pat) * effort_factor
        finish = advance_work(work_start, effort, pat["region"], config, off_rate)
        step = GateStep(gate=gate, status="pending", started_at=moment, queue_load=load)
        steps.append(step)

        if stop_at is not None and finish > stop_at:
            # Still in this gate at the cut-off time; later gates are waiting.
            step.reminders = _reminders(moment, stop_at)
            steps.extend(GateStep(gate=later) for later in path[index + 1:])
            return steps, "in_progress", None
        if finish > event_time:
            steps.extend(GateStep(gate=later) for later in path[index + 1:])
            step.reminders = _reminders(moment, event_time)
            return steps, "terminated", event_time
        step.status = "approved"
        step.completed_at = finish
        step.reminders = _reminders(moment, finish)
        moment = finish
    return steps, "approved", moment


def _reminders(start: datetime, end: datetime) -> list[datetime]:
    """Workflow sends a reminder every 24 hours a gate stays pending."""
    reminders = []
    moment = start + timedelta(hours=24)
    while moment < end:
        reminders.append(moment)
        moment += timedelta(hours=24)
    return reminders


def _event_time(rng: random.Random, base: datetime, region: str, config: dict[str, Any]) -> datetime:
    tz = region_timezone(region, config)
    local = base.astimezone(tz)
    roll = rng.random()
    if roll < 0.10:
        # Weekend event.
        days_to_saturday = (5 - local.weekday()) % 7
        local = local + timedelta(days=days_to_saturday + rng.choice([0, 1]))
        hour = rng.randint(9, 20)
    elif roll < 0.27:
        hour = rng.randint(18, 22)
    else:
        hour = rng.randint(8, 17)
    local = datetime(local.year, local.month, local.day, hour, rng.choice([0, 15, 30, 45]), tzinfo=tz)
    return local.astimezone(UTC)


def _lead_hours(rng: random.Random) -> float:
    if rng.random() < 0.20:
        return rng.uniform(4, 30)
    return max(30.0, _lognormal(rng, 144, 0.7))


def _base_pat(rng: random.Random, config: dict[str, Any], pat_id: str, event_time: datetime, region: str) -> dict[str, Any]:
    event_type = rng.choices(EVENT_TYPES, EVENT_TYPE_WEIGHTS)[0]
    content = _event_content(rng, event_type)
    return {
        "id": pat_id,
        "event_type": event_type,
        "region": region,
        "city": rng.choice(config["regions"][region]["cities"]),
        "event_time": event_time,
        "speaker_role": rng.choice(SPEAKER_ROLES),
        "submitter": f"analyst{rng.randint(1, 60):02d}",
        "cross_region": rng.random() < 0.1,
        **content,
    }


def generate_history(
    count: int = 5000,
    now: datetime | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
) -> list[Pat]:
    """Historical PATs with events in the year before `now`."""
    config = config or load_config()
    now = now or datetime.now(UTC)
    rng = random.Random(seed)
    history: list[Pat] = []
    for index in range(count):
        region = rng.choices(REGIONS, REGION_WEIGHTS)[0]
        base = now - timedelta(days=rng.uniform(7, 365))
        event_time = _event_time(rng, base, region, config)
        pat = _base_pat(rng, config, f"H-{index:05d}", event_time, region)
        submitted_at = event_time - timedelta(hours=_lead_hours(rng))
        gates, status, completed_at = simulate_gates(rng, pat, submitted_at, config)
        history.append(
            Pat(
                **pat,
                submitted_at=submitted_at,
                gates=gates,
                status=status,
                completed_at=completed_at,
            )
        )
    return history


def new_in_flight_id() -> str:
    return "PAT-" + secrets.token_hex(3).upper()


def _scenario(
    config: dict[str, Any],
    now: datetime,
    *,
    title: str,
    description: str,
    event_type: str,
    region: str,
    city: str,
    hours_to_event: float,
    submitted_hours_ago: float,
    approved: list[tuple[str, float]],
    pending_since_hours: float,
    securities: list[str] | None = None,
    sovereigns: list[str] | None = None,
) -> Pat:
    """A hand-built in-flight PAT. `approved` holds (gate, hours ago) pairs."""
    path = gate_path(event_type, config)
    gates = []
    for gate in path:
        done = next((ago for code, ago in approved if code == gate), None)
        if done is not None:
            gates.append(GateStep(gate=gate, status="approved", completed_at=now - timedelta(hours=done)))
        elif not any(step.status == "pending" for step in gates):
            started = now - timedelta(hours=pending_since_hours)
            gates.append(GateStep(gate=gate, status="pending", started_at=started, reminders=_reminders(started, now)))
        else:
            gates.append(GateStep(gate=gate))
    previous_end = now - timedelta(hours=submitted_hours_ago)
    for step in gates:
        if step.status == "approved":
            step.started_at = previous_end
            previous_end = step.completed_at
    return Pat(
        id=new_in_flight_id(),
        title=title,
        description=description,
        event_type=event_type,
        region=region,
        city=city,
        event_time=now + timedelta(hours=hours_to_event),
        submitter="analyst07",
        securities=securities or [],
        sovereign_entities=sovereigns or [],
        submitted_at=now - timedelta(hours=submitted_hours_ago),
        gates=gates,
    )


def demo_scenarios(now: datetime, config: dict[str, Any] | None = None) -> list[Pat]:
    """Three in-flight PATs that reliably show critical, at-risk and on-track states."""
    config = config or load_config()
    return [
        _scenario(
            config, now,
            title="Oil and gas market update - media interview",
            description="Speaking at a financial news panel about commodity prices. Will reference Northwind Energy and Proseware Mining. Discussion includes the outlook for Saudi Arabia.",
            event_type="media", region="AMER", city="New York",
            hours_to_event=3.5, submitted_hours_ago=32,
            approved=[("RM", 29), ("SA", 25)], pending_since_hours=25,
            securities=["Northwind Energy", "Proseware Mining"], sovereigns=["Saudi Arabia"],
        ),
        _scenario(
            config, now,
            title="Bank capital rules - client webinar",
            description="Speaking at a client webinar about bank capital rules. Will reference Contoso Bank, Woodgrove Financial and Fabrikam Semiconductors.",
            event_type="client", region="EMEA", city="London",
            hours_to_event=20, submitted_hours_ago=6,
            approved=[("RM", 4)], pending_since_hours=4,
            securities=["Contoso Bank", "Woodgrove Financial", "Fabrikam Semiconductors"],
        ),
        _scenario(
            config, now,
            title="Interest rate outlook - investor summit",
            description="Speaking at an investor summit about the interest rate outlook.",
            event_type="conference", region="APAC", city="Singapore",
            hours_to_event=96, submitted_hours_ago=2,
            approved=[], pending_since_hours=2,
        ),
    ]


def generate_in_flight(
    count: int = 20,
    now: datetime | None = None,
    seed: int | None = None,
    config: dict[str, Any] | None = None,
) -> list[Pat]:
    """In-flight PATs at `now`: the demo scenarios plus randomly progressed ones."""
    config = config or load_config()
    now = now or datetime.now(UTC)
    rng = random.Random(seed)
    pats = demo_scenarios(now, config)
    attempts = 0
    while len(pats) < count and attempts < count * 50:
        attempts += 1
        region = rng.choices(REGIONS, REGION_WEIGHTS)[0]
        submitted_at = now - timedelta(hours=rng.uniform(0.5, 72))
        event_time = _event_time(rng, submitted_at + timedelta(hours=_lead_hours(rng)), region, config)
        if event_time <= now + timedelta(hours=1):
            continue
        pat = _base_pat(rng, config, new_in_flight_id(), event_time, region)
        gates, status, _ = simulate_gates(rng, pat, submitted_at, config, stop_at=now)
        if status != "in_progress":
            continue
        for step in gates:
            step.queue_load = None  # live load is computed from the current queue
        pats.append(Pat(**pat, submitted_at=submitted_at, gates=gates))
    return pats


def history_stats(history: list[Pat], last_minute_hours: float = 3) -> dict[str, Any]:
    """Headline numbers comparable to the proposal's historical analysis."""
    approved = [pat for pat in history if pat.status == "approved"]
    terminated = [pat for pat in history if pat.status == "terminated"]
    last_minute = [
        pat for pat in approved
        if hours_between(pat.completed_at, pat.event_time) <= last_minute_hours
    ]
    blocking: dict[str, int] = {}
    for pat in terminated:
        gate = next(step.gate for step in pat.gates if step.status == "pending")
        blocking[gate] = blocking.get(gate, 0) + 1
    last_gate: dict[str, int] = {}
    for pat in last_minute:
        gate = pat.gates[-1].gate
        last_gate[gate] = last_gate.get(gate, 0) + 1
    return {
        "total": len(history),
        "approved": len(approved),
        "terminated": len(terminated),
        "terminatedPct": round(100 * len(terminated) / max(len(history), 1), 2),
        "terminationsByGate": blocking,
        "lastMinuteApprovals": len(last_minute),
        "lastMinutePct": round(100 * len(last_minute) / max(len(approved), 1), 1),
        "lastMinuteByFinalGate": last_gate,
    }
