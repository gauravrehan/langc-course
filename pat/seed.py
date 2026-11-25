"""Generate synthetic PAT data and load it into MongoDB.

    uv run python -m pat.seed                 # 5000 historical + 20 in-flight PATs
    uv run python -m pat.seed --in-flight-only  # refresh in-flight PATs for a demo

Only the PAT collections are replaced; LangGraph checkpoint collections are left alone.
In-flight PATs get fresh IDs each run, so old agent threads are never reused.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

from pat.data import generate_history, generate_in_flight, history_stats
from pat.repository import PatRepository, build_repository


def seed(
    repository: PatRepository,
    history_count: int = 5000,
    in_flight_count: int = 20,
    in_flight_only: bool = False,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    summary: dict = {}
    if not in_flight_only:
        history = generate_history(history_count, now=now)
        repository.replace_history(history)
        summary["history"] = history_stats(history)
    in_flight = generate_in_flight(in_flight_count, now=now)
    repository.replace_in_flight(in_flight)
    summary["inFlight"] = [pat.id for pat in in_flight]
    return summary


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history", type=int, default=5000)
    parser.add_argument("--in-flight", type=int, default=20)
    parser.add_argument("--in-flight-only", action="store_true")
    args = parser.parse_args()

    if not os.getenv("MONGODB_URI"):
        sys.exit(
            "MONGODB_URI is not set. Without MongoDB the API seeds an in-memory "
            "store automatically on startup, so there is nothing to do."
        )
    summary = seed(
        build_repository(),
        history_count=args.history,
        in_flight_count=args.in_flight,
        in_flight_only=args.in_flight_only,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
