"""PAT data storage: MongoDB when MONGODB_URI is set, otherwise in memory.

Collections live in the same database as the LangGraph checkpointer but never
touch the checkpoint collections.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from pat.models import GateEvent, InterventionRecord, Pat, PatStatus

logger = logging.getLogger("pat")

HISTORY = "pat_history"
REQUESTS = "pat_requests"
EVENTS = "pat_gate_events"
INTERVENTIONS = "pat_interventions"
LESSONS = "pat_lessons"


class PatRepository(ABC):
    @abstractmethod
    def replace_history(self, pats: list[Pat]) -> None: ...

    @abstractmethod
    def load_history(self) -> list[Pat]: ...

    @abstractmethod
    def replace_in_flight(self, pats: list[Pat]) -> None:
        """Replace all submitted PATs and clear their events and interventions."""

    @abstractmethod
    def get_pat(self, pat_id: str) -> Pat | None: ...

    @abstractmethod
    def save_pat(self, pat: Pat) -> None: ...

    @abstractmethod
    def list_pats(self, status: PatStatus | None = None) -> list[Pat]: ...

    @abstractmethod
    def add_event(self, event: GateEvent) -> None: ...

    @abstractmethod
    def list_events(self, pat_id: str) -> list[GateEvent]: ...

    @abstractmethod
    def add_intervention(self, record: InterventionRecord) -> None: ...

    @abstractmethod
    def list_interventions(self, pat_id: str) -> list[InterventionRecord]: ...


class InMemoryPatRepository(PatRepository):
    def __init__(self) -> None:
        self.history: list[Pat] = []
        self.pats: dict[str, Pat] = {}
        self.events: list[GateEvent] = []
        self.interventions: list[InterventionRecord] = []

    def replace_history(self, pats: list[Pat]) -> None:
        self.history = [pat.model_copy(deep=True) for pat in pats]

    def load_history(self) -> list[Pat]:
        return [pat.model_copy(deep=True) for pat in self.history]

    def replace_in_flight(self, pats: list[Pat]) -> None:
        self.pats = {pat.id: pat.model_copy(deep=True) for pat in pats}
        self.events = []
        self.interventions = []

    def get_pat(self, pat_id: str) -> Pat | None:
        pat = self.pats.get(pat_id)
        return pat.model_copy(deep=True) if pat else None

    def save_pat(self, pat: Pat) -> None:
        self.pats[pat.id] = pat.model_copy(deep=True)

    def list_pats(self, status: PatStatus | None = None) -> list[Pat]:
        return [
            pat.model_copy(deep=True)
            for pat in self.pats.values()
            if status is None or pat.status == status
        ]

    def add_event(self, event: GateEvent) -> None:
        self.events.append(event.model_copy(deep=True))

    def list_events(self, pat_id: str) -> list[GateEvent]:
        return sorted(
            (event for event in self.events if event.pat_id == pat_id),
            key=lambda event: event.timestamp,
        )

    def add_intervention(self, record: InterventionRecord) -> None:
        self.interventions.append(record.model_copy(deep=True))

    def list_interventions(self, pat_id: str) -> list[InterventionRecord]:
        return sorted(
            (item for item in self.interventions if item.pat_id == pat_id),
            key=lambda item: item.created_at,
        )


def _to_doc(model: Any, key: str) -> dict[str, Any]:
    doc = model.model_dump(mode="python")
    doc["_id"] = doc[key]
    return doc


def _from_doc(doc: dict[str, Any]) -> dict[str, Any]:
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


class MongoPatRepository(PatRepository):
    def __init__(self, database: Any) -> None:
        self.db = database
        self.db[HISTORY].create_index([("event_type", 1), ("region", 1)])
        self.db[HISTORY].create_index("submitted_at")
        self.db[REQUESTS].create_index("status")
        self.db[EVENTS].create_index([("pat_id", 1), ("timestamp", 1)])
        self.db[INTERVENTIONS].create_index([("pat_id", 1), ("created_at", 1)])

    def replace_history(self, pats: list[Pat]) -> None:
        self.db[HISTORY].delete_many({})
        if pats:
            self.db[HISTORY].insert_many([_to_doc(pat, "id") for pat in pats])

    def load_history(self) -> list[Pat]:
        return [Pat.model_validate(_from_doc(doc)) for doc in self.db[HISTORY].find()]

    def replace_in_flight(self, pats: list[Pat]) -> None:
        for name in (REQUESTS, EVENTS, INTERVENTIONS):
            self.db[name].delete_many({})
        if pats:
            self.db[REQUESTS].insert_many([_to_doc(pat, "id") for pat in pats])

    def get_pat(self, pat_id: str) -> Pat | None:
        doc = self.db[REQUESTS].find_one({"_id": pat_id})
        return Pat.model_validate(_from_doc(doc)) if doc else None

    def save_pat(self, pat: Pat) -> None:
        self.db[REQUESTS].replace_one({"_id": pat.id}, _to_doc(pat, "id"), upsert=True)

    def list_pats(self, status: PatStatus | None = None) -> list[Pat]:
        query = {"status": status} if status else {}
        return [Pat.model_validate(_from_doc(doc)) for doc in self.db[REQUESTS].find(query)]

    def add_event(self, event: GateEvent) -> None:
        self.db[EVENTS].insert_one(event.model_dump(mode="python"))

    def list_events(self, pat_id: str) -> list[GateEvent]:
        cursor = self.db[EVENTS].find({"pat_id": pat_id}).sort("timestamp", 1)
        return [GateEvent.model_validate(_from_doc(doc)) for doc in cursor]

    def add_intervention(self, record: InterventionRecord) -> None:
        self.db[INTERVENTIONS].insert_one(_to_doc(record, "id"))

    def list_interventions(self, pat_id: str) -> list[InterventionRecord]:
        cursor = self.db[INTERVENTIONS].find({"pat_id": pat_id}).sort("created_at", 1)
        return [InterventionRecord.model_validate(_from_doc(doc)) for doc in cursor]


def build_repository() -> PatRepository:
    """MongoDB repository when MONGODB_URI is set, otherwise in memory."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        logger.warning("MONGODB_URI is not configured; PAT data is kept in memory.")
        return InMemoryPatRepository()
    from pymongo import MongoClient

    client = MongoClient(
        uri,
        tz_aware=True,
        serverSelectionTimeoutMS=int(os.getenv("MONGODB_TIMEOUT_MS", "5000")),
    )
    database = client[os.getenv("MONGODB_DATABASE", "langgraph_chat")]
    return MongoPatRepository(database)
