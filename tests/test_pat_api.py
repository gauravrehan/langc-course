"""Offline tests for the /pat endpoints with in-memory storage and a fake LLM."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import pat_fakes  # noqa: E402  (clears MONGODB_URI before the app is imported)
from fastapi.testclient import TestClient  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import chat  # noqa: E402
from pat import api as pat_api  # noqa: E402
from pat import llm as pat_llm  # noqa: E402
from pat.advisor_graph import builder as advisor_builder  # noqa: E402
from pat.monitor_graph import builder as monitor_builder  # noqa: E402


class PatApiTests(TestCase):
    def setUp(self):
        self.context, self.pats = pat_fakes.make_context()
        pat_api.set_services(pat_api.PatServices(
            context=self.context,
            advisor=advisor_builder.compile(),
            monitor=monitor_builder.compile(checkpointer=InMemorySaver()),
        ))
        self.addCleanup(pat_api.set_services, None)
        self.fake = pat_fakes.FakeLLM(tool_script=pat_fakes.default_tool_script())
        patcher = patch.object(pat_llm, "llm", self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(chat.app)

    def _body(self, hours_to_event=30):
        return {
            "title": "Rates outlook - TV interview",
            "description": "Speaking at a television interview about the interest rate outlook. Will reference Contoso Bank.",
            "eventType": "media",
            "region": "AMER",
            "city": "New York",
            "eventTime": (datetime.now(UTC) + timedelta(hours=hours_to_event)).isoformat(),
        }

    def test_advise_returns_camel_case_outlook(self):
        response = self.client.post("/pat/advise", json=self._body())
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("probabilityPct", data["outlook"])
        self.assertEqual(data["signals"]["securities"], ["Contoso Bank"])
        self.assertEqual(data["summarySource"], "llm")

    def test_advise_rejects_past_event(self):
        response = self.client.post("/pat/advise", json=self._body(hours_to_event=-1))
        self.assertEqual(response.status_code, 422)

    def test_submit_approve_gates_and_tick(self):
        created = self.client.post("/pat", json=self._body(hours_to_event=200)).json()
        pat_id = created["id"]
        self.assertEqual([gate["status"] for gate in created["gates"]], ["pending", "waiting", "waiting", "waiting"])

        approved = self.client.post(f"/pat/{pat_id}/events", json={"type": "gate_approved"}).json()
        self.assertEqual(approved["gates"][1]["status"], "pending")
        wrong_gate = self.client.post(f"/pat/{pat_id}/events", json={"type": "gate_approved", "gate": "CC"})
        self.assertEqual(wrong_gate.status_code, 409)

        tick = self.client.post(f"/pat/{pat_id}/tick", json={})
        self.assertEqual(tick.status_code, 200, tick.text)
        self.assertEqual(tick.json()["status"], "completed")

        health = self.client.get(f"/pat/{pat_id}/health").json()
        self.assertEqual(health["blockingGate"], "SA")
        self.assertEqual(len(health["history"]), 1)

        listed = self.client.get("/pat").json()
        self.assertIn(pat_id, [row["id"] for row in listed])

    def test_escalation_flow_over_http(self):
        self.fake.handlers["HealthJudgement"] = lambda *_: {"health": "critical", "explanation": "CG review is pending."}
        self.fake.handlers["InterventionChoice"] = lambda *_: {"action": "escalate_to_backup", "urgency": "high", "reason": "Critical."}
        pat_id = self.pats["critical"].id
        as_of = pat_fakes.NOW.isoformat()

        tick = self.client.post(f"/pat/{pat_id}/tick", json={"asOf": as_of}).json()
        self.assertEqual(tick["status"], "awaiting_confirmation")
        self.assertEqual(self.client.post(f"/pat/{pat_id}/tick", json={"asOf": as_of}).status_code, 409)

        resumed = self.client.post(f"/pat/{pat_id}/resume", json={"message": "yes"}).json()
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["intervention"]["outcome"], "sent")

        notifications = self.client.get(f"/pat/{pat_id}/notifications").json()
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["recipient"], "research-pat-cg-leads")
        self.assertEqual(self.client.post(f"/pat/{pat_id}/resume", json={"message": "yes"}).status_code, 409)

    def test_unknown_pat_returns_404(self):
        self.assertEqual(self.client.post("/pat/PAT-NOPE/tick", json={}).status_code, 404)
        self.assertEqual(self.client.get("/pat/PAT-NOPE/health").status_code, 404)
