"""Offline tests for the Approval Intelligence Agent graph with a fake LLM."""

import sys
from datetime import timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import pat_fakes  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from pat import llm as pat_llm  # noqa: E402
from pat.monitor_graph import builder, pending_interrupt, resume_tick, start_tick  # noqa: E402
from pat.tools import TOOL_FUNCTIONS  # noqa: E402


def _fixed_health(health):
    return lambda *_: {"health": health, "explanation": "The blocking review is still pending."}


def _fixed_action(action):
    return lambda *_: {"action": action, "urgency": "high", "reason": "Chosen from the facts."}


class MonitorGraphTests(TestCase):
    def setUp(self):
        self.context, self.pats = pat_fakes.make_context()
        self.graph = builder.compile(checkpointer=InMemorySaver())

    def _use(self, fake):
        patcher = patch.object(pat_llm, "llm", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def _tick(self, pat, offset_hours=0.0):
        return start_tick(self.graph, self.context, pat.id, pat_fakes.NOW + timedelta(hours=offset_hours))

    def _resume(self, pat, message):
        return resume_tick(self.graph, self.context, pat.id, message)

    def _escalation_fake(self):
        return self._use(pat_fakes.FakeLLM(
            {"HealthJudgement": _fixed_health("critical"), "InterventionChoice": _fixed_action("escalate_to_backup")},
            tool_script=pat_fakes.default_tool_script(),
        ))

    def test_agent_uses_tools_then_waits_for_escalation_confirmation(self):
        self._escalation_fake()
        pat = self.pats["critical"]
        result = self._tick(pat)

        self.assertEqual(result["reasoning_mode"], "tools")
        self.assertEqual(
            result["tool_calls"],
            ["get_gate_status", "predict_remaining_time", "get_group_coverage", "get_intervention_history"],
        )
        self.assertIn("blocker", result["agent_notes"])
        payload = result["__interrupt__"][0].value
        self.assertEqual(payload["kind"], "confirm_escalation")
        self.assertEqual(payload["parameters"]["recipient"], "research-pat-cg-leads")
        self.assertIsNotNone(pending_interrupt(self.graph, pat.id))
        self.assertEqual(self.context.repo.list_interventions(pat.id), [])

    def test_confirmed_escalation_sends_notification_to_backup(self):
        self._escalation_fake()
        pat = self.pats["critical"]
        self._tick(pat)
        result = self._resume(pat, "yes, go ahead")

        record = result["intervention"]
        self.assertEqual(record["outcome"], "sent")
        self.assertEqual(record["action"], "escalate_to_backup")
        sent = self.context.notifier.sent
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].audience, "backup")
        self.assertEqual(sent[0].recipient, "research-pat-cg-leads")
        self.assertEqual(sent[0].blocking_gate, "CG")
        self.assertEqual(sent[0].source, "llm")
        self.assertIsNone(pending_interrupt(self.graph, pat.id))
        self.assertEqual(len(self.context.repo.list_interventions(pat.id)), 1)

    def test_rejected_escalation_sends_nothing(self):
        self._escalation_fake()
        pat = self.pats["critical"]
        self._tick(pat)
        result = self._resume(pat, "no, leave it")

        self.assertEqual(result["intervention"]["outcome"], "cancelled")
        self.assertEqual(self.context.notifier.sent, [])

    def test_unclear_replies_ask_again_then_cancel(self):
        self._escalation_fake()
        pat = self.pats["critical"]
        self._tick(pat)
        for _ in range(2):
            result = self._resume(pat, "hmm, maybe")
            self.assertIn("didn't understand", result["__interrupt__"][0].value["message"])
        result = self._resume(pat, "hmm, maybe")
        self.assertEqual(result["intervention"]["outcome"], "cancelled")

    def test_llm_cannot_understate_health(self):
        self._use(pat_fakes.FakeLLM(
            {"HealthJudgement": _fixed_health("on_track"), "InterventionChoice": _fixed_action("none")},
        ))
        result = self._tick(self.pats["critical"])
        assessment = result["assessment"]
        self.assertTrue(assessment["overridden"])
        self.assertEqual(assessment["llm_health"], "on_track")
        self.assertEqual(assessment["health"], assessment["computed_health"])
        self.assertNotEqual(assessment["health"], "on_track")

    def test_alerts_for_on_track_pat_are_dropped(self):
        self._use(pat_fakes.FakeLLM(
            {"HealthJudgement": _fixed_health("on_track"), "InterventionChoice": _fixed_action("notify_submitter")},
        ))
        result = self._tick(self.pats["on_track"])
        record = result["intervention"]
        self.assertEqual(record["assessment"]["health"], "on_track")
        self.assertEqual(record["proposed_action"], "notify_submitter")
        self.assertEqual(record["outcome"], "no_action")
        self.assertIn("on track", record["policy_notes"][0])
        self.assertEqual(self.context.notifier.sent, [])

    def test_critical_pat_always_gets_a_notification(self):
        self._use(pat_fakes.FakeLLM(
            {
                "HealthJudgement": _fixed_health("critical"),
                "InterventionChoice": lambda *_: {"action": "none", "urgency": "low", "reason": "Nothing to do."},
            },
        ))
        result = self._tick(self.pats["critical"])
        record = result["intervention"]
        self.assertEqual(record["proposed_action"], "none")
        self.assertEqual(record["action"], "notify_submitter")
        self.assertEqual(record["urgency"], "high")
        self.assertEqual(record["outcome"], "sent")
        self.assertIn("Critical PATs", record["policy_notes"][0])

    def test_repeat_notification_within_cooldown_is_suppressed(self):
        self._use(pat_fakes.FakeLLM(
            {"HealthJudgement": _fixed_health("at_risk"), "InterventionChoice": _fixed_action("notify_submitter")},
        ))
        pat = self.pats["at_risk"]
        first = self._tick(pat)
        second = self._tick(pat, offset_hours=1)

        if first["assessment"]["health"] == "critical":
            self.skipTest("at-risk scenario computed as critical for this seed")
        self.assertEqual(first["intervention"]["outcome"], "sent")
        self.assertEqual(second["intervention"]["outcome"], "suppressed")
        self.assertEqual(len(self.context.notifier.sent), 1)
        self.assertEqual(len(self.context.repo.list_interventions(pat.id)), 2)

    def test_escalation_for_non_critical_pat_becomes_a_nudge(self):
        self._use(pat_fakes.FakeLLM(
            {"HealthJudgement": _fixed_health("at_risk"), "InterventionChoice": _fixed_action("escalate_to_backup")},
        ))
        result = self._tick(self.pats["at_risk"])
        record = result["intervention"]
        if record["assessment"]["health"] == "critical":
            self.skipTest("at-risk scenario computed as critical for this seed")
        self.assertEqual(record["action"], "nudge_approval_group")
        self.assertEqual(record["outcome"], "sent")
        self.assertEqual(record["notification"]["audience"], "approval_group")

    def test_tool_loop_failure_falls_back_to_all_tools(self):
        self._use(pat_fakes.FakeLLM(tool_script=[RuntimeError("tool calling unsupported")]))
        with self.assertLogs("pat", "WARNING"):
            result = self._tick(self.pats["on_track"])
        self.assertEqual(result["reasoning_mode"], "fallback")
        self.assertEqual(set(result["tool_results"]), set(TOOL_FUNCTIONS))
        self.assertIn("intervention", result)

    def test_ungrounded_message_uses_template(self):
        self._escalation_fake().handlers["DraftMessage"] = lambda *_: {
            "title": "Urgent", "body": "Only 777 minutes left.", "next_steps": [], "approval_group_note": None,
        }
        pat = self.pats["critical"]
        self._tick(pat)
        result = self._resume(pat, "yes")
        notification = result["intervention"]["notification"]
        self.assertEqual(notification["source"], "template")
        self.assertNotIn("777", notification["body"])
        self.assertTrue(notification["recommended_steps"])

    def test_tick_after_event_terminates_pat(self):
        fake = self._use(pat_fakes.FakeLLM())
        pat = self.pats["critical"]
        result = self._tick(pat, offset_hours=10)
        self.assertTrue(result["closed"])
        self.assertIn("terminated", result["message"])
        self.assertEqual(self.context.repo.get_pat(pat.id).status, "terminated")
        self.assertEqual(fake.calls, [])
        events = self.context.repo.list_events(pat.id)
        self.assertEqual(events[-1].type, "terminated")
