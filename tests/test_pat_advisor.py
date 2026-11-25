"""Offline tests for the Submission Advisor graph with a fake LLM."""

import sys
from datetime import timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import pat_fakes  # noqa: E402
from pat import llm as pat_llm  # noqa: E402
from pat.advisor_graph import builder, run_advisor  # noqa: E402
from pat.models import PatRequest  # noqa: E402


def _request(hours_to_event: float = 10) -> PatRequest:
    return PatRequest(
        title="Bank capital rules - TV interview",
        description="Speaking at a television interview about bank capital rules. Will reference Contoso Bank.",
        event_type="media",
        region="AMER",
        city="New York",
        event_time=pat_fakes.NOW + timedelta(hours=hours_to_event),
    )


class AdvisorGraphTests(TestCase):
    def setUp(self):
        self.context, _ = pat_fakes.make_context()
        self.graph = builder.compile()

    def _run(self, fake, request=None):
        with patch.object(pat_llm, "llm", fake):
            return run_advisor(self.graph, self.context, request or _request(), pat_fakes.NOW)

    def test_happy_path_uses_llm_output_after_checks(self):
        fake = pat_fakes.FakeLLM()
        response = self._run(fake)

        self.assertEqual(response.signals.securities, ["Contoso Bank"])
        self.assertEqual(response.signals.rejected, ["Initech"])
        self.assertEqual(response.summary_source, "llm")
        self.assertIn(f"{response.outlook.probability_pct}%", response.summary)
        self.assertEqual([action.id for action in response.recommended_actions], ["submit_now", "enable_monitoring"])
        ids = [driver.id for driver in response.risk_drivers]
        self.assertIn("short_lead_time", ids)
        self.assertIn("media_event", ids)
        self.assertTrue(response.suggested_monitoring)
        self.assertEqual(
            fake.calls,
            ["SignalExtraction", "RiskRanking", "ActionChoices", "AdvisorSummary"],
        )

    def test_llm_ranking_order_is_kept(self):
        seen = []

        def ranking(_schema, messages):
            ids = [driver["id"] for driver in pat_fakes.human_json(messages)["riskDrivers"]]
            seen.extend(ids)
            return {"drivers": [{"id": driver_id, "why": "This matters for timing."} for driver_id in reversed(ids)]}

        response = self._run(pat_fakes.FakeLLM({"RiskRanking": ranking}))
        self.assertGreater(len(seen), 1)
        self.assertEqual([driver.id for driver in response.risk_drivers], list(reversed(seen)))
        self.assertTrue(all(driver.why == "This matters for timing." for driver in response.risk_drivers))

    def test_drivers_the_llm_skips_are_still_returned(self):
        response = self._run(pat_fakes.FakeLLM({
            "RiskRanking": lambda *_: {"drivers": [{"id": "media_event", "why": "Extra review."}]},
        }))
        ids = [driver.id for driver in response.risk_drivers]
        self.assertEqual(ids[0], "media_event")
        self.assertIn("short_lead_time", ids)

    def test_invented_number_triggers_one_retry(self):
        attempts = []

        def summary(_schema, messages):
            attempts.append(messages[0][1])
            if len(attempts) == 1:
                return {"summary": "This PAT has a 12345% chance of approval."}
            facts = pat_fakes.human_json(messages)
            return {"summary": f"About {facts['outlook']['probabilityOnTimePct']}% likely to be approved in time."}

        response = self._run(pat_fakes.FakeLLM({"AdvisorSummary": summary}))
        self.assertEqual(len(attempts), 2)
        self.assertIn("12345", attempts[1])
        self.assertEqual(response.summary_source, "llm")

    def test_repeated_invented_numbers_fall_back_to_template(self):
        response = self._run(pat_fakes.FakeLLM({
            "AdvisorSummary": lambda *_: {"summary": "Approval in 999 hours via SA, RM and the XX gate."},
        }))
        self.assertEqual(response.summary_source, "template")
        self.assertIn(f"{response.outlook.probability_pct}%", response.summary)

    def test_ungrounded_reason_is_replaced_with_catalogue_text(self):
        response = self._run(pat_fakes.FakeLLM({
            "ActionChoices": lambda *_: {"actions": [{"id": "submit_now", "reason": "Gives 4242 extra hours."}]},
        }))
        self.assertEqual(response.recommended_actions[0].id, "submit_now")
        self.assertNotIn("4242", response.recommended_actions[0].reason)

    def test_llm_failures_fall_back_to_code(self):
        failing = {name: (lambda *_: RuntimeError("model down")) for name in pat_fakes.default_handlers()}
        with self.assertLogs("pat", "WARNING"):
            response = self._run(pat_fakes.FakeLLM(failing))
        self.assertEqual(response.signals.securities, [])
        self.assertEqual(response.summary_source, "template")
        self.assertIn("submit_now", [action.id for action in response.recommended_actions])
        self.assertTrue(response.risk_drivers)
        self.assertEqual(response.risk_drivers[0].impact, "high")
