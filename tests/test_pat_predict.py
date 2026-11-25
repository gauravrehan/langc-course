"""Offline tests for synthetic data, the prediction engine and guardrails."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).parent))
import pat_fakes  # noqa: E402
from pat.clock import advance_work, in_coverage  # noqa: E402
from pat.config import load_config  # noqa: E402
from pat.data import history_stats  # noqa: E402
from pat.guards import (  # noqa: E402
    allowed_numbers,
    entity_in_text,
    filter_entities,
    grounding_problems,
    time_left_problems,
)
from pat.models import PatRequest  # noqa: E402
from pat.predict import current_loads  # noqa: E402


class SyntheticDataTests(TestCase):
    def test_history_reproduces_proposal_patterns(self):
        stats = history_stats(pat_fakes.history())
        self.assertGreater(stats["terminatedPct"], 0.3)
        self.assertLess(stats["terminatedPct"], 4)
        self.assertGreater(stats["lastMinutePct"], 10)
        self.assertLess(stats["lastMinutePct"], 25)
        by_gate = stats["terminationsByGate"]
        self.assertEqual(max(by_gate, key=by_gate.get), "CG")

    def test_history_is_reproducible(self):
        first = pat_fakes.generate_history(50, now=pat_fakes.NOW)
        second = pat_fakes.generate_history(50, now=pat_fakes.NOW)
        self.assertEqual(
            [pat.model_dump() for pat in first], [pat.model_dump() for pat in second]
        )


class PredictionTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context, cls.pats = pat_fakes.make_context()
        cls.config = load_config()
        cls.loads = current_loads(cls.context.repo.list_pats("in_progress"), cls.config)

    def _request(self, hours_to_event: float) -> PatRequest:
        return PatRequest(
            title="Rates outlook - TV interview",
            description="Speaking at a television interview about the interest rate outlook.",
            event_type="media",
            region="AMER",
            city="New York",
            event_time=pat_fakes.NOW + timedelta(hours=hours_to_event),
        )

    def test_late_submission_has_lower_probability(self):
        model = self.context.model
        early = model.outlook(self._request(240), pat_fakes.NOW, self.loads)
        late = model.outlook(self._request(8), pat_fakes.NOW, self.loads)
        self.assertGreater(early.probability, late.probability)
        self.assertEqual([step.gate for step in early.journey], ["RM", "SA", "CG", "CC"])

    def test_stuck_gate_close_to_event_is_riskier_than_fresh_pat(self):
        model = self.context.model
        critical = model.outlook(self.pats["critical"], pat_fakes.NOW, self.loads)
        on_track = model.outlook(self.pats["on_track"], pat_fakes.NOW, self.loads)
        self.assertLess(critical.probability, 0.8)
        self.assertGreaterEqual(on_track.probability, 0.8)

    def test_outlook_is_deterministic(self):
        model = self.context.model
        first = model.outlook(self.pats["at_risk"], pat_fakes.NOW, self.loads)
        second = model.outlook(self.pats["at_risk"], pat_fakes.NOW, self.loads)
        self.assertEqual(first, second)

    def test_similar_cases_match_event_type(self):
        summary = self.context.model.similar(self.pats["critical"], pat_fakes.NOW)
        self.assertEqual(summary.count, self.config["similarCases"])
        self.assertTrue(all(example.event_type == "media" for example in summary.examples))

    def test_queue_load_bands(self):
        pats = self.context.repo.list_pats("in_progress")
        loads = current_loads(pats, self.config)
        self.assertEqual(set(loads), {"RM", "SA", "CG", "CC"})
        self.assertEqual(current_loads([], self.config)["CG"], "low")


class ClockTests(TestCase):
    def test_work_pauses_over_the_weekend(self):
        config = load_config()
        # Friday 18:00 New York.
        friday = datetime(2026, 11, 27, 23, 0, tzinfo=UTC)
        finish = advance_work(friday, 2, "AMER", config, off_hours_rate=0)
        # One hour on Friday, the second hour on Monday morning.
        self.assertEqual(finish, datetime(2026, 11, 30, 14, 0, tzinfo=UTC))
        self.assertFalse(in_coverage(datetime(2026, 11, 28, 15, 0, tzinfo=UTC), "AMER", config))


class GuardTests(TestCase):
    def test_entity_must_appear_in_text(self):
        text = "Will reference Contoso Bank and the outlook for Brazil."
        self.assertTrue(entity_in_text("contoso bank", text))
        self.assertFalse(entity_in_text("Initech", text))
        kept, rejected = filter_entities(["Contoso Bank", "Initech", "Brazil"], text)
        self.assertEqual(kept, ["Contoso Bank", "Brazil"])
        self.assertEqual(rejected, ["Initech"])

    def test_numbers_and_gates_must_come_from_facts(self):
        allowed = allowed_numbers({"probabilityOnTimePct": 74, "hoursToEvent": 3.46})
        self.assertEqual(grounding_problems("74% chance, event in 3.5 hours, CG pending.", allowed, {"CG"}), [])
        problems = grounding_problems("90% chance; SA pending.", allowed, {"CG"})
        self.assertEqual(len(problems), 2)

    def test_time_left_must_match_the_deadline(self):
        self.assertEqual(time_left_problems("Only 3.5 hours left before the event.", 3.46), [])
        self.assertEqual(time_left_problems("CG has been pending for 25 hours.", 3.46), [])
        self.assertEqual(len(time_left_problems("CG is pending with 25 hours remaining.", 3.46)), 1)
