"""Actual review/repair control flow; run only inside a reviewed Scheduler job."""
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from four_role_discovery import RoleHistory, supervise_result, execute_four_role_attempt
from private_inference import RuntimeIsolationError
from test_four_role import hypothesis, plan, result
from two_role_discovery import QUESTION
from test_decision_review import consistent_review


def reviewed(*, accepted=False, diagnostic=True, invalid=False):
    return {"requested_test_status": "fulfilled" if accepted else "unfulfilled",
        "scientific_validity": "adequate" if accepted else "invalid" if invalid else "unverified",
        "claim_consistency": "consistent" if accepted else "contradicted",
        "required_assumptions_status": "satisfied" if accepted else "violated" if invalid else "unverified",
        "diagnostic_required": diagnostic,
        "required_diagnostic_metrics": [{"name": "gain", "role": "target_estimate"}] if diagnostic else [],
        "issues": [] if accepted else [{"source": "diagnostic_code", "finding":
            "First-user assignment does not establish the requested cluster unit." if invalid else "Requested comparison is missing."}],
        "next_action": "accept" if accepted else "repair", "explanation": "Source-grounded fixture review."}


class ScientificSupervisionTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Reviewed Scheduler allocation required")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config.json").write_text(json.dumps({"question": QUESTION}))
        self.session = self.root / "attempt-0001"
        self.session.mkdir()
        self.history = RoleHistory(self.root)
        self.record = {}
        self.facts = {"analysis_status": "succeeded", "evidence_id": "result-0001",
            "estimates": result(plan()["analysis"])["estimates"], "metrics": [{"n": "100"}],
            "execution_reuse": {"source_attempt": 1, "reason": "same primary action"}}

    def tearDown(self):
        self.tmp.cleanup()

    def supervise(self, model, runner, diagnostic=None):
        if diagnostic is not None:
            self.facts["agent_r_diagnostic"] = diagnostic
            self.record["diagnostic_executions"] = [{"code_sha256": hashlib.sha256(diagnostic["code"].encode()).hexdigest()}]
        return supervise_result(self.session, 1, None, model, self.history, self.record,
            contract=hypothesis(), plan=plan(), facts=self.facts, evidence="No prior evidence.",
            design_evidence="No prior evidence.", supplied={}, research_question=QUESTION,
            inquiry="Compare predictive models with valid uncertainty.", diagnostic_runner=runner,
            shared_memory_enabled=False)

    def test_missing_code_is_executed_then_reassessed_and_reviewed(self):
        calls = []
        runner = Mock(return_value={"status": "completed", "metrics": {"gain": 0.03}})
        def model(stage, prompt, schema, tokens, seed):
            calls.append(stage)
            if stage == "empiricist_assessment":
                return {"execution_assessment": "Actual receipt checked.", "diagnostic_action": "adequate_for_interpretation" if runner.called else "design_problem"}
            if stage == "orchestrator_scientific_review":
                self.assertIn("Private orchestrator conversation", prompt)
                return reviewed(accepted=runner.called)
            if stage == "empiricist_supervised_repair":
                self.assertIn("Requested comparison is missing", prompt)
                return {"action": "repair", "diagnostic_r_code": "valid_comparison()", "reason": "Execute missing comparison."}
            self.fail(stage)
        diagnostic, _, review, _, _ = self.supervise(model, runner)
        self.assertEqual(calls, ["empiricist_assessment", "orchestrator_scientific_review",
            "empiricist_supervised_repair", "empiricist_assessment", "orchestrator_scientific_review"])
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(diagnostic["code"], "valid_comparison()")
        self.assertEqual(review["status"], "accepted")
        self.assertEqual(len(self.history._items("orchestrator")), 2)
        first = json.loads((self.session / "supervision-0001/evidence.json").read_text())
        self.assertIsNone(first["diagnostic_receipt"])
        second = json.loads((self.session / "supervision-0002/evidence.json").read_text())
        self.assertEqual(second["diagnostic_receipt"]["metrics"]["gain"], 0.03)
        for source in review["sources"] + [review["final_review_source"], diagnostic["receipt_source"]]:
            self.assertEqual(hashlib.sha256((self.root/source["path"]).read_bytes()).hexdigest(), source["sha256"])

    def test_finite_but_invalid_cluster_code_requires_new_execution(self):
        original = {"status": "completed", "code": "first_user_per_set()", "metrics": {"gain": 0.04, "obsolete_ci": 0.1}}
        runner = Mock(return_value={"status": "completed", "metrics": {"gain": 0.02}})
        def model(stage, prompt, schema, tokens, seed):
            if stage == "empiricist_assessment":
                return {"execution_assessment": "Finite numbers alone do not prove validity.", "diagnostic_action": "additional_test"}
            if stage == "orchestrator_scientific_review":
                return reviewed(accepted=runner.called, invalid=not runner.called)
            if stage == "empiricist_supervised_repair":
                self.assertIn("First-user assignment", prompt)
                self.assertIn(original["code"], prompt)
                return {"action": "repair", "diagnostic_r_code": "validated_unit_comparison()", "reason": "Correct inference unit."}
            self.fail(stage)
        diagnostic, _, review, _, _ = self.supervise(model, runner, original)
        self.assertEqual(review["status"], "accepted")
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(diagnostic["metrics"]["gain"], 0.02)
        first = json.loads((self.session/"supervision-0001/orchestrator-review.json").read_text())
        self.assertEqual(first["scientific_validity"], "invalid")
        self.assertEqual(original["metrics"]["gain"], 0.04)
        self.assertEqual(self.facts["metrics"][0]["agent_diagnostic_gain"], 0.02)
        self.assertNotIn("agent_diagnostic_obsolete_ci", self.facts["metrics"][0])

    def test_unchanged_code_is_blocked_without_execution(self):
        diagnostic = {"status": "completed", "code": "invalid_comparison()", "metrics": {"gain": 0.04}}
        def model(stage, prompt, schema, tokens, seed):
            if stage == "empiricist_assessment":
                return {"execution_assessment": "Invalid uncertainty.", "diagnostic_action": "design_problem"}
            if stage == "orchestrator_scientific_review": return reviewed(invalid=True)
            return {"action": "repair", "diagnostic_r_code": diagnostic["code"], "reason": "Repeat."}
        runner = Mock()
        _, _, review, _, _ = self.supervise(model, runner, diagnostic)
        self.assertEqual(review["status"], "blocked")
        self.assertIn("No substantive progress", review["blocking_reason"])
        runner.assert_not_called()
        self.assertTrue((self.root/review["resolution_source"]["path"]).exists())

    def test_all_na_cannot_be_accepted_and_explicit_blocker_stops(self):
        diagnostic = {"status": "completed", "code": "write_na()", "metrics": {"gain": None}}
        def model(stage, prompt, schema, tokens, seed):
            if stage == "empiricist_assessment":
                return {"execution_assessment": "Output unavailable.", "diagnostic_action": "design_problem"}
            if stage == "orchestrator_scientific_review": return reviewed(accepted=True)
            return {"action": "unavailable", "diagnostic_r_code": None, "reason": "Required comparison cannot be identified."}
        runner = Mock()
        preserved, _, review, _, _ = self.supervise(model, runner, diagnostic)
        self.assertEqual(review["status"], "blocked")
        self.assertIsNone(preserved["metrics"]["gain"])
        runner.assert_not_called()

    def test_deadline_preserves_review_and_proposal_without_fake_completion(self):
        def model(stage, prompt, schema, tokens, seed):
            if stage == "empiricist_assessment":
                return {"execution_assessment": "Missing diagnostic.", "diagnostic_action": "additional_test"}
            if stage == "orchestrator_scientific_review": return reviewed()
            error = RuntimeIsolationError("Allocation ended")
            error.deadline_exhausted = True
            raise error
        with self.assertRaises(RuntimeIsolationError):
            self.supervise(model, Mock())
        self.assertTrue((self.session/"supervision-0001/orchestrator-review.json").exists())
        self.assertFalse((self.session/"scientific-review.json").exists())

    def test_primary_failure_does_not_trigger_diagnostic_only_repairs(self):
        self.facts["analysis_status"] = "failed"
        calls = []
        def model(stage, prompt, schema, tokens, seed):
            calls.append(stage)
            if stage == "empiricist_assessment":
                return {"execution_assessment": "Primary model execution failed.", "diagnostic_action": "design_problem"}
            if stage == "orchestrator_scientific_review": return reviewed()
            self.fail("Invalid diagnostic-only repair of failed primary: " + stage)
        runner = Mock()
        _, _, review, _, _ = self.supervise(model, runner)
        self.assertEqual(review["status"], "blocked")
        self.assertEqual(review["blocking_domain"], "primary_execution_pending")
        self.assertIn("repair the primary analysis", review["pending_work"])
        runner.assert_not_called()
        self.assertEqual(calls, ["empiricist_assessment", "orchestrator_scientific_review"])

    def test_false_assessment_is_rewritten_without_changing_code_or_results(self):
        calls = []
        corrected = False
        def model(stage, prompt, schema, tokens, seed):
            nonlocal corrected
            calls.append(stage)
            if stage == "empiricist_assessment":
                return {"execution_assessment": "A causal effect was proven.", "diagnostic_action": "adequate_for_interpretation"}
            if stage == "empiricist_assessment_repair":
                self.assertIn("Revise only the assessment claims", prompt)
                corrected = True
                return {"execution_assessment": "Only an association was tested.", "diagnostic_action": "adequate_for_interpretation"}
            if stage == "orchestrator_scientific_review":
                value = reviewed(accepted=True, diagnostic=False)
                if not corrected:
                    value.update(claim_consistency="contradicted", next_action="repair",
                        issues=[{"source": "empirical_assessment", "finding": "Association does not establish causation."}])
                return value
            self.fail(stage)
        runner = Mock()
        original = deepcopy(self.facts)
        diagnostic, assessment, review, _, _ = self.supervise(model, runner)
        runner.assert_not_called()
        self.assertIsNone(diagnostic)
        self.assertEqual(self.facts, original)
        self.assertEqual(assessment["execution_assessment"], "Only an association was tested.")
        self.assertEqual(review["status"], "accepted")
        self.assertEqual(calls, ["empiricist_assessment", "orchestrator_scientific_review",
                                 "empiricist_assessment_repair", "orchestrator_scientific_review"])

    def test_same_false_assessment_blocks_without_runner(self):
        def model(stage, prompt, schema, tokens, seed):
            if stage in {"empiricist_assessment", "empiricist_assessment_repair"}:
                return {"execution_assessment": "Causality proven.", "diagnostic_action": "adequate_for_interpretation"}
            if stage == "orchestrator_scientific_review":
                return {**reviewed(accepted=True, diagnostic=False), "claim_consistency": "contradicted",
                        "next_action": "repair"}
            self.fail(stage)
        runner = Mock()
        _, _, review, _, _ = self.supervise(model, runner)
        self.assertEqual(review["status"], "blocked")
        self.assertIn("same contradicted assessment", review["blocking_reason"])
        runner.assert_not_called()

    def test_initial_assessment_deadline_preserves_result_and_pending_review(self):
        def model(stage, prompt, schema, tokens, seed):
            if stage == "hypothesis": return hypothesis()
            if stage == "empiricist_plan": return {**plan(), "diagnostic_r_code": None}
            if stage == "empiricist_assessment":
                error = RuntimeIsolationError("Allocation ended during assessment.")
                error.deadline_exhausted = True
                raise error
            self.fail(stage)
        record, _, _ = execute_four_role_attempt(self.session, 1, None, model,
            lambda folder, action: result(action), data_identity="fixture", condition="B")
        self.assertEqual(record["status"], "censored", record)
        self.assertEqual(record["supervision_status"], "interrupted_review_pending")
        saved = json.loads((self.session/"tool-result.json").read_text())
        self.assertTrue(saved["scientific_review_pending"])
        self.assertEqual(saved["estimates"], result(plan()["analysis"])["estimates"])
        ledger = json.loads((self.session/"orchestrator.json").read_text())
        self.assertEqual(len(ledger["events"]), 4)
        self.assertEqual(ledger["events"][-1]["stage"], "empiricist_assessment")

    def test_full_attempt_publishes_only_final_evidence_and_decision_sees_review(self):
        runner = Mock(return_value={"status": "completed", "metrics": {"gain": 0.03}})
        def model(stage, prompt, schema, tokens, seed):
            if stage == "hypothesis": return hypothesis()
            if stage == "empiricist_plan": return {**plan(), "diagnostic_r_code": None}
            if stage == "empiricist_assessment":
                return {"execution_assessment": "Actual evidence checked.", "diagnostic_action": "adequate_for_interpretation" if runner.called else "design_problem"}
            if stage == "orchestrator_scientific_review":
                self.assertFalse((self.session/"tool-result.json").exists())
                return reviewed(accepted=runner.called)
            if stage == "empiricist_supervised_repair":
                return {"action": "repair", "diagnostic_r_code": "comparison()", "reason": "Execute requested work."}
            if stage == "hypothesis_decision":
                self.assertIn("Orchestrator scientific review", prompt)
                self.assertIn('"status": "accepted"', prompt)
                return {"action": "maintain", "basis_keywords": "Comparison executed and reviewed.",
                    "next_inquiry": "Assess external validity.", "evidence_ids": ["result-0001"]}
            if stage == "orchestrator_decision_review": return consistent_review()
            self.fail(stage)
        record, _, _ = execute_four_role_attempt(self.session, 1, None, model,
            lambda folder, action: result(action), diagnostic_runner=runner, data_identity="fixture",
            condition="B", inquiry="Compare predictive models with valid uncertainty.")
        self.assertEqual(record["status"], "completed", record)
        self.assertEqual(record["scientific_review_status"], "accepted")
        workflow = json.loads((self.session/"orchestrator.json").read_text())
        self.assertEqual(workflow["status"], "handoffs_completed")
        for event in workflow["events"]:
            for source in event["sources"]:
                self.assertEqual(hashlib.sha256((self.session/source["path"]).read_bytes()).hexdigest(), source["sha256"])
        facts = json.loads((self.session/"current-result-facts.json").read_text())
        actual = json.loads((self.session/"tool-result.json").read_text())
        self.assertEqual(facts["agent_r_diagnostic"], actual["agent_r_diagnostic"])
        self.assertEqual(facts["metrics"], actual["metrics"])
        self.assertNotIn("agent_r_diagnostic", json.loads((self.session/"primary-tool-result.json").read_text()))
