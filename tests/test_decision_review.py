"""Decision review must preserve uncertainty without imposing new analyses."""
import unittest
import hashlib
import json
import os
from pathlib import Path
import tempfile

from decision_review import adjudicate_decision_review, assessment_only_repair, decision_review_prompt


def consistent_review():
    return {"claim_consistency": "consistent", "issues": [], "next_action": "accept",
            "explanation": "Decision matches the actual evidence and explicit pending work."}


class DecisionReviewTests(unittest.TestCase):
    def test_consistent_maintain_is_accepted_without_new_requirements(self):
        review = consistent_review()
        verdict = adjudicate_decision_review(review)
        self.assertEqual(verdict["status"], "accepted")
        self.assertEqual(verdict["review"], review)
        self.assertNotIn("required_diagnostic_metrics", verdict)

    def test_contradictory_accept_is_sent_for_decision_revision(self):
        review = {**consistent_review(), "claim_consistency": "contradicted",
            "issues": [{"source": "scientific_review", "finding": "Blocked comparison was called completed."}]}
        verdict = adjudicate_decision_review(review)
        self.assertEqual(verdict["status"], "revision_required")
        self.assertEqual(verdict["next_action"], "revise")
        self.assertEqual(review["next_action"], "accept")

    def test_unknown_is_blocked_not_accepted(self):
        verdict = adjudicate_decision_review({**consistent_review(), "claim_consistency": "unverified"})
        self.assertEqual(verdict["status"], "blocked")
        self.assertEqual(verdict["claim_consistency"], "unverified")

    def test_prompt_uses_existing_review_without_requesting_new_tests(self):
        prompt = decision_review_prompt("Broad question", "Compare models", {}, {"p_value": 0.192},
            {"status": "blocked"}, {"action": "maintain", "basis_keywords": "Comparison remains unperformed."})
        self.assertIn('"status": "blocked"', prompt)
        self.assertIn('"p_value": 0.192', prompt)
        self.assertIn("Maintaining a hypothesis can be appropriate", prompt)
        self.assertIn("not a new analysis/design review", prompt)
        self.assertIn("explicitly leaving prior work unresolved", prompt)

    def test_unknown_issue_source_is_rejected(self):
        with self.assertRaises(ValueError):
            adjudicate_decision_review({**consistent_review(), "issues": [
                {"source": "hidden_paper", "finding": "Does not match the paper."}]})

    def test_only_interpretation_failure_uses_assessment_repair(self):
        review = {"requested_test_status": "fulfilled", "scientific_validity": "adequate",
            "required_assumptions_status": "satisfied", "claim_consistency": "contradicted", "next_action": "repair"}
        self.assertTrue(assessment_only_repair(review))
        self.assertFalse(assessment_only_repair({**review, "scientific_validity": "invalid"}))
        self.assertFalse(assessment_only_repair({**review, "requested_test_status": "unfulfilled"}))
        self.assertFalse(assessment_only_repair({**review, "required_assumptions_status": "unverified"}))


class DecisionHandbackTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Reviewed Scheduler allocation required")
        from four_role_discovery import RoleHistory
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.session = self.root/"attempt-0001"
        self.session.mkdir()
        self.history = RoleHistory(self.root)
        self.record = {}
        self.facts = {"evidence_id": "result-0001", "analysis_status": "succeeded"}
        self.decision = {"action": "maintain", "basis_keywords": "Comparison completed successfully.",
            "next_inquiry": "Investigate generalizability.", "evidence_ids": ["result-0001"]}

    def tearDown(self):
        self.tmp.cleanup()

    def run_review(self, model):
        from four_role_discovery import review_final_decision
        return review_final_decision(self.session, 1, None, model, self.history, self.record,
            decision=self.decision, contract={}, facts=self.facts,
            scientific_review={"status": "blocked", "requested_test_status": "unfulfilled"},
            supplied={}, research_question="Research question", inquiry="Compare the models",
            shared_memory_enabled=False)

    def test_false_completion_is_repaired_and_reviewed_before_acceptance(self):
        calls = []
        repaired = {**self.decision, "basis_keywords": "Comparison remains unperformed; existing association retained."}
        def model(stage, prompt, schema, tokens, seed):
            calls.append(stage)
            if stage == "hypothesis_decision_repair":
                self.assertIn("Do not change the actual analysis", prompt)
                self.assertIn("Blocked comparison was called completed", prompt)
                return repaired
            self.assertEqual(stage, "orchestrator_decision_review")
            if len(calls) == 1:
                return {**consistent_review(), "claim_consistency": "contradicted", "next_action": "revise",
                    "issues": [{"source": "scientific_review", "finding": "Blocked comparison was called completed."}]}
            return consistent_review()
        decision, verdict = self.run_review(model)
        self.assertEqual(decision, repaired)
        self.assertEqual(verdict["status"], "accepted")
        self.assertEqual(calls, ["orchestrator_decision_review", "hypothesis_decision_repair", "orchestrator_decision_review"])
        self.assertEqual(json.loads((self.session/"decision-proposal-0001.json").read_text()), self.decision)
        self.assertEqual(json.loads((self.session/"decision-proposal-0002.json").read_text()), repaired)
        for source in verdict["sources"]:
            self.assertEqual(hashlib.sha256((self.root/source["path"]).read_bytes()).hexdigest(), source["sha256"])

    def test_identical_rejected_decision_stops_without_publishing(self):
        calls = []
        def model(stage, prompt, schema, tokens, seed):
            calls.append(stage)
            if stage == "hypothesis_decision_repair": return self.decision
            return {**consistent_review(), "claim_consistency": "contradicted", "next_action": "revise"}
        decision, verdict = self.run_review(model)
        self.assertIsNone(decision)
        self.assertEqual(verdict["status"], "blocked")
        self.assertEqual(len(calls), 2)
        self.assertFalse((self.session/"decision.json").exists())
        self.assertTrue((self.root/verdict["repeated_proposal_source"]["path"]).exists())

    def test_review_deadline_preserves_rejected_proposal_and_raw_review(self):
        from private_inference import RuntimeIsolationError
        def model(stage, prompt, schema, tokens, seed):
            if stage == "hypothesis_decision_repair":
                error = RuntimeIsolationError("Allocation deadline")
                error.deadline_exhausted = True
                raise error
            return {**consistent_review(), "claim_consistency": "contradicted", "next_action": "revise"}
        with self.assertRaises(RuntimeIsolationError):
            self.run_review(model)
        self.assertTrue((self.session/"decision-proposal-0001.json").exists())
        self.assertTrue((self.session/"decision-review-0001.json").exists())
        self.assertFalse((self.session/"decision.json").exists())

    def test_rejected_final_decision_never_enters_shared_graph(self):
        from four_role_discovery import execute_four_role_attempt
        from research_memory.two_role import TwoRoleMemory
        from test_four_role import hypothesis, plan, result
        from test_orchestrator_review import favorable
        from two_role_discovery import QUESTION
        (self.root/"config.json").write_text(json.dumps({"question": QUESTION}))
        memory = TwoRoleMemory(self.root)
        def model(stage, prompt, schema, tokens, seed):
            if stage == "hypothesis": value = hypothesis()
            elif stage == "empiricist_plan": value = {**plan(), "diagnostic_r_code": None}
            elif stage == "empiricist_assessment":
                value = {"execution_assessment": "Association only.", "diagnostic_action": "adequate_for_interpretation"}
            elif stage == "orchestrator_scientific_review": value = favorable()
            elif stage == "hypothesis_decision": value = self.decision
            elif stage == "orchestrator_decision_review":
                value = {**consistent_review(), "claim_consistency": "unverified", "next_action": "blocked"}
            else: self.fail(stage)
            if any({"action", "query", "answer"}.issubset(branch.get("properties", {})) for branch in schema.get("anyOf", [])):
                return {"action": "answer", "query": None, "answer": value}
            return value
        record, _, decision = execute_four_role_attempt(self.session, 1, memory, model,
            lambda folder, action: result(action), data_identity={"sha256": "fixture"})
        self.assertEqual(record["status"], "partial", record)
        self.assertEqual(record["decision_status"], "blocked")
        self.assertEqual(decision, {})
        self.assertNotIn("decision-0001", memory.nodes)
        self.assertIn("result-0001", memory.nodes)
        self.assertFalse((self.session/"decision.json").exists())
        self.assertEqual(json.loads((self.session/"orchestrator.json").read_text())["status"], "decision_blocked")
