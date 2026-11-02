"""A resumed Decision must inherit its actual scientific review unchanged."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from four_role_discovery import execute_four_role_attempt
from four_role_resume import pending_result, resume_result
from memory_iteration import publish_cursor, source_reference
from private_inference import RuntimeIsolationError
from research_memory.two_role import TwoRoleMemory
from test_four_role import hypothesis, plan, result
from test_scientific_supervision import reviewed
from test_decision_review import consistent_review
from two_role_discovery import QUESTION


def own_answer(value, schema):
    """Adapt transport only; never supply or replace a scientific judgment."""
    if any({"action", "query", "answer"}.issubset(branch.get("properties", {}))
           for branch in schema.get("anyOf", [])):
        return {"action": "answer", "query": None, "answer": value}
    return value


class SupervisedResumeTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Reviewed Scheduler allocation required")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/"config.json").write_text(json.dumps({"question": QUESTION}))
        self.original = self.root/"attempt-0001"
        self.original.mkdir()
        self.memory = TwoRoleMemory(self.root)
        self.calls = []
        self.runner_calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def create_censored_decision(self, *, blocked):
        expected_review = reviewed(accepted=not blocked, diagnostic=blocked, invalid=blocked)
        if blocked:
            expected_review["next_action"] = "blocked"
        def model(stage, prompt, schema, tokens, seed):
            self.calls.append(stage)
            if stage == "hypothesis": value = hypothesis()
            elif stage == "empiricist_plan": value = {**plan(), "diagnostic_r_code": None}
            elif stage == "empiricist_assessment":
                value = {"execution_assessment": "Requested comparison is unresolved." if blocked else "Only the requested association was tested.",
                         "diagnostic_action": "design_problem" if blocked else "adequate_for_interpretation"}
            elif stage == "orchestrator_scientific_review": value = expected_review
            elif stage == "hypothesis_decision":
                error = RuntimeIsolationError("Fixture allocation ended at Decision.")
                error.deadline_exhausted = True
                raise error
            else: self.fail("Unexpected extra work: " + stage)
            return own_answer(value, schema)
        def runner(folder, action):
            self.runner_calls.append(action)
            return result(action)
        record, _, _ = execute_four_role_attempt(self.original, 1, self.memory, model, runner,
            data_identity={"sha256": "fixture"}, inquiry="Compare the requested models." if blocked else QUESTION)
        self.assertEqual(record["status"], "censored", record)
        self.assertEqual(record["scientific_review"]["status"], "blocked" if blocked else "accepted")
        self.assertEqual(record["scientific_review"]["review"], expected_review)
        self.assertEqual(record["decision_status"], "censored")
        (self.original/"attempt.json").write_text(json.dumps(record))
        refs = [source_reference(self.original/name, self.root) for name in ("plan.json", "attempt.json")]
        publish_cursor(self.root, 1, "old-job", *refs, None)
        cursor = json.loads((self.root/"CURRENT.json").read_text())
        self.assertEqual(pending_result(self.root, cursor), self.original)
        return record

    def resume(self, expected_status):
        session = self.original/"resume-new-job"
        session.mkdir()
        calls = []
        def model(stage, prompt, schema, tokens, seed):
            calls.append(stage)
            if stage == "orchestrator_decision_review":
                return own_answer(consistent_review(), schema)
            self.assertEqual(stage, "hypothesis_decision")
            self.assertIn("Orchestrator scientific review and pending work", prompt)
            self.assertIn('"status": "' + expected_status + '"', prompt)
            value = {"action": "additional_test" if expected_status == "blocked" else "maintain",
                "basis_keywords": "Preserve the original scientific review and its unresolved work.",
                "next_inquiry": "Resolve the missing test." if expected_status == "blocked" else "Assess generalizability.",
                "evidence_ids": ["result-0001"]}
            return own_answer(value, schema)
        record, _, decision = resume_result(session, self.original, self.memory, model,
                                            data_identity={"sha256": "fixture"})
        self.assertEqual(calls, ["hypothesis_decision", "orchestrator_decision_review"])
        return session, record, decision

    def test_blocked_review_survives_resume_as_partial_without_refit(self):
        original_record = self.create_censored_decision(blocked=True)
        original_bytes = {path: path.read_bytes() for path in self.original.rglob("*") if path.is_file()}
        session, record, decision = self.resume("blocked")
        self.assertEqual(record["status"], "partial", record)
        self.assertEqual(record["scientific_review_status"], "blocked")
        self.assertEqual(record["research_task_status"], "unfulfilled")
        self.assertEqual(record["scientific_review"], original_record["scientific_review"])
        self.assertFalse(record["runner_reexecuted"])
        self.assertEqual(len(self.runner_calls), 1)
        self.assertEqual(self.memory.nodes["decision-0001"]["payload"], decision)
        self.assertEqual(json.loads((session/"orchestrator.json").read_text())["status"], "handoffs_completed")
        for path, content in original_bytes.items():
            self.assertEqual(path.read_bytes(), content)

    def test_accepted_review_remains_accepted_without_new_review_call(self):
        original_record = self.create_censored_decision(blocked=False)
        _, record, _ = self.resume("accepted")
        self.assertEqual(record["status"], "completed", record)
        self.assertEqual(record["scientific_review"], original_record["scientific_review"])
        self.assertEqual(record["scientific_review_status"], "accepted")
        self.assertEqual(len(self.runner_calls), 1)

    def test_changed_source_review_is_rejected_before_model_call(self):
        original_record = self.create_censored_decision(blocked=True)
        reference = original_record["scientific_review"]["final_review_source"]
        (self.root/reference["path"]).write_text("{}")
        session = self.original/"resume-tampered"
        session.mkdir()
        def forbidden(*args): self.fail("Tampered review reached the model")
        with self.assertRaisesRegex(ValueError, "Scientific review source changed"):
            resume_result(session, self.original, self.memory, forbidden, data_identity={"sha256": "fixture"})

    def test_changed_review_summary_cannot_override_original_record(self):
        self.create_censored_decision(blocked=True)
        path = self.original/"scientific-review.json"
        changed = json.loads(path.read_text())
        changed["status"] = "accepted"
        path.write_text(json.dumps(changed))
        session = self.original/"resume-tampered"
        session.mkdir()
        def forbidden(*args): self.fail("Forged review reached the model")
        with self.assertRaisesRegex(ValueError, "Saved scientific review differs"):
            resume_result(session, self.original, self.memory, forbidden, data_identity={"sha256": "fixture"})

    def test_interrupted_review_is_actually_completed_on_resume_not_treated_as_legacy(self):
        def initial(stage, prompt, schema, tokens, seed):
            if stage == "hypothesis": value = hypothesis()
            elif stage == "empiricist_plan": value = {**plan(), "diagnostic_r_code": None}
            elif stage == "empiricist_assessment":
                error = RuntimeIsolationError("Assessment allocation exhausted")
                error.deadline_exhausted = True
                raise error
            else: self.fail(stage)
            return own_answer(value, schema)
        record, _, _ = execute_four_role_attempt(self.original, 1, self.memory, initial,
            lambda folder, action: result(action), data_identity={"sha256": "fixture"})
        self.assertEqual(record["status"], "censored", record)
        self.assertNotIn("scientific_review", record)
        self.assertEqual(record["supervision_status"], "interrupted_review_pending")
        (self.original/"attempt.json").write_text(json.dumps(record))
        calls = []
        def resumed(stage, prompt, schema, tokens, seed):
            calls.append(stage)
            if stage == "empiricist_assessment":
                value = {"execution_assessment": "Requested comparison has not been executed.", "diagnostic_action": "design_problem"}
            elif stage == "orchestrator_scientific_review":
                value = reviewed(invalid=True)
                value["next_action"] = "blocked"
            elif stage == "hypothesis_decision":
                self.assertIn('"status": "blocked"', prompt)
                self.assertIn("Post-result resume reviews saved work", prompt)
                value = {"action": "additional_test", "basis_keywords": "Comparison remains pending.",
                    "next_inquiry": "Resolve the missing comparison design.", "evidence_ids": ["result-0001"]}
            elif stage == "orchestrator_decision_review": value = consistent_review()
            else: self.fail("Resume unexpectedly executed earlier work: " + stage)
            return own_answer(value, schema)
        session = self.original/"resume-pending-review"
        session.mkdir()
        resumed_record, _, _ = resume_result(session, self.original, self.memory, resumed,
            data_identity={"sha256": "fixture"})
        self.assertEqual(calls, ["empiricist_assessment", "orchestrator_scientific_review", "hypothesis_decision", "orchestrator_decision_review"])
        self.assertEqual(resumed_record["status"], "partial", resumed_record)
        self.assertEqual(resumed_record["scientific_review_status"], "blocked")
        self.assertFalse(resumed_record["runner_reexecuted"])
