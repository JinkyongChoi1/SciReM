"""Post-result resumption must preserve immutable originals and never refit."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from four_role_discovery import execute_four_role_attempt
from four_role_resume import pending_result, resume_result as _resume_result, verify_reused_result
from analysis_reuse import VerifiedAnalysisCache
from memory_iteration import load_progress, publish_cursor, source_reference
from private_inference import RuntimeIsolationError, ContextCapacityError
from research_memory.two_role import TwoRoleMemory
from research_memory.orchestration import FourRoleOrchestrator
from test_four_role import hypothesis, plan, result, answer_envelope_adapter
from two_role_discovery import QUESTION


def resume_result(*args, **kwargs):
    args = list(args)
    if len(args) > 3 and callable(args[3]):
        args[3] = answer_envelope_adapter(args[3])
    elif callable(kwargs.get("model")):
        kwargs["model"] = answer_envelope_adapter(kwargs["model"])
    return _resume_result(*args, **kwargs)


class PostResultResumeTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Reviewed SGE allocation required")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config.json").write_text(json.dumps({"question": QUESTION}))
        self.original = self.root / "attempt-0001"
        self.original.mkdir()
        self.memory = TwoRoleMemory(self.root)
        self.runner_calls = []
        def model(label, prompt, schema, tokens, seed):
            if label == "hypothesis":
                return hypothesis()
            if label == "empiricist_plan":
                return plan()
            error = RuntimeIsolationError("fixture deadline")
            error.deadline_exhausted = True
            raise error
        def runner(session, action):
            self.runner_calls.append(action)
            return result(action)
        self.record, self.plan, _ = execute_four_role_attempt(self.original, 1, self.memory,
            answer_envelope_adapter(model), runner, data_identity={"sha256": "fixture"})
        self.assertEqual(self.record["status"], "censored", self.record)
        (self.original / "attempt.json").write_text(json.dumps(self.record))
        self.refs = [source_reference(self.original / name, self.root)
                     for name in ("plan.json", "attempt.json")]
        publish_cursor(self.root, 1, "old-job", *self.refs, None)
        self.cursor = json.loads((self.root / "CURRENT.json").read_text())
        self.session = self.original / "resume-new-job"
        self.session.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, label, prompt, schema, tokens, seed):
        self.calls.append(label)
        if label == "empiricist_assessment":
            return {"execution_assessment": "Only association evaluated.",
                    "diagnostic_action": "additional_test"}
        if label == "hypothesis_decision":
            return {"action": "additional_test", "basis_keywords": "Result does not settle rival.",
                    "next_inquiry": "Check the rival explanation.", "evidence_ids": ["result-0001"]}
        raise AssertionError("Resume repeated an earlier role: " + label)

    def test_reused_result_source_is_verified_and_tampering_rejected(self):
        cache = VerifiedAnalysisCache({"sha256": "fixture"})
        action = {"family": "logit", "terms": ["typicality"]}
        source = self.root / "attempt-0002"
        source.mkdir()
        first = cache.run(source, action, lambda session, requested: result(requested))
        reused = cache.run(self.original, action, lambda *_: self.fail("unexpected refit"))
        verify_reused_result(reused, self.root, {"sha256": "fixture"})
        self.assertEqual({k: v for k, v in reused.items() if k != "execution_reuse"}, first)
        next(source.glob("executed-analysis-*.json")).write_text("{}")
        with self.assertRaisesRegex(ValueError, "source changed"):
            verify_reused_result(reused, self.root, {"sha256": "fixture"})

    def test_post_result_resume_only_calls_handbacks_and_preserves_sources(self):
        original_bytes = {p.name: p.read_bytes() for p in self.original.iterdir() if p.is_file()}
        self.assertEqual(pending_result(self.root, self.cursor), self.original)
        self.calls = []
        record, _, decision = resume_result(self.session, self.original, self.memory, self.model,
                                            data_identity={"sha256": "fixture"})
        self.assertEqual(record["status"], "completed", record)
        self.assertEqual(self.calls, ["empiricist_assessment", "hypothesis_decision"])
        self.assertEqual(len(self.runner_calls), 1)
        self.assertFalse(record["runner_reexecuted"])
        self.assertEqual(self.memory.nodes["decision-0001"]["payload"], decision)
        for name, content in original_bytes.items():
            self.assertEqual((self.original / name).read_bytes(), content)
        (self.session / "attempt.json").write_text(json.dumps(record))
        publish_cursor(self.root, 2, "new-job", *self.refs,
                       source_reference(self.session / "decision.json", self.root))
        progress = load_progress(self.root)
        self.assertEqual(progress["next_attempt"], 2)
        self.assertEqual(progress["next_inquiry"], decision["next_inquiry"])
        self.assertIsNone(pending_result(self.root, progress["cursor"]))
        workflow = json.loads((self.session / "orchestrator.json").read_text())
        self.assertEqual([e.get("execution") for e in workflow["events"][:3]],
                         ["restored_receipt_only"] * 3)

    def test_changed_original_result_is_rejected_before_model_call(self):
        path = self.original / "tool-result.json"
        changed = json.loads(path.read_text())
        changed["estimates"][0]["p_value"] = ".001"
        path.write_text(json.dumps(changed))
        self.calls = []
        with self.assertRaises(ValueError):
            resume_result(self.session, self.original, self.memory, self.model,
                          data_identity={"sha256": "fixture"})
        self.assertEqual(self.calls, [])

    def test_changed_question_and_data_are_rejected(self):
        self.calls = []
        for kwargs in ({"data_identity": {"sha256": "fixture"}, "research_question": "Different RQ"},
                       {"data_identity": {"sha256": "other"}}):
            with self.assertRaises(ValueError):
                resume_result(self.session, self.original, self.memory, self.model, **kwargs)
        self.assertEqual(self.calls, [])

    def test_changed_cursor_bound_attempt_is_rejected(self):
        (self.original / "attempt.json").write_text("{}")
        with self.assertRaises(ValueError):
            pending_result(self.root, self.cursor)

    def test_timeout_preserves_result_and_does_not_publish_decision(self):
        def timeout(*args):
            error = RuntimeIsolationError("fixture deadline")
            error.deadline_exhausted = True
            raise error
        record, _, decision = resume_result(self.session, self.original, self.memory, timeout,
                                            data_identity={"sha256": "fixture"})
        self.assertEqual(record["status"], "censored")
        self.assertEqual(record["analysis_status"], "succeeded")
        self.assertEqual(decision, {})
        self.assertNotIn("decision-0001", self.memory.nodes)
        self.assertEqual(len(self.runner_calls), 1)

    def test_on_demand_context_capacity_stops_resume_after_assessment_or_decision(self):
        for failed_stage in ("empiricist_assessment", "hypothesis_decision"):
            with self.subTest(stage=failed_stage):
                session = self.original / ("resume-capacity-" + failed_stage)
                session.mkdir()
                original_bytes = {p.name: p.read_bytes() for p in self.original.iterdir() if p.is_file()}
                calls = []

                def model(label, prompt, schema, tokens, seed):
                    calls.append(label)
                    if label == "empiricist_assessment":
                        assessment = {"execution_assessment": "Association result needs context.",
                                      "diagnostic_action": "additional_test"}
                        if failed_stage == label:
                            if len(calls) == 1:
                                return {"action": "retrieve_memory", "query": "result-0001", "answer": None}
                            raise ContextCapacityError("Retrieved assessment context exceeds capacity")
                        return assessment
                    if label == "hypothesis_decision":
                        if failed_stage == label and calls.count(label) == 1:
                            return {"action": "retrieve_memory", "query": "result-0001", "answer": None}
                        if failed_stage == label:
                            raise ContextCapacityError("Retrieved decision context exceeds capacity")
                    raise AssertionError(label)

                record, _, _ = resume_result(session, self.original, self.memory, answer_envelope_adapter(model),
                                             data_identity={"sha256": "fixture"})
                self.assertEqual(record["status"], "needs-attention", record)
                self.assertEqual(record["failure_type"], "ContextCapacityError")
                self.assertEqual(record["failure_domain"], "context_capacity")
                self.assertTrue((session / "memory-tool-01.json").exists())
                self.assertEqual(len(record["on_demand_retrievals"]), 1)
                self.assertEqual(len(self.runner_calls), 1)
                self.assertEqual({p.name: p.read_bytes() for p in self.original.iterdir() if p.is_file()},
                                 original_bytes)
                (session / "attempt.json").write_text(json.dumps(record))
                self.assertIsNone(pending_result(self.root, self.cursor))

    def test_saved_assessment_resumes_only_decision(self):
        empirical = {"execution_assessment": "Only association evaluated.", "diagnostic_action": "additional_test"}
        (self.original / "empiricist-assessment.json").write_text(json.dumps(empirical))
        old = json.loads((self.original / "orchestrator.json").read_text())
        flow = FourRoleOrchestrator(self.original, 1, research_question=QUESTION, inquiry=QUESTION)
        for event in old["events"]:
            flow.route(event["stage"], event["actor"], event["artifact"])
        facts = json.loads((self.original / "current-result-facts.json").read_text())
        flow.route("decision", "hypothesis-agent", {"result_id": "result-0001",
            "result_facts": facts, "empiricist_assessment": empirical})
        flow.write(self.original / "orchestrator.json", status="censored")
        self.calls = []
        record, _, _ = resume_result(self.session, self.original, self.memory, self.model,
                                     data_identity={"sha256": "fixture"})
        self.assertEqual(record["status"], "completed", record)
        self.assertEqual(self.calls, ["hypothesis_decision"])

    def censor_decision(self, session):
        self.calls = []
        def model(label, *args):
            if label == "hypothesis_decision":
                self.calls.append(label)
                error = RuntimeIsolationError("fixture decision deadline")
                error.deadline_exhausted = True
                raise error
            return self.model(label, *args)
        record, _, _ = resume_result(session, self.original, self.memory, model,
                                     data_identity={"sha256": "fixture"})
        self.assertEqual(record["status"], "censored", record)
        self.assertEqual(record["decision_status"], "censored")

    def test_repeated_resume_timeouts_reuse_assessment_without_refit(self):
        original_record = (self.original / "attempt.json").read_bytes()
        self.censor_decision(self.session)
        self.assertEqual(self.calls, ["empiricist_assessment", "hypothesis_decision"])
        second = self.original / "resume-second"
        second.mkdir()
        self.censor_decision(second)
        self.assertEqual(self.calls, ["hypothesis_decision"])
        third = self.original / "resume-third"
        third.mkdir()
        self.calls = []
        record, _, _ = resume_result(third, self.original, self.memory, self.model,
                                     data_identity={"sha256": "fixture"})
        self.assertEqual(record["status"], "completed", record)
        self.assertEqual(self.calls, ["hypothesis_decision"])
        self.assertEqual(len(self.runner_calls), 1)
        self.assertEqual((self.original / "attempt.json").read_bytes(), original_record)

    def test_changed_ancestor_assessment_rejected_across_resume_chain(self):
        self.censor_decision(self.session)
        second = self.original / "resume-second"
        second.mkdir()
        self.censor_decision(second)
        path = self.session / "empiricist-assessment.json"
        changed = json.loads(path.read_text())
        changed["execution_assessment"] = "Altered earlier interpretation."
        path.write_text(json.dumps(changed))
        third = self.original / "resume-third"
        third.mkdir()
        self.calls = []
        with self.assertRaises(ValueError):
            resume_result(third, self.original, self.memory, self.model,
                          data_identity={"sha256": "fixture"})
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
