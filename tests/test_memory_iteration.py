"""Run only in a reviewed Scheduler allocation; fixtures contain no research data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import os
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import memory_iteration as runner


class MemoryIterationTests(unittest.TestCase):
    def test_attempts_use_the_actual_inference_deadline_not_an_extra_two_minute_stop(self):
        self.assertTrue(runner.may_start_attempt("C", 200, 3500, 3600))
        self.assertFalse(runner.may_start_attempt("C", 200, 3555, 3600))

    def test_main_runner_repeats_fresh_oneshots_without_receipt_or_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "synthetic.csv"
            dataset.write_text("fixture")
            trajectory = root / "trajectory"
            trajectory.mkdir()
            calls, fits = [], []
            def execute(session, index, arm, memory, model, analysis_runner, **kwargs):
                calls.append((session, index, kwargs["query"], kwargs["previous_plan"]))
                analysis_runner(session, {"action": "same"})
                return ({"attempt": index, "status": "completed", "decision_status": "succeeded"}, {},
                        {"next_inquiry": "SENTINEL_FROM_PREVIOUS_EPISODE"})
            def fit(session, action):
                fits.append(session)
                return {"returncode": 0}
            args = SimpleNamespace(condition="A", round=1, embedding_backend="lexical", retrieval_strategy="ppr",
                backend="codex", codex_model="fixture", device="cpu", runtime=Path("/fixture"), cuda_toolkit=None,
                budget_seconds=900, max_model_calls=None, repeat_one_shot=True)
            with patch.dict(os.environ, {"JOB_ID": "synthetic-job", "NSLOTS": "1"}), \
                 patch.object(runner, "ITERATIONS", root / "outputs"), patch.object(runner, "DATA", dataset), \
                 patch.object(runner, "BENCHMARK_CONDITIONS", ("A", "B", "C")), \
                 patch.object(runner, "SOURCE_FILES", ()), patch.object(runner, "file_sha256", return_value="fixture"), \
                 patch.object(runner, "execute_attempt", side_effect=execute), patch.object(runner, "run_analysis", side_effect=fit), \
                 patch.object(runner, "may_start_attempt", side_effect=lambda condition, index, elapsed, budget, repeat: index <= 2):
                runner.run_round(args, trajectory, time.monotonic())
            self.assertEqual(len(calls), 2)
            self.assertEqual([call[1] for call in calls], [1, 1])
            self.assertTrue(all(call[2] == runner.QUESTION and call[3] is None for call in calls))
            self.assertNotEqual(calls[0][0].parent, calls[1][0].parent)
            self.assertEqual(len(fits), 2)
            report = json.loads((root / "outputs/A/synthetic-job/report.json").read_text())
            self.assertEqual([row["episode"] for row in report["attempts"]], [1, 2])

    def test_runtime_failure_stops_round_but_scientific_attempt_failure_can_continue(self):
        for failure in ("OSError", "PermissionError", "FileNotFoundError"):
            self.assertEqual(runner.fatal_attempt_stop_reason({"failure_type": failure}),
                             "filesystem_failure_no_retry")
        for failure in ("ModuleNotFoundError", "ImportError"):
            self.assertEqual(runner.fatal_attempt_stop_reason({"failure_type": failure}),
                             "environment_dependency_no_retry")
        self.assertEqual(runner.fatal_attempt_stop_reason({"failure_type": "RuntimeIsolationError"}),
                         "runtime_failed_closed_no_retry")
        self.assertEqual(runner.fatal_attempt_stop_reason({"failure_type": "ContextCapacityError"}),
                         "context_capacity_no_retry")
        self.assertIsNone(runner.fatal_attempt_stop_reason({"failure_type": "PlanRepairError"}))
        self.assertEqual(runner.fatal_attempt_stop_reason({"failure_type": "ModelCallBudgetExceeded"}),
                         "model_call_budget_reached")
        for failure in ("DataInsufficientDesign", "UnsupportedModelDesign"):
            with self.subTest(failure=failure):
                self.assertIsNone(runner.fatal_attempt_stop_reason({"failure_type": failure}))
        self.assertIsNone(runner.fatal_attempt_stop_reason({"failure_type": "PlanValidationError"}))
        self.assertIsNone(runner.fatal_attempt_stop_reason({"failure_type": None}))

    def test_codex_dependency_preflight_checks_schema_validator(self):
        with patch.object(runner.importlib, "import_module", side_effect=ModuleNotFoundError("jsonschema")) as loader:
            with self.assertRaises(ModuleNotFoundError):
                runner.verify_codex_dependencies()
            loader.assert_called_once_with("jsonschema")

    def test_missing_dependency_stops_full_attempt_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "synthetic.csv"
            dataset.write_text("fixture")
            trajectory = root / "trajectory"
            trajectory.mkdir()
            args = SimpleNamespace(condition="A", round=1, embedding_backend="lexical", retrieval_strategy="ppr",
                backend="codex", codex_model="fixture", device="cpu", runtime=Path("/fixture"), cuda_toolkit=None,
                budget_seconds=900, max_model_calls=None, repeat_one_shot=True)
            failure = ({"attempt": 1, "status": "failed", "analysis_status": "not_started",
                        "decision_status": "not_started", "failure_type": "ModuleNotFoundError"}, None, None)
            with patch.dict(os.environ, {"JOB_ID": "synthetic-job", "NSLOTS": "1"}), \
                 patch.object(runner, "ITERATIONS", root / "outputs"), patch.object(runner, "DATA", dataset), \
                 patch.object(runner, "BENCHMARK_CONDITIONS", ("A", "B", "C")), \
                 patch.object(runner, "SOURCE_FILES", ()), patch.object(runner, "file_sha256", return_value="fixture"), \
                 patch.object(runner, "execute_attempt", return_value=failure) as execute:
                runner.run_round(args, trajectory, time.monotonic())
            report = json.loads((root / "outputs/A/synthetic-job/report.json").read_text())
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(report["stop_reason"], "environment_dependency_no_retry")
            self.assertEqual(report["status"], "failed")

    def test_context_overflow_receipt_is_bounded_and_marks_attempt_not_started(self):
        record = runner.context_capacity_attempt(7, "hypothesis", {
            "rendered_prompt_tokens": 33313, "requested_generation_tokens": 768,
            "context_limit": 32768, "safety_margin_tokens": 32, "required_tokens": 34113,
            "capacity_status": "exceeded", "raw_prompt": "must not be retained"})
        self.assertEqual(record["status"], "needs-attention")
        self.assertEqual(record["failure_stage"], "hypothesis")
        self.assertEqual(record["context"]["rendered_prompt_tokens"], 33313)
        self.assertNotIn("raw_prompt", record["context"])
        self.assertEqual(runner.fatal_attempt_stop_reason(record), "context_capacity_no_retry")

    def test_worker_deadline_reserves_native_cleanup_and_report_time(self):
        for budget, deadline in ((900, 1855), (1800, 2755)):
            with self.subTest(budget=budget), patch.object(runner, "PrivateInference") as factory:
                worker = runner.private_worker(Path("/reviewed/job/inference"), 1000, budget)
            self.assertIs(worker, factory.return_value)
            self.assertEqual(factory.call_args.args[0], Path("/reviewed/job/inference"))
            self.assertEqual(factory.call_args.args[-1], deadline)
        self.assertEqual(runner.INFERENCE_DEADLINE_RESERVE_SECONDS, 45)

    def test_native_character_limit_metadata_is_not_confused_with_token_limit(self):
        record = runner.context_capacity_attempt(20, "empiricist_assessment", {
            "prompt_chars": 1048577, "prompt_bytes": 1048577,
            "transport_input_char_limit": 1048576, "completion_submitted": False,
            "raw_prompt": "private input"})
        self.assertEqual(record["context"]["transport_input_char_limit"], 1048576)
        self.assertEqual(record["context"]["prompt_chars"], 1048577)
        self.assertFalse(record["context"]["completion_submitted"])
        self.assertIsNone(record["context"]["context_limit"])
        self.assertNotIn("raw_prompt", record["context"])

    def test_reviewed_path_rejects_relative_parent_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "trajectories"
            with patch.object(runner, "TRAJECTORIES", base):
                self.assertEqual(runner.reviewed_trajectory(base / "agreed"), base / "agreed")
                for path in ("relative", base, base / ".." / "evaluator"):
                    with self.assertRaises(ValueError):
                        runner.reviewed_trajectory(path)

    def test_symlink_cannot_escape_reviewed_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "trajectories"
            base.mkdir()
            (base / "escape").symlink_to(Path(directory), target_is_directory=True)
            with patch.object(runner, "TRAJECTORIES", base), self.assertRaises(ValueError):
                runner.reviewed_trajectory(base / "escape")

    def test_fresh_or_censored_attempts_have_unique_monotonic_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(runner.load_progress(root)["next_attempt"], 1)
            (root / "attempt-0001").mkdir()
            (root / "attempt-0004").mkdir()
            (root / "attempt-bad").mkdir()
            result = runner.load_progress(root)
            self.assertEqual(result["next_attempt"], 5)
            self.assertEqual(result["next_inquiry"], runner.QUESTION)
            self.assertIsNone(result["previous_plan"])

    def _fixture(self, root):
        session = root / "attempt-0001"
        session.mkdir()
        for filename, value in (("plan.json", {"hypothesis": "fixture"}),
            ("attempt.json", {"attempt": 1, "hypothesis_id": "hypothesis-0001",
                "decision_id": "decision-0001", "decision_status": "succeeded"}),
            ("decision.json", {"next_inquiry": "own scientific inquiry", "action": "maintain"})):
            (session / filename).write_text(json.dumps(value))
        references = [runner.source_reference(session / filename, root)
                      for filename in ("plan.json", "attempt.json", "decision.json")]
        runner.publish_cursor(root, 1, "fixture-job", *references)
        return session

    def test_resume_uses_bound_own_decision_not_qa_or_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            (root / "qa.json").write_text(json.dumps({"next_inquiry": "FORBIDDEN evaluator hint"}))
            result = runner.load_progress(root)
            self.assertEqual(result["next_inquiry"], "own scientific inquiry")
            self.assertEqual(result["previous_graph"]["decision_id"], "decision-0001")
            self.assertEqual(result["next_attempt"], 2)

    def test_modified_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self._fixture(root)
            (session / "decision.json").write_text('{"next_inquiry":"tampered"}')
            with self.assertRaises(ValueError):
                runner.load_progress(root)

    def test_cursor_cannot_expand_with_feedback_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            path = root / "CURRENT.json"
            cursor = json.loads(path.read_text())
            cursor["qa_feedback"] = "FORBIDDEN evaluator feedback"
            path.write_text(json.dumps(cursor))
            with self.assertRaises(ValueError):
                runner.load_progress(root)

    def test_cursor_sources_cannot_escape_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trajectory"
            root.mkdir()
            outside = Path(directory) / "qa.json"
            outside.write_text("{}")
            with self.assertRaises(ValueError):
                runner.read_reference({"path": "../qa.json", "sha256": "fake"}, root)

    def test_initial_weights_digest_is_reused_only_with_unchanged_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            weights, binary = Path(directory) / "weights", Path(directory) / "server"
            weights.write_bytes(b"fixture weights")
            binary.write_bytes(b"fixture binary")
            initial = runner.pinned_model_identity(weights, binary)
            with patch.object(runner, "file_sha256", wraps=runner.file_sha256) as digest:
                self.assertEqual(runner.pinned_model_identity(weights, binary, initial), initial)
                self.assertEqual([call.args[0] for call in digest.call_args_list], [binary])
            weights.write_bytes(b"changed fixture weights")
            with self.assertRaises(ValueError):
                runner.pinned_model_identity(weights, binary, initial)

    def test_attempt_checkpoint_resumes_latest_decision_without_final_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "attempt-0002"
            session.mkdir()
            for filename, value in (("plan.json", {"hypothesis": "Latest own hypothesis."}),
                ("attempt.json", {"attempt": 2, "hypothesis_id": "hypothesis-0002",
                    "decision_id": "decision-0002", "decision_status": "succeeded"}),
                ("decision.json", {"next_inquiry": "Latest own inquiry.", "action": "maintain"})):
                (session / filename).write_text(json.dumps(value))
            refs = [runner.source_reference(session / name, root)
                    for name in ("plan.json", "attempt.json", "decision.json")]
            runner.publish_cursor(root, 4, "checkpoint-job", *refs, checkpoint_only=True)
            self.assertFalse((root / "round-04-checkpoint-job.cursor.json").exists())
            progress = runner.load_progress(root)
            self.assertEqual(progress["next_inquiry"], "Latest own inquiry.")
            self.assertEqual(progress["previous_graph"]["decision_id"], "decision-0002")
            self.assertEqual(progress["cursor"]["round"], 4)
            (session / "decision.json").write_text('{"next_inquiry":"changed"}')
            with self.assertRaises(ValueError):
                runner.load_progress(root)

    def test_runtime_shared_library_change_is_not_hidden_by_unchanged_main_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, binary, library = root / "weights", root / "llama-server", root / "libggml-cuda.so"
            weights.write_bytes(b"pinned weights")
            binary.write_bytes(b"unchanged main router")
            library.write_bytes(b"VMM allocator")
            initial = runner.pinned_model_identity(weights, binary)
            self.assertIn("libggml-cuda.so", initial["runtime_library_sha256"])
            library.write_bytes(b"non-VMM allocator")
            with self.assertRaises(ValueError):
                runner.pinned_model_identity(weights, binary, initial)


if __name__ == "__main__":
    unittest.main()
