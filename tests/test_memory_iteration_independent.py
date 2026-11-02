"""Synthetic resume/isolation checks; execute only under reviewed SGE."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import memory_iteration as runner


class IndependentMemoryIterationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.trajectories = self.root / "trajectories"
        self.trajectories.mkdir()
        self.trajectory = self.trajectories / "own-A"
        self.trajectory.mkdir()
        self.paths = patch.object(runner, "TRAJECTORIES", self.trajectories)
        self.paths.start()

    def tearDown(self):
        self.paths.stop()
        self.tmp.cleanup()

    def write(self, relative, value):
        path = self.trajectory / relative
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(value))
        return path

    def cursor(self, attempt=1):
        prefix = f"attempt-{attempt:04d}/"
        plan = self.write(prefix + "plan.json", {"hypothesis": "Own hypothesis.", "analysis": {}})
        state = self.write(prefix + "attempt.json", {"attempt": attempt, "hypothesis_id": f"hypothesis-{attempt:04d}",
            "decision_status": "succeeded", "decision_id": f"decision-{attempt:04d}"})
        decision = self.write(prefix + "decision.json", {"action": "maintain", "next_inquiry": "Own next question.",
            "evidence_ids": [f"result-{attempt:04d}"]})
        return {"format": "memory-iteration-cursor-v1", "round": 1, "job_id": "QA_SYNTHETIC",
            "last_plan": runner.source_reference(plan, self.trajectory),
            "last_attempt": runner.source_reference(state, self.trajectory),
            "last_decision": runner.source_reference(decision, self.trajectory),
            "last_decision_attempt": runner.source_reference(state, self.trajectory)}

    def test_reviewed_path_rejects_parent_relative_and_symlink_escape(self):
        for path in ("relative", self.trajectories, self.root / "outside"):
            with self.subTest(path=str(path)), self.assertRaises(ValueError):
                runner.reviewed_trajectory(path)
        outside = self.root / "outside"
        outside.mkdir()
        (self.trajectories / "escape").symlink_to(outside)
        with self.assertRaises(ValueError):
            runner.reviewed_trajectory(self.trajectories / "escape")
        self.assertEqual(runner.reviewed_trajectory(self.trajectory), self.trajectory)

    def test_fresh_trajectory_has_no_inherited_query_or_graph(self):
        progress = runner.load_progress(self.trajectory)
        self.assertEqual(progress["next_attempt"], 1)
        self.assertEqual(progress["next_inquiry"], runner.QUESTION)
        self.assertIsNone(progress["previous_plan"])
        self.assertIsNone(progress["previous_graph"])
        self.assertEqual(progress["cursor"], {})

    def test_partial_attempt_directories_reserve_global_ids(self):
        (self.trajectory / "attempt-0002").mkdir()
        (self.trajectory / "attempt-0011").mkdir()
        (self.trajectory / "attempt-IGNORE_QA").mkdir()
        self.assertEqual(runner.load_progress(self.trajectory)["next_attempt"], 12)

    def test_resume_only_own_hash_verified_plan_and_successful_decision(self):
        cursor = self.cursor(3)
        self.write("CURRENT.json", cursor)
        progress = runner.load_progress(self.trajectory)
        self.assertEqual(progress["next_attempt"], 4)
        self.assertEqual(progress["next_inquiry"], "Own next question.")
        self.assertEqual(progress["previous_graph"], {"hypothesis_id": "hypothesis-0003", "decision_id": "decision-0003"})
        self.write("attempt-0003/decision.json", {"next_inquiry": "TAMPERED_QA_HINT"})
        with self.assertRaises(ValueError):
            runner.load_progress(self.trajectory)

    def test_resume_rejects_qa_or_evaluator_cursor_channel(self):
        for field in ("qa_feedback", "evaluator_reference", "target_hypothesis", "next_query"):
            cursor = self.cursor()
            cursor[field] = "HIDDEN_TARGET"
            self.write("CURRENT.json", cursor)
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.load_progress(self.trajectory)

    def test_qa_json_cannot_be_substituted_as_attempt_record(self):
        cursor = self.cursor()
        qa = self.write("qa/report.json", {"attempt": 1, "hypothesis_id": "hypothesis-0001"})
        cursor["last_attempt"] = runner.source_reference(qa, self.trajectory)
        self.write("CURRENT.json", cursor)
        with self.assertRaises(ValueError):
            runner.load_progress(self.trajectory)

    def test_qa_json_cannot_be_substituted_as_next_decision(self):
        cursor = self.cursor()
        qa = self.write("qa/decision.json", {"next_inquiry": "HIDDEN_TARGET"})
        cursor["last_decision"] = runner.source_reference(qa, self.trajectory)
        self.write("CURRENT.json", cursor)
        with self.assertRaises(ValueError):
            runner.load_progress(self.trajectory)

    def test_failed_decision_never_becomes_next_question(self):
        cursor = self.cursor()
        self.write("attempt-0001/attempt.json", {"attempt": 1, "hypothesis_id": "hypothesis-0001",
            "decision_status": "failed", "decision_id": "decision-0001"})
        cursor["last_attempt"] = runner.source_reference(self.trajectory / "attempt-0001/attempt.json", self.trajectory)
        self.write("CURRENT.json", cursor)
        with self.assertRaises(ValueError):
            runner.load_progress(self.trajectory)

    def test_resuming_one_trajectory_does_not_seed_fresh_other_trajectory(self):
        self.write("CURRENT.json", self.cursor(7))
        fresh = self.trajectories / "fresh-B"
        fresh.mkdir()
        self.assertEqual(runner.load_progress(self.trajectory)["next_inquiry"], "Own next question.")
        self.assertEqual(runner.load_progress(fresh)["next_inquiry"], runner.QUESTION)
        self.assertIsNone(runner.load_progress(fresh)["previous_graph"])

    def test_cursor_archive_is_append_only(self):
        cursor = self.cursor()
        args = (self.trajectory, 1, "QA_SYNTHETIC", cursor["last_plan"], cursor["last_attempt"], cursor["last_decision"])
        runner.publish_cursor(*args)
        self.assertEqual(json.loads((self.trajectory / "CURRENT.json").read_text()), cursor)
        with self.assertRaises(FileExistsError):
            runner.publish_cursor(*args)

    def test_pinned_identity_hashes_actual_file_content(self):
        source = self.trajectory / "synthetic-identity.txt"
        source.write_text("abc")
        self.assertEqual(runner.file_sha256(source), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        source.write_text("different")
        self.assertNotEqual(runner.file_sha256(source), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
