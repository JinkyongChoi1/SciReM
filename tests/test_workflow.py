from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from research_memory.memory import submit_candidate, build_snapshot, retrieve
from research_memory import workflow as flow
from research_memory.viewer import render_dashboard


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.record = {"memory_id": "prior", "agent_id": "agent-a", "track_id": "FD",
                       "text": "Competition prior failed design", "aliases": ["competition"], "source": "synthetic fixture"}
        submit_candidate(self.root, self.record)
        build_snapshot(self.root)
        self.task = flow.create_task(self.root, "FD", "competition", "agent-b")

    def tearDown(self):
        self.temp.cleanup()

    def ready(self):
        trace = retrieve(self.root, "competition", "FD")
        flow.save_retrieval(self.root, trace)
        flow.attach_retrieval(self.root, self.task["id"], trace["request_id"])
        return trace

    def test_lifecycle_requires_both_receipts(self):
        with self.assertRaises(ValueError):
            flow.complete_task(self.root, self.task["id"])
        trace = self.ready()
        with self.assertRaises(ValueError):
            flow.complete_task(self.root, self.task["id"])
        record = dict(self.record, memory_id="next", agent_id="agent-b")
        attempt = flow.record_attempt(self.root, self.task["id"], record, trace["selected_ids"], "revise", "synthetic outcome")
        self.assertEqual(attempt["snapshot_id"], trace["snapshot_id"])
        task = flow.complete_task(self.root, self.task["id"])
        self.assertEqual(task["status"], "completed")

    def test_reject_unseen_memory_and_wrong_question(self):
        trace = retrieve(self.root, "unrelated", "FD")
        flow.save_retrieval(self.root, trace)
        with self.assertRaises(ValueError):
            flow.attach_retrieval(self.root, self.task["id"], trace["request_id"])
        self.ready()
        record = dict(self.record, memory_id="next", agent_id="agent-b")
        with self.assertRaises(ValueError):
            flow.record_attempt(self.root, self.task["id"], record, ["never-delivered"], "revise", "synthetic outcome")

    def test_completion_rechecks_retrieval_and_candidate_integrity(self):
        trace = self.ready()
        record = dict(self.record, memory_id="next", agent_id="agent-b")
        attempt = flow.record_attempt(self.root, self.task["id"], record, trace["selected_ids"], "revise", "synthetic outcome")
        path = self.root / "retrievals" / trace["request_id"] / "brief.md"
        original = path.read_text()
        path.write_text("tampered evidence")
        with self.assertRaises(ValueError):
            flow.complete_task(self.root, self.task["id"])
        path.write_text(original)
        candidate = self.root / attempt["candidate"]
        candidate.write_text("{}")
        with self.assertRaises(ValueError):
            flow.complete_task(self.root, self.task["id"])

    def test_decision_verbatim_not_general_approval(self):
        text = "Design 수정에는 동의하지만 GPU 사용은 승인하지 않음."
        decision = flow.record_decision(self.root, self.task["id"], text)
        self.assertEqual(decision["text"], text)
        self.assertEqual(decision["authority"], "user_statement")

    def test_metrics_first_faithful_and_failures(self):
        trial = flow.create_trial(self.root, self.task["id"], "shared-ppr", ["table-1", "table-2"])
        base = datetime.fromisoformat(trial["created_at"])
        events = [
            {"kind": "submitted", "run_id": "run1", "at": (base + timedelta(seconds=2)).isoformat()},
            {"kind": "started", "run_id": "run1", "at": (base + timedelta(seconds=12)).isoformat()},
            {"kind": "error", "run_id": "run1", "at": (base + timedelta(seconds=18)).isoformat()},
            {"kind": "finished", "run_id": "run1", "at": (base + timedelta(seconds=22)).isoformat()},
            {"kind": "reference_check", "checks": {"table-1": True, "table-2": False}, "at": (base + timedelta(seconds=30)).isoformat()},
            {"kind": "reference_check", "checks": {"table-1": True, "table-2": True}, "at": (base + timedelta(seconds=40)).isoformat()},
        ]
        for event in events:
            result = flow.record_trial_event(self.root, trial["id"], dict(event, source="synthetic test event"))
        self.assertEqual(result["metrics"]["errors"], 1)
        self.assertEqual(result["metrics"]["queue_time"], 10)
        self.assertEqual(result["metrics"]["compute_time"], 10)
        self.assertEqual(result["metrics"]["time_to_first_faithful_seconds"], 40)
        self.assertEqual(result["metrics"]["fidelity"], 1)
        self.assertIsNone(result["metrics"]["mechanism_validity"])

    def test_incomplete_reference_check_not_success(self):
        trial = flow.create_trial(self.root, self.task["id"], "shared-ppr", ["a", "b"])
        with self.assertRaises(ValueError):
            flow.record_trial_event(self.root, trial["id"], {"kind": "reference_check", "checks": {"a": True}, "source": "fixture"})

    def test_duplicate_stage_and_pretrial_events_rejected(self):
        trial = flow.create_trial(self.root, self.task["id"], "shared-ppr", ["a"])
        base = datetime.fromisoformat(trial["created_at"])
        event = {"kind": "started", "run_id": "run1", "source": "synthetic job log", "at": (base + timedelta(seconds=10)).isoformat()}
        flow.record_trial_event(self.root, trial["id"], event)
        with self.assertRaises(ValueError):
            flow.record_trial_event(self.root, trial["id"], dict(event, at=(base + timedelta(seconds=20)).isoformat()))
        with self.assertRaises(ValueError):
            flow.record_trial_event(self.root, trial["id"], {"kind": "reference_check", "checks": {"a": True}, "source": "synthetic check", "at": (base - timedelta(seconds=1)).isoformat()})

    def test_dashboard_cannot_execute_source_text(self):
        trace = self.ready()
        trace["query"] = "</script><script>alert(1)</script>"
        trace["nodes"][0]["text"] = "<img src=x onerror=alert(1)>"
        html = render_dashboard(trace)
        self.assertNotIn("<img src=x", html)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("\\u003c", html)
        self.assertNotIn("fetch(", html)
        self.assertIn("snapshot_id", html)


if __name__ == "__main__":
    unittest.main()
