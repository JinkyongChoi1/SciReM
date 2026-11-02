"""Human review receipts are authenticated annotations, not scientific execution."""
import json
import stat
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from research_memory.live import ArtifactController, _recorded_activity, create_server
import test_memory_inventory as inventory_fixture


class MemoryReviewTests(unittest.TestCase):
    setUp = inventory_fixture.MemoryInventoryTests.setUp
    publish = inventory_fixture.MemoryInventoryTests.publish

    def test_review_is_immutable_private_and_does_not_modify_research(self):
        folder = self.publish()
        report_before, graph_before = self.report.read_bytes(), (folder / "graph.json").read_bytes()
        controller = ArtifactController(self.report, self.trajectory)
        first = controller.memory_review("h-1", "approve", "Check recorded evidence")
        second = controller.memory_review("h-1", "request_change", "Add a sensitivity test")
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(first["application_status"], "not_applied")
        self.assertEqual(self.report.read_bytes(), report_before)
        self.assertEqual((folder / "graph.json").read_bytes(), graph_before)
        fresh = ArtifactController(self.report, self.trajectory).state()
        self.assertEqual(len(fresh["human_reviews"]), 2)
        self.assertTrue(fresh["capabilities"]["memory_review"])
        for receipt in fresh["human_reviews"]:
            path = self.trajectory / "human-decisions" / (receipt["id"] + ".json")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(receipt["status"], "recorded")

    def test_review_rejects_unknown_targets_and_invalid_actions(self):
        self.publish()
        controller = ArtifactController(self.report, self.trajectory)
        for memory_id, action in (("other-arm-memory", "approve"), ("h-1", "execute")):
            with self.assertRaises(ValueError):
                controller.memory_review(memory_id, action, "")
        self.assertFalse((self.trajectory / "human-decisions").exists())

    def test_review_cannot_follow_symlinked_output_directory(self):
        self.publish()
        outside = self.root / "outside"
        outside.mkdir()
        (self.trajectory / "human-decisions").symlink_to(outside, target_is_directory=True)
        controller = ArtifactController(self.report, self.trajectory)
        with self.assertRaises(ValueError):
            controller.memory_review("h-1", "approve", "")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse(controller.state()["capabilities"]["memory_review"])

    def test_corrupted_inventory_cannot_receive_review(self):
        folder = self.publish()
        (folder / "graph.json").write_text("{}")
        controller = ArtifactController(self.report, self.trajectory)
        with self.assertRaises(ValueError):
            controller.memory_review("h-1", "approve", "")
        self.assertFalse(controller.state()["capabilities"]["memory_review"])

    def test_review_http_requires_token_and_origin_and_preserves_readonly_routes(self):
        self.publish()
        token = "synthetic-review-session-token-123456789"
        server = create_server(ArtifactController(self.report, self.trajectory), token=token)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        origin = "http://127.0.0.1:%d" % server.server_port
        body = json.dumps({"memory_id": "h-1", "action": "approve", "note": "Human note"}).encode()
        def request(auth=True, same_origin=True, route="/api/memory-review"):
            headers = {"Content-Type": "application/json", "Origin": origin if same_origin else "https://outside.example"}
            if auth:
                headers["Authorization"] = "Bearer " + token
            return Request(origin + route, data=body, headers=headers)
        for args in ({"auth": False}, {"same_origin": False}):
            with self.assertRaises(HTTPError) as failure:
                urlopen(request(**args))
            self.assertEqual(failure.exception.code, 403)
        with urlopen(request()) as response:
            self.assertEqual(json.load(response)["application_status"], "not_applied")
        with self.assertRaises(HTTPError) as failure:
            urlopen(request(route="/api/run"))
        self.assertEqual(failure.exception.code, 405)

    def test_activity_distinguishes_retrieval_delivery_and_model_calls(self):
        trace = {"snapshot_id": "s-old", "selected_ids": ["h-1"], "outcome": "retrieved"}
        experiment = {"traces": {"Hypothesis": trace},
            "report": {"model_calls": [{"stage": "hypothesis", "attempt": 1, "parse_status": "ok"}]}}
        records = _recorded_activity(experiment)
        self.assertEqual([item["kind"] for item in records], ["retrieval", "model_call"])
        self.assertNotIn("read", records[0]["label"].lower())
        trace["actual_delivery"] = {"full_packet_delivered": True}
        records = _recorded_activity(experiment)
        self.assertEqual([item["kind"] for item in records], ["retrieval", "context_delivery", "model_call"])
        self.assertIn("not verified", records[1]["label"])
