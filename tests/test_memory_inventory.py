"""The full published store is distinct from query-specific retrieval evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.live import ArtifactController, _memory_inventory


class MemoryInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trajectory = self.root / "trajectory"
        self.store = self.trajectory / "shared-memory"
        self.store.mkdir(parents=True)
        self.report = self.root / "report.json"
        self.report.write_text(json.dumps({"trajectory_root": str(self.trajectory),
            "status": "completed", "attempts": [{"attempt": 1}]}))
        self.nodes = [{"memory_id": "h-1", "memory_type": "Hypothesis", "track_id": "PILOT",
                       "payload": {"hypothesis": "Previously tested hypothesis"}},
                      {"memory_id": "h-2", "memory_type": "Hypothesis", "track_id": "PILOT",
                       "payload": {"hypothesis": "Stored but absent from selected retrieval"}}]

    def publish(self, nodes=None, edges=None, snapshot="s-test"):
        folder = self.store / "snapshots" / snapshot
        folder.mkdir(parents=True)
        raw = json.dumps({"nodes": self.nodes if nodes is None else nodes, "edges": edges or []}).encode()
        (folder / "graph.json").write_bytes(raw)
        (folder / "manifest.json").write_text(json.dumps({"snapshot_id": snapshot,
            "files": {"graph.json": hashlib.sha256(raw).hexdigest()}}))
        (self.store / "CURRENT").write_text(snapshot + "\n")
        return folder

    def test_full_inventory_preserves_inactive_nodes_without_fabricated_scores(self):
        self.publish()
        attempt = self.trajectory / "attempt-0001"
        attempt.mkdir()
        trace = {"snapshot_id": "s-earlier", "nodes": [dict(self.nodes[0], ppr=.75)], "edges": []}
        (attempt / "hypothesis-retrieval.json").write_text(json.dumps(trace))
        state = ArtifactController(self.report, self.trajectory).state()
        inventory = state["memory_inventory"]
        self.assertTrue(inventory["complete"])
        self.assertEqual(inventory["nodes"], self.nodes)
        self.assertEqual(inventory["snapshot_id"], "s-test")
        self.assertNotIn("ppr", inventory["nodes"][1])
        self.assertEqual(state["experiment"]["traces"]["Hypothesis"], trace)
        self.assertEqual(state["experiment"]["memory_inventory"], inventory)

    def test_missing_store_is_not_an_empty_published_store(self):
        self.assertEqual(_memory_inventory(self.store)["status"], "unavailable")
        self.publish(nodes=[])
        inventory = _memory_inventory(self.store)
        self.assertEqual(inventory["status"], "available")
        self.assertTrue(inventory["complete"])
        self.assertEqual(inventory["nodes"], [])

    def test_atomic_current_update_is_observed(self):
        self.publish()
        controller = ArtifactController(self.report, self.trajectory)
        self.assertEqual(len(controller.state()["memory_inventory"]["nodes"]), 2)
        self.publish(nodes=self.nodes[:1], snapshot="s-next")
        self.assertEqual(controller.state()["memory_inventory"]["snapshot_id"], "s-next")
        self.assertEqual(len(controller.state()["memory_inventory"]["nodes"]), 1)

    def test_graph_hash_and_endpoint_integrity_are_required(self):
        folder = self.publish()
        (folder / "graph.json").write_text('{"nodes": [], "edges": []}')
        self.assertEqual(_memory_inventory(self.store)["status"], "error")
        self.publish(edges=[{"origin": "h-1", "target": "outside", "relation": "revision_of"}],
                     snapshot="s-bad-edge")
        self.assertEqual(_memory_inventory(self.store)["status"], "error")

    def test_symlinked_graph_is_not_read(self):
        folder = self.publish()
        graph = folder / "graph.json"
        outside = self.root / "outside.json"
        graph.rename(outside)
        graph.symlink_to(outside)
        inventory = _memory_inventory(self.store)
        self.assertEqual(inventory["status"], "error")
        self.assertEqual(inventory["nodes"], [])

    def test_current_cannot_escape_or_follow_an_external_store(self):
        (self.store / "CURRENT").write_text("../../outside")
        self.assertEqual(_memory_inventory(self.store)["status"], "error")
        (self.store / "CURRENT").unlink()
        outside = self.root / "outside-current"
        outside.write_text("private-store")
        (self.store / "CURRENT").symlink_to(outside)
        self.assertEqual(_memory_inventory(self.store)["status"], "error")

    def test_generic_inventory_scopes_records_and_edges_to_selected_track(self):
        self.publish(nodes=[*self.nodes, {"memory_id": "other", "memory_type": "Hypothesis", "track_id": "OTHER"}],
                     edges=[{"origin": "h-2", "target": "h-1", "relation": "revision_of"},
                            {"origin": "other", "target": "h-1", "relation": "related_to"}])
        inventory = _memory_inventory(self.store, "PILOT")
        self.assertEqual(inventory["nodes"], self.nodes)
        self.assertEqual(len(inventory["edges"]), 1)
        self.assertEqual(inventory["track_id"], "PILOT")


if __name__ == "__main__":
    unittest.main()
