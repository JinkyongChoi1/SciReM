"""Run these tests on a scheduler compute node, never the Login Node."""
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.memory import build_snapshot, retrieve, submit_candidate


class SharedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture = Path(__file__).resolve().parents[1] / "examples/synthetic/memories.jsonl"
        self.records = [json.loads(line) for line in fixture.read_text().splitlines()]
        for record in self.records:
            submit_candidate(self.root, record)

    def tearDown(self):
        self.temp.cleanup()

    def test_shared_retrieval_and_trace(self):
        build_snapshot(self.root)
        result = retrieve(self.root, "firm differentiation", "FD", top_k=3)
        self.assertEqual(result["anchors"][0]["id"], "fd-concept")
        self.assertIn("fd-revision", result["selected_ids"])
        self.assertNotIn("dc-private", result["ppr"])
        self.assertAlmostEqual(sum(result["ppr"].values()), 1.0, places=8)
        self.assertTrue(result["converged"])
        self.assertIn("not semantic", result["algorithm"])
        self.assertIn("Source:", result["brief"])
        self.assertEqual({r["memory_id"] for r in result["nodes"] if r["included"]}, set(result["selected_ids"]))

    def test_embedding_free_lexical_anchors_with_personalized_pagerank(self):
        manifest = build_snapshot(self.root, embedding_backend="lexical")
        result = retrieve(self.root, "firm differentiation", "FD", top_k=3, strategy="ppr")
        self.assertEqual(manifest["backend"], "lexical")
        self.assertEqual(result["backend"], "lexical")
        self.assertEqual(result["strategy"], "ppr")
        self.assertTrue(result["anchors"])
        self.assertTrue(result["converged"])
        self.assertGreater(result["iterations"], 0)
        self.assertIn("fd-revision", result["selected_ids"])
        self.assertNotIn("models", [path.name for path in self.root.iterdir()])
        self.assertIn("lexical-RAG (not semantic)+anchors+PPR", result["algorithm"])

    def test_immutable_candidate_and_provenance(self):
        with self.assertRaises(FileExistsError):
            submit_candidate(self.root, self.records[0])
        with self.assertRaises(ValueError):
            submit_candidate(self.root, dict(self.records[0], memory_id="other", source=""))
        with self.assertRaises(ValueError):
            submit_candidate(self.root, dict(self.records[0], agent_id="../escape"))

    def test_snapshot_pinning_and_tamper_detection(self):
        first = build_snapshot(self.root)
        submit_candidate(self.root, dict(self.records[2], memory_id="new-result", aliases=[]))
        second = build_snapshot(self.root)
        self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
        result = retrieve(self.root, "competition", "FD", snapshot_id=first["snapshot_id"])
        self.assertNotIn("new-result", result["ppr"])
        path = self.root / "snapshots" / first["snapshot_id"] / "graph.json"
        path.write_text("{}")
        with self.assertRaises(ValueError):
            retrieve(self.root, "competition", "FD", snapshot_id=first["snapshot_id"])

    def test_no_match_is_explicit(self):
        build_snapshot(self.root)
        result = retrieve(self.root, "xylophone zebra", "FD")
        self.assertEqual(result["outcome"], "no-match")
        self.assertEqual(result["selected_ids"], [])

    def test_first_task_with_empty_memory(self):
        empty = self.root / "empty"
        build_snapshot(empty)
        result = retrieve(empty, "new research question", "FD")
        self.assertEqual(result["outcome"], "no-match")
        self.assertEqual(result["nodes"], [])

    def test_external_vectors_and_dimension_validation(self):
        external = self.root / "external"
        for i, record in enumerate(self.records):
            submit_candidate(external, dict(record, vector=[1, i]))
        build_snapshot(external, "external", "fixture-model-v1")
        result = retrieve(external, "unseen semantic phrase", "FD", query_vector=[1, 0])
        self.assertTrue(result["selected_ids"])
        with self.assertRaises(ValueError):
            retrieve(external, "query", "FD")
        with self.assertRaises(ValueError):
            retrieve(external, "query", "FD", query_vector=[1, 0, 0])

    def test_dense_baseline_does_not_run_pagerank(self):
        build_snapshot(self.root)
        result = retrieve(self.root, "firm differentiation", "FD", strategy="dense")
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(sum(result["ppr"].values()), 0)
        self.assertEqual(result["selected_ids"], ["fd-concept"])

    def test_embedding_free_similarity_baseline_is_labeled_lexical(self):
        manifest = build_snapshot(self.root, embedding_backend="lexical")
        result = retrieve(self.root, "firm differentiation", "FD", strategy="similarity")
        self.assertEqual(manifest["backend"], "lexical")
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(sum(result["ppr"].values()), 0)
        self.assertEqual(result["selected_ids"], ["fd-concept"])
        self.assertEqual(result["algorithm"], "lexical-RAG (not semantic) [lexical similarity only]")
        self.assertNotIn("models", [path.name for path in self.root.iterdir()])

    def test_writer_lock_preserves_current(self):
        manifest = build_snapshot(self.root)
        (self.root / ".index-lock").mkdir()
        with self.assertRaises(RuntimeError):
            build_snapshot(self.root)
        self.assertEqual((self.root / "CURRENT").read_text().strip(), manifest["snapshot_id"])


if __name__ == "__main__":
    unittest.main()
