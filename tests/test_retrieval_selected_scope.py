"""Selected retrieval must not silently expand into the complete scientific graph."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_memory import evidence
from research_memory.memory import retrieve
from research_memory.tabular import render_tsv
from research_memory.two_role import TwoRoleMemory, selected_context_nodes
from audit_tsv_iteration import audit_trace
from test_two_role_independent import hypothesis, analysis, tool
from two_role_discovery import build_plan


class SelectedScopeTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Reviewed Scheduler allocation required")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = TwoRoleMemory(self.root)
        config = self.save(self.root / "config.json", {"question": "Synthetic question"})
        previous_hypothesis = previous_decision = None
        for index in range(1, 9):
            folder = self.root / f"attempt-{index:04d}"
            h = hypothesis()
            h["hypothesis"] += " unique-marker-" + str(index)
            hid = self.memory.publish_hypothesis(index, h, self.save(folder / "hypothesis.json", h),
                "Synthetic question", config, revision_of=previous_hypothesis,
                follows_decision=previous_decision)
            plan = build_plan(h, analysis())
            plan_path = self.save(folder / "plan.json", plan)
            mid = self.memory.publish_model(index, hid, plan, plan_path)
            result = tool(plan["analysis"])
            result["estimates"].append({"term": "Intercept", "coefficient": "-2.5",
                                       "se": ".22", "p_value": ".001"})
            rid = self.memory.publish_result(index, mid, self.save(folder / "tool-result.json", result),
                plan_path, data_identity="synthetic", sample_identity="sample-" + str(index))
            decision = {"action": "additional_test", "basis_keywords": "Synthetic",
                        "next_inquiry": "Inspect alternatives", "evidence_ids": [rid]}
            previous_decision = self.memory.publish_decision(index, hid, rid, decision,
                self.save(folder / "decision.json", decision))
            previous_hypothesis = hid
        self.raw = retrieve(self.memory.workspace, "typicality", "PILOT", top_k=20)
        self.assertGreater(len(self.raw["nodes"]), 20)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def selected(self, seeds):
        raw = deepcopy(self.raw)
        raw["selected_ids"] = seeds
        with patch("research_memory.two_role.retrieve", return_value=raw):
            return self.memory.retrieve_tsv("typicality")

    def test_selected_decision_closes_result_model_hypothesis_not_revision_history(self):
        trace, records = self.selected(["decision-0008"])
        expected = {"research-question", "hypothesis-0008", "model-0008", "result-0008", "decision-0008"}
        self.assertEqual(set(trace["context_manifest"]["source_node_ids"]), expected)
        self.assertEqual({node["memory_id"] for node in trace["nodes"] if node["included"]}, expected)
        self.assertEqual(set(records), {"result-0008"})
        self.assertNotIn("unique-marker-7", trace["brief"])
        self.assertEqual(len(trace["context_packet"]["evidence"][0]["estimates"]), 2)
        self.assertEqual(trace["context_manifest"]["coefficient_row_count"], 2)
        self.assertEqual(trace["brief"], render_tsv(trace["context_packet"]))
        self.assertEqual(set(self.memory.available_records(trace)), {"result-0008"})
        checked = audit_trace(trace, self.memory.nodes, self.root, 9, "hypothesis", evidence, render_tsv)
        self.assertEqual(checked["node_count"], 5)
        self.assertEqual(len(trace["context_exclusions"]), len(self.raw["nodes"]) - 5)

    def test_no_ranked_match_delivers_fixed_question_only(self):
        trace, records = self.selected([])
        self.assertEqual(set(trace["context_manifest"]["source_node_ids"]), {"research-question"})
        self.assertEqual(records, {})
        self.assertNotIn("unique-marker-", trace["brief"])

    def test_exact_result_lookup_does_not_expand_to_ranked_neighbors(self):
        trace, records = self.memory.retrieve_tsv(
            "Source-verified full output for result-0008, including model specification, coefficients, metrics, and diagnostics.",
            role="analysis", source_lookup=True)
        self.assertEqual(trace["upstream_selected_ids"], ["result-0008"])
        self.assertEqual(set(records), {"result-0008"})
        self.assertEqual(set(trace["context_manifest"]["source_node_ids"]),
                         {"research-question", "hypothesis-0008", "model-0008", "result-0008"})
        self.assertNotIn("unique-marker-7", trace["brief"])
        self.assertEqual(trace["context_manifest"]["coefficient_row_count"], 2)
        self.assertNotIn("context_char_budget", trace)
        self.assertEqual(trace["context_manifest"]["rendered_chars"], len(trace["brief"]))
        self.assertEqual(set(self.memory.available_records(trace)), {"result-0008"})
        audit_trace(trace, self.memory.nodes, self.root, 9, "analysis", evidence, render_tsv)

    def test_multiple_exact_ids_and_missing_source_are_explicit(self):
        trace, records = self.memory.retrieve_tsv("Compare result-0008 and result-0002 and result-9999", source_lookup=True)
        self.assertEqual(set(records), {"result-0008", "result-0002"})
        self.assertEqual(trace["missing_ids"], ["result-9999"])
        self.assertIn("Unavailable source IDs in this snapshot: result-9999", trace["brief"])
        self.assertEqual(trace["context_manifest"]["coefficient_row_count"], 4)
        absent, records = self.memory.retrieve_tsv("Full result-9999 output typicality", source_lookup=True)
        self.assertEqual(records, {})
        self.assertEqual(absent["outcome"], "no-match")
        self.assertNotIn("unique-marker-", absent["brief"])

    def test_explicit_source_selection_has_no_arbitrary_twenty_seed_failure(self):
        seeds = [f"result-{index:04d}" for index in range(1, 26)]
        nodes = {key: {"edges": []} for key in seeds}
        self.assertEqual(selected_context_nodes(nodes, seeds), set(seeds))
        with self.assertRaises(ValueError):
            selected_context_nodes(nodes, seeds + [seeds[0]])

    def test_id_substring_is_not_an_exact_source_lookup(self):
        trace, _ = self.memory.retrieve_tsv("typicality prefixresult-0008suffix", source_lookup=True)
        self.assertEqual(trace["requested_ids"], [])
        self.assertEqual(trace["upstream_selected_ids"], trace["upstream_ranked_ids"])

    def test_broad_research_query_keeps_ppr_even_when_it_mentions_a_result(self):
        trace, _ = self.memory.retrieve_tsv("Alternatives to result-0008 typicality association")
        self.assertEqual(trace["requested_ids"], [])
        self.assertEqual(trace["upstream_selected_ids"], trace["upstream_ranked_ids"])
        self.assertGreater(len(trace["context_manifest"]["executed_result_ids"]), 1)

    def test_selected_source_and_complete_rows_still_verified(self):
        trace, _ = self.selected(["result-0008"])
        changed = deepcopy(trace)
        changed["context_packet"]["evidence"][0]["estimates"].pop()
        changed["brief"] = render_tsv(changed["context_packet"])
        with self.assertRaises(ValueError):
            self.memory.available_records(changed)
        path = self.root / "attempt-0008/tool-result.json"
        original = json.loads(path.read_text())
        original["estimates"][0]["coefficient"] = "999"
        self.save(path, original)
        with self.assertRaises(ValueError):
            self.memory.available_records(trace)

    def test_false_visual_inclusion_and_missing_source_endpoint_rejected(self):
        trace, _ = self.selected(["result-0008"])
        next(node for node in trace["nodes"] if node["memory_id"] == "result-0001")["included"] = True
        with self.assertRaises(ValueError):
            self.memory.available_records(trace)
        raw = deepcopy(self.raw)
        raw["selected_ids"] = ["result-0008"]
        raw["nodes"] = [node for node in raw["nodes"] if node["memory_id"] != "model-0008"]
        with patch("research_memory.two_role.retrieve", return_value=raw), self.assertRaises(ValueError):
            self.memory.retrieve_tsv("typicality")


if __name__ == "__main__":
    unittest.main()
