"""Full-context TSV regressions; execute only in a reviewed Scheduler allocation."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.tabular import render_table, render_tsv
from research_memory.two_role import TwoRoleMemory


def tables(text):
    result, section, header = {}, None, None
    for row in csv.reader(io.StringIO(text), delimiter="\t"):
        if len(row) == 1 and row[0].startswith("# "):
            section, header = row[0][2:], None
            result[section] = []
        elif section and row:
            if header is None:
                header = row
            else:
                result[section].append(dict(zip(header, row)))
    return result


class TsvMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.save("config.json", {"question": "Why\tcategories?\nAlternative explanations?"})
        self.memory = TwoRoleMemory(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    def publish(self, index, variant=None):
        variant = variant or index
        curve = variant == 7
        term = "I(typicality^2)" if curve else "typicality"
        hypothesis = {"hypothesis": "Typicality\tassociation.\nCheck alternatives.",
            "expected_pattern": "negative" if curve else "positive", "rival_explanation": "selection",
            "focal_test": {"effect_type": "curvature" if curve else "main_effect", "term": term,
                           "expected_direction": "negative" if curve else "positive"},
            "adaptation": {"decision": "initial", "evidence_ids": []}, "historical_claims": []}
        base = f"attempt-{index:04d}/"
        h = self.memory.publish_hypothesis(index, hypothesis, self.save(base + "hypothesis.json", hypothesis),
            "Why\tcategories?\nAlternative explanations?", self.config)
        action = {"family": "logit", "outcome": "tagged", "terms": ["typicality", term] if curve else [term],
                  "strata": "none", "cluster": "none", "filter": "all"}
        plan = {**hypothesis, "analysis": action, "design_rationale": "Binary outcome."}
        plan_path = self.save(base + "plan.json", plan)
        model = self.memory.publish_model(index, h, plan, plan_path)
        row = {"term": term, "coefficient": None if curve else "0.850069632400955123456789",
               "se": None if curve else ".01234567890123456789", "p_value": None if curve else ".10", "ci_low": None}
        tool = {"returncode": 1 if variant == 6 else 0, "effective_action": action,
                "requested_action": action, "input_normalized": False,
                "estimates": [] if variant == 6 else [row],
                "metrics": [{"n": "100", "SE_type": "iid", "mean_outcome": ".5"}]}
        result = self.memory.publish_result(index, model, self.save(base + "tool-result.json", tool),
            plan_path, data_identity="dataset-sha", sample_identity="rows-" + str(variant))
        decision = {"action": "additional_test", "basis_keywords": "numeric facts",
                    "next_inquiry": "Check\trobustness.\nKeep limits explicit.", "evidence_ids": [result]}
        self.memory.publish_decision(index, h, result, decision, self.save(base + "decision.json", decision))

    def test_eight_groups_all_rows_curvature_null_and_exact_precision(self):
        for index in range(1, 9):
            self.publish(index)
        trace, records = self.memory.retrieve_tsv("typicality", role="hypothesis")
        self.assertEqual(len(records), 8)
        self.assertGreater(len(trace["brief"]), 3200)
        self.assertEqual(trace["context_manifest"]["omitted_coefficient_count"], 0)
        self.assertEqual(trace["brief"], render_tsv(trace["context_packet"]))
        parsed = tables(trace["brief"])
        self.assertEqual(len(parsed["COEFFICIENTS"]), 7)
        self.assertEqual(len(parsed["HYPOTHESES"]), 8)
        self.assertEqual(len(parsed["MODELS"]), 8)
        self.assertEqual(len(parsed["DECISIONS"]), 8)
        self.assertTrue(any(r["term"] == "I(typicality^2)" and r["coefficient"] == r"\N" for r in parsed["COEFFICIENTS"]))
        self.assertTrue(any(r["coefficient"] == "0.850069632400955123456789" for r in parsed["COEFFICIENTS"]))
        self.assertEqual({r["n"] for r in parsed["METRICS"]}, {"100"})
        self.assertEqual({r["SE_type"] for r in parsed["METRICS"]}, {"iid"})
        self.assertEqual({r["mean_outcome"] for r in parsed["METRICS"]}, {".5"})

    def test_duplicate_groups_keep_delivered_event_ids_and_allow_own_source_claims(self):
        for index in range(1, 21):
            self.publish(index, variant=1)
        trace, records = self.memory.retrieve_tsv("typicality", role="analysis")
        self.assertEqual(len(records), 1)
        observed = trace["context_packet"]["evidence"][0]["observed_in"]
        delivered_results = {node["memory_id"] for node in trace["context_packet"]["graph_nodes"]
                             if node["memory_type"] == "ResultAssessment"}
        self.assertEqual(set(observed), delivered_results)
        self.assertEqual(set(self.memory.available_records(trace)), delivered_results)
        claim = {"evidence_id": observed[-1], "term": "typicality", "field": "coefficient", "value": "0.850069632400955123456789"}
        self.assertEqual(self.memory.validate_context_claims([claim], trace)[0]["status"], "verified")
        claim["value"] = "999"
        self.assertEqual(self.memory.validate_context_claims([claim], trace)[0]["status"], "contradicted")

    def test_tabs_newlines_and_literal_null_escape_round_trip(self):
        text = render_table("SYNTHETIC", ["text", "missing", "literal", "exact"],
            [{"text": 'a\tb\n"quoted"', "missing": None, "literal": r"\N", "exact": "0.10000000000000000001"}])
        row = tables(text)["SYNTHETIC"][0]
        self.assertEqual(row["text"], 'a\tb\n"quoted"')
        self.assertEqual(row["missing"], r"\N")
        self.assertEqual(row["literal"], r"\\N")
        self.assertEqual(row["exact"], "0.10000000000000000001")

    def test_source_tampering_rejected(self):
        self.publish(1)
        trace, _ = self.memory.retrieve_tsv("typicality")
        self.save("attempt-0001/tool-result.json", {"returncode": 0})
        with self.assertRaises(ValueError):
            self.memory.available_records(trace)

    def test_first_empty_receipts_record_frozen_lexical_ppr_configuration(self):
        memory = TwoRoleMemory(self.root, embedding_backend="lexical", retrieval_strategy="ppr")
        trace, records = memory.retrieve_tsv("first research question")
        self.assertEqual(records, {})
        self.assertEqual(trace["backend"], "lexical")
        self.assertEqual(trace["strategy"], "ppr")
        self.assertEqual(trace["retrieval_status"], "not_run_no_history")
        self.assertEqual(trace["outcome"], "no-match")
        self.assertIsNone(trace["snapshot_id"])
        self.assertTrue(trace["request_id"].startswith("r-"))

        legacy_trace, legacy_records = memory.retrieve("first research question")
        self.assertEqual(legacy_records, {})
        self.assertEqual(legacy_trace["backend"], "lexical")
        self.assertEqual(legacy_trace["strategy"], "ppr")
        self.assertEqual(legacy_trace["retrieval_status"], "not_run_no_history")

    def test_packet_row_or_whole_group_omission_rejected(self):
        self.publish(1)
        trace, _ = self.memory.retrieve_tsv("typicality")
        altered = deepcopy(trace)
        altered["context_packet"]["evidence"][0]["estimates"] = []
        altered["brief"] = render_tsv(altered["context_packet"])
        with self.assertRaises(ValueError):
            self.memory.available_records(altered)
        altered = deepcopy(trace)
        altered["context_packet"]["evidence"] = []
        altered["selected_ids"] = []
        altered["brief"] = render_tsv(altered["context_packet"])
        with self.assertRaises(ValueError):
            self.memory.available_records(altered)

    def test_strict_control_has_no_nodes_or_past_results(self):
        self.publish(1)
        control = TwoRoleMemory(self.root, arm="no-memory")
        trace, records = control.retrieve_tsv("typicality")
        self.assertEqual(records, {})
        self.assertEqual(trace["context_packet"]["graph_nodes"], [])
        self.assertEqual(trace["context_packet"]["evidence"], [])
        self.assertEqual(control.available_records(trace), {})

    def test_fresh_publisher_nodes_equal_validated_disk_and_resumed_nodes(self):
        self.publish(1)
        resumed = TwoRoleMemory(self.root)
        stored = {node["memory_id"]: node for node in
            (json.loads(path.read_text()) for path in self.memory.workspace.glob("candidates/*/*.json"))}
        self.assertEqual(self.memory.nodes, stored)
        self.assertEqual(resumed.nodes, stored)
        self.assertTrue(all(node.get("created_at") for node in self.memory.nodes.values()))


if __name__ == "__main__":
    unittest.main()
