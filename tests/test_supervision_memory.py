"""Scheduler-only source integrity and supervisor handback retrieval regressions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.evidence import result_assessment, verify_record
from research_memory.two_role import TwoRoleMemory, evidence_group_key
from research_memory.tabular import render_tsv


class SupervisionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.action = {"family": "logit", "outcome": "outcome", "terms": ["predictor"],
                       "strata": "none", "cluster": "none", "filter": "all"}
        self.tool = {"returncode": 0, "effective_action": self.action,
                     "estimates": [{"term": "predictor", "coefficient": ".2", "se": ".1", "p_value": ".05"}],
                     "metrics": [{"n": "100"}]}
        self.review = {"status": "blocked", "requested_test_status": "unfulfilled",
                       "scientific_validity": "invalid", "claim_consistency": "consistent",
                       "required_assumptions_status": "violated", "next_action": "blocked",
                       "pending_work": "Repair the uncertainty estimator before reusing this comparison.",
                       "issues": [{"source": "diagnostic_code", "finding": "The uncertainty calculation ignores required dependence."}],
                       "explanation": "A finite comparison estimate does not validate its confidence interval."}

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = content.encode() if isinstance(content, str) else json.dumps(content).encode()
        path.write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    def record(self, tool):
        source = self.artifact("attempt-0001/tool-result.json", tool)
        return result_assessment("result-0001", self.root / source["path"], self.root)

    def supervised_tool(self, index=1, review=None):
        tool = deepcopy(self.tool)
        tool["scientific_review"] = deepcopy(self.review if review is None else review)
        tool["scientific_review_source"] = self.artifact(
            f"attempt-{index:04d}/scientific-review.json", tool["scientific_review"])
        return tool

    def diagnostic_tool(self, nested=True):
        folder = "attempt-0001/" + ("supervision-0001/execution/" if nested else "") + "agent-r-diagnostic/"
        code = self.artifact(folder + "diagnostic.R", "write.csv(data.frame(metric='comparison', value=0), '/work/diagnostics.csv')")
        output = self.artifact(folder + "diagnostics.csv", "metric,value\ncomparison,0\n")
        tool = deepcopy(self.tool)
        tool["agent_r_diagnostic"] = {"status": "completed", "code_sha256": code["sha256"],
            "output_sha256": output["sha256"], "metrics": {"comparison": 0}}
        if nested:
            tool["agent_r_diagnostic"]["source"] = {**code, "path_scope": "trajectory"}
        return tool, code, output

    def test_nested_repaired_code_and_output_verified_at_actual_location(self):
        tool, code, _ = self.diagnostic_tool()
        record = self.record(tool)
        self.assertEqual(verify_record(record, self.root), record)
        (self.root / code["path"]).write_text("altered code")
        with self.assertRaises(ValueError):
            verify_record(record, self.root)

    def test_nested_output_tampering_is_rejected(self):
        tool, _, output = self.diagnostic_tool()
        record = self.record(tool)
        (self.root / output["path"]).write_text("metric,value\ncomparison,999\n")
        with self.assertRaises(ValueError):
            verify_record(record, self.root)

    def test_code_shown_to_agent_must_match_executed_code_hash(self):
        tool, code, _ = self.diagnostic_tool()
        tool['agent_r_diagnostic']['code'] = (self.root / code['path']).read_text()
        verify_record(self.record(tool), self.root)
        tool['agent_r_diagnostic']['code'] = 'different code shown to the reviewer'
        with self.assertRaises(ValueError):
            verify_record(self.record(tool), self.root)

    def test_legacy_diagnostic_without_source_uses_original_location(self):
        tool, _, _ = self.diagnostic_tool(nested=False)
        record = self.record(tool)
        self.assertEqual(verify_record(record, self.root), record)

    def test_diagnostic_symlink_alias_and_outside_path_are_rejected(self):
        tool, code, _ = self.diagnostic_tool()
        path = self.root / code["path"]
        replacement = path.with_name("actual.R")
        path.rename(replacement)
        path.symlink_to(replacement.name)
        with self.assertRaises(ValueError):
            verify_record(self.record(tool), self.root)
        tool["agent_r_diagnostic"]["source"]["path"] = "../outside.R"
        with self.assertRaises(ValueError):
            verify_record(self.record(tool), self.root)

    def test_parent_directory_symlink_cannot_bypass_artifact_containment(self):
        tool, code, _ = self.diagnostic_tool()
        folder = (self.root / code["path"]).parent
        actual = folder.with_name("actual-diagnostic")
        folder.rename(actual)
        folder.symlink_to(actual.name, target_is_directory=True)
        with self.assertRaises(ValueError):
            verify_record(self.record(tool), self.root)

    def test_review_record_and_source_tampering_are_rejected(self):
        tool = self.supervised_tool()
        record = self.record(tool)
        verify_record(record, self.root)
        altered = deepcopy(record)
        altered["assessment"]["scientific_review"]["status"] = "accepted"
        with self.assertRaises(ValueError):
            verify_record(altered, self.root)
        (self.root / tool["scientific_review_source"]["path"]).write_text("{}")
        with self.assertRaises(ValueError):
            verify_record(record, self.root)

    def test_review_linked_sources_are_hashed_and_trajectory_bound(self):
        review = deepcopy(self.review)
        source = self.artifact("attempt-0001/supervision-0001/evidence.json", {"receipt": "observed"})
        review["sources"] = [source]
        tool = self.supervised_tool(review=review)
        record = self.record(tool)
        verify_record(record, self.root)
        (self.root / source["path"]).write_text("{}")
        with self.assertRaises(ValueError):
            verify_record(record, self.root)
        review["sources"][0]["path"] = "../other-trajectory/evidence.json"
        with self.assertRaises(ValueError):
            verify_record(self.record(self.supervised_tool(review=review)), self.root)

    def test_legacy_is_not_promoted_to_reviewed_and_review_affects_dedup(self):
        legacy = self.record(self.tool)
        self.assertNotIn("scientific_review", legacy["assessment"])
        reviewed = self.record(self.supervised_tool())
        self.assertNotEqual(evidence_group_key(legacy, "data", "rows"),
                            evidence_group_key(reviewed, "data", "rows"))
        accepted = deepcopy(reviewed)
        accepted["assessment"]["scientific_review"]["status"] = "accepted"
        self.assertNotEqual(evidence_group_key(reviewed, "data", "rows"),
                            evidence_group_key(accepted, "data", "rows"))

    def publish(self, memory, index, tool):
        hypothesis = {"hypothesis": "Predictor is positively associated with outcome.",
            "expected_pattern": "positive", "rival_explanation": "selection",
            "focal_test": {"effect_type": "main_effect", "term": "predictor", "expected_direction": "positive"},
            "adaptation": {"decision": "initial", "evidence_ids": []}, "historical_claims": []}
        base = f"attempt-{index:04d}/"
        config = self.artifact("config.json", {"question": "Explain the outcome"})
        h_source = self.artifact(base + "hypothesis.json", hypothesis)
        h_id = memory.publish_hypothesis(index, hypothesis, self.root / h_source["path"],
                                        "Explain the outcome", self.root / config["path"])
        plan = {**hypothesis, "analysis": deepcopy(self.action), "design_rationale": "Association only"}
        plan_source = self.artifact(base + "plan.json", plan)
        model_id = memory.publish_model(index, h_id, plan, self.root / plan_source["path"])
        source = self.artifact(base + "tool-result.json", tool)
        return memory.publish_result(index, model_id, self.root / source["path"], self.root / plan_source["path"],
                                     data_identity="same-data", sample_identity="same-rows")

    def test_blocked_review_survives_graph_retrieval_separate_from_numeric_facts(self):
        memory = TwoRoleMemory(self.root)
        result_id = self.publish(memory, 1, self.supervised_tool())
        trace, _ = memory.retrieve_tsv(result_id, source_lookup=True)
        records = memory.available_records(trace)
        self.assertEqual(records[result_id]["assessment"]["scientific_review"], self.review)
        self.assertEqual(records[result_id]["result"]["estimates"], self.tool["estimates"])
        self.assertIn("# SUPERVISOR_ASSESSMENTS", trace["brief"])
        self.assertIn("# SUPERVISOR_ISSUES", trace["brief"])
        self.assertIn(self.review["pending_work"], trace["brief"])
        self.assertIn(self.review["issues"][0]["finding"], trace["brief"])
        altered = deepcopy(trace)
        altered["context_packet"]["evidence"][0]["scientific_review"]["status"] = "accepted"
        altered["brief"] = render_tsv(altered["context_packet"])
        with self.assertRaises(ValueError):
            memory.available_records(altered)

    def test_pending_review_remains_pending_in_retrieval(self):
        memory = TwoRoleMemory(self.root)
        result_id = self.publish(memory, 1, {**deepcopy(self.tool), "scientific_review_pending": True})
        trace, _ = memory.retrieve_tsv(result_id, source_lookup=True)
        record = memory.available_records(trace)[result_id]
        self.assertTrue(record["assessment"]["scientific_review_pending"])
        self.assertNotIn("scientific_review", record["assessment"])
        self.assertIn("pending", trace["brief"])

    def test_same_numeric_result_with_different_reviews_is_not_merged(self):
        memory = TwoRoleMemory(self.root)
        first = self.publish(memory, 1, self.supervised_tool())
        accepted = {**deepcopy(self.review), "status": "accepted", "requested_test_status": "fulfilled",
                    "scientific_validity": "adequate", "required_assumptions_status": "satisfied",
                    "next_action": "accept", "issues": [], "pending_work": None}
        second = self.publish(memory, 2, self.supervised_tool(index=2, review=accepted))
        trace, _ = memory.retrieve_tsv(first + " " + second, source_lookup=True)
        self.assertEqual(len(trace["context_packet"]["evidence"]), 2)
        self.assertEqual({view["scientific_review"]["status"] for view in trace["context_packet"]["evidence"]},
                         {"blocked", "accepted"})
