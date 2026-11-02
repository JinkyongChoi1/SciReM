"""Scheduler-only regressions for lossless evidence, factual recall and bounded packets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.evidence import (result_assessment, verify_record, test_fact, validate_claims, ground_trace,
    loop_candidates, encode, validate_focal_test, statistical_assessment, DesignAlignmentError)
from research_memory.memory import submit_candidate, build_snapshot, retrieve


class CoefficientLabelTests(unittest.TestCase):
    def test_nonprimary_threshold_and_missing_values(self):
        from research_memory.evidence import coefficient_summaries
        rows = [{"term": str(p), "coefficient": "0.03", "p_value": p}
                for p in ("0.0891020922080603", "0.10", "0.10001", "NA", "NaN", "2")]
        self.assertEqual([row['significance'] for row in coefficient_summaries(rows)],
                         ['O', 'O', 'X', '?', '?', '?'])
        self.assertNotIn('significance', rows[0])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.session = self.root / "attempt-0001"
        self.session.mkdir()
        self.term = "distinctive:expertise_genre"
        self.tool = {"returncode": 0, "effective_action": {"family": "logit", "outcome": "tagged",
            "terms": ["distinctive", "expertise_genre", "typicality", "selfclaim_lv", self.term],
            "strata": "choice_set", "cluster": "user_label", "filter": "all"},
            "metrics": [{"n": "1043", "clusters": "500", "SE_type": "cluster_robust"}],
            "estimates": [{"term": key, "coefficient": "0.1", "se": "0.02", "p_value": "0.01"} for key in
                          ("distinctive", "expertise_genre", "typicality", "selfclaim_lv")]
                + [{"term": self.term, "coefficient": "0.0194", "se": "0.0149", "p_value": "0.192", "ci_low": "-0.01", "ci_high": "0.05"}]}
        self.source = self.session / "tool-result.json"
        self.source.write_text(json.dumps(self.tool))
        self.record = result_assessment("result-0001", self.source, self.root, {"interpretation": "not supported"})

    def tearDown(self):
        self.temp.cleanup()

    def claim(self, field="test_status", value="not_in_model", evidence_id="result-0001"):
        return {"evidence_id": evidence_id, "term": self.term, "field": field, "value": value}

    def test_fifth_interaction_and_pvalue_ci_are_preserved(self):
        self.assertEqual(self.record["result"]["estimates"], self.tool["estimates"])
        self.assertEqual(self.record["result"]["analysis"], self.tool["effective_action"])
        self.assertEqual(self.record["result"]["estimates"][4]["p_value"], "0.192")
        self.assertEqual(self.record["result"]["estimates"][4]["ci_high"], "0.05")
        self.assertEqual(self.record["assessment"]["status"], "unverified_agent_interpretation")

    def test_false_no_test_claim_is_contradicted_by_original_result(self):
        verified = validate_claims([self.claim()], {"result-0001": self.record}, self.root)[0]
        self.assertEqual(verified["status"], "contradicted")
        self.assertEqual(verified["expected"], "estimated")

    def test_valid_and_invalid_numbers_are_source_checked(self):
        claims = [self.claim("p_value", "0.192"), self.claim("p_value", "0.001"), self.claim("coefficient", "0.0194")]
        result = validate_claims(claims, {"result-0001": self.record}, self.root)
        self.assertEqual([c["status"] for c in result], ["verified", "contradicted", "verified"])

    def test_missing_packet_evidence_is_unknown_not_no_test(self):
        result = validate_claims([self.claim()], {}, self.root)[0]
        self.assertEqual(result["status"], "unverifiable")
        self.assertIsNone(result["expected"])

    def test_specified_but_unavailable_test_is_distinguished(self):
        altered = copy.deepcopy(self.record)
        altered["result"]["estimates"][4]["p_value"] = "NA"
        self.assertEqual(test_fact(altered, self.term)["status"], "specified_result_unavailable")
        altered["result"]["returncode"] = 1
        self.assertEqual(test_fact(altered, self.term)["status"], "execution_failed")
        self.assertEqual(test_fact(self.record, "not_a_predictor")["status"], "not_in_model")

    def test_tampered_source_and_stored_result_are_not_trusted(self):
        altered = copy.deepcopy(self.record)
        altered["result"]["estimates"][4]["p_value"] = ".001"
        with self.assertRaises(ValueError):
            verify_record(altered, self.root)
        self.source.write_text("{}")
        checked = validate_claims([self.claim()], {"result-0001": self.record}, self.root)[0]
        self.assertEqual(checked["status"], "unverifiable")

    def test_outside_trajectory_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            result_assessment("other", self.source, self.root / "unrelated")

    def trace(self):
        return {"nodes": [{"memory_id": "result-0001", "evidence": self.record, "result_id": "result-0001"}],
                "selected_ids": ["result-0001"], "edges": []}

    def test_fifth_coefficient_survives_query_projection_and_whole_json_budget(self):
        trace, records = ground_trace(self.trace(), self.root, "interaction " + self.term, budget=1800)
        packet = json.loads(trace["brief"])
        rows = packet["evidence"][0]["estimates"]
        self.assertTrue(any(r["term"] == self.term and r["p_value"] == "0.192" for r in rows))
        self.assertLessEqual(len(trace["brief"]), 1800)
        self.assertEqual(trace["brief"], encode(trace["context_packet"]))
        self.assertIn("result-0001", records)

    def test_small_budget_omits_whole_record_and_trace_inclusion_is_honest(self):
        trace, records = ground_trace(self.trace(), self.root, self.term, budget=256)
        self.assertEqual(trace["selected_ids"], [])
        self.assertFalse(trace["nodes"][0]["included"])
        self.assertEqual(json.loads(trace["brief"])["evidence"], [])
        self.assertEqual(records, {})

    def test_five_node_loop_publishes_and_retrieval_hydrates_from_source(self):
        plan = {"hypothesis": "test genre expertise interaction"}
        interpretation = {"next_step": "check uncertainty"}
        candidates, prior = loop_candidates(1, "question", plan, interpretation, self.record, self.session, self.root/"config.json")
        self.assertEqual({c["memory_type"] for c in candidates},
                         {"ResearchQuestion", "Hypothesis", "AnalysisModel", "ResultAssessment", "Decision"})
        memory = self.root / "memory"
        for candidate in candidates:
            submit_candidate(memory, candidate)
        build_snapshot(memory)
        trace = retrieve(memory, self.term, "PILOT", top_k=5)
        trace, supplied = ground_trace(trace, self.root, "interaction " + self.term)
        self.assertTrue(trace["converged"])
        self.assertIn("result-0001", supplied)
        self.assertIn("0.192", trace["brief"])
        new = copy.deepcopy(self.record)
        new["evidence_id"] = "result-0002"
        later, _ = loop_candidates(2, "question", plan, interpretation, new, self.session, self.root/"config.json", prior)
        self.assertEqual(len(later), 4)
        self.assertIn({"target": "hypothesis-0001", "relation": "revision_of", "source": str(self.session/"plan.json")}, later[0]["edges"])

    def primary_plan(self):
        return {"hypothesis": "Genre expertise moderates distinctiveness", "expected_pattern": "positive interaction",
            "focal_test": {"effect_type": "interaction", "term": self.term, "expected_direction": "positive"},
            "analysis": copy.deepcopy(self.tool["effective_action"])}

    def test_moderation_requires_declared_and_executed_interaction(self):
        plan = self.primary_plan()
        self.assertEqual(validate_focal_test(plan), plan["focal_test"])
        plan["analysis"]["terms"].remove(self.term)
        with self.assertRaises(DesignAlignmentError):
            validate_focal_test(plan)
        self.assertEqual(statistical_assessment(plan, self.tool)["support"], "inconclusive")

    def test_primary_not_significant_despite_significant_controls(self):
        plan = self.primary_plan()
        for p in ("0.192", "0.470", "0.10000000001", "1"):
            tool = copy.deepcopy(self.tool)
            tool["estimates"][-1]["p_value"] = p
            verdict = statistical_assessment(plan, tool)
            self.assertEqual(verdict["support"], "not_supported")
            self.assertEqual(verdict["estimate"]["p_value"], p)
            self.assertEqual(verdict["scientific_validity"], "unverified")

    def test_significant_primary_and_direction_have_deterministic_status(self):
        tool = copy.deepcopy(self.tool)
        for p in ("0.05", "0.075", "0.10"):
            tool["estimates"][-1]["p_value"] = p
            self.assertEqual(statistical_assessment(self.primary_plan(), tool)["support"], "supported")
        tool["estimates"][-1]["p_value"] = "6.74e-14"
        self.assertEqual(statistical_assessment(self.primary_plan(), tool)["support"], "supported")
        tool["estimates"][-1]["coefficient"] = "-0.0194"
        self.assertEqual(statistical_assessment(self.primary_plan(), tool)["support"], "not_supported")
        tool["estimates"][-1]["p_value"] = "NA"
        self.assertEqual(statistical_assessment(self.primary_plan(), tool)["support"], "inconclusive")
        tool["returncode"] = 1
        self.assertEqual(statistical_assessment(self.primary_plan(), tool)["support"], "inconclusive")

    def test_effective_model_missing_primary_is_never_supported(self):
        tool = copy.deepcopy(self.tool)
        tool["effective_action"]["terms"].remove(self.term)
        tool["estimates"][-1]["p_value"] = "0.001"
        self.assertEqual(statistical_assessment(self.primary_plan(), tool)["support"], "inconclusive")

    def test_statistical_check_is_bound_to_source_plan_and_tool(self):
        plan = self.primary_plan()
        origin = self.session / "plan.json"
        origin.write_text(json.dumps(plan))
        check = statistical_assessment(plan, self.tool)
        record = result_assessment("result-0001", self.source, self.root,
                                  statistical_check=check, plan_source=origin)
        verify_record(record, self.root)
        altered = copy.deepcopy(record)
        altered["assessment"]["statistical_check"]["support"] = "supported"
        with self.assertRaises(ValueError):
            verify_record(altered, self.root)
        origin.write_text("{}")
        with self.assertRaises(ValueError):
            verify_record(record, self.root)


if __name__ == "__main__":
    unittest.main()
