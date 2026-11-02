"""Independent adversarial tests. Execute only in a reviewed SGE allocation."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from research_memory.two_role import TwoRoleMemory, evidence_group_key
from research_memory.evidence import verify_record
from two_role_discovery import (execute_attempt, build_plan, QUESTION, ANALYSIS_SCHEMA,
    ACTION_SCHEMA, UnsupportedModelDesign, DataInsufficientDesign)
from research_memory.model_catalog import (data_gaps, runner_status, PILOT_ANALYSIS_GUIDANCE,
    DESIGNS, ESTIMATORS, RUNNER_ESTIMATORS)


def hypothesis():
    return {"hypothesis": "Typicality positively predicts tagging.",
            "expected_pattern": "Positive primary coefficient.", "rival_explanation": "Selection.",
            "focal_test": {"effect_type": "main_effect", "term": "typicality", "expected_direction": "positive"},
            "adaptation": {"decision": "initial", "basis": "No earlier evidence.", "evidence_ids": []}, "historical_claims": []}


def analysis():
    return {"analysis": {"family": "logit", "outcome": "tagged", "terms": ["typicality"],
                         "strata": "none", "cluster": "none", "filter": "all"},
            "design_rationale": "Binary outcome.", "model_specification": {
                "design_type": "observational_association", "outcome_type": "binary", "estimator": "logistic_regression",
                "dependent_variable": "tagged", "focal_predictor": "typicality", "focal_terms": ["typicality"],
                "control_variables": [],
                "treatment_variable": None, "instrument_variables": [], "moderator_interaction": None,
                "panel_unit": None, "time_variable": None, "treatment_timing": None,
                "running_variable": None, "cutoff": None, "fixed_effects": [], "strata": [],
                "standard_error_method": "model_based", "cluster_variables": [], "estimand": "associational",
                "identification_assumptions": "No causal claim.", "diagnostics": []}}


def tool(action):
    return {"returncode": 0, "effective_action": action,
            "estimates": [{"term": "typicality", "coefficient": ".02", "se": ".015", "p_value": ".192"}],
            "metrics": [{"n": "100", "sample_membership_md5": "a" * 32}]}


class TwoRoleIndependentTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Reviewed SGE allocation required")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.session = self.root / "attempt-0001"
        self.session.mkdir()
        self.save(self.root / "config.json", {"question": QUESTION})
        self.memory = TwoRoleMemory(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, path, value):
        path.write_text(json.dumps(value))
        return path

    def publish(self, index=1, sample="sample-A"):
        session = self.root / ("attempt-%04d" % index)
        session.mkdir(exist_ok=True)
        h = hypothesis()
        h_id = self.memory.publish_hypothesis(index, h, self.save(session / "hypothesis.json", h),
                                               QUESTION, self.root / "config.json")
        plan = build_plan(h, analysis())
        source = self.save(session / "plan.json", plan)
        m_id = self.memory.publish_model(index, h_id, plan, source)
        r_id = self.memory.publish_result(index, m_id, self.save(session / "tool-result.json", tool(plan["analysis"])),
            source, data_identity="dataset-A", sample_identity=sample)
        return h_id, m_id, r_id, plan

    def test_hypothesis_role_cannot_publish_model_or_verdict(self):
        for extra in ("analysis", "support", "coefficient"):
            h = {**hypothesis(), extra: "POISON"}
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.memory.publish_hypothesis(1, h, self.save(self.session / "hypothesis.json", h),
                                               QUESTION, self.root / "config.json")

    def test_analysis_role_cannot_change_predeclared_focal_direction(self):
        h_id, _, _, plan = self.publish()
        altered = copy.deepcopy(plan)
        altered["focal_test"]["expected_direction"] = "negative"
        with self.assertRaises(ValueError):
            self.memory.publish_model(2, h_id, altered, self.save(self.session / "tampered.json", altered))
        with self.assertRaises(ValueError):
            build_plan(hypothesis(), {**analysis(), "hypothesis": "POISON"})

    def test_model_specification_must_match_effective_estimator_and_cannot_claim_causality(self):
        h, request = hypothesis(), analysis()
        self.assertEqual(build_plan(h, request)["model_specification"]["estimator"], "logistic_regression")
        for field, value in (("estimator", "OLS"), ("dependent_variable", "typicality")):
            changed = copy.deepcopy(request)
            changed["model_specification"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_plan(h, changed)

    def test_broad_catalog_records_unsupported_design_without_estimator_substitution(self):
        fields = ANALYSIS_SCHEMA["properties"]["model_specification"]["properties"]
        designs = fields["design_type"]["enum"]
        self.assertEqual(designs, ["observational_association"])
        self.assertEqual(ANALYSIS_SCHEMA["properties"]["analysis"], ACTION_SCHEMA)
        self.assertNotIn("estimator", fields)
        self.assertNotIn("fixed_effects", fields)
        self.assertNotIn("strata", fields)
        self.assertTrue({"difference_in_differences", "instrumental_variables", "regression_discontinuity"}.issubset(DESIGNS))
        self.assertTrue({"negative_binomial", "fractional_logit", "Cox", "SEM"}.issubset(ESTIMATORS))
        self.assertEqual(fields["dependent_variable"]["type"], "string")
        self.assertEqual(ANALYSIS_SCHEMA["properties"]["model_specification"]["required"].count("focal_terms"), 1)
        self.assertEqual(fields["focal_terms"]["maxItems"], 12)
        self.assertEqual(fields["control_variables"]["maxItems"], 12)
        self.assertEqual(fields["cluster_variables"]["items"]["type"], "string")
        request = analysis()
        request["analysis"] = None
        request["model_specification"].update(design_type="difference_in_differences", estimator="difference_in_differences")
        with self.assertRaisesRegex(DataInsufficientDesign, "treatment_assignment.*parallel-trends"):
            build_plan(hypothesis(), request)
        self.assertEqual(data_gaps("observational_association"), [])
        self.assertEqual(data_gaps("difference_in_differences"),
                         ["time_variable", "treatment_assignment", "treatment_timing"])
        self.assertEqual(runner_status({"design_type": "difference_in_differences"}, {})["status"], "data_insufficient")
        self.assertEqual(runner_status({"design_type": "descriptive"}, {})["status"], "unsupported")
        self.assertIn("Observational association is available", PILOT_ANALYSIS_GUIDANCE)
        self.assertIn("add at most two", PILOT_ANALYSIS_GUIDANCE.lower())
        self.assertIn("Never report the same grouping as both", PILOT_ANALYSIS_GUIDANCE)
        request = analysis()
        request["analysis"] = None
        with self.assertRaisesRegex(ValueError, "requires an exact Runner action"):
            build_plan(hypothesis(), request)

    def test_focal_terms_and_controls_partition_model_terms(self):
        request = analysis()
        request["analysis"]["terms"].append("distinctive")
        with self.assertRaises(ValueError):
            build_plan(hypothesis(), request)
        request["model_specification"]["control_variables"] = ["distinctive"]
        self.assertEqual(build_plan(hypothesis(), request)["model_specification"]["control_variables"], ["distinctive"])

    def test_quadratic_basis_terms_are_focal_terms_not_controls(self):
        contract = hypothesis()
        contract["required_terms"] = ["typicality", "I(typicality^2)"]
        request = analysis()
        request["analysis"]["terms"] = ["typicality", "I(typicality^2)", "distinctive"]
        request["model_specification"]["focal_terms"] = ["typicality", "I(typicality^2)"]
        request["model_specification"]["control_variables"] = ["distinctive"]
        plan = build_plan(contract, request)
        self.assertEqual(plan["model_specification"]["focal_terms"], ["typicality", "I(typicality^2)"])
        self.assertEqual(plan["model_specification"]["control_variables"], ["distinctive"])

        request["analysis"]["terms"] = ["typicality", "I(typicality^2)"]
        request["model_specification"]["control_variables"] = []
        self.assertEqual(build_plan(contract, request)["model_specification"]["control_variables"], [])

    def test_required_term_resolves_unique_dictionary_spelling_without_changing_action(self):
        contract = hypothesis()
        contract["required_terms"] = ["typicality", "self_claim_lv"]
        request = analysis()
        request["analysis"]["terms"] = ["typicality", "selfclaim_lv"]
        request["model_specification"]["control_variables"] = ["selfclaim_lv"]
        plan = build_plan(contract, request)
        self.assertEqual(plan["analysis"]["terms"], ["typicality", "selfclaim_lv"])
        self.assertEqual(plan["required_term_resolutions"], [{"requested": "self_claim_lv",
            "executed": "selfclaim_lv", "basis": "unique_data_dictionary_identifier_match_ignoring_underscores"}])

    def test_required_term_alias_requires_unique_dictionary_match(self):
        contract = hypothesis()
        contract["required_terms"] = ["typicality", "self_claim_unknown"]
        request = analysis()
        with self.assertRaisesRegex(ValueError, "without a unique dictionary-field match"):
            build_plan(contract, request)

    def test_required_term_alias_does_not_ignore_case(self):
        contract = hypothesis()
        contract["required_terms"] = ["typicality", "Self_claim_lv"]
        request = analysis()
        request["analysis"]["terms"].append("selfclaim_lv")
        with self.assertRaisesRegex(ValueError, "without a unique dictionary-field match"):
            build_plan(contract, request)

    def test_focal_and_control_terms_cannot_overlap_or_omit_runner_terms(self):
        request = analysis()
        request["analysis"]["terms"].append("distinctive")
        request["model_specification"]["focal_terms"].append("distinctive")
        request["model_specification"]["control_variables"] = ["distinctive"]
        with self.assertRaisesRegex(ValueError, "partition"):
            build_plan(hypothesis(), request)

    def test_conditional_logit_strata_are_not_mislabeled_as_fixed_effects(self):
        request = analysis()
        request["analysis"].update(family="conditional_logit", strata="choice_set")
        request["model_specification"].update(estimator="conditional_logistic_regression",
            fixed_effects=[], strata=["choice_set"])
        plan = build_plan(hypothesis(), request)
        self.assertEqual(plan["model_specification"]["execution_support"]["status"], "supported")
        request["model_specification"]["fixed_effects"] = ["choice_set"]
        with self.assertRaises(ValueError):
            build_plan(hypothesis(), request)

    def test_runner_action_derives_estimator_and_group_labels(self):
        request = analysis()
        request["analysis"].update(family="logit", strata="choice_set")
        spec = request["model_specification"]
        for key in ("estimator", "fixed_effects", "strata"):
            spec.pop(key)
        plan = build_plan(hypothesis(), request)
        self.assertEqual(plan["model_specification"]["estimator"], "fixed_effects_logistic_regression")
        self.assertEqual(plan["model_specification"]["fixed_effects"], ["choice_set"])
        self.assertEqual(plan["model_specification"]["strata"], [])

    def test_decision_cannot_override_nonsignificant_machine_result(self):
        h_id, _, r_id, _ = self.publish()
        self.assertEqual(self.memory.nodes[r_id]["evidence"]["assessment"]["statistical_check"]["support"], "not_supported")
        decision = {"action": "maintain", "basis_keywords": "recent receipt", "next_inquiry": "Robustness.",
                    "evidence_ids": [r_id], "support": "supported"}
        with self.assertRaises(ValueError):
            self.memory.publish_decision(1, h_id, r_id, decision, self.save(self.session / "decision.json", decision))

    def test_decision_requires_a_published_immutable_result_receipt(self):
        h = hypothesis()
        h_id = self.memory.publish_hypothesis(1, h, self.save(self.session / "hypothesis.json", h),
                                               QUESTION, self.root / "config.json")
        plan = build_plan(h, analysis())
        self.memory.publish_model(1, h_id, plan, self.save(self.session / "plan.json", plan))
        decision = {"action": "maintain", "basis_keywords": "No receipt.", "next_inquiry": "Stop.",
                    "evidence_ids": ["result-0001"]}
        with self.assertRaises(ValueError):
            self.memory.publish_decision(1, h_id, "result-0001", decision,
                                         self.save(self.session / "decision.json", decision))

    def test_tampered_statistical_assessment_is_rejected(self):
        _, _, r_id, _ = self.publish()
        record = copy.deepcopy(self.memory.nodes[r_id]["evidence"])
        record["assessment"]["statistical_check"]["support"] = "supported"
        with self.assertRaises(ValueError):
            verify_record(record, self.root)

    def test_tsv_context_discloses_retrieval_cap_instead_of_claiming_full_graph(self):
        self.publish()
        trace, _ = self.memory.retrieve_tsv("typicality", role="hypothesis")
        self.assertIn("up to 20 ranked seeds plus required scientific source endpoints", trace["context_scope"])
        self.assertEqual(trace["context_packet"]["reason"], "retrieved_own_context_top_k_20")

    def test_duplicate_group_requires_data_sample_model_results_and_preserves_ids(self):
        _, _, first, _ = self.publish()
        _, _, second, _ = self.publish(2)
        a, b = self.memory.nodes[first], self.memory.nodes[second]
        self.assertEqual(a["evidence_group"], b["evidence_group"])
        self.assertIsNone(evidence_group_key(a["evidence"], "dataset-A", None))
        self.assertNotEqual(a["evidence_group"], evidence_group_key(b["evidence"], "dataset-A", "other-rows"))
        changed = copy.deepcopy(b["evidence"])
        changed["result"]["analysis"]["cluster"] = "user_label"
        self.assertNotEqual(a["evidence_group"], evidence_group_key(changed, "dataset-A", "sample-A"))
        trace, supplied = self.memory.retrieve("typicality", budget=6000)
        views = trace["context_packet"]["evidence"]
        self.assertEqual(len(views), 1)
        self.assertEqual(set(views[0]["observed_in"]), {first, second})
        self.assertEqual(set(supplied), set(trace["selected_ids"]))

    def test_five_node_types_typed_edges_and_immutable_revision(self):
        h_id, m_id, r_id, _ = self.publish()
        decision = {"action": "additional_test", "basis_keywords": "recent receipt", "next_inquiry": "Robustness.", "evidence_ids": [r_id]}
        d_id = self.memory.publish_decision(1, h_id, r_id, decision, self.save(self.session / "decision.json", decision))
        nodes = self.memory.nodes
        self.assertEqual({n["memory_type"] for n in nodes.values()}, set(TwoRoleMemory.OWNERS))
        self.assertIn({"target": h_id, "relation": "intended_test_for"},
                      [{"target": e["target"], "relation": e["relation"]} for e in nodes[m_id]["edges"]])
        for node in nodes.values():
            self.assertEqual(node["actor"], TwoRoleMemory.OWNERS[node["memory_type"]])
            self.assertEqual(len(node["provenance"]["sha256"]), 64)
            for edge in node["edges"]:
                self.assertIn(edge["target"], nodes)
                self.assertEqual(len(edge["provenance"]["sha256"]), 64)
        h = hypothesis()
        self.memory.publish_hypothesis(2, h, self.save(self.session / "next.json", h), QUESTION,
            self.root / "config.json", revision_of=h_id, follows_decision=d_id)
        with self.assertRaises(FileExistsError):
            self.memory.publish_hypothesis(1, h, self.session / "hypothesis.json", QUESTION, self.root / "config.json")

    def test_no_memory_never_retrieves_or_includes_prior_query_plan_error(self):
        memory = TwoRoleMemory(self.root, arm="no-memory")
        memory.retrieve = Mock(side_effect=AssertionError("Control retrieval forbidden"))
        prompts = []
        def model(label, prompt, schema, tokens, seed):
            prompts.append(prompt)
            return {"hypothesis": hypothesis(), "analysis_plan": analysis(), "decision": {
                "action": "maintain", "basis_keywords": "Current facts.", "next_inquiry": "Stop.",
                "evidence_ids": ["result-0001"]}}[label]
        record, _, _ = execute_attempt(self.session, 1, "no-memory", memory, model,
            analysis_runner=lambda session, action: tool(action), query="POISON_QUERY_PRIOR_ERROR",
            previous_plan={**build_plan(hypothesis(), analysis()), "hypothesis": "POISON_PRIOR_HYPOTHESIS"}, data_identity="dataset-A")
        self.assertEqual(record["status"], "completed", record)
        self.assertNotIn("POISON", "".join(prompts))
        memory.retrieve.assert_not_called()

    def test_decision_json_failure_leaves_searchable_machine_result(self):
        def model(label, prompt, schema, tokens, seed):
            if label == "decision":
                raise json.JSONDecodeError("truncated", "{", 1)
            return hypothesis() if label == "hypothesis" else analysis()
        record, _, _ = execute_attempt(self.session, 1, "memory", self.memory, model,
            analysis_runner=lambda session, action: tool(action), data_identity="dataset-A")
        self.assertEqual(record["analysis_status"], "succeeded", record)
        self.assertEqual(record["decision_status"], "failed", record)
        self.assertIn("result-0001", self.memory.nodes)
        _, supplied = self.memory.retrieve("typicality", budget=6000)
        self.assertIn("result-0001", supplied)

    def test_contradicted_historical_claim_is_rejected_before_analysis(self):
        self.publish()
        session = self.root / "attempt-0002"
        session.mkdir()
        false = hypothesis()
        false["historical_claims"] = [{"evidence_id": "result-0001", "term": "typicality", "field": "p_value", "value": "0.001"}]
        runner = Mock(side_effect=AssertionError("False claims must not execute"))
        record, _, _ = execute_attempt(session, 2, "memory", self.memory,
            lambda *args: false, analysis_runner=runner, data_identity="dataset-A")
        runner.assert_not_called()
        self.assertEqual(record["status"], "failed")
        self.assertNotIn("hypothesis-0002", self.memory.nodes)

    def test_recent_receipt_is_not_automatically_a_temporal_error(self):
        contract = hypothesis()
        contract["adaptation"] = {"decision": "continue", "basis": "The recent receipt is consistent.", "evidence_ids": []}
        def model(label, *args):
            return {"hypothesis": contract, "analysis_plan": analysis(), "decision": {
                "action": "maintain", "basis_keywords": "Recent receipt.", "next_inquiry": "Robustness.",
                "evidence_ids": ["result-0001"]}}[label]
        record, _, _ = execute_attempt(self.session, 1, "memory", self.memory, model,
            analysis_runner=lambda session, action: tool(action), data_identity="dataset-A")
        self.assertEqual(record["status"], "completed", record)
        self.assertFalse(any("temporal" in issue for issue in record["adaptation"]["validation_issues"]))

    def test_twenty_duplicate_results_do_not_crowd_out_primary_numeric_facts(self):
        ids = []
        for index in range(1, 21):
            h_id, _, result_id, _ = self.publish(index)
            ids.append(result_id)
            session = self.root / ("attempt-%04d" % index)
            decision = {"action": "maintain", "basis_keywords": "Same estimate.",
                        "next_inquiry": "Robustness.", "evidence_ids": [result_id]}
            self.memory.publish_decision(index, h_id, result_id, decision,
                self.save(session / "decision.json", decision))
        trace, supplied = self.memory.retrieve("typicality", role="hypothesis", budget=3200)
        self.assertTrue(supplied, trace["brief"])
        self.assertLessEqual(len(trace["brief"]), 3200)
        evidence = trace["context_packet"]["evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertIn(".192", json.dumps(evidence))
        self.assertEqual(len([n for n in self.memory.nodes.values() if n["memory_type"] == "ResultAssessment"]), 20)

    def test_retrieval_rejects_tampered_hypothesis_source(self):
        self.publish()
        altered = hypothesis()
        altered["hypothesis"] = "POISON changed after publication."
        self.save(self.session / "hypothesis.json", altered)
        with self.assertRaises(ValueError):
            self.memory.retrieve("typicality", budget=3200)


if __name__ == "__main__":
    unittest.main()
