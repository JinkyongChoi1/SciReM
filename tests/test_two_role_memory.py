"""Run only within a reviewed Scheduler allocation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.two_role import TwoRoleMemory, evidence_group_key


class TwoRoleMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.save("config.json", {"question": "Why tagged?"})
        self.memory = TwoRoleMemory(self.root)
        self.hypothesis = {"hypothesis": "typicality -> tagging (+)",
            "expected_pattern": "positive", "focal_test": {"effect_type": "main_effect",
            "term": "typicality", "expected_direction": "positive"}}
        self.plan = {**self.hypothesis, "analysis": {"family": "logit", "outcome": "tagged",
            "terms": ["typicality"], "strata": "none", "cluster": "none", "filter": "all"}}
        self.tool = {"returncode": 0, "effective_action": self.plan["analysis"],
            "metrics": [{"n": "100"}], "estimates": [{"term": "typicality",
            "coefficient": "0.2", "se": "0.1", "p_value": "0.1"}]}

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, payload):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))
        return target

    def stages(self, attempt=1, sample="rows-sha-A"):
        base = f"attempt-{attempt:04d}/"
        h = self.memory.publish_hypothesis(attempt, self.hypothesis,
            self.save(base + "hypothesis.json", self.hypothesis), "Why tagged?", self.config)
        plan_path = self.save(base + "plan.json", self.plan)
        model = self.memory.publish_model(attempt, h, self.plan, plan_path)
        result = self.memory.publish_result(attempt, model, self.save(base + "tool-result.json", self.tool),
            plan_path, data_identity="dataset-sha", sample_identity=sample)
        return h, model, result

    def test_research_question_is_a_published_immutable_root(self):
        identifier = self.memory.ensure_research_question("Why tagged?", self.config)
        self.assertEqual(identifier, "research-question")
        self.assertEqual(self.memory.nodes[identifier]["memory_type"], "ResearchQuestion")
        self.assertEqual(self.memory.nodes[identifier]["payload"], {"question": "Why tagged?"})
        self.assertEqual(self.memory.ensure_research_question("Why tagged?", self.config), identifier)
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.memory.ensure_research_question("A different question", self.config)

    def test_machine_result_published_before_optional_decision(self):
        _, _, result = self.stages()
        trace, supplied = self.memory.retrieve("typicality", budget=12000)
        self.assertIn(result, supplied)
        self.assertFalse(any(n["memory_type"] == "Decision" for n in self.memory.nodes.values()))
        view = trace["context_packet"]["evidence"][0]
        self.assertEqual(view["statistical_check"]["significance"], "O")
        self.assertTrue(view["result_facts_available"])
        self.assertFalse(view["agent_prose_included"])
        self.assertEqual(view["hypotheses"][0]["hypothesis_id"], "hypothesis-0001")

    def test_role_ownership_and_edge_provenance(self):
        h, model, result = self.stages()
        self.assertEqual(self.memory.nodes[h]["actor"], "hypothesis-agent")
        self.assertEqual(self.memory.nodes[model]["actor"], "analysis-agent")
        edge = self.memory.nodes[result]["edges"][0]
        self.assertEqual((edge["target"], edge["relation"]), (model, "produced_by"))
        self.assertEqual(edge["provenance"], self.memory.nodes[result]["provenance"])
        with self.assertRaises(FileExistsError):
            self.stages()

    def test_identical_results_group_only_with_identical_sample(self):
        self.stages(1)
        self.stages(2)
        trace, _ = self.memory.retrieve("typicality", budget=16000)
        self.assertEqual(len(trace["context_packet"]["evidence"]), 1)
        view = trace["context_packet"]["evidence"][0]
        self.assertEqual(len(view["observed_in"]), 2)
        self.assertEqual(len(view["hypotheses"]), 1)
        self.assertEqual(view["omitted_hypothesis_count"], 1)
        self.assertEqual(len(next(iter(trace["equivalence_groups"].values()))["hypotheses"]), 2)
        self.stages(3, sample="rows-sha-B")
        trace, supplied = self.memory.retrieve("typicality", budget=20000)
        self.assertEqual(len(supplied), 2)

    def test_missing_sample_identity_is_not_n_based_equivalence(self):
        _, _, result = self.stages(sample=None)
        record = self.memory.nodes[result]["evidence"]
        self.assertIsNone(evidence_group_key(record, "dataset-sha", None))

    def test_hypothesis_cannot_inject_result_or_analysis(self):
        bad = {**self.hypothesis, "analysis": self.plan["analysis"]}
        with self.assertRaises(ValueError):
            self.memory.publish_hypothesis(1, bad, self.save("bad.json", bad), "Why?", self.config)

    def test_analysis_cannot_mutate_contract(self):
        h, _, _ = self.stages()
        bad = deepcopy(self.plan)
        bad["focal_test"]["expected_direction"] = "negative"
        with self.assertRaises(ValueError):
            self.memory.publish_model(2, h, bad, self.save("bad-plan.json", bad))

    def test_strict_control_never_returns_graph_history(self):
        self.stages()
        control = TwoRoleMemory(self.root, arm="no-memory")
        trace, supplied = control.retrieve("typicality")
        self.assertEqual(supplied, {})
        self.assertEqual(trace["nodes"], [])
        self.assertFalse(trace["context_packet"]["result_facts_available"])

    def test_tampered_source_rejected_before_decision(self):
        h, _, result = self.stages()
        self.save("attempt-0001/hypothesis.json", {"hypothesis": "changed"})
        decision = {"action": "maintain", "basis_keywords": ["support"], "next_inquiry": "controls",
                    "evidence_ids": [result]}
        with self.assertRaises(ValueError):
            self.memory.publish_decision(1, h, result, decision, self.save("decision.json", decision))

    def test_twenty_repetitions_do_not_crowd_out_numeric_facts(self):
        for attempt in range(1, 21):
            h, _, result = self.stages(attempt)
            decision = {"action": "maintain", "basis_keywords": "support",
                        "next_inquiry": "Check robustness.", "evidence_ids": [result]}
            self.memory.publish_decision(attempt, h, result, decision,
                self.save(f"attempt-{attempt:04d}/decision.json", decision))
        trace, supplied = self.memory.retrieve("typicality", budget=3200)
        self.assertEqual(len(supplied), 1)
        self.assertLessEqual(len(trace["brief"]), 3200)
        view = trace["context_packet"]["evidence"][0]
        self.assertEqual(view["statistical_check"]["significance"], "O")
        self.assertEqual(len(view["observed_in"]), 4)
        self.assertEqual(view["omitted_observation_count"], 16)
        self.assertEqual(view["omitted_hypothesis_count"], 19)
        self.assertEqual(view["omitted_decision_count"], 19)
        self.assertEqual(view["decisions"][0]["decision_id"], "decision-0020")
        self.assertEqual(len(next(iter(trace["equivalence_groups"].values()))["sources"]), 20)
        analysis, records = self.memory.retrieve("typicality", role="analysis", budget=3200)
        self.assertEqual(len(records), 1)
        analysis_view = analysis["context_packet"]["evidence"][0]
        self.assertNotIn("hypotheses", analysis_view)
        self.assertNotIn("decisions", analysis_view)
        self.assertIn("hypothesis_ids", analysis_view)


if __name__ == "__main__":
    unittest.main()
