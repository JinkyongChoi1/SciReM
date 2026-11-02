"""Regressions for context-visible equivalent IDs; Scheduler execution only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.evidence import encode
from research_memory.two_role import TwoRoleMemory


class EquivalentContextIdsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.save("config.json", {"question": "Why categories?"})
        self.memory = TwoRoleMemory(self.root)
        self.statement = "Typicality positively predicts tagging. " + "H" * 140
        self.statement = self.statement[:180]
        self.hypothesis = {"hypothesis": self.statement, "expected_pattern": "positive",
            "focal_test": {"effect_type": "main_effect", "term": "typicality", "expected_direction": "positive"}}
        self.plan = {**self.hypothesis, "analysis": {"family": "logit", "outcome": "tagged",
            "terms": ["typicality", "distinctive"], "strata": "none", "cluster": "none", "filter": "all"}}
        self.tool = {"returncode": 0, "effective_action": self.plan["analysis"],
            "metrics": [{"n": "100", "sample_membership_md5": "a" * 32}],
            "estimates": [{"term": "typicality", "coefficient": "0.850069632400955", "se": ".01", "p_value": ".001"},
                          {"term": "distinctive", "coefficient": ".07", "se": ".02", "p_value": ".02"}]}
        for index in range(1, 7):
            self.publish(index)
        self.trace, self.representatives = self.memory.retrieve("typicality", budget=3200)
        self.assertTrue(self.representatives, "Repetition metadata must not erase numeric facts")
        self.view = self.trace["context_packet"]["evidence"][0]
        self.alias = next(identifier for identifier in self.view["observed_in"] if identifier != self.view["evidence_id"])

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def publish(self, index):
        base = f"attempt-{index:04d}/"
        h = self.memory.publish_hypothesis(index, self.hypothesis,
            self.save(base + "hypothesis.json", self.hypothesis), "Why categories?", self.config)
        plan_path = self.save(base + "plan.json", self.plan)
        model = self.memory.publish_model(index, h, self.plan, plan_path)
        result = self.memory.publish_result(index, model, self.save(base + "tool-result.json", self.tool),
            plan_path, data_identity="dataset-sha", sample_identity="md5:" + "a" * 32)
        decision = {"action": "maintain", "basis_keywords": "K" * 180,
                    "next_inquiry": "Q" * 180, "evidence_ids": [result]}
        self.memory.publish_decision(index, h, result, decision, self.save(base + "decision.json", decision))

    def claim(self, identifier=None, value="0.850069632400955", term="typicality"):
        return {"evidence_id": identifier or self.alias, "term": term, "field": "coefficient", "value": value}

    def test_visible_equivalent_id_resolves_without_changing_representative_selection(self):
        selected = list(self.trace["selected_ids"])
        available = self.memory.available_records(self.trace)
        self.assertIn(self.alias, available)
        self.assertEqual(self.trace["selected_ids"], selected)
        self.assertEqual(self.trace["selected_ids"], [v["evidence_id"] for v in self.trace["context_packet"]["evidence"]])
        self.assertIn(self.alias, self.trace["delivered_equivalent_ids"])
        self.assertEqual(available[self.alias]["evidence_id"], self.alias)
        self.assertEqual(available[self.alias]["result"]["source"]["path"],
                         f"attempt-{int(self.alias.split('-')[1]):04d}/tool-result.json")
        self.assertEqual(self.memory.validate_context_claims([self.claim()], self.trace)[0]["status"], "verified")

    def test_hidden_equivalent_id_does_not_become_available(self):
        shown = set(self.view["observed_in"]) | {self.view["evidence_id"]}
        hidden = next(f"result-{i:04d}" for i in range(1, 7) if f"result-{i:04d}" not in shown)
        self.assertNotIn(hidden, self.memory.available_records(self.trace))
        self.assertEqual(self.memory.validate_context_claims([self.claim(hidden)], self.trace)[0]["status"], "unverifiable")

    def test_wrong_number_for_visible_alias_remains_contradicted(self):
        self.assertEqual(self.memory.validate_context_claims([self.claim(value="999")], self.trace)[0]["status"], "contradicted")

    def test_alias_chronology_uses_own_source_not_representative_order(self):
        order = int(self.alias.split("-")[1])
        claim = {"evidence_id": self.alias, "term": "typicality", "field": "attempt_order", "value": str(order)}
        self.assertEqual(self.memory.validate_context_claims([claim], self.trace)[0]["status"], "verified")

    def test_omitted_coefficient_cannot_be_reported_as_read(self):
        trace = deepcopy(self.trace)
        view = trace["context_packet"]["evidence"][0]
        view["estimates"] = [row for row in view["estimates"] if row["term"] != "distinctive"]
        if "distinctive" not in view["omitted_estimate_terms"]:
            view["omitted_estimate_terms"].append("distinctive")
        trace["brief"] = encode(trace["context_packet"])
        claims = [self.claim(value=".07", term="distinctive"), self.claim()]
        outcomes = self.memory.validate_context_claims(claims, trace)
        self.assertEqual([outcome["status"] for outcome in outcomes], ["unverifiable", "verified"])

    def test_alias_source_tampering_rejected(self):
        source = self.root / f"attempt-{int(self.alias.split('-')[1]):04d}/tool-result.json"
        changed = deepcopy(self.tool)
        changed["estimates"][0]["coefficient"] = "999"
        source.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            self.memory.available_records(self.trace)

    def test_visible_term_does_not_grant_an_omitted_numeric_field(self):
        trace = deepcopy(self.trace)
        view = trace["context_packet"]["evidence"][0]
        for row in view["estimates"]:
            if row["term"] == "typicality":
                row.pop("p_value", None)
        view["omitted_estimate_fields"] = {"typicality": ["p_value"]}
        # The verdict itself also cannot be a hidden second route to p-value access.
        trace["brief"] = encode(trace["context_packet"])
        claim = {**self.claim(), "field": "p_value", "value": ".001"}
        outcomes = self.memory.validate_context_claims([claim, self.claim()], trace)
        self.assertEqual([outcome["status"] for outcome in outcomes], ["unverifiable", "verified"])

    def test_alias_sample_identity_mismatch_rejected(self):
        self.memory.nodes[self.alias]["sample_identity"] = "md5:" + "b" * 32
        with self.assertRaises(ValueError):
            self.memory.available_records(self.trace)

    def test_packet_brief_mismatch_rejected(self):
        trace = deepcopy(self.trace)
        trace["context_packet"]["evidence"][0]["observed_in"].append("result-9999")
        with self.assertRaises(ValueError):
            self.memory.available_records(trace)

    def test_twenty_repeats_maximum_keyword_strings_preserve_numeric_context(self):
        for index in range(7, 21):
            self.publish(index)
        trace, representatives = self.memory.retrieve("typicality", budget=3200)
        self.assertTrue(representatives)
        self.assertLessEqual(len(trace["brief"]), 3200)
        available = self.memory.available_records(trace)
        delivered = {i for view in trace["context_packet"]["evidence"] for i in view["observed_in"]}
        self.assertEqual(set(available), set(trace["selected_ids"]) | delivered)
        self.assertLessEqual(len(available), 5)
        self.assertTrue(trace["context_packet"]["evidence"][0]["estimates"])


if __name__ == "__main__":
    unittest.main()
