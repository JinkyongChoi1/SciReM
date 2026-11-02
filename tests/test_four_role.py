"""Synthetic contract tests for the Orchestrator→Hypothesis→Empiricist→Runner loop."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from research_memory.orchestration import FourRoleOrchestrator, HandoffError
from research_memory.two_role import TwoRoleMemory
from private_inference import RuntimeIsolationError, ContextCapacityError
from four_role_discovery import execute_four_role_attempt as _execute_four_role_attempt, call_role_with_memory, validate_retrieved_claims
from four_role_memory_iteration import execute as _execute, configure_three_condition
from four_role_memory_iteration import configure
import memory_iteration
from four_role_discovery import RoleHistory


class UnicodeHistoryTests(unittest.TestCase):
    def test_matching_history_policy_preserves_same_history_for_b_and_c(self):
        import four_role_discovery
        with tempfile.TemporaryDirectory() as directory:
            history = RoleHistory(directory)
            history.add('hypothesis', 'proposal', {'value': 'first'})
            history.add('hypothesis', 'proposal', {'value': 'second'})
            with patch.object(four_role_discovery, 'COMPACT_SHARED_HISTORY', False):
                self.assertEqual(history.prompt('hypothesis', 'current', shared_memory=False),
                                 history.prompt('hypothesis', 'current', shared_memory=True))

    def test_jsonl_unicode_separators_survive_reload_and_prior_result_lookup(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            history = RoleHistory(directory)
            text = "query\u2028continued\u2029paragraph\u0085next"
            history.add("hypothesis", "decision", {"basis": text, "evidence_ids": ["result-0001"]})
            restored = RoleHistory(directory)
            for compact in (False, True):
                self.assertIn(text, restored.prompt("hypothesis", "next", shared_memory=compact))
            self.assertEqual(restored.prior_result_ids(), {"result-0001"})
from four_role_discovery import hypothesis_schema
from two_role_discovery import QUESTION
from two_role_discovery import build_plan


def hypothesis():
    return {"hypothesis": "Typicality positively predicts tagging.", "expected_pattern": "Positive effect.",
            "rival_explanation": "Selection.",
            "focal_test": {"effect_type": "main_effect", "term": "typicality", "expected_direction": "positive"},
            "adaptation": {"decision": "initial", "basis": "Initial attempt; no prior evidence.", "evidence_ids": []},
            "historical_claims": []}


def plan():
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


def result(action):
    return {"returncode": 0, "effective_action": action,
            "estimates": [{"term": "typicality", "coefficient": ".02", "se": ".015", "p_value": ".192"}],
            "metrics": [{"n": "100", "sample_membership_md5": "b" * 32}]}


def answer_envelope_adapter(model):
    """Let legacy fixture callbacks answer newly enveloped requests without weakening production."""
    def adapted(label, prompt, schema, tokens, seed):
        if label == "orchestrator_scientific_review":
            # Legacy tests isolate other contracts; dedicated supervisor tests use explicit reviews.
            from test_orchestrator_review import favorable
            response = favorable()
        elif label == "orchestrator_decision_review":
            response = {"claim_consistency": "consistent", "issues": [], "next_action": "accept",
                        "explanation": "Legacy fixture Decision agrees with its supplied evidence."}
        else:
            response = model(label, prompt, schema, tokens, seed)
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        branches = schema.get("anyOf", []) if isinstance(schema, dict) else []
        envelope = ("action" in properties and "query" in properties and "answer" in properties) or any(
            isinstance(branch, dict) and {"action", "query", "answer"}.issubset(
                branch.get("properties", {})) for branch in branches)
        if envelope:
            if not (isinstance(response, dict) and response.get("action") in {"answer", "retrieve_memory"}):
                return {"action": "answer", "query": None, "answer": response}
        return response
    return adapted


def execute_four_role_attempt(*args, **kwargs):
    args = list(args)
    if len(args) > 3 and callable(args[3]):
        args[3] = answer_envelope_adapter(args[3])
    elif callable(kwargs.get("model")):
        kwargs["model"] = answer_envelope_adapter(kwargs["model"])
    return _execute_four_role_attempt(*args, **kwargs)


def execute(*args, **kwargs):
    args = list(args)
    if len(args) > 4 and callable(args[4]):
        args[4] = answer_envelope_adapter(args[4])
    elif callable(kwargs.get("model")):
        kwargs["model"] = answer_envelope_adapter(kwargs["model"])
    return _execute(*args, **kwargs)


class FourRoleTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Reviewed SGE allocation required")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config.json").write_text(json.dumps({"question": QUESTION}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_repeated_one_shots_are_independent_full_cycles(self):
        from memory_iteration import one_shot_episode, may_start_attempt
        prompts = []
        for ordinal in (1, 2):
            session = one_shot_episode(self.root, ordinal)
            def model(label, prompt, schema, tokens, seed):
                if label == "hypothesis":
                    prompts.append(prompt)
                    return hypothesis()
                if label == "empiricist_plan": return plan()
                if label == "empiricist_assessment":
                    return {"execution_assessment": "Evidence available.", "diagnostic_action": "additional_test"}
                if label == "hypothesis_decision":
                    return {"action": "maintain", "basis_keywords": "SENTINEL_PREVIOUS_EPISODE",
                            "next_inquiry": "SENTINEL_PREVIOUS_EPISODE", "evidence_ids": ["result-0001"]}
                raise AssertionError(label)
            record, _, _ = execute_four_role_attempt(session, 1, None, model,
                lambda _, action: result(action), data_identity="fixture", condition="A")
            self.assertEqual(record["status"], "completed", record)
            self.assertFalse(record["shared_memory_enabled"])
        self.assertNotIn("SENTINEL_PREVIOUS_EPISODE", prompts[1])
        self.assertIn("No cross-attempt evidence supplied", prompts[1])
        self.assertTrue(may_start_attempt("A", 2, 100, 3600, repeat_one_shot=True))
        self.assertFalse(may_start_attempt("A", 2, 3555, 3600, repeat_one_shot=True))
        self.assertFalse((self.root / "role-history").exists())

    def test_agent_diagnostics_available_in_all_conditions(self):
        for condition in ("A", "B", "C"):
            with self.subTest(condition=condition):
                root = self.root / condition
                root.mkdir()
                (root / "config.json").write_text(json.dumps({"question": QUESTION}))
                session = root / "attempt-0001"
                session.mkdir()
                memory = TwoRoleMemory(root) if condition == "C" else None
                def model(label, prompt, schema, tokens, seed):
                    if label == "hypothesis": return hypothesis()
                    if label == "empiricist_plan": return {**plan(), "diagnostic_r_code": "synthetic code"}
                    if label == "empiricist_assessment":
                        self.assertIn("agent_diagnostic_mean", prompt)
                        return {"execution_assessment": "Diagnostic remains unverified.", "diagnostic_action": "additional_test"}
                    if label == "hypothesis_decision":
                        return {"action": "maintain", "basis_keywords": "One result.",
                                "next_inquiry": "Assess robustness.", "evidence_ids": ["result-0001"]}
                    raise AssertionError(label)
                def diagnostic_fixture(folder, code):
                    import hashlib
                    target = folder / "agent-r-diagnostic"
                    target.mkdir()
                    source = target / "diagnostic.R"
                    source.write_text(code)
                    return {"status": "completed", "metrics": {"mean": 2.0}, "scope": "unverified",
                            "source": {"path": str(source.relative_to(folder)),
                                       "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
                with patch("agent_r_diagnostic.run_agent_r", side_effect=diagnostic_fixture) as run:
                    from agent_r_diagnostic import run_agent_r
                    record, _, _ = execute_four_role_attempt(session, 1, memory, model,
                        lambda _, action: result(action), data_identity="fixture", condition=condition,
                        diagnostic_runner=run_agent_r)
                self.assertEqual(record["status"], "completed", record)
                self.assertEqual(run.call_count, 1)
                facts = json.loads((session / "current-result-facts.json").read_text())
                self.assertEqual(facts["metrics"][0]["agent_diagnostic_mean"], 2.0)

    def test_orchestrator_rejects_out_of_order_or_receiptless_handoffs(self):
        flow = FourRoleOrchestrator(self.root, 1, research_question=QUESTION, inquiry=QUESTION)
        with self.assertRaises(HandoffError):
            flow.route("runner", "runner")
        flow.route("hypothesis", "hypothesis-agent", {"state": "initial", "research_question": QUESTION,
                                                     "current_inquiry": QUESTION})
        with self.assertRaises(HandoffError):
            flow.route("empiricist_assessment", "empiricist-agent")
        (self.root / "hypothesis.json").write_text(json.dumps(hypothesis()))
        flow.route("empiricist_plan", "empiricist-agent",
                   {"hypothesis_id": "hypothesis-0001", "hypothesis": hypothesis()})
        with self.assertRaises(HandoffError):
            flow.route("runner", "runner")
        self.assertEqual(flow.cursor, 2)

    def test_complete_cycle_has_role_order_runner_actor_and_receipt_bound_decision(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory, calls = TwoRoleMemory(self.root), []
        def model(label, prompt, schema, tokens, seed):
            calls.append((label, prompt))
            if label == "hypothesis":
                return hypothesis()
            if label == "empiricist_plan":
                return plan()
            if label == "empiricist_assessment":
                return {"execution_assessment": "Result receipt available.", "diagnostic_action": "adequate_for_interpretation"}
            if label == "hypothesis_decision":
                return {"action": "maintain", "basis_keywords": "Current receipt.", "next_inquiry": "Robustness.",
                        "evidence_ids": ["result-0001"]}
            raise AssertionError(label)
        record, _, _ = execute_four_role_attempt(session, 1, memory, model,
            lambda _, action: result(action), data_identity="fixture")
        self.assertEqual(record["status"], "completed", record)
        workflow = json.loads((session / "orchestrator.json").read_text())
        self.assertEqual([event["stage"] for event in workflow["events"]],
                         ["hypothesis", "empiricist_plan", "runner", "empiricist_assessment", "decision"])
        self.assertEqual(memory.nodes["result-0001"]["actor"], "runner")
        self.assertEqual(memory.nodes["result-0001"]["sample_identity"], "md5:" + "b" * 32)
        self.assertEqual(memory.nodes["result-0001"]["sample_recipe"]["n_obs"], "100")
        self.assertEqual(memory.nodes["result-0001"]["sample_recipe"]["filter"], "all")
        self.assertTrue((self.root / "role-history/hypothesis.jsonl").exists())
        self.assertTrue((self.root / "role-history/empiricist.jsonl").exists())
        self.assertTrue((self.root / "role-history/orchestrator.jsonl").exists())
        self.assertEqual([label for label, _ in calls],
                         ["hypothesis", "empiricist_plan", "empiricist_assessment", "hypothesis_decision"])
        self.assertEqual(memory.nodes["research-question"]["memory_type"], "ResearchQuestion")
        self.assertEqual(workflow["research_question"], QUESTION)
        self.assertEqual(workflow["routing_mode"], "deterministic")
        self.assertEqual(record["orchestration_mode"], "deterministic")
        self.assertTrue(all(event["sources"] for event in workflow["events"][1:]))
        empirical_prompt = next(prompt for label, prompt in calls if label == "empiricist_plan")
        self.assertIn("cluster='none' requires model_based SE", empirical_prompt)

    def test_failed_plan_does_not_publish_orphan_hypothesis_to_shared_graph(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)

        def model(label, prompt, schema, tokens, seed):
            if label == "hypothesis":
                return hypothesis()
            if label == "empiricist_plan":
                invalid = plan()
                invalid["analysis"]["terms"] = ["distinctive"]
                return invalid
            raise AssertionError(label)

        record, _, _ = execute_four_role_attempt(session, 1, memory, model,
            lambda _, action: result(action), data_identity="fixture", condition="C")
        self.assertEqual(record["status"], "failed")
        self.assertIn("research-question", memory.nodes)
        self.assertNotIn("hypothesis-0001", memory.nodes)

    def test_shared_memory_history_compacts_inputs_but_keeps_archive_and_outputs(self):
        history = RoleHistory(self.root)
        for index in range(10):
            history.add("hypothesis", "decision", {"basis": f"Prior result {index}"},
                        f"UNIQUE_PRIOR_FACT_{index}: p_value=0.{index:03d}\n"
                        "SHARED_PRIOR_FACT: same evidence\nSHARED_LINE: same")
        archive_before = (self.root / "role-history/hypothesis.jsonl").read_bytes()
        current = "current request\nSHARED_LINE: same"
        prompt = RoleHistory(self.root).prompt("hypothesis", current, shared_memory=True)
        self.assertIn("Prior result 9", prompt)
        self.assertNotIn("Prior result 0", prompt)
        for index in range(10):
            self.assertIn(f"UNIQUE_PRIOR_FACT_{index}: p_value=0.{index:03d}", archive_before.decode())
        self.assertNotIn("UNIQUE_PRIOR_FACT_", prompt)
        self.assertNotIn("SHARED_PRIOR_FACT: same evidence", prompt)
        self.assertEqual(prompt.count("SHARED_LINE: same"), 1)
        self.assertIn("current request", prompt)
        self.assertEqual((self.root / "role-history/hypothesis.jsonl").read_bytes(), archive_before)

    def test_memory_request_fetches_one_verified_packet_and_resumes_role(self):
        memory = Mock()
        trace = {"request_id": "r-fixture", "query": "result-0002", "selected_ids": ["result-0002"],
                 "upstream_selected_ids": ["hypothesis-0002", "result-0002"],
                 "context_manifest": {"source_node_ids": ["hypothesis-0002", "result-0002"]},
                 "anchors": [{"id": "result-0002"}],
                 "brief": "RESULTS\nevidence_id\tp_value\nresult-0002\t0.470\n"}
        memory.retrieve_tsv.return_value = (trace, {"result-0002": {"verified": True}})
        history, record, calls = RoleHistory(self.root), {}, []
        schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                  "required": ["answer"], "additionalProperties": False}

        def model(label, prompt, requested_schema, tokens, seed):
            calls.append((prompt, requested_schema, seed))
            if len(calls) == 1:
                return {"action": "retrieve_memory", "query": "result-0002", "answer": None}
            return {"answer": "grounded final answer"}

        answer, final_prompt, delivered_trace, delivered = call_role_with_memory(
            self.root, "hypothesis", "hypothesis", "hypothesis", "Current question", schema, 100, 7,
            model, memory, history, record, True,
            final_schema_builder=lambda extra: {**schema, "available_ids": sorted({"result-0001"} | set(extra))})
        self.assertEqual(answer, {"answer": "grounded final answer"})
        self.assertEqual(len(calls), 2)
        retrieve_branch, answer_branch = calls[0][1]["anyOf"]
        self.assertEqual(retrieve_branch["properties"]["action"]["const"], "retrieve_memory")
        self.assertEqual(retrieve_branch["properties"]["answer"], {"type": "null"})
        self.assertEqual(answer_branch["properties"]["action"]["const"], "answer")
        self.assertEqual(answer_branch["properties"]["query"], {"type": "null"})
        self.assertEqual(answer_branch["properties"]["answer"], schema)
        self.assertNotIn("action", calls[1][1]["properties"])
        self.assertEqual(calls[0][2], 7)
        self.assertEqual(calls[1][2], 8)
        self.assertEqual(delivered_trace, trace)
        self.assertIn("result-0002", delivered)
        self.assertEqual(calls[1][1]["available_ids"], ["result-0001", "result-0002"])
        self.assertIn(trace["brief"], final_prompt)
        self.assertEqual(memory.retrieve_tsv.call_args.args[0], "result-0002")
        self.assertEqual(memory.retrieve_tsv.call_args.kwargs["role"], "hypothesis")
        self.assertEqual(record["on_demand_retrievals"][0]["context_node_ids"],
                         ["hypothesis-0002", "result-0002"])
        self.assertEqual(record["on_demand_retrievals"][0]["result_ids"], ["result-0002"])
        self.assertEqual(json.loads((self.root / "memory-tool-01.json").read_text()), trace)
        request_artifact = json.loads((self.root / "memory-tool-01-request.json").read_text())
        self.assertEqual(request_artifact["request_response"],
                         {"action": "retrieve_memory", "query": "result-0002", "answer": None})
        self.assertIn("Current question", request_artifact["request_prompt"])
        archived = (self.root / "role-history/hypothesis.jsonl").read_text()
        self.assertIn("result-0002", archived)
        self.assertIn('"query": "result-0002"', archived)
        self.assertTrue((self.root / "memory-tool-01.json").exists())

    def test_plan_gate_maps_required_strata_and_cluster_to_runner_fields(self):
        contract = {**hypothesis(), "required_terms": ["typicality", "I(choice_set)", "I(user_label)"]}
        proposed = plan()
        proposed["analysis"].update(strata="choice_set", cluster="user_label")
        proposed["model_specification"].update(
            estimator="fixed_effects_logistic_regression", fixed_effects=["choice_set"],
            standard_error_method="cluster_robust", cluster_variables=["user_label"])
        result_plan = build_plan(contract, proposed)
        self.assertEqual(result_plan["required_term_resolutions"], [
            {"requested": "I(choice_set)", "executed": "strata:choice_set", "basis": "runner_strata_field"},
            {"requested": "I(user_label)", "executed": "cluster:user_label", "basis": "runner_cluster_field"},
        ])

    def test_plan_gate_canonicalizes_focal_label_from_immutable_hypothesis(self):
        contract = {**hypothesis(), "focal_test": {
            "effect_type": "main_effect", "term": "distinctive", "expected_direction": "positive"}}
        proposed = plan()
        proposed["analysis"]["terms"] = ["typicality", "distinctive"]
        proposed["model_specification"].update(focal_predictor="typicality",
                                                focal_terms=["typicality", "distinctive"])
        result_plan = build_plan(contract, proposed)
        self.assertEqual(result_plan["model_specification"]["dependent_variable"], "tagged")
        self.assertEqual(result_plan["model_specification"]["focal_predictor"], "distinctive")
        self.assertEqual(result_plan["model_specification_resolutions"], [
            {"field": "focal_predictor", "requested": "typicality", "effective": "distinctive",
             "basis": "immutable hypothesis focal test"},
        ])

    def test_plan_gate_canonicalizes_interaction_order_and_duplicate_terms_with_receipts(self):
        contract = {**hypothesis(), "focal_test": {
            "effect_type": "interaction", "term": "distinctive:typicality", "expected_direction": "positive"},
            "required_terms": ["typicality:distinctive", "typicality", "distinctive"]}
        proposed = plan()
        proposed["analysis"]["terms"] = ["typicality", "distinctive", "typicality:distinctive",
                                          "typicality:distinctive"]
        proposed["model_specification"].update(
            focal_predictor="typicality", focal_terms=["typicality", "distinctive",
                "typicality:distinctive", "distinctive:typicality"],
            moderator_interaction="typicality:distinctive")
        result_plan = build_plan(contract, proposed)
        self.assertEqual(result_plan["analysis"]["terms"],
                         ["typicality", "distinctive", "distinctive:typicality"])
        self.assertEqual(result_plan["model_specification"]["focal_terms"],
                         ["typicality", "distinctive", "distinctive:typicality"])
        self.assertEqual(result_plan["model_specification"]["moderator_interaction"],
                         "distinctive:typicality")
        self.assertEqual(result_plan["action_resolutions"][0]["effective_action"],
                         result_plan["analysis"])
        self.assertIn({"requested": "typicality:distinctive", "effective": "distinctive:typicality",
                       "basis": "commutative_interaction_canonicalization"},
                      result_plan["required_term_resolutions"])
        self.assertIn({"field": "focal_terms", "requested": "distinctive:typicality",
                       "effective": "distinctive:typicality", "basis": "duplicate_equivalent_model_term"},
                      result_plan["model_specification_resolutions"])

    def test_missing_executed_interaction_still_fails_the_gate(self):
        contract = {**hypothesis(), "focal_test": {
            "effect_type": "interaction", "term": "distinctive:typicality", "expected_direction": "positive"},
            "required_terms": ["typicality:distinctive", "typicality", "distinctive"]}
        proposed = plan()
        proposed["analysis"]["terms"] = ["typicality", "distinctive"]
        proposed["model_specification"].update(
            focal_predictor="typicality", focal_terms=["typicality", "distinctive"],
            moderator_interaction="typicality:distinctive")
        with self.assertRaisesRegex(ValueError, "omits required terms.*distinctive:typicality"):
            build_plan(contract, proposed)

    def test_claims_can_be_verified_against_a_later_on_demand_trace(self):
        class SourceCheck:
            def validate_context_claims(self, claims, trace):
                return [{"claim": claim, "status": "verified" if claim["evidence_id"] in trace["selected_ids"]
                         else "unverifiable"} for claim in claims]

        claim = {"evidence_id": "result-0002", "term": "interaction", "field": "p_value", "value": ".470"}
        initial = {"selected_ids": ["result-0001"], "delivered_equivalent_ids": []}
        requested = {"selected_ids": ["result-0002"], "delivered_equivalent_ids": []}
        outcome = validate_retrieved_claims(SourceCheck(), [claim], [initial, requested])
        self.assertEqual(outcome, [{"claim": claim, "status": "verified"}])

    def test_memory_request_is_bounded_and_unavailable_to_no_memory_roles(self):
        memory, history, record = Mock(), RoleHistory(self.root), {}
        schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                  "required": ["answer"], "additionalProperties": False}

        def requested(query):
            return lambda label, prompt, requested_schema, tokens, seed: {
                "action": "retrieve_memory", "query": query, "answer": None}

        for query in ("", "x" * 241, ["bad"]):
            with self.subTest(query=type(query).__name__), self.assertRaises(ValueError):
                call_role_with_memory(self.root, "hypothesis", "hypothesis", "hypothesis", "question",
                    schema, 80, 1, requested(query), memory, history, record, True)
        memory.retrieve_tsv.assert_not_called()
        self.assertFalse(record.get("on_demand_retrievals"))

        calls = []
        history.add("hypothesis", "hypothesis", {"claim": "prior role result"}, "full prior prompt")
        answer, _, _, _ = call_role_with_memory(self.root, "hypothesis", "hypothesis", "hypothesis", "question",
            schema, 80, 1, lambda *args: calls.append(args) or {"answer": "B response"},
            None, history, record, False)
        self.assertEqual(answer, {"answer": "B response"})
        self.assertEqual(len(calls), 1)
        self.assertNotIn("action", calls[0][2]["properties"])
        self.assertIn("prior role result", calls[0][1])
        self.assertIn("full prior prompt", (history.root / "hypothesis.jsonl").read_text())

    def test_context_overflow_after_memory_tool_keeps_receipt_and_stops(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        calls = []

        def model(label, prompt, schema, tokens, seed):
            calls.append((label, prompt))
            if label == "hypothesis" and len(calls) == 1:
                return {"action": "retrieve_memory", "query": "research-question", "answer": None}
            if label == "hypothesis":
                self.assertIn("research-question", prompt)
                raise ContextCapacityError("Retrieved context exceeds model capacity")
            raise AssertionError(label)

        research_runner = Mock(side_effect=AssertionError("No plan may run after capacity failure"))
        record, _, _ = execute_four_role_attempt(session, 1, memory, model, research_runner,
            data_identity="fixture", condition="C")
        self.assertEqual(record["failure_type"], "ContextCapacityError")
        self.assertEqual(record["status"], "needs-attention")
        self.assertEqual(len(record["on_demand_retrievals"]), 1)
        self.assertTrue((session / "memory-tool-01.json").exists())
        self.assertTrue((session / "memory-tool-01-request.json").exists())
        trace = json.loads((session / "memory-tool-01.json").read_text())
        self.assertEqual(trace["query"], "research-question")
        self.assertIn("research-question", trace["upstream_selected_ids"])
        self.assertIn("research-question", trace["context_manifest"]["source_node_ids"])
        self.assertTrue(any(anchor["id"] == "research-question" for anchor in trace["anchors"]))
        self.assertIn("TSV_CONTEXT_V1", calls[1][1])
        self.assertIn("research-question", calls[1][1])
        self.assertIn("research-question", record["on_demand_retrievals"][0]["context_node_ids"])
        self.assertIn("research-question", (self.root / "role-history/hypothesis.jsonl").read_text())
        self.assertEqual(memory_iteration.fatal_attempt_stop_reason(record), "context_capacity_no_retry")
        self.assertEqual(len(calls), 2)
        research_runner.assert_not_called()

    def test_shared_memory_keeps_full_hypothesis_chat_history_when_disabled(self):
        history = RoleHistory(self.root)
        history.add("hypothesis", "decision", {"basis": "Observed prior result."},
                    "FULL_PRIOR_TSV_PACKET: p_value=0.192")
        prompt = RoleHistory(self.root).prompt("hypothesis", "current request")
        self.assertIn("Observed prior result", prompt)
        self.assertIn("FULL_PRIOR_TSV_PACKET", prompt)

    def test_c_context_overflow_is_attention_state_and_never_publishes_candidate(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)

        def model(label, prompt, schema, tokens, seed):
            if label == "hypothesis":
                return hypothesis()
            if label == "empiricist_plan":
                raise ContextCapacityError("Full context does not fit")
            raise AssertionError(label)

        runner = Mock(side_effect=AssertionError("Runner must not execute"))
        record, _, _ = execute_four_role_attempt(session, 1, memory, model, runner,
            data_identity="fixture", condition="C")
        self.assertEqual(record["status"], "needs-attention")
        self.assertEqual(record["failure_domain"], "context_capacity")
        self.assertEqual(record["analysis_status"], "not_started")
        self.assertNotIn("hypothesis-0001", memory.nodes)
        runner.assert_not_called()

    def test_empiricist_gate_mismatch_gets_one_repair_in_same_attempt(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory, calls = TwoRoleMemory(self.root), []
        contract = {**hypothesis(), "required_terms": ["typicality", "self_claim_lv"]}
        def model(label, prompt, schema, tokens, seed):
            calls.append((label, prompt))
            if label == "hypothesis":
                return contract
            if label == "empiricist_plan":
                return plan()
            if label == "empiricist_plan_repair":
                fixed = plan()
                fixed["analysis"]["terms"] = ["typicality", "selfclaim_lv"]
                fixed["model_specification"]["control_variables"] = ["selfclaim_lv"]
                return fixed
            if label == "empiricist_assessment":
                return {"execution_assessment": "Result receipt available.",
                        "diagnostic_action": "adequate_for_interpretation"}
            if label == "hypothesis_decision":
                return {"action": "maintain", "basis_keywords": "Observed result.",
                        "next_inquiry": "Check robustness.", "evidence_ids": ["result-0001"]}
            raise AssertionError(label)

        runner = Mock(return_value=result({"family": "logit", "outcome": "tagged",
            "terms": ["typicality", "selfclaim_lv"], "strata": "none", "cluster": "none", "filter": "all"}))
        record, _, _ = execute_four_role_attempt(session, 1, memory, model, runner,
            data_identity="fixture", condition="C")
        self.assertEqual(record["status"], "completed", record)
        self.assertEqual(record["plan_repair_status"], "succeeded")
        self.assertEqual([label for label, _ in calls], ["hypothesis", "empiricist_plan",
                         "empiricist_plan_repair", "empiricist_assessment", "hypothesis_decision"])
        self.assertIn("self_claim_lv", next(prompt for label, prompt in calls
                                             if label == "empiricist_plan_repair"))
        self.assertIn("hypothesis-0001", memory.nodes)
        self.assertIn("model-0001", memory.nodes)
        self.assertTrue((session / "empiricist-plan.json").exists())
        self.assertTrue((session / "empiricist-plan-repair.json").exists())
        runner.assert_called_once()

    def test_failed_single_plan_repair_stops_before_graph_publication_and_runner(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        calls = []
        contract = {**hypothesis(), "required_terms": ["typicality", "selfclaim_lv"]}
        def model(label, prompt, schema, tokens, seed):
            calls.append(label)
            if label == "hypothesis":
                return contract
            if label in ("empiricist_plan", "empiricist_plan_repair"):
                return plan()
            raise AssertionError(label)

        runner = Mock(side_effect=AssertionError("Invalid plan must not reach Runner"))
        record, _, _ = execute_four_role_attempt(session, 1, memory, model, runner,
            data_identity="fixture", condition="C")
        self.assertEqual(record["failure_type"], "PlanRepairError")
        self.assertEqual(record["plan_repair_status"], "failed")
        self.assertEqual(calls, ["hypothesis", "empiricist_plan", "empiricist_plan_repair"])
        self.assertNotIn("hypothesis-0001", memory.nodes)
        self.assertNotIn("model-0001", memory.nodes)
        self.assertTrue((session / "plan-repair-failure.json").exists())
        history = (session.parent / "role-history/empiricist.jsonl").read_text().splitlines()
        rejection = json.loads(history[-1])
        self.assertEqual(rejection["stage"], "plan_rejection")
        self.assertIn("required", rejection["value"]["gate_error"])
        runner.assert_not_called()

    def test_data_insufficient_plan_does_not_use_repair_call(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        calls = []
        def model(label, prompt, schema, tokens, seed):
            calls.append(label)
            if label == "hypothesis":
                return hypothesis()
            if label == "empiricist_plan":
                unavailable = plan()
                unavailable["analysis"] = None
                unavailable["model_specification"].update(
                    design_type="difference_in_differences", estimator="difference_in_differences")
                return unavailable
            raise AssertionError(label)

        runner = Mock(side_effect=AssertionError("Data-insufficient design must not reach Runner"))
        record, _, _ = execute_four_role_attempt(session, 1, memory, model, runner,
            data_identity="fixture", condition="C")
        self.assertEqual(record["failure_type"], "DataInsufficientDesign")
        self.assertEqual(record["status"], "needs-attention")
        self.assertEqual(calls, ["hypothesis", "empiricist_plan"])
        self.assertNotIn("hypothesis-0001", memory.nodes)
        self.assertTrue((session / "plan-validation-rejection.json").exists())
        runner.assert_not_called()

    def test_repair_response_preserves_data_insufficient_classification(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        calls = []
        contract = {**hypothesis(), "required_terms": ["typicality", "selfclaim_lv"]}
        def model(label, prompt, schema, tokens, seed):
            calls.append(label)
            if label == "hypothesis":
                return contract
            if label == "empiricist_plan":
                return plan()
            if label == "empiricist_plan_repair":
                unavailable = plan()
                unavailable["analysis"] = None
                unavailable["model_specification"].update(
                    design_type="difference_in_differences", estimator="difference_in_differences")
                return unavailable
            raise AssertionError(label)

        runner = Mock(side_effect=AssertionError("Data-insufficient plan must not reach Runner"))
        record, _, _ = execute_four_role_attempt(session, 1, memory, model, runner,
            data_identity="fixture", condition="C")
        self.assertEqual(record["failure_type"], "DataInsufficientDesign")
        self.assertEqual(record["plan_repair_status"], "failed")
        self.assertEqual(calls, ["hypothesis", "empiricist_plan", "empiricist_plan_repair"])
        self.assertNotIn("hypothesis-0001", memory.nodes)
        self.assertTrue((session / "plan-repair-failure.json").exists())
        runner.assert_not_called()

    def test_context_overflow_during_single_repair_is_terminal(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        calls = []
        contract = {**hypothesis(), "required_terms": ["typicality", "selfclaim_lv"]}
        def model(label, prompt, schema, tokens, seed):
            calls.append(label)
            if label == "hypothesis":
                return contract
            if label == "empiricist_plan":
                return plan()
            if label == "empiricist_plan_repair":
                raise ContextCapacityError("Repair prompt exceeds context")
            raise AssertionError(label)

        runner = Mock(side_effect=AssertionError("Context overflow must stop before Runner"))
        record, _, _ = execute_four_role_attempt(session, 1, memory, model, runner,
            data_identity="fixture", condition="C")
        self.assertEqual(record["status"], "needs-attention")
        self.assertEqual(record["failure_type"], "ContextCapacityError")
        self.assertEqual(calls, ["hypothesis", "empiricist_plan", "empiricist_plan_repair"])
        self.assertNotIn("hypothesis-0001", memory.nodes)
        runner.assert_not_called()

    def test_terminal_four_role_receipts_reach_common_retry_gate(self):
        cases = (
            ("context", "ContextCapacityError", "context_capacity_no_retry"),
            # A repaired-plan rejection is a recorded attempt. The next bounded
            # attempt may adapt; only non-recoverable capacity failures stop the arm.
            ("repair", "PlanRepairError", None),
            ("unsupported", "DataInsufficientDesign", None),
        )
        for name, failure_type, expected_stop in cases:
            with self.subTest(failure=failure_type):
                session = self.root / name / "attempt-0001"
                session.mkdir(parents=True)
                (session.parent / "config.json").write_text(json.dumps({"question": QUESTION}))
                memory = TwoRoleMemory(session.parent)

                def model(label, prompt, schema, tokens, seed):
                    if label == "hypothesis":
                        return hypothesis()
                    if failure_type == "ContextCapacityError" and label == "empiricist_plan":
                        raise ContextCapacityError("Synthetic overflow")
                    if label == "empiricist_plan":
                        invalid = plan()
                        if failure_type == "PlanRepairError":
                            invalid["analysis"]["terms"] = ["distinctive"]
                            return invalid
                        invalid["analysis"] = None
                        invalid["model_specification"].update(
                            design_type="difference_in_differences", estimator="difference_in_differences")
                        return invalid
                    if label == "empiricist_plan_repair":
                        return {**plan(), "analysis": {**plan()["analysis"], "terms": ["distinctive"]}}
                    raise AssertionError(label)

                research_runner = Mock(side_effect=AssertionError("Terminal plan must not run"))
                record, _, _ = execute(session, 1, "memory", memory, model, research_runner,
                    condition="C", data_identity="fixture")
                attempt_file = session / "attempt.json"
                attempt_file.write_text(json.dumps(record))
                self.assertEqual(json.loads(attempt_file.read_text())["failure_type"], failure_type)
                self.assertEqual(memory_iteration.fatal_attempt_stop_reason(record), expected_stop)
                self.assertNotIn("hypothesis-0001", memory.nodes)
                research_runner.assert_not_called()

    def test_c_followup_inquiry_does_not_change_immutable_research_question(self):
        root = self.root / "C-followup"
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"question": QUESTION}))
        memory = TwoRoleMemory(root)
        calls = []

        def run_cycle(index, inquiry, previous_plan=None, previous_graph=None):
            session = root / f"attempt-{index:04d}"
            session.mkdir()
            def model(label, prompt, schema, tokens, seed):
                calls.append((index, label, prompt))
                if label == "hypothesis":
                    proposed = hypothesis()
                    if index > 1:
                        proposed["adaptation"] = {
                            "decision": "refine", "basis": "Use the prior result.",
                            "evidence_ids": ["result-0001"]}
                    return proposed
                if label == "empiricist_plan":
                    return plan()
                if label == "empiricist_assessment":
                    return {"execution_assessment": "Result available.",
                            "diagnostic_action": "adequate_for_interpretation"}
                if label == "hypothesis_decision":
                    return {"action": "additional_test", "basis_keywords": "Current result.",
                            "next_inquiry": "Check a rival explanation.",
                            "evidence_ids": [f"result-{index:04d}"]}
                raise AssertionError(label)

            return execute(session, index, "memory", memory, model,
                lambda _, action: result(action), data_identity="fixture", condition="C",
                research_question=QUESTION, query=inquiry,
                previous_plan=previous_plan, previous_graph=previous_graph)

        first, first_plan, _ = run_cycle(1, QUESTION)
        followup = "Check a rival explanation."
        second, _, _ = run_cycle(2, followup, first_plan, first)
        self.assertEqual(first["status"], "completed", first)
        self.assertEqual(second["status"], "completed", second)
        self.assertEqual(memory.nodes["research-question"]["payload"], {"question": QUESTION})
        trace = json.loads((root / "attempt-0002/hypothesis-retrieval.json").read_text())
        self.assertIn(QUESTION, trace["query"])
        self.assertIn(followup, trace["query"])
        self.assertFalse(any(label == "orchestrator" for _, label, _ in calls))
        receipt = json.loads((root / "attempt-0002/orchestrator.json").read_text())
        self.assertEqual(receipt["research_question"], QUESTION)
        self.assertEqual(receipt["current_inquiry"], followup)
        hypothesis_prompt = next(prompt for attempt, label, prompt in reversed(calls)
                                 if attempt == 2 and label == "hypothesis")
        self.assertIn(QUESTION, hypothesis_prompt)
        self.assertIn(followup, hypothesis_prompt)
        self.assertIn("typicality\t.02\t.015\t.192", hypothesis_prompt)
        self.assertNotIn("TURN decision\nINPUT:", hypothesis_prompt)
        self.assertIn("TURN decision\nAGENT OUTPUT:", hypothesis_prompt)

    def test_controller_blocks_mismatched_runner_receipt_before_execution(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        def model(label, prompt, schema, tokens, seed):
            return {"hypothesis": hypothesis(), "empiricist_plan": plan()}[label]
        runner = Mock(side_effect=AssertionError("Runner must not execute"))
        original_route = FourRoleOrchestrator.route
        def corrupt_receipt(flow, stage, actor, artifact=None):
            if stage == "runner":
                artifact = {**artifact, "model_id": "model-9999"}
            return original_route(flow, stage, actor, artifact)
        with patch.object(FourRoleOrchestrator, "route", corrupt_receipt):
            record, _, _ = execute_four_role_attempt(session, 1, memory, model, runner, data_identity="fixture")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_type"], "HandoffError")
        runner.assert_not_called()
        receipt = json.loads((session / "orchestrator.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual([event["stage"] for event in receipt["events"]], ["hypothesis", "empiricist_plan"])

    def test_inference_deadline_timeout_is_censored_not_tool_failure(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        def model(label, prompt, schema, tokens, seed):
            if label == "hypothesis":
                error = RuntimeIsolationError("Private runtime transport/cache boundary failed")
                error.deadline_exhausted = True
                raise error from TimeoutError()
            raise AssertionError(label)
        record, _, _ = execute_four_role_attempt(session, 1, memory, model,
            lambda _, action: result(action), data_identity="fixture")
        self.assertEqual(record["status"], "censored")
        self.assertEqual(record["failure_domain"], "inference_deadline_censor")
        receipt = json.loads((session / "orchestrator.json").read_text())
        self.assertEqual(receipt["status"], "censored")
        self.assertEqual([event["stage"] for event in receipt["events"]], ["hypothesis"])

    def test_early_runtime_timeout_remains_failed(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        def model(label, prompt, schema, tokens, seed):
            if label == "hypothesis":
                error = RuntimeIsolationError("Private runtime transport/cache boundary failed")
                error.deadline_exhausted = False
                raise error from TimeoutError()
            raise AssertionError(label)
        record, _, _ = execute_four_role_attempt(session, 1, memory, model,
            lambda _, action: result(action), data_identity="fixture")
        self.assertEqual(record["status"], "failed")

    def test_role_history_persists_full_exchanges_without_cross_role_leakage(self):
        history = RoleHistory(self.root)
        history.add("hypothesis", "decision", {"basis": "observed result"},
                    "RESULT_FACTS: coefficient=0.02; p_value=0.192")
        restored = RoleHistory(self.root)
        prompt = restored.prompt("hypothesis", "new inquiry")
        other_role = restored.prompt("empiricist", "new analysis")
        self.assertIn("coefficient=0.02", prompt)
        self.assertIn("observed result", prompt)
        self.assertNotIn("coefficient=0.02", other_role)

    def test_first_attempt_schema_cannot_claim_prior_history(self):
        schema = hypothesis_schema(1)["properties"]
        self.assertEqual(schema["adaptation"]["properties"]["decision"], {"const": "initial"})
        self.assertEqual(schema["adaptation"]["properties"]["basis"],
                         {"const": "Initial attempt; no prior evidence."})
        self.assertEqual(schema["adaptation"]["properties"]["evidence_ids"]["maxItems"], 0)
        self.assertEqual(schema["historical_claims"]["maxItems"], 0)
        later = hypothesis_schema(2, {"result-0001"})["properties"]
        self.assertEqual(later["historical_claims"]["maxItems"], 2)
        self.assertNotIn("initial", later["adaptation"]["properties"]["decision"]["enum"])
        self.assertEqual(later["adaptation"]["properties"]["evidence_ids"]["items"],
                         {"enum": ["result-0001"]})

    def test_controller_rejects_wrong_question_actor_and_changed_source(self):
        flow = FourRoleOrchestrator(self.root, 1, research_question=QUESTION, inquiry="Current inquiry")
        initial = {"state": "initial", "research_question": QUESTION, "current_inquiry": "Current inquiry"}
        for actor, receipt in (("runner", initial),
                               ("hypothesis-agent", {**initial, "research_question": "Changed RQ"})):
            with self.assertRaises(HandoffError):
                flow.route("hypothesis", actor, receipt)
            self.assertEqual(flow.cursor, 0)
        flow.route("hypothesis", "hypothesis-agent", initial)
        path = self.root / "hypothesis.json"
        path.write_text(json.dumps(hypothesis()))
        flow.route("empiricist_plan", "empiricist-agent",
                   {"hypothesis_id": "hypothesis-0001", "hypothesis": hypothesis()})
        (self.root / "plan.json").write_text(json.dumps(plan()))
        path.write_text(json.dumps({**hypothesis(), "hypothesis": "Tampered hypothesis"}))
        with self.assertRaisesRegex(HandoffError, "source changed"):
            flow.route("runner", "runner", {"model_id": "model-0001", "analysis": plan()["analysis"],
                                             "model_specification": plan()["model_specification"]})
        self.assertEqual(flow.cursor, 2)

    def test_a_and_b_run_without_shared_memory_and_b_history_retrieves_own_result(self):
        for condition in ("A", "B"):
            with self.subTest(condition=condition):
                root = self.root / condition
                root.mkdir()
                session = root / "attempt-0001"
                session.mkdir()
                calls = []
                current_index = 1

                def model(label, prompt, schema, tokens, seed):
                    calls.append((label, prompt))
                    if label == "hypothesis":
                        proposed = hypothesis()
                        if current_index > 1:
                            proposed["adaptation"] = {"decision": "revise", "basis": "Prior result was not significant.",
                                                       "evidence_ids": ["result-0001"]}
                        return proposed
                    if label == "empiricist_plan":
                        return plan()
                    if label == "empiricist_assessment":
                        return {"execution_assessment": "Current result available.",
                                "diagnostic_action": "adequate_for_interpretation"}
                    if label == "hypothesis_decision":
                        return {"action": "maintain", "basis_keywords": "Current result.",
                                "next_inquiry": "Check a rival explanation.",
                                "evidence_ids": ["result-%04d" % current_index]}
                    raise AssertionError(label)

                record, _, _ = execute_four_role_attempt(session, 1, None, model,
                    lambda _, action: result(action), data_identity="fixture", condition=condition)
                self.assertEqual(record["status"], "completed", record)
                self.assertFalse(record["shared_memory_enabled"])
                self.assertEqual(record["retrieved_ids"], [])
                self.assertFalse((root / "memory").exists())
                self.assertEqual([label for label, _ in calls],
                                 ["hypothesis", "empiricist_plan", "empiricist_assessment", "hypothesis_decision"])
                self.assertIn("No cross-attempt evidence supplied", calls[0][1])
                if condition == "B":
                    next_session = root / "attempt-0002"
                    next_session.mkdir()
                    calls.clear()
                    current_index = 2
                    record, _, _ = execute_four_role_attempt(next_session, 2, None, model,
                        lambda _, action: result(action), data_identity="fixture", condition="B")
                    h_prompt = next(prompt for label, prompt in calls if label == "hypothesis")
                    self.assertEqual(record["status"], "completed", record)
                    self.assertIn("typicality\t.02\t.015\t.192", h_prompt)
                    self.assertIn("result-0001", h_prompt)

    def test_missing_or_external_hypothesis_receipt_cannot_advance(self):
        flow = FourRoleOrchestrator(self.root, 1, research_question=QUESTION, inquiry=QUESTION)
        flow.route("hypothesis", "hypothesis-agent", {"state": "initial", "research_question": QUESTION,
                                                     "current_inquiry": QUESTION})
        receipt = {"hypothesis_id": "hypothesis-0001", "hypothesis": hypothesis()}
        with self.assertRaises(HandoffError):
            flow.route("empiricist_plan", "empiricist-agent", receipt)
        with tempfile.TemporaryDirectory() as elsewhere:
            source = Path(elsewhere) / "hypothesis.json"
            source.write_text(json.dumps(hypothesis()))
            (self.root / "hypothesis.json").symlink_to(source)
            with self.assertRaisesRegex(HandoffError, "outside this trajectory"):
                flow.route("empiricist_plan", "empiricist-agent", receipt)
        self.assertEqual(flow.cursor, 1)

    def test_post_result_handoffs_reject_stale_ids_and_missing_assessment(self):
        for target in ("empiricist_assessment", "decision"):
            with self.subTest(target=target):
                root = self.root / target
                root.mkdir()
                session = root / "attempt-0001"
                session.mkdir()
                model = Mock(side_effect=lambda label, *args: {
                    "hypothesis": hypothesis(), "empiricist_plan": plan(),
                    "empiricist_assessment": {"execution_assessment": "Result available.",
                                              "diagnostic_action": "additional_test"}}[label])
                original_route = FourRoleOrchestrator.route
                def corrupt(flow, stage, actor, artifact=None):
                    if stage == target:
                        if target == "empiricist_assessment":
                            artifact = {**artifact, "result_id": "result-9999"}
                        else:
                            (session / "empiricist-assessment.json").unlink()
                    return original_route(flow, stage, actor, artifact)
                with patch.object(FourRoleOrchestrator, "route", corrupt):
                    record, _, _ = execute_four_role_attempt(session, 1, None, model,
                        lambda _, action: result(action), data_identity="fixture", condition="B")
                self.assertEqual(record["failure_type"], "HandoffError", record)
                self.assertEqual(record["analysis_status"], "succeeded")
                self.assertTrue((session / "tool-result.json").exists())
                self.assertNotIn("hypothesis_decision", [call.args[0] for call in model.call_args_list])

    def test_result_survives_post_analysis_inference_deadline(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        def model(label, *args):
            if label == "empiricist_assessment":
                error = RuntimeIsolationError("Allocation deadline")
                error.deadline_exhausted = True
                raise error from TimeoutError()
            return {"hypothesis": hypothesis(), "empiricist_plan": plan()}[label]
        record, _, _ = execute_four_role_attempt(session, 1, memory, model,
            lambda _, action: result(action), data_identity="fixture")
        self.assertEqual(record["status"], "censored", record)
        self.assertEqual(record["analysis_status"], "succeeded")
        self.assertEqual(record["statistical_assessment"]["support"], "not_supported")
        self.assertIn("result-0001", memory.nodes)
        receipt = json.loads((session / "orchestrator.json").read_text())
        self.assertEqual(receipt["status"], "censored")
        self.assertEqual([event["stage"] for event in receipt["events"]],
                         ["hypothesis", "empiricist_plan", "runner", "empiricist_assessment"])

    def test_scientific_decision_still_requires_current_result(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        def model(label, *args):
            return {"hypothesis": hypothesis(), "empiricist_plan": plan(),
                    "empiricist_assessment": {"execution_assessment": "Available.",
                                              "diagnostic_action": "additional_test"},
                    "hypothesis_decision": {"action": "maintain", "basis_keywords": "Wrong receipt.",
                                            "next_inquiry": "Next test.", "evidence_ids": ["result-9999"]}}[label]
        record, _, _ = execute_four_role_attempt(session, 1, None, model,
            lambda _, action: result(action), data_identity="fixture", condition="B")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["decision_status"], "failed")
        self.assertIn("current result receipt", record["diagnostic"])
        self.assertFalse((session / "decision.json").exists())
        self.assertEqual(json.loads((session / "orchestrator.json").read_text())["status"], "failed")

    def test_later_attempt_rejects_initial_or_unavailable_adaptation(self):
        root = self.root / "later"
        root.mkdir()
        history = RoleHistory(root)
        history.add("hypothesis", "decision", {"evidence_ids": ["result-0001"]}, "prior result")
        session = root / "attempt-0002"
        session.mkdir()
        def model(label, prompt, schema, tokens, seed):
            proposed = hypothesis()
            proposed["adaptation"] = {"decision": "initial", "basis": "No history.", "evidence_ids": []}
            return proposed

        runner = Mock(side_effect=AssertionError("Invalid hypothesis must stop before analysis"))
        record, _, _ = execute_four_role_attempt(session, 2, None, model, runner,
            data_identity="fixture", condition="B")
        self.assertEqual(record["status"], "failed")
        self.assertIn("non-initial", record["diagnostic"])
        runner.assert_not_called()

    def test_three_condition_start_policy(self):
        from memory_iteration import may_start_attempt
        self.assertTrue(may_start_attempt("A", 1, 1000, 3600))
        self.assertFalse(may_start_attempt("A", 2, 10, 3600))
        self.assertTrue(may_start_attempt("B", 4, 2000, 3600))
        self.assertFalse(may_start_attempt("C", 4, 3555, 3600))

    def test_three_condition_entrypoint_isolated_from_legacy_memory_series(self):
        original = (memory_iteration.ITERATIONS, memory_iteration.TRAJECTORIES,
                    memory_iteration.MAX_ROUNDS, memory_iteration.MEMORY_FORMAT,
                    memory_iteration.TOPOLOGY, memory_iteration.SOURCE_FILES,
                    memory_iteration.execute_attempt, memory_iteration.BENCHMARK_CONDITIONS)
        try:
            configured = configure_three_condition()
            self.assertEqual(configured.BENCHMARK_CONDITIONS, ("A", "B", "C"))
            self.assertEqual(configured.MAX_ROUNDS, 1)
            self.assertEqual(configured.TRAJECTORIES,
                             configured.ROOT / "workspaces/three-condition-benchmark-20261024/trajectories")
            self.assertNotEqual(configured.ITERATIONS, original[0])
        finally:
            (memory_iteration.ITERATIONS, memory_iteration.TRAJECTORIES,
             memory_iteration.MAX_ROUNDS, memory_iteration.MEMORY_FORMAT,
             memory_iteration.TOPOLOGY, memory_iteration.SOURCE_FILES,
             memory_iteration.execute_attempt, memory_iteration.BENCHMARK_CONDITIONS) = original

    def test_first_attempt_rejects_claim_to_missing_prior_result(self):
        session = self.root / "attempt-0001"
        session.mkdir()
        memory = TwoRoleMemory(self.root)
        proposed = hypothesis()
        proposed["historical_claims"] = [{"evidence_id": "missing-result", "term": "typicality",
            "field": "p_value", "value": "garbage"}]
        def model(label, prompt, schema, tokens, seed):
            if label == "hypothesis":
                return proposed
            if label == "empiricist_plan":
                return plan()
            if label == "empiricist_assessment":
                return {"execution_assessment": "Current result available.", "diagnostic_action": "adequate_for_interpretation"}
            if label == "hypothesis_decision":
                return {"action": "maintain", "basis_keywords": "Current result.", "next_inquiry": "Next test.",
                        "evidence_ids": ["result-0001"]}
            raise AssertionError(label)
        record, _, _ = execute_four_role_attempt(session, 1, memory, model,
            lambda _, action: result(action), data_identity="fixture")
        self.assertEqual(record["status"], "failed", record)
        response_claims = json.loads((session / "hypothesis-response.json").read_text())["historical_claims"]
        self.assertEqual(len(response_claims), 1)
        self.assertNotIn("hypothesis-0001", memory.nodes)

    def test_future_entrypoint_rejects_unreviewed_nonmemory_arm(self):
        with self.assertRaises(ValueError):
            execute(None, 1, "no-memory", None, None, None)

    def test_dense_retrieval_is_an_explicit_isolated_configuration(self):
        memory = TwoRoleMemory(self.root, retrieval_strategy="dense")
        self.assertEqual(memory.retrieval_strategy, "dense")
        with self.assertRaises(ValueError):
            TwoRoleMemory(self.root, retrieval_strategy="flat")

    def test_four_role_runner_configuration_and_repeated_stage_receipts(self):
        original = (memory_iteration.ITERATIONS, memory_iteration.TRAJECTORIES, memory_iteration.MAX_ROUNDS,
                    memory_iteration.MEMORY_FORMAT, memory_iteration.TOPOLOGY, memory_iteration.SOURCE_FILES,
                    memory_iteration.execute_attempt)
        try:
            configured = configure()
            self.assertEqual(configured.TOPOLOGY, "orchestrator-hypothesis-empiricist-runner")
            self.assertEqual(configured.MEMORY_FORMAT, "tsv")
            self.assertIn("tools/four_role_discovery.py", configured.SOURCE_FILES)
            counts = {}
            self.assertEqual([memory_iteration.stage_receipt_label(counts, "orchestrator") for _ in range(3)],
                             ["orchestrator", "orchestrator-2", "orchestrator-3"])
        finally:
            (memory_iteration.ITERATIONS, memory_iteration.TRAJECTORIES, memory_iteration.MAX_ROUNDS,
             memory_iteration.MEMORY_FORMAT, memory_iteration.TOPOLOGY, memory_iteration.SOURCE_FILES,
             memory_iteration.execute_attempt) = original

    def test_four_role_auditor_requires_runner_actor_and_workflow_receipt(self):
        import audit_four_role_iteration as auditor
        self.assertEqual(auditor.EXPECTED_ACTORS["ResultAssessment"], "runner")
        self.assertEqual(auditor.RETRIEVAL_ROLES, (("hypothesis", "hypothesis"), ("empiricist", "analysis")))
        self.assertTrue(auditor.REQUIRE_WORKFLOW)


if __name__ == "__main__":
    unittest.main()
