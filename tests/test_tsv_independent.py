"""Independent synthetic TSV delivery and offline QA regressions; SGE only."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from research_memory import evidence
from research_memory.tabular import render_tsv
from research_memory.two_role import TwoRoleMemory
from test_two_role_independent import hypothesis, analysis
from two_role_discovery import build_plan
from audit_tsv_iteration import audit_trace, audit_round, validate_runtime_contract, expected_context_nodes


class IndependentTsvTests(unittest.TestCase):
    def test_research_question_is_expected_even_before_any_scientific_nodes(self):
        root = {"research-question": {"memory_type": "ResearchQuestion"}}
        self.assertEqual(expected_context_nodes(root, 1, "hypothesis"), {"research-question"})

    def setUp(self):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Reviewed SGE allocation required")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.trajectory = self.root / "trajectory"
        self.trajectory.mkdir()
        self.task = {"question": "Synthetic question.", "embedding_backend": "lexical", "retrieval_strategy": "ppr",
                     "data_identity": {"sha256": "synthetic-data"},
                     "model_identity": {"weights_sha256": "synthetic-model", "server_binary_sha256": "synthetic-server"}}
        self.save(self.trajectory / "config.json", self.task)
        self.memory = TwoRoleMemory(self.trajectory)
        self.memory.ensure_research_question(self.task["question"], self.trajectory / "config.json")
        self.states = []
        for index in range(1, 4):
            self.publish(index)
        self.trace, _ = self.memory.retrieve_tsv("typicality", role="hypothesis")

    def tearDown(self):
        self.temp.cleanup()

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def publish(self, index):
        folder = self.trajectory / f"attempt-{index:04d}"
        contract, request = hypothesis(), analysis()
        if index == 1:
            contract["required_terms"] = ["typicality", "self_claim_lv"]
            request["analysis"]["terms"].append("selfclaim_lv")
            request["model_specification"]["control_variables"] = ["selfclaim_lv"]
        if index == 3:
            contract["hypothesis"] = "Typicality has negative curvature."
            contract["focal_test"] = {"effect_type": "curvature", "term": "I(typicality^2)", "expected_direction": "negative"}
            contract["required_terms"] = ["typicality", "I(typicality^2)"]
            request["analysis"]["terms"].append("I(typicality^2)")
            request["model_specification"].update(focal_predictor="I(typicality^2)",
                focal_terms=["typicality", "I(typicality^2)"], control_variables=[])
        trace, _ = self.memory.retrieve_tsv("typicality", role="hypothesis")
        self.save(folder / "hypothesis-retrieval.json", trace)
        h = self.memory.publish_hypothesis(index, contract, self.save(folder / "hypothesis.json", contract),
            self.task["question"], self.trajectory / "config.json")
        trace, _ = self.memory.retrieve_tsv("typicality", role="analysis")
        self.save(folder / "analysis-retrieval.json", trace)
        plan = build_plan(contract, request)
        plan_source = self.save(folder / "plan.json", plan)
        model = self.memory.publish_model(index, h, plan, plan_source)
        p = "0.5" if index == 2 else "0.1"
        tool = {"returncode": 0, "effective_action": plan["analysis"],
            "metrics": [{"n": "100", "sample_membership_md5": "a" * 32}],
            "estimates": [{"term": "typicality", "coefficient": "0.1", "se": "0.1", "p_value": p}],
            "diagnostic": "A tab\tand newline\nplus literal \\N remain exact."}
        if index == 3:
            tool["estimates"].append({"term": "I(typicality^2)", "coefficient": "-0.2", "se": "0.1", "p_value": "0.1"})
        result = self.memory.publish_result(index, model, self.save(folder / "tool-result.json", tool), plan_source,
            data_identity=self.task["data_identity"], sample_identity="md5:" + "a" * 32)
        self.save(folder / "statistical-assessment.json", evidence.statistical_assessment(plan, tool))
        decision = {"action": "maintain", "basis_keywords": "Own results.", "next_inquiry": "Own follow-up.", "evidence_ids": [result]}
        d = self.memory.publish_decision(index, h, result, decision, self.save(folder / "decision.json", decision))
        self.states.append({"attempt": index, "status": "completed", "analysis_status": "succeeded",
            "decision_status": "succeeded", "hypothesis_id": h, "model_id": model, "result_id": result, "decision_id": d})

    def check_trace(self, trace):
        return audit_trace(trace, self.memory.nodes, self.trajectory, 4, "hypothesis", evidence, render_tsv)

    def refresh(self, trace):
        trace["brief"] = render_tsv(trace["context_packet"])
        return trace

    def test_all_positive_nonpositive_and_curvature_results_reach_actual_tsv(self):
        checked = self.check_trace(self.trace)
        self.assertEqual(checked["group_count"], 3)
        self.assertEqual(checked["delivered_support"]["supported"], 2)
        self.assertEqual(checked["delivered_support"]["not_supported"], 1)
        self.assertEqual(checked["delivered_support"]["curvature"], 1)
        self.assertIn('"A tab\tand newline\nplus literal \\N remain exact."', self.trace["brief"])
        self.assertIn("observational_association", self.trace["brief"])
        self.assertIn("logistic_regression", self.trace["brief"])
        self.assertIn("dependent_variable", self.trace["brief"])
        self.assertIn("focal_terms", self.trace["brief"])
        self.assertIn("unique_data_dictionary_identifier_match_ignoring_underscores", self.trace["brief"])
        self.assertIn("self_claim_lv", self.trace["brief"])
        self.assertIn("I(typicality^2)", self.trace["brief"])
        self.assertIn("control_variables", self.trace["brief"])
        self.assertIn("execution_support", self.trace["brief"])

    def test_silent_group_drop_rejected_even_with_new_valid_tsv(self):
        trace = copy.deepcopy(self.trace)
        removed = trace["context_packet"]["evidence"].pop()
        trace["selected_ids"].remove(removed["evidence_id"])
        with self.assertRaises(ValueError):
            self.check_trace(self.refresh(trace))

    def test_silent_coefficient_drop_rejected_even_with_new_valid_tsv(self):
        trace = copy.deepcopy(self.trace)
        view = next(view for view in trace["context_packet"]["evidence"] if len(view["estimates"]) == 2)
        view["estimates"].pop()
        with self.assertRaises(ValueError):
            self.check_trace(self.refresh(trace))

    def test_hypothesis_prose_not_promoted_to_machine_result(self):
        trace = copy.deepcopy(self.trace)
        trace["context_packet"]["evidence"][0]["full_statistical_check"]["support"] = "supported-by-prose"
        with self.assertRaises(ValueError):
            self.check_trace(self.refresh(trace))

    def test_manifest_and_rendered_string_are_both_verified(self):
        trace = copy.deepcopy(self.trace)
        trace["context_manifest"]["coefficient_row_count"] += 1
        with self.assertRaises(ValueError):
            self.check_trace(trace)
        trace = copy.deepcopy(self.trace)
        trace["brief"] += "\nUNLOGGED EXTRA EVIDENCE"
        with self.assertRaises(ValueError):
            self.check_trace(trace)

    def test_manifest_rendered_size_verified_and_legacy_absence_supported(self):
        trace = copy.deepcopy(self.trace)
        self.assertEqual(trace["context_manifest"]["rendered_chars"], len(trace["brief"]))
        self.check_trace(trace)
        trace["context_manifest"]["rendered_chars"] += 1
        with self.assertRaises(ValueError):
            self.check_trace(trace)
        del trace["context_manifest"]["rendered_chars"]
        self.check_trace(trace)
        trace["context_manifest"]["unknown_field"] = "unvalidated"
        with self.assertRaises(ValueError):
            self.check_trace(trace)

    def run_fixture(self, status="window-ended", censored=False, failure_type=None, stop_reason=None, cleanup=True):
        run = self.root / "run"
        (run / "source").mkdir(parents=True, exist_ok=True)
        project = Path(__file__).resolve().parents[1]
        sources = ("src/research_memory/evidence.py", "src/research_memory/tabular.py", "src/research_memory/two_role.py",
                   "src/research_memory/model_catalog.py",
                   "tools/two_role_discovery.py", "tools/private_inference.py", "tools/discovery_analysis.py")
        hashes = {}
        from audit_memory_iteration import digest
        for relative in sources:
            target = run / "source" / Path(relative).name
            shutil.copyfile(project / relative, target)
            hashes[relative] = digest(target)
        states = copy.deepcopy(self.states)
        if censored:
            states[-1].update(status="censored", decision_status="failed", failure_type="TimeoutError")
        report = {**self.task, "round": 1, "job_id": "900001", "arm": "memory", "memory_format": "tsv",
            "trajectory_root": str(self.trajectory), "source_sha256": hashes,
            "embedding_backend": "lexical", "retrieval_strategy": "ppr",
            "topology": "hypothesis-analysis-hypothesis",
            "server_binary_sha256": self.task["model_identity"]["server_binary_sha256"],
            "status": status, "attempts": states, "active_attempt": None, "model_calls": [{
                "capacity_scope": "actual_rendered_chat_prompt", "rendered_prompt_tokens": 100,
                "requested_generation_tokens": 768, "safety_margin_tokens": 32,
                "required_tokens": 900, "context_limit": 32768, "context_shift_enabled": False,
                "capacity_status": "fits", "completion_submitted": True, "truncated": False}],
            "runtime_cleanup_verified": cleanup, "failure_type": failure_type, "stop_reason": stop_reason}
        self.save(run / "config.json", report)
        self.save(run / "report.json", report)
        return self.audit_fixture()

    def audit_fixture(self):
        import audit_tsv_iteration as auditor
        from audit_memory_iteration import OWNERS
        names = ("REQUIRED_SOURCE_BASENAMES", "EXPECTED_ACTORS", "RETRIEVAL_ROLES", "EXPECTED_TOPOLOGY",
                 "REQUIRE_WORKFLOW", "REQUIRE_SAMPLE_RECIPE")
        saved = {name: getattr(auditor, name) for name in names}
        auditor.REQUIRED_SOURCE_BASENAMES = {"evidence.py", "tabular.py", "two_role.py", "model_catalog.py", "two_role_discovery.py",
            "private_inference.py", "discovery_analysis.py"}
        auditor.EXPECTED_ACTORS = dict(OWNERS)
        auditor.RETRIEVAL_ROLES = (("hypothesis", "hypothesis"), ("analysis", "analysis"))
        auditor.EXPECTED_TOPOLOGY = "hypothesis-analysis-hypothesis"
        auditor.REQUIRE_WORKFLOW = False
        auditor.REQUIRE_SAMPLE_RECIPE = False
        try:
            return audit_round(self.root / "run", self.trajectory, 1, "900001")
        finally:
            for name, value in saved.items():
                setattr(auditor, name, value)

    def test_normal_exit_passes_source_integrity(self):
        report = self.run_fixture()
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["source_integrity"], "passed")

    def test_verified_budget_censor_is_partial_not_source_corruption(self):
        report = self.run_fixture(status="failed", censored=True, failure_type="TimeoutError", stop_reason="budget deadline")
        self.assertEqual(report["status"], "partial", report)
        self.assertEqual(report["source_integrity"], "passed")

    def test_unexplained_transport_failure_is_not_excused(self):
        report = self.run_fixture(status="failed", failure_type="RuntimeIsolationError")
        self.assertEqual(report["status"], "failed", report)
        self.assertEqual(report["source_integrity"], "passed")
        self.assertEqual(report["runtime_health"], "failed")

    def test_explicit_startup_deadline_receipt_is_validated_and_censored(self):
        self.run_fixture()
        path = self.root / "run/report.json"
        report = json.loads(path.read_text())
        report.update(status="window-ended", stop_reason="allocation deadline",
            active_wall_seconds=1790, budget_seconds=1800, inference_deadline_reserve_seconds=45)
        report["attempts"][-1].update(status="censored", decision_status="failed",
            failure_type="AllocationDeadlineExceeded")
        report["model_calls"][0].update(failure_type="AllocationDeadlineExceeded",
            failure_boundary="runtime_startup", failure_cause="AllocationDeadlineExceeded",
            deadline_exhausted=True, parse_status="failed", completion_submitted=False)
        path.write_text(json.dumps(report))
        (self.root / "run/config.json").write_text(json.dumps(report))
        audited = self.audit_fixture()
        self.assertEqual(audited["source_integrity"], "passed", audited)
        self.assertEqual(audited["status"], "partial", audited)
        self.assertEqual(audited["censored_attempt_count"], 1)

    def test_empty_graph_is_valid_only_for_prepublication_failures(self):
        self.run_fixture()
        shared = self.trajectory / "shared-memory"
        shutil.rmtree(shared / "candidates")
        shutil.rmtree(shared / "snapshots")
        (shared / "CURRENT").unlink()
        for directory in self.trajectory.glob("attempt-*"):
            shutil.rmtree(directory)
        path = self.root / "run/report.json"
        report = json.loads(path.read_text())
        report.update(status="window-ended", stop_reason="allocation deadline",
            active_wall_seconds=1790, budget_seconds=1800, inference_deadline_reserve_seconds=45,
            attempts=[{"attempt": 1, "status": "failed", "analysis_status": "not_started",
                "decision_status": "failed", "failure_type": "AllocationDeadlineExceeded"}],
            model_calls=[{"failure_type": "AllocationDeadlineExceeded", "failure_boundary": "runtime_startup",
                "failure_cause": "AllocationDeadlineExceeded", "deadline_exhausted": True,
                "parse_status": "failed", "completion_submitted": False}], runtime_cleanup_verified=True)
        path.write_text(json.dumps(report))
        (self.root / "run/config.json").write_text(json.dumps(report))
        audited = self.audit_fixture()
        self.assertEqual(audited["source_integrity"], "passed", audited)
        self.assertEqual(audited["status"], "failed", audited)
        self.assertEqual(audited["failed_attempt_count"], 1)
        self.assertEqual(audited["actual_fit_count"], 0)

    def test_graph_candidates_without_snapshot_are_rejected(self):
        self.run_fixture()
        shared = self.trajectory / "shared-memory"
        shutil.rmtree(shared / "snapshots")
        (shared / "CURRENT").unlink()
        audited = self.audit_fixture()
        self.assertEqual(audited["source_integrity"], "unverified", audited)
        self.assertTrue(any("without a published snapshot" in flag["detail"] for flag in audited["flags"]))

    def test_unpublished_candidate_provenance_is_verified_when_all_attempts_failed(self):
        self.run_fixture()
        shared = self.trajectory / "shared-memory"
        shutil.rmtree(shared / "snapshots")
        (shared / "CURRENT").unlink()
        path = self.root / "run/report.json"
        report = json.loads(path.read_text())
        for state in report["attempts"]:
            state.update(status="failed", analysis_status="not_started", decision_status="failed",
                         failure_type="RuntimeIsolationError")
        report.update(status="failed", stop_reason="memory index build failed", model_calls=[], runtime_cleanup_verified=True)
        path.write_text(json.dumps(report))
        (self.root / "run/config.json").write_text(json.dumps(report))
        audited = self.audit_fixture()
        self.assertEqual(audited["source_integrity"], "passed", audited)
        self.assertEqual(audited["runtime_health"], "failed", audited)
        self.assertTrue(any(flag["code"] == "graph_snapshot_missing" for flag in audited["flags"]))

    def test_unrecorded_future_candidate_is_not_hidden_by_missing_snapshot(self):
        self.run_fixture()
        shared = self.trajectory / "shared-memory"
        source = next((shared / "candidates").glob("*/*.json"))
        future = json.loads(source.read_text())
        future["memory_id"] = "hypothesis-0004"
        target = shared / "candidates" / "hypothesis" / "hypothesis-0004.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(future))
        shutil.rmtree(shared / "snapshots")
        (shared / "CURRENT").unlink()
        audited = self.audit_fixture()
        self.assertEqual(audited["source_integrity"], "unverified", audited)
        self.assertTrue(any("outside the recorded attempt window" in flag["detail"] for flag in audited["flags"]))

    def test_cleanup_failure_preserves_source_proof_without_claiming_success(self):
        report = self.run_fixture(cleanup=False)
        self.assertEqual(report["source_integrity"], "passed", report)
        self.assertEqual(report["runtime_health"], "failed")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["actual_fit_count"], 3)

    def test_corrupt_source_is_unverified_not_excused_as_runtime_failure(self):
        self.run_fixture()
        source = self.root / "run/source/evidence.py"
        source.write_text(source.read_text() + "\n# changed after freeze\n")
        report = audit_round(self.root / "run", self.trajectory, 1, "900001")
        self.assertEqual(report["source_integrity"], "unverified")
        self.assertEqual(report["runtime_health"], "unverified")
        self.assertEqual(report["status"], "failed")

    def gpu_contract(self):
        runtime = {"device": "gpu", "runtime": "/reviewed/runtime/bin", "cuda_toolkit": "/reviewed/cuda"}
        config, task = {"runtime_contract": runtime}, {"runtime_contract": runtime}
        report = {"runtime_contract": runtime, "gpu": True, "device_report": {
            "requested_gpu_count": 1, "device_count": 1, "cuda_offloaded_layers": 40, "exclusive": False}}
        return config, report, task

    def test_gpu_contract_requires_one_actual_offloaded_device(self):
        config, report, task = self.gpu_contract()
        validate_runtime_contract(config, report, task)
        for key, value in (("device_count", 2), ("requested_gpu_count", 2), ("cuda_offloaded_layers", 0), ("exclusive", True)):
            changed = copy.deepcopy(report)
            changed["device_report"][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_runtime_contract(config, changed, task)

    def test_gpu_runtime_contract_change_rejected(self):
        config, report, task = self.gpu_contract()
        changed = copy.deepcopy(report)
        changed["runtime_contract"]["cuda_toolkit"] = "/different/cuda"
        with self.assertRaises(ValueError):
            validate_runtime_contract(config, changed, task)


if __name__ == "__main__":
    unittest.main()
