"""Two scripted fixture cycles through the actual graph/research controller.

This checks integration, not model capability, statistics or research performance.
The fixture responses and estimates come from the shipped regression tests.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from test_four_role import hypothesis, plan, result, answer_envelope_adapter
import four_role_discovery as flow
from research_memory.two_role import TwoRoleMemory
from research_memory.viewer import render_dashboard
from two_role_discovery import QUESTION


def main():
    if not os.environ.get("JOB_ID"):
        raise SystemExit("Submit through the Job Scheduler; this demo computes graph retrieval.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-statistics", action="store_true",
                        help="Use the actual sandboxed R Runner on generated synthetic rows; model replies stay scripted")
    args = parser.parse_args()
    os.umask(0o077)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    (root / "config.json").write_text(json.dumps({"question": QUESTION, "synthetic_only": True}))
    flow.RESEARCH_ROLE_LAYOUT = "unified"
    flow.ORCHESTRATOR_ASSESSMENT = False
    flow.ADAPTIVE_SHARED_CONTEXT = True
    memory = TwoRoleMemory(root, embedding_backend="lexical", retrieval_strategy="ppr")
    records, calls = [], []
    analysis_runner = lambda folder, action: result(action)
    if args.real_statistics:
        from discovery_pilot import run_analysis
        rng = random.Random(112)
        data = root / "synthetic.csv"
        with data.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["tagged", "typicality"])
            for _ in range(400):
                writer.writerow([rng.randrange(2), rng.gauss(0, 1)])
        analysis_runner = lambda folder, action: run_analysis(folder, action, data=data)
    for index in (1, 2):
        session = root / f"attempt-{index:04d}"
        session.mkdir(mode=0o700)
        def model(stage, prompt, schema, tokens, seed):
            calls.append({"attempt": index, "stage": stage, "backend": "scripted_fixture"})
            if stage == "hypothesis":
                response = hypothesis()
                if index == 2:
                    response['adaptation'] = {'decision': 'continue', 'basis': 'Scripted repeated fixture; no discovery claim.', 'evidence_ids': ['result-0001']}
                return response
            if stage == "empiricist_plan":
                return {**plan(), "diagnostic_r_code": None}
            if stage == "empiricist_assessment":
                return {"execution_assessment": "Actual R on synthetic rows; interpretation is scripted." if args.real_statistics else "Synthetic fixture result; no real statistical analysis.",
                        "diagnostic_action": "adequate_for_interpretation"}
            if stage == "hypothesis_decision":
                return {"action": "maintain", "basis_keywords": "Scripted decision; inspect machine-derived statistical assessment separately.",
                        "next_inquiry": "Inspect the stored synthetic association.",
                        "evidence_ids": [f"result-{index:04d}"]}
            raise AssertionError(stage)
        record, _, _ = flow.execute_four_role_attempt(session, index, memory,
            answer_envelope_adapter(model), analysis_runner,
            data_identity="synthetic-fixture-not-real-data", condition="C")
        records.append(record)
        assert record['status'] == 'completed', record
        if not args.real_statistics:
            assert record['statistical_assessment']['support'] == 'not_supported'
    trace, _ = memory.retrieve_tsv("Inspect the stored synthetic typicality association.", role="hypothesis")
    (root / "trace.json").write_text(json.dumps(trace, indent=2))
    (root / "memory.tsv").write_text(trace['brief'])
    html = render_dashboard(trace)
    label = 'SYNTHETIC ONLY: scripted model responses; ' + ('actual R estimates on generated data.' if args.real_statistics else 'fixed statistical receipts.')
    html = html.replace('<body>', '<body><p><strong>' + label + ' No research-performance claim.</strong></p>', 1)
    (root / "dashboard.html").write_text(html)
    report = {"passed": True, "synthetic_only": True, "model_backend": "scripted_fixture",
              "statistics_backend": "actual_R_synthetic_data" if args.real_statistics else "fixed_test_receipts", "controller": "execute_four_role_attempt",
              "graph_backend": "lexical_plus_ppr", "cycles": len(records),
              "graph_node_count": len(memory.nodes), "calls": calls, "records": records}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": True, "synthetic_only": True, "cycles": 2, "output": str(root)}))


if __name__ == "__main__":
    main()
