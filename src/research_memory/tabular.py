"""Full scientific-context TSV rendering with provenance kept in the trace."""
import csv
import io

from .evidence import encode, statistical_summary, attempt_order


def _cell(value):
    if value is None:
        return r"\N"
    if isinstance(value, str):
        return "\\" + value if value.startswith("\\") else value
    return encode(value)


def render_table(name, columns, rows):
    output = io.StringIO(newline="")
    output.write("\n# " + name + "\n")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(column)) for column in columns])
    return output.getvalue()


def render_tsv(packet):
    """Keep exact scalar strings and every scientific row; never substring-cut."""
    output = io.StringIO(newline="")
    scope = ("Selected relevant scientific nodes and their required source endpoints; all rows of selected results. "
             "Historical lineage edge targets may be outside this packet; absence is not evidence of no previous test. "
             if packet.get("selection_policy") == "ranked_seeds_with_scientific_source_closure" else
             "All published own-trajectory scientific nodes and distinct executed results, not raw data. ")
    output.write("TSV_CONTEXT_V1: \\N=null; leading backslash strings escaped. " + scope +
        "O/X/? are statistical facts, not scientific validity. Table order is relevance, not recency; "
        "source_attempt is the original source order, and observed_in lists equivalent observations, not new independent evidence. "
        "Supervisor tables are source-linked agent assessments, not independently verified scientific facts.\n")
    if packet.get("requested_ids"):
        output.write("Requested source IDs: " + ", ".join(packet["requested_ids"]) + ".\n")
    if packet.get("missing_ids"):
        output.write("Unavailable source IDs in this snapshot: " + ", ".join(packet["missing_ids"]) + ".\n")
    def table(name, columns, rows):
        output.write(render_table(name, columns, rows))

    questions, hypotheses, models, decisions, edges = [], [], [], [], []
    for node in packet["graph_nodes"]:
        payload, kind = node["payload"], node["memory_type"]
        identifier = node["memory_id"]
        if kind == "ResearchQuestion":
            questions.append({"id": identifier, "question": payload["question"]})
        elif kind == "Hypothesis":
            focal = payload["focal_test"]
            hypotheses.append({"id": identifier, "claim": payload["hypothesis"],
                "expected": payload.get("expected_pattern"), "rival": payload.get("rival_explanation"),
                "required_terms": payload.get("required_terms"),
                "focal_type": focal["effect_type"], "term": focal["term"], "direction": focal["expected_direction"],
                "adaptation": payload.get("adaptation"), "historical_claims": payload.get("historical_claims")})
        elif kind == "AnalysisModel":
            hypothesis_ids = [e["target"] for e in node["edges"] if e["relation"] == "intended_test_for"]
            specification = payload.get("model_specification") or {}
            models.append({"id": identifier, "hypothesis_id": hypothesis_ids,
                **payload.get("analysis", payload.get("effective_action", {})),
                "design_rationale": specification.get("design_rationale", payload.get("design_rationale")),
                "required_term_resolutions": payload.get("required_term_resolutions"),
                **specification,
                "sample_id": node.get("sample_identity")})
        elif kind == "Decision":
            decisions.append({"id": identifier, **payload})
        edges.extend({"origin": identifier, **edge}
                     for edge in node["edges"])
    table("RESEARCH_QUESTIONS", ["id", "question"], questions)
    table("HYPOTHESES", ["id", "claim", "expected", "rival", "focal_type", "term", "direction", "required_terms", "adaptation", "historical_claims"], hypotheses)
    table("MODELS", ["id", "hypothesis_id", "design_type", "outcome_type", "estimator", "dependent_variable",
        "focal_predictor", "focal_terms", "control_variables", "treatment_variable", "instrument_variables", "moderator_interaction",
        "panel_unit", "time_variable", "treatment_timing", "running_variable", "cutoff", "fixed_effects", "strata",
        "standard_error_method", "cluster_variables", "estimand", "identification_assumptions", "diagnostics",
        "execution_support", "family", "outcome", "terms", "cluster", "filter",
        "sample_id", "design_rationale", "required_term_resolutions"], models)
    result_rows, supervision_rows, supervision_issues = [], [], []
    for view in packet["evidence"]:
        check = view["full_statistical_check"] or {}
        focal = check.get("primary_test") or {}
        extra = {key: value for key, value in view["tool"].items() if key not in (
            "estimates", "metrics", "effective_action", "scientific_review", "scientific_review_source", "scientific_review_pending")}
        review = view.get("scientific_review") or {}
        supervision_rows.append({"evidence_id": view["evidence_id"],
            "status": review.get("status", "pending" if view.get("scientific_review_pending") else "legacy_not_reviewed"),
            "requested_test_status": review.get("requested_test_status"),
            "scientific_validity": review.get("scientific_validity"),
            "claim_consistency": review.get("claim_consistency"),
            "required_assumptions_status": review.get("required_assumptions_status"),
            "next_action": review.get("next_action"),
            "pending_work": review.get("pending_work", review.get("blocking_reason")),
            "explanation": review.get("explanation")})
        supervision_issues.extend({"evidence_id": view["evidence_id"], **issue} for issue in review.get("issues", []))
        result_rows.append({**extra, "evidence_id": view["evidence_id"], "observed_in": view["observed_in"],
            "source_attempt": attempt_order({"result": {"source": view["source"]}}),
            "sample_identity": view["sample_identity"], "sample_recipe": view.get("sample_recipe"), **statistical_summary(check),
            "support_enum": check.get("support"), "primary_term": focal.get("term"),
            "expected_direction": focal.get("expected_direction"), "reason": check.get("reason"),
            "scope": check.get("scope"), "scientific_validity": check.get("scientific_validity")})
    result_columns = ["evidence_id", "source_attempt", "observed_in", "sample_recipe", "sample_identity", "returncode", "significance", "direction", "support",
                      "alpha", "primary_test", "support_enum", "primary_term", "expected_direction", "reason", "scope", "scientific_validity"]
    result_columns += [key for row in result_rows for key in row if key not in result_columns]
    table("RESULTS", list(dict.fromkeys(result_columns)), result_rows)
    table("SUPERVISOR_ASSESSMENTS", ["evidence_id", "status", "requested_test_status", "scientific_validity",
        "claim_consistency", "required_assumptions_status", "next_action", "pending_work", "explanation"], supervision_rows)
    table("SUPERVISOR_ISSUES", ["evidence_id", "source", "finding"], supervision_issues)
    for name, field, preferred in (("COEFFICIENTS", "estimates", ["term", "coefficient", "se", "p_value"]),
                                   ("METRICS", "metrics", ["n", "SE_type", "sample_membership_md5"])):
        rows = [{"evidence_id": view["evidence_id"], "row": index, **row}
                for view in packet["evidence"] for index, row in enumerate(view[field], 1)]
        fields = list(dict.fromkeys(key for row in rows for key in row))
        columns = ["evidence_id", "row"] + [key for key in preferred if key in fields]
        columns += [key for key in fields if key not in columns]
        table(name, columns, rows)
    table("DECISIONS", ["id", "action", "basis_keywords", "next_inquiry", "evidence_ids"], decisions)
    table("EDGES", ["origin", "relation", "target", "assertion_status", "scope", "basis",
                    "scientific_validity", "source", "provenance"], edges)
    return output.getvalue()
