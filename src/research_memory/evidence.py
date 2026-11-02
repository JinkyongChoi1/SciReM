"""Source-grounded result records and bounded, whole-record memory packets."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _source(path, root):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence source outside this trajectory")
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()}


def _verified_artifact(reference, root):
    """Read an immutable trajectory artifact without following symlink aliases."""
    root = Path(root).resolve()
    path = Path(reference["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact source outside this trajectory")
    path = root / path
    if any(part.is_symlink() for part in (path, *path.parents) if part != root and part.is_relative_to(root)):
        raise ValueError("Artifact source must not be a symlink")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Artifact source outside this trajectory")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Artifact differs from immutable source hash")
    return raw


def supervision_fields(tool):
    return {key: deepcopy(tool[key]) for key in
            ("scientific_review", "scientific_review_source", "scientific_review_pending") if key in tool}


def result_assessment(evidence_id, source, root, assessment=None, historical_validation=None, statistical_check=None, plan_source=None):
    tool, provenance = _source(source, root)
    if not isinstance(tool.get("returncode"), int) or not isinstance(tool.get("effective_action"), dict):
        raise ValueError("Evidence requires an executed tool result and effective specification")
    if not isinstance(tool.get("estimates", []), list) or not isinstance(tool.get("metrics", []), list):
        raise ValueError("Malformed tool result")
    return {"node_type": "ResultAssessment", "evidence_id": evidence_id,
        "result": {"source": provenance, "returncode": tool["returncode"],
                   "analysis": deepcopy(tool["effective_action"]),
                   "estimates": deepcopy(tool.get("estimates", [])), "metrics": deepcopy(tool.get("metrics", []))},
        "assessment": {"status": "unverified_agent_interpretation", "content": deepcopy(assessment or {}),
                       "historical_claim_validation": deepcopy(historical_validation or []),
                       "statistical_check": deepcopy(statistical_check),
                       "plan_source": _source(plan_source, root)[1] if plan_source else None,
                       **supervision_fields(tool)}}


def verify_record(record, root):
    source = record["result"]["source"]
    tool, actual = _source(Path(root) / source["path"], root)
    expected = {"source": actual, "returncode": tool["returncode"], "analysis": tool["effective_action"],
                "estimates": tool.get("estimates", []), "metrics": tool.get("metrics", [])}
    if record["result"] != expected:
        raise ValueError("Evidence differs from immutable source or source hash")
    diagnostic = tool.get("agent_r_diagnostic") or {}
    if diagnostic.get("code_sha256"):
        if "code" in diagnostic and hashlib.sha256(diagnostic["code"].encode()).hexdigest() != diagnostic["code_sha256"]:
            raise ValueError("Diagnostic code in context differs from executed source")
        reference = deepcopy(diagnostic.get("source"))
        if reference is None:
            reference = {"path": str(Path(source["path"]).parent / "agent-r-diagnostic" / "diagnostic.R"),
                         "sha256": diagnostic["code_sha256"]}
        elif reference.get("path_scope") != "trajectory":
            reference["path"] = str(Path(source["path"]).parent / reference["path"])
        if reference["sha256"] != diagnostic["code_sha256"]:
            raise ValueError("Agent-written diagnostic code differs from source receipt")
        _verified_artifact(reference, root)
        if diagnostic.get("output_sha256"):
            _verified_artifact({"path": str(Path(reference["path"]).parent / "diagnostics.csv"),
                                "sha256": diagnostic["output_sha256"]}, root)
    supervision = supervision_fields(tool)
    if supervision_fields(record.get("assessment", {})) != supervision:
        raise ValueError("Scientific supervision differs from immutable result source")
    if "scientific_review_pending" in supervision and not isinstance(supervision["scientific_review_pending"], bool):
        raise ValueError("Scientific review pending status must be boolean")
    if supervision.get("scientific_review") is not None:
        review = supervision["scientific_review"]
        review_source = supervision.get("scientific_review_source")
        if not isinstance(review, dict) or not isinstance(review_source, dict):
            raise ValueError("Scientific review requires its immutable source")
        if json.loads(_verified_artifact(review_source, root)) != review:
            raise ValueError("Scientific review differs from its source artifact")
        references = list(review.get("sources", []))
        references.extend(review[key] for key in ("final_review_source", "resolution_source", "blocking_diagnostic_source")
                          if review.get(key) is not None)
        for reference in references:
            _verified_artifact(reference, root)
    check = record.get("assessment", {}).get("statistical_check")
    if check is not None:
        plan_source = record["assessment"]["plan_source"]
        plan, provenance = _source(Path(root) / plan_source["path"], root)
        if provenance != plan_source or check != statistical_assessment(plan, tool):
            raise ValueError("Statistical assessment differs from source plan and executed result")
    return record


def _term(value):
    value = re.sub(r"\s+", "", value)
    while value.startswith("I(I(") and value.endswith("))"):
        value = value[2:-1]
    return ":".join(sorted(value.split(":"))) if ":" in value else value


def test_fact(record, term):
    result = record["result"]
    if result["returncode"] != 0:
        return {"status": "execution_failed", "estimate": None}
    key = _term(term)
    rows = [row for row in result["estimates"] if _term(row["term"]) == key]
    included = any(_term(value) == key for value in result["analysis"].get("terms", []))
    if not included and not rows:
        return {"status": "not_in_model", "estimate": None}
    if len(rows) == 1:
        try:
            values = [float(rows[0][name]) for name in ("coefficient", "se", "p_value")]
            if all(math.isfinite(x) for x in values) and values[1] >= 0 and 0 <= values[2] <= 1:
                return {"status": "estimated", "estimate": rows[0]}
        except (KeyError, TypeError, ValueError):
            pass
    return {"status": "specified_result_unavailable", "estimate": None}


class DesignAlignmentError(ValueError):
    """A declared primary hypothesis is not represented by the proposed model."""


def validate_focal_test(plan):
    focal = plan.get("focal_test", {})
    kind, term = focal.get("effect_type"), focal.get("term", "")
    direction = focal.get("expected_direction")
    identifier = r"[a-z_][a-z0-9_]*"
    interaction = rf"{identifier}:{identifier}"
    valid_term = ((kind == "interaction" and re.fullmatch(interaction, term)) or
                  (kind == "curvature" and re.fullmatch(rf"I\({identifier}\^2\)", term)) or
                  (kind == "main_effect" and re.fullmatch(identifier, term)))
    if not valid_term or direction not in ("positive", "negative", "nonzero"):
        raise DesignAlignmentError("Declare one well-formed primary coefficient test and expected direction")
    if _term(term) not in {_term(t) for t in plan["analysis"]["terms"]}:
        raise DesignAlignmentError("Declared primary test is missing from the proposed model: " + term)
    # shortcut: explicit terminology only; 'moderate levels' is not moderation.
    wording = " ".join(str(plan.get(k, "")) for k in ("hypothesis", "expected_pattern"))
    executed_interactions = [t for t in plan["analysis"]["terms"]
                             if re.fullmatch(interaction, _term(t))]
    if (re.search(r"\b(?:moderation|moderators?|moderat(?:es|ed|ing)|interactions?|interact(?:s|ed|ing)?)\b", wording, re.I)
            and kind != "interaction" and not executed_interactions):
        raise DesignAlignmentError("A moderation/interaction hypothesis requires a declared interaction test")
    return deepcopy(focal)


def statistical_assessment(plan, tool):
    """A fixed directional coefficient test, never a scientific-validity certificate."""
    check = {"support": "inconclusive", "reason": "", "alpha": 0.10,
             "primary_test": deepcopy(plan.get("focal_test")),
             "scope": "exploratory primary coefficient test only; not causal or hypothesis validity",
             "scientific_validity": "unverified"}
    try:
        focal = validate_focal_test(plan)
    except (DesignAlignmentError, KeyError, TypeError) as error:
        check["reason"] = "design_invalid: " + str(error)
        return check
    actual = {"result": {"returncode": tool.get("returncode"),
                        "analysis": tool.get("effective_action", {}), "estimates": tool.get("estimates", [])}}
    if _term(focal["term"]) not in {_term(t) for t in actual["result"]["analysis"].get("terms", [])}:
        check["reason"] = "declared_test_missing_from_effective_model"
        return check
    fact = test_fact(actual, focal["term"])
    check["reason"] = fact["status"]
    if fact["estimate"] is None:
        return check
    row = fact["estimate"]
    check["estimate"] = deepcopy(row)
    coefficient, p_value = Decimal(str(row["coefficient"])), Decimal(str(row["p_value"]))
    expected = focal["expected_direction"]
    aligned = ((expected == "positive" and coefficient > 0) or
               (expected == "negative" and coefficient < 0) or (expected == "nonzero" and coefficient != 0))
    if p_value > Decimal("0.10"):
        check.update(support="not_supported", reason="primary_test_not_significant; not proof of a null effect")
    elif not aligned:
        check.update(support="not_supported", reason="significant_effect_opposes_declared_direction")
    else:
        check.update(support="supported", reason="primary_coefficient_sign_and_p_value_match_declared_test")
    return check


def validate_claims(claims, supplied_records, root):
    """Validate explicit factual claims only; never certify free prose or hypothesis truth."""
    checked = []
    for claim in claims:
        outcome = {"claim": deepcopy(claim), "status": "unverifiable", "expected": None}
        record = supplied_records.get(claim.get("evidence_id"))
        if record is not None:
            try:
                verify_record(record, root)
                fact = test_fact(record, claim.get("term", ""))
                field, value = claim.get("field"), claim.get("value")
                if field == "attempt_order":
                    expected = attempt_order(record)
                    outcome.update(expected=expected, status="verified" if expected is not None and str(value) == str(expected) else "contradicted" if expected is not None else "unverifiable")
                elif field == "test_status":
                    outcome.update(expected=fact["status"], status="verified" if value == fact["status"] else "contradicted")
                elif field in ("coefficient", "se", "p_value") and fact["estimate"] is not None:
                    expected = str(fact["estimate"][field])
                    observed, target = Decimal(str(value)), Decimal(expected)
                    if not observed.is_finite() or not target.is_finite():
                        raise ValueError("Nonfinite claim")
                    outcome.update(expected=expected, status="verified" if observed == target else "contradicted")
                outcome["source"] = record["result"]["source"]
            except (OSError, ValueError, KeyError, TypeError, InvalidOperation):
                outcome.update(status="unverifiable", expected=None)
        checked.append(outcome)
    return checked


def attempt_order(record):
    """Order comes from verified source location, not agent prose or result naming."""
    match = re.fullmatch(r"attempt-(\d+)/tool-result\.json", record["result"]["source"]["path"])
    return int(match[1]) if match else None


def analysis_key(action):
    normalized = deepcopy(action)
    normalized["terms"] = sorted({_term(t) for t in action.get("terms", [])})
    return hashlib.sha256(encode(dict(sorted(normalized.items()))).encode()).hexdigest()


def coefficient_summaries(rows):
    """Deterministic labels for every reported term, including non-focal controls."""
    summaries = []
    for row in rows:
        item = {"term": row.get("term"), "p_value": row.get("p_value"),
                "significance": "?", "direction": "?", "alpha": 0.10}
        try:
            p, beta = Decimal(str(row.get("p_value"))), Decimal(str(row.get("coefficient")))
            if p.is_finite() and beta.is_finite() and 0 <= p <= 1:
                item.update(significance="O" if p <= Decimal("0.10") else "X",
                            direction="+" if beta > 0 else "-" if beta < 0 else "0")
        except InvalidOperation:
            pass
        summaries.append(item)
    return summaries


def statistical_summary(check):
    check = check or {}
    row = check.get("estimate")
    significance, direction = "?", "?"
    if row:
        significance = "O" if Decimal(str(row["p_value"])) <= Decimal("0.10") else "X"
        coefficient = Decimal(str(row["coefficient"]))
        direction = "+" if coefficient > 0 else "-" if coefficient < 0 else "0"
    return {"significance": significance, "direction": direction,
            "support": {"supported": "O", "not_supported": "X"}.get(check.get("support"), "?"),
            "alpha": 0.10, "primary_test": check.get("primary_test")}


def validate_interpretation(content, check, limit=600):
    """Narrow numeric readback and output-quality checks, not semantic certification."""
    issues = []
    expected = check.get("estimate")
    readback = content.get("primary_readback")
    if expected is not None:
        if not isinstance(readback, dict) or any(str(readback.get(k)) != str(expected[k]) for k in ("term", "coefficient", "se", "p_value")):
            issues.append("primary_readback_mismatch")
    elif readback is not None:
        issues.append("primary_readback_without_estimate")
    for field in ("interpretation", "limitations", "next_step"):
        text = content.get(field, "").strip()
        if not text or text[-1:] not in (".", "!", "?") or len(text) >= limit:
            issues.append(field + ":incomplete_or_at_character_boundary")
    return {"status": "flagged" if issues else "narrow_checks_passed", "issues": issues,
            "scientific_validity": "unverified", "free_text_direction_and_chronology": "unverified"}


def ground_trace(trace, root, query, budget=3200):
    """Hydrate ranked IDs from own-arm sources; never substring-cut scientific JSON."""
    if budget < 128:
        raise ValueError("Context budget too small")
    trace = deepcopy(trace)
    nodes = {node["memory_id"]: node for node in trace["nodes"]}
    ranked = list(trace["selected_ids"])
    packing_order = list(ranked)
    verified = []
    for key in ranked:
        evidence_id = nodes[key].get("result_id", key)
        record = nodes.get(evidence_id, {}).get("evidence")
        if record is None:
            continue
        try:
            verify_record(record, root)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        verified.append((key, record))
    if verified:
        first_key, first = verified[0]
        alternatives = [(key, record) for key, record in verified
                        if analysis_key(record["result"]["analysis"]) != analysis_key(first["result"]["analysis"])]
        if alternatives:
            latest_key, _ = max(alternatives, key=lambda pair: attempt_order(pair[1]) or -1)
            packing_order = [first_key, latest_key] + [key for key in ranked if key not in (first_key, latest_key)]
    packet = {"format": "facts-v7", "evidence": [], "omitted_evidence_ids": [],
              "policy": "O/X=statistical only; ?=unknown. Omitted tests unknown. Prose excluded."}
    supplied, seen = {}, set()
    for key in packing_order:
        node = nodes[key]
        evidence_id = node.get("result_id", key)
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        target = nodes.get(evidence_id, {})
        record = target.get("evidence")
        if record is None:
            packet["omitted_evidence_ids"].append(evidence_id)
            continue
        try:
            verify_record(record, root)
        except (OSError, ValueError, KeyError, TypeError):
            packet["omitted_evidence_ids"].append(evidence_id)
            continue
        view = {"evidence_id": evidence_id, "source": record["result"]["source"],
                "attempt_order": attempt_order(record), "analysis_key": analysis_key(record["result"]["analysis"]),
                "returncode": record["result"]["returncode"], "analysis": record["result"]["analysis"],
                "metrics": record["result"]["metrics"], "estimates": [], "omitted_estimate_terms": [],
                "statistical_check": statistical_summary(record["assessment"].get("statistical_check")),
                "assessment": {"status": "agent_prose_excluded", "content": {},
                    "omitted_fields": list(record["assessment"].get("content", {})),
                    **supervision_fields(record["assessment"])}}
        rows = record["result"]["estimates"]
        query_lower = query.lower()
        def priority(row):
            term = row["term"].lower()
            primary = (record["assessment"].get("statistical_check") or {}).get("primary_test") or {}
            focal = _term(term) == _term(primary.get("term", ""))
            exact = term in query_lower
            interaction = ":" in term and any(word in query_lower for word in ("interaction", "moderation", "moderator", ":"))
            overlap = sum(part in query_lower for part in term.split(":"))
            return (-focal, -exact, -interaction, -overlap)
        rows = sorted(rows, key=priority)
        view["omitted_estimate_terms"] = [row["term"] for row in rows]
        packet["evidence"].append(view)
        # Reserve one primary/query-relevant row per result before spending on extras.
        if rows:
            view["estimates"].append(rows[0])
            view["omitted_estimate_terms"].remove(rows[0]["term"])
        if len(encode(packet)) > budget or (rows and not view["estimates"]):
            packet["evidence"].pop()
            packet["omitted_evidence_ids"].append(evidence_id)
        else:
            supplied[evidence_id] = record
    for view in packet["evidence"]:
        rows = supplied[view["evidence_id"]]["result"]["estimates"]
        for row in rows:
            if row in view["estimates"]:
                continue
            view["estimates"].append(row)
            view["omitted_estimate_terms"].remove(row["term"])
            if len(encode(packet)) > budget:
                view["estimates"].pop()
                view["omitted_estimate_terms"].append(row["term"])
    # Omitted-ID metadata itself is budgeted; records are removed whole, never cut.
    while len(encode(packet)) > budget and packet["evidence"]:
        removed = packet["evidence"].pop()
        supplied.pop(removed["evidence_id"])
        packet["omitted_evidence_ids"].append(removed["evidence_id"])
    if len(encode(packet)) > budget:
        packet = {"evidence": [], "omitted_count": len(seen), "policy": "No complete evidence record fit; past tests are unknown."}
        supplied = {}
    trace.update(ranked_selected_ids=ranked, packing_order=packing_order,
                 packing_policy="top_relevance_then_latest_distinct_model_then_remaining; numeric_rows_before_prose",
                 selected_ids=list(supplied), context_packet=packet,
                 brief=encode(packet), context_char_budget=budget, context_exclusions=packet.get("omitted_evidence_ids", []))
    for node in trace["nodes"]:
        node["included"] = node["memory_id"] in supplied
    return trace, supplied


def loop_candidates(index, question, plan, interpretation, evidence, session, config_source, previous=None):
    """Five primary types, immutable backward provenance edges across inquiry versions."""
    suffix = f"{index:04d}"
    h, model, result, decision = [prefix + "-" + suffix for prefix in ("hypothesis", "model", "result", "decision")]
    source = str(Path(session) / "tool-result.json")
    def node(key, kind, text, origin, edges=(), **extra):
        return {"memory_id": key, "agent_id": "attempt-" + suffix, "track_id": "PILOT", "memory_type": kind,
                "status": "proposed", "text": text or "No statement supplied", "source": str(origin),
                "aliases": ([evidence["assessment"]["statistical_check"]["primary_test"]["term"]]
                            if kind == "ResultAssessment" and evidence["assessment"].get("statistical_check") else []),
                "edges": [{"target": target, "relation": relation, "source": str(origin)} for target, relation in edges], **extra}
    records = []
    if evidence["evidence_id"] != result:
        raise ValueError("Result node identity mismatch")
    if previous is None:
        records.append(node("research-question", "ResearchQuestion", question, config_source))
    hypothesis_edges = [("research-question", "addresses")]
    if previous is not None:
        hypothesis_edges += [(previous["decision"], "follows_decision"), (previous["hypothesis"], "revision_of")]
    records += [node(h, "Hypothesis", plan["hypothesis"], Path(session)/"plan.json", hypothesis_edges, result_id=result),
        node(model, "AnalysisModel", encode({"effective_action": evidence["result"]["analysis"],
             "model_specification": plan.get("model_specification"),
             "design_rationale": plan.get("design_rationale"),
             "focal_test": plan.get("focal_test"), "alignment": "declared_primary_term_present" if plan.get("focal_test") else "unreviewed"}),
             source, [(h, "intended_test_for")], result_id=result),
        node(result, "ResultAssessment", "Executed results for " + plan["hypothesis"], source,
             [(model, "produced_by")], result_id=result, evidence=evidence),
        node(decision, "Decision", encode({"declared_decision": plan.get("adaptation", {}).get("decision"),
             "status": "self_reported_not_scientific_validation",
             "result": statistical_summary(evidence["assessment"].get("statistical_check"))}), Path(session)/"plan.json",
             [(result, "based_on")], result_id=result)]
    return records, {"hypothesis": h, "decision": decision, "result": result}
