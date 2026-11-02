"""Resume only post-result scientific handbacks, preserving the original attempt."""
from pathlib import Path
import hashlib
import json

from discovery_pilot import QUESTION, write
from four_role_discovery import RoleHistory, finish_result
from private_inference import RuntimeIsolationError, ContextCapacityError
from research_memory.evidence import _source, statistical_assessment, statistical_summary, verify_record
from research_memory.orchestration import FourRoleOrchestrator


def verify_reused_result(result, root, data_identity):
    reuse = result.get("execution_reuse")
    if not reuse:
        return
    source_attempt = reuse.get("source_attempt", "")
    if not source_attempt.startswith("attempt-") or not source_attempt[8:].isdigit():
        raise ValueError("Invalid reused analysis source")
    key = json.dumps([data_identity, result.get("effective_action")], sort_keys=True, separators=(",", ":"))
    path = root / source_attempt / ("executed-analysis-" + hashlib.sha256(key.encode()).hexdigest()[:16] + ".json")
    content = path.read_bytes()
    original = {k: v for k, v in result.items() if k not in (
        "execution_reuse", "agent_r_diagnostic", "scientific_review", "scientific_review_source", "scientific_review_pending")}
    diagnostic = result.get("agent_r_diagnostic")
    if diagnostic is not None:
        # Only diagnostic-derived additions may differ from the immutable primary cache.
        original = json.loads(json.dumps(original))
        expected = {"agent_diagnostic_status": diagnostic.get("status"), **{
            "agent_diagnostic_" + key: value for key, value in diagnostic.get("metrics", {}).items()}}
        for index, row in enumerate(original.get("metrics", [])):
            additions = {key: value for key, value in row.items() if key.startswith("agent_diagnostic_")}
            if additions and (index != 0 or additions != expected):
                raise ValueError("Diagnostic metrics differ from their execution receipt")
            for key in additions:
                del row[key]
    if hashlib.sha256(content).hexdigest() != reuse.get("source_sha256") or json.loads(content) != original:
        raise ValueError("Reused analysis source changed or differs from receipt")


def verified_handback(pointer, original, refs, research_question, inquiry, seen=None):
    """Validate an interrupted resume and its immutable inheritance chain."""
    root = original.parent
    seen = set() if seen is None else seen
    if set(pointer) != {"format", "attempt", "manifest", "ledger", "assessment"} or pointer["format"] != "pending-handback-v1":
        raise ValueError("Invalid pending handback pointer")
    index = int(original.name.split("-")[1])
    if pointer["attempt"] != index:
        raise ValueError("Pending handback belongs to another attempt")
    folder = (root / pointer["manifest"]["path"]).parent
    if folder.parent != original or not folder.name.startswith("resume-") or str(folder) in seen:
        raise ValueError("Invalid or cyclic handback chain")
    seen.add(str(folder))
    values = {}
    for key, name in (("manifest", "resume-source.json"), ("ledger", "orchestrator.json"),
                      ("assessment", "empiricist-assessment.json")):
        value, actual = _source(folder / name, root)
        if actual != pointer[key]:
            raise ValueError("Pending handback source changed")
        values[key] = value
    manifest, ledger = values["manifest"], values["ledger"]
    if (manifest.get("format") != "post-result-resume-v1" or manifest.get("sources") != refs or
            manifest.get("original_attempt") != index or manifest.get("runner_reexecuted") is not False):
        raise ValueError("Handback no longer matches its original result sources")
    for name in ("hypothesis.json", "plan.json", "tool-result.json", "statistical-assessment.json", "current-result-facts.json"):
        if _source(folder / name, root)[0] != _source(original / name, root)[0]:
            raise ValueError("Restored scientific evidence differs from original")
    if (ledger.get("research_question") != research_question or ledger.get("current_inquiry") != inquiry or
            ledger.get("status") != "censored" or len(ledger.get("events", [])) != 5):
        raise ValueError("Pending handback has no censored Decision stage")
    verifier = FourRoleOrchestrator(folder, index, research_question=research_question, inquiry=inquiry)
    for event in ledger["events"]:
        if verifier.route(event["stage"], event["actor"], event["artifact"])["sources"] != event["sources"]:
            raise ValueError("Pending handback receipt changed")
    ancestor = manifest.get("prior_handback")
    if ancestor is not None and verified_handback(ancestor, original, refs, research_question, inquiry, seen) != values["assessment"]:
        raise ValueError("Inherited assessment changed across resumes")
    return values["assessment"]


def pending_result(trajectory, cursor):
    """Only a cursor-bound successful analysis without a scientific Decision qualifies."""
    if not cursor or not cursor.get("last_attempt"):
        return None
    root = Path(trajectory)
    reference = cursor["last_attempt"]
    record, actual = _source(root / reference["path"], root)
    if actual != reference:
        raise ValueError("Resume attempt changed after cursor publication")
    if (record.get("topology") != "orchestrator-hypothesis-empiricist-runner" or
            record.get("status") != "censored" or record.get("analysis_status") != "succeeded" or
            record.get("decision_status") == "succeeded"):
        return None
    folder = root / f"attempt-{record['attempt']:04d}"
    if folder / "attempt.json" != root / reference["path"]:
        raise ValueError("Pending analysis must be an original own-trajectory attempt")
    if not (folder / "tool-result.json").is_file():
        # A primary fit alone cannot resume a diagnostic/review stage as a finished result.
        return None
    for resume_attempt in folder.glob("resume-*/attempt.json"):
        if resume_attempt.is_symlink() or resume_attempt.parent.is_symlink():
            raise ValueError("Resume receipt cannot be a symbolic link")
        resumed, _ = _source(resume_attempt, root)
        if (resumed.get("attempt") == record["attempt"] and
                resumed.get("original_attempt_source") == reference and
                resumed.get("status") == "needs-attention" and
                resumed.get("failure_type") == "ContextCapacityError"):
            return None
    if cursor.get("last_decision"):
        decision, _ = _source(root / cursor["last_decision"]["path"], root)
        if record.get("result_id") in decision.get("evidence_ids", []):
            return None
    return folder


def resume_result(session, original, memory, model, *, data_identity, research_question=QUESTION):
    """Use verified own sources; never rerun the model or overwrite historical artifacts."""
    session, original = Path(session), Path(original)
    root = original.parent
    if session.parent != original or not session.name.startswith("resume-"):
        raise ValueError("Resume output must be a private child of the original attempt")
    originals, refs = {}, {}
    for name in ("attempt.json", "hypothesis.json", "plan.json", "tool-result.json",
                 "statistical-assessment.json", "current-result-facts.json", "orchestrator.json"):
        originals[name], refs[name] = _source(original / name, root)
    old = originals["attempt.json"]
    index = old["attempt"]
    if (original.name != f"attempt-{index:04d}" or old["status"] != "censored" or
            old["analysis_status"] != "succeeded" or old.get("decision_status") == "succeeded"):
        raise ValueError("Only a censored post-result attempt can resume")
    ledger = originals["orchestrator.json"]
    scientific_review = None
    if old.get("scientific_review") is not None:
        scientific_review, refs["scientific-review.json"] = _source(original / "scientific-review.json", root)
        if scientific_review != old["scientific_review"]:
            raise ValueError("Saved scientific review differs from the attempt receipt")
        review_sources = list(scientific_review.get("sources", []))
        for key in ("final_review_source", "resolution_source"):
            if scientific_review.get(key):
                review_sources.append(scientific_review[key])
        for reference in review_sources:
            if _source(root / reference["path"], root)[1] != reference:
                raise ValueError("Scientific review source changed before resume")
    if ledger["research_question"] != research_question:
        raise ValueError("Resume fixed ResearchQuestion changed")
    contract, plan, result = (originals[name] for name in
                              ("hypothesis.json", "plan.json", "tool-result.json"))
    if "scientific_review" in result and result["scientific_review"] != scientific_review:
        raise ValueError("Executed result and saved supervision receipt differ")
    check = statistical_assessment(plan, result)
    if result["returncode"] != 0 or check != originals["statistical-assessment.json"]:
        raise ValueError("Saved statistical result differs from source")
    expected_facts = {**statistical_summary(check), "estimate": check.get("estimate"),
        "estimates": result.get("estimates", []), "metrics": result.get("metrics", []),
        "analysis": result.get("effective_action"), "analysis_status": "succeeded",
        "evidence_id": f"result-{index:04d}", "result_facts_available": True,
        "agent_prose_included": False, "scope": check["scope"]}
    if result.get("execution_reuse"):
        verify_reused_result(result, root, data_identity)
        expected_facts["execution_reuse"] = result["execution_reuse"]
    # New handoffs retain diagnostic failures; legacy saved facts predate this field.
    if "agent_r_diagnostic" in originals["current-result-facts.json"]:
        expected_facts["agent_r_diagnostic"] = result.get("agent_r_diagnostic")
    if "scientific_review" in originals["current-result-facts.json"]:
        expected_facts["scientific_review"] = result.get("scientific_review")
    for key in ("scientific_review_source", "scientific_review_pending"):
        if key in originals["current-result-facts.json"]:
            expected_facts[key] = result.get(key)
    if result.get("scientific_review_source"):
        source = result["scientific_review_source"]
        value, actual = _source(root / source["path"], root)
        if actual != source or value != scientific_review:
            raise ValueError("Scientific review pointer differs from saved supervision")
    if originals["current-result-facts.json"] != expected_facts:
        raise ValueError("Saved result facts differ from complete source evidence")
    memory.ensure_research_question(research_question, root / "config.json")
    for identifier, kind, name in ((f"hypothesis-{index:04d}", "Hypothesis", "hypothesis.json"),
                                  (f"model-{index:04d}", "AnalysisModel", "plan.json"),
                                  (f"result-{index:04d}", "ResultAssessment", "tool-result.json")):
        node = memory._require(identifier, kind)
        if node["provenance"] != refs[name]:
            raise ValueError("Resume input differs from published graph source")
        if kind == "ResultAssessment":
            verify_record(node["evidence"], root)
            if node["data_identity"] != data_identity:
                raise ValueError("Resume result uses different data")
    if f"decision-{index:04d}" in memory.nodes:
        raise ValueError("Scientific Decision already exists; do not publish twice")
    verifier = FourRoleOrchestrator(original, index, research_question=research_question,
                                    inquiry=ledger["current_inquiry"])
    events = ledger["events"]
    if len(events) not in (4, 5):
        raise ValueError("Saved handoff trace did not reach the result")
    for event in events:
        restored = verifier.route(event["stage"], event["actor"], event["artifact"])
        if restored["sources"] != event["sources"]:
            raise ValueError("Saved handoff source receipt changed")
    saved_assessment = None
    if len(events) == 5:
        saved_assessment, refs["empiricist-assessment.json"] = _source(original / "empiricist-assessment.json", root)
    pending_path = original / "PENDING-HANDBACK.json"
    prior_handback = json.loads(pending_path.read_text()) if pending_path.exists() else None
    if prior_handback is not None:
        saved_assessment = verified_handback(prior_handback, original, refs, research_question, ledger["current_inquiry"])
    # Copies keep the legacy source paths immutable; manifest ties every input back to it.
    write(session / "resume-source.json", {"format": "post-result-resume-v1", "sources": refs,
        "original_attempt": index, "runner_reexecuted": False, "prior_handback": prior_handback})
    for name in ("hypothesis.json", "plan.json", "tool-result.json",
                 "statistical-assessment.json", "current-result-facts.json"):
        write(session / name, originals[name])
    if scientific_review is not None:
        write(session / "scientific-review.json", scientific_review)
    workflow = FourRoleOrchestrator(session, index, research_question=research_question,
                                    inquiry=ledger["current_inquiry"])
    for event in events[:3]:
        replay = workflow.route(event["stage"], event["actor"], event["artifact"])
        replay["execution"] = "restored_receipt_only"
    histories = RoleHistory(root)
    record = {"attempt": index, "topology": old["topology"], "condition": "C",
        "orchestration_mode": "deterministic", "shared_memory_enabled": True,
        "role_history_enabled": True, "status": "started", "analysis_status": "succeeded",
        "decision_status": "not_started", "hypothesis_id": f"hypothesis-{index:04d}",
        "model_id": f"model-{index:04d}", "result_id": f"result-{index:04d}",
        "statistical_assessment": check, "resumed_result": refs["tool-result.json"],
        "runner_reexecuted": False, "original_attempt_source": refs["attempt.json"]}
    if scientific_review is not None:
        record["scientific_review"] = scientific_review
    if old.get("scientific_supervision"):
        record["scientific_supervision"] = old["scientific_supervision"]

    def route(stage, actor, artifact):
        event = workflow.route(stage, actor, artifact)
        if stage == "empiricist_assessment" and saved_assessment is not None:
            event["execution"] = "restored_receipt_only"
        workflow.write(session / "orchestrator.json", status="in_progress")
        histories.add("orchestrator", stage, {"routing_mode": "deterministic", "next_stage": stage,
            "research_question": research_question, "current_inquiry": ledger["current_inquiry"],
            "sources": event["sources"], "resumed_attempt": index})

    decision = {}
    try:
        query = research_question + "\nCurrent hypothesis: " + contract["hypothesis"]
        trace, supplied = memory.retrieve_tsv(query, role="hypothesis")
        design_trace, _ = memory.retrieve_tsv(query, role="analysis")
        for name, value in (("hypothesis-retrieval.json", trace), ("empiricist-retrieval.json", design_trace)):
            write(session / name, value)
        record["retrieved_ids"] = trace["selected_ids"]
        decision = finish_result(session, index, memory, model, route, histories, record,
            contract=contract, plan=plan, facts=expected_facts, evidence=trace["brief"],
            design_evidence=design_trace["brief"], supplied=supplied,
            research_question=research_question, inquiry=ledger["current_inquiry"],
            saved_assessment=saved_assessment, scientific_review=scientific_review)
        workflow.write(session / "orchestrator.json", status=record.get("workflow_status") or
                       ("handoffs_completed" if record.get("scientific_review") else "completed"))
    except Exception as error:
        censored = isinstance(error, RuntimeIsolationError) and getattr(error, "deadline_exhausted", False)
        capacity = isinstance(error, ContextCapacityError)
        record.update(status="censored" if censored else "needs-attention" if capacity else "failed",
                      failure_type=type(error).__name__, diagnostic=str(error)[:300],
                      failure_domain="inference_deadline_censor" if censored else
                          "context_capacity" if capacity else "resume_handback")
        if record["decision_status"] == "running":
            record["decision_status"] = "censored" if censored else "failed"
        workflow.write(session / "orchestrator.json", status=record["status"])
        if censored and record["decision_status"] == "censored":
            pointer = {"format": "pending-handback-v1", "attempt": index,
                "manifest": _source(session / "resume-source.json", root)[1],
                "ledger": _source(session / "orchestrator.json", root)[1],
                "assessment": _source(session / "empiricist-assessment.json", root)[1]}
            # Only this derived pointer is replaced; source attempts and resume receipts stay immutable.
            pending = original / "PENDING-HANDBACK.pending.json"
            write(pending, pointer)
            pending.replace(pending_path)
    return record, plan, decision
