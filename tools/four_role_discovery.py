"""Scheduler-only four-role research loop; the Runner is deterministic code."""
from copy import deepcopy
from collections import Counter
import json
import hashlib
import math
import re
from pathlib import Path

from discovery_pilot import QUESTION, encode, write
from research_memory.evidence import statistical_assessment, statistical_summary
from research_memory.orchestration import FourRoleOrchestrator
from private_inference import RuntimeIsolationError, ContextCapacityError
from agent_r_diagnostic import DIAGNOSTIC_ENVIRONMENT
from two_role_discovery import (ANALYSIS_SCHEMA, DECISION_SCHEMA, HYPOTHESIS_SCHEMA, NO_EVIDENCE,
    DataInsufficientDesign, PILOT_ANALYSIS_GUIDANCE,
                                SHORT, UnsupportedModelDesign, build_plan, decision_schema, role_prompt)


EMPIRICIST_ASSESSMENT_SCHEMA = {"type": "object", "properties": {
    "execution_assessment": SHORT,
    "diagnostic_action": {"enum": ["adequate_for_interpretation", "additional_test", "design_problem"]}},
    "required": ["execution_assessment", "diagnostic_action"], "additionalProperties": False}

CODING_ANALYSIS_SCHEMA = deepcopy(ANALYSIS_SCHEMA)
CODING_ANALYSIS_SCHEMA["properties"]["diagnostic_r_code"] = {
    "anyOf": [{"type": "null"}, {"type": "string", "maxLength": 16000}]}
CODING_ANALYSIS_SCHEMA["required"].append("diagnostic_r_code")

DIAGNOSTIC_REPAIR_SCHEMA = {"type": "object", "properties": {
    "action": {"enum": ["repair", "unavailable"]},
    "diagnostic_r_code": {"anyOf": [{"type": "null"}, {"type": "string", "minLength": 1}]},
    "reason": {"type": "string", "minLength": 1}},
    "required": ["action", "diagnostic_r_code", "reason"], "additionalProperties": False}


class PlanRepairError(ValueError):
    """The Empiricist plan remained invalid after its one bounded repair call."""


def execute_diagnostic_work(session, code, diagnostic_runner, model, memory, histories, record,
                            *, contract, inquiry, shared_memory_enabled, artifact_root=None):
    """Repair the requested diagnostic with source-linked failures, not another primary fit."""
    from agent_r_diagnostic import DiagnosticOutputError
    session = Path(session)
    artifact_root = Path(artifact_root) if artifact_root is not None else session.parent
    seen = set()
    seen_failures = set()
    ordinal = 0
    while True:
        ordinal += 1
        digest = hashlib.sha256(code.encode()).hexdigest()
        seen.add(digest)
        work = session if ordinal == 1 else session / ("diagnostic-repair-%04d" % (ordinal - 1))
        if work != session:
            work.mkdir(mode=0o700)
        if diagnostic_runner is None:
            diagnostic = {"status": "blocked", "reason": "No reviewed dataset-bound diagnostic runner",
                          "error": {"category": "runner_unavailable", "message": "No diagnostic runner was supplied."}}
        else:
            try:
                diagnostic = diagnostic_runner(work, code)
            except (ValueError, OSError) as error:
                diagnostic = deepcopy(getattr(error, "receipt", None)) or {
                    "status": "failed", "reason": str(error) if isinstance(error, DiagnosticOutputError) else type(error).__name__,
                    "error": {"category": "diagnostic_runner_error", "message":
                              str(error) if isinstance(error, DiagnosticOutputError) else type(error).__name__}}
        diagnostic = {**diagnostic, "code": code, "code_sha256": digest,
                      "scope": "Agent-written diagnostic execution; scientific adequacy is not independently validated."}
        if diagnostic.get("status") == "completed" and not any(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in diagnostic.get("metrics", {}).values()):
            diagnostic.update(status="unavailable", execution_status="completed", reason="No numeric diagnostic result",
                error={"category": "no_numeric_result", "stage": "output_validation",
                       "message": "R completed but returned no finite numeric diagnostic values.",
                       "repair_guidance": "Determine why the requested result is unavailable; repair the analysis if possible or report the actual blocker. Preserve NA for unavailable results; never replace NA with a fabricated zero."})
        for field in ("source", "stderr_source", "runner_source"):
            if diagnostic.get(field):
                reference = diagnostic[field]
                artifact = (work / reference["path"]).resolve()
                if not artifact.is_relative_to(work.resolve()):
                    raise ValueError("Diagnostic source escaped its private execution directory")
                if hashlib.sha256(artifact.read_bytes()).hexdigest() != reference["sha256"]:
                    raise ValueError("Diagnostic source differs from its execution receipt")
                diagnostic[field] = {**reference, "path": str(artifact.relative_to(artifact_root.resolve())),
                                     "path_scope": "trajectory"}
        receipt_path = work / "diagnostic-receipt.json"
        write(receipt_path, diagnostic)
        source = {"path": str(receipt_path.relative_to(artifact_root)),
                  "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
        diagnostic["receipt_source"] = source
        record.setdefault("diagnostic_executions", []).append({"ordinal": ordinal,
            "status": diagnostic.get("status"), "code_sha256": digest, "source": source,
            "error": diagnostic.get("error")})
        histories.add("empiricist", "diagnostic_execution", diagnostic)
        if diagnostic.get("status") in {"completed", "blocked"}:
            return diagnostic
        failure = diagnostic.get("error", {})
        signature = (failure.get("category"), failure.get("symbol"), failure.get("message"))
        if failure.get("repairable_from_receipt") is False or signature in seen_failures:
            return {**diagnostic, "status": "blocked", "scientific_test_completed": False,
                    "repair_status": "error_inspection_required" if failure.get("repairable_from_receipt") is False else "same_failure_after_repair",
                    "blocking_reason": "Concrete error inspection is required: the error detail is unavailable or the proposed repair left the same execution failure. This is an engineering blocker, not data infeasibility."}
        seen_failures.add(signature)
        repair_prompt = (
            "Repair the failed diagnostic for the existing research request. Do not rerun the primary fit as a substitute.\n"
            "Current inquiry: " + inquiry + "\nCurrent hypothesis/test contract:\n" + encode(contract) +
            "\nExact failed code and source-linked, sanitized failure receipt:\n" + encode(diagnostic) +
            "\nReturn action=repair with changed standalone R code and a reason explaining how it fixes this failure, "
            "or action=unavailable with diagnostic_r_code=null and the specific missing capability/data/design requirement. "
            "Preserve the requested comparison, sample, estimand and uncertainty target. Do not silently substitute "
            "a coefficient p-value for predictive comparison or remove a failing check to claim success. "
            "Use established packages for statistical methods. "
            "The same sandbox provides /data/styles.csv read-only and /work; write aggregate metric,value CSV "
            "to /work/diagnostics.csv under the existing diagnostic output contract. "
            "The raw failure remains recorded; it is not evidence of no effect. " + DIAGNOSTIC_ENVIRONMENT)
        repair, rendered, _, _ = call_role_with_memory(session, "empiricist_diagnostic_repair", "empiricist",
            "analysis", repair_prompt, DIAGNOSTIC_REPAIR_SCHEMA, 768, 20400000 + ordinal,
            model, memory, histories, record, shared_memory_enabled)
        write(work / "diagnostic-repair-response.json", repair)
        histories.add("empiricist", "diagnostic_repair", repair, rendered)
        replacement = repair.get("diagnostic_r_code")
        if repair.get("action") == "unavailable" and replacement is None:
            return {**diagnostic, "status": "blocked", "repair_status": "unavailable",
                    "blocking_reason": repair.get("reason"), "scientific_test_completed": False}
        if repair.get("action") != "repair" or not isinstance(replacement, str) or not replacement.strip():
            raise ValueError("Diagnostic repair must supply changed code or explicitly report unavailable")
        if hashlib.sha256(replacement.encode()).hexdigest() in seen:
            return {**diagnostic, "status": "blocked", "repair_status": "unchanged_failed_code",
                    "blocking_reason": "The proposed repair repeats already failed code; no identical execution was launched.",
                    "scientific_test_completed": False}
        code = replacement


def diagnostic_metrics(primary_metrics, diagnostic):
    rows = [{key: deepcopy(value) for key, value in row.items() if not key.startswith("agent_diagnostic_")}
            for row in primary_metrics]
    if rows and diagnostic is not None:
        rows[0]["agent_diagnostic_status"] = diagnostic["status"]
        rows[0].update({"agent_diagnostic_" + key: value for key, value in diagnostic.get("metrics", {}).items()})
    return rows


def supervise_result(session, index, memory, model, histories, record, *, contract, plan,
                     facts, evidence, design_evidence, supplied, research_question, inquiry,
                     diagnostic_runner, shared_memory_enabled):
    """Review actual research work, execute justified repairs, then review the new evidence."""
    from orchestrator_review import REVIEW_SCHEMA, review_prompt, adjudicate_review
    from decision_review import assessment_only_repair
    session = Path(session)
    diagnostic = facts.get("agent_r_diagnostic")
    primary_metrics = deepcopy(facts.get("metrics", []))
    ordinal = 0
    seen_code = {item["code_sha256"] for item in record.get("diagnostic_executions", [])}
    assessment_feedback = None
    rejected_assessments = set()
    while True:
        ordinal += 1
        work = session / ("supervision-%04d" % ordinal)
        work.mkdir(mode=0o700)
        if diagnostic is not None:
            facts["agent_r_diagnostic"] = diagnostic
        facts["metrics"] = diagnostic_metrics(primary_metrics, diagnostic)
        actual = {"inquiry": inquiry, "contract": contract, "plan": plan,
                  "primary_facts": facts, "diagnostic_code": (diagnostic or {}).get("code"),
                  "diagnostic_receipt": diagnostic}
        write(work / "evidence.json", actual)
        assessment_prompt = (
            role_prompt("Empiricist agent: execution assessment", contract, design_evidence, facts, memory_format="tsv") +
            "\nFixed Research Question: " + research_question + "\nCurrent inquiry: " + inquiry +
            "\nActual plan, full diagnostic code and execution receipts:\n" + encode(actual) +
            "\nAssess the actual requested test, assumptions, and interpretation against this evidence. "
            "A cached primary fit is not a new diagnostic. Code exit zero or a significant coefficient does not "
            "establish a requested comparison. Mark unavailable or invalid diagnostics as unresolved. "
            "Do not change authoritative statistical values or claim unexecuted work was performed.")
        if assessment_feedback is not None:
            assessment_prompt += (
                "\nRevise only the assessment claims using this source-grounded review. "
                "The actual analysis/code/results have not changed; do not propose or claim new execution.\n" +
                encode(assessment_feedback))
        assessment_stage = "empiricist_assessment_repair" if assessment_feedback is not None else "empiricist_assessment"
        empirical, rendered, trace, extra = call_role_with_memory(work, assessment_stage, "empiricist",
            "analysis", assessment_prompt, EMPIRICIST_ASSESSMENT_SCHEMA, 240, 20330000 + index * 10 + ordinal,
            model, memory, histories, record, shared_memory_enabled)
        merge_supplied(supplied, extra)
        evidence = append_retrieval_evidence(evidence, trace)
        design_evidence = append_retrieval_evidence(design_evidence, trace)
        write(work / "empiricist-assessment.json", empirical)
        histories.add("empiricist", "assessment", empirical, rendered)
        prompt = "Fixed Research Question: " + research_question + "\n" + review_prompt(
            inquiry=inquiry, contract=contract, plan=plan, primary_facts=facts,
            diagnostic_code=(diagnostic or {}).get("code"), diagnostic_receipt=diagnostic,
            empirical_assessment=empirical)
        proposed, rendered, trace, extra = call_role_with_memory(work, "orchestrator_scientific_review",
            "orchestrator", "analysis", prompt, REVIEW_SCHEMA, 768, 20500000 + index * 10 + ordinal,
            model, memory, histories, record, shared_memory_enabled)
        merge_supplied(supplied, extra)
        evidence = append_retrieval_evidence(evidence, trace)
        review = adjudicate_review(proposed, facts, diagnostic, diagnostic_code=(diagnostic or {}).get("code"))
        review["sources"] = [{"path": str(path.relative_to(session.parent)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (work / "evidence.json", work / "empiricist-assessment.json")]
        write(work / "orchestrator-review.json", review)
        review_source = {"path": str((work / "orchestrator-review.json").relative_to(session.parent)),
            "sha256": hashlib.sha256((work / "orchestrator-review.json").read_bytes()).hexdigest()}
        record.setdefault("scientific_reviews", []).append({"status": review["status"], "source": review_source})
        histories.add("orchestrator", "scientific_review", review, rendered)
        histories.add("empiricist", "supervisor_feedback", review)
        if facts.get("analysis_status") != "succeeded":
            review = {**review, "status": "blocked", "next_action": "blocked",
                      "blocking_domain": "primary_execution_pending",
                      "blocking_reason": "Repair the primary analysis plan/execution before another diagnostic; no valid primary fit is available.",
                      "pending_work": "Empiricist must repair the primary analysis for Runner, not change only diagnostic code."}
            break
        if review["status"] == "accepted" or review.get("next_action") != "repair":
            break
        if assessment_only_repair(review):
            signature = hashlib.sha256(json.dumps(empirical, sort_keys=True).encode()).hexdigest()
            if signature in rejected_assessments:
                review = {**review, "status": "blocked", "next_action": "blocked",
                          "blocking_reason": "No substantive progress: the same contradicted assessment was repeated."}
                break
            rejected_assessments.add(signature)
            assessment_feedback = review
            continue
        assessment_feedback = None
        if diagnostic and diagnostic.get("status") == "blocked":
            review = {**review, "status": "blocked", "next_action": "blocked",
                      "blocking_reason": diagnostic.get("blocking_reason", diagnostic.get("reason")),
                      "blocking_diagnostic_source": diagnostic.get("receipt_source")}
            break
        repair_prompt = (
            "The Orchestrator reviewed your actual work and found the requested scientific task unresolved. "
            "Implement the missing diagnostic or repair its scientific design; do not repeat the cached primary fit.\n"
            "Current inquiry: " + inquiry + "\nHypothesis/test contract:\n" + encode(contract) +
            "\nActual executed work:\n" + encode(actual) + "\nSource-grounded review:\n" + encode(review) +
            "\nReturn action=repair with new or changed standalone R code, or action=unavailable with null code "
            "and the specific missing data/design/capability. Preserve the requested estimand and uncertainty target. "
            "Do not fabricate zeros for NA, drop checks, or replace a comparison by an association. "
            "The Runner executes the returned code in the existing sandbox (/data/styles.csv read-only, /work writable). "
            "Use established statistical packages and write aggregate metric,value CSV to /work/diagnostics.csv. "
            "Your planning session having no shell is not evidence the Runner host is unavailable. " + DIAGNOSTIC_ENVIRONMENT)
        repair, rendered, trace, extra = call_role_with_memory(work, "empiricist_supervised_repair", "empiricist",
            "analysis", repair_prompt, DIAGNOSTIC_REPAIR_SCHEMA, 768, 20600000 + index * 10 + ordinal,
            model, memory, histories, record, shared_memory_enabled)
        merge_supplied(supplied, extra)
        design_evidence = append_retrieval_evidence(design_evidence, trace)
        write(work / "repair-proposal.json", repair)
        histories.add("empiricist", "supervised_repair", repair, rendered)
        code = repair.get("diagnostic_r_code")
        if repair.get("action") == "unavailable" and code is None:
            review = {**review, "status": "blocked", "next_action": "blocked", "blocking_reason": repair["reason"],
                      "resolution_source": {"path": str((work / "repair-proposal.json").relative_to(session.parent)),
                          "sha256": hashlib.sha256((work / "repair-proposal.json").read_bytes()).hexdigest()}}
            break
        if repair.get("action") != "repair" or not isinstance(code, str) or not code.strip():
            raise ValueError("Scientific repair must supply code or a specific unavailable reason")
        digest = hashlib.sha256(code.encode()).hexdigest()
        if digest in seen_code:
            review = {**review, "status": "blocked", "next_action": "blocked",
                      "blocking_reason": "No substantive progress: proposed code repeats an already executed diagnostic.",
                      "resolution_source": {"path": str((work / "repair-proposal.json").relative_to(session.parent)),
                          "sha256": hashlib.sha256((work / "repair-proposal.json").read_bytes()).hexdigest()}}
            break
        seen_code.add(digest)
        execution = work / "execution"
        execution.mkdir(mode=0o700)
        diagnostic = execute_diagnostic_work(execution, code, diagnostic_runner, model, memory, histories,
            record, contract=contract, inquiry=inquiry, shared_memory_enabled=shared_memory_enabled,
            artifact_root=session.parent)
        seen_code.update(item["code_sha256"] for item in record.get("diagnostic_executions", []))
    review = {**review, "final_review_source": review_source,
              "final_review_source_scope": "Last independent reviewer verdict before controller resolution."}
    write(session / "scientific-review.json", review)
    record["scientific_review"] = review
    record["scientific_review_source"] = {"path": str((session / "scientific-review.json").relative_to(session.parent)),
        "sha256": hashlib.sha256((session / "scientific-review.json").read_bytes()).hexdigest()}
    return diagnostic, empirical, review, evidence, design_evidence


def review_final_decision(session, index, memory, model, histories, record, *, decision,
                          contract, facts, scientific_review, supplied, research_question,
                          inquiry, shared_memory_enabled):
    """Only an evidence-consistent Decision may become the published scientific handback."""
    from decision_review import DECISION_REVIEW_SCHEMA, decision_review_prompt, adjudicate_decision_review
    session = Path(session)
    root = histories.root.parent
    evidence_path = session / "decision-evidence.json"
    write(evidence_path, {"research_question": research_question, "inquiry": inquiry,
        "contract": contract, "result_facts": facts, "scientific_review": scientific_review})
    seen, ordinal = set(), 0
    while True:
        ordinal += 1
        if set(decision) != set(DECISION_SCHEMA["properties"]) or facts["evidence_id"] not in decision["evidence_ids"]:
            raise ValueError("Decision lacks the current result receipt")
        proposal = session / ("decision-proposal-%04d.json" % ordinal)
        write(proposal, decision)
        histories.add("hypothesis", "decision_proposal", decision)
        signature = hashlib.sha256(json.dumps(decision, sort_keys=True).encode()).hexdigest()
        if signature in seen:
            verdict = {**verdict, "status": "blocked", "next_action": "blocked",
                       "blocking_reason": "No substantive progress: the same rejected Decision was repeated.",
                       "repeated_proposal_source": {"path": str(proposal.relative_to(root)),
                           "sha256": hashlib.sha256(proposal.read_bytes()).hexdigest()}}
            break
        seen.add(signature)
        prompt = decision_review_prompt(research_question, inquiry, contract, facts, scientific_review, decision)
        response, rendered, _, extra = call_role_with_memory(session, "orchestrator_decision_review",
            "orchestrator", "hypothesis", prompt, DECISION_REVIEW_SCHEMA, 512,
            20700000 + index * 10 + ordinal, model, memory, histories, record, shared_memory_enabled)
        merge_supplied(supplied, extra)
        verdict = adjudicate_decision_review(response)
        verdict["sources"] = [{"path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (evidence_path, proposal)]
        review_path = session / ("decision-review-%04d.json" % ordinal)
        write(review_path, verdict)
        record.setdefault("decision_reviews", []).append({"status": verdict["status"], "source": {
            "path": str(review_path.relative_to(root)), "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest()}})
        histories.add("orchestrator", "decision_review", verdict, rendered)
        histories.add("hypothesis", "decision_review", verdict)
        if verdict["status"] != "revision_required":
            break
        prompt = (
            "Revise only your scientific Decision to address the Orchestrator's concrete evidence-consistency issues. "
            "Do not change the actual analysis, code, numerical results, or existing scientific review. "
            "Do not claim a pending test was executed; distinguish a proposed next action from completed work. "
            "Maintaining a hypothesis is allowed when consistent with the evidence.\n" +
            decision_review_prompt(research_question, inquiry, contract, facts, scientific_review, decision) +
            "\nDecision review requiring revision:\n" + encode(verdict) +
            "\nReturn the revised Decision using the Decision schema, not the review schema. Cite the current result.")
        decision, rendered, _, extra = call_role_with_memory(session, "hypothesis_decision_repair",
            "hypothesis", "hypothesis", prompt, decision_schema(facts["evidence_id"], set(supplied)), 320,
            20800000 + index * 10 + ordinal, model, memory, histories, record, shared_memory_enabled,
            final_schema_builder=lambda extra: decision_schema(facts["evidence_id"], set(supplied) | set(extra)))
        merge_supplied(supplied, extra)
        histories.add("hypothesis", "decision_repair", decision, rendered)
    write(session / "decision-review.json", verdict)
    record["decision_review"] = verdict
    record["decision_review_source"] = {"path": str((session / "decision-review.json").relative_to(root)),
        "sha256": hashlib.sha256((session / "decision-review.json").read_bytes()).hexdigest()}
    return (decision if verdict["status"] == "accepted" else None), verdict


def focal_execution_instruction(contract):
    focal = contract["focal_test"]["term"]
    kind = contract["focal_test"]["effect_type"]
    if kind == "interaction":
        left, right = focal.split(":", 1)
        return (f"Primary focal test is interaction `{focal}`. Include that exact interaction and both main effects "
                f"`{left}` and `{right}` in analysis.terms and model_specification.focal_terms; set "
                f"moderator_interaction exactly to `{focal}`.")
    if kind == "curvature":
        base = focal[2:-3] if focal.startswith("I(") and focal.endswith("^2)") else ""
        return (f"Primary focal test is curvature term `{focal}`. Include the exact term in analysis.terms and "
                f"model_specification.focal_terms" +
                (f", with base predictor `{base}` in both fields." if base else "."))
    return (f"Primary focal term is `{focal}`. Include it in analysis.terms and "
            "model_specification.focal_terms; additional basis terms belong there only if they define this focal claim.")


def hypothesis_schema(attempt, prior_result_ids=()):
    """Do not let a fresh episode invent a prior attempt or evidence receipt."""
    schema = deepcopy(HYPOTHESIS_SCHEMA)
    schema["properties"]["required_terms"] = {"type": "array", "items": {"type": "string", "maxLength": 100},
                                                "minItems": 1, "maxItems": 12, "uniqueItems": True}
    schema["required"].append("required_terms")
    if attempt != 1:
        schema["properties"]["adaptation"]["properties"]["decision"] = {
            "enum": ["continue", "refine", "revise", "abandon"]}
        ids = sorted(set(prior_result_ids))
        schema["properties"]["adaptation"]["properties"]["evidence_ids"] = {
            "type": "array", "items": {"enum": ids} if ids else {"type": "string"},
            "minItems": 1 if ids else 0, "maxItems": 4 if ids else 0}
        return schema
    schema["properties"]["adaptation"] = {
        "type": "object",
        "properties": {
            "decision": {"const": "initial"},
            "basis": {"const": "Initial attempt; no prior evidence."},
            "evidence_ids": {"type": "array", "items": {"type": "string", "maxLength": 64}, "maxItems": 0}},
        "required": ["decision", "basis", "evidence_ids"], "additionalProperties": False}
    schema["properties"]["historical_claims"] = {"type": "array", "items": HYPOTHESIS_SCHEMA[
        "properties"]["historical_claims"]["items"], "maxItems": 0}
    return schema


def sample_provenance(data_identity, result):
    """Human-readable effective-sample recipe; row hash remains optional audit metadata."""
    metrics = next((row for row in result.get("metrics", []) if row.get("n") is not None), {})
    action = result.get("effective_action") or {}
    data_sha256 = data_identity.get("sha256") if isinstance(data_identity, dict) else data_identity
    return {"base_data": "styles.csv", "data_sha256": data_sha256,
            "merge": "none", "filter": action.get("filter"),
            "model_exclusions": "complete cases on outcome and model variables",
            "n_obs": metrics.get("n"), "rows_input": metrics.get("rows_input"),
            "rows_after_filter": metrics.get("rows_after_filter"),
            "complete_case_dropped": metrics.get("complete_case_dropped")}


COMPACT_SHARED_HISTORY = True
ADAPTIVE_SHARED_CONTEXT = False
RESEARCH_ROLE_LAYOUT = "split"
ORCHESTRATOR_ASSESSMENT = True


class RoleHistory:
    """Append-only role-local exchanges; memory retrieval remains a separate channel."""

    def __init__(self, root):
        self.root = Path(root) / "role-history"
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.layout = RESEARCH_ROLE_LAYOUT
        if self.layout not in {"split", "unified"}:
            raise ValueError("Unknown research role layout")

    def _key(self, role):
        if role not in {"hypothesis", "empiricist", "orchestrator", "researcher"}:
            raise ValueError("Unknown private history role")
        if self.layout == "unified" and role in {"hypothesis", "empiricist"}:
            return "researcher"
        if role == "researcher" and self.layout != "unified":
            raise ValueError("Unified history is unavailable in split layout")
        return role

    def _identity_prompt(self, role, prompt):
        if self._key(role) != "researcher":
            return prompt
        prompt = re.sub(r"(?m)^Role: (?:Hypothesis|Empiricist) agent", "Role: Researcher", prompt, count=1)
        return ("You are one Researcher responsible for hypothesis, analysis, interpretation and revision. "
                "Stage labels denote steps of your own work, not different researchers.\n" + prompt)

    def _items(self, role):
        role = self._key(role)
        path = self.root / (role + ".jsonl")
        return [json.loads(line) for line in path.read_text().split("\n") if line] if path.exists() else []

    def recall(self, role, query):
        role = self._key(role)
        match = re.fullmatch(r"private-turn:([1-9][0-9]*):([0-9]+)", query)
        if not match:
            raise ValueError("Private recall requires private-turn:TURN:OFFSET")
        number, offset = map(int, match.groups())
        items = self._items(role)
        if number > len(items):
            raise ValueError("Private turn does not exist")
        text = json.dumps(items[number - 1], ensure_ascii=False)
        if offset >= len(text):
            raise ValueError("Private recall offset exceeds record length")
        end = min(len(text), offset + 12000)
        return {"role": role, "turn": number, "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "offset": offset, "total_chars": len(text), "next_offset": end if end < len(text) else None,
                "content": text[offset:end], "scope": "Exact own-role archive page; agent prose is not verified evidence."}

    def _adaptive_prompt(self, role, prompt, items):
        checkpoint_path = self.root / (role + "-checkpoint.json")
        header = ("C ACTIVE CHECKPOINT: complete raw exchanges remain in your private archive, not in this prompt. "
                  "Missing history is unknown, not proof something was never tested. Latest outputs below are agent "
                  "statements, not verified statistical facts. Use shared-memory retrieval for source-verified results. "
                  "To read any own-role original input/output, request retrieve_memory with query "
                  "private-turn:TURN:OFFSET (start OFFSET=0); pages report next_offset.\n"
                  f"Private role={role}; archived turn numbers=1..{len(items)}.\n")
        tail = "\nCurrent request (complete):\n" + prompt
        blocks, selected, seen = [], [], set()
        for number in range(len(items), 0, -1):
            item = items[number - 1]
            if item["stage"] in seen:
                continue
            seen.add(item["stage"])
            value = json.dumps(item["value"], ensure_ascii=False)
            block = f"ARCHIVED TURN {number}; stage={item['stage']}; latest output:\n{value}\n"
            blocks.append(block)
            selected.append(number)
        rendered = lossless_prompt(header + "".join(reversed(blocks)) + tail)
        checkpoint = {"policy": "bounded_c_checkpoint_with_private_recall", "role": role,
              "archived_turns": len(items), "included_output_turns": sorted(selected),
              "omitted_turns": len(items) - len(selected), "active_chars": len(rendered),
              "selection": "latest complete output per stage; no character cutoff",
              "current_request_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        # Replace the derived pointer only; the raw exchange archive stays append-only.
        pending = checkpoint_path.with_suffix(".pending")
        pending.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))
        pending.replace(checkpoint_path)
        return rendered

    def prompt(self, role, prompt, *, shared_memory=False):
        prompt = self._identity_prompt(role, prompt)
        role = self._key(role)
        path = self.root / (role + ".jsonl")
        if shared_memory and ADAPTIVE_SHARED_CONTEXT:
            return self._adaptive_prompt(role, prompt, self._items(role))
        turns = []
        if path.exists():
            # JSON strings may contain Unicode separators; JSONL uses only LF as its delimiter.
            items = self._items(role)
            if shared_memory and COMPACT_SHARED_HISTORY and not ADAPTIVE_SHARED_CONTEXT:
                # Full exchanges remain archived; the current TSV and one latest output per stage are the checkpoint.
                latest, seen = [], set()
                for item in reversed(items):
                    if item["stage"] not in seen:
                        seen.add(item["stage"])
                        latest.append(item)
                for item in reversed(latest):
                    turns.append("LATEST TURN %s\nAGENT OUTPUT:\n%s" % (
                        item["stage"], json.dumps(item["value"], ensure_ascii=False)))
            else:
                for item in items:
                    turns.append("TURN %s\nINPUT:\n%s\nOUTPUT:\n%s" % (
                        item["stage"], item.get("prompt", ""), json.dumps(item["value"], ensure_ascii=False)))
        history = "\n\n".join(turns) or "No prior role-local conversation."
        checkpoint = ("Compact C checkpoint: latest role output per stage only; agent prose is not a verified fact. "
                      "Full raw exchanges remain archived; request source-verified memory for missing evidence.\n"
                      if shared_memory and COMPACT_SHARED_HISTORY and not ADAPTIVE_SHARED_CONTEXT else "")
        full = (checkpoint + "Private %s conversation history follows. It is not shared with other roles.\n%s\n"
                "Current request:\n%s" % (role, history, prompt))
        return lossless_prompt(full)

    def add(self, role, stage, value, prompt=""):
        prompt = self._identity_prompt(role, prompt) if prompt else prompt
        origin_role = role
        role = self._key(role)
        path = self.root / (role + ".jsonl")
        with path.open("a") as stream:
            item = {"stage": stage, "prompt": prompt, "value": value}
            if self.layout == "unified":
                item["origin_stage_role"] = origin_role
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    def prior_result_ids(self, role="hypothesis"):
        role = self._key(role)
        path = self.root / (role + ".jsonl")
        if not path.exists():
            return set()
        return {evidence_id for line in path.read_text().split("\n") if line
                if (item := json.loads(line)).get("stage") == "decision"
                for evidence_id in item.get("value", {}).get("evidence_ids", [])
                if re.fullmatch(r"result-\d{4}", evidence_id)}


def lossless_prompt(full):
    # Keep repeated lines lossless, including current memory packets, without truncating evidence.
    lines = full.split("\n")
    counts = Counter(lines)
    repeated = {}
    # Literal archive text must never be mistaken for an alias we generated.
    prefix = "SAME"
    while "[" + prefix + " " in full:
        prefix += "_"
    for line in dict.fromkeys(lines):
        if counts[line] < 2:
            continue
        label = "L" + prefix[4:] + str(len(repeated) + 1)
        reference = "[" + prefix + " " + label + "]"
        if (counts[line] - 1) * len(line) > counts[line] * len(reference) + len(label) + 2:
            repeated[line] = label
    if not repeated:
        return full
    dictionary = "\n".join(label + "\t" + line for line, label in repeated.items())
    body = "\n".join("[" + prefix + " " + repeated[line] + "]" if line in repeated else line for line in lines)
    encoded = ("Lossless role-local text dictionary. [" + prefix + " Lx] means the exact full text of Lx below. "
               "Repeated evidence is not independent evidence. Historical requests are history; follow Current request.\n"
               + dictionary + "\nEND TEXT DICTIONARY\n" + body)
    return encoded if len(encoded) < len(full) else full

def finish_result(session, index, memory, model, route, histories, record, *, contract, plan,
                  facts, evidence, design_evidence, supplied, research_question, inquiry,
                  shared_memory_enabled=True, saved_assessment=None, scientific_review=None):
    """Complete scientific handbacks; also used after a verified saved Runner result."""
    result_id = facts["evidence_id"]
    route("empiricist_assessment", "empiricist-agent", {"result_id": result_id,
        "analysis_status": record["analysis_status"], "result_facts": facts})
    assessment_prompt = (
        role_prompt("Empiricist agent: execution assessment", contract, design_evidence, facts, memory_format="tsv") +
        "\nFixed Research Question: " + research_question +
        "\nAssess execution/design limits only. Do not rewrite p-values, support, or hypothesis. "
        "Compare the hypothesis and rival explanation with the exact executed terms/sample/inference. "
        "State what this analysis tested and what remains unresolved; use design_problem when a broader claim "
        "was not tested. A standalone association does not establish relative strength, a mechanism, or causality. "
        "A quadratic coefficient alone does not establish a turning point within observed support. "
        "Do not invent diagnostics or a completed comparison that the result receipt does not contain.")
    empirical = saved_assessment
    if empirical is None:
        empirical, assessment_prompt, assessment_trace, assessment_supplied = call_role_with_memory(
            session, "empiricist_assessment", "empiricist", "analysis", assessment_prompt,
            EMPIRICIST_ASSESSMENT_SCHEMA, 240, 20330000 + index * 10,
            model, memory, histories, record, shared_memory_enabled)
        merge_supplied(supplied, assessment_supplied)
        evidence = append_retrieval_evidence(evidence, assessment_trace)
        design_evidence = append_retrieval_evidence(design_evidence, assessment_trace)
    write(session / "empiricist-assessment.json", empirical)
    if saved_assessment is None:
        histories.add("empiricist", "assessment", empirical, assessment_prompt)
    if scientific_review is None and record.get("scientific_supervision"):
        from orchestrator_review import REVIEW_SCHEMA, review_prompt, adjudicate_review
        diagnostic = facts.get("agent_r_diagnostic")
        prompt = "Fixed Research Question: " + research_question + "\n" + review_prompt(
            inquiry=inquiry, contract=contract, plan=plan, primary_facts=facts,
            diagnostic_code=(diagnostic or {}).get("code"), diagnostic_receipt=diagnostic,
            empirical_assessment=empirical)
        proposed, rendered, trace, extra = call_role_with_memory(session, "orchestrator_scientific_review",
            "orchestrator", "analysis", prompt, REVIEW_SCHEMA, 768, 20500000 + index * 10,
            model, memory, histories, record, shared_memory_enabled)
        merge_supplied(supplied, extra)
        evidence = append_retrieval_evidence(evidence, trace)
        scientific_review = adjudicate_review(proposed, facts, diagnostic, diagnostic_code=(diagnostic or {}).get("code"))
        root = histories.root.parent
        scientific_review["sources"] = [{"path": str((session / name).relative_to(root)),
            "sha256": hashlib.sha256((session / name).read_bytes()).hexdigest()}
            for name in ("plan.json", "tool-result.json", "current-result-facts.json", "empiricist-assessment.json")]
        scientific_review["execution_scope"] = "Post-result resume reviews saved work; requested repairs remain pending for the next research execution."
        write(session / "scientific-review.json", scientific_review)
        record["scientific_review"] = scientific_review
        record["scientific_review_source"] = {"path": str((session / "scientific-review.json").relative_to(root)),
            "sha256": hashlib.sha256((session / "scientific-review.json").read_bytes()).hexdigest()}
        histories.add("orchestrator", "scientific_review", scientific_review, rendered)
    route("decision", "hypothesis-agent", {"result_id": result_id,
        "result_facts": facts, "empiricist_assessment": empirical})
    requested = decision_schema(result_id, set(supplied))
    decision_prompt = (
        role_prompt("Hypothesis agent: scientific decision", contract, evidence, facts, memory_format="tsv") +
        "\nFixed Research Question: " + research_question +
        "\nCurrent inquiry: " + inquiry +
        "\nEmpiricist assessment:\n" + json.dumps(empirical) +
        "\nOrchestrator scientific review and pending work (review is an assessment, not proof):\n" +
        json.dumps(scientific_review or {"status": "disabled_by_condition" if record.get("orchestrator_assessment_enabled") is False else "legacy_not_reviewed"}) +
        "\nReturn only action, basis_keywords, next_inquiry and evidence_ids. Cite the current evidence_id. "
        "In basis_keywords state the learned fact and remaining uncertainty. Choose a next inquiry that "
        "resolves that uncertainty or distinguishes the rival explanation, not a search for significance. "
        "Maintaining, revising or abandoning is allowed; justify repeated tests by reproducibility or robustness. "
        "An execution_reuse receipt means the identical prior analysis was reused, not a new independent test. "
        "Maintaining a conclusion is valid; when choosing revise, distinguish a proposed change from an actually implemented change. "
        "Do not infer absence of an effect from non-significance, or a stronger effect from coefficients "
        "on different scales/models/samples. Leave unavailable comparisons unresolved.")
    record["decision_status"] = "running"
    decision, decision_prompt, _, _ = call_role_with_memory(
        session, "hypothesis_decision", "hypothesis", "hypothesis", decision_prompt,
        requested, 320, 20340000 + index * 10, model, memory, histories, record,
        shared_memory_enabled,
        final_schema_builder=lambda extra: decision_schema(result_id, set(supplied) | set(extra)))
    if set(decision) != set(DECISION_SCHEMA["properties"]) or result_id not in decision["evidence_ids"]:
        raise ValueError("Decision lacks the current result receipt")
    if record.get("scientific_supervision") or scientific_review is not None:
        decision, decision_review = review_final_decision(session, index, memory, model, histories, record,
            decision=decision, contract=contract, facts=facts, scientific_review=scientific_review,
            supplied=supplied, research_question=research_question, inquiry=inquiry,
            shared_memory_enabled=shared_memory_enabled)
        if decision is None:
            record.update(status="partial", decision_status="blocked", analysis=plan["analysis"],
                          pending_work=decision_review, workflow_status="decision_blocked",
                          research_task_status=(scientific_review or {}).get("requested_test_status", "unverifiable"))
            return {}
    write(session / "decision.json", decision)
    histories.add("hypothesis", "decision", decision, decision_prompt)
    record["decision_id"] = (memory.publish_decision(index, record["hypothesis_id"], result_id,
        decision, session / "decision.json") if shared_memory_enabled else "decision-%04d" % index)
    check = record["statistical_assessment"]
    diagnostic = facts.get("agent_r_diagnostic")
    unresolved_diagnostic = bool(diagnostic and diagnostic.get("status") != "completed")
    unresolved_review = scientific_review is not None and scientific_review.get("status") != "accepted"
    record.update(status="partial" if unresolved_diagnostic or unresolved_review else "completed", decision_status="succeeded", analysis=plan["analysis"],
                  research_task_status=("diagnostic_unresolved" if unresolved_diagnostic else
                      "diagnostic_executed_unvalidated" if diagnostic else "primary_only"),
                  assessment_followup=empirical.get("diagnostic_action"),
                  support_claim=check["support"], support_scope=check["scope"])
    if scientific_review is not None:
        record.update(research_task_status=scientific_review["requested_test_status"],
                      scientific_review_status=scientific_review["status"],
                      scientific_validity=scientific_review["scientific_validity"],
                      claim_consistency=scientific_review["claim_consistency"],
                      workflow_status="handoffs_completed")
    return decision


def merge_supplied(supplied, additional):
    for identifier, record in additional.items():
        if identifier in supplied and supplied[identifier] != record:
            raise ValueError("On-demand evidence differs from the existing source-verified record")
        supplied[identifier] = record


def append_retrieval_evidence(evidence, trace):
    if trace is None or not trace.get("brief") or trace["brief"] in evidence:
        return evidence
    return evidence + "\n\n" + trace["brief"]


def validate_retrieved_claims(memory, claims, traces):
    traces = [part for trace in traces for part in trace.get("retrieval_traces", [trace])]
    outcomes = []
    for claim in claims:
        identifier = claim.get("evidence_id") if isinstance(claim, dict) else None
        matches = [trace for trace in traces if identifier in
                   (set(trace.get("selected_ids", [])) | set(trace.get("delivered_equivalent_ids", [])))]
        readable = [trace for trace in matches if trace.get("actual_delivery", {}).get("full_packet_delivered") is not False]
        if matches and not readable:
            outcomes.append({"claim": claim, "status": "unverifiable", "reason": "Source packet was only partially delivered; exact field visibility not verified."})
            continue
        candidates = [memory.validate_context_claims([claim], trace)[0] for trace in readable]
        if candidates:
            outcomes.append(next((item for item in candidates if item["status"] == "verified"), candidates[0]))
        else:
            outcomes.append(memory.validate_context_claims([claim], traces[0])[0])
    return outcomes


def call_role_with_memory(session, stage, role, retrieval_role, prompt, schema, tokens, seed,
                          model, memory, histories, record, shared_memory_enabled,
                          final_schema_builder=None):
    """Research roles answer directly or retrieve source-linked evidence before answering."""
    if not shared_memory_enabled:
        current_prompt = histories.prompt(role, prompt, shared_memory=False)
        return model(stage, current_prompt, schema, tokens, seed), prompt, None, {}
    if memory is None:
        raise ValueError("Shared-memory role cannot request retrieval without its isolated memory store")
    if ADAPTIVE_SHARED_CONTEXT:
        return adaptive_memory_loop(session, stage, role, retrieval_role, prompt, schema, tokens, seed,
                                    model, memory, histories, record, final_schema_builder)
    current_prompt = histories.prompt(role, prompt, shared_memory=True)

    request_schema = {"anyOf": [
        {"type": "object", "properties": {
            "action": {"const": "retrieve_memory"},
            "query": {"type": "string", "minLength": 1, "maxLength": 240},
            "answer": {"type": "null"}},
            "required": ["action", "query", "answer"], "additionalProperties": False},
        {"type": "object", "properties": {
            "action": {"const": "answer"}, "query": {"type": "null"},
            "answer": deepcopy(schema)},
            "required": ["action", "query", "answer"], "additionalProperties": False},
    ]}
    action_instruction = (
        "\nChoose exactly one action. For a shared-memory lookup, return action=retrieve_memory, a concise "
        "query or exact evidence ID, and answer=null. For a final answer without lookup, return action=answer, "
        "query=null, and the complete answer under the answer field. A retrieval request does not require or "
        "accept a draft scientific answer; after lookup you receive the exact TSV and answer once under the original schema.")
    request_prompt = current_prompt + action_instruction
    response = model(stage, request_prompt, request_schema, tokens, seed)
    if not isinstance(response, dict) or response.get("action") not in {"retrieve_memory", "answer"}:
        raise ValueError("Memory-tool response must choose retrieve_memory or answer")
    action = response["action"]
    query = response.get("query")
    answer = response.get("answer")
    if action == "answer":
        if query is not None or not isinstance(answer, dict):
            raise ValueError("Final answer action requires query=null and an answer object")
        return answer, prompt + action_instruction, None, {}
    if answer is not None:
        raise ValueError("Memory retrieval action must not contain a draft answer")
    if not isinstance(query, str) or not query.strip() or len(query) > 240:
        raise ValueError("Memory query must be a non-empty string of at most 240 characters")
    ordinal = len(record.setdefault("on_demand_retrievals", [])) + 1
    request_path = Path(session) / ("memory-tool-%02d-request.json" % ordinal)
    write(request_path, {"stage": stage, "role": role, "query": query,
                         "request_response": response, "request_prompt": request_prompt})
    histories.add(role, stage + "_memory_request", {"query": query}, prompt + action_instruction)
    trace, tool_supplied = memory.retrieve_tsv(query, role=retrieval_role)
    if (trace.get("query") != query or not isinstance(trace.get("brief"), str) or
            not isinstance(trace.get("request_id"), str)):
        raise ValueError("Memory tool returned a mismatched or incomplete retrieval trace")
    trace_path = Path(session) / ("memory-tool-%02d.json" % ordinal)
    write(trace_path, trace)
    request = {"stage": stage, "role": role, "query": query, "request_id": trace["request_id"],
               "context_node_ids": trace.get("context_manifest", {}).get(
                   "source_node_ids", trace.get("upstream_selected_ids", [])),
               "result_ids": trace.get("selected_ids", []),
               "anchor_ids": [anchor["id"] for anchor in trace.get("anchors", [])],
               "trace_path": trace_path.name}
    record["on_demand_retrievals"].append(request)

    final_prompt = prompt + "\n\nON_DEMAND_SHARED_MEMORY (source-verified TSV; relevance is not truth):\n" + trace["brief"] + (
        "\nUse this retrieved evidence where relevant and return your final response under the original schema. "
        "No further memory request is available in this stage.")
    rendered_final_prompt = histories.prompt(role, final_prompt, shared_memory=True)
    final_schema = final_schema_builder(tool_supplied) if final_schema_builder else schema
    return model(stage, rendered_final_prompt, final_schema, tokens, seed + 1), final_prompt, trace, tool_supplied


def adaptive_memory_loop(session, stage, role, retrieval_role, prompt, schema, tokens, seed,
                         model, memory, histories, record, final_schema_builder):
    """C may read additional evidence/pages until it answers or its allocation ends."""
    supplied, traces, receipts, seen = {}, [], [], {}
    archived = {}
    current = prompt
    current_paged = False
    paged_queries = {}
    capacity_failure = None

    def archive_view(text, label):
        """Archive complete text and advertise exact source pages; never discard its middle silently."""
        histories.add(role, "context_source", {"kind": label, "chars": len(text)}, text)
        number = len(histories._items(role))
        first = histories.recall(role, f"private-turn:{number}:0")
        last = histories.recall(role, f"private-turn:{number}:{max(0, first['total_chars'] - 12000)}")
        return (f"\n{label}: COMPLETE SOURCE ARCHIVED as private-turn:{number}:0. "
                "The following are first/last pages, not the whole source; retrieve middle pages using next_offset "
                "before making claims about absent fields.\n" + json.dumps(first, ensure_ascii=False) +
                ("\n" + json.dumps(last, ensure_ascii=False) if last["offset"] else "")), number
    instruction = (
        "\nChoose action=answer with query=null and your complete answer, or action=retrieve_memory "
        "with a concise query/exact evidence ID and answer=null. You may retrieve another source or page "
        "before answering. For your original private exchanges use private-turn:TURN:OFFSET, starting at "
        "OFFSET=0 and following next_offset if needed. Private prose is unverified; machine result receipts "
        "take precedence. Do not repeat an identical query already returned below. Answer when the evidence "
        "is sufficient; retrieval does not itself require a new scientific test.")
    calls = 0
    while True:
        final_schema = final_schema_builder(supplied) if final_schema_builder else schema
        request_schema = {"anyOf": [
            {"type": "object", "properties": {"action": {"const": "retrieve_memory"},
                "query": {"type": "string", "minLength": 1}, "answer": {"type": "null"}},
             "required": ["action", "query", "answer"], "additionalProperties": False},
            {"type": "object", "properties": {"action": {"const": "answer"},
                "query": {"type": "null"}, "answer": deepcopy(final_schema)},
             "required": ["action", "query", "answer"], "additionalProperties": False}]}
        def working_prompt():
            index = "\nARCHIVED TOOL RECEIPTS (exact own-role sources; retrieve if needed):\n" + "\n".join(
                query + " -> private-turn:" + str(number) + ":0" for query, number in archived.items()) if archived else ""
            return current + "".join(text for _, text in receipts) + index + instruction
        final_prompt = working_prompt()
        rendered = histories.prompt(role, final_prompt, shared_memory=True)
        transport_limit = getattr(model, "input_character_limit", None)
        if not isinstance(transport_limit, int):
            transport_limit = None
        if capacity_failure is not None:
            previous_size, _ = capacity_failure
            transport_limit = min(transport_limit, previous_size - 1) if transport_limit is not None else previous_size - 1
        while transport_limit is not None and len(rendered) > transport_limit:
            base_rendered = histories.prompt(role, current + instruction, shared_memory=True)
            if len(base_rendered) > transport_limit:
                if current_paged:
                    raise ContextCapacityError("Transport cannot fit the fixed instructions and source-page metadata")
                current, _ = archive_view(current, "Current request")
                current_paged = True
                final_prompt = working_prompt()
                rendered = histories.prompt(role, final_prompt, shared_memory=True)
                continue
            if receipts:
                query, text = receipts.pop(0)
                cached_page = paged_queries.get(query)
                page_view, number = cached_page if cached_page and text == cached_page[0] else archive_view(text, "Tool receipt " + query)
                archived[query] = number
                # Keep the newest result visible, paged if it alone exceeds the actual transport.
                if not receipts:
                    if cached_page and text == cached_page[0]:
                        if current_paged:
                            raise ContextCapacityError("Transport cannot fit source-page metadata and fixed request")
                        current, _ = archive_view(current, "Current request")
                        current_paged = True
                    receipts.append((query, page_view))
                    paged_queries[query] = (page_view, number)
            else:
                if current_paged:
                    raise ContextCapacityError("Transport cannot fit the source index and fixed instructions")
                current, _ = archive_view(current, "Current request")
                current_paged = True
            final_prompt = working_prompt()
            rendered = histories.prompt(role, final_prompt, shared_memory=True)
        if capacity_failure is not None and len(rendered) >= capacity_failure[0]:
            raise capacity_failure[1]
        try:
            response = model(stage, rendered, request_schema, tokens, seed + calls)
        except ContextCapacityError as error:
            # Retry only a genuinely smaller source-linked working view, never the same oversized call.
            capacity_failure = (len(rendered), error)
            record.setdefault("context_compaction_recoveries", []).append({
                "stage": stage, "failed_prompt_chars": len(rendered),
                "failed_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "failure_type": type(error).__name__, "status": "compacting_before_retry"})
            calls += 1
            continue
        if capacity_failure is not None:
            record["context_compaction_recoveries"][-1].update(status="recovered", retry_prompt_chars=len(rendered))
            capacity_failure = None
        delivered_queries = {query for query, text in receipts if query in seen and text == seen[query]}
        for trace in traces:
            delivery = trace.setdefault("actual_delivery", {"full_packet_delivered": False})
            delivery["full_packet_delivered"] |= trace["query"] in delivered_queries
            delivery["status"] = "complete" if trace["query"] in delivered_queries else "partial_or_archived"
        record.setdefault("context_deliveries", []).append({"stage": stage, "call": calls + 1,
            "current_request_complete": not current_paged, "prompt_chars": len(rendered),
            "full_tool_queries": sorted(delivered_queries),
            "archived_tool_queries": sorted(archived),
            "source_trace_delivery": [{"request_id": trace["request_id"], **trace["actual_delivery"]} for trace in traces]})
        calls += 1
        if not isinstance(response, dict) or response.get("action") not in {"retrieve_memory", "answer"}:
            raise ValueError("Memory-tool response must choose retrieve_memory or answer")
        query, answer = response.get("query"), response.get("answer")
        if response["action"] == "answer":
            if query is not None or not isinstance(answer, dict):
                raise ValueError("Final answer action requires query=null and an answer object")
            combined = None
            if traces:
                combined = dict(traces[-1])
                combined.update(retrieval_traces=traces,
                    brief="\n\n".join(trace["brief"] for trace in traces),
                    selected_ids=list(dict.fromkeys(identifier for trace in traces for identifier in trace["selected_ids"])),
                    delivered_equivalent_ids=list(dict.fromkeys(identifier for trace in traces
                        for identifier in trace.get("delivered_equivalent_ids", []))))
            return answer, final_prompt, combined, supplied
        if answer is not None or not isinstance(query, str) or not query.strip():
            raise ValueError("Memory retrieval requires a non-empty query and answer=null")
        ordinal = len(record.setdefault("memory_tool_requests", [])) + 1
        request_path = Path(session) / ("memory-tool-%02d-request.json" % ordinal)
        write(request_path, {"stage": stage, "role": role, "query": query,
                            "request_response": response, "request_prompt": rendered})
        record["memory_tool_requests"].append({"stage": stage, "role": role, "query": query,
                                               "request_path": request_path.name})
        histories.add(role, stage + "_memory_request", {"query": query}, final_prompt)
        if query in seen:
            # Return feedback instead of re-running the same retrieval or duplicating its packet.
            feedback = "\nTOOL FEEDBACK: identical query already delivered: " + query + ". Use its receipt or request a different page/source.\n"
            if query in archived:
                receipts.append((query, paged_queries[query][0] if query in paged_queries else seen[query]))
                del archived[query]
            elif not any(text == feedback for _, text in receipts):
                receipts.append((query, feedback))
            continue
        if query.startswith("private-turn:"):
            try:
                page = histories.recall(role, query)
            except ValueError as error:
                seen[query] = "\nPRIVATE RECALL ERROR: " + str(error) + "\n"
                receipts.append((query, seen[query]))
                continue
            write(Path(session) / ("private-history-%02d.json" % ordinal), page)
            record.setdefault("private_history_recalls", []).append({key: page[key] for key in
                ("role", "turn", "offset", "next_offset", "total_chars", "source_sha256")})
            seen[query] = "\nEXACT PRIVATE ARCHIVE PAGE:\n" + json.dumps(page, ensure_ascii=False)
            receipts.append((query, seen[query]))
            continue
        trace, additional = memory.retrieve_tsv(query, role=retrieval_role, source_lookup=True)
        if (trace.get("query") != query or not isinstance(trace.get("brief"), str) or
                not isinstance(trace.get("request_id"), str)):
            raise ValueError("Memory tool returned a mismatched or incomplete retrieval trace")
        merge_supplied(supplied, additional)
        trace["actual_delivery"] = {"full_packet_delivered": False, "status": "retrieved_not_yet_delivered"}
        trace_path = Path(session) / ("memory-tool-%02d.json" % ordinal)
        write(trace_path, trace)
        record.setdefault("on_demand_retrievals", []).append({"stage": stage, "role": role,
            "query": query, "request_id": trace["request_id"], "context_node_ids": trace.get(
                "context_manifest", {}).get("source_node_ids", trace.get("upstream_selected_ids", [])),
            "result_ids": trace.get("selected_ids", []),
            "anchor_ids": [anchor["id"] for anchor in trace.get("anchors", [])], "trace_path": trace_path.name})
        traces.append(trace)
        seen[query] = "\nON_DEMAND_SHARED_MEMORY (source-verified TSV; relevance is not truth):\n" + trace["brief"]
        receipts.append((query, seen[query]))


def execute_four_role_attempt(session, index, memory, model, analysis_runner, *, data_identity, diagnostic_runner=None,
                              research_question=QUESTION, inquiry=QUESTION,
                              previous_plan=None, previous_graph=None,
                              memory_format="tsv", condition="C"):
    """One checked Orchestrator→Hypothesis→Empiricist→Runner cycle and handbacks."""
    if memory_format != "tsv":
        raise ValueError("Four-role pilot requires complete TSV evidence context")
    session = Path(session)
    shared_memory_enabled = condition == "C"
    if condition not in ("A", "B", "C"):
        raise ValueError("Condition must be A, B, or C")
    if shared_memory_enabled and memory is None:
        raise ValueError("Condition C requires its isolated shared-memory store")
    if not shared_memory_enabled and memory is not None:
        raise ValueError("Conditions A/B must not receive a shared-memory interface")
    if not research_question.strip() or not inquiry.strip():
        raise ValueError("Research question and current inquiry must be non-empty")
    if shared_memory_enabled:
        memory.ensure_research_question(research_question, session.parent / "config.json")
    workflow = FourRoleOrchestrator(session, index, research_question=research_question, inquiry=inquiry)
    histories = RoleHistory(session.parent)
    record = {"attempt": index, "topology": "orchestrator-hypothesis-empiricist-runner",
              "orchestration_mode": "deterministic",
              "scientific_supervision": "separate_orchestrator_model_review_with_execution_handback" if ORCHESTRATOR_ASSESSMENT else None,
              "research_role_layout": RESEARCH_ROLE_LAYOUT,
              "orchestrator_assessment_enabled": ORCHESTRATOR_ASSESSMENT,
              "actor_instances": {"hypothesis-agent": histories._key("hypothesis"),
                                  "empiricist-agent": histories._key("empiricist"),
                                  "orchestrator": "orchestrator", "runner": "deterministic_tool"},
              "condition": condition,
              "shared_memory_enabled": shared_memory_enabled,
              "role_history_policy": ("bounded_c_checkpoint_with_private_recall" if shared_memory_enabled and ADAPTIVE_SHARED_CONTEXT else
                                      "latest_output_per_stage_with_full_archive" if shared_memory_enabled and COMPACT_SHARED_HISTORY else
                                      "full_role_local_history"),
              "role_history_enabled": True,
              "status": "started", "analysis_status": "not_started", "decision_status": "not_started",
              "retrieved_ids": []}
    plan, decision = {}, {}

    def route(stage, actor, artifact=None):
        event = workflow.route(stage, actor, artifact)
        workflow.write(session / "orchestrator.json", status="in_progress")
        histories.add("orchestrator", stage, {"routing_mode": "deterministic", "next_stage": stage,
            "research_question": research_question, "current_inquiry": inquiry, "sources": event["sources"]})

    try:
        retrieval_query = (research_question if inquiry == research_question else
                           research_question + "\nCurrent inquiry: " + inquiry)
        trace, supplied = memory.retrieve_tsv(retrieval_query, role="hypothesis") if shared_memory_enabled else (
            {"brief": NO_EVIDENCE, "selected_ids": [], "nodes": [], "edges": []}, [])
        evidence = trace["brief"] if trace["brief"] else NO_EVIDENCE
        prior_result_ids = histories.prior_result_ids()
        prior_result_ids.update(item for item in trace["selected_ids"]
                                if re.fullmatch(r"result-\d{4}", item))
        record["retrieved_ids"] = list(trace["selected_ids"])
        if shared_memory_enabled and ADAPTIVE_SHARED_CONTEXT:
            trace["actual_delivery"] = {"full_packet_delivered": False, "status": "retrieved_not_yet_delivered"}
        write(session / "hypothesis-retrieval.json", trace)
        (session / "hypothesis-memory.tsv").write_text(evidence)
        route("hypothesis", "hypothesis-agent", {"state": "initial",
              "research_question": research_question, "current_inquiry": inquiry})
        hypothesis_prompt = (
            role_prompt("Hypothesis agent", evidence=evidence, memory_format="tsv") +
            "\nFixed Research Question: " + research_question +
            "\nCurrent inquiry: " + inquiry +
            "\nPropose one falsifiable hypothesis and primary focal_test; do NOT choose an estimator. "
            "Use the documented data meanings and unit of observation. State a substantive rival_explanation "
            "and use expected_pattern to say what observation would distinguish it, or explicitly say that "
            "the available data cannot distinguish it. Choose the next test for information, not significance. "
        "In required_terms list the exact predictor/covariate terms necessary for this claim, including the focal term; "
            "do not list strata, fixed-effect, or cluster identifiers because those use separate Runner fields. "
            "List every term needed by expected_pattern in required_terms, up to the Runner's 12-term capacity. "
            "Supported terms are dataset predictors, I(x^2), poly(x,2), and pairwise x:y interactions; "
            "higher powers are unavailable, so do not describe them as an executable next test. "
            "Write complete scientific statements within the output bounds. Do not claim relative strength "
            "or a causal mechanism from a standalone association. Shape or conditional claims require their "
            "corresponding terms and adequate diagnostic evidence, not merely a significant coefficient. "
            "Expected direction precedes results. Historical claims require supplied IDs/numbers. "
            + ("This is the first attempt in a fresh condition: use adaptation=initial, the exact initial basis, "
               "and no prior evidence IDs or historical claims. " if index == 1 else
               "This is attempt %s, not an initial attempt. Do not use adaptation=initial. "
               "Use a decision and evidence ID from these available prior results: %s. " %
               (index, ", ".join(sorted(prior_result_ids)) or "none")))
        contract, hypothesis_prompt, hypothesis_tool_trace, hypothesis_tool_supplied = call_role_with_memory(
            session, "hypothesis", "hypothesis", "hypothesis", hypothesis_prompt,
            hypothesis_schema(index, prior_result_ids), 768, 20310000 + index * 10,
            model, memory, histories, record, shared_memory_enabled,
            final_schema_builder=lambda extra: hypothesis_schema(index, prior_result_ids | set(extra)))
        if shared_memory_enabled and ADAPTIVE_SHARED_CONTEXT:
            delivered = any(item["stage"] == "hypothesis" and item["current_request_complete"]
                            for item in record.get("context_deliveries", []))
            trace["actual_delivery"] = {"full_packet_delivered": delivered,
                "status": "complete" if delivered else "partial_or_archived"}
            write(session / "hypothesis-context-delivery.json", {"request_id": trace.get("request_id"),
                **trace["actual_delivery"], "calls": record.get("context_deliveries", [])})
        if hypothesis_tool_trace is not None:
            merge_supplied(supplied, hypothesis_tool_supplied)
            prior_result_ids.update(identifier for identifier in hypothesis_tool_supplied
                                    if re.fullmatch(r"result-\d{4}", identifier))
            trace_for_claims = [trace, hypothesis_tool_trace]
        else:
            trace_for_claims = [trace]
        write(session / "hypothesis-response.json", contract)
        if index == 1 and (contract["adaptation"] != {
                "decision": "initial", "basis": "Initial attempt; no prior evidence.", "evidence_ids": []}
                or contract["historical_claims"]):
            raise ValueError("First attempt cannot cite or invent prior history")
        if index > 1:
            adaptation = contract["adaptation"]
            cited = set(adaptation["evidence_ids"])
            if adaptation["decision"] == "initial" or not cited.issubset(prior_result_ids) or (
                    prior_result_ids and not cited):
                raise ValueError("Later attempt must make a non-initial decision grounded in available prior results")
        claims = (validate_retrieved_claims(memory, contract["historical_claims"], trace_for_claims)
                  if shared_memory_enabled else [{"claim": item, "status": "conversation_only_unverified"}
                                                  for item in contract["historical_claims"]])
        write(session / "historical-validation.json", claims)
        # An invalid optional citation must not discard a viable research cycle.
        # Preserve the raw response and validation receipt; only verified claims enter shared memory.
        if shared_memory_enabled:
            contract["historical_claims"] = [item["claim"] for item in claims if item["status"] == "verified"]
        write(session / "hypothesis.json", contract)
        histories.add("hypothesis", "hypothesis", contract, hypothesis_prompt)
        links = {}
        if previous_graph:
            if previous_plan and any(contract[field] != previous_plan.get(field)
                                     for field in ("hypothesis", "expected_pattern", "focal_test")):
                links["revision_of"] = previous_graph["hypothesis_id"]
            if previous_graph.get("decision_id"):
                links["follows_decision"] = previous_graph["decision_id"]
        h_id = "hypothesis-%04d" % index
        record["hypothesis_id"] = h_id

        design_query = research_question + "\nCurrent hypothesis: " + contract["hypothesis"]
        design_trace, _ = memory.retrieve_tsv(design_query, role="analysis") if shared_memory_enabled else (
            {"brief": NO_EVIDENCE, "selected_ids": [], "nodes": [], "edges": []}, [])
        design_evidence = design_trace["brief"] if design_trace["brief"] else NO_EVIDENCE
        write(session / "empiricist-retrieval.json", design_trace)
        (session / "empiricist-memory.tsv").write_text(design_evidence)
        route("empiricist_plan", "empiricist-agent", {"hypothesis_id": h_id, "hypothesis": contract})
        analysis_prompt = (
            role_prompt("Empiricist agent", contract, design_evidence, memory_format="tsv") +
            "\nFixed Research Question: " + research_question +
            "\nCurrent inquiry: " + inquiry +
            "\n" + PILOT_ANALYSIS_GUIDANCE + "\n"
            "Select the scientifically appropriate design, then specify a package-based analysis testing exactly this focal contract. "
            "Include every required_terms entry. In design_rationale say which claim this model tests; "
            "Copy dataset field names exactly from the data dictionary; do not add or remove underscores. "
            "in diagnostics identify any needed but unperformed comparison or design check. "
            "Do not describe a rival explanation as excluded unless this actual design can distinguish it. "
            "Do not alter the hypothesis/focal term. Put every term that defines the focal relationship in focal_terms: include nonlinear basis terms and interaction terms with their constituent main effects. Put only additional covariates in control_variables; focal_terms and control_variables must partition the Runner terms. Record DV/outcome type, focal predictor, focal_terms, controls, treatment/instrument, panel/time/cutoff fields when relevant, FE, SE/clusters, estimand, assumptions and diagnostics. "
            "Use only the documented dataset-supported design and exact Runner action in the schema; never invent or substitute a model. "
            "This is the planning stage: the Runner will execute the diagnostic_r_code you submit after this response. "
            "The planning LLM has no shell; that does not mean the reviewed execution host is unavailable. "
            "Do not report submitted code as already executed or claim the host is unavailable without an actual execution receipt. "
            "The Runner action is the executable specification: cluster='none' requires model_based SE and no cluster_variables; "
            "cluster='user_label' requires cluster_robust SE and cluster_variables=['user_label']. Never report an SE/cluster combination that differs from the action. "
            "For an additional diagnostic not offered by the standard tool, set diagnostic_r_code to standalone R code; otherwise use null. "
            "The sandbox exposes the CSV read-only at /data/styles.csv and a private /work directory, without network. "
            "Write 1–200 aggregate numeric diagnostics to /work/diagnostics.csv with exactly metric,value columns; "
            "metric names must be unique letters/digits/underscores and start with a letter; unavailable values may be NA. "
            "never output individual observations. Custom outputs are unverified and do not replace the primary model.")
        analysis_prompt += ("\n" + focal_execution_instruction(contract) +
            "\nThe current hypothesis explicitly requires these executed terms (including its declared adjustment variables): " +
            json.dumps(contract.get("required_terms", [])) +
            ". Include all of them in the proposed Runner terms; assign focal_terms vs control_variables according to their role. "
            "Do not postpone required adjustment variables as optional future robustness checks.\n" + DIAGNOSTIC_ENVIRONMENT)
        analysis, analysis_prompt, plan_tool_trace, plan_tool_supplied = call_role_with_memory(
            session, "empiricist_plan", "empiricist", "analysis", analysis_prompt,
            CODING_ANALYSIS_SCHEMA, 768, 20320000 + index * 10,
            model, memory, histories, record, shared_memory_enabled)
        if plan_tool_trace is not None:
            merge_supplied(supplied, plan_tool_supplied)
            design_evidence = append_retrieval_evidence(design_evidence, plan_tool_trace)
        write(session / "empiricist-plan.json", analysis)
        diagnostic_code = analysis.get("diagnostic_r_code")
        analysis = {key: value for key, value in analysis.items() if key != "diagnostic_r_code"}
        try:
            plan = build_plan(contract, analysis)
        except UnsupportedModelDesign as unsupported_error:
            write(session / "plan-validation-rejection.json", {"error": str(unsupported_error),
                "plan": analysis})
            raise
        except ValueError as initial_error:
            record["plan_repair_status"] = "running"
            repair_prompt = (
                role_prompt("Empiricist agent: deterministic plan repair", contract, design_evidence,
                            memory_format="tsv") +
                "\nFixed Research Question: " + research_question +
                "\nCurrent inquiry: " + inquiry +
                "\nOriginal rejected plan:\n" + encode(analysis) +
                "\nDeterministic gate error:\n" + str(initial_error) +
                "\nReturn one corrected analysis plan using the required schema. Repair only the action/specification; "
                "do not change the immutable hypothesis, estimand, or focal contract. Use exact dataset field names. "
                "Every unique Runner action term must occur exactly once across focal_terms and control_variables. "
                "Put nonlinear/interaction basis terms in focal_terms only when they define this focal claim; "
                "put other prespecified covariates in controls. Remove duplicate or unjustified terms; do not add "
                "terms just to satisfy the gate. required_terms are predictor/covariate terms; represent strata and "
                "cluster identifiers only in their dedicated action fields. "
                + focal_execution_instruction(contract) + " "
                "If the necessary analysis is unsupported or the data are insufficient, do not substitute an unrelated estimator.")
            repaired, repair_prompt, repair_trace, repair_supplied = call_role_with_memory(
                session, "empiricist_plan_repair", "empiricist", "analysis", repair_prompt,
                CODING_ANALYSIS_SCHEMA, 768, 20320001 + index * 10,
                model, memory, histories, record, shared_memory_enabled)
            if repair_trace is not None:
                merge_supplied(supplied, repair_supplied)
                design_evidence = append_retrieval_evidence(design_evidence, repair_trace)
            write(session / "empiricist-plan-repair.json", repaired)
            diagnostic_code = repaired.get("diagnostic_r_code")
            repaired = {key: value for key, value in repaired.items() if key != "diagnostic_r_code"}
            try:
                plan = build_plan(contract, repaired)
            except UnsupportedModelDesign as repair_error:
                write(session / "plan-repair-failure.json", {"first_error": str(initial_error),
                    "repair_error": str(repair_error), "initial_plan": analysis, "repaired_plan": repaired})
                record["plan_repair_status"] = "failed"
                raise
            except ValueError as repair_error:
                write(session / "plan-repair-failure.json", {"first_error": str(initial_error),
                    "repair_error": str(repair_error), "initial_plan": analysis, "repaired_plan": repaired})
                record["plan_repair_status"] = "failed"
                rejection = {
                    "attempt": index, "focal_test": contract["focal_test"],
                    "gate_error": str(repair_error),
                    "rejected_action_terms": (repaired.get("analysis") or {}).get("terms", []),
                    "source": {"path": str((session / "plan-repair-failure.json").relative_to(session.parent)),
                               "sha256": hashlib.sha256((session / "plan-repair-failure.json").read_bytes()).hexdigest()},
                    "scope": "Historical proposal rejected before analysis; this is not a statistical result.",
                }
                histories.add("empiricist", "plan_rejection", rejection, repair_prompt)
                histories.add("hypothesis", "plan_rejection", rejection)
                raise PlanRepairError("Plan failed the deterministic gate after one repair") from repair_error
            analysis, analysis_prompt = repaired, repair_prompt
            record["plan_repair_status"] = "succeeded"
        write(session / "plan.json", plan)
        # Keep unvalidated proposals in attempt artifacts, not the scientific graph.
        if shared_memory_enabled:
            memory.publish_hypothesis(index, contract, session / "hypothesis.json", research_question,
                                      session.parent / "config.json", **links)
        m_id = memory.publish_model(index, h_id, plan, session / "plan.json") if shared_memory_enabled else "model-%04d" % index
        record["model_id"] = m_id
        plan_source_path = session / ("empiricist-plan-repair.json" if record.get("plan_repair_status") == "succeeded" else "empiricist-plan.json")
        histories.add("empiricist", "plan", {**analysis, "diagnostic_r_code": diagnostic_code,
            "proposal_source": {"path": str(plan_source_path.relative_to(session.parent)),
                                "sha256": hashlib.sha256(plan_source_path.read_bytes()).hexdigest()}}, analysis_prompt)

        route("runner", "runner", {"model_id": m_id, "analysis": plan["analysis"],
                                     "model_specification": plan["model_specification"]})
        record["analysis_status"] = "running"
        result = analysis_runner(session, plan["analysis"])
        # Preserve the executed primary fit even if diagnostic repair consumes the remaining allocation.
        write(session / "primary-tool-result.json", result)
        record["tool_returncode"] = result["returncode"]
        record["analysis_status"] = "succeeded" if result["returncode"] == 0 else "failed"
        if diagnostic_code and result.get("returncode") == 0:
            diagnostic = execute_diagnostic_work(session, diagnostic_code, diagnostic_runner, model,
                memory, histories, record, contract=contract, inquiry=inquiry,
                shared_memory_enabled=shared_memory_enabled)
            result["agent_r_diagnostic"] = diagnostic
            record["diagnostic_work_status"] = diagnostic.get("status")
        record["tool_returncode"] = result["returncode"]
        record["analysis_status"] = "succeeded" if result["returncode"] == 0 else "failed"
        check = statistical_assessment(plan, result)
        write(session / "statistical-assessment.json", check)
        record["statistical_assessment"] = check
        sample_md5 = next((row.get("sample_membership_md5") for row in result.get("metrics", [])
                           if row.get("sample_membership_md5")), None)
        sample_id = "md5:" + sample_md5 if isinstance(sample_md5, str) and re.fullmatch(r"[a-fA-F0-9]{32}", sample_md5) else None
        result_id = "result-%04d" % index
        record["result_id"] = result_id
        facts = {**statistical_summary(check), "estimate": check.get("estimate"),
                 "estimates": result.get("estimates", []), "metrics": result.get("metrics", []),
                 "analysis": result.get("effective_action"), "analysis_status": record["analysis_status"],
                 "evidence_id": result_id, "result_facts_available": True,
                 "agent_prose_included": False, "scope": check["scope"]}
        if result.get("execution_reuse"):
            facts["execution_reuse"] = result["execution_reuse"]
        if result.get("agent_r_diagnostic"):
            facts["agent_r_diagnostic"] = result["agent_r_diagnostic"]
        diagnostic, empirical, review = facts.get("agent_r_diagnostic"), None, None
        if ORCHESTRATOR_ASSESSMENT:
            diagnostic, empirical, review, evidence, design_evidence = supervise_result(
                session, index, memory, model, histories, record, contract=contract, plan=plan,
                facts=facts, evidence=evidence, design_evidence=design_evidence, supplied=supplied,
                research_question=research_question, inquiry=inquiry, diagnostic_runner=diagnostic_runner,
                shared_memory_enabled=shared_memory_enabled)
        if diagnostic is not None:
            result["agent_r_diagnostic"] = diagnostic
            facts["agent_r_diagnostic"] = diagnostic
            record["diagnostic_work_status"] = diagnostic.get("status")
        result["metrics"] = deepcopy(facts["metrics"])
        if review is not None:
            result["scientific_review"] = facts["scientific_review"] = review
            result["scientific_review_source"] = facts["scientific_review_source"] = record["scientific_review_source"]
        else:
            record["scientific_validity"] = "unverified"
            record["review_policy"] = "disabled_by_experimental_condition; not an approval"
        # Publish once after supervision: accepted handoffs never point to an overwritten source.
        write(session / "tool-result.json", result)
        if shared_memory_enabled:
            published = memory.publish_result(index, m_id, session / "tool-result.json", session / "plan.json",
                data_identity=data_identity, sample_identity=sample_id,
                sample_recipe=sample_provenance(data_identity, result), actor="runner")
            if published != result_id:
                raise ValueError("Published result ID differs from the supervised evidence ID")
        write(session / "current-result-facts.json", facts)

        decision = finish_result(session, index, memory, model, route, histories, record,
            contract=contract, plan=plan, facts=facts, evidence=evidence, design_evidence=design_evidence,
            supplied=supplied, research_question=research_question, inquiry=inquiry,
            shared_memory_enabled=shared_memory_enabled, saved_assessment=empirical, scientific_review=review)
        workflow.write(session / "orchestrator.json", status=record.get("workflow_status", "handoffs_completed"))
    except Exception as error:
        if (record.get("analysis_status") == "succeeded" and "facts" in locals()
                and not (session / "tool-result.json").exists()):
            # Interrupted supervision must not erase a successful primary analysis or its latest diagnostic.
            try:
                if record.get("diagnostic_executions"):
                    source = record["diagnostic_executions"][-1]["source"]
                    path = session.parent / source["path"]
                    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                        raise ValueError("Interrupted diagnostic receipt changed")
                    facts["agent_r_diagnostic"] = {**json.loads(path.read_text()), "receipt_source": source}
                pending_diagnostic = facts.get("agent_r_diagnostic")
                facts["metrics"] = diagnostic_metrics(facts.get("metrics", []), pending_diagnostic)
                result["metrics"] = deepcopy(facts["metrics"])
                result["scientific_review_pending"] = facts["scientific_review_pending"] = True
                if pending_diagnostic is not None:
                    result["agent_r_diagnostic"] = pending_diagnostic
                write(session / "tool-result.json", result)
                if shared_memory_enabled:
                    memory.publish_result(index, m_id, session / "tool-result.json", session / "plan.json",
                        data_identity=data_identity, sample_identity=sample_id,
                        sample_recipe=sample_provenance(data_identity, result), actor="runner")
                write(session / "current-result-facts.json", facts)
                record["supervision_status"] = "interrupted_review_pending"
                if workflow.cursor == 3:
                    route("empiricist_assessment", "empiricist-agent", {"result_id": result_id,
                        "analysis_status": record["analysis_status"], "result_facts": facts})
            except Exception as preservation_error:
                record["post_result_preservation_error"] = type(preservation_error).__name__ + ": " + str(preservation_error)
        deadline_censored = isinstance(error, RuntimeIsolationError) and getattr(error, "deadline_exhausted", False) is True
        model_budget_censored = type(error).__name__ == "ModelCallBudgetExceeded"
        censored = deadline_censored or model_budget_censored
        context_capacity = isinstance(error, ContextCapacityError)
        plan_repair_failed = isinstance(error, PlanRepairError)
        unsupported_plan = isinstance(error, UnsupportedModelDesign)
        if unsupported_plan:
            rejection = {"attempt": index, "failure_type": type(error).__name__,
                "reason": str(error), "scope": "Proposal unavailable with current data/tool capabilities; no statistical test occurred.",
                "hypothesis": locals().get("contract", {}), "plan": locals().get("analysis", {})}
            rejection_path = session / "unsupported-design.json"
            write(rejection_path, rejection)
            rejection["source"] = {"path": str(rejection_path.relative_to(session.parent)),
                                    "sha256": hashlib.sha256(rejection_path.read_bytes()).hexdigest()}
            histories.add("empiricist", "plan_rejection", rejection)
            histories.add("hypothesis", "plan_rejection", rejection)
        if record["decision_status"] == "running":
            record["decision_status"] = "censored" if censored else "failed"
        record.update(status="censored" if censored else
                      "needs-attention" if context_capacity or plan_repair_failed or unsupported_plan else "failed",
                      failure_type=type(error).__name__, diagnostic=str(error)[:300],
                      failure_domain="model_call_budget" if model_budget_censored else
                      "context_capacity" if context_capacity else
                      "plan_repair_failed" if plan_repair_failed else
                      "inference_deadline_censor" if deadline_censored else
                      "data_insufficient" if isinstance(error, DataInsufficientDesign) else
                          "unsupported_method" if isinstance(error, UnsupportedModelDesign) else
                          "four_role_harness_or_contract; not a scientific-error score")
        workflow.write(session / "orchestrator.json", status=record["status"])
    return record, plan, decision
