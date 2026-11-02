"""Single-writer graph publication in an allocation or explicit local workstation mode."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import uuid

from .evidence import (_source, _term, analysis_key, encode, ground_trace, result_assessment,
                       statistical_assessment, statistical_summary, supervision_fields, validate_claims, validate_focal_test, verify_record)
from .memory import build_snapshot, retrieve, submit_candidate
from .tabular import render_tsv


JUDGMENT_RELATIONS = {"supports", "challenges", "assumption_constraint"}


def judgment_edges(kind, target, payload, source, provenance, check=None):
    """Scoped, source-derived relations; none certifies scientific validity."""
    if kind == "AnalysisModel":
        assumptions = (payload.get("model_specification") or {}).get("identification_assumptions")
        if not isinstance(assumptions, str) or not assumptions.strip():
            return []
        details = {"relation": "assumption_constraint", "basis": assumptions,
                   "assertion_status": "declared_not_validated",
                   "scope": "Interpretation of this hypothesis through this model is conditional on the declared assumptions; their validity is unverified."}
    elif kind == "ResultAssessment":
        if check.get("support") == "supported":
            relation = "supports"
        elif (check.get("reason") == "significant_effect_opposes_declared_direction" and
              ((check["primary_test"]["expected_direction"] == "positive" and Decimal(str(check["estimate"]["coefficient"])) < 0) or
               (check["primary_test"]["expected_direction"] == "negative" and Decimal(str(check["estimate"]["coefficient"])) > 0))):
            relation = "challenges"
        else:
            return []  # Non-significance or an unavailable test is not contradictory evidence.
        details = {"relation": relation, "assertion_status": "derived_primary_statistical_test",
                   "basis": {key: deepcopy(check.get(key)) for key in
                             ("primary_test", "estimate", "alpha", "reason")},
                   "scope": check["scope"]}
    else:
        return []
    return [{"target": target, "source": str(Path(source).resolve()),
             "provenance": deepcopy(provenance), "scientific_validity": "unverified", **details}]


def selected_context_nodes(nodes, ranked_ids):
    """Close ranked evidence over scientific sources, not the entire revision history."""
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Retrieval must supply distinct selected seeds")
    pending = list(ranked_ids)
    if "research-question" in nodes:
        pending.append("research-question")
    included = set()
    while pending:
        identifier = pending.pop()
        if identifier in included:
            continue
        if identifier not in nodes:
            raise ValueError("Selected scientific endpoint is absent from the snapshot")
        included.add(identifier)
        node = nodes[identifier]
        result_id = node.get("result_id")
        if result_id in nodes:
            pending.append(result_id)
        pending.extend(edge["target"] for edge in node.get("edges", [])
                       if edge["relation"] in {"produced_by", "intended_test_for", "addresses", "based_on"} | JUDGMENT_RELATIONS)
    return included


def evidence_group_key(record, data_identity, sample_identity):
    """Missing row-membership identity must never be replaced by a sample count."""
    if not data_identity or not sample_identity:
        return None
    payload = {"data": data_identity, "sample": sample_identity,
               "action": analysis_key(record["result"]["analysis"]),
               "estimates": record["result"]["estimates"],
               "metrics": record["result"]["metrics"],
               "returncode": record["result"]["returncode"],
               "primary_check": record["assessment"].get("statistical_check")}
    if supervision_fields(record["assessment"]):
        payload["supervision"] = supervision_fields(record["assessment"])
    return hashlib.sha256(encode(payload).encode()).hexdigest()


class TwoRoleMemory:
    OWNERS = {"ResearchQuestion": "user", "Hypothesis": "hypothesis-agent",
              "AnalysisModel": "analysis-agent", "ResultAssessment": "analysis-agent",
              "Decision": "hypothesis-agent"}

    def __init__(self, root, *, arm="memory", embedding_backend="lexical", model_name=None, retrieval_strategy="ppr", local=False):
        from .execution import require_execution
        require_execution(local=local)
        if arm not in ("memory", "no-memory"):
            raise ValueError("Unknown experimental arm")
        if retrieval_strategy not in ("ppr", "dense", "similarity"):
            raise ValueError("Unknown retrieval strategy")
        self.root = Path(root).resolve()
        self.workspace = self.root / "shared-memory"
        self.arm, self.backend, self.model_name, self.retrieval_strategy = arm, embedding_backend, model_name, retrieval_strategy
        self.nodes = {}
        for path in self.workspace.glob("candidates/*/*.json"):
            node = json.loads(path.read_text())
            self.nodes[node["memory_id"]] = node

    def _require(self, identifier, kind):
        node = self.nodes.get(identifier)
        if node is None or node["memory_type"] != kind:
            raise ValueError("Missing typed graph endpoint: " + str(identifier))
        original, provenance = _source(node["source"], self.root)
        if provenance != node["provenance"]:
            raise ValueError("Graph node source has changed")
        if kind in ("Hypothesis", "AnalysisModel", "Decision") and original != node["payload"]:
            raise ValueError("Graph payload differs from source")
        if node.get("judgment_edge_schema") == "scirem-judgment-v1":
            if kind == "AnalysisModel":
                hypothesis = next(e["target"] for e in node["edges"] if e["relation"] == "intended_test_for")
                self._require(hypothesis, "Hypothesis")
                expected = judgment_edges(kind, hypothesis, original, node["source"], provenance)
            else:
                model_id = next(e["target"] for e in node["edges"] if e["relation"] == "produced_by")
                model = self._require(model_id, "AnalysisModel")
                hypothesis = next(e["target"] for e in model["edges"] if e["relation"] == "intended_test_for")
                expected = judgment_edges(kind, hypothesis, original, node["source"], provenance,
                                          statistical_assessment(model["payload"], original))
            actual = [edge for edge in node["edges"] if edge["relation"] in JUDGMENT_RELATIONS]
            if actual != expected:
                raise ValueError("Judgment edges differ from their source-derived meaning")
        return node

    def _publish(self, identifier, kind, payload, source, edges=(), actor=None, judgment=None, **extra):
        _, provenance = _source(source, self.root)
        role = self.OWNERS[kind]
        node = {"memory_id": identifier, "agent_id": actor or role, "actor": actor or role,
                "track_id": "PILOT", "memory_type": kind, "status": "proposed",
                "text": encode(payload), "payload": deepcopy(payload),
                "source": str(Path(source).resolve()), "provenance": provenance,
                "aliases": [], "edges": [{"target": target, "relation": relation,
                    "source": str(Path(source).resolve()), "provenance": provenance}
                    for target, relation in edges], **extra}
        if judgment is not None:
            node["edges"].extend(deepcopy(judgment))
            node["judgment_edge_schema"] = "scirem-judgment-v1"
        target = submit_candidate(self.workspace, node)
        self.nodes[identifier] = json.loads(target.read_text())
        build_snapshot(self.workspace, embedding_backend=self.backend, model_name=self.model_name)
        return identifier

    def ensure_research_question(self, question, source):
        """Publish the immutable trajectory root before any agent stage runs."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A non-empty research question is required")
        identifier = "research-question"
        if identifier in self.nodes:
            node = self._require(identifier, "ResearchQuestion")
            if node["payload"] != {"question": question}:
                raise ValueError("The trajectory research question is immutable")
            return identifier
        return self._publish(identifier, "ResearchQuestion", {"question": question}, source)

    def publish_hypothesis(self, attempt, hypothesis, source, question, config_source,
                           revision_of=None, follows_decision=None):
        allowed = {"hypothesis", "expected_pattern", "rival_explanation", "focal_test",
                   "adaptation", "historical_claims", "required_terms"}
        if not isinstance(hypothesis, dict) or set(hypothesis) - allowed:
            raise ValueError("Hypothesis role cannot publish analysis/results/support")
        if not hypothesis.get("hypothesis") or not isinstance(hypothesis.get("focal_test"), dict):
            raise ValueError("Hypothesis requires a statement and declared test contract")
        original, _ = _source(source, self.root)
        if original != hypothesis:
            raise ValueError("Hypothesis payload differs from immutable source")
        self.ensure_research_question(question, config_source)
        edges = [("research-question", "addresses")]
        if revision_of:
            self._require(revision_of, "Hypothesis")
            edges.append((revision_of, "revision_of"))
        if follows_decision:
            self._require(follows_decision, "Decision")
            edges.append((follows_decision, "follows_decision"))
        return self._publish(f"hypothesis-{attempt:04d}", "Hypothesis", hypothesis, source, edges,
                             result_id=f"result-{attempt:04d}")

    def publish_model(self, attempt, hypothesis_id, plan, source):
        hypothesis = self._require(hypothesis_id, "Hypothesis")["payload"]
        original, _ = _source(source, self.root)
        if original != plan:
            raise ValueError("Analysis plan differs from immutable source")
        for field in ("hypothesis", "expected_pattern", "focal_test", "required_terms"):
            if plan.get(field) != hypothesis.get(field):
                raise ValueError("Analysis role changed hypothesis contract: " + field)
        validate_focal_test(plan)
        _, provenance = _source(source, self.root)
        return self._publish(f"model-{attempt:04d}", "AnalysisModel", plan, source,
                             [(hypothesis_id, "intended_test_for")], result_id=f"result-{attempt:04d}",
                             judgment=judgment_edges("AnalysisModel", hypothesis_id, plan, source, provenance))

    def publish_result(self, attempt, model_id, source, plan_source, *, data_identity,
                       sample_identity=None, sample_recipe=None, actor=None):
        model = self._require(model_id, "AnalysisModel")
        plan, provenance = _source(plan_source, self.root)
        if plan != model["payload"] or provenance != model["provenance"]:
            raise ValueError("Result plan is not the published analysis model")
        tool, _ = _source(source, self.root)
        evidence = result_assessment(f"result-{attempt:04d}", source, self.root,
            statistical_check=statistical_assessment(plan, tool), plan_source=plan_source)
        verify_record(evidence, self.root)
        review = evidence["assessment"].get("scientific_review") or {}
        hypothesis = next(e["target"] for e in model["edges"] if e["relation"] == "intended_test_for")
        _, result_provenance = _source(source, self.root)
        return self._publish(f"result-{attempt:04d}", "ResultAssessment",
            {"analysis_status": "succeeded" if tool["returncode"] == 0 else "failed",
             "agent_assessment_status": "not_yet_provided",
             **({"supervisor_status": review.get("status", "pending"),
                 "supervisor_issues": review.get("issues", []),
                 "supervisor_next_action": review.get("next_action"),
                 "supervisor_pending_work": review.get("pending_work", review.get("blocking_reason"))}
                if supervision_fields(evidence["assessment"]) else {})}, source, [(model_id, "produced_by")],
            evidence=evidence, result_id=f"result-{attempt:04d}",
            data_identity=deepcopy(data_identity), sample_identity=deepcopy(sample_identity),
            sample_recipe=deepcopy(sample_recipe),
            evidence_group=evidence_group_key(evidence, data_identity, sample_identity),
            aliases=[plan["focal_test"]["term"]], actor=actor,
            judgment=judgment_edges("ResultAssessment", hypothesis, tool, source, result_provenance,
                                    evidence["assessment"]["statistical_check"]))

    def publish_decision(self, attempt, hypothesis_id, result_id, decision, source):
        self._require(hypothesis_id, "Hypothesis")
        result = self._require(result_id, "ResultAssessment")
        model_id = next(e["target"] for e in result["edges"] if e["relation"] == "produced_by")
        model = self._require(model_id, "AnalysisModel")
        if not any(e["target"] == hypothesis_id and e["relation"] == "intended_test_for" for e in model["edges"]):
            raise ValueError("Decision result belongs to a different hypothesis")
        allowed = {"action", "basis_keywords", "next_inquiry", "evidence_ids"}
        if not isinstance(decision, dict) or set(decision) - allowed:
            raise ValueError("Decision cannot override machine results")
        if decision.get("action") not in ("maintain", "revise", "abandon", "additional_test"):
            raise ValueError("Unknown hypothesis decision")
        if result_id not in decision.get("evidence_ids", []):
            raise ValueError("Decision must cite its current result")
        for evidence_id in decision["evidence_ids"]:
            self._require(evidence_id, "ResultAssessment")
        original, _ = _source(source, self.root)
        if original != decision:
            raise ValueError("Decision differs from immutable source")
        edges = [(identifier, "based_on") for identifier in decision["evidence_ids"]]
        return self._publish(f"decision-{attempt:04d}", "Decision", decision, source, edges)

    def available_records(self, trace):
        """Resolve only delivered equivalent IDs; keep representative selection unchanged."""
        packet = trace["context_packet"]
        views = packet["evidence"]
        expected_brief = render_tsv(packet) if packet.get("format") == "facts-tsv-v1" else encode(packet)
        if trace["brief"] != expected_brief or trace["selected_ids"] != [v["evidence_id"] for v in views]:
            raise ValueError("Delivered packet differs from retrieval trace")
        if self.arm == "no-memory" and views:
            raise ValueError("Strict control cannot receive historical evidence")
        if packet.get("format") == "facts-tsv-v1":
            if self.arm == "no-memory" and packet["graph_nodes"]:
                raise ValueError("Strict control cannot receive historical graph nodes")
            graph_ids = [n["memory_id"] for n in packet["graph_nodes"]]
            trace_nodes = {n["memory_id"]: n for n in trace["nodes"]}
            if packet.get("selection_policy") == "ranked_seeds_with_scientific_source_closure":
                expected_ids = selected_context_nodes(
                    {key: self.nodes[key] for key in trace_nodes}, trace["upstream_selected_ids"])
                if any(n.get("included") != (key in expected_ids) for key, n in trace_nodes.items()):
                    raise ValueError("Retrieval visualization differs from selected context")
            else:
                expected_ids = set(trace_nodes)
            if len(graph_ids) != len(set(graph_ids)) or set(graph_ids) != expected_ids:
                raise ValueError("TSV graph scope differs from its published snapshot trace")
            delivered_results = [identifier for view in views for identifier in view["observed_in"]]
            expected_results = {key for key in expected_ids if trace_nodes[key]["memory_type"] == "ResultAssessment"}
            if len(delivered_results) != len(set(delivered_results)) or set(delivered_results) != expected_results:
                raise ValueError("TSV does not cover every published executed result exactly once")
            for delivered in packet["graph_nodes"]:
                original = self._require(delivered["memory_id"], delivered["memory_type"])
                for key in ("payload", "source", "provenance", "edges", "actor", "memory_type"):
                    if delivered[key] != original[key]:
                        raise ValueError("Delivered graph node differs from its own source")
                if delivered["memory_type"] == "AnalysisModel":
                    samples = []
                    for result_node in self.nodes.values():
                        if result_node["memory_type"] == "ResultAssessment" and any(
                            e["relation"] == "produced_by" and e["target"] == delivered["memory_id"] for e in result_node["edges"]):
                            sample = result_node.get("sample_identity")
                            if sample not in samples:
                                samples.append(sample)
                    expected_sample = samples[0] if len(samples) == 1 else sorted(samples, key=str) if samples else None
                    if delivered.get("sample_identity") != expected_sample:
                        raise ValueError("Delivered model sample identity differs from executed results")
        records, aliases = {}, []
        for view in views:
            representative = self._require(view["evidence_id"], "ResultAssessment")
            record = representative["evidence"]
            verify_record(record, self.root)
            if view["source"] != record["result"]["source"] or view["analysis"] != record["result"]["analysis"]:
                raise ValueError("Delivered representative provenance or model differs from source")
            if packet.get("format") == "facts-tsv-v1":
                tool, _ = _source(representative["source"], self.root)
                if (view["tool"] != tool or view["metrics"] != record["result"]["metrics"] or
                    view["estimates"] != record["result"]["estimates"] or
                    view["full_statistical_check"] != record["assessment"]["statistical_check"] or
                    supervision_fields(view) != supervision_fields(record["assessment"]) or
                    view.get("sample_identity") != representative.get("sample_identity") or
                    view.get("sample_recipe") != representative.get("sample_recipe") or
                    view.get("data_identity") != representative.get("data_identity")):
                    raise ValueError("TSV scientific fields differ from complete original evidence")
            for delivered in view["estimates"]:
                rows = [r for r in record["result"]["estimates"] if _term(r["term"]) == _term(delivered["term"])]
                if len(rows) != 1 or any(key not in rows[0] or rows[0][key] != value for key, value in delivered.items()):
                    raise ValueError("Delivered numeric fields differ from source")
            group = evidence_group_key(record, representative.get("data_identity"), representative.get("sample_identity"))
            for identifier in dict.fromkeys([view["evidence_id"], *view.get("observed_in", [])]):
                node = self._require(identifier, "ResultAssessment")
                other = node["evidence"]
                verify_record(other, self.root)
                if other["evidence_id"] != identifier or other["result"]["source"] != node["provenance"]:
                    raise ValueError("Delivered ID is not bound to its own result source")
                if identifier != view["evidence_id"]:
                    actual = evidence_group_key(other, node.get("data_identity"), node.get("sample_identity"))
                    if group is None or actual != group or node.get("evidence_group") != group or representative.get("evidence_group") != group:
                        raise ValueError("Delivered alias is not source-verified equivalent evidence")
                    aliases.append(identifier)
                records[identifier] = deepcopy(other)
        # This manifest is trace-only; no extra text can displace numerical context.
        trace["delivered_equivalent_ids"] = list(dict.fromkeys(aliases))
        return records

    def validate_context_claims(self, claims, trace):
        """Verify originals, then restrict factual readback to delivered terms and fields."""
        records = self.available_records(trace)
        outcomes = validate_claims(claims, records, self.root)
        views = trace["context_packet"]["evidence"]
        for claim, outcome in zip(claims, outcomes):
            identifier, field = claim.get("evidence_id"), claim.get("field")
            if identifier not in records or field == "attempt_order":
                continue
            visible = [v for v in views if identifier == v["evidence_id"] or identifier in v.get("observed_in", [])]
            term = _term(claim.get("term", ""))
            rows = [r for v in visible for r in v["estimates"] if _term(r["term"]) == term]
            if field in ("coefficient", "se", "p_value"):
                available = any(field in row for row in rows)
            elif field == "test_status":
                # An omitted row cannot establish whether its test was estimated.
                available = bool(rows) and all(key in rows[0] for key in ("coefficient", "se", "p_value"))
            else:
                available = False
            if not available:
                outcome.update(status="unverifiable", expected=None, reason="term_or_field_not_delivered")
        return outcomes

    def retrieve_tsv(self, query, role="hypothesis", *, source_lookup=False):
        """Deliver ranked evidence and required sources with complete selected result rows."""
        if role not in ("hypothesis", "analysis"):
            raise ValueError("Unknown retrieval role")
        empty = self.arm == "no-memory" or not (self.workspace / "CURRENT").exists()
        empty_reason = "strict_control" if self.arm == "no-memory" else "no_history"
        raw = ({"request_id": "r-" + uuid.uuid4().hex, "query": query, "track_id": "PILOT",
                "snapshot_id": None, "backend": self.backend, "strategy": self.retrieval_strategy,
                "retrieval_status": "not_run_" + empty_reason, "outcome": "no-match",
                "nodes": [], "edges": [], "anchors": [], "selected_ids": [], "ppr": {},
                "iterations": 0, "converged": True}
               if empty else retrieve(self.workspace, query, "PILOT", top_k=20, strategy=self.retrieval_strategy))
        nodes = {n["memory_id"]: n for n in raw["nodes"]}
        requested_ids = (list(dict.fromkeys(re.findall(
            r"(?<![\w-])(?:hypothesis|model|result|decision)-\d{4,}(?![\w-])", query)))
            if source_lookup else [])
        # An explicit source lookup is not a request for its whole PPR neighborhood.
        ranked_ids = list(raw["selected_ids"])
        upstream = [key for key in requested_ids if key in nodes] if requested_ids else ranked_ids
        missing_ids = [key for key in requested_ids if key not in nodes]
        included = selected_context_nodes({key: self.nodes[key] for key in nodes}, upstream)
        ranked = sorted((nodes[key] for key in included),
                        key=lambda n: (-n.get("final_score", 0), n["memory_id"]))
        groups, sample_by_model = {}, {}
        for event in ranked:
            identifier = event.get("result_id", event["memory_id"])
            node = nodes.get(identifier, {})
            if node.get("memory_type") != "ResultAssessment":
                continue
            original = self._require(identifier, "ResultAssessment")
            record = original["evidence"]
            verify_record(record, self.root)
            key = evidence_group_key(record, original.get("data_identity"), original.get("sample_identity"))
            if key != original.get("evidence_group"):
                raise ValueError("Stored evidence group differs from original scientific identity")
            key = key or identifier
            if key not in groups:
                tool, _ = _source(original["source"], self.root)
                groups[key] = {"evidence_id": identifier, "observed_in": [],
                    "source": record["result"]["source"], "analysis": deepcopy(record["result"]["analysis"]),
                    "returncode": record["result"]["returncode"], "metrics": deepcopy(record["result"]["metrics"]),
                    "estimates": deepcopy(record["result"]["estimates"]), "omitted_estimate_terms": [],
                    "statistical_check": statistical_summary(record["assessment"]["statistical_check"]),
                    "full_statistical_check": deepcopy(record["assessment"]["statistical_check"]),
                    **supervision_fields(record["assessment"]),
                    "tool": tool, "data_identity": deepcopy(original.get("data_identity")),
                    "sample_identity": original.get("sample_identity"),
                    "sample_recipe": deepcopy(original.get("sample_recipe")),
                    "result_facts_available": True, "agent_prose_included": False}
            if identifier not in groups[key]["observed_in"]:
                groups[key]["observed_in"].append(identifier)
            for edge in original["edges"]:
                if edge["relation"] == "produced_by":
                    samples = sample_by_model.setdefault(edge["target"], [])
                    if original.get("sample_identity") not in samples:
                        samples.append(original.get("sample_identity"))
        graph_nodes = []
        for ranked_node in ranked:
            original = self._require(ranked_node["memory_id"], ranked_node["memory_type"])
            for edge in original["edges"]:
                if edge["target"] not in nodes:
                    raise ValueError("Published graph is not closed over its typed endpoints")
            node = deepcopy(original)
            if node["memory_type"] == "AnalysisModel":
                samples = sample_by_model.get(node["memory_id"], [])
                node["sample_identity"] = samples[0] if len(samples) == 1 else sorted(samples, key=str) if samples else None
            graph_nodes.append(node)
        views = list(groups.values())
        for view in views:
            view["observed_in"].sort()
        packet = {"format": "facts-tsv-v1", "evidence": views, "graph_nodes": graph_nodes,
            "selection_policy": "ranked_seeds_with_scientific_source_closure",
            "result_facts_available": bool(views), "agent_prose_included": False,
            "agent_prose_scope": "previous ResultAssessment interpretation only",
            "hypothesis_decision_text_included": bool(graph_nodes),
            "context_scope": "up to 20 ranked seeds plus required scientific source endpoints and fixed RQ; complete selected results; revision-lineage references may point outside this packet; absent evidence is unknown",
            "omitted_evidence_ids": [], "reason": "strict_control" if self.arm == "no-memory" else "retrieved_own_context_top_k_20"}
        if requested_ids and self.arm != "no-memory":
            packet.update(context_scope="Explicitly requested IDs plus required scientific source endpoints and fixed RQ; complete selected results; no unrelated ranked neighbors; absent evidence is unknown",
                          reason="requested_source_ids", requested_ids=requested_ids, missing_ids=missing_ids)
        raw.pop("context_char_budget", None)  # Legacy prose budget does not constrain TSV.
        raw.update(upstream_ranked_ids=ranked_ids, upstream_selected_ids=upstream,
            requested_ids=requested_ids, missing_ids=missing_ids,
            selected_ids=[v["evidence_id"] for v in views],
            context_packet=packet, brief=render_tsv(packet), logical_view=role,
            context_scope=packet["context_scope"],
            context_exclusions=[{"memory_id": key, "reason": "outside_ranked_source_closure"}
                                for key in sorted(set(nodes) - included)],
            context_manifest={"source_node_ids": [n["memory_id"] for n in graph_nodes],
                "executed_result_ids": sorted({i for v in views for i in v["observed_in"]}),
                "representative_ids": [v["evidence_id"] for v in views],
                "coefficient_row_count": sum(len(v["estimates"]) for v in views),
                "metric_row_count": sum(len(v["metrics"]) for v in views),
                "omitted_evidence_ids": [], "omitted_coefficient_count": 0})
        raw["context_manifest"]["rendered_chars"] = len(raw["brief"])
        if requested_ids and not empty:
            raw["outcome"] = "retrieved" if upstream else "no-match"
        for node in raw["nodes"]:
            node["included"] = node["memory_id"] in included
            node["numeric_representative_included"] = node["memory_id"] in raw["selected_ids"]
            if requested_ids:
                node["selection_reason"] = ("explicit source lookup" if node["memory_id"] in upstream else
                    "required scientific source endpoint" if node["included"] else "outside requested source closure")
        self.available_records(raw)
        return raw, {v["evidence_id"]: deepcopy(self.nodes[v["evidence_id"]]["evidence"]) for v in views}

    def retrieve(self, query, role="hypothesis", budget=3200):
        if role not in ("hypothesis", "analysis"):
            raise ValueError("Unknown retrieval role")
        if self.arm == "no-memory" or not (self.workspace / "CURRENT").exists():
            reason = "strict_control" if self.arm == "no-memory" else "no_history"
            packet = {"format": "facts-v8", "evidence": [], "result_facts_available": False,
                      "agent_prose_included": False, "reason": reason}
            return {"request_id": "r-" + uuid.uuid4().hex, "query": query, "track_id": "PILOT",
                    "snapshot_id": None, "backend": self.backend, "strategy": self.retrieval_strategy,
                    "retrieval_status": "not_run_" + reason, "outcome": "no-match",
                    "nodes": [], "edges": [], "anchors": [], "selected_ids": [],
                    "iterations": 0, "converged": True,
                    "context_packet": packet, "brief": encode(packet)}, {}
        raw = retrieve(self.workspace, query, "PILOT", top_k=20, strategy=self.retrieval_strategy)
        nodes = {n["memory_id"]: n for n in raw["nodes"]}
        raw["upstream_selected_ids"] = list(raw["selected_ids"])
        ranked = sorted((n for n in raw["nodes"] if n.get("final_score", 0) > 0),
                        key=lambda n: (-n["final_score"], n["memory_id"]))
        groups, selected = {}, []
        candidate_groups, verified_results = set(), set()
        for event in ranked:
            key = event["memory_id"]
            result_id = event.get("result_id", key)
            node = nodes.get(result_id, {})
            if node.get("memory_type") != "ResultAssessment" or "evidence" not in node:
                continue
            if result_id not in verified_results:
                verify_record(node["evidence"], self.root)
                verified_results.add(result_id)
            group = node.get("evidence_group") or result_id
            candidate_groups.add(group)
            # Apply the same finite cap to distinct evidence, not repeated events.
            if group not in groups and len(groups) < 20:
                groups[group] = {"representative": result_id, "observed_in": []}
                selected.append(result_id)
        raw.update(candidate_event_count=len(ranked), candidate_result_count=len(verified_results),
                   candidate_group_count=len(candidate_groups), selected_group_count=len(groups),
                   candidate_group_limit=20)
        # Preserve every equivalent event in the trace, not only top-k ranking hits.
        for identifier, node in nodes.items():
            if "evidence" not in node:
                continue
            group = node.get("evidence_group") or identifier
            if group in groups:
                verify_record(node["evidence"], self.root)
                groups[group]["observed_in"].append(identifier)
        raw["selected_ids"] = selected
        # Reserve bounded role metadata before packing extra coefficient rows.
        numeric_budget = max(128, budget - min(1400, budget // 2))
        trace, supplied = ground_trace(raw, self.root, query, budget=numeric_budget)
        packet = trace["context_packet"]
        packet.update(format="facts-v8", result_facts_available=bool(supplied), agent_prose_included=False)
        packet["policy"] = "Numbers available; prior interpretation prose excluded. O/X are statistical only."
        for view in packet["evidence"]:
            group = nodes[view["evidence_id"]].get("evidence_group") or view["evidence_id"]
            observed = sorted(set(groups[group]["observed_in"]))
            sources = [{"evidence_id": identifier, "source": nodes[identifier]["evidence"]["result"]["source"]}
                       for identifier in observed]
            view["observed_in"] = observed if len(observed) <= 4 else observed[:1] + observed[-3:]
            view["omitted_observation_count"] = len(observed) - len(view["observed_in"])
            view["omitted_source_count"] = len(sources) - 1
            hypotheses, decisions = {}, []
            for identifier in observed:
                model_id = next(e["target"] for e in self.nodes[identifier]["edges"] if e["relation"] == "produced_by")
                model = self._require(model_id, "AnalysisModel")
                hypothesis_id = next(e["target"] for e in model["edges"] if e["relation"] == "intended_test_for")
                hypothesis_node = self._require(hypothesis_id, "Hypothesis")
                hypothesis = hypothesis_node["payload"]
                hypotheses[hypothesis_id] = {"hypothesis_id": hypothesis_id,
                    "hypothesis": hypothesis["hypothesis"], "focal_test": hypothesis["focal_test"],
                    "source": hypothesis_node["provenance"]}
                for decision in self.nodes.values():
                    if decision["memory_type"] == "Decision" and any(e["target"] == identifier for e in decision["edges"]):
                        self._require(decision["memory_id"], "Decision")
                        decisions.append({"decision_id": decision["memory_id"],
                                          "source": decision["provenance"], **decision["payload"]})
            decisions = list({decision["decision_id"]: decision for decision in decisions}.values())
            decisions.sort(key=lambda decision: decision["decision_id"])
            all_hypotheses = list(hypotheses.values())
            first = all_hypotheses[0]
            distinct = [h for h in all_hypotheses[1:]
                        if (h["hypothesis"], h["focal_test"]) != (first["hypothesis"], first["focal_test"])]
            compact_hypotheses = [first] + distinct[-1:]
            latest_decision = decisions[-1:]
            groups[group].update(sources=sources, hypotheses=all_hypotheses, decisions=decisions)
            if role == "hypothesis":
                view["hypotheses"] = compact_hypotheses
                view["decisions"] = latest_decision
            else:
                view["hypothesis_ids"] = [h["hypothesis_id"] for h in compact_hypotheses]
                view["decision_ids"] = [d["decision_id"] for d in latest_decision]
            view["omitted_hypothesis_count"] = len(all_hypotheses) - len(compact_hypotheses)
            view["omitted_decision_count"] = len(decisions) - len(latest_decision)
            view["result_facts_available"] = True
            view["agent_prose_included"] = False
        while len(encode(packet)) > budget and packet["evidence"]:
            removed = packet["evidence"].pop()
            supplied.pop(removed["evidence_id"], None)
            packet.setdefault("omitted_evidence_ids", []).append(removed["evidence_id"])
        if len(encode(packet)) > budget:
            packet = {"format": "facts-v8", "evidence": [], "result_facts_available": False,
                      "agent_prose_included": False, "reason": "no_complete_record_fits"}
            supplied = {}
        packet["result_facts_available"] = bool(supplied)
        trace.update(context_packet=packet, brief=encode(packet), selected_ids=list(supplied),
                     logical_view=role, context_char_budget=budget, equivalence_groups=groups,
                     context_exclusions=packet.get("omitted_evidence_ids", []))
        for node in trace["nodes"]:
            node["included"] = node["memory_id"] in supplied
        return trace, supplied
