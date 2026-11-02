"""Small Scheduler-owned handoff ledger for the four-role research loop."""
from copy import deepcopy
import json
from pathlib import Path
import time

from .evidence import _source


class HandoffError(ValueError):
    """A role tried to bypass the reviewed research-loop order."""


class FourRoleOrchestrator:
    """Enforce Orchestrator → Hypothesis → Empiricist → Runner → handback."""

    STEPS = (("hypothesis", "hypothesis-agent"), ("empiricist_plan", "empiricist-agent"),
             ("runner", "runner"), ("empiricist_assessment", "empiricist-agent"),
             ("decision", "hypothesis-agent"))

    def __init__(self, root, attempt, *, research_question, inquiry):
        self.root, self.attempt, self.cursor, self.events = Path(root), attempt, 0, []
        if not research_question.strip() or not inquiry.strip():
            raise HandoffError("Research question and inquiry are required")
        self.research_question, self.inquiry = research_question, inquiry
        self.started = time.monotonic()

    def validate_receipt(self, stage, artifact):
        sources = []

        def read(name):
            value, source = _source(self.root / name, self.root)
            sources.append(source)
            return value

        try:
            # Earlier handoffs must still refer to the same immutable evidence.
            for event in self.events:
                for source in event["sources"]:
                    if _source(self.root / source["path"], self.root)[1] != source:
                        raise HandoffError("Previously accepted handoff source changed")
            if stage == "hypothesis":
                expected = {"state": "initial", "research_question": self.research_question,
                            "current_inquiry": self.inquiry}
            elif stage == "empiricist_plan":
                expected = {"hypothesis_id": f"hypothesis-{self.attempt:04d}",
                            "hypothesis": read("hypothesis.json")}
            elif stage == "runner":
                plan = read("plan.json")
                expected = {"model_id": f"model-{self.attempt:04d}", "analysis": plan["analysis"],
                            "model_specification": plan["model_specification"]}
            else:
                facts = read("current-result-facts.json")
                result_id = f"result-{self.attempt:04d}"
                if facts["evidence_id"] != result_id:
                    raise HandoffError("Result receipt belongs to another attempt")
                expected = {"result_id": result_id, "result_facts": facts}
                if stage == "empiricist_assessment":
                    expected["analysis_status"] = facts["analysis_status"]
                    read("tool-result.json")
                    read("statistical-assessment.json")
                else:
                    expected["empiricist_assessment"] = read("empiricist-assessment.json")
            if artifact != expected:
                raise HandoffError("%s handoff does not match its current source receipt" % stage)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HandoffError("%s receipt validation failed: %s" % (stage, error)) from error
        return sources

    def route(self, stage, actor, artifact=None):
        if self.cursor >= len(self.STEPS):
            raise HandoffError("Attempt has already completed")
        expected_stage, expected_actor = self.STEPS[self.cursor]
        if (stage, actor) != (expected_stage, expected_actor):
            raise HandoffError("Expected %s from %s" % (expected_stage, expected_actor))
        sources = self.validate_receipt(stage, artifact)
        event = {"attempt": self.attempt, "actor": actor, "stage": stage, "artifact": deepcopy(artifact),
                 "sources": sources, "elapsed_seconds": round(time.monotonic() - self.started, 6)}
        self.events.append(event)
        self.cursor += 1
        return event

    def complete(self):
        if self.cursor != len(self.STEPS):
            raise HandoffError("Cannot complete before every role handoff")
        return self.snapshot("completed")

    def snapshot(self, status):
        return {"attempt": self.attempt, "status": status, "routing_mode": "deterministic",
                "research_question": self.research_question, "current_inquiry": self.inquiry,
                "events": deepcopy(self.events)}

    def write(self, path, status="completed"):
        target = Path(path)
        receipt = self.complete() if status == "completed" else self.snapshot(status)
        pending = target.with_suffix(".pending.json")
        pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        pending.replace(target)
        return target
