"""Role/context ablations; execute under the Scheduler with the connected suite."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import four_role_discovery as flow
from test_four_role import hypothesis, plan, result, answer_envelope_adapter
from research_memory.two_role import TwoRoleMemory
from two_role_discovery import QUESTION


class RoleLayoutTests(unittest.TestCase):
    def test_unified_history_and_recall_share_scientific_work_but_not_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(flow, 'RESEARCH_ROLE_LAYOUT', 'unified'), patch.object(flow, 'ADAPTIVE_SHARED_CONTEXT', True):
            history = flow.RoleHistory(tmp)
            history.add('hypothesis', 'hypothesis', {'text': 'HYPOTHESIS_MARKER'})
            history.add('empiricist', 'plan', {'text': 'PLAN_MARKER'})
            history.add('orchestrator', 'review', {'text': 'PRIVATE_REVIEWER_MARKER'})
            prompt = history.prompt('hypothesis', 'Role: Hypothesis agent.\nCurrent request.', shared_memory=True)
            self.assertIn('PLAN_MARKER', prompt)
            self.assertIn('HYPOTHESIS_MARKER', prompt)
            self.assertNotIn('PRIVATE_REVIEWER_MARKER', prompt)
            self.assertIn('Role: Researcher.', prompt)
            self.assertEqual(history.recall('empiricist', 'private-turn:1:0')['role'], 'researcher')
            self.assertIn('HYPOTHESIS_MARKER', history.recall('empiricist', 'private-turn:1:0')['content'])
            history.add('hypothesis', 'decision', {'evidence_ids': ['result-0001']})
            self.assertEqual(history.prior_result_ids(), {'result-0001'})

    def test_split_history_does_not_acquire_other_researcher_private_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = flow.RoleHistory(tmp)
            history.add('empiricist', 'plan', {'text': 'PLAN_MARKER'})
            self.assertNotIn('PLAN_MARKER', history.prompt('hypothesis', 'current', shared_memory=True))

    def test_review_off_keeps_real_memory_numerical_gate_and_self_assessment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(flow, 'RESEARCH_ROLE_LAYOUT', 'unified'), patch.object(flow, 'ORCHESTRATOR_ASSESSMENT', False), patch.object(flow, 'ADAPTIVE_SHARED_CONTEXT', True):
            root = Path(tmp)
            (root/'config.json').write_text(json.dumps({'question': QUESTION}))
            session = root/'attempt-0001'; session.mkdir()
            memory = TwoRoleMemory(root, embedding_backend='lexical', retrieval_strategy='ppr')
            calls = []
            def model(stage, prompt, schema, tokens, seed):
                calls.append(stage)
                self.assertFalse(stage.startswith('orchestrator_'), stage)
                if stage == 'hypothesis': return hypothesis()
                if stage == 'empiricist_plan': return {**plan(), 'diagnostic_r_code': None}
                if stage == 'empiricist_assessment':
                    return {'execution_assessment': 'Primary association only.', 'diagnostic_action': 'adequate_for_interpretation'}
                if stage == 'hypothesis_decision':
                    return {'action': 'maintain', 'basis_keywords': 'Primary not supported.', 'next_inquiry': 'Assess robustness.', 'evidence_ids': ['result-0001']}
                raise AssertionError(stage)
            record, _, _ = flow.execute_four_role_attempt(session, 1, memory, answer_envelope_adapter(model),
                lambda folder, action: result(action), data_identity='fixture', condition='C')
            self.assertEqual(record['status'], 'completed', record)
            self.assertEqual(calls, ['hypothesis', 'empiricist_plan', 'empiricist_assessment', 'hypothesis_decision'])
            self.assertEqual(record['statistical_assessment']['support'], 'not_supported')
            self.assertFalse(record['orchestrator_assessment_enabled'])
            self.assertEqual(record['scientific_validity'], 'unverified')
            self.assertEqual(record['actor_instances']['empiricist-agent'], 'researcher')
            self.assertTrue((session/'tool-result.json').exists())
            self.assertFalse((session/'scientific-review.json').exists())
            self.assertFalse((session/'decision-review.json').exists())
