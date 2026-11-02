"""Regression tests for source-grounded diagnostic repair; Scheduler only."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from agent_r_diagnostic import DiagnosticOutputError
from four_role_discovery import RoleHistory, execute_diagnostic_work, execute_four_role_attempt
from private_inference import RuntimeIsolationError
from test_four_role import hypothesis, plan, result
from test_scientific_supervision import reviewed
from test_decision_review import consistent_review
from two_role_discovery import QUESTION


class DiagnosticHandoffTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get('JOB_ID'):
            raise RuntimeError('Reviewed Scheduler allocation required')
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root/'config.json').write_text(json.dumps({'question': QUESTION}))
        self.session = self.root/'attempt-0001'
        self.session.mkdir()
        self.history = RoleHistory(self.root)
        self.record = {}

    def tearDown(self):
        self.temporary.cleanup()

    def failure(self, code):
        return {'status': 'failed', 'returncode': 1, 'code': code,
                'error': {'category': 'combinatorial_enumeration_overflow',
                          'message': 'combn exhausted allocation capacity',
                          'repair_guidance': 'Use an appropriate package-supported scoring method.'}}

    def run_work(self, code, runner, model):
        return execute_diagnostic_work(self.session, code, runner, model, None, self.history, self.record,
            contract=hypothesis(), inquiry='Compare the two models on held-out choice sets.',
            shared_memory_enabled=False)

    def test_failed_code_and_typed_error_reach_changed_code_execution(self):
        failed_code = 'combn(seq_len(n), k, simplify=FALSE)'
        repaired_code = 'package_supported_scoring()'
        calls = []
        def runner(folder, code):
            calls.append((folder, code))
            return self.failure(code) if len(calls) == 1 else {'status': 'completed', 'metrics': {'gain': 0.01}}
        def model(stage, prompt, schema, tokens, seed):
            self.assertEqual(stage, 'empiricist_diagnostic_repair')
            self.assertIn(failed_code, prompt)
            self.assertIn('combinatorial_enumeration_overflow', prompt)
            self.assertIn('held-out choice sets', prompt)
            return {'action': 'repair', 'diagnostic_r_code': repaired_code, 'reason': 'Avoid enumerating combinations.'}
        receipt = self.run_work(failed_code, runner, model)
        self.assertEqual([code for _, code in calls], [failed_code, repaired_code])
        self.assertNotEqual(calls[0][0], calls[1][0])
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['code'], repaired_code)
        self.assertEqual([item['status'] for item in self.record['diagnostic_executions']], ['failed', 'completed'])
        original = json.loads((self.session/'diagnostic-receipt.json').read_text())
        self.assertEqual(original['code'], failed_code)
        self.assertEqual(original['status'], 'failed')
        self.assertIn(failed_code, json.dumps(self.history._items('empiricist')))
        self.assertIn(repaired_code, json.dumps(self.history._items('empiricist')))
        source = self.root/receipt['receipt_source']['path']
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), receipt['receipt_source']['sha256'])

    def test_identical_failed_code_is_not_executed_again(self):
        runner = Mock(return_value=self.failure('same code'))
        model = Mock(return_value={'action': 'repair', 'diagnostic_r_code': 'same code', 'reason': 'Retry.'})
        receipt = self.run_work('same code', runner, model)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(receipt['status'], 'blocked')
        self.assertEqual(receipt['repair_status'], 'unchanged_failed_code')
        self.assertFalse(receipt['scientific_test_completed'])

    def test_changed_code_with_same_error_returns_to_supervisor(self):
        runner = Mock(side_effect=lambda folder, code: self.failure(code))
        model = Mock(return_value={'action': 'repair', 'diagnostic_r_code': 'different code', 'reason': 'Changed formatting.'})
        receipt = self.run_work('first code', runner, model)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(receipt['repair_status'], 'same_failure_after_repair')
        self.assertEqual(receipt['status'], 'blocked')

    def test_no_error_detail_does_not_request_blind_llm_repair(self):
        runner = Mock(return_value={'status': 'failed', 'error': {
            'category': 'unclassified_execution_error', 'repairable_from_receipt': False}})
        model = Mock()
        receipt = self.run_work('first code', runner, model)
        model.assert_not_called()
        self.assertEqual(receipt['repair_status'], 'error_inspection_required')

    def test_repaired_source_paths_are_bound_to_their_actual_execution_folder(self):
        calls = []
        def runner(folder, code):
            calls.append(code)
            source = folder/'agent-r-diagnostic'
            source.mkdir()
            (source/'diagnostic.R').write_text(code)
            (source/'stderr.log').write_text('known failure' if len(calls) == 1 else '')
            receipt = self.failure(code) if len(calls) == 1 else {'status': 'completed', 'metrics': {'gain': 0.1}}
            for field, name in [('source', 'diagnostic.R'), ('stderr_source', 'stderr.log')]:
                artifact = source/name
                receipt[field] = {'path': str(artifact.relative_to(folder)),
                                  'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
            return receipt
        model = Mock(return_value={'action': 'repair', 'diagnostic_r_code': 'fixed code', 'reason': 'Fixed cause.'})
        receipt = self.run_work('original failure', runner, model)
        self.assertEqual(receipt['source']['path_scope'], 'trajectory')
        source = self.root/receipt['source']['path']
        self.assertIn('diagnostic-repair-0001', receipt['source']['path'])
        self.assertEqual(source.read_text(), 'fixed code')

    def test_explicit_unavailable_is_recorded_without_fake_metric(self):
        runner = Mock(return_value=self.failure('failed code'))
        model = Mock(return_value={'action': 'unavailable', 'diagnostic_r_code': None,
                                   'reason': 'Required comparison capability is unavailable.'})
        receipt = self.run_work('failed code', runner, model)
        self.assertEqual(receipt['status'], 'blocked')
        self.assertNotIn('metrics', receipt)
        self.assertEqual(runner.call_count, 1)
        self.assertIn('Required comparison', receipt['blocking_reason'])

    def test_successful_exit_with_all_na_metrics_is_not_a_completed_comparison(self):
        runner = Mock(return_value={'status': 'completed', 'returncode': 0,
                                    'metrics': {'gain': None, 'ci_low': None, 'ci_high': None}})
        def model(stage, prompt, schema, tokens, seed):
            self.assertIn('no_numeric_result', prompt)
            self.assertIn('never replace NA with a fabricated zero', prompt)
            return {'action': 'unavailable', 'diagnostic_r_code': None,
                    'reason': 'No valid uncertainty estimate has been computed.'}
        receipt = self.run_work('write.csv(all_NA)', runner, model)
        self.assertEqual(receipt['status'], 'blocked')
        self.assertEqual(receipt['execution_status'], 'completed')
        self.assertEqual(receipt['returncode'], 0)
        self.assertEqual(receipt['error']['category'], 'no_numeric_result')
        self.assertTrue(all(value is None for value in receipt['metrics'].values()))
        self.assertEqual(self.record['diagnostic_executions'][0]['status'], 'unavailable')

    def test_output_contract_error_preserves_its_typed_receipt(self):
        failure = {'status': 'failed', 'returncode': 0, 'error': {'category': 'invalid_output_contract',
                    'message': 'Output needs exactly metric,value columns'}}
        runner = Mock(side_effect=DiagnosticOutputError('Output needs exactly metric,value columns', failure))
        def model(stage, prompt, schema, tokens, seed):
            self.assertIn('invalid_output_contract', prompt)
            self.assertIn('metric,value columns', prompt)
            return {'action': 'unavailable', 'diagnostic_r_code': None, 'reason': 'Explicitly unresolved.'}
        receipt = self.run_work('write.csv()', runner, model)
        self.assertEqual(receipt['error']['category'], 'invalid_output_contract')
        self.assertEqual(receipt['returncode'], 0)
        self.assertEqual(receipt['status'], 'blocked')

    def test_isolation_or_deadline_error_is_not_an_inference_retry(self):
        runner = Mock(return_value=self.failure('failure'))
        model = Mock(side_effect=RuntimeIsolationError('allocation ended'))
        with self.assertRaises(RuntimeIsolationError):
            self.run_work('failure', runner, model)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(model.call_count, 1)
        self.assertTrue((self.session/'diagnostic-receipt.json').exists())

    def test_primary_success_does_not_mask_failed_comparison(self):
        failed_code = 'combn(seq_len(n), k)'
        def model(stage, prompt, schema, tokens, seed):
            if stage == 'hypothesis': return hypothesis()
            if stage == 'empiricist_plan': return {**plan(), 'diagnostic_r_code': failed_code}
            if stage == 'empiricist_diagnostic_repair':
                return {'action': 'unavailable', 'diagnostic_r_code': None, 'reason': 'Requested diagnostic remains unresolved.'}
            if stage == 'empiricist_assessment':
                self.assertIn('CURRENT_DIAGNOSTIC_EXECUTION', prompt)
                self.assertIn('combinatorial_enumeration_overflow', prompt)
                self.assertIn(failed_code, prompt)
                return {'execution_assessment': 'Primary fit is available; requested comparison failed.', 'diagnostic_action': 'design_problem'}
            if stage == 'orchestrator_scientific_review': return reviewed()
            if stage == 'orchestrator_decision_review': return consistent_review()
            if stage == 'hypothesis_decision':
                return {'action': 'additional_test', 'basis_keywords': 'Failed diagnostic is unresolved.',
                        'next_inquiry': 'Resolve the blocked comparison.', 'evidence_ids': ['result-0001']}
            raise AssertionError(stage)
        record, _, _ = execute_four_role_attempt(self.session, 1, None, model,
            lambda _, action: result(action), data_identity='fixture', condition='B',
            diagnostic_runner=lambda folder, code: self.failure(code))
        self.assertEqual(record['status'], 'partial', record)
        self.assertEqual(record['analysis_status'], 'succeeded')
        self.assertEqual(record['research_task_status'], 'unfulfilled')
        facts = json.loads((self.session/'current-result-facts.json').read_text())
        self.assertEqual(facts['agent_r_diagnostic']['status'], 'blocked')
        self.assertEqual(facts['estimates'][0]['p_value'], '.192')
        original_plan = next(item for item in self.history._items('empiricist') if item['stage'] == 'plan')
        self.assertEqual(original_plan['value']['diagnostic_r_code'], failed_code)
        self.assertTrue((self.root/original_plan['value']['proposal_source']['path']).exists())

    def test_omitted_comparison_code_is_not_labeled_diagnostic_execution(self):
        def model(stage, prompt, schema, tokens, seed):
            if stage == 'hypothesis': return hypothesis()
            if stage == 'empiricist_plan':
                self.assertIn('This is the planning stage', prompt)
                self.assertIn('does not mean the reviewed execution host is unavailable', prompt)
                return {**plan(), 'diagnostic_r_code': None}
            if stage == 'empiricist_assessment':
                return {'execution_assessment': 'Requested comparison was not executed.', 'diagnostic_action': 'design_problem'}
            if stage == 'orchestrator_scientific_review': return reviewed()
            if stage == 'orchestrator_decision_review': return consistent_review()
            if stage == 'empiricist_supervised_repair':
                return {'action': 'unavailable', 'diagnostic_r_code': None, 'reason': 'Required comparison design remains unspecified.'}
            if stage == 'hypothesis_decision':
                return {'action': 'additional_test', 'basis_keywords': 'Comparison is not performed.',
                        'next_inquiry': 'Execute the comparison.', 'evidence_ids': ['result-0001']}
            raise AssertionError(stage)
        diagnostic = Mock()
        record, _, _ = execute_four_role_attempt(self.session, 1, None, model,
            lambda _, action: result(action), data_identity='fixture', condition='B', diagnostic_runner=diagnostic)
        diagnostic.assert_not_called()
        self.assertEqual(record['status'], 'partial', record)
        self.assertEqual(record['research_task_status'], 'unfulfilled')
        self.assertEqual(record['assessment_followup'], 'design_problem')
        self.assertNotIn('diagnostic_executions', record)

    def test_primary_receipt_survives_allocation_end_during_diagnostic_repair(self):
        def model(stage, prompt, schema, tokens, seed):
            if stage == 'hypothesis': return hypothesis()
            if stage == 'empiricist_plan': return {**plan(), 'diagnostic_r_code': 'failed code'}
            if stage == 'empiricist_diagnostic_repair':
                error = RuntimeIsolationError('Allocation exhausted during repair.')
                error.deadline_exhausted = True
                raise error
            raise AssertionError(stage)
        record, _, _ = execute_four_role_attempt(self.session, 1, None, model,
            lambda _, action: result(action), data_identity='fixture', condition='B',
            diagnostic_runner=lambda folder, code: self.failure(code))
        self.assertEqual(record['status'], 'censored', record)
        self.assertEqual(record['analysis_status'], 'succeeded')
        self.assertTrue((self.session/'primary-tool-result.json').exists())
        self.assertEqual(json.loads((self.session/'diagnostic-receipt.json').read_text())['status'], 'failed')
