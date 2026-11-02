"""Structured R failure feedback; run only within a reviewed Scheduler allocation."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import agent_r_diagnostic as module


class DiagnosticFeedbackTests(unittest.TestCase):
    def test_missing_logical_coefficient_has_actionable_source_grounded_feedback(self):
        code='e <- research_estimate(fit, "distinctive:subsample_q25")'
        error=module.execution_error('Error in research_estimate(fit, "distinctive:subsample_q25") : \n  Requested coefficient is absent from the fitted model\nPRIVATE_ROW_SENTINEL',code,1)
        self.assertEqual(error['category'],'coefficient_name_mismatch')
        self.assertIn('TRUE',error['repair_guidance'])
        self.assertNotIn('PRIVATE_ROW_SENTINEL',json.dumps(error))

    def test_absorbed_main_effect_is_not_silently_filled(self):
        code='e <- research_estimate(fit, "subsample_1m")'
        error=module.execution_error('Error in research_estimate(fit, "subsample_1m") : Requested coefficient or covariance is not estimable',code,1)
        self.assertEqual(error['category'],'coefficient_not_estimable')
        self.assertIn('never replace',error['repair_guidance'])

    def test_real_r_deparse_quote_change_preserves_source_match(self):
        code = "e <- research_estimate(fit, 'distinctive:subsample_q25')"
        stderr = 'Error in research_estimate(fit, "distinctive:subsample_q25") : \n  Requested coefficient is absent from the fitted model\n'
        error = module.execution_error(stderr, code, 1)
        self.assertEqual(error['category'], 'coefficient_name_mismatch')
        self.assertEqual(error['source_lines'], [1])
        mismatch = module.execution_error(stderr.replace('subsample_q25', 'PRIVATE_ROW_SENTINEL'), code, 1)
        self.assertEqual(mismatch['category'], 'unclassified_execution_error')
        self.assertNotIn('PRIVATE_ROW_SENTINEL', json.dumps(mismatch))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.code = 'library(survival)\n# agent-authored diagnostic\n'

    def tearDown(self):
        self.tmp.cleanup()

    def run_fixture(self, stderr='', output=None, returncode=1):
        def run(command, **kwargs):
            kwargs['stderr'].write(stderr)
            if output is not None:
                (Path(kwargs['stdout'].name).parent / 'diagnostics.csv').write_text(output)
            return SimpleNamespace(returncode=returncode)
        with patch.dict(os.environ, {'JOB_ID': 'synthetic-test'}), \
                patch.object(module, 'sandbox', return_value=['sandbox', '--']), \
                patch.object(module.subprocess, 'run', side_effect=run):
            return module.run_agent_r(self.root, self.code, data=self.root / 'unused.csv')

    def test_combinatorial_failure_keeps_code_and_actionable_safe_error(self):
        stderr = ('PRIVATE_ROW_SENTINEL\n' * 400 +
                  'Error in vector("list", count) : vector size cannot be NA\n'
                  'Calls: data.frame -> condlog -> combn\n'
                  'NAs introduced by coercion to integer range\nExecution halted\n')
        receipt = self.run_fixture(stderr)
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['error']['category'], 'combinatorial_enumeration_overflow')
        self.assertEqual(receipt['error']['stage'], 'execution')
        self.assertIn('enumerat', receipt['error']['repair_guidance'])
        self.assertEqual(receipt['code'], self.code)
        self.assertEqual(receipt['code_sha256'], hashlib.sha256(self.code.encode()).hexdigest())
        self.assertEqual(receipt['source']['sha256'], receipt['code_sha256'])
        self.assertEqual(receipt['stderr_source']['sha256'], hashlib.sha256(stderr.encode()).hexdigest())
        self.assertNotIn('PRIVATE_ROW_SENTINEL', json.dumps(receipt))
        self.assertEqual((self.root / 'agent-r-diagnostic/stderr.log').read_text(), stderr)

    def test_missing_package_does_not_echo_untrusted_package_name(self):
        receipt = self.run_fixture("Error in library(x): there is no package called 'PRIVATE_ROW_SENTINEL'\n")
        self.assertEqual(receipt['error']['category'], 'package_unavailable')
        self.assertNotIn('PRIVATE_ROW_SENTINEL', json.dumps(receipt))
        self.assertEqual(receipt['reason'], 'R package unavailable')

    def test_unknown_error_is_not_silently_reinterpreted_or_echoed(self):
        receipt = self.run_fixture('PRIVATE_ROW_SENTINEL custom fatal failure\n')
        self.assertEqual(receipt['error']['category'], 'unclassified_execution_error')
        self.assertNotIn('PRIVATE_ROW_SENTINEL', json.dumps(receipt))
        self.assertEqual(receipt['returncode'], 1)
        self.assertEqual(receipt['code'], self.code)
        self.assertFalse(receipt['error']['repairable_from_receipt'])

    def test_missing_function_preserves_code_grounded_symbol_and_line(self):
        self.code = 'd <- read.csv("/data/styles.csv")\nstrata(d$choice_set)\n'
        receipt = self.run_fixture('PRIVATE_ROW_SENTINEL\nError in strata(choice_set) : could not find function "strata"\n')
        self.assertEqual(receipt['error']['category'], 'function_not_found')
        self.assertEqual(receipt['error']['symbol'], 'strata')
        self.assertEqual(receipt['error']['source_lines'], [2])
        self.assertNotIn('PRIVATE_ROW_SENTINEL', json.dumps(receipt))
        setup = self.root / receipt['runner_source']['path']
        self.assertIn('library(survival)', setup.read_text())
        self.assertEqual(receipt['runner_source']['sha256'], hashlib.sha256(setup.read_bytes()).hexdigest())

    def test_unknown_symbol_not_present_in_code_is_not_echoed(self):
        receipt = self.run_fixture('Error: could not find function "PRIVATE_ROW_SENTINEL"\n')
        self.assertEqual(receipt['error']['category'], 'unclassified_execution_error')
        self.assertNotIn('PRIVATE_ROW_SENTINEL', json.dumps(receipt))

    def test_real_zero_length_extraction_error_keeps_code_location(self):
        self.code = 'sm <- summary(fit)\nmetrics$value[2] <- unname(sm$ robust.se["typicality"])\n'
        receipt = self.run_fixture('PRIVATE_ROW_SENTINEL\nError in metrics$value[2] <- unname(sm$robust.se["typicality"]) : \n  replacement has length zero\n')
        self.assertEqual(receipt['error']['category'], 'invalid_result_extraction')
        self.assertEqual(receipt['error']['source_lines'], [2])
        self.assertIn('sm$robust.se', receipt['error']['failed_expression'])
        self.assertIn('research_estimate', receipt['error']['repair_guidance'])
        self.assertNotIn('PRIVATE_ROW_SENTINEL', json.dumps(receipt))

    def test_output_contract_exception_carries_repair_receipt_without_raw_rows(self):
        with self.assertRaises(module.DiagnosticOutputError) as captured:
            self.run_fixture(output='metric,value\nPRIVATE_ROW_SENTINEL,not_numeric\n', returncode=0)
        receipt = captured.exception.receipt
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['error']['stage'], 'output_validation')
        self.assertEqual(receipt['code'], self.code)
        self.assertNotIn('PRIVATE_ROW_SENTINEL', json.dumps(receipt))
        self.assertIn('numeric', receipt['error']['message'])

    def test_malformed_csv_is_an_actionable_contract_error(self):
        with self.assertRaises(module.DiagnosticOutputError) as captured:
            self.run_fixture(output='metric,value\nmean_x\n', returncode=0)
        self.assertEqual(captured.exception.receipt['error']['category'], 'invalid_output_contract')

    def test_completed_output_preserves_all_valid_values_and_source(self):
        receipt = self.run_fixture(output='metric,value\nmean_x,0.0194\np_value,0.192\nmissing,NA\n', returncode=0)
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['metrics'], {'mean_x': 0.0194, 'p_value': 0.192, 'missing': None})
        self.assertEqual(receipt['code'], self.code)
        self.assertNotIn('error', receipt)

    def test_assertion_keeps_compatible_reason_and_specific_feedback(self):
        receipt = self.run_fixture('Error: all(PRIVATE_ROW_SENTINEL) is not TRUE\n')
        self.assertEqual(receipt['reason'], 'R assertion failed; check diagnostic assumptions')
        self.assertEqual(receipt['error']['category'], 'assertion_failed')
        self.assertNotIn('PRIVATE_ROW_SENTINEL', json.dumps(receipt))
