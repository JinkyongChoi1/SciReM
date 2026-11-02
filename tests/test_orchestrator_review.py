"""Orchestrator must not promote workflow completion into scientific completion."""
import json
import unittest

from orchestrator_review import REVIEW_SCHEMA, adjudicate_review, review_prompt


def favorable(**overrides):
    value = {'requested_test_status': 'fulfilled', 'scientific_validity': 'adequate',
             'claim_consistency': 'consistent', 'required_assumptions_status': 'satisfied',
             'diagnostic_required': False, 'required_diagnostic_metrics': [], 'issues': [],
             'next_action': 'accept', 'explanation': 'The exact requested source test was evaluated.'}
    value.update(overrides)
    return value


def primary():
    return {'analysis_status': 'succeeded', 'estimates': [
        {'term': 'predictor', 'coefficient': '.0194', 'se': '.0101', 'p_value': '.192'}]}


def comparison(**overrides):
    return favorable(diagnostic_required=True, required_diagnostic_metrics=[
        {'name': 'score_difference', 'role': 'target_estimate'},
        {'name': 'difference_se', 'role': 'uncertainty'}], **overrides)


class OrchestratorReviewTests(unittest.TestCase):
    def test_scalar_primary_test_needs_no_invented_diagnostic(self):
        verdict = adjudicate_review(favorable(), primary())
        self.assertEqual(verdict['status'], 'accepted')
        self.assertEqual(verdict['workflow_status'], 'review_completed')
        self.assertEqual(verdict['requested_test_status'], 'fulfilled')
        self.assertIn('not independent scientific validation', verdict['scope'])
        self.assertEqual(verdict['review'], favorable())

    def test_failed_primary_overrides_favorable_review_without_rewriting_numbers(self):
        facts = {'analysis_status': 'failed', 'estimates': []}
        verdict = adjudicate_review(favorable(), facts)
        self.assertEqual(verdict['status'], 'revision_required')
        self.assertEqual(verdict['requested_test_status'], 'unfulfilled')
        self.assertEqual(verdict['scientific_validity'], 'unverified')
        self.assertEqual(facts, {'analysis_status': 'failed', 'estimates': []})

    def test_failed_diagnostic_overrides_even_if_reviewer_says_not_required(self):
        verdict = adjudicate_review(favorable(), primary(), {'status': 'failed'}, 'code')
        self.assertEqual(verdict['requested_test_status'], 'unfulfilled')
        self.assertEqual(verdict['next_action'], 'repair')

    def test_missing_and_all_na_outputs_do_not_complete_comparison(self):
        for metrics in ({}, {'score_difference': None, 'difference_se': 'NA'},
                        {'n': 100, 'score_difference': float('nan'), 'difference_se': float('inf')}):
            with self.subTest(metrics=metrics):
                verdict = adjudicate_review(comparison(), primary(), {'status': 'completed', 'metrics': metrics})
                self.assertEqual(verdict['requested_test_status'], 'unfulfilled')
                self.assertNotEqual(verdict['status'], 'accepted')

    def test_required_metric_names_are_exact_and_boolean_is_not_a_numeric_result(self):
        for metrics in ({'score_diff': .2, 'difference_se': .1},
                        {'score_difference': True, 'difference_se': .1}):
            verdict = adjudicate_review(comparison(), primary(), {'status': 'completed', 'metrics': metrics})
            self.assertTrue(any(item['code'] == 'required_metrics_unavailable' for item in verdict['deterministic_checks']))
            self.assertEqual(verdict['status'], 'revision_required')

    def test_count_only_completed_code_does_not_certify_the_requested_work(self):
        review = favorable(diagnostic_required=True, required_diagnostic_metrics=[
            {'name': 'n_observations', 'role': 'sample_count'},
            {'name': 'successful_exit', 'role': 'execution_metadata'}])
        verdict = adjudicate_review(review, primary(), {'status': 'completed', 'metrics': {
            'n_observations': 842, 'successful_exit': 1}})
        self.assertEqual(verdict['requested_test_status'], 'unverifiable')
        self.assertNotEqual(verdict['status'], 'accepted')

    def test_zero_gain_is_a_valid_completed_comparison_not_a_failure(self):
        verdict = adjudicate_review(comparison(), primary(), {'status': 'completed', 'metrics': {
            'score_difference': 0, 'difference_se': .1}})
        self.assertEqual(verdict['status'], 'accepted')

    def test_unknown_necessary_assumption_cannot_be_accepted(self):
        verdict = adjudicate_review(comparison(required_assumptions_status='unverified'), primary(),
                                    {'status': 'completed', 'metrics': {'score_difference': .2, 'difference_se': .1}})
        self.assertEqual(verdict['scientific_validity'], 'unverified')
        self.assertEqual(verdict['next_action'], 'blocked')
        self.assertEqual(verdict['status'], 'unknown')

    def test_violated_assumption_and_contradicted_claim_remain_adverse(self):
        verdict = adjudicate_review(favorable(required_assumptions_status='violated',
            scientific_validity='invalid', claim_consistency='contradicted'), primary())
        self.assertEqual(verdict['scientific_validity'], 'invalid')
        self.assertEqual(verdict['claim_consistency'], 'contradicted')
        self.assertEqual(verdict['status'], 'revision_required')

    def test_missing_primary_receipt_is_unknown_not_success(self):
        verdict = adjudicate_review(favorable(), {})
        self.assertEqual(verdict['requested_test_status'], 'unverifiable')
        self.assertEqual(verdict['status'], 'unknown')

    def test_review_prompt_contains_actual_sources_and_no_nesting_assumption(self):
        prompt = review_prompt('Compare the models', {'hypothesis': 'current'}, {'analysis': 'actual'},
                               primary(), 'actual program', {'status': 'failed'}, {'claim': 'current assessment'})
        payload = json.loads(prompt.split('REVIEW ARTIFACTS:\n', 1)[1])
        self.assertEqual(payload['diagnostic_code'], 'actual program')
        self.assertEqual(payload['diagnostic_receipt']['status'], 'failed')
        self.assertIn('Do not infer grouping, nesting or assignment from column', prompt)

    def test_invalid_review_schema_cannot_silently_accept(self):
        review = favorable()
        review['requested_test_status'] = 'probably_done'
        with self.assertRaises(ValueError):
            adjudicate_review(review, primary())
        review = favorable(required_diagnostic_metrics=[{'name': 'x', 'role': 'target_estimate'},
                                                       {'name': 'x', 'role': 'uncertainty'}])
        with self.assertRaises(ValueError):
            adjudicate_review(review, primary())
        self.assertEqual(set(favorable()), set(REVIEW_SCHEMA['required']))

    def test_required_metric_identifier_matches_runner_contract(self):
        for name in ('descriptive text instead of a metric', '1gain', 'x' * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                adjudicate_review(favorable(required_diagnostic_metrics=[
                    {'name': name, 'role': 'target_estimate'}]), primary())

    def test_normal_exit_is_distinguished_from_missing_output(self):
        prompt = review_prompt('Compare models', {}, {}, primary(), 'actual program',
                               {'returncode': 0, 'status': 'failed', 'reason': 'missing output'})
        payload = json.loads(prompt.split('REVIEW ARTIFACTS:\n', 1)[1])
        self.assertEqual(payload['execution_status']['diagnostic_process'], 'exited_successfully')
        self.assertEqual(payload['execution_status']['diagnostic_output_contract'], 'failed')
        self.assertIn('do not say the code was not executed', prompt)
        verdict = adjudicate_review(comparison(), primary(), payload['diagnostic_receipt'], 'actual program')
        self.assertEqual(verdict['requested_test_status'], 'unfulfilled')
        self.assertEqual(verdict['next_action'], 'repair')

    def test_missing_returncode_does_not_invent_execution(self):
        prompt = review_prompt('Compare models', {}, {}, primary(), 'actual program', {'status': 'failed'})
        payload = json.loads(prompt.split('REVIEW ARTIFACTS:\n', 1)[1])
        self.assertEqual(payload['execution_status']['diagnostic_process'], 'unknown')

    def test_prompt_uses_minimal_existing_evidence_without_moving_goalposts(self):
        prompt = review_prompt('Compare models', {}, {}, primary())
        self.assertIn('not a reporting wish list', prompt)
        self.assertIn('exact existing source metric names', prompt)
        self.assertIn('facts already established by those sources', prompt)
        self.assertIn('Do not expand the inquiry', prompt)
        self.assertIn('Separate the parameter-estimation method from the evaluation/scoring criterion', prompt)
        self.assertIn('do not require the fitting objective', prompt.lower())

    def test_non_numeric_provenance_and_composite_intervals_are_not_metric_requirements(self):
        prompt = review_prompt('Compare models', {}, {}, primary())
        self.assertIn('finite numeric scalar output', prompt)
        self.assertIn('never in this numeric metric list', prompt)
        self.assertIn('existing scalar lower-bound and upper-bound metric names', prompt)
        self.assertIn('unknown or unchecked assumption is unverified, not violated', prompt)

    def test_honest_all_na_failure_claim_remains_consistent(self):
        review = comparison(requested_test_status='unfulfilled', scientific_validity='unverified',
                            next_action='repair', explanation='All diagnostic outputs are unavailable; no comparison was established.')
        verdict = adjudicate_review(review, primary(), {'status': 'completed', 'metrics': {
            'score_difference': None, 'difference_se': 'NA'}})
        self.assertEqual(verdict['claim_consistency'], 'consistent')
        self.assertEqual(verdict['requested_test_status'], 'unfulfilled')
        self.assertEqual(verdict['scientific_validity'], 'unverified')
        self.assertEqual(verdict['status'], 'revision_required')

    def test_failed_execution_does_not_override_reviewed_claim_consistency(self):
        for consistency in ('consistent', 'contradicted', 'unverified'):
            with self.subTest(consistency=consistency):
                verdict = adjudicate_review(comparison(claim_consistency=consistency), primary(),
                                            {'status': 'failed', 'returncode': 1})
                self.assertEqual(verdict['claim_consistency'], consistency)
                self.assertEqual(verdict['requested_test_status'], 'unfulfilled')
