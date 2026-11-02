"""Current-result diagnostic evidence must survive both review policies."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import two_role_discovery as module


class DiagnosticTableTests(unittest.TestCase):
    def test_sample_flag_interaction_is_available_to_schema_and_runner(self):
        from discovery_analysis import validate_action,render_r,FILTERS
        from discovery_pilot import ALLOWED_TERMS,FOCAL_SCHEMA
        from jsonschema import Draft202012Validator
        from research_memory.evidence import validate_focal_test,statistical_assessment
        for flag in FILTERS-{'all'}:
            term='distinctive:'+flag
            action={'family':'conditional_logit','outcome':'tagged','terms':['distinctive',flag,term],
                    'strata':'choice_set','cluster':'user_label','filter':'all'}
            self.assertIn(term,ALLOWED_TERMS)
            Draft202012Validator(FOCAL_SCHEMA).validate({'effect_type':'interaction','term':term,'expected_direction':'nonzero'})
            self.assertIn(term,validate_action(action)['terms'])
            focal={'effect_type':'interaction','term':term,'expected_direction':'nonzero'}
            plan={'hypothesis':'Test the interaction','focal_test':focal,'analysis':action}
            self.assertEqual(validate_focal_test(plan),focal)
            check=statistical_assessment(plan,{'returncode':0,'effective_action':action,
                'estimates':[{'term':term,'coefficient':'0.03','se':'0.02','p_value':'0.13'}]})
            self.assertEqual(check['support'],'not_supported')
            self.assertNotIn('design_invalid',check['reason'])
            code=render_r(action)
            self.assertIn(term,code)
            self.assertIn('cluster=user_label',code)
            self.assertIn('if (is.logical(d[[column]])) d[[column]] <- as.integer(d[[column]])',code)
        with self.assertRaises(ValueError):
            validate_action({**action,'terms':['distinctive:unknown_flag']})

    def test_complete_current_diagnostic_values_without_primary_metric_merge(self):
        facts={"metrics":[{"n":"228319"}],"agent_r_diagnostic":{
            "status":"completed","metrics":{"nested_gain":9.45518983029251,
            "flag_coefficient":0.0357601385289304,"unknown":None,"zero":0.0}}}
        before=deepcopy(facts)
        prompt=module.role_prompt("scientific decision",current_result=facts,memory_format="tsv")
        self.assertIn("CURRENT_DIAGNOSTIC_METRICS",prompt)
        for name in facts['agent_r_diagnostic']['metrics']:self.assertIn(name,prompt)
        self.assertIn("0.0357601385289304",prompt)
        self.assertIn("9.45518983029251",prompt)
        self.assertIn("independently validated",prompt)
        self.assertEqual(facts,before)

    def test_empty_or_failed_diagnostic_does_not_invent_values(self):
        for diagnostic in [None,{}, {"status":"failed","error":{"message":"failure"}}]:
            self.assertEqual(module.diagnostic_metrics_table(diagnostic),"")

    def test_control_omits_only_the_new_table(self):
        facts={"agent_r_diagnostic":{"status":"completed","metrics":{"probe_metric":7.25}}}
        treated=module.role_prompt("scientific decision",current_result=facts,memory_format="tsv")
        table=module.diagnostic_metrics_table(facts['agent_r_diagnostic'])
        with patch.object(module,'diagnostic_metrics_table',return_value=''):
            control=module.role_prompt("scientific decision",current_result=facts,memory_format="tsv")
        self.assertEqual(treated.replace(table,'',1),control)
