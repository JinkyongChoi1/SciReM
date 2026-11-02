"""Scoped scientific relations through real graph publication/retrieval; Scheduler only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.two_role import TwoRoleMemory, JUDGMENT_RELATIONS
from research_memory.viewer import render_dashboard
from research_memory import evidence
from audit_tsv_iteration import audit_judgment_edges
from test_tsv_memory import tables


class JudgmentEdgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = TwoRoleMemory(self.root)
        self.config = self.save('config.json', {'question': 'Why category assignment?'})

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def publish(self, index=1, coefficient='.2', p_value='.1', returncode=0, missing=False, assumptions='Unmeasured selection remains possible.', direction='positive'):
        h = {'hypothesis': 'Typicality association.', 'expected_pattern': direction,
             'focal_test': {'effect_type': 'main_effect', 'term': 'typicality', 'expected_direction': direction}}
        prefix = f'attempt-{index:04d}/'
        hid = self.memory.publish_hypothesis(index, h, self.save(prefix+'hypothesis.json', h), 'Why category assignment?', self.config)
        plan = {**h, 'analysis': {'family': 'logit', 'outcome': 'tagged', 'terms': ['typicality'], 'strata': 'none', 'cluster': 'none', 'filter': 'all'},
                'model_specification': {'identification_assumptions': assumptions}}
        path = self.save(prefix+'plan.json', plan)
        mid = self.memory.publish_model(index, hid, plan, path)
        tool = {'returncode': returncode, 'effective_action': plan['analysis'], 'metrics': [{'n': '100'}],
                'estimates': [] if missing else [{'term': 'typicality', 'coefficient': coefficient, 'se': '.1', 'p_value': p_value}]}
        rid = self.memory.publish_result(index, mid, self.save(prefix+'tool-result.json', tool), path, data_identity='synthetic')
        return hid, mid, rid

    def test_statistical_support_challenge_and_unknown_are_distinct(self):
        cases = [('.2', '.1', 0, False, {'supports'}), ('-.2', '.05', 0, False, {'challenges'}),
                 ('-.2', '.192', 0, False, set()), ('.2', '.470', 0, False, set()),
                 ('.2', '.01', 1, False, set()), ('.2', '.01', 0, True, set()),
                 ('0', '.01', 0, False, set())]
        for i, (coef, p, rc, missing, expected) in enumerate(cases, 1):
            with self.subTest(case=i):
                hid, _, rid = self.publish(i, coef, p, rc, missing)
                edges = [e for e in self.memory.nodes[rid]['edges'] if e['relation'] in JUDGMENT_RELATIONS]
                self.assertEqual({e['relation'] for e in edges}, expected)
                for edge in edges:
                    self.assertEqual(edge['target'], hid)
                    self.assertEqual(edge['scientific_validity'], 'unverified')
                    self.assertEqual(edge['basis']['estimate']['p_value'], p)
                    self.assertEqual(edge['provenance'], self.memory.nodes[rid]['provenance'])

    def test_negative_expected_direction_is_scoped_correctly(self):
        _, _, rid = self.publish(coefficient='.2', p_value='.01', direction='negative')
        self.assertEqual(self.memory.nodes[rid]['edges'][-1]['relation'], 'challenges')

    def test_declared_assumption_is_independent_edge_not_validated_truth(self):
        text = 'Selection\tmay remain.\nConditional association only.'
        hid, mid, _ = self.publish(assumptions=text)
        edge = self.memory.nodes[mid]['edges'][-1]
        self.assertEqual((edge['relation'], edge['target']), ('assumption_constraint', hid))
        self.assertEqual(edge['basis'], text)
        self.assertEqual(edge['assertion_status'], 'declared_not_validated')
        self.assertEqual(edge['provenance'], self.memory.nodes[mid]['provenance'])
        _, no_assumption, _ = self.publish(2, assumptions='')
        self.assertFalse(any(e['relation']=='assumption_constraint' for e in self.memory.nodes[no_assumption]['edges']))

    def test_reload_ppr_tsv_and_viewer_keep_same_scoped_edges(self):
        hid, mid, rid = self.publish()
        restored = TwoRoleMemory(self.root)
        trace, records = restored.retrieve_tsv('typicality category assignment')
        self.assertTrue(trace['converged'])
        raw = [e for e in trace['edges'] if e['relation'] in JUDGMENT_RELATIONS]
        self.assertEqual({e['relation'] for e in raw}, {'supports','assumption_constraint'})
        self.assertIn(rid, records)
        delivered = tables(trace['brief'])['EDGES']
        support = next(e for e in delivered if e['relation']=='supports')
        self.assertEqual((support['origin'],support['target']), (rid,hid))
        self.assertEqual(support['scientific_validity'],'unverified')
        self.assertEqual(json.loads(support['provenance']), restored.nodes[rid]['provenance'])
        html = render_dashboard(trace)
        self.assertIn('assumption_constraint', html)
        self.assertIn('derived_primary_statistical_test', html)
        self.assertEqual({n['memory_type'] for n in restored.nodes.values()},
                         {'ResearchQuestion','Hypothesis','AnalysisModel','ResultAssessment'})

    def test_edge_metadata_and_original_source_tampering_are_rejected(self):
        _, mid, rid = self.publish()
        original = deepcopy(self.memory.nodes[rid]['edges'])
        self.memory.nodes[rid]['edges'][-1]['relation']='challenges'
        with self.assertRaisesRegex(ValueError, 'Judgment edges'):
            self.memory._require(rid, 'ResultAssessment')
        self.memory.nodes[rid]['edges']=original
        self.memory.nodes[mid]['edges'][-1]['basis']='Verified causal identification'
        with self.assertRaisesRegex(ValueError, 'Judgment edges'):
            self.memory._require(mid, 'AnalysisModel')
        restored = TwoRoleMemory(self.root)
        path = self.root/'attempt-0001/plan.json'
        plan=json.loads(path.read_text());plan['model_specification']['identification_assumptions']='Changed source'
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'source has changed'):
            restored.retrieve_tsv('typicality')

    def test_independent_auditor_recomputes_meaning_not_just_edge_types(self):
        _, mid, rid = self.publish()
        _, _, nonsig = self.publish(2, p_value='.192')
        for node in self.memory.nodes.values():
            audit_judgment_edges(node, self.memory.nodes, self.root, evidence)
        forged = deepcopy(self.memory.nodes[nonsig])
        edge = deepcopy(self.memory.nodes[rid]['edges'][-1])
        edge.update(target='hypothesis-0002', source=forged['source'], provenance=forged['provenance'])
        forged['edges'].append(edge)
        with self.assertRaisesRegex(ValueError, 'source recomputation'):
            audit_judgment_edges(forged, self.memory.nodes, self.root, evidence)
        wrong_type = deepcopy(self.memory.nodes[rid])
        wrong_type['edges'][-1]['target'] = mid
        with self.assertRaisesRegex(ValueError, 'source recomputation'):
            audit_judgment_edges(wrong_type, self.memory.nodes, self.root, evidence)
        promoted = deepcopy(self.memory.nodes[mid])
        promoted['edges'][-1]['assertion_status'] = 'scientifically_validated'
        with self.assertRaisesRegex(ValueError, 'source recomputation'):
            audit_judgment_edges(promoted, self.memory.nodes, self.root, evidence)

    def test_legacy_candidates_without_new_relations_remain_readable(self):
        self.publish()
        for node in self.memory.nodes.values():
            legacy = deepcopy(node)
            legacy.pop('judgment_edge_schema', None)
            legacy['edges'] = [e for e in legacy['edges'] if e['relation'] not in JUDGMENT_RELATIONS]
            audit_judgment_edges(legacy, self.memory.nodes, self.root, evidence)
