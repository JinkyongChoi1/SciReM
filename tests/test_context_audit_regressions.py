"""Independent regressions for stage-local retrieval growth and source validation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import four_role_discovery as module


class ContextAuditRegressions(unittest.TestCase):
    def test_literal_dictionary_marker_is_not_reinterpreted_as_a_reference(self):
        row = 'result-0001\tcoefficient=.0194\tp=.192\t' + 'scientific context ' * 40
        original = '[SAME L1]\n' + '\n'.join([row] * 8)
        rendered = module.lossless_prompt(original)
        if rendered != original:
            dictionary = rendered.partition('\nEND TEXT DICTIONARY\n')[0]
            self.assertNotIn('[SAME Lx] means', dictionary,
                             'Literal source marker collides with an adapter-generated alias')

    def test_many_private_pages_do_not_replay_entire_stage_transcript(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(module, 'ADAPTIVE_SHARED_CONTEXT', True):
            root = Path(folder)
            history = module.RoleHistory(root)
            # Distinct long single-line pages cannot disappear via repeated-line aliases.
            original_input = ''.join('%06d:distinct-source-value|' % n for n in range(70000))
            history.add('hypothesis', 'decision', {'fact': 'archived output'}, original_input)
            original_archive = (history.root / 'hypothesis.jsonl').read_bytes()
            calls = []
            record = {}
            memory = Mock()
            schema = {'type': 'object', 'properties': {'done': {'type': 'boolean'}},
                      'required': ['done'], 'additionalProperties': False}

            def model(stage, prompt, requested, tokens, seed):
                calls.append(len(prompt))
                self.assertLess(len(prompt), 1040000,
                                'Stage retrieval replay approaches the real CLI transport limit')
                ordinal = len(calls) - 1
                if ordinal < 100:
                    return {'action': 'retrieve_memory',
                            'query': 'private-turn:1:%d' % (ordinal * 12000), 'answer': None}
                return {'action': 'answer', 'query': None, 'answer': {'done': True}}

            model.input_character_limit = 1040000

            answer, _, trace, supplied = module.call_role_with_memory(
                root, 'hypothesis', 'hypothesis', 'hypothesis', 'Fixed RQ; inspect original pages.',
                schema, 100, 1, model, memory, history, record, True)
            self.assertEqual(answer, {'done': True})
            self.assertEqual(len(record['private_history_recalls']), 100)
            self.assertEqual((history.root / 'hypothesis.jsonl').read_bytes()[:len(original_archive)],
                             original_archive)
            self.assertIsNone(trace)
            self.assertEqual(supplied, {})
            memory.retrieve_tsv.assert_not_called()

    def test_combined_retrieval_claims_validate_against_each_original_trace(self):
        memory = Mock()
        first = {'request_id': 'one', 'selected_ids': ['result-0001'], 'brief': 'first source'}
        second = {'request_id': 'two', 'selected_ids': ['result-0002'], 'brief': 'second source'}
        combined = dict(second, retrieval_traces=[first, second],
                        selected_ids=['result-0001', 'result-0002'], brief='combined')
        claims = [{'evidence_id': 'result-0001', 'field': 'p_value', 'value': .192},
                  {'evidence_id': 'result-0002', 'field': 'coefficient', 'value': .0194}]

        def validate(batch, trace):
            self.assertIn(trace, [first, second])
            self.assertIn(batch[0]['evidence_id'], trace['selected_ids'])
            return [{'claim': batch[0], 'status': 'verified'}]

        memory.validate_context_claims.side_effect = validate
        outcomes = module.validate_retrieved_claims(memory, claims, [combined])
        self.assertEqual([item['status'] for item in outcomes], ['verified', 'verified'])

    def test_source_match_does_not_certify_reading_undelivered_middle_rows(self):
        claim = {'evidence_id': 'result-0001', 'field': 'p_value', 'value': .192}
        trace = {'request_id': 'paged', 'selected_ids': ['result-0001'],
                 'brief': 'complete source containing a middle coefficient row',
                 'actual_delivery': {'status': 'partial', 'full_packet_delivered': False}}
        memory = Mock()
        memory.validate_context_claims.return_value = [{'claim': claim, 'status': 'verified'}]
        outcome = module.validate_retrieved_claims(memory, [claim], [trace])[0]
        self.assertEqual(outcome['status'], 'unverifiable')

    def test_evicted_oversized_source_can_be_requested_again(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(module, 'ADAPTIVE_SHARED_CONTEXT', True):
            root = Path(folder)
            history = module.RoleHistory(root)
            record = {}
            queries = ['first source', 'second source', 'first source']
            memory = Mock()
            memory.retrieve_tsv.side_effect = lambda query, **kwargs: ({
                'request_id': query, 'query': query, 'selected_ids': [],
                'brief': query.replace(' ', '-') * 120000}, {})
            calls = []

            def model(stage, prompt, schema, tokens, seed):
                calls.append(prompt)
                self.assertLess(len(prompt), 1040000)
                if len(calls) <= len(queries):
                    return {'action': 'retrieve_memory', 'query': queries[len(calls) - 1], 'answer': None}
                return {'action': 'answer', 'query': None, 'answer': {'done': True}}

            model.input_character_limit = 1040000
            answer, _, _, _ = module.call_role_with_memory(
                root, 'hypothesis', 'hypothesis', 'hypothesis', 'Fixed request.', {}, 100, 1,
                model, memory, history, record, True)
            self.assertEqual(answer, {'done': True})
            self.assertEqual(memory.retrieve_tsv.call_count, 2)


if __name__ == '__main__':
    unittest.main()
