"""C-only bounded context, exact private recall and integration regressions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import four_role_discovery as module
from four_role_discovery import RoleHistory, call_role_with_memory
from private_inference import ContextCapacityError


class AdaptiveContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.history = RoleHistory(self.root)
        self.flag = patch.object(module, 'ADAPTIVE_SHARED_CONTEXT', True)
        self.flag.start()

    def tearDown(self):
        self.flag.stop()
        self.tmp.cleanup()

    def grow(self, count=120):
        for n in range(count):
            self.history.add('hypothesis', 'decision',
                             {'evidence_ids': ['result-%04d' % (n + 1)], 'basis': 'UNIQUE-%d-' % n + str(n) * 700},
                             'ORIGINAL-INPUT-%d-' % n + str(n) * 8000)

    def test_distinct_growth_is_bounded_archive_survives_and_continues(self):
        self.grow()
        archive = self.history.root / 'hypothesis.jsonl'
        original = archive.read_bytes()
        self.assertGreater(len(original), 1048576)
        request = 'CURRENT RECEIPT: interaction coefficient=.0194; p=.192; significance=X; RQ unchanged.'
        first = self.history.prompt('hypothesis', request, shared_memory=True)
        self.assertLess(len(first), 10000)
        self.assertIn(request, first)
        self.assertNotIn('ORIGINAL-INPUT-0-', first)
        self.assertEqual(original, archive.read_bytes())
        # A new process/session can continue from the same archive/checkpoint without reloading all turns into input.
        reloaded = RoleHistory(self.root)
        reloaded.add('hypothesis', 'hypothesis', {'claim': 'A new inquiry after compaction'}, 'next input')
        second = reloaded.prompt('hypothesis', request, shared_memory=True)
        self.assertLess(len(second), 10000)
        self.assertIn('A new inquiry after compaction', second)
        self.assertEqual(archive.read_bytes()[:len(original)], original)
        checkpoint = json.loads((self.history.root / 'hypothesis-checkpoint.json').read_text())
        self.assertEqual(checkpoint['archived_turns'], 121)
        self.assertGreater(checkpoint['omitted_turns'], 0)

    def test_old_unique_fact_is_exactly_recoverable_with_pagination(self):
        text = 'OLD INTERACTION: coefficient=.0194, p=.192, SE=.0101; Unicode\u2028still one record. ' + 'original' * 5000
        self.history.add('hypothesis', 'decision', {'fact': 'NOT SIGNIFICANT'}, text)
        original = json.dumps(self.history._items('hypothesis')[0], ensure_ascii=False)
        pages = [];offset = 0
        while True:
            page = self.history.recall('hypothesis', 'private-turn:1:%d' % offset)
            pages.append(page['content'])
            self.assertEqual(page['source_sha256'], hashlib.sha256(original.encode()).hexdigest())
            self.assertLessEqual(len(page['content']), 12000)
            if page['next_offset'] is None:
                break
            offset = page['next_offset']
        self.assertEqual(''.join(pages), original)
        self.assertEqual(json.loads(''.join(pages))['prompt'], text)

    def test_private_recall_call_path_uses_no_graph_and_returns_final_schema(self):
        self.history.add('hypothesis', 'decision', {'fact': 'p=.192; significance=X'}, 'old unique input')
        self.history.add('empiricist', 'plan', {'secret': 'OTHER_ROLE_SECRET'}, 'other role input')
        memory = Mock();calls = [];record = {}
        schema = {'type': 'object', 'properties': {'value': {'type': 'string'}},
                  'required': ['value'], 'additionalProperties': False}
        def model(stage, prompt, requested, tokens, seed):
            calls.append((prompt, requested))
            if len(calls) == 1:
                return {'action': 'retrieve_memory', 'query': 'private-turn:1:0', 'answer': None}
            self.assertIn('p=.192; significance=X', prompt)
            self.assertIn('old unique input', prompt)
            self.assertNotIn('OTHER_ROLE_SECRET', prompt)
            self.assertEqual(requested['anyOf'][1]['properties']['answer'], schema)
            return {'action': 'answer', 'query': None, 'answer': {'value': 'Read original correctly'}}
        answer, _, trace, supplied = call_role_with_memory(
            self.root, 'hypothesis', 'hypothesis', 'hypothesis', 'Current test', schema,
            100, 1, model, memory, self.history, record, True)
        self.assertEqual(answer, {'value': 'Read original correctly'})
        self.assertEqual(len(calls), 2)
        memory.retrieve_tsv.assert_not_called()
        self.assertIsNone(trace)
        self.assertEqual(supplied, {})
        self.assertEqual(record['private_history_recalls'][0]['turn'], 1)
        self.assertTrue((self.root / 'private-history-01.json').is_file())

    def test_b_receives_no_custom_compaction_or_recall_interface(self):
        self.grow()
        with patch.object(module, 'COMPACT_SHARED_HISTORY', False):
            baseline = self.history.prompt('hypothesis', 'CURRENT B', shared_memory=False)
            with patch.object(module, 'ADAPTIVE_SHARED_CONTEXT', False):
                self.assertEqual(baseline, self.history.prompt('hypothesis', 'CURRENT B', shared_memory=False))
        self.assertIn('ORIGINAL-INPUT-0-', baseline)
        self.assertFalse((self.history.root / 'hypothesis-checkpoint.json').exists())
        model = Mock(return_value={'answer': 'normal'})
        call_role_with_memory(self.root, 'decision', 'hypothesis', 'hypothesis', 'B current', {},
                              100, 1, model, None, self.history, {}, False)
        self.assertEqual(model.call_count, 1)
        self.assertNotIn('private-turn:', model.call_args.args[1])

    def test_unknown_paths_roles_and_offsets_rejected(self):
        self.history.add('hypothesis', 'decision', {'fact': 'own'}, 'input')
        for query in ['../../empiricist.jsonl', 'private-turn:0:0', 'private-turn:1:-1',
                      'private-turn:2:0', 'private-turn:1:999999']:
            with self.assertRaises(ValueError):
                self.history.recall('hypothesis', query)
        with self.assertRaises(ValueError):
            self.history.recall('../empiricist', 'private-turn:1:0')

    def test_current_request_is_not_silently_truncated(self):
        current = ''.join('UNIQUE_CURRENT_%06d; coefficient=.0194; p=.192\n' % n for n in range(5000))
        self.assertGreater(len(current), 160000)
        rendered = self.history.prompt('hypothesis', current, shared_memory=True)
        self.assertIn(current, rendered)

    def test_history_shrinking_does_not_reinflate_after_checkpoint(self):
        self.grow()
        self.history.prompt('hypothesis', 'current', shared_memory=True)
        # Same archive, shorter subsequent current request: checkpoint still applies.
        rendered = RoleHistory(self.root).prompt('hypothesis', 'short', shared_memory=True)
        self.assertLess(len(rendered), 10000)
        self.assertIn('C ACTIVE CHECKPOINT', rendered)

    def test_cli_defaults_to_c_only_and_retains_explicit_historical_policy(self):
        from memory_iteration import use_adaptive_shared_context
        self.assertTrue(use_adaptive_shared_context('C', False))
        self.assertFalse(use_adaptive_shared_context('C', True))
        self.assertFalse(use_adaptive_shared_context('A', False))
        self.assertFalse(use_adaptive_shared_context('B', False))
        with self.assertRaises(ValueError):
            use_adaptive_shared_context('B', False, True)
        with self.assertRaises(ValueError):
            use_adaptive_shared_context('C', True, True)

    def test_current_repeated_tsv_rows_are_losslessly_encoded(self):
        row = 'result-0020\tinteraction\t.0194\t.0101\t.192\tX\t842\t' + 'source-provenance-' * 20
        rendered = self.history.prompt('hypothesis', 'RQ fixed\n' + (row + '\n') * 800, shared_memory=True)
        self.assertIn(row, rendered)
        self.assertEqual(rendered.count(row), 1)
        self.assertIn('END TEXT DICTIONARY', rendered)
        self.assertIn('RQ fixed', rendered)

    def test_private_recall_can_read_more_than_one_page_in_same_stage(self):
        self.history.add('hypothesis', 'decision', {'value': 'END FACT p=.192'}, 'x' * 18000)
        record = {}; calls = []
        schema = {'type': 'object', 'properties': {'value': {'type': 'string'}}}
        def model(stage, prompt, requested, tokens, seed):
            calls.append(prompt)
            if len(calls) == 1:
                return {'action': 'retrieve_memory', 'query': 'private-turn:1:0', 'answer': None}
            if len(calls) == 2:
                self.assertIn('"next_offset": 12000', prompt)
                return {'action': 'retrieve_memory', 'query': 'private-turn:1:12000', 'answer': None}
            self.assertIn('END FACT p=.192', prompt)
            return {'action': 'answer', 'query': None, 'answer': {'value': 'Read all pages'}}
        result, _, _, _ = call_role_with_memory(self.root, 'decision', 'hypothesis', 'hypothesis',
            'Read exact archived input and output', schema, 100, 1, model, Mock(), self.history, record, True)
        self.assertEqual(result['value'], 'Read all pages')
        self.assertEqual(len(record['private_history_recalls']), 2)
        self.assertEqual(len(list(self.root.glob('memory-tool-*-request.json'))), 2)

    def test_repeated_query_does_not_refetch_and_multiple_traces_are_returned(self):
        memory = Mock();record = {};calls = []
        def retrieve(query, **kwargs):
            self.assertTrue(kwargs['source_lookup'])
            return ({'query': query, 'request_id': query, 'brief': query + '\tp=.192',
                     'selected_ids': [query]}, {query: {'verified': True}})
        memory.retrieve_tsv.side_effect = retrieve
        def model(stage, prompt, schema, tokens, seed):
            calls.append(prompt)
            if len(calls) <= 3:
                return {'action': 'retrieve_memory', 'query': ['result-0001', 'result-0001', 'result-0002'][len(calls)-1], 'answer': None}
            self.assertIn('identical query already delivered', prompt)
            return {'action': 'answer', 'query': None, 'answer': {'value': 'done'}}
        result, _, trace, supplied = call_role_with_memory(self.root, 'decision', 'hypothesis', 'hypothesis',
            'Current RQ', {}, 100, 1, model, memory, self.history, record, True)
        self.assertEqual(memory.retrieve_tsv.call_count, 2)
        self.assertEqual(set(trace['selected_ids']), {'result-0001', 'result-0002'})
        self.assertEqual(set(supplied), {'result-0001', 'result-0002'})
        self.assertEqual(len(trace['retrieval_traces']), 2)

    def test_current_source_beyond_real_transport_is_archived_and_pageable(self):
        current = ''.join('UNIQUE-CURRENT-%06d\n' % n for n in range(10000))
        prompts = []
        def model(stage, prompt, schema, tokens, seed):
            prompts.append(prompt)
            self.assertLessEqual(len(prompt), model.input_character_limit)
            self.assertIn('COMPLETE SOURCE ARCHIVED', prompt)
            self.assertIn('first/last pages, not the whole source', prompt)
            return {'action': 'answer', 'query': None, 'answer': {'value': 'Source is paged, not absent'}}
        model.input_character_limit = 100000
        call_role_with_memory(self.root, 'hypothesis', 'hypothesis', 'hypothesis', current,
            {}, 100, 1, model, Mock(), self.history, {}, True)
        self.assertEqual(self.history._items('hypothesis')[0]['prompt'], current)
        self.assertEqual(len(prompts), 1)

    def test_provider_capacity_failure_retries_only_a_smaller_working_view(self):
        current = ''.join('UNIQUE-CURRENT-%06d\n' % n for n in range(10000))
        sizes = []; record = {}
        def model(stage, prompt, schema, tokens, seed):
            sizes.append(len(prompt))
            if len(sizes) == 1:
                raise ContextCapacityError('Provider input context exceeded')
            self.assertLess(sizes[-1], sizes[0])
            self.assertIn('COMPLETE SOURCE ARCHIVED', prompt)
            return {'action': 'answer', 'query': None, 'answer': {'value': 'read source'}}
        call_role_with_memory(self.root, 'hypothesis', 'hypothesis', 'hypothesis', current,
            {}, 100, 1, model, Mock(), self.history, record, True)
        self.assertEqual(record['context_compaction_recoveries'][0]['status'], 'recovered')
        self.assertEqual(self.history._items('hypothesis')[0]['prompt'], current)
        self.assertEqual(len(sizes), 2)

    def test_non_capacity_failure_is_not_retried(self):
        model = Mock(side_effect=RuntimeError('quota exhausted'))
        with self.assertRaisesRegex(RuntimeError, 'quota exhausted'):
            call_role_with_memory(self.root, 'hypothesis', 'hypothesis', 'hypothesis', 'Current RQ',
                {}, 100, 1, model, Mock(), self.history, {}, True)
        self.assertEqual(model.call_count, 1)
