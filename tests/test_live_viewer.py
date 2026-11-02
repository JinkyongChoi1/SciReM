"""Render boundaries; actual browser behavior is verified by Scheduler QA."""
import json
import re
import unittest
from research_memory.live_viewer import render_live_dashboard


class LiveViewerTests(unittest.TestCase):
    def test_server_html_never_contains_access_token(self):
        token = 'private-access-token-do-not-serve'
        html = render_live_dashboard(token)
        self.assertNotIn(token, html)
        self.assertIn("new URLSearchParams(location.hash.slice(1)).get('token')", html)
        self.assertIn("history.replaceState(null,'',location.pathname+location.search)", html)
        self.assertRegex(html, r"['\"]Authorization['\"]\s*:\s*['\"]Bearer ")

    def test_initial_state_is_json_not_executable_markup(self):
        state = {'tasks': [{'question': "</script><script>alert('unsafe')</script>&"}]}
        html = render_live_dashboard('', state)
        self.assertNotIn(state['tasks'][0]['question'], html)
        payload = re.search(r'<script id="bootstrap" type="application/json">(.*?)</script>', html, re.S).group(1)
        self.assertEqual(json.loads(payload), {'state': state})
        self.assertNotIn('innerHTML', html)

    def test_memory_ui_uses_english_and_keeps_connection_state(self):
        html = render_live_dashboard('')
        for identifier in ('activity', 'rank-table', 'connection', 'attempt-status'):
            self.assertIn('id="' + identifier + '"', html)
        self.assertIn('<html lang="en">', html)
        self.assertIsNone(re.search('[\uac00-\ud7a3\u3131-\u318e]', html))
        self.assertIn('Synthetic', html)
        self.assertNotIn('Math.random', html)

    def test_memory_only_interface_has_no_research_workflow_or_role_cards(self):
        html = render_live_dashboard('')
        self.assertNotIn('class="role-card', html)
        self.assertNotIn('class="cycle-step', html)
        self.assertNotIn('Research workflow', html)
        self.assertNotIn('Research cycle', html)
        self.assertIn('memory_inventory', html)
        self.assertIn('id="memory-type-legend"', html)
        self.assertIn('memory-inactive', html)
        self.assertIn('<details', html)
        self.assertIn('evidence', html.lower())

    def test_query_ranking_and_memory_selection_are_source_backed(self):
        html = render_live_dashboard('')
        for identifier in ('memory-query', 'memory-graph', 'rank-table', 'evidence'):
            self.assertIn('id="' + identifier + '"', html)
        for field in ('selectedMemoryId', 'anchor_weight', 'ppr', 'actual_delivery',
                      'full_packet_delivered', 'decision-edge', 'follows_decision', 'based_on'):
            self.assertIn(field, html)
        self.assertIn('Score', html)
        self.assertIn('Rank', html)

    def test_model_design_and_research_question_remain_inspectable(self):
        specification = {'identification_assumptions': 'Synthetic assumption to inspect, not a validated claim.'}
        state = {'experiment': {'artifacts': {'plan': {'model_specification': specification}}}}
        html = render_live_dashboard('', state)
        payload = re.search(r'<script id="bootstrap" type="application/json">(.*?)</script>', html, re.S).group(1)
        self.assertEqual(json.loads(payload)['state']['experiment']['artifacts']['plan']['model_specification'], specification)
        self.assertIn('appendMemoryRecord', html)
        self.assertIn('appendReadable', html)
        self.assertIn('research question', html.lower())
        self.assertIn('Context', html)

    def test_failed_connection_is_bounded_and_offline_snapshots_do_not_poll(self):
        html = render_live_dashboard('')
        self.assertIn('AbortController', html)
        self.assertIn('abort.abort()', html)
        self.assertIn('document.hidden', html)
        self.assertIn('Disconnected', html)
        offline = render_live_dashboard('', {'capabilities': {'agent_run': False}}, offline=True)
        payload = re.search(r'<script id="bootstrap" type="application/json">(.*?)</script>', offline, re.S).group(1)
        self.assertTrue(json.loads(payload)['offline'])
        self.assertIn('offlineSnapshot', offline)
        self.assertIn('Saved snapshot', offline)

    def test_compact_canvas_has_collapsed_relevance_and_source_linked_human_review(self):
        html = render_live_dashboard('')
        for identifier in ('relevance-drawer', 'ai-activity', 'review-memory-id',
                           'review-note', 'review-approve', 'review-request-change',
                           'review-status', 'human-reviews'):
            self.assertIn('id="' + identifier + '"', html)
        drawer = re.search(r'<details\b[^>]*id="relevance-drawer"[^>]*>', html).group(0)
        self.assertNotRegex(drawer, r'\bopen(?:\s|=|>)')
        self.assertIn('/api/memory-review', html)
        self.assertIn('memory_review', html)
        self.assertIn('not applied', html.lower())
