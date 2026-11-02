"""Render boundary checks; browser interaction is verified by the SGE QA job."""

import json
import re
import unittest

from research_memory.live_viewer import render_live_dashboard


class LiveViewerTests(unittest.TestCase):
    def test_server_html_never_contains_access_token(self):
        token = "private-access-token-do-not-serve"
        html = render_live_dashboard(token)
        self.assertNotIn(token, html)
        self.assertIn("new URLSearchParams(location.hash.slice(1)).get('token')", html)
        self.assertIn("history.replaceState(null,'',location.pathname+location.search)", html)
        self.assertIn("'Authorization':'Bearer '+sessionToken", html)

    def test_initial_state_is_json_not_executable_markup(self):
        state = {"tasks": [{"question": "</script><script>alert('unsafe')</script>&"}]}
        html = render_live_dashboard("", state)
        self.assertNotIn(state["tasks"][0]["question"], html)
        payload = re.search(r'<script id="bootstrap" type="application/json">(.*?)</script>',
                            html, re.S).group(1)
        self.assertEqual(json.loads(payload), {"state": state})
        self.assertNotIn("innerHTML", html)

    def test_visible_ui_has_bounded_forms_and_no_fake_metrics(self):
        html = render_live_dashboard("")
        self.assertEqual(html.count('maxlength="8000"'), 2)
        for identifier in ("question", "topic", "run-task", "save-decision", "activity",
                           "result", "memory-list", "trials", "connection"):
            self.assertIn('id="' + identifier + '"', html)
        self.assertIn("queue_time", html)
        self.assertIn("compute_time", html)
        self.assertIn("미측정", html)
        self.assertIn("Synthetic demo", html)

    def test_research_flow_is_icon_first_and_details_are_optional(self):
        html = render_live_dashboard("")
        self.assertIn("<h2>연구 흐름</h2>", html)
        self.assertIn("grid-template-columns:1fr", html)
        for identifier in ("state-question", "state-hypothesis", "state-model",
                           "state-result", "state-decision"):
            self.assertIn('id="' + identifier + '"', html)
        self.assertIn("근거", html)
        self.assertNotIn("trajectory", html.lower())

    def test_question_anchor_and_memory_selection_survive_refresh(self):
        html = render_live_dashboard("")
        self.assertIn('id="memory-question-anchor"', html)
        self.assertIn("Research Question", html)
        self.assertIn("let selectedMemoryId=null", html)
        self.assertIn("function traceContextIds(trace,nodes)", html)
        self.assertIn("ids=traceContextIds(trace,all)", html)

    def test_model_design_is_readable_and_question_falls_back_to_active_report(self):
        html = render_live_dashboard("")
        self.assertIn("experiment?.report?.question", html)
        self.assertIn("function modelSummary(plan,result)", html)
        self.assertIn("dependent_variable:", html)
        self.assertIn("identification_assumptions:", html)
        self.assertIn("새 연구의 Research Question", html)
        self.assertIn("새 연구 프로젝트의 질문을 입력하세요", html)
        self.assertIn("Correlation test, DiD, IV/2SLS, RDD", html)
        self.assertIn("실제 context 포함", html)

    def test_failed_connection_has_timeout_and_disables_writes(self):
        html = render_live_dashboard("")
        self.assertIn("setTimeout(()=>abort.abort(),5000)", html)
        self.assertIn("if(inFlight||document.hidden)return", html)
        self.assertIn("$('save-decision').disabled=posting||!connected||busy", html)
        self.assertIn("selectedTask=state.active_task_id||tasks[0]?.id", html)

