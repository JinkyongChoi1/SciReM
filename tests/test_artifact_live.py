"""Read-only live observation must reflect recorded evidence, not invented progress."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from research_memory.live import ArtifactController, create_server


class ArtifactControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trajectory = self.root / 'trajectory'
        self.trajectory.mkdir()
        self.report_path = self.root / 'report.json'
        self.report = {'trajectory_root': str(self.trajectory), 'status': 'running',
                       'question': 'Synthetic research question', 'synthetic_only': True,
                       'active_attempt': {'attempt': 1}, 'active_stage': 'hypothesis',
                       'attempts': [], 'model_calls': []}
        self.publish()

    def publish(self):
        temporary = self.report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report))
        temporary.replace(self.report_path)

    def controller(self):
        return ArtifactController(self.report_path, self.trajectory)

    def test_new_report_is_observed_without_controller_restart(self):
        controller = self.controller()
        first = controller.state()
        self.assertFalse(first['capabilities']['agent_run'])
        self.assertEqual(first['experiment']['report']['active_stage'], 'hypothesis')
        self.report['active_stage'] = 'runner'
        self.publish()
        self.assertEqual(controller.state()['experiment']['report']['active_stage'], 'runner')
        self.report.update(status='completed', active_attempt=None, active_stage=None,
                           attempts=[{'attempt': 1, 'status': 'completed'}])
        self.publish()
        self.assertEqual(controller.state()['experiment']['status'], 'completed')

    def test_exact_source_trace_and_model_specification_survive(self):
        folder = self.trajectory / 'attempt-0001'
        folder.mkdir()
        trace = {'nodes': [{'memory_id': 'h-1', 'memory_type': 'Hypothesis',
                            'anchor_weight': .4, 'ppr': .8}],
                 'edges': [], 'selected_ids': ['h-1'], 'brief': 'ID\ttype\nh-1\tHypothesis'}
        (folder / 'hypothesis-retrieval.json').write_text(json.dumps(trace))
        spec = {'design_type': 'observational_association', 'estimator': 'OLS',
                'dependent_variable': 'y', 'focal_predictor': 'x'}
        (folder / 'plan.json').write_text(json.dumps({'analysis': {'family': 'linear'},
                                                    'model_specification': spec}))
        state = self.controller().state()['experiment']
        self.assertEqual(state['traces']['Hypothesis'], trace)
        self.assertEqual(state['artifacts']['plan']['model_specification'], spec)

    def test_old_running_report_is_stale(self):
        old = time.time() - 601
        os.utime(self.report_path, (old, old))
        self.assertEqual(self.controller().state()['experiment']['report']['freshness'], 'stale')

    def test_report_cannot_redirect_to_unapproved_trajectory(self):
        controller = self.controller()
        self.report['trajectory_root'] = str(self.root / 'outside')
        self.publish()
        self.assertEqual(controller.state()['experiment']['status'], 'error')

    def test_explicit_episode_evidence_uses_execution_attempt_identity(self):
        folder = self.trajectory / 'episode-0007' / 'attempt-0001'
        folder.mkdir(parents=True)
        (folder / 'hypothesis.json').write_text(json.dumps({'hypothesis': 'Current independent episode'}))
        self.report['active_attempt'] = {'attempt': 7, 'execution_attempt': 1,
                                         'artifact_directory': 'episode-0007/attempt-0001'}
        self.publish()
        observed = self.controller().state()['experiment']
        self.assertEqual(observed['artifacts']['hypothesis']['hypothesis'], 'Current independent episode')
        self.assertEqual(observed['report']['active_attempt']['execution_attempt'], 1)
        self.report.update(status='completed', active_attempt=None, active_stage=None,
                           attempts=[{'attempt': 1, 'episode': 7, 'status': 'completed',
                                      'artifact_directory': 'episode-0007/attempt-0001'}])
        self.publish()
        self.assertEqual(self.controller().state()['experiment']['artifacts']['hypothesis']['hypothesis'],
                         'Current independent episode')

    def test_completed_resume_shows_its_saved_evidence(self):
        folder = self.trajectory / 'attempt-0001' / 'resume-42'
        folder.mkdir(parents=True)
        (folder / 'decision.json').write_text(json.dumps({'action': 'revise', 'basis_keywords': 'Resume evidence'}))
        self.report.update(status='completed', active_attempt=None, active_stage=None, attempts=[],
                           resumptions=[{'attempt': 1, 'status': 'completed',
                                         'artifact_directory': 'attempt-0001/resume-42'}])
        self.publish()
        self.assertEqual(self.controller().state()['experiment']['artifacts']['decision']['basis_keywords'],
                         'Resume evidence')

    def test_explicit_artifact_path_cannot_escape_selected_run(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (self.trajectory / 'linked').symlink_to(outside, target_is_directory=True)
        for path in ('../outside', str(outside), 'linked'):
            with self.subTest(path=path):
                self.report['active_attempt'] = {'attempt': 1, 'artifact_directory': path}
                self.publish()
                self.assertEqual(self.controller().state()['experiment']['status'], 'error')

    def test_symlinked_attempt_and_trace_are_not_followed(self):
        folder = self.trajectory / 'attempt-0001'
        outside = self.root / 'outside'
        outside.mkdir()
        folder.symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.controller().state()['experiment']['status'], 'error')
        folder.unlink()
        folder.mkdir()
        target = outside / 'private.json'
        target.write_text(json.dumps({'private': 'must-not-be-returned'}))
        (folder / 'hypothesis-retrieval.json').symlink_to(target)
        state = self.controller().state()
        self.assertEqual(state['experiment']['status'], 'error')
        self.assertNotIn('must-not-be-returned', json.dumps(state))

    def test_real_http_requires_token_and_refuses_writes(self):
        server = create_server(self.controller(), token='test-only-session-token-123456789')
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = 'http://127.0.0.1:' + str(server.server_port)
            html = urlopen(base).read().decode()
            self.assertNotIn(server.token, html)
            with self.assertRaises(HTTPError) as missing:
                urlopen(base + '/api/state')
            self.assertEqual(missing.exception.code, 403)
            headers = {'Authorization': 'Bearer ' + server.token}
            response = json.loads(urlopen(Request(base + '/api/state', headers=headers)).read())
            self.assertEqual(response['experiment']['report']['question'], self.report['question'])
            headers.update({'Origin': base, 'Content-Type': 'application/json'})
            with self.assertRaises(HTTPError) as write:
                urlopen(Request(base + '/api/task', data=b'{"track_id":"FD","question":"do not create"}', headers=headers))
            self.assertIn(write.exception.code, (403, 405))
            self.assertFalse((self.trajectory / 'tasks').exists())
            with self.assertRaises(HTTPError) as host:
                urlopen(Request(base + '/api/state', headers={'Authorization': 'Bearer ' + server.token, 'Host': 'attacker.example'}))
            self.assertEqual(host.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
