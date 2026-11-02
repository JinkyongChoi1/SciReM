import json
from unittest.mock import patch
from test_codex_role_backend import CodexRoleBackendTests
from codex_role_backend import safe_cli_diagnostic
from private_inference import RuntimeIsolationError, ContextCapacityError

class CliDiagnosticTests(CodexRoleBackendTests):
    def test_cli_character_limit_rejects_before_any_model_request(self):
        backend=self.backend()
        with patch.object(backend,'_run') as run:
            with self.assertRaises(ContextCapacityError):
                backend.complete('x'*1048577,{'type':'object'},10,1)
        run.assert_not_called()
        self.assertFalse(backend.last_metadata['completion_submitted'])
        self.assertEqual(backend.last_metadata['prompt_chars'],1048577)
        self.assertEqual(backend.last_metadata['failure_category'],'cli_input_capacity')

    def test_transport_limit_counts_characters_not_utf8_bytes(self):
        backend=self.backend()
        def success(*args):
            (backend.directory/'call-0001/work/result.json').write_text('{}')
            return 0,'',''
        with patch.object(backend,'_run',side_effect=success):
            backend.complete('é'*600000,{'type':'object'},10,1)
        self.assertEqual(backend.last_metadata['prompt_chars'],600000)
        self.assertEqual(backend.last_metadata['prompt_bytes'],1200000)

    def test_redaction_and_failure_preservation(self):
        backend=self.backend()
        message='fixture startup failure bearer SECRET access_token="TOKEN" sk-fixture123 https://hooks.slack.com/services/fixture/secret'
        with patch.object(backend,'_run',return_value=(1,'',message)):
            with self.assertRaises(RuntimeIsolationError):
                backend.complete('synthetic',{'type':'object','properties':{}},10,1)
        diagnostic=backend.last_metadata['cli_diagnostic']
        for secret in ('SECRET','TOKEN','sk-fixture123','services/fixture'):
            self.assertNotIn(secret,diagnostic)
        self.assertIn('startup failure',diagnostic)
        self.assertEqual(backend.last_metadata['failure_boundary'],'model_execution')
        self.assertEqual(backend.last_metadata['prompt_bytes'],9)
        saved=backend.directory/'call-0001/work/cli-error.json'
        self.assertEqual(json.loads(saved.read_text())['diagnostic'],diagnostic)
        self.assertEqual(saved.stat().st_mode&0o777,0o600)
        self.assertLessEqual(len(safe_cli_diagnostic('x'*5000)),2500)
