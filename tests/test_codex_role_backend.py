"""Adapter contract tests; execute only in the reviewed Scheduler QA allocation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from codex_role_backend import CodexRoleBackend, root_wire_schema, failure_category
from private_inference import AllocationDeadlineExceeded, ContextCapacityError, RuntimeIsolationError


class CodexRoleBackendTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Scheduler allocation required for tests")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.binary = self.root / "codex"
        self.binary.touch()
        self.auth = self.root / "auth"
        self.auth.mkdir()
        (self.auth / "auth.json").write_text("synthetic credential fixture")
        self.env = patch.dict(os.environ, {"RESEARCH_AGENT_ALLOW_CHILD_CLEANUP": "1",
            "OPENAI_API_KEY": "must-not-inherit", "CODEX_API_KEY": "must-not-inherit"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def backend(self):
        with patch("codex_role_backend.shutil.which", return_value="/usr/bin/bwrap"):
            return CodexRoleBackend(self.root / "backend", time.monotonic() + 60,
                                    model="test-model", codex_binary=self.binary, auth_home=self.auth)

    def test_login_node_rejected_before_files_or_processes(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeIsolationError):
                CodexRoleBackend(self.root / "never-created", 1, model="test")
        self.assertFalse((self.root / "never-created").exists())

    def test_only_current_call_and_auth_are_mounted_not_repo_or_old_history(self):
        backend = self.backend()
        folder = backend._folder("fixture")
        command = backend._sandbox(folder)
        self.assertIn("--unshare-pid", command)
        self.assertNotIn("--unshare-net", command)
        self.assertNotIn(str(self.root), command)
        self.assertNotIn(str(self.auth), command)
        self.assertIn(str(self.auth / "auth.json"), command)
        tmpfs_targets = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--tmpfs"]
        for target in ("/tmp", "/codex-home/tmp", "/codex-home/memories"):
            self.assertIn(target, tmpfs_targets)
        self.assertNotIn("/codex-home/plugins",tmpfs_targets)
        self.assertNotIn("/codex-home/skills",tmpfs_targets)
        self.assertFalse((folder/'home').exists())
        self.assertIn(str(backend.home),command)
        self.assertNotIn("/work", tmpfs_targets)
        cli = backend._cli()
        self.assertIn('forced_login_method="chatgpt"', cli)
        self.assertIn("features.shell_tool=false", cli)
        self.assertIn('web_search="disabled"', cli)
        self.assertIn('memories.use_memories=false',cli)
        self.assertIn('memories.generate_memories=false',cli)
        self.assertIn('sqlite_home="/tmp/codex-state"',cli)

    def test_call_work_separate_but_run_profile_shared(self):
        backend=self.backend()
        first,second=backend._folder('first'),backend._folder('second')
        for folder in (first,second):
            command=backend._sandbox(folder)
            self.assertIn(str(backend.home),command)
            self.assertIn(str(folder/'work'),command)
        self.assertNotIn(str(first/'work'),backend._sandbox(second))

    def test_preflight_rejects_api_auth(self):
        backend = self.backend()
        with patch.object(backend, "_run", side_effect=[(0, "", ""), (0, "API key login", "")]):
            with self.assertRaisesRegex(RuntimeIsolationError, "account authentication"):
                backend.__enter__()

    def test_preflight_reports_initiating_filesystem_error_before_login(self):
        backend = self.backend()
        with patch.object(backend, "_run", return_value=(1, "", "bwrap: Unknown option --clearenv")) as run:
            with self.assertRaisesRegex(RuntimeIsolationError, "Unknown option"):
                backend.__enter__()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(backend.last_metadata["failure_category"], "filesystem_preflight")

    def test_each_call_is_fresh_receives_complete_prompt_and_records_usage(self):
        backend = self.backend()
        prompts, commands = [], []
        def run(command, prompt):
            commands.append(command)
            prompts.append(prompt)
            output = backend.directory / ("call-%04d" % backend.calls) / "work/result.json"
            output.write_text('{"answer":"fixture"}')
            return 0, json.dumps({"type": "turn.completed", "usage": {"input_tokens": 50}}), ""
        with patch.object(backend, "_run", side_effect=run):
            for n in range(2):
                value, metadata, content = backend.complete("all evidence " * 100, {"type": "object"}, 10, n)
                self.assertEqual(value, {"answer": "fixture"})
                self.assertEqual(metadata["usage"], [{"input_tokens": 50}])
                self.assertFalse(metadata["seed_supported"])
                self.assertFalse(metadata["context_truncated_by_adapter"])
        self.assertEqual(prompts, ["all evidence " * 100] * 2)
        self.assertTrue(all("--ephemeral" in c and "resume" not in c for c in commands))
        self.assertTrue((backend.directory / "call-0001/work/result.json").exists())
        self.assertTrue((backend.directory / "call-0002/work/result.json").exists())

    def test_native_tool_events_are_rejected_not_silently_accepted(self):
        backend = self.backend()
        event = {"type": "item.completed", "item": {"type": "command_execution", "command": "fixture"}}
        with patch.object(backend, "_run", return_value=(0, json.dumps(event), "")):
            with self.assertRaisesRegex(RuntimeIsolationError, "Unexpected native"):
                backend.complete("prompt", {}, 10, 1)
        self.assertEqual(backend.last_metadata["failure_type"], "RuntimeIsolationError")

    def test_error_event_is_recorded_and_does_not_mask_a_valid_response(self):
        backend = self.backend()
        def run(command, prompt):
            (backend.directory / "call-0001/work/result.json").write_text('{"answer":"ok"}')
            return 0, json.dumps({"type": "item.completed", "item": {"type": "error"}}), ""
        with patch.object(backend, "_run", side_effect=run):
            value, metadata, _ = backend.complete("prompt", {"type": "object"}, 10, 1)
        self.assertEqual(value["answer"], "ok")
        self.assertEqual(metadata["error_event_count"], 1)

    def test_context_error_preserved_without_retry_or_raw_error_metadata(self):
        backend = self.backend()
        with patch.object(backend, "_run", return_value=(1, "", "context_length_exceeded sensitive prompt")) as run:
            with self.assertRaises(ContextCapacityError):
                backend.complete("prompt", {}, 10, 1)
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("sensitive", json.dumps(backend.last_metadata))
        self.assertEqual(backend.last_content, "")

    def test_timeout_cleans_only_owned_child_group_and_excludes_api_credentials(self):
        backend = self.backend()
        process = Mock(pid=987654, returncode=-9)
        process.poll.return_value = None
        process.communicate.side_effect = [subprocess.TimeoutExpired(["fixture"], 1), ("", "")]
        with patch("codex_role_backend.subprocess.Popen", return_value=process) as launch, \
             patch("codex_role_backend.os.killpg") as kill:
            with self.assertRaisesRegex(RuntimeIsolationError, "timed out"):
                backend._run(["fixture"], "prompt")
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertNotIn("OPENAI_API_KEY", launch.call_args.kwargs["env"])
        self.assertNotIn("CODEX_API_KEY", launch.call_args.kwargs["env"])
        self.assertEqual(launch.call_args.kwargs["env"], {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        self.assertEqual(kill.call_args.args[0], 987654)

    def test_expired_allocation_does_not_launch(self):
        backend = self.backend()
        backend.deadline = time.monotonic() - 1
        with patch("codex_role_backend.subprocess.Popen") as launch:
            with self.assertRaises(AllocationDeadlineExceeded):
                backend._run(["fixture"])
        launch.assert_not_called()

    def test_default_call_uses_allocation_deadline_without_separate_time_cap(self):
        backend = self.backend()
        self.assertIsNone(backend.call_timeout)
        backend.deadline = 10000
        process = Mock(returncode=0)
        process.communicate.return_value = ("", "")
        process.poll.return_value = 0
        with patch("codex_role_backend.time.monotonic", return_value=1000), \
             patch("codex_role_backend.subprocess.Popen", return_value=process):
            backend._run(["fixture"], "prompt")
        self.assertEqual(process.communicate.call_args.kwargs["timeout"], 9000)

    def test_wire_conversion_preserves_original_tuple_and_action_validation(self):
        from jsonschema import Draft202012Validator
        schema = {"anyOf": [
            {"type": "object", "properties": {"action": {"const": "retrieve_memory"},
                "answer": {"type": "null"}}, "required": ["action", "answer"], "additionalProperties": False},
            {"type": "object", "properties": {"action": {"const": "answer"},
                "answer": {"type": "array", "prefixItems": [{"const": "current"}, {"const": "prior"}],
                           "minItems": 2, "maxItems": 2}}, "required": ["action", "answer"], "additionalProperties": False}]}
        wire = root_wire_schema(schema)
        self.assertEqual(wire["type"], "object")
        self.assertNotIn("prefixItems", json.dumps(wire))
        mixed = {"action": "retrieve_memory", "answer": ["prior", "current"]}
        self.assertFalse(Draft202012Validator(wire).is_valid({"response": mixed}))
        self.assertFalse(Draft202012Validator(schema).is_valid(mixed))
        valid = {"action": "retrieve_memory", "answer": None}
        self.assertTrue(Draft202012Validator(wire).is_valid({"response": valid}))
        self.assertTrue(Draft202012Validator(schema).is_valid(valid))

    def test_original_schema_violation_fails_closed(self):
        backend = self.backend()
        def run(*args):
            (backend.directory / "call-0001/work/result.json").write_text('{"answer":"too long"}')
            return 0, "", ""
        with patch.object(backend, "_run", side_effect=run):
            with self.assertRaisesRegex(RuntimeIsolationError, "original schema"):
                backend.complete("prompt", {"type": "object", "properties": {
                    "answer": {"type": "string", "maxLength": 2}}}, 10, 1)
        self.assertEqual(backend.last_metadata["failure_category"], "original_schema_validation")

    def test_safe_error_categories_distinguish_actionable_roots(self):
        for text, expected in (("invalid_json_schema secret", "output_schema"),
                ("rate_limit_exceeded", "account_usage_limit"),
                ("model not supported", "model_unavailable"),
                ("401 unauthorized", "account_authentication"),
                ("rename auth.json permission denied", "auth_refresh_write_blocked")):
            self.assertEqual(failure_category(text), expected)
