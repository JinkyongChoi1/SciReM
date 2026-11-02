"""Native-token capacity policy fixtures; run only in reviewed SGE allocations."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from private_inference import PrivateInference, ContextCapacityError, RuntimeIsolationError


class ContextCapacityTests(unittest.TestCase):
    def worker(self, token_ids, response=None, reuse=True):
        worker = PrivateInference(Path("unused"), Path("unused"), "fixed", 9999999999)
        worker.fixed_tokens = 7
        worker.rendered_fixed_prefix = "fixed"
        worker.process = SimpleNamespace(pid=123)
        worker.render = Mock(side_effect=lambda content: content)
        events = ([{"n_restored": 7}] if reuse else []) + [{"tokens": token_ids}]
        if response is not None:
            events.append(response)
        events.append({})
        worker.request = Mock(side_effect=events)
        return worker

    def test_exact_budget_boundary_allows_full_rendered_prompt(self):
        worker = self.worker([1] * 4, {"content": '{"ok":true}'})
        worker.context_limit = 100
        result, metadata, _ = worker.complete("\twhole TSV\n", {}, 64, 42)
        self.assertTrue(result["ok"])
        self.assertEqual(metadata["required_tokens"], 100)
        self.assertEqual(metadata["capacity_status"], "fits")
        self.assertEqual(worker.request.call_args_list[1].args[1], {
            "content": "fixed\twhole TSV\n", "add_special": True, "parse_special": True, "with_pieces": False})
        self.assertEqual(worker.request.call_args_list[2].args[1]["prompt"], "fixed\twhole TSV\n")

    def test_overflow_is_not_transport_failure_and_never_generates_or_trims(self):
        worker = self.worker([1] * 5)
        worker.context_limit = 100
        with self.assertRaises(ContextCapacityError):
            worker.complete("FULL_EVIDENCE", {}, 64, 42)
        self.assertFalse(worker.failed)
        self.assertEqual([call.args[0] for call in worker.request.call_args_list],
                         ["/slots/0?action=restore", "/tokenize", "/slots/0?action=erase"])
        self.assertEqual(worker.last_metadata["required_tokens"], 101)
        self.assertEqual(worker.last_metadata["capacity_status"], "exceeded")
        self.assertFalse(worker.last_metadata["completion_submitted"])
        self.assertNotIsInstance(ContextCapacityError(), RuntimeIsolationError)

    def test_capacity_uses_actual_rendering_not_dynamic_text_length(self):
        worker = self.worker([1] * 10, {"content": '{"ok":true}'})
        worker.render = Mock(return_value="fixedACTUAL_CHAT_TEMPLATE_AND_FULL_TSV_SUFFIX")
        _, metadata, _ = worker.complete("short", {}, 64, 42)
        token_payload = worker.request.call_args_list[1].args[1]
        self.assertEqual(token_payload["content"], "fixedACTUAL_CHAT_TEMPLATE_AND_FULL_TSV_SUFFIX")
        self.assertEqual(metadata["rendered_prompt_tokens"], 10)

    def test_native_tokenizer_malformed_response_fails_closed(self):
        for tokens in (None, "not-token-ids", [True], [-1], ["1"]):
            worker = self.worker(tokens)
            with self.subTest(tokens=tokens), self.assertRaises(RuntimeIsolationError):
                worker.complete("whole", {}, 64, 42)
            self.assertTrue(worker.failed)
            self.assertNotIn("/completion", [call.args[0] for call in worker.request.call_args_list])

    def test_valid_json_from_truncated_context_is_rejected_and_metadata_preserved(self):
        worker = self.worker([1], {"content": '{"scientific_result":"looks valid"}', "truncated": True,
                                  "stop_type": "eos", "timings": {"predicted_n": 12}})
        with self.assertRaises(ContextCapacityError):
            worker.complete("whole", {}, 64, 42)
        self.assertEqual(worker.last_content, '{"scientific_result":"looks valid"}')
        self.assertEqual(worker.last_metadata["capacity_status"], "server_reported_truncation")
        self.assertEqual(worker.last_metadata["timings"]["predicted_n"], 12)
        self.assertEqual(worker.request.call_args_list[-1].args[0], "/slots/0?action=erase")
        self.assertFalse(worker.failed)

    def test_uncached_requests_still_receive_native_capacity_precheck(self):
        worker = self.worker([1] * 5, reuse=False)
        worker.context_limit = 100
        with self.assertRaises(ContextCapacityError):
            worker.complete("whole", {}, 64, 42, reuse_prefix=False)
        self.assertEqual([call.args[0] for call in worker.request.call_args_list],
                         ["/tokenize", "/slots/0?action=erase"])

    def test_failed_erase_after_capacity_rejection_is_transport_failure(self):
        worker = self.worker([1] * 5)
        worker.context_limit = 100
        worker.request.side_effect = [{"n_restored": 7}, {"tokens": [1] * 5}, OSError("erase failed")]
        with self.assertRaises(RuntimeIsolationError):
            worker.complete("whole", {}, 64, 42)
        self.assertTrue(worker.failed)
        self.assertEqual(worker.last_metadata["capacity_status"], "exceeded")

    def test_completion_timeout_preserves_that_generation_request_was_submitted(self):
        worker = self.worker([1])
        worker.request.side_effect = [{"n_restored": 7}, {"tokens": [1]}, TimeoutError("completion timed out")]
        with self.assertRaises(RuntimeIsolationError):
            worker.complete("whole", {}, 64, 42)
        self.assertTrue(worker.failed)
        self.assertTrue(worker.last_metadata["completion_submitted"])
        self.assertEqual(worker.last_metadata["capacity_status"], "fits")
        self.assertEqual(worker.request.call_args_list[-1].args[0], "/completion")

    def test_startup_disables_context_shift_and_checks_effective_native_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            (runtime / "llama-server").touch()
            worker = PrivateInference(root / "worker", runtime, "fixed", 9999999999)
            worker.render = Mock(side_effect=lambda content: "CHAT:" + content)
            worker.request = Mock(side_effect=[{}, {"default_generation_settings": {"n_ctx": 16384}},
                {"tokens": [1, 2]}, {}, {"n_saved": 2}, {}])
            process = Mock()
            process.poll.return_value = None
            with patch("private_inference.subprocess.Popen", return_value=process) as popen, patch("private_inference.os.chmod"):
                worker.__enter__()
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index("-c") + 1], "32768")
                self.assertIn("--no-context-shift", command)
                self.assertEqual(command[command.index("-b") + 1], "256")
                self.assertEqual(command[command.index("-ub") + 1], "256")
                self.assertEqual(worker.context_limit, 16384)
                self.assertEqual([call.args[0] for call in worker.request.call_args_list], [
                    "/health", "/props", "/tokenize", "/completion", "/slots/0?action=save", "/slots/0?action=erase"])
                worker.__exit__()
            process.stdin.write.assert_called_once_with("cmd_router_to_child:exit\n")
            process.kill.assert_not_called()
            process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
