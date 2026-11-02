"""Local opt-in must not weaken the university Login Node execution rule."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_memory.__main__ import main
from research_memory.execution import require_execution
from research_memory.two_role import TwoRoleMemory


class PortabilityTests(unittest.TestCase):
    def test_workstation_requires_explicit_opt_in(self):
        with patch.dict(os.environ, {}, clear=True), patch("socket.gethostname", return_value="my-laptop"):
            with self.assertRaises(RuntimeError):
                require_execution()
            require_execution(local=True)
            self.assertNotIn("JOB_ID", os.environ)

    def test_known_cluster_hosts_reject_local_without_scheduler(self):
        with patch.dict(os.environ, {}, clear=True):
            for hostname in ("researchint01", "research15.grid.gsb", "gridtest.grid.gsb", "Research13."):
                with self.subTest(hostname=hostname), patch("socket.gethostname", return_value=hostname):
                    with self.assertRaises(RuntimeError):
                        require_execution(local=True)

    def test_actual_scheduler_allocation_needs_no_local_flag(self):
        if not os.environ.get("JOB_ID"):
            self.skipTest("Allocation verification is run in Scheduler QA")
        require_execution()

    def test_graph_controller_has_explicit_local_argument(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
                patch("socket.gethostname", return_value="scirem-local"):
            with self.assertRaises(RuntimeError):
                TwoRoleMemory(folder, embedding_backend="lexical")
            memory = TwoRoleMemory(folder, embedding_backend="lexical", local=True)
            self.assertEqual(memory.nodes, {})
            self.assertNotIn("JOB_ID", os.environ)

    def test_packaged_demo_runs_away_from_checkout_without_job_id(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
                patch("socket.gethostname", return_value="scirem-local"), contextlib.redirect_stdout(io.StringIO()):
            workspace = Path(folder) / "memory"
            previous = Path.cwd()
            try:
                os.chdir(folder)
                output = main(["--local", "--workspace", str(workspace), "demo", "--backend", "lexical"])
            finally:
                os.chdir(previous)
            self.assertTrue(output["synthetic"])
            self.assertTrue(Path(output["dashboard"]).is_file())
            trace = json.loads((workspace / "retrievals" / output["request_id"] / "trace.json").read_text())
            self.assertTrue(trace["nodes"])
            self.assertNotIn("dc-private", [node["memory_id"] for node in trace["nodes"]])
            self.assertNotIn("JOB_ID", os.environ)
