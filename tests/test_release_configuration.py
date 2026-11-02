"""Release-only configuration must select the actual experimental controller."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import four_role_discovery as flow
import memory_iteration
from four_role_memory_iteration import execute
from run_scirem import configure_release


class ReleaseConfigurationTests(unittest.TestCase):
    def test_paths_and_ablation_select_existing_controller(self):
        with patch.dict(os.environ, {"SCIREM_OUTPUT": "/tmp/scirem-test", "SCIREM_ROLE_LAYOUT": "split", "SCIREM_REVIEW": "on"}), \
                patch.object(flow, "RESEARCH_ROLE_LAYOUT"), patch.object(flow, "ORCHESTRATOR_ASSESSMENT"), \
                patch.multiple(memory_iteration, ITERATIONS=None, TRAJECTORIES=None, MAX_ROUNDS=3,
                               MEMORY_FORMAT="tsv", TOPOLOGY="", RESUME_RESULT=None,
                               SOURCE_FILES=memory_iteration.SOURCE_FILES, execute_attempt=None):
            runner = configure_release()
            self.assertEqual(runner.TRAJECTORIES, Path('/tmp/scirem-test/trajectories'))
            self.assertIs(runner.execute_attempt, execute)
            self.assertEqual(flow.RESEARCH_ROLE_LAYOUT, 'split')
            self.assertTrue(flow.ORCHESTRATOR_ASSESSMENT)

    def test_unknown_review_is_rejected(self):
        with patch.dict(os.environ, {"SCIREM_REVIEW": "maybe"}):
            with self.assertRaises(ValueError):
                configure_release()
