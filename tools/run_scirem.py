"""Configure the existing research loop; scientific execution remains unchanged."""
import os
from pathlib import Path

import four_role_discovery as flow
from four_role_memory_iteration import configure


def configure_release():
    layout = os.environ.get("SCIREM_ROLE_LAYOUT", "unified")
    review = os.environ.get("SCIREM_REVIEW", "off")
    if layout not in ("unified", "split") or review not in ("on", "off"):
        raise ValueError("SCIREM_ROLE_LAYOUT must be unified/split; SCIREM_REVIEW must be on/off")
    flow.RESEARCH_ROLE_LAYOUT = layout
    flow.ORCHESTRATOR_ASSESSMENT = review == "on"
    runner = configure()
    runner.ITERATIONS = Path(os.environ.get("SCIREM_OUTPUT", runner.ROOT / "workspaces/scirem")).expanduser().resolve()
    runner.TRAJECTORIES = runner.ITERATIONS / "trajectories"
    runner.SOURCE_FILES = (*runner.SOURCE_FILES, "tools/run_scirem.py",
                           "tools/orchestrator_review.py", "tools/decision_review.py")
    return runner


if __name__ == "__main__":
    configure_release().main()
