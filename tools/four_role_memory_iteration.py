"""Future Scheduler entrypoint for the tested four-role memory-only loop."""
import memory_iteration as runner
from four_role_discovery import execute_four_role_attempt
from four_role_resume import resume_result


def execute(session, index, arm, memory, model, analysis_runner, *, condition=None, **kwargs):
    if "query" in kwargs:
        if "inquiry" in kwargs:
            raise ValueError("Pass either inquiry or the legacy query field, not both")
        kwargs["inquiry"] = kwargs.pop("query")
    if condition is None:
        if arm != "memory":
            raise ValueError("Four-role memory-only entrypoint rejects non-memory arms")
        condition = "C"
    expected_arm = "memory" if condition == "C" else "no-memory"
    if arm != expected_arm:
        raise ValueError("Condition and shared-memory access policy disagree")
    return execute_four_role_attempt(session, index, memory, model, analysis_runner,
                                     condition=condition, **kwargs)


def configure():
    """Apply the isolated four-role namespace before the common runner parses CLI args."""
    runner.ITERATIONS = runner.ROOT / "workspaces/four-role-20261023"
    runner.TRAJECTORIES = runner.ITERATIONS / "trajectories"
    runner.MAX_ROUNDS = 3
    runner.MEMORY_FORMAT = "tsv"
    runner.TOPOLOGY = "orchestrator-hypothesis-empiricist-runner"
    runner.RESUME_RESULT = resume_result
    runner.SOURCE_FILES = (*runner.SOURCE_FILES, "tools/four_role_memory_iteration.py",
                           "tools/four_role_discovery.py", "tools/four_role_resume.py", "src/research_memory/orchestration.py")
    runner.execute_attempt = execute
    return runner


def configure_three_condition():
    """Configure a fresh, bounded real-data A/B/C pilot entrypoint."""
    runner.ITERATIONS = runner.ROOT / "workspaces/three-condition-benchmark-20261024"
    runner.TRAJECTORIES = runner.ITERATIONS / "trajectories"
    runner.MAX_ROUNDS = 1
    runner.MEMORY_FORMAT = "tsv"
    runner.TOPOLOGY = "orchestrator-hypothesis-empiricist-runner"
    runner.BENCHMARK_CONDITIONS = ("A", "B", "C")
    runner.RESUME_RESULT = None
    runner.SOURCE_FILES = (*runner.SOURCE_FILES, "tools/four_role_memory_iteration.py",
                           "tools/four_role_discovery.py", "src/research_memory/orchestration.py")
    runner.execute_attempt = execute
    return runner


if __name__ == "__main__":
    configure_three_condition().main()
