"""The reviewed three-round TSV series, reusing the finite memory-only runner."""
import memory_iteration as runner

if __name__ == "__main__":
    runner.ITERATIONS = runner.ROOT / "workspaces/tsv-memory-iterations-20261023"
    runner.TRAJECTORIES = runner.ITERATIONS / "trajectories"
    runner.MAX_ROUNDS = 3
    runner.MEMORY_FORMAT = "tsv"
    runner.main()
