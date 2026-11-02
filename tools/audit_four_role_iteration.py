"""Four-role TSV audit entrypoint with four-role ownership and handoff checks."""
import audit_tsv_iteration as auditor
from audit_memory_iteration import OWNERS

auditor.BASE = auditor.ROOT / "workspaces/four-role-20261023"
auditor.EXPECTED_ACTORS = {**OWNERS, "ResultAssessment": "runner"}
auditor.REQUIRED_SOURCE_BASENAMES = {*auditor.REQUIRED_SOURCE_BASENAMES,
                                    "memory.py", "orchestration.py", "four_role_discovery.py",
                                    "four_role_memory_iteration.py"}
auditor.RETRIEVAL_ROLES = (("hypothesis", "hypothesis"), ("empiricist", "analysis"))
auditor.EXPECTED_TOPOLOGY = "orchestrator-hypothesis-empiricist-runner"
auditor.REQUIRE_WORKFLOW = True
auditor.REQUIRE_SAMPLE_RECIPE = True
EXPECTED_ACTORS = auditor.EXPECTED_ACTORS
RETRIEVAL_ROLES = auditor.RETRIEVAL_ROLES
REQUIRE_WORKFLOW = auditor.REQUIRE_WORKFLOW
REQUIRE_SAMPLE_RECIPE = auditor.REQUIRE_SAMPLE_RECIPE

if __name__ == "__main__":
    auditor.main()
