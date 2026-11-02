# Release status

Visual-fidelity follow-up: the live canvas now uses the bundled Noto Sans font,
larger node labels, score pills sourced from recorded PageRank, a compact RQ
header and legend below the graph. Five separate Firefox controller/graph QA
passes verified font availability, label bounds, source-backed scores and the
existing memory/review contracts; an additional 1536px browser pass and mobile
checks were run. These checks use synthetic records and scripted agent replies.
The supplied concept remains an illustration: graph content, node positions,
and the final font rendering can vary with the run, viewport and environment.
Standalone offline snapshots fall back to an available system font if the
bundled font asset is not available alongside the viewer service.

Follow-up visual QA: the selected concept and the actual Firefox screenshot were
compared, and the canvas palette, query chip and review dock were refined. A
Scheduler-hosted browser test verified the same published graph inventory,
recorded anchors and PageRank, unconfirmed-delivery handling, evidence links and
human-review receipts. The local live smoke used scripted agent responses and
synthetic data with actual R execution. The concept image is illustrative;
actual graph layout and visible records depend on the selected research run.

Memory-canvas update (2026-11-02): the implemented dashboard follows the selected
canvas-and-dock design in the README. It shows the complete selected store,
source-recorded anchor/PPR/context overlays, source-linked Decision edges,
AI activity receipts and separate human review annotations. Relevance and full
evidence expand on demand. It preserves offline history and explicit stale or
disconnected states, without inventing activity or relevance scores.

Three engineering QA rounds completed:

| Round | Verification | Outcome |
|---|---|---|
| 1 | 397 unit tests + 12 handoff tests, actual-R synthetic smoke, real Firefox UI and authenticated human-review persistence | Passed; screenshot review identified layout issues repaired before round 2 |
| 2 | 402 unit tests + 12 handoff tests, non-SGE CLI and actual-R synthetic smoke, revised UI and local live service | Passed; full Python source parity checked when reusing already-passing checks |
| 3 | Install package to a separate target, run installed CLI from a clean directory, validate installed-engine browser behavior and public local smoke command | Passed |

Browser checks covered ten actual controller stages through scripted cycles,
exact scores, inactive stored records, source selection, Decision provenance,
human review persistence, unauthorized writes, offline states and live-service
shutdown. Human reviews remain **recorded / not_applied**; no agent uptake or
scientific validity is implied. No LLM inference or empirical research data were
used. These are implementation tests, not evidence of research-performance gains.

Non-SGE validation used a Linux UTS namespace with a non-cluster hostname and no
`JOB_ID`, inside an actual allocated compute job. Firefox ran outside that nested
namespace and accessed the service over loopback; its sandbox was not disabled.
Failed temporary-directory/nested-browser tests and a test interaction with an
open drawer were retained and corrected. Installed-engine imports were verified
to come from the separate installation; the controller/test fixtures still came
from the source checkout and existing system dependencies were reused. This is
not a fresh-machine dependency-installation test. Mac/Windows, site-specific
tunnels and the original empirical launcher's non-SGE portability remain
unverified. The canvas prioritizes desktop use; its mobile graph is scaled down.
The observer is not a publicly hosted service.

A final receipt audit found that older retrieval traces without a delivery flag
could appear delivered. The viewer now requires explicit positive confirmation;
missing and negative receipts remain unconfirmed while the packet, anchors and
PPR stay inspectable. Browser regressions verify all three states. This closing
repair changed only the viewer; the other Python files were verified identical
to the installed-engine test snapshot before reusing its evidence.

Prepared 2026-11-01 from the current experimental code for https://github.com/JinkyongChoi1/SciReM. The author selected public posting without a code reuse license; reuse permissions remain undecided. Third-party font license remains included. This prototype is not a complete rights-cleared empirical replication package.

Validation results will be recorded here after checking the exported tree. Packaging tests use isolated source and an environment with existing system Python dependencies; this is not proof that all dependencies install on every clean machine. The scripted smoke uses the real controller, memory graph and retrieval with fixture LLM/statistical outputs. No new model calls or real-data experiment are included in release QA.

The public-tree scan checks tracked file contents for obvious secrets, institutional paths and excluded data/runtime artifacts. Such checks are bounded checks, not a guarantee against every disclosure. Scientific benchmark data access/citation, human judgment revision integration, native-B behavior, non-SGE deployment and live-browser access remain outside this release verification.

Verified release QA: isolated source installation succeeded;367 discovered tests plus12 separate diagnostic-handoff tests passed (379 total). Both the fixture smoke and actual sandboxed-R/synthetic-data smoke completed two research cycles through the exported controller and shared graph. The real-R mode used existing site R packages; no model inference or empirical research data were used. A source-content scan of84 files found no matching private host paths, obvious key/token/webhook patterns or excluded runtime directories. Test-only fake redaction inputs were retained. No test was removed to obtain the pass.

First packaging QA failed on omitted audit dependencies and a synthetic fixture. These were restored from the actual codebase, after which the exported suite passed. Package installation used a new virtual environment with system site packages and no dependency downloads; fully independent dependency installation, another institution's environment and browser/live-server access remain unverified.

Judgment-edge update (2026-11-01): the expanded suite passed374 tests plus12 separate handoff tests (386 total), including7 new judgment-edge/auditor tests. The actual R/synthetic-data two-cycle smoke also passed. The first expanded QA exposed7 legacy-auditor failures; the auditor was extended with independent semantic recomputation and no tests were weakened. Production and release copies of the changed graph/TSV/auditor code match. This verifies implementation and source fidelity, not improved research performance or human-review effectiveness. Publication still awaits GitHub connection/account and license selection.

Publication decision update: the public repository `JinkyongChoi1/SciReM` exists and write access is confirmed. The author explicitly authorized public posting without a code license, with reuse permissions to be decided later. Earlier statements that connection/account/license choice was pending describe the preceding preparation state.
