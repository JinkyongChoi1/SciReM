# SciReM — Scientific Research Memory

SciReM is a memory system for cumulative, inspectable human–AI social-science research. 

The aim of SciReM is to connect **evidence → interpretation → decision → next inquiry**, so earlier judgments can be inspected and reconsidered. The prototype implements a source-linked scientific graph, query-anchored retrieval with personalized PageRank, compact TSV packets, statistical execution receipts, and a research controller. Improved scientific judgment, reduced repetition, and better human review remain research questions.

**Start here:** [paper–code alignment](docs/CLAIMS.md), [technical appendix](docs/technical-appendix.md), [preliminary evidence](docs/preliminary-evidence.md), [evaluation plan](docs/evaluation.md).

**Live dashboard:** [watch an active research run](#watch-a-research-run) or [try the demo](#quick-start). The dashboard follows recorded execution and memory retrieval, with an offline view after completion.

## Dashboard preview

![SciReM memory canvas with query anchors, connected memories, AI activity and human decisions](docs/images/memory-canvas-concept.png)

## What is available 

1. **If you don't have an AI model account or dataset:** two scripted cycles through the real controller, graph publication, retrieval and dashboard generation. This validates wiring; responses and statistics are fixtures.
2. **If you have an AI model account and compatible data:** the actual memory-enabled research loop using the same controller and R Runner as the development experiments. The current tool/data dictionary is specific to the processed category-classification pilot. Setting a different CSV path does not support arbitrary datasets.

Python 3.10+ is required. The memory CLI has no mandatory third-party dependencies; the scripted research demo adds `jsonschema`. Explicit `--local` execution supports a workstation without SGE while retaining the university Login Node restriction. The original empirical launcher still requires Linux/SGE and bubblewrap; real statistical execution also needs R, `survival`, `sandwich`, `lmtest` and `fixest`. Hosted calls require an installed Codex CLI and the user's own authorized account/model access.

## Quick start

On a non-cluster computer, run the demo:

```bash
python -m pip install '.[experiment]'
python tools/smoke_scirem.py --local --serve --duration 300 \
  --output ./workspaces/local-demo-01
```

Open the `browser_url` in the connection file printed by the command. The demo starts when you connect and uses synthetic data and scripted responses. Choose a new output directory for each run.

On a cluster, run the demo inside a scheduler allocation and omit `--local`. Do not run code on a Login Node.

## Watch a research run

Inside a compute allocation, provide the run's report and its exact `trajectory_root`:

```bash
python -m research_memory --workspace /path/to/viewer watch \
  --report /path/to/run/report.json \
  --trajectory /path/to/trajectory \
  --port 8765 --duration 3600
```

Open the private `browser_url` in the printed connection file. To connect from another computer, forward the port to the compute node with SSH. For a non-cluster computer, add `--local` before `--workspace`.

The dashboard displays memory and activity. Human reviews are saved separately and do not automatically change or resume the research.

## Run the pilot

The pilot requires compatible data and a Linux/SGE allocation. See [data and runtime requirements](docs/data-and-runtime.md), then configure:

```bash
export PYTHONPATH="$PWD/src:$PWD/tools:$PWD/tests"
export SCIREM_DATA=/absolute/path/to/compatible/styles.csv
export SCIREM_R_HOME=/absolute/path/to/R/home
export SCIREM_R_LIBRARY=/absolute/path/to/R/site-library
export SCIREM_OUTPUT=/absolute/path/to/new-run-output
export SCIREM_ROLE_LAYOUT=unified
export SCIREM_REVIEW=off
export RESEARCH_AGENT_ALLOW_CHILD_CLEANUP=1

python tools/run_scirem.py \
  --round 1 --budget-seconds 300 \
  --backend codex --codex-model YOUR_ACCESSIBLE_MODEL \
  --embedding-backend lexical --adaptive-shared-context \
  --trajectory-root "$SCIREM_OUTPUT/trajectories/run-01"
```

## Tests

Inside a compute allocation:

```bash
python -m unittest discover -s tests -v
python -m unittest newtest_diagnostic_handoff -v
```

