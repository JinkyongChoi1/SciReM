# Minimal export changes

This tree copies the experiment controller used in the development research runs.
It does not replace that controller with a demo. Unchanged files retain their
experimental implementation, including historical entrypoints and names.

Release portability changes:

- `tools/discovery_pilot.py`: `SCIREM_DATA`, `SCIREM_R_HOME`,
  `SCIREM_R_LIBRARY`, and `SCIREM_LLAMA_RUNTIME` replace institution-specific paths.
- `tools/validate_local_synthetic.py`: `SCIREM_MODEL_WEIGHTS` replaces the private
  weight path. Weights and binaries are not distributed.
- `tools/run_scirem.py`: a small entrypoint configures the existing controller's
  output root, unified/split roles and on/off independent Orchestrator review.
  Default unified/no-review matches the latest loop validation. Review-off does
  not disable numerical validation or the researcher's own assessment.
- `pyproject.toml`: optional `experiment` dependencies include `jsonschema`.

Primary tools remain dataset-specific: `styles.csv`, the English question,
provided variable dictionary, permitted terms and tested analysis families.
`SCIREM_DATA` selects a file with that schema; it does not make an arbitrary CSV
compatible. No hidden reference, raw data, account credentials, individual-run
histories, scheduler logs or model weights are included. A new research domain
requires deliberate edits to its dictionary, plan schema and analysis tool.

The hosted backend uses fresh isolated Codex CLI calls with harness-supplied role
history. It is not a native continued Codex conversation. The legacy A/B/C
entrypoint is retained for code provenance; its B condition is not certified as
native B and should not be used for the proposed native-session comparison.
The recommended release entrypoint is memory-enabled C only. No performance
advantage is implied by artifact execution.

Current deployment is Linux with SGE, bubblewrap, Python >=3.10, R and needed
R packages. The release retains compute-allocation and isolation checks; do not
invent JOB_ID to bypass them. Local synthetic unit tests do not require a model
or research dataset. Other schedulers/platforms are not validated.

After the initial export, the production graph and this release both gained independent
`supports`, `challenges`, and `assumption_constraint` edges with source-derived
metadata and TSV delivery. This is a versioned scientific representation change,
not a portability-only edit or a retroactive change to earlier experiment results.

Release smoke has an optional `--real-statistics` mode: the same controller uses
`run_analysis` in the actual R sandbox on generated synthetic rows, while model
responses remain scripted. It adds no new scientific model or production loop.
The independent TSV auditor was updated with the same relation contract and
separate source-based semantic recomputation. Existing legacy-archive checks
remain; metadata is not accepted solely because an edge name is allowed.

Live-dashboard release repair: an `ArtifactController` and `watch` command
observe explicit report/trajectory paths without the older generic controller's
private registry or historical workspace assumptions. The actual research loop is
unchanged. The smoke writes incremental execution checkpoints and optionally
serves them after the first authenticated viewer connects. The live viewer
distinguishes recorded activity, stale state, disconnection and offline snapshots;
it preserves model specification and retrieval-delivery evidence. This is an
observer/portability change, not a new research condition or performance result.
The common research runner records the exact relative artifact directory for
normal, independent-episode and resumed attempts, and checkpoints diagnostic
Runner entry. These telemetry fields prevent the viewer from selecting an old
attempt or attributing a running diagnostic to the preceding model call.

The memory canvas reads the selected trajectory's complete, verified published
snapshot and overlays its recorded retrieval. Human-review submissions write only
separate append-only annotations, marked not applied. The observer remains
unable to launch research, modify the report or publish a new scientific graph.
