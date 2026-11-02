# Paper–artifact alignment

SciReM's proposed contribution is to preserve the evidential basis of scientific judgments so that agents and human collaborators can inspect and revise them. Neither more agents nor PageRank alone is the claimed novelty. This prototype exports the existing experimental implementation; it is not a new implementation of the entire proposed vision.

| Paper concept | Current implementation | Boundary / proposed work |
|---|---|---|
| Cumulative research memory | Immutable source-linked ResearchQuestion, Hypothesis, AnalysisModel, ResultAssessment, Decision records in `src/research_memory/two_role.py` | Agent interpretations are fallible; an archive is not necessarily active model context. |
| Link a decision to evidence | `based_on`, `produced_by`, `intended_test_for`, `addresses`, `revision_of`, `follows_decision`; Decision basis keywords and next inquiry; independent `supports`, `challenges`, `assumption_constraint` edges | Support/challenge concerns the declared primary statistical test. Assumption edges preserve declarations, not validation. General methodological-concern nodes and human-confirmation revisions remain future work. |
| Distinguish evidence from interpretation | `evidence.py` preserves machine results; focal statistical checks computed from declared test/actual output; model assessments stored separately | Statistical significance at the configured alpha is not causal validity, novelty or overall scientific support. |
| Retrieve connected context | `memory.py` and `two_role.py`: query anchors, personalized PageRank, recorded selection and TSV context | Latest loops support lexical anchors without embeddings; semantic backend optional. Ranking is relevance, not truth. PPR benefit unestablished. |
| Adaptive research loop | Existing `four_role_discovery.py` controller, unified/split roles, Runner, decisions, optional review and repair | A declared revision is not evidence it was executed. Neither extra roles nor LLM review is established to improve science. |
| Human inspectability | Existing artifact/live viewers, sources and retrieval traces; generic workflow records user decisions | Generic user-decision storage is not a fully integrated human-confirmed revision loop in the typed scientific graph. No controlled human study yet. |
| Reusable infrastructure | Export of current harness, synthetic smoke, tests, configurable local paths | Pilot data dictionary/formula support remains task-specific. Real model/data/runtime access needed for empirical reruns. |
| Better scientific decisions / less repetition | Feasibility and failure observations in `preliminary-evidence.md` | Hypotheses for future evaluation, not demonstrated treatment effects. |

The scientific graph currently stores compact decision bases rather than complete structured human rationale. The proposal should promise to extend and evaluate this capability, not state that it is already validated. Do not interpret `status=proposed`, provenance hashes, successful workflow completion, or a high retrieval score as human approval or scientific truth.
