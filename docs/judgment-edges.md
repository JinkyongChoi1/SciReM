# Independent judgment edges

The current five primary node types are unchanged. New model/result publications create independently typed edges inside the persisted graph, rather than hiding these relations only in prose.

| Relation | Origin → target | Basis and scope |
|---|---|---|
| `supports` | ResultAssessment → Hypothesis | Valid declared primary statistical test, significant at the recorded alpha, expected direction. Exploratory statistical support only. |
| `challenges` | ResultAssessment → Hypothesis | Valid significant primary effect in the opposite declared direction. Non-significance is not evidence of contradiction. |
| `assumption_constraint` | AnalysisModel → Hypothesis | Actual `model_specification.identification_assumptions` text. `declared_not_validated`; constrains interpretation, not a verified identification claim. |

Each edge retains `source`, `provenance`, `basis`, `scope`, `assertion_status` and `scientific_validity=unverified`. Result publication derives the relation from the source-grounded primary statistical assessment; the LLM does not assign the authoritative label. Failed, unavailable and zero-effect tests do not automatically create support/challenge edges. The source retains non-significant evidence even when neither relation applies.

The existing single-writer publication path includes the edges in graph snapshots. Personalized PageRank traverses them for relevance; it does not treat support as truth or subtract challenge edges as negative evidence. Selected context includes the required endpoints, and the EDGES TSV table carries the metadata. The original source is rechecked on retrieval. The independent TSV auditor separately reconstructs the expected edge meaning from the recorded plan/results and rejects forged support, wrong endpoints or promotion of declared assumptions to verified facts.

Older archives retain their original semantics and are not retroactively rewritten. This addition does not introduce separate assumption nodes, cross-project adjudication, causal-validity certification or human-confirmed revision workflow. There is no experiment yet showing that these added relations improve scientific research performance.
