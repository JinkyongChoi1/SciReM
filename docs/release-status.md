# Release status

SciReM is a research-memory prototype. The dashboard displays stored records,
retrieval evidence, AI activity and human reviews. Review submissions are saved
but are not automatically applied to the research graph or passed to an agent.

Validation covers 402 unit tests, 12 handoff tests, browser checks and two-cycle
synthetic-data smoke runs with actual R. Model responses in the smoke runs are
scripted; these tests do not establish research-performance improvements.

The memory CLI and demo support explicit non-SGE execution. The empirical
launcher remains SGE-specific; Mac and Windows have not been verified. The
repository does not currently specify a code reuse license.
