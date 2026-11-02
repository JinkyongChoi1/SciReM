# Suggested proposal inserts

These passages supplement the author's SciReM draft; they are not a complete1,500–2,000word submission. Check the final references, empirical artifact accessibility and actual page layout before submission.

## Core contribution

SciReM investigates whether preserving the evidential basis of scientific judgments can improve cumulative human–AI research. Its proposed contribution is not a new language model or ranking algorithm, but a research-memory representation that connects hypotheses, executed analyses, interpretations and subsequent decisions. These connections are intended to help an agent distinguish a contradicted explanation from an inadequately tested one, and help a researcher reconstruct why a direction was pursued before deciding whether to redirect it. The central empirical question is whether this organization produces better justified next actions and more effective human supervision than access to the same research records through general-purpose retrieval. Improvements in scientific quality, novelty and efficiency remain outcomes to be tested rather than assumptions of the design.

## What has already been built and learned

I have implemented a research-loop prototype with a source-linked graph, query-anchored retrieval, personalized PageRank, tabular evidence packets and a statistical Runner. It records a trajectory from research question to hypothesis, analysis specification, numerical results, assessment and decision. The implementation supports unified or separate research-role contexts and optional independent review. Numerical evidence is preserved separately from agent interpretation, and actual source artifacts can be inspected. The current version partially represents judgment through model assumptions, rival explanations, evidence links and decision bases; a fully integrated researcher-confirmed revision workflow remains to be developed.

Development experiments establish feasibility while exposing important limits. In one real-data probe, an agent proposed and executed seven pooled conditional-logit interactions to compare prespecified samples with their complements. Independently written package-based code reproduced all49 reported aggregate outputs. In a separate fixed-result probe, surfacing already available metrics in a concise table increased acknowledgement of completed diagnostics from0/3 to3/3 responses in one case. This small presentation intervention does not isolate a memory or PageRank effect. Subsequent full-loop tests did not establish reduced repetition and revealed a more consequential failure: an agent interpreted a successful primary model while failing to acknowledge an unsuccessful additional diagnostic. After source-grounded error feedback was repaired, two failed diagnostic programs were each corrected in one model call and reproduced the expected numerical outputs. These observations motivate evaluating actual execution and claim–evidence agreement, rather than treating fluent accounts of collaboration as evidence that collaboration occurred.

## Risks and intended development

The principal risk is that structured memory may preserve and amplify mistaken judgments rather than improve them. Source links can establish where a claim came from without establishing whether it is justified; highly connected records can dominate retrieval even when their assumptions are unresolved. Other risks include a mismatch between proposed tests and available tools, incomplete error handoffs, model-specific context behavior, and the cost of reviewing a growing research record. I will address these risks through explicit separation of numerical results and interpretation, inspectable retrieval traces, tests of actual execution, and independently rated research decisions. The evaluation will compare matched records and resources, preserve unsuccessful trajectories, and distinguish justified maintenance from gratuitous hypothesis changes. Human checkpoint studies will test whether researchers can detect unsupported claims and issue guidance that subsequent analyses actually implement.

If successful, SciReM would make the basis of research decisions easier to transfer across collaborators and sessions without requiring agreement on a single predetermined theory. Its value would lie in enabling researchers to reopen consequential questions, identify unresolved assumptions and direct effort toward informative tests. The intended outcome is an inspectable infrastructure for cumulative organizational inquiry, not simply a larger volume of automated analyses.

## Reference checks

Verified starting points for the final bibliography:

- Davis, M.S.(1971). That's Interesting!: Towards a Phenomenology of Sociology and a Sociology of Phenomenology. *Philosophy of the Social Sciences,1*(2),309–344. https://doi.org/10.1177/004839317100100211
- Timmermans, S., & Tavory, I.(2012). Theory Construction in Qualitative Research: From Grounded Theory to Abductive Analysis. *Sociological Theory,30*(3),167–186. https://doi.org/10.1177/0735275112457914
- Gutiérrez et al.(2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. https://arxiv.org/abs/2405.14831 (retrieve the complete author list and chosen publication version for the final bibliography).

The draft's abbreviated Anthropic2025, Sarkar2026, Swanson2025 and other entries are not a verified bibliography. Do not invent missing titles or infer a general expert-level capability claim from an unspecified source. The proposal's mechanism is a hypothesis informed by the literature, not a result established by citing it.
