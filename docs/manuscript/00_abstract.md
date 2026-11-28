# Abstract {#sec:abstract}
![{{GRAPHICAL_ABSTRACT_CAPTION}}](../output/figures/graphical_abstract.png){#fig:graphical_abstract width=100%}

GEO-INFER is a modular geospatial inference framework spanning spatial
analysis, active inference, Bayesian methods, domain modeling, agent
workflows, and repository validation. This manuscript is generated from the
tracked checkout at commit `{{RESEARCH_COMMIT}}`: the repository contains
`{{MODULE_COUNT}}` modules that ship a `src/` package, grouped into
`{{MODULE_THEME_COUNT}}` themes, with `{{SOURCE_FILE_COUNT}}` Python source
files and `{{TEST_FILE_COUNT}}` Python test files across its measured
evidence surfaces. [@fig:graphical_abstract] summarises the resulting
pipeline end to end: the themed module set, the geospatial and probabilistic
models it composes into, the evidence loop that verifies it, and the claims
that verification underwrites.

The research contribution is an auditable composition contract. Domain
modules own their implementations, cross-module workflows expose shared
spatial and probabilistic interfaces, and validation commands make
correctness, reproducibility, and documentation claims executable. The
Methods section states that contract term by term for the Active Inference,
Bayesian, and risk spine: the free-energy decomposition an inference call
returns, the sampler and diagnostic obligations a posterior estimate carries,
and the bounds a risk result must satisfy before it is reported. Figures,
captions, and quantitative statements in this manuscript are produced from
the same repository inventory and source fingerprint used to render it.

The published evidence record was measured at the
`{{VERIFICATION_RECORD_TIER}}` tier, which defines
`{{VERIFICATION_DEFINED_COUNT}}` verification command groups. Of those,
`{{VERIFICATION_PASS_COUNT}}` passed, `{{VERIFICATION_FAIL_COUNT}}` failed,
and `{{VERIFICATION_UNRUN_COUNT}}` did not run; the record summarises as
`{{VERIFICATION_STATUS}}`, and every group is published with its own
outcome. The three counts are taken against the record's own tier and are
checked to sum to it, so the summary cannot report more outcomes than the
definition it is measured against admits. A build invoked with
`--publication` refuses an empty or failing record, so unrun and failed
checks are never represented as evidence of success.
