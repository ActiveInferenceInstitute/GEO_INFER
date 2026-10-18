# Limitations and Next Steps {#sec:limitations_next_steps}

## Current Limitations

- Repository inventory counts measure implementation and verification surfaces;
  they do not establish domain validity, predictive accuracy, calibration, or
  causal efficacy.
- The generated figures summarize file-backed evidence surfaces and should not
  be read as benchmark results. PNG, SVG, JSON, and CSV companions are multiple
  representations of that evidence rather than independent confirmations.
- A verification record is only as complete as the commands requested during
  the generation that produced it, and that generation is not necessarily this
  one: a build that requests none republishes the stored record instead of
  emptying it. The record published here was measured at the
  `{{VERIFICATION_RECORD_TIER}}` tier, which defines
  `{{VERIFICATION_DEFINED_COUNT}}` command groups; it reports
  `{{VERIFICATION_STATUS}}` with `{{VERIFICATION_UNRUN_COUNT}}` of those
  groups unrun, is {{VERIFICATION_RECORD_PROVENANCE}}, and exposes the
  command-level record for audit.
- The composition contract constrains interfaces, provenance, and refusal
  behavior. It is not a proof of correctness: no clause in it establishes that
  a free-energy value, a posterior, or a return level matches an external
  reference implementation or a field observation.
- Shared H3 identifiers, coordinate arrays, and UTC timestamps do not establish
  compatible spatial or temporal support. Aggregation, disaggregation,
  ecological inference, and sensitivity to resolution require explicit
  study-level treatment [@gotway_spatial_support_2002].
- The local categorical free-energy calculator accepts a reference vector via
  an argument named `preferences`; the local policy score includes an entropy
  exploration proxy and a variance-weighted risk term. These implementation
  conventions are described explicitly rather than treated as interchangeable
  with every theoretical or external-backend free-energy formulation.
- Reproducible seeds address a source of computational variability, not all
  variation across libraries, platforms, floating-point kernels, or hardware.
  Selected Python 3.15 transport and profiling evidence does not qualify the
  entire scientific stack on that interpreter. A profiler's output is not an
  uninstrumented benchmark [@python315_profiling_docs].
- Agent interfaces execute local Python code with the process's authority.
  Process-local handles have no persistence across server restarts, and failed
  batches do not undo effects. Native lifecycle checks are software behavior,
  not an application-level recovery protocol.
- Module theme membership is a declared editorial grouping, not a measured
  dependency structure. The manuscript reports no import graph, no coupling
  metric, and no evidence that the themes partition the framework's actual
  interactions.
- Documentation-page counts are scoped to `GEO-INFER-INTRA/docs/`, the
  cross-module documentation tree. No freshness, ownership, or staleness check
  runs over those pages, so the count says how many exist, not how many are
  current.
- External scientific claims require resolved entries in `references.bib` and
  should be expanded with domain-specific datasets and preregistered analyses.
  Bibliographic resolution alone does not show that a citation supports a
  claim. Model adequacy remains conditional and empirically assessed
  [@oreskes_model_validation_1994].

## Next Steps

1. For each candidate publication, resolve any unrun or failing groups at its
   recorded tier and run `--full-validation --publication` from clean, current
   source. Archive `research_verification.json`, `research_inventory.json`,
   the figure registry, and all companion artifacts beside the rendered
   outputs. Existing results remain attributed to the source that produced
   them rather than being relabeled for a new candidate.
2. Extend the inventory with domain datasets and benchmark results only when
   their producers emit machine-readable provenance and uncertainty.
3. Add preregistered comparisons for the Active Inference, Bayesian, and RISK
   research spine without replacing real model outputs with file-count proxies.
4. Add a documentation-freshness measurement so the documentation-page count
   can be qualified by currency rather than reported as a bare total.
5. Promote the composition contract's clauses into executable checks where they
   are not already covered by a module test, so the contract is verified by the
   same evidence record that reports the rest of the build.
6. Evaluate representative composed studies with declared spatial support,
   observation error, priors, calibration data, and decision costs. Compare
   methods against explicit numerical or empirical references before making a
   predictive, causal, or performance-superiority claim.

## Boundary Note

Changing values must enter the manuscript through the generator. The tracked
source contains tokens; resolved output is disposable and must never be edited
by hand. The one tracked file the generator itself writes is
`manuscript/config.yaml`, for the reason recorded in the Reproducibility
section.
