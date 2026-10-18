# Supplemental Source Surface {#sec:source_surface}

This supplement records the source surfaces used by the generated manuscript
evidence bundle. The current source fingerprint is
`{{RESEARCH_SOURCE_HASH}}` at commit `{{RESEARCH_COMMIT}}`.

| Surface | Role |
|---|---|
| `GEO-INFER-*/src/` | Python implementation surface measured per module. |
| `GEO-INFER-*/tests/` | Module behavior and contract evidence surface. |
| `GEO-INFER-TEST/` | Repository validators, unified suites, and reproducibility checks. |
| `GEO-INFER-INTRA/docs/` | Cross-module conceptual and user-facing documentation. |
| `GEO-INFER-INTRA/docs/research_grade_inference_contracts.md` | Authored source for the ACT, BAYES, and RISK interface obligations reproduced in the Composition Contract. |
| `GEO-INFER-ACT/src/geo_infer_act/core/types.py` | Typed results — `FreeEnergyBreakdown`, `PolicyEvaluation`, `ActiveInferenceStepResult` — that the contract is expressed over. |
| `GEO-INFER-API/src/geo_infer_api/agent_runtime.py` and `agent_schema.py` | Native object retention, scientific codecs, request validation, and explicit lifecycle for shared script, JSONL, and MCP execution. |
| `pyproject.toml` and `uv.lock` | Project version and environment provenance. |
| `output/data/` | Generated inventory, variables, verification, and manifest records. |
| `output/figures` | Generated figures and their caption and provenance registry. |

: Source and output surfaces associated with the evidence bundle at source hash
`{{RESEARCH_SOURCE_HASH}}`. The first four rows are measured populations —
the Python implementation and test trees, the unified test framework in
`GEO-INFER-TEST`, and the cross-module documentation under
`GEO-INFER-INTRA/docs/`. The implementation and packaging rows identify
tracked inputs to review. The final output rows identify generated destinations;
they are not authored evidence or a substitute for source custody.
{#tbl:source_surface}

## Authored and Generated Boundary

[@tbl:source_surface] lists what is measured. This subsection records who
writes what, which is the invariant the Reproducibility contract depends on.

| Path | Written by | Tracked |
|---|---|---|
| `manuscript/0*.md`, `manuscript/9*.md`, `manuscript/S*.md` | Author | yes |
| `manuscript/sections/*.md` | Author | yes |
| `manuscript/references.bib` | Author | yes |
| `manuscript/preamble.md` | Author | yes |
| `manuscript/generate_research_artifacts.py` | Author | yes |
| `manuscript/config.yaml` | Author, except four generator-owned fields marked in place | yes |
| `output/data/`, `output/figures`, `output/manuscript/` | Generator | no |
| `output/pdf/`, `output/web/` | Renderer | no |

: Authored versus generated files. Every tracked file under `manuscript/` is
authored except four generator-owned fields marked in place in
`manuscript/config.yaml` (project version, title-page date, publication year,
licence); everything under `output/` is disposable, rebuilt each build. Read
the table as the write-ownership contract behind Reproducibility: every value
is hand-reviewable at a tracked path or regenerated from a measured one.
{#tbl:authored_generated}

## Reproducing This Build

The command that regenerates every measured value, figure, caption, and
resolved manuscript copy in this document, and that a publication build must
use, is:

```bash
uv run python manuscript/generate_research_artifacts.py \
    --full-validation --publication
```

That invocation runs both tiers of verification command group, and
`--publication` refuses to finish while the evidence record is empty or holds
a failed group. Clean generation also requires current tracked metadata.
After committing authored source, run `--refresh-config`, review and commit
the config-only change, and generate from the resulting clean checkout.
`--allow-dirty` is confined to explicitly marked development builds;
combining it with `--publication` is rejected. `--check` validates the
published evidence and artifact hashes against the measured checkout without
writing files. These checks establish source and artifact correspondence, not
scientific model adequacy.

## External References

Each externally-supported claim area in this manuscript resolves to a verified
BibTeX entry rather than to a repository-derived count:

- Active inference and the free-energy formulation
  [@friston_free_energy_2010; @parr_active_inference_2022], and the
  `inferactively-pymdp` backend the categorical path depends on
  [@heins_pymdp_2022].
- Bayesian posterior estimation and convergence diagnostics
  [@gelman_bda_2013].
- Extreme-value return levels underlying the risk module's Gumbel fit
  [@coles_extremes_2001].
- The H3 hierarchical spatial index [@h3_docs].
- Spatial support and the interpretation of incompatible spatial inputs
  [@gotway_spatial_support_2002].
- Computational reproducibility and reusable research metadata
  [@sandve_reproducible_2013; @wilkinson_fair_2016].
- Figure communication and the distinction between profiling and benchmarking
  [@rougier_figures_2014; @python315_profiling_docs].
- Conditional assessment of numerical earth-science models
  [@oreskes_model_validation_1994].

The generator fails the build on an unresolved citation and names any entry no
section cites. That check establishes bibliographic closure, not scholarly
support. Publication years, author lists, identifiers, and the relationship
between a reference and its claim require separate review against primary
sources. Mutable documentation entries identify their version and access date
instead of presenting the access year as a publication year.
