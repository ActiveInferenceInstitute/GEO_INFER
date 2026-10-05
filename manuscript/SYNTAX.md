# Manuscript Syntax - GEO-INFER

This manuscript uses the shared template conventions from `docs/guides/manuscript-semantics.md` in the sibling template repository.

## Section Labels

| File | H1 | Label |
|---|---|---|
| `00_abstract.md` | Abstract | `{#sec:abstract}` |
| `01_introduction.md` | Introduction | `{#sec:introduction}` |
| `02_system_context.md` | System Context | `{#sec:system_context}` |
| `03_methods.md` | Methods | `{#sec:methods}` |
| `04_artifacts_and_evidence.md` | Artifacts and Evidence | `{#sec:artifacts_evidence}` |
| `05_reproducibility.md` | Reproducibility | `{#sec:reproducibility}` |
| `06_limitations_and_next_steps.md` | Limitations and Next Steps | `{#sec:limitations_next_steps}` |
| `S01_source_surface.md` | Supplemental Source Surface | `{#sec:source_surface}` |
| `S02_module_catalog.md` | Supplemental Module Catalog | `{#sec:module_catalog}` |
| `98_symbols_glossary.md` | Symbols and Glossary | `{#sec:symbols_glossary}` |
| `99_references.md` | References | `{#sec:references}` |

## Module Catalog Sections

The per-module entries of the Supplemental Module Catalog live in
`sections/<lowercase-slug>.md` (one file per module, e.g. `sections/sec.md`
for GEO-INFER-SEC). They are not published files in the table above: the
render script concatenates every `manuscript/sections/*.md` in alphabetical
stem order after `S02_module_catalog.md` when it builds the combined
document (`MODULE_CATALOG_ENTRY` and `_module_catalog_sections` in
`scripts/render_manuscript_pdf.py`). A missing or empty `sections/`
directory fails the render rather than publishing a truncated catalog.

Catalog entry files are plain prose with these constraints:

- The first line is `## GEO-INFER-<SLUG> — <Title>`.
- No `{{TOKENS}}`: the render combines them from the tracked source without
  token substitution, so a token would reach the PDF literally. Quantities
  belong in the generated tables of the main sections.
- No duplicated source/test census or suite-size rankings: the generated
  catalog index owns those measurements. Describe concrete verification
  surfaces and their evidence boundaries instead.
- No figures, labels (`{#...}`), raw LaTeX, or HTML: entries are H2 prose
  under the catalog's single H1.

## Citations

Use Pandoc citation syntax only, for example `[@real_key]`. Every key must exist in `references.bib` before it appears in prose.

## Figures

Generated figures are produced by `generate_research_artifacts.py`, registered
with dynamic captions, and live under `../output/figures/` in the authored
manuscript source. They are referenced with labels such as:

```markdown
![Generated caption text.](../output/figures/example.png){#fig:example width=80%}
```

The generator rewrites the relative figure path for resolved copies under
`output/manuscript/`, and rejects unresolved uppercase variable tokens.

## Claims

Changing quantitative claims must be represented by `{{TOKENS}}` in authored
sections and computed by the manuscript generator. Publication claims require
source-backed artifacts, passing verification, and resolved bibliography
entries.
