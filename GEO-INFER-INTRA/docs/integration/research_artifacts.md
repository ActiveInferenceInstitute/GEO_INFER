# Reusable research figures and publication artifacts

The repository's manuscript producer is
`manuscript/generate_research_artifacts.py`. It measures the current source,
resolves authored tokens, draws four figures, and records executed verification
separately from repository file counts. Generated files belong under ignored
`output/`; authored Markdown and bibliography remain the reviewable sources.

## Build and check a clean publication

Commit source changes first. Refresh the four generator-owned metadata fields
and commit only `manuscript/config.yaml` before measuring the clean checkout:

```bash
uv run --frozen python manuscript/generate_research_artifacts.py --refresh-config
git add manuscript/config.yaml
git commit -m "Refresh manuscript source metadata"
uv run --frozen python manuscript/generate_research_artifacts.py --full-validation --publication
uv run --frozen python manuscript/generate_research_artifacts.py --check
```

The explicit metadata commit reaches a fixed point because the manuscript's
source date excludes config-only commits. Any subsequent source change requires
a new refresh. `--allow-dirty` is a development rendering option; it cannot be
combined with `--publication`. A carried verification record keeps its original
source identity, tier, and measured-elsewhere status.

`--check` writes nothing. It compares the current source fingerprint, tracked
config metadata, and verification summary, then verifies every delivered figure,
data table, resolved manuscript file, and supporting evidence file against its
recorded SHA-256. Missing files, changed bytes, duplicate figure identifiers,
unsafe artifact names, symlinked files, and incomplete digest sets fail the
check. These hashes establish byte consistency; they are not a signature or
independent scientific validation.

## Figure bundle contract

Each figure has four files in `output/figures/`:

| Format | Purpose |
| --- | --- |
| PNG | Fixed-DPI manuscript rendering, with enforced print dimensions |
| SVG | Editable vector artwork with text, title, description and source identity |
| JSON | Typed tables and explicit measurement semantics |
| CSV | Flat table rows with table name, source commit and source hash |

`figure_registry.json` retains the primary PNG's `filename` and `sha256`, plus
`label`, `caption`, `alt_text`, and `generated_by`. Its additive `artifacts` list
records each companion's filename, media type, and SHA-256. Companion files
must be regular files in the registry's directory. SVG exports omit a build
timestamp and use a source-derived identifier salt. Their text remains text,
which supports editing and assistive readers; this does not establish a tagged
PDF accessibility certification.

The JSON schema is `geo-infer-manuscript-figure-data/v1`. The top-level fields
are `source_commit`, `source_hash`, `figure_label`, `measurement`, and `tables`.
The inventory and research-spine tables expose module/package identity,
source/test file counts, declared theme, and research-spine membership. The
validation figure exports test-directory categories and repository surfaces.
The graphical abstract exports module themes and the displayed default-tier
verification statuses; missing results remain `not-run`.

File counts are implementation surfaces. They do not measure test coverage,
executed cases, model quality, causal validity, or forecasting skill. Theme
membership is an editorial classification. Read the source-bound verification
record and actual runner receipts for execution evidence.

`output/data/research_manifest.json` lists resolved manuscript paths and an
`artifact_sha256` mapping covering all delivered dependencies other than the
manifest itself. The figure registry binds captions and alt text as well as
the figures; the manifest binds that registry and the resolved bibliography.
Retain this entire bundle when sharing a PDF so its measurements remain
inspectable and reusable.

Bibliography preflight checks the resolved top-level text in memory and the
authored module chapters that the PDF renderer appends after the catalog.
Fatal preflight findings stop before verification and output writes. The
post-generation audit and `--check` apply the same citation scope.
Module chapters are appended literally; generated variable substitution applies
to the top-level manuscript files.

## Visualization and profiling interfaces

For reusable map construction, use the owning `GeoArt` APIs described in
[geospatial visualization](../geospatial/visualization/index.md). Discover
their current signatures through `geo-infer catalog ART`, then
`geo-infer describe`. Python scripts, persistent CLI sessions, MCP `geo_call`,
and module skills share the same public methods and object handles.

Use [backend profiling](backend_profiling.md) for actual workload execution and
read-only hotspot reports. Python 3.15 tracing and sampling have different
measurement units; neither an instrumented profile nor the repository surface
charts establish a throughput improvement. Keep successful and failed receipts
with their original artifact bytes.
