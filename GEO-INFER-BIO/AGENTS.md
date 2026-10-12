# Agent Instructions: GEO-INFER-BIO

## Scope

- Owning module: `GEO-INFER-BIO`
- Python package: `geo_infer_bio`
- Directory role: Bioinformatics and biological data analysis with geospatial context for spatial omics, landscape genetics, phylogeography, and microbial ecology.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_bio` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `docs/`
- `examples/`
- `src/`
- `tests/`
- `Dockerfile`
- `SKILL.md`
- `docker-compose.yml`
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module BIO
```


## Visualization Guidance

- Keep spatial plotting inputs nonempty, finite, geographically bounded, and
  explicit about required columns; return figures for programmatic inspection.
- Create nested output directories before saving and close saved figures.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
