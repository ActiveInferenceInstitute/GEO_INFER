# Agent Instructions: GEO-INFER-INTRA/docs

## Scope

- Owning module: `GEO-INFER-INTRA`
- Python package: `geo_infer_intra`
- Directory role: Docs workspace within `GEO-INFER-INTRA`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_intra` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `advanced/`
- `alphaearth/`
- `api/`
- `architecture/`
- `deployment/`
- `developer_guide/`
- `examples/`
- `geospatial/`
- `getting_started/`
- `guides/`
- `integration/`
- `knowledge_base/`
- `materiality/`
- `modules/`
- `ontology/`
- `realms/`
- `releases/`
- `security/`
- `support/`
- `tnfd/`
- `tutorials/`
- `user_guide/`
- `workflows/`
- `DOCUMENTATION_IMPROVEMENTS.md`
- `DOCUMENTATION_IMPROVEMENTS_SUMMARY.md`
- `DOCUMENTATION_STANDARDS.md`
- `active_inference_guide.md`
- `api_schema.yaml`
- `bayesian_inference_guide.md`
- `data_dictionary.md`
- `documentation_guide.md`
- `examples_gallery.md`
- `geospatial_standards.md`
- `index.md`
- `installation.md`
- `module_readme_template.md`
- `overview.md`
- `research_grade_inference_contracts.md`
- `temporal_analysis_guide.md`
- `terminology.md`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
