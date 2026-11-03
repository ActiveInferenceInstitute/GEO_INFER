# Agent Instructions: tests

## Scope

- Owning module: none (repository-level directory outside the GEO-INFER-* module fleet)
- Directory role: Tests workspace within GEO-INFER.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through the repository's module-level public contracts; this directory defines no Python package.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `conftest.py`
- `test_manuscript_bibliography.py`
- `test_manuscript_coverage.py`
- `test_manuscript_evidence_durability.py`
- `test_manuscript_figures.py`
- `test_manuscript_fingerprint.py`
- `test_manuscript_inventory.py`
- `test_manuscript_labels.py`
- `test_manuscript_paths.py`
- `test_manuscript_pdf_layout.py`
- `test_manuscript_preamble.py`
- `test_manuscript_provenance.py`
- `test_manuscript_source_lifecycle.py`
- `test_manuscript_verification_table.py`

## Validation

```bash
uv run python -m pytest tests
```


## Manuscript Test Rules

- These tests validate the manuscript pipeline, not module behavior; module tests
  live under each `GEO-INFER-*/tests/`.
- Run `uv run python -m pytest tests/` after regenerating artifacts.
- A missing artifact is a generation failure; regenerate rather than editing
  expected data.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
