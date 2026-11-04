# tests

Tests workspace within GEO-INFER.

## Contents

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

## Public Interface

- `conftest.py:git_environment` (function)
- `conftest.py:generator` (function)
- `conftest.py:repo_root` (function)
- `conftest.py:repo_inventory` (function)
- `conftest.py:figure_specs` (function)
- `conftest.py:git_repo` (function)
- `conftest.py:shipped_artifacts_are_read_only` (function)
- `conftest.py:generatable_checkout` (function)


## Validation

```bash
uv run python -m pytest tests
```


## Manuscript Test Suite

- These fourteen tracked tests validate manuscript structure, bibliography,
  figures, provenance, fingerprints, and PDF layout; module behavior is tested
  under each `GEO-INFER-*/tests/`.
- Run them from the repository root with `uv run python -m pytest tests/`.
- Tests read the generated artifact set; run
  `uv run python manuscript/generate_research_artifacts.py` first when a test
  reports missing outputs.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
