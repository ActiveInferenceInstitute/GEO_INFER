# InstituteOS integration — GEO_INFER

## Identity

| Field | Value |
| --- | --- |
| Repository | `ActiveInferenceInstitute/GEO_INFER` |
| Affiliation | `institute` |
| Category | `research-software` |
| Status | `active` |
| Maturity | `released` |
| Steward | Active Inference Institute |
| Default branch | `main` |
| Current release | — |
| License | `CC-BY-NC-SA-4.0` |
| Citation | Active Inference Institute — GEO-INFER. |

## Federation

| Setting | Value |
| --- | --- |
| Dashboard mode | `repos` |
| Analytics | `true` |
| Dashboard | `true` |
| Sync policy | `pull` |
| Registries | — |
| Tags | geo-infer, geospatial, active-inference, modules, html |

## Ecosystem relations

| Repository | Relation | Note |
| --- | --- | --- |
| `ActiveInferenceInstitute/InstituteOS` | `referenced-by` | Federated via the InstituteOS .aii manifest. |
| `ActiveInferenceInstitute/Active_Inference_Ontology` | `references` | Uses Active Inference Ontology terms. |
| `ActiveInferenceInstitute/pymdp` | `depends-on` | Consumes the pymdp active-inference library. |

## Key artifacts

| Path | Kind | Description |
| --- | --- | --- |
| `README.md` | `doc` | Repository overview. |
| `pyproject.toml` | `code` | Python package config. |
| `LICENSE` | `doc` | License. |
| `AGENTS.md` | `doc` | Agent guidance. |
| `CITATION.cff` | `citation` | Machine-readable citation metadata. |
| `tests` | `test` | Test suite. |
| `manuscript` | `publication` | Manuscript sources and artifacts. |
| `CHANGELOG.md` | `doc` | Changelog. |

## Portable tasks

| Task | Command | Working directory | Description |
| --- | --- | --- | --- |
| `test` | `uv run python GEO-INFER-TEST/run_unified_tests.py --category unit` | `.` | Run the test suite. |
| `lint` | `uv run --with 'ruff>=0.15.6,<0.16' ruff check .` | `.` | Run the repository lint gate. |
| `validate` | `uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke` | `.` | Validate repository layout and documentation contracts. |

## Validation

```bash
python -m aii_platform.aii_sidecar.validate <repo> --completeness
python -m aii_platform.aii_sidecar docs <repo> --overwrite
python -m aii_platform.aii_sidecar.run <repo> --list
```

The sidecar is valid when `validate` prints `OK` and reports no missing artifact,
manifest, link, or declared-doc path. Completeness is a curation target, not a
runtime requirement.

## Data boundary

This sidecar carries metadata only. It does not grant execution permission, does not
replace the repository's own documentation, and must not be used to move private
content into a public repository. Task execution is limited to trusted checkouts.
