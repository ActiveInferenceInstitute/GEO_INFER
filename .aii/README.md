# .aii — InstituteOS sidecar

This folder is the committed [InstituteOS](https://github.com/ActiveInferenceInstitute/InstituteOS)
sidecar for **`ActiveInferenceInstitute/GEO_INFER`**. It records curated institutional metadata that the GitHub
API cannot provide — identity, category, maturity, ecosystem relations, key artifacts,
provenance, and portable tasks — so InstituteOS can federate a unified view across
Institute repositories without touching this project's own content. The schema is
`aii-sidecar/v1`.

## Contents

| File | Purpose |
| --- | --- |
| [`config.yaml`](config.yaml) | Machine-readable metadata; the single source of truth. |
| [`readiness.json`](readiness.json) | Declared tier, baseline commit, and waivers for the AII readiness standard. |
| [`docs/`](docs/README.md) | Sidecar documents: the [`README.md`](docs/README.md) · [`AGENTS.md`](docs/AGENTS.md) folder guides, and [`INTEGRATION.md`](docs/INTEGRATION.md) — how this repository is federated, validated, and operated through InstituteOS. |
| [`README.md`](README.md) · [`AGENTS.md`](AGENTS.md) | Human and agent guidance for this folder. |

## Repository profile

| Field | Value |
| --- | --- |
| Repository | `ActiveInferenceInstitute/GEO_INFER` |
| Category | `research-software` |
| Status | `active` |
| Maturity | `released` |
| License | `CC-BY-NC-SA-4.0` |
| Default branch | `main` |
| Homepage | — |
| Tags | `geo-infer`, `geospatial`, `active-inference`, `modules`, `html` |

## Capabilities

- python-package
- tested
- ci
- citation
- manuscript-publication

## Portable tasks

| Task | Command | Working directory | Description |
| --- | --- | --- | --- |
| `test` | `uv run --frozen python scripts/gate.py test-unit` | `.` | Run the canonical unit-test arm. |
| `lint` | `uv run --frozen python scripts/gate.py lint` | `.` | Run repository lint and format checks. |
| `validate` | `uv run --frozen python scripts/gate.py gates` | `.` | Run strict repository, import, model, and secret gates. |

## Links

- **github:** https://github.com/ActiveInferenceInstitute/GEO_INFER
- **website:** https://www.activeinference.org

## Validate

The authoritative validator ships in the private `aii-platform` kernel:

```bash
python -m aii_platform.aii_sidecar.validate <repo> --completeness
python -m aii_platform.aii_sidecar docs <repo> --overwrite
python -m aii_platform.aii_sidecar.run <repo> --list
```

If the private environment is unavailable, review `config.yaml` as plain YAML and
check that every declared path exists in the repository.
