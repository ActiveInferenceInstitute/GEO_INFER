# Deployment and Execution

GEO-INFER is currently documented and validated as a Python workspace. Deployment
means packaging one or more module APIs, runners, or analysis jobs for a target
environment; there is no repository-wide server command or default production
topology.

## Reproducible environment

```bash
uv sync --all-packages --all-extras --all-groups
uv run python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
```

Pin the repository state with Git and retain pyproject.toml, uv.lock, and
.python-version together.

## Running a module API

Only deploy a module that exposes the required application surface. For
GEO-INFER-API, application construction requires a nonempty process-environment
`SECRET_KEY`. This local example generates an ephemeral value for that command:

```bash
SECRET_KEY="$(uv run --frozen python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  uv run --frozen uvicorn geo_infer_api.app:main_app --host 127.0.0.1 --port 8000
```

This command is a local example, not a claim that a hosted service or public
endpoint exists. The current application has no authentication middleware; the
required secret does not add authentication. Polygon features use a process-local
in-memory store, so restarts lose data and workers do not share it. Configure and
qualify authentication, persistence, CORS, secrets, logging, and resource limits
for the intended deployment. See the [GeoJSON API contract](../../../GEO-INFER-API/docs/geojson_api.md).

## Batch and scenario jobs

- Use module CLIs and examples documented in the module root.
- Pass an explicit output directory for scenario and visualization artifacts.
- Keep generated reports under .geo-infer-test-results/ or a job-specific
  artifact directory.
- Record the exact package versions, input CRS, H3 resolution, random seed, and
  validator results with the job output.

## CI and release evidence

The hosted CI workflow runs Python 3.11 and 3.12 on CPU-only Ubuntu runners.
It installs the workspace with a small native-only exclusion list, then runs
source, documentation, model, test, and H3 gates. Treat CI configuration as the
deployment/release contract until a separate deployment manifest is added.

## Further reading

- [Installation](../getting_started/installation_guide.md)
- [Architecture](../architecture/index.md)
- [Testing guide](../developer_guide/testing_guide.md)
- [Security policy](../../../SECURITY.md)
