# GEO-INFER-GIT/src/geo_infer_git

Geo Infer Git workspace within `GEO-INFER-GIT`.

## Contents

- `api/`
- `config/`
- `core/`
- `utils/`
- `__init__.py`
- `cli.py`
- `main.py`

## Public Interface

- `cli.py:setup_logging` (function)
- `cli.py:load_repo_list` (function)
- `cli.py:clone_command` (function)
- `cli.py:sync_command` (function)
- `cli.py:status_command` (function)
- `cli.py:branch_command` (function)
- `cli.py:main` (function)
- `main.py:parse_arguments` (function)
- `main.py:create_gitignore_entry` (function)
- `main.py:generate_report` (function)
- `main.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-GIT`
- Package: `geo_infer_git`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-git`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module GIT`

## Dependencies

- `urllib3>=2.0.6`
- `fastapi>=0.104.0`
- `starlette>=0.27.0`
- `GitPython>=3.1.0`
- `jsonschema>=4.17.0`
- `pydantic>=2.5.0`
- `pyyaml>=6.0`
- `requests>=2.28.1`
- `tqdm>=4.65.0`
- `uvicorn[standard]>=0.24.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module GIT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
