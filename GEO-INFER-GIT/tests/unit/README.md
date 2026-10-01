# GEO-INFER-GIT/tests/unit

Unit workspace within `GEO-INFER-GIT`.

## Contents

- `test_advanced_git.py`
- `test_cli.py`
- `test_config_loader.py`
- `test_config_loader_packaged_config.py`
- `test_distributed_coordinator.py`
- `test_error_handler.py`
- `test_error_recovery_strategies.py`
- `test_github_api.py`
- `test_intelligent_cache_prefetch.py`
- `test_main.py`
- `test_main_entrypoint.py`
- `test_multi_platform_api.py`
- `test_repo_analyzer.py`
- `test_repo_cloner.py`
- `test_repo_manager.py`
- `test_rest_api.py`
- `test_rest_api_error_handling.py`
- `test_validation.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-GIT`
- Package: `geo_infer_git`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-git`
- Tests: `uv run python -m pytest GEO-INFER-GIT/tests/unit`

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
uv run python -m pytest GEO-INFER-GIT/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
