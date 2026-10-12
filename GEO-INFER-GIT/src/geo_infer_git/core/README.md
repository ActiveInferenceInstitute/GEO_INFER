# GEO-INFER-GIT/src/geo_infer_git/core

Core workspace within `GEO-INFER-GIT`.

## Contents

- `__init__.py`
- `advanced_git.py`
- `distributed_coordinator.py`
- `github_api.py`
- `multi_platform_api.py`
- `repo_analyzer.py`
- `repo_cloner.py`
- `repo_manager.py`

## Public Interface

- `advanced_git.py:SubmoduleInfo` (class)
- `advanced_git.py:MergeConflict` (class)
- `advanced_git.py:CherryPickOperation` (class)
- `advanced_git.py:RebaseOperation` (class)
- `advanced_git.py:SubmoduleManager` (class)
- `advanced_git.py:CherryPickManager` (class)
- `advanced_git.py:RebaseManager` (class)
- `advanced_git.py:AdvancedGitOperations` (class)
- `advanced_git.py:create_advanced_git_operations` (function)
- `distributed_coordinator.py:NodeInfo` (class)
- `distributed_coordinator.py:JobInfo` (class)
- `distributed_coordinator.py:CoordinationMessage` (class)
- `distributed_coordinator.py:DistributedCoordinator` (class)
- `distributed_coordinator.py:create_distributed_coordinator` (function)
- `github_api.py:GitHubRepository` (class)
- `github_api.py:RateLimit` (class)
- `github_api.py:GitHubAPI` (class)
- `multi_platform_api.py:GitLabRepository` (class)
- `multi_platform_api.py:BitbucketRepository` (class)
- `multi_platform_api.py:LocalRepository` (class)

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
