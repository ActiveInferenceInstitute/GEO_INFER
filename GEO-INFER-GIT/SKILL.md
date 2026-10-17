---
name: geo-infer-git
description: Git repository cloning, management, and REST API tooling for geospatial projects. Use when cloning many GitHub repositories (by user or by list), managing local git worktrees, analyzing repositories, or running the GEO-INFER-GIT FastAPI service.
prerequisites:
  required: []
  recommended: []
difficulty: beginner
estimated_time: 30min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-GIT

## Instructions

### Core Capabilities

- **Bulk repository cloning**: Clone repositories from target users and organizations with parallel workers, progress tracking, and clone reports
- **Repository management**: Clone/sync/status/branch operations across local worktrees (`RepoManager`)
- **Multi-platform API clients**: GitHub, GitLab, Bitbucket, and local repository discovery (`GitHubAPI`, `MultiPlatformAPI`)
- **Repository analysis**: Code complexity, documentation coverage, and geospatial content detection (`RepositoryAnalyzer`)
- **Error handling and recovery**: Retry, classification, and recovery strategies (`geo_infer_git.utils.error_handler`)
- **Caching**: SQLite/Redis-backed advanced cache (`geo_infer_git.utils.advanced_cache`)

### Key Imports

```python
from geo_infer_git.core.repo_manager import RepoManager
from geo_infer_git.core.repo_cloner import RepoCloner
from geo_infer_git.core.github_api import GitHubAPI
from geo_infer_git.core.multi_platform_api import MultiPlatformAPI
from geo_infer_git.core.repo_analyzer import RepositoryAnalyzer
from geo_infer_git.api import app, run_api
```

## Examples

```python
from geo_infer_git.core.repo_manager import RepoManager
from geo_infer_git.core.repo_cloner import RepoCloner
from geo_infer_git.utils.config_loader import CloneConfig

# Manage local git worktrees under a base directory
manager = RepoManager(config_path="config/example.yaml")
status = manager.check_repo_status()  # dict of repo name -> status

# Clone repositories in parallel
config = CloneConfig(output_dir="./cloned_repositories", max_workers=4)
cloner = RepoCloner(config)
results = cloner.clone_multiple_repositories([("owner", "repo", "main")])
cloner.close()

# GitHub search + filtering (requires network and optionally GITHUB_TOKEN)
from geo_infer_git.core.github_api import GitHubAPI

github = GitHubAPI()
repos = github.search_repositories("geospatial", language="Python", max_results=10)
filtered = github.filter_repositories(repos, min_stars=10, exclude_forks=True)
```

### CLI

The package installs a `geo-infer-git` console script (`geo_infer_git.cli:main`) with
`clone`, `sync`, `status`, and `branch` commands over `RepoManager`.

### Agent interfaces

Use `uv run --frozen geo-infer catalog GIT` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- `RepoCloner` and `RepoManager` create their output/base directories lazily on first write;
  constructing them has no filesystem side effects.
- `cleanup_failed_clones` only removes `owner/repo`-shaped directories under the output
  directory that lack a `.git` folder; unrelated files are never touched.
- The REST API serves repository management, clone/sync, and branch endpoints only.
  Data versioning, workflow, and deployment endpoints are descoped: no backing
  domain behavior exists in src/geo_infer_git for them.

### Integrations

- Works with cloned geospatial dataset repositories used by GEO-INFER-DATA
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module GIT`