#!/usr/bin/env python3

"""
REST API implementation for GEO-INFER-GIT.

This module provides a FastAPI-based REST API that implements the endpoints
defined in the OpenAPI schema for repository management operations.
"""

import time
from typing import Any, cast
from datetime import datetime, UTC

import git
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

from ..core.repo_manager import RepoManager
from ..core.github_api import GitHubAPI
from ..utils.config_loader import ConfigLoader
from ..utils.logging_utils import setup_logging
from .errors import register_error_handlers

# Create FastAPI app
app = FastAPI(
    title="GEO-INFER-GIT API",
    description="Version control integration and repository management system for the GEO-INFER framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

# Global instances
repo_manager: RepoManager | None = None
github_api: GitHubAPI | None = None
config_loader: ConfigLoader | None = None
logger: Any | None = None
repository_records: dict[str, dict[str, Any]] = {}


# Pydantic models for request/response
class RepositoryRequest(BaseModel):
    """Request model for repository operations."""

    clone_url: str = Field(..., description="Repository clone URL")
    name: str | None = Field(None, description="Custom name for the repository")
    description: str | None = Field(None, description="Repository description")
    platform: str = Field(
        "github", description="Git platform (github, gitlab, bitbucket)"
    )
    credentials: dict[str, Any] | None = Field(
        None, description="Authentication credentials"
    )
    auto_sync: bool = Field(True, description="Enable automatic synchronization")
    sync_interval: int = Field(3600, description="Sync interval in seconds")

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        if v not in ["github", "gitlab", "bitbucket", "local"]:
            raise ValueError(
                "Platform must be one of: github, gitlab, bitbucket, local"
            )
        return v


class RepositoryResponse(BaseModel):
    """Response model for repository information."""

    id: str
    name: str
    full_name: str
    description: str | None
    platform: str
    clone_url: str
    ssh_url: str | None
    default_branch: str
    language: str | None
    size: int
    branch_count: int
    commit_count: int
    status: str
    last_sync: datetime | None
    created_at: datetime
    updated_at: datetime


class CloneRequest(BaseModel):
    """Request model for repository cloning."""

    branch: str | None = Field(None, description="Branch to clone")
    depth: int = Field(1, description="Clone depth")
    recursive: bool = Field(False, description="Clone submodules recursively")
    lfs: bool = Field(True, description="Include Git LFS files")


class CloneResponse(BaseModel):
    """Response model for clone operations."""

    job_id: str
    status: str
    progress: float
    estimated_completion: datetime | None


class SyncRequest(BaseModel):
    """Request model for repository synchronization."""

    force: bool = Field(False, description="Force synchronization")
    prune: bool = Field(True, description="Prune deleted branches")
    branches: list[str] | None = Field(None, description="Specific branches to sync")


class SyncResponse(BaseModel):
    """Response model for sync operations."""

    job_id: str
    status: str
    changes_detected: bool
    started_at: datetime


class BranchRequest(BaseModel):
    """Request model for branch operations."""

    name: str = Field(..., description="Branch name")
    base: str = Field(..., description="Base branch")
    protected: bool = Field(False, description="Whether branch is protected")


class BranchResponse(BaseModel):
    """Response model for branch information."""

    name: str
    commit_sha: str
    commit_message: str
    author: str
    created_at: datetime
    updated_at: datetime
    protected: bool
    ahead: int
    behind: int
    repository_id: str


class MergeRequest(BaseModel):
    """Request model for merge operations."""

    target_branch: str = Field(..., description="Target branch for merge")
    message: str | None = Field(None, description="Merge commit message")
    strategy: str = Field("merge", description="Merge strategy")
    delete_source: bool = Field(False, description="Delete source branch after merge")


class MergeResponse(BaseModel):
    """Response model for merge operations."""

    merge_commit_sha: str | None
    merged: bool
    message: str
    conflicts: list[str]


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str
    timestamp: datetime
    components: dict[str, dict[str, Any]]


class SystemStatusResponse(BaseModel):
    """Response model for system status."""

    version: str
    uptime: int
    repository_count: int
    active_workflows: int
    storage_usage: dict[str, Any]
    git_version: str


# Dependency functions
def get_repo_manager() -> RepoManager:
    """Get repository manager instance."""
    if repo_manager is None:
        raise HTTPException(
            status_code=500, detail="Repository manager not initialized"
        )
    return repo_manager


def get_github_api() -> GitHubAPI:
    """Get GitHub API instance."""
    if github_api is None:
        raise HTTPException(status_code=500, detail="GitHub API not initialized")
    return github_api


def get_logger() -> Any:
    """Get logger instance."""
    if logger is None:
        raise HTTPException(status_code=500, detail="Logger not initialized")
    return logger


# API Endpoints


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).replace(tzinfo=None),
        components={
            "repository_manager": {"status": "up" if repo_manager else "down"},
            "github_api": {"status": "up" if github_api else "down"},
            "database": {"status": "up"},
        },
    )


@app.get("/repositories", response_model=dict[str, Any], tags=["repositories"])
async def list_repositories(
    status_filter: str | None = None,
    platform: str | None = None,
    organization: str | None = None,
    language: str | None = None,
    manager: RepoManager = Depends(get_repo_manager),
) -> dict[str, Any]:
    """List managed repositories with optional filtering.

    Note: status semantics are filesystem-derived — ``active`` means the
    directory exists and contains a ``.git`` folder, ``error`` the opposite.
    There is no persistent status tracking; repository records are
    process-local in-memory state and are lost on restart.
    ``platform``/``organization``/``language`` filters are accepted but not
    applied to filesystem-backed repository discovery.
    """
    try:
        # Get all repositories
        all_repos = manager._get_all_repo_paths()

        # Apply filters
        filtered_repos = {}
        for name, path in all_repos.items():
            include_repo = True

            # Status filter (simplified - would need more sophisticated status tracking)
            if status_filter:
                if status_filter == "active" and not (path / ".git").exists():
                    include_repo = False
                elif status_filter == "error":
                    include_repo = not path.exists() or not (path / ".git").exists()
                elif status_filter not in {"active", "error"}:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported status_filter: {status_filter}",
                    )

            if include_repo:
                filtered_repos[name] = {
                    "path": str(path),
                    "exists": path.exists(),
                    "is_git": (path / ".git").exists(),
                }

        return {"repositories": filtered_repos, "total": len(filtered_repos)}

    except HTTPException:
        raise


@app.post("/repositories", response_model=RepositoryResponse, tags=["repositories"])
async def add_repository(
    request: RepositoryRequest,
    background_tasks: BackgroundTasks,
    manager: RepoManager = Depends(get_repo_manager),
) -> RepositoryResponse:
    """Add a new repository to the management system."""
    try:
        # Validate repository URL
        if request.platform == "github" and not request.clone_url.startswith(
            "https://github.com/"
        ):
            raise HTTPException(status_code=400, detail="Invalid GitHub repository URL")

        # Create repository entry
        repo_id = f"{request.platform}_{request.name or 'unknown'}"
        now = datetime.now(UTC).replace(tzinfo=None)
        repo_data = {
            "id": repo_id,
            "name": request.name or "unknown",
            "full_name": request.clone_url.split("/")[-2]
            + "/"
            + request.clone_url.split("/")[-1].replace(".git", ""),
            "description": request.description,
            "platform": request.platform,
            "clone_url": request.clone_url,
            "ssh_url": None,
            "default_branch": manager.config.get("repositories", {}).get(
                "default_branch", "main"
            ),
            "language": None,
            "size": 0,
            "branch_count": 0,
            "commit_count": 0,
            "status": "pending",
            "last_sync": None,
            "created_at": now,
            "updated_at": now,
        }
        repository_records[repo_id] = repo_data

        # Start clone operation in background
        background_tasks.add_task(clone_repository_background, repo_data, request)

        return RepositoryResponse(**repo_data)

    except HTTPException:
        raise


@app.get(
    "/repositories/{repo_id}", response_model=RepositoryResponse, tags=["repositories"]
)
async def get_repository(
    repo_id: str, manager: RepoManager = Depends(get_repo_manager)
) -> RepositoryResponse:
    """Get detailed information about a specific repository."""
    try:
        record = repository_records.get(repo_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Repository {repo_id} not found"
            )
        return RepositoryResponse(**record)

    except HTTPException:
        raise


@app.post(
    "/repositories/{repo_id}/clone", response_model=CloneResponse, tags=["repositories"]
)
async def clone_repository(
    repo_id: str,
    request: CloneRequest,
    background_tasks: BackgroundTasks,
    manager: RepoManager = Depends(get_repo_manager),
) -> CloneResponse:
    """Clone a repository."""
    try:
        if repo_id not in repository_records:
            raise HTTPException(
                status_code=404, detail=f"Repository {repo_id} not found"
            )
        # Generate job ID
        job_id = f"clone_{repo_id}_{int(time.time())}"

        # Start clone operation in background
        background_tasks.add_task(clone_repository_background, repo_id, request)

        return CloneResponse(
            job_id=job_id,
            status="queued",
            progress=0.0,
            estimated_completion=datetime.now(UTC).replace(tzinfo=None),
        )

    except HTTPException:
        raise


@app.post(
    "/repositories/{repo_id}/sync", response_model=SyncResponse, tags=["repositories"]
)
async def sync_repository(
    repo_id: str,
    request: SyncRequest,
    background_tasks: BackgroundTasks,
    manager: RepoManager = Depends(get_repo_manager),
) -> SyncResponse:
    """Synchronize a repository."""
    try:
        if repo_id not in repository_records:
            raise HTTPException(
                status_code=404, detail=f"Repository {repo_id} not found"
            )
        # Generate job ID
        job_id = f"sync_{repo_id}_{int(time.time())}"

        # Start sync operation in background
        background_tasks.add_task(sync_repository_background, repo_id, request)

        return SyncResponse(
            job_id=job_id,
            status="queued",
            changes_detected=False,  # Would be determined during sync
            started_at=datetime.now(UTC).replace(tzinfo=None),
        )

    except HTTPException:
        raise


@app.get(
    "/repositories/{repo_id}/branches", response_model=dict[str, Any], tags=["branches"]
)
async def list_branches(
    repo_id: str,
    status_filter: str | None = None,
    protected: bool | None = None,
    manager: RepoManager = Depends(get_repo_manager),
) -> dict[str, Any]:
    """List branches for a repository."""
    try:
        branches = manager.list_branches(repo_id)
        if protected is not None:
            branches = [
                branch
                for branch in branches
                if branch.get("protected", False) == protected
            ]
        if status_filter:
            if status_filter not in {"active", "protected"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported branch status: {status_filter}",
                )
            if status_filter == "protected":
                branches = [
                    branch for branch in branches if branch.get("protected", False)
                ]
        return {"branches": branches, "total": len(branches)}

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/repositories/{repo_id}/branches", response_model=BranchResponse, tags=["branches"]
)
async def create_branch(
    repo_id: str,
    request: BranchRequest,
    manager: RepoManager = Depends(get_repo_manager),
) -> BranchResponse:
    """Create a new branch."""
    try:
        branch = manager.create_branch_for_repository(
            repo_id, request.name, request.base, protected=request.protected
        )
        branch["repository_id"] = repo_id
        return BranchResponse(**cast(dict[str, Any], branch))
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise


@app.post(
    "/repositories/{repo_id}/branches/{branch_name}/merge",
    response_model=MergeResponse,
    tags=["branches"],
)
async def merge_branch(
    repo_id: str,
    branch_name: str,
    request: MergeRequest,
    manager: RepoManager = Depends(get_repo_manager),
) -> MergeResponse:
    """Merge a branch."""
    try:
        return MergeResponse(
            **cast(
                dict[str, Any],
                manager.merge_branch(
                    repo_id,
                    branch_name,
                    request.target_branch,
                    strategy=request.strategy,
                    message=request.message,
                    delete_source=request.delete_source,
                ),
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/system/status", response_model=SystemStatusResponse, tags=["system"])
async def get_system_status() -> SystemStatusResponse:
    """Get comprehensive system status."""
    import git

    records: dict[str, Any] = repo_manager.check_repo_status() if repo_manager else {}
    active_repositories = sum(1 for value in records.values() if "error" not in value)
    return SystemStatusResponse(
        version="1.0.0",
        uptime=0,
        repository_count=active_repositories,
        active_workflows=0,
        storage_usage={"tracked_records": len(repository_records)},
        git_version=git.__version__,
    )


# Background task functions
async def clone_repository_background(
    repo_id: str | dict[str, Any],
    clone_request: RepositoryRequest | CloneRequest,
) -> None:
    """Background task for repository cloning."""
    try:
        if repo_manager is None:
            raise RuntimeError("Repository manager is not initialized")
        if isinstance(repo_id, dict):
            record = repo_id
            identifier = record["id"]
            clone_config: dict[str, Any] = {
                "url": record["clone_url"],
                "name": record["name"],
                "branch": getattr(clone_request, "branch", None),
                "depth": getattr(clone_request, "depth", None),
            }
        else:
            repo_record = repository_records.get(repo_id)
            if repo_record is None:
                raise FileNotFoundError(f"Repository not found: {repo_id}")
            identifier = repo_id
            clone_config = {
                "url": repo_record["clone_url"],
                "name": repo_record["name"],
                "branch": getattr(clone_request, "branch", None),
                "depth": getattr(clone_request, "depth", None),
            }
        if logger:
            logger.info("Starting background clone for %s", identifier)
        result = repo_manager.clone_repositories([clone_config], parallel=False)
        success = bool(result.get(clone_config["name"]))
        updated_record = repository_records.get(identifier)
        if updated_record is not None:
            updated_record["status"] = "active" if success else "error"
            updated_record["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
            if success:
                branches = repo_manager.list_branches(clone_config["name"])
                updated_record["branch_count"] = len(branches)
                updated_record["commit_count"] = sum(
                    1
                    for _ in git.Repo(
                        repo_manager._resolve_repo_path(clone_config["name"])
                    ).iter_commits()
                )
    except Exception as e:
        if isinstance(repo_id, str) and repo_id in repository_records:
            repository_records[repo_id]["status"] = "error"
        if logger:
            logger.error("Background clone failed for %s: %s", repo_id, e)


async def sync_repository_background(repo_id: str, sync_request: SyncRequest) -> None:
    """Background task for repository synchronization."""
    try:
        if repo_manager is None:
            raise RuntimeError("Repository manager is not initialized")
        record = repository_records.get(repo_id)
        if record is None:
            raise FileNotFoundError(f"Repository not found: {repo_id}")
        if logger:
            logger.info("Starting background sync for %s", repo_id)
        repo_name = record["name"]
        result = repo_manager.sync_repositories([repo_name])
        success = bool(result.get(repo_name))
        record["status"] = "active" if success else "error"
        record["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
        if success:
            record["last_sync"] = record["updated_at"]
    except Exception as e:
        if repo_id in repository_records:
            repository_records[repo_id]["status"] = "error"
        if logger:
            logger.error("Background sync failed for %s: %s", repo_id, e)


# Initialization function
def initialize_api(config_path: str | None = None) -> None:
    """Initialize the API with configuration."""
    global repo_manager, github_api, config_loader, logger

    # Load configuration
    config_loader = ConfigLoader(config_path)

    # Load clone configuration
    clone_config = config_loader.load_clone_config()

    # Initialize components
    repo_manager = RepoManager()
    github_api = GitHubAPI(
        token=clone_config.github_token,
        api_url=clone_config.github_api_url,
        wait_on_rate_limit=clone_config.github_wait_on_rate_limit,
        max_retries=clone_config.github_max_retries,
        retry_delay=clone_config.github_retry_delay,
    )

    # Setup logging
    logger = setup_logging(
        {
            "level": clone_config.log_level,
            "format": clone_config.log_format,
            "file": f"{clone_config.output_dir}/api.log",
        }
    )


def run_api(
    host: str = "0.0.0.0", port: int = 8000, config_path: str | None = None
) -> None:
    """Run the FastAPI server."""
    initialize_api(config_path)

    uvicorn.run(
        "geo_infer_git.api.rest_api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    run_api()
