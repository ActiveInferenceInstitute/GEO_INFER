#!/usr/bin/env python3

"""
Repository cloning functionality for GEO-INFER-GIT.

This module provides functionality to clone GitHub repositories
using Git commands with various options, including parallel cloning,
progress tracking, and error handling.
"""

import os
import logging
import shutil
import threading
from typing import Any, cast
from concurrent.futures import ThreadPoolExecutor, as_completed, Executor
import time
from pathlib import Path
import git

from ..utils.config_loader import CloneConfig
from .github_api import GitHubRepository

logger = logging.getLogger(__name__)


class CloneProgress:
    """Track cloning progress and statistics."""

    def __init__(self) -> None:
        self.total_repos = 0
        self.completed_repos = 0
        self.failed_repos = 0
        self.skipped_repos = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def increment_completed(self) -> None:
        """Increment completed repository count."""
        with self.lock:
            self.completed_repos += 1

    def increment_failed(self) -> None:
        """Increment failed repository count."""
        with self.lock:
            self.failed_repos += 1

    def increment_skipped(self) -> None:
        """Increment skipped repository count."""
        with self.lock:
            self.skipped_repos += 1

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_repos == 0:
            return 0.0
        return (self.completed_repos / self.total_repos) * 100.0

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        return {
            "total": self.total_repos,
            "completed": self.completed_repos,
            "failed": self.failed_repos,
            "skipped": self.skipped_repos,
            "elapsed_time": self.elapsed_time,
            "success_rate": self.success_rate,
        }


class RepoCloner:
    """
    Repository cloner with parallel execution and progress tracking.

    Provides functionality for:
    - Cloning repositories with various options
    - Parallel cloning for improved performance
    - Progress tracking and statistics
    - Error handling and recovery
    - Git LFS support
    - Sparse checkout support
    """

    def __init__(self, config: CloneConfig):
        """
        Initialize the repository cloner.

        Args:
            config: CloneConfig object with cloning parameters
        """
        self.config = config
        self.progress = CloneProgress()

        # Create output directory if it doesn't exist
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up Git environment
        self.git_env = os.environ.copy()
        if config.github_token:
            # For GitHub token authentication via HTTPS
            self.git_env["GIT_TOKEN"] = config.github_token

    def clone_repository(
        self, owner: str, repo: str, branch: str | None = None
    ) -> bool:
        """
        Clone a single repository.

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch to clone (uses default if None)

        Returns:
            True if successful, False otherwise
        """
        repo_path = self.output_dir / owner / repo
        clone_url = f"https://github.com/{owner}/{repo}.git"

        # Check if repository already exists
        if repo_path.exists():
            if (repo_path / ".git").exists():
                logger.info(f"Repository {owner}/{repo} already exists, skipping")
                self.progress.increment_skipped()
                return True
            else:
                # Remove non-git directory
                shutil.rmtree(repo_path)

        # Create parent directory
        repo_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Cloning {owner}/{repo} to {repo_path}")

            # Prepare clone options
            clone_kwargs: dict[str, Any] = {
                "branch": branch or self.config.default_branch,
                "depth": self.config.clone_depth,
                "recursive": False,  # We'll handle submodules separately if needed
            }

            # Handle authentication
            if self.config.auth_method == "token" and self.config.github_token:
                # Use token in URL for GitHub
                if "github.com" in clone_url:
                    clone_url = clone_url.replace(
                        "https://", f"https://{self.config.github_token}@"
                    )

            # Perform the clone
            git_repo = git.Repo.clone_from(clone_url, repo_path, **clone_kwargs)

            # Handle LFS if configured
            if self.config.github_token:  # Assume LFS support if authenticated
                self._setup_lfs(git_repo)

            logger.info(f"Successfully cloned {owner}/{repo}")
            self.progress.increment_completed()
            return True

        except git.GitCommandError as e:
            logger.error(f"Git error while cloning {owner}/{repo}: {e}")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            self.progress.increment_failed()
            return False
        except Exception as e:
            logger.error(f"Unexpected error while cloning {owner}/{repo}: {e}")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            self.progress.increment_failed()
            return False

    def clone_repositories_for_user(
        self, username: str, repositories: list[GitHubRepository]
    ) -> tuple[int, int]:
        """
        Clone repositories for a specific user.

        Args:
            username: GitHub username
            repositories: List of GitHubRepository objects to clone

        Returns:
            Tuple of (success_count, total_count)
        """
        self.progress.total_repos = len(repositories)
        success_count = 0

        if self.config.concurrency_enabled and len(repositories) > 1:
            # Parallel cloning
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_repo = {
                    executor.submit(self._clone_repo_from_github_obj, repo): repo
                    for repo in repositories
                }

                for future in as_completed(future_to_repo):
                    if future.result():
                        success_count += 1
        else:
            # Sequential cloning
            for repo in repositories:
                if self._clone_repo_from_github_obj(repo):
                    success_count += 1

        return success_count, len(repositories)

    def _clone_repo_from_github_obj(self, github_repo: GitHubRepository) -> bool:
        """
        Clone a repository from GitHub API object.

        Args:
            github_repo: GitHubRepository object

        Returns:
            True if successful, False otherwise
        """
        return self.clone_repository(
            github_repo.owner, github_repo.name, github_repo.default_branch
        )

    def _setup_lfs(self, repo: git.Repo) -> None:
        """
        Set up Git LFS for a repository if needed.

        Args:
            repo: Git repository object
        """
        try:
            # Check if repository uses LFS
            lfs_config = repo.git.config(
                "--get", "filter.lfs.required", with_exceptions=False
            )
            if lfs_config:
                logger.info("Setting up Git LFS")
                # Pull LFS objects
                repo.git.lfs("pull", "origin")
        except git.GitCommandError:
            # LFS not available or not needed
            pass
        except Exception as e:
            logger.warning(f"Error setting up LFS: {e}")

    def clone_multiple_repositories(
        self, repositories: list[tuple[str, str, str]]
    ) -> dict[str, bool]:
        """
        Clone multiple repositories in parallel.

        Args:
            repositories: List of (owner, repo, branch) tuples

        Returns:
            Dictionary mapping repository names to success status
        """
        self.progress.total_repos = len(repositories)
        results = {}

        if self.config.concurrency_enabled and len(repositories) > 1:
            # Parallel cloning
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_repo = {
                    executor.submit(
                        self.clone_repository, owner, repo, branch
                    ): f"{owner}/{repo}"
                    for owner, repo, branch in repositories
                }

                for future in as_completed(future_to_repo):
                    repo_name = future_to_repo[future]
                    try:
                        success = future.result()
                        results[repo_name] = success
                    except Exception as e:
                        logger.error(f"Exception while cloning {repo_name}: {e}")
                        results[repo_name] = False
        else:
            # Sequential cloning
            for owner, repo, branch in repositories:
                repo_name = f"{owner}/{repo}"
                try:
                    success = self.clone_repository(owner, repo, branch)
                    results[repo_name] = success
                except Exception as e:
                    logger.error(f"Exception while cloning {repo_name}: {e}")
                    results[repo_name] = False

        return results

    def get_disk_usage(self) -> dict[str, Any]:
        """
        Get disk usage information for the output directory.

        Returns:
            Dictionary with disk usage statistics
        """
        try:
            usage = shutil.disk_usage(self.output_dir)
            return {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "used_percent": (usage.used / usage.total) * 100
                if usage.total > 0
                else 0,
            }
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return {}

    def cleanup_failed_clones(self) -> int:
        """
        Clean up partially cloned repositories left by failed clone attempts.

        Only ``owner/repo``-shaped directories (exactly two levels under the
        output directory, matching where :meth:`clone_repository` places
        clones) that lack a ``.git`` folder are removed. Any other files or
        directories in the output directory are left untouched.

        Returns:
            Number of directories cleaned up
        """
        cleaned_count = 0

        try:
            for owner_dir in self.output_dir.iterdir():
                if not owner_dir.is_dir() or owner_dir.name.startswith("."):
                    continue
                for repo_dir in owner_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue
                    git_dir = repo_dir / ".git"
                    if not git_dir.exists():
                        # owner/repo-shaped directory without .git: a failed
                        # or interrupted clone attempt, safe to remove.
                        try:
                            shutil.rmtree(repo_dir)
                            cleaned_count += 1
                            logger.info(f"Cleaned up failed clone: {repo_dir}")
                        except Exception as e:
                            logger.warning(f"Error cleaning up {repo_dir}: {e}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        return cleaned_count

    def get_clone_stats(self) -> dict[str, Any]:
        """
        Get cloning statistics and progress.

        Returns:
            Dictionary with cloning statistics
        """
        return self.progress.get_stats()

    def reset_progress(self) -> None:
        """Reset progress tracking."""
        self.progress = CloneProgress()

    def close(self) -> None:
        """Clean up resources and reset internal state."""
        self.reset_progress()
        executor: Any = getattr(self, "_executor", None)
        if executor is not None:
            cast(Executor, executor).shutdown(wait=False)
            self._executor = None

    def estimate_clone_time(
        self, repo_count: int, avg_repo_size: int = 50000
    ) -> dict[str, Any]:
        """
        Estimate cloning time based on repository count and average size.

        Args:
            repo_count: Number of repositories to clone
            avg_repo_size: Average repository size in KB

        Returns:
            Dictionary with time estimates
        """
        # Rough estimates based on typical Git operations
        base_clone_time = 10  # seconds for basic clone setup
        per_kb_time = 0.001  # additional time per KB
        parallel_overhead = 2  # seconds per repository for parallel coordination

        if self.config.concurrency_enabled:
            # Parallel cloning estimate
            workers = min(self.config.max_workers, repo_count)
            parallel_time = (
                base_clone_time + (avg_repo_size * per_kb_time) + parallel_overhead
            ) * (repo_count / workers)
            sequential_time = (
                base_clone_time + (avg_repo_size * per_kb_time)
            ) * repo_count

            return {
                "parallel_estimate_seconds": parallel_time,
                "sequential_estimate_seconds": sequential_time,
                "recommended_approach": "parallel"
                if parallel_time < sequential_time
                else "sequential",
                "workers_used": workers,
                "speedup_factor": sequential_time / parallel_time
                if parallel_time > 0
                else 1,
            }
        else:
            # Sequential cloning estimate
            sequential_time = (
                base_clone_time + (avg_repo_size * per_kb_time)
            ) * repo_count

            return {
                "sequential_estimate_seconds": sequential_time,
                "recommended_approach": "sequential",
            }
