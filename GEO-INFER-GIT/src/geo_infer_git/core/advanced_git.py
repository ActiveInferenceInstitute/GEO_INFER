#!/usr/bin/env python3

"""
Advanced Git operations for GEO-INFER-GIT.

This module provides sophisticated Git functionality including:
- Submodule management and synchronization
- Cherry-picking and advanced merging strategies
- Conflict resolution and merge strategies
- Interactive rebasing and history manipulation
- Advanced branch management and workflows
"""

import re
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
import git

from ..utils.error_handler import GitOperationError
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _git_path_exists(repo: "git.Repo", repo_path: Path, git_file: str) -> bool:
    """Return True if a path reported by ``git rev-parse --git-path`` exists.

    Version-robust in-progress detection: state files such as
    ``CHERRY_PICK_HEAD`` or the ``rebase-merge``/``rebase-apply``/``sequencer``
    directories exist exactly while the corresponding operation is mid-flight,
    independent of human-readable git output wording.
    """
    try:
        reported = repo.git.rev_parse("--git-path", git_file).strip()
    except git.GitCommandError:
        return False
    if not reported:
        return False
    candidate = Path(reported)
    if not candidate.is_absolute():
        candidate = repo_path / candidate
    return candidate.exists()


@dataclass
class SubmoduleInfo:
    """Information about a Git submodule."""

    path: str
    url: str
    branch: str = "main"
    commit: str = ""
    status: str = "unknown"  # initialized, updated, dirty, missing
    recursive: bool = False


@dataclass
class MergeConflict:
    """Information about a merge conflict."""

    file_path: str
    conflict_type: str  # "both_modified", "deleted_by_us", "deleted_by_them", etc.
    our_content: str = ""
    their_content: str = ""
    ancestor_content: str = ""
    resolution: str = ""  # "ours", "theirs", "manual", "resolved"


@dataclass
class CherryPickOperation:
    """Information about a cherry-pick operation."""

    commit_sha: str
    status: str = "pending"  # pending, applied, skipped, failed
    message: str = ""
    conflicts: list[MergeConflict] = field(default_factory=list)


@dataclass
class RebaseOperation:
    """Information about a rebase operation."""

    base_commit: str
    target_branch: str
    status: str = "pending"  # pending, in_progress, completed, failed, aborted
    message: str = ""
    current_step: int = 0
    total_steps: int = 0
    conflicts: list[MergeConflict] = field(default_factory=list)


class SubmoduleManager:
    """
    Advanced submodule management for Git repositories.

    Provides functionality for:
    - Submodule initialization and synchronization
    - Recursive submodule handling
    - Submodule status monitoring
    - Submodule dependency resolution
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize submodule manager.

        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.submodules: dict[str, SubmoduleInfo] = {}

        # Load existing submodules
        self._load_submodules()

    def _load_submodules(self) -> None:
        """Load existing submodules from .gitmodules file."""
        gitmodules_path = self.repo_path / ".gitmodules"

        if not gitmodules_path.exists():
            return

        try:
            import configparser

            config = configparser.ConfigParser()
            config.read(gitmodules_path)

            for section in config.sections():
                if section.startswith("submodule "):
                    submodule_name = section[10:].strip('"')
                    submodule_path = config.get(section, "path")
                    submodule_url = config.get(section, "url")

                    self.submodules[submodule_name] = SubmoduleInfo(
                        path=submodule_path, url=submodule_url
                    )

        except Exception as e:
            logger.warning(f"Error loading submodules: {e}")

    def initialize_submodules(self, recursive: bool = False) -> dict[str, bool]:
        """
        Initialize all submodules in the repository.

        Args:
            recursive: Whether to initialize submodules recursively

        Returns:
            Dictionary mapping submodule names to success status
        """
        results = {}

        try:
            # Initialize submodules
            self.repo.git.submodule(
                "update", "--init", "--recursive" if recursive else "--init"
            )

            # Update submodule information
            for name, submodule in self.submodules.items():
                submodule_path = self.repo_path / submodule.path
                if submodule_path.exists():
                    try:
                        sub_repo = git.Repo(submodule_path)
                        submodule.commit = sub_repo.head.commit.hexsha
                        submodule.status = "initialized"
                        results[name] = True
                    except Exception:
                        submodule.status = "error"
                        results[name] = False
                else:
                    submodule.status = "missing"
                    results[name] = False

        except git.GitCommandError as e:
            logger.error(f"Error initializing submodules: {e}")
            raise GitOperationError(f"Submodule initialization failed: {e}") from e

        return results

    def update_submodules(self, recursive: bool = False) -> dict[str, bool]:
        """
        Update all submodules to their latest commits.

        Args:
            recursive: Whether to update submodules recursively

        Returns:
            Dictionary mapping submodule names to success status
        """
        results = {}

        try:
            # Update submodules
            self.repo.git.submodule(
                "update", "--remote", "--recursive" if recursive else "--remote"
            )

            # Update submodule information
            for name, submodule in self.submodules.items():
                submodule_path = self.repo_path / submodule.path
                if submodule_path.exists():
                    try:
                        sub_repo = git.Repo(submodule_path)
                        submodule.commit = sub_repo.head.commit.hexsha
                        submodule.status = "updated"

                        # Check if submodule is dirty
                        if sub_repo.is_dirty():
                            submodule.status = "dirty"

                        results[name] = True
                    except Exception:
                        submodule.status = "error"
                        results[name] = False
                else:
                    submodule.status = "missing"
                    results[name] = False

        except git.GitCommandError as e:
            logger.error(f"Error updating submodules: {e}")
            raise GitOperationError(f"Submodule update failed: {e}") from e

        return results

    def sync_submodules(self, recursive: bool = False) -> dict[str, bool]:
        """
        Synchronize submodules with their remote repositories.

        Args:
            recursive: Whether to sync submodules recursively

        Returns:
            Dictionary mapping submodule names to success status
        """
        results = {}

        for name, submodule in self.submodules.items():
            try:
                submodule_path = self.repo_path / submodule.path

                if not submodule_path.exists():
                    results[name] = False
                    continue

                # Sync individual submodule
                sub_repo = git.Repo(submodule_path)

                # Fetch latest changes
                for remote in sub_repo.remotes:
                    remote.fetch()

                # Update to latest commit on tracked branch
                if getattr(sub_repo.head, "is_tracking", False):
                    sub_repo.git.pull("origin", sub_repo.active_branch.name)

                submodule.status = "synced"
                results[name] = True

            except Exception as e:
                logger.error(f"Error syncing submodule {name}: {e}")
                submodule.status = "error"
                results[name] = False

        return results

    def get_submodule_status(self) -> dict[str, dict[str, Any]]:
        """
        Get detailed status of all submodules.

        Returns:
            Dictionary with submodule status information
        """
        status_info = {}

        for name, submodule in self.submodules.items():
            submodule_path = self.repo_path / submodule.path

            info = {
                "path": submodule.path,
                "url": submodule.url,
                "branch": submodule.branch,
                "commit": submodule.commit,
                "status": submodule.status,
                "exists": submodule_path.exists(),
                "is_git_repo": (submodule_path / ".git").exists()
                if submodule_path.exists()
                else False,
            }

            if submodule_path.exists():
                try:
                    sub_repo = git.Repo(submodule_path)
                    info["is_dirty"] = sub_repo.is_dirty()
                    info["ahead_behind"] = self._get_ahead_behind(sub_repo)
                    info["remotes"] = [remote.name for remote in sub_repo.remotes]

                    # Get submodule dependencies if recursive
                    if submodule.recursive:
                        info["dependencies"] = self._get_submodule_dependencies(
                            submodule_path
                        )

                except Exception as e:
                    info["error"] = str(e)

            status_info[name] = info

        return status_info

    def _get_ahead_behind(self, repo: git.Repo) -> dict[str, int]:
        """Get ahead/behind information for a repository."""
        try:
            if repo.remotes and getattr(repo.head, "is_tracking", False):
                tracking = getattr(repo.head, "tracking_branch", lambda: None)()
                if tracking is not None:
                    remote_name = tracking.remote_name
                    remote_ref = f"{remote_name}/{repo.active_branch.name}"

                    ahead = len(list(repo.iter_commits(f"{remote_ref}..HEAD")))
                    behind = len(list(repo.iter_commits(f"HEAD..{remote_ref}")))

                    return {"ahead": ahead, "behind": behind}
        except Exception as exc:
            logger.warning("Could not determine ahead/behind status: %s", exc)

        return {"ahead": 0, "behind": 0}

    def _get_submodule_dependencies(self, submodule_path: Path) -> list[str]:
        """Get dependencies of a submodule by scanning its .gitmodules file.

        Returns the list of nested submodule paths declared in
        ``<submodule_path>/.gitmodules``.
        """
        dependencies: list[str] = []
        gitmodules = submodule_path / ".gitmodules"
        if not gitmodules.exists():
            return dependencies
        try:
            import configparser

            config = configparser.ConfigParser()
            config.read(gitmodules)
            for section in config.sections():
                if section.startswith("submodule "):
                    sub_path = config.get(section, "path", fallback=None)
                    if sub_path:
                        dependencies.append(sub_path)
        except Exception as e:
            logger.warning(
                f"Could not parse submodule dependencies from {gitmodules}: {e}"
            )
        return dependencies


class CherryPickManager:
    """
    Advanced cherry-picking functionality for selective commit application.

    Provides functionality for:
    - Cherry-picking individual commits
    - Handling merge conflicts during cherry-pick
    - Batch cherry-picking operations
    - Conflict resolution strategies
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize cherry-pick manager.

        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.operations: list[CherryPickOperation] = []

    def cherry_pick_commit(
        self, commit_sha: str, strategy: str = "recursive"
    ) -> CherryPickOperation:
        """
        Cherry-pick a specific commit.

        Args:
            commit_sha: SHA of commit to cherry-pick
            strategy: Merge strategy for conflict resolution

        Returns:
            CherryPickOperation result
        """
        operation = CherryPickOperation(commit_sha=commit_sha)

        try:
            # Get the commit object
            self.repo.commit(commit_sha)

            # Perform cherry-pick
            self.repo.git.cherry_pick(commit_sha, strategy=strategy)

            operation.status = "applied"
            operation.message = f"Successfully cherry-picked {commit_sha}"

        except git.GitCommandError as e:
            if self._cherry_pick_in_progress():
                operation.status = "conflicts"
                operation.conflicts = self._detect_conflicts()
                operation.message = (
                    f"Conflicts detected during cherry-pick of {commit_sha}"
                )
            else:
                operation.status = "failed"
                operation.message = f"Cherry-pick failed: {e}"

        except Exception as e:
            operation.status = "failed"
            operation.message = f"Error during cherry-pick: {e}"

        self.operations.append(operation)
        return operation

    def cherry_pick_range(
        self, start_sha: str, end_sha: str, stop_on_conflict: bool = True
    ) -> list[CherryPickOperation]:
        """
        Cherry-pick a range of commits.

        Args:
            start_sha: Starting commit SHA (exclusive)
            end_sha: Ending commit SHA (inclusive)
            stop_on_conflict: Whether to stop on first conflict

        Returns:
            List of CherryPickOperation results
        """
        operations = []

        try:
            # Get commits in range
            commits = list(self.repo.iter_commits(f"{start_sha}..{end_sha}"))

            for commit in commits:
                operation = self.cherry_pick_commit(commit.hexsha)

                if operation.status == "conflicts" and stop_on_conflict:
                    break

                operations.append(operation)

        except Exception as e:
            logger.error(f"Error in cherry-pick range: {e}")
            # Add failed operation
            failed_op = CherryPickOperation(
                commit_sha=end_sha,
                status="failed",
                message=f"Range cherry-pick failed: {e}",
            )
            operations.append(failed_op)

        return operations

    def resolve_conflicts(
        self, operation_index: int, resolution_strategy: str = "ours"
    ) -> bool:
        """
        Resolve conflicts in a cherry-pick operation.

        Args:
            operation_index: Index of the operation with conflicts
            resolution_strategy: Strategy for conflict resolution

        Returns:
            True if conflicts were resolved successfully
        """
        if operation_index >= len(self.operations):
            raise ValueError(f"Operation index {operation_index} out of range")

        operation = self.operations[operation_index]

        if operation.status != "conflicts":
            logger.warning(f"Operation {operation_index} has no conflicts to resolve")
            return True

        try:
            # Apply resolution strategy
            if resolution_strategy == "ours":
                # Keep our changes, discard theirs
                for conflict in operation.conflicts:
                    if conflict.conflict_type == "both_modified":
                        # Write our content
                        conflict_file = self.repo_path / conflict.file_path
                        conflict_file.write_text(conflict.our_content)

            elif resolution_strategy == "theirs":
                # Keep their changes, discard ours
                for conflict in operation.conflicts:
                    if conflict.conflict_type == "both_modified":
                        # Write their content
                        conflict_file = self.repo_path / conflict.file_path
                        conflict_file.write_text(conflict.their_content)

            elif resolution_strategy == "manual":
                # Mark as manually resolved
                logger.info(
                    f"Manual conflict resolution required for operation {operation_index}"
                )
                return False

            # Add resolved files
            self.repo.git.add(".")

            # Continue cherry-pick
            self.repo.git.cherry_pick("--continue")

            operation.status = "applied"
            operation.message = (
                f"Conflicts resolved using {resolution_strategy} strategy"
            )

            return True

        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")
            operation.message = f"Conflict resolution failed: {e}"
            return False

    def _cherry_pick_in_progress(self) -> bool:
        """Detect an in-progress cherry-pick via git state files, not output text.

        ``CHERRY_PICK_HEAD`` (and the ``sequencer`` directory on newer git)
        exist exactly while a cherry-pick is stopped on a conflict, regardless
        of stderr wording across git versions.
        """
        return _git_path_exists(
            self.repo, self.repo_path, "CHERRY_PICK_HEAD"
        ) or _git_path_exists(self.repo, self.repo_path, "sequencer")

    def _detect_conflicts(self) -> list[MergeConflict]:
        """Detect current merge conflicts in the repository."""
        conflicts = []

        try:
            # Get git status to find conflicted files
            status_output = self.repo.git.status("--porcelain")

            for line in status_output.split("\n"):
                if line.strip():
                    status_code = line[:2]
                    file_path = line[3:]

                    if (
                        status_code.startswith("UU")
                        or status_code.startswith("AA")
                        or status_code.startswith("DD")
                    ):
                        conflict = self._analyze_conflict(file_path, status_code)
                        if conflict:
                            conflicts.append(conflict)

        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")

        return conflicts

    def _analyze_conflict(
        self, file_path: str, status_code: str
    ) -> MergeConflict | None:
        """Analyze a specific conflict in detail."""
        try:
            conflict_file = self.repo_path / file_path

            # Read conflict markers
            with open(conflict_file) as f:
                content = f.read()

            # Parse conflict markers
            our_pattern = r"<<<<<<< HEAD\n(.*?)\n======="
            their_pattern = r"=======\n(.*?)\n>>>>>>> "

            our_match = re.search(our_pattern, content, re.DOTALL)
            their_match = re.search(their_pattern, content, re.DOTALL)

            our_content = our_match.group(1) if our_match else ""
            their_content = their_match.group(1) if their_match else ""

            return MergeConflict(
                file_path=file_path,
                conflict_type="both_modified",
                our_content=our_content,
                their_content=their_content,
            )

        except OSError as exc:
            logger.warning(
                "Could not read conflict markers from %s: %s", file_path, exc
            )
            return None


class RebaseManager:
    """
    Advanced rebasing functionality for history manipulation.

    Provides functionality for:
    - Interactive rebasing with conflict resolution
    - Automatic conflict resolution strategies
    - Rebase progress tracking and recovery
    - History cleanup and optimization
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize rebase manager.

        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.current_rebase: RebaseOperation | None = None

    def start_interactive_rebase(
        self, base_commit: str, target_branch: str | None = None
    ) -> RebaseOperation:
        """
        Start an interactive rebase operation.

        Args:
            base_commit: Base commit to rebase onto
            target_branch: Target branch (current branch if None)

        Returns:
            RebaseOperation instance
        """
        operation = RebaseOperation(
            base_commit=base_commit,
            target_branch=target_branch or self.repo.active_branch.name,
        )

        try:
            # Get commits to rebase
            commits = list(self.repo.iter_commits(f"{base_commit}..HEAD"))
            operation.total_steps = len(commits)

            # Start interactive rebase
            self.repo.git.rebase("-i", base_commit)

            operation.status = "in_progress"
            operation.current_step = 0

        except git.GitCommandError as e:
            if self._rebase_in_progress():
                operation.status = "conflicts"
                operation.conflicts = self._detect_rebase_conflicts()
                operation.message = f"Conflicts detected during rebase of {base_commit}"
            else:
                operation.status = "failed"
                operation.message = f"Failed to start rebase: {e}"

        except Exception as e:
            operation.status = "failed"
            operation.message = f"Error starting rebase: {e}"

        self.current_rebase = operation
        return operation

    def continue_rebase(self) -> bool:
        """
        Continue an ongoing rebase operation.

        Returns:
            True if rebase continued successfully
        """
        if not self.current_rebase or self.current_rebase.status not in (
            "in_progress",
            "conflicts",
        ):
            logger.error("No active rebase to continue")
            return False

        try:
            self.repo.git.rebase("--continue")
            self.current_rebase.current_step += 1

            # The rebase is complete once git tears down its state directory;
            # while it still exists, more steps remain regardless of the
            # pre-computed step count.
            if not self._rebase_in_progress():
                self.current_rebase.status = "completed"
                self.current_rebase = None

            return True

        except git.GitCommandError:
            assert self.current_rebase is not None
            if self._rebase_in_progress():
                self.current_rebase.status = "conflicts"
                self.current_rebase.conflicts = self._detect_rebase_conflicts()
                return False
            else:
                self.current_rebase.status = "failed"
                return False

    def _rebase_in_progress(self) -> bool:
        """Detect an in-progress rebase via git state directories, not output text.

        The ``rebase-merge`` (interactive/merge backend) or ``rebase-apply``
        (apply backend) directory exists exactly while a rebase is stopped,
        regardless of stderr wording across git versions.
        """
        return _git_path_exists(
            self.repo, self.repo_path, "rebase-merge"
        ) or _git_path_exists(self.repo, self.repo_path, "rebase-apply")

    def abort_rebase(self) -> bool:
        """
        Abort the current rebase operation.

        Returns:
            True if rebase was aborted successfully
        """
        if not self.current_rebase:
            logger.error("No active rebase to abort")
            return False

        try:
            self.repo.git.rebase("--abort")
            self.current_rebase.status = "aborted"
            self.current_rebase = None
            return True

        except git.GitCommandError as e:
            logger.error(f"Failed to abort rebase: {e}")
            return False

    def _detect_rebase_conflicts(self) -> list[MergeConflict]:
        """Detect conflicts during rebase by parsing git status --porcelain output."""
        conflicts = []
        try:
            status_output = self.repo.git.status("--porcelain")
            for line in status_output.split("\n"):
                line = line.strip()
                if not line:
                    continue
                status_code = line[:2]
                file_path = line[3:]
                # UU, AA, DD markers all indicate conflict
                if any(
                    status_code.startswith(code)
                    for code in ("UU", "AA", "DD", "AU", "UA")
                ):
                    conflict = MergeConflict(
                        file_path=file_path,
                        conflict_type={
                            "UU": "both_modified",
                            "AA": "both_added",
                            "DD": "both_deleted",
                            "AU": "added_by_us",
                            "UA": "added_by_them",
                        }.get(status_code[:2], "unknown"),
                    )
                    # Attempt to read conflict markers from the file
                    try:
                        conflict_file = self.repo_path / file_path
                        if conflict_file.exists():
                            content = conflict_file.read_text(errors="replace")
                            import re as re_mod

                            our_m = re_mod.search(
                                r"<<<<<<< .+?\n(.*?)\n=======", content, re_mod.DOTALL
                            )
                            their_m = re_mod.search(
                                r"=======\n(.*?)\n>>>>>>> ", content, re_mod.DOTALL
                            )
                            conflict.our_content = our_m.group(1) if our_m else ""
                            conflict.their_content = their_m.group(1) if their_m else ""
                    except Exception as exc:
                        logger.warning(
                            "Could not read conflict markers from %s: %s",
                            file_path,
                            exc,
                        )
                    conflicts.append(conflict)
        except Exception as e:
            logger.error(f"Error detecting rebase conflicts: {e}")
        return conflicts


class AdvancedGitOperations:
    """
    Comprehensive advanced Git operations manager.

    Combines submodule management, cherry-picking, and rebasing
    into a unified interface for complex Git workflows.
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize advanced Git operations manager.

        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)

        # Initialize sub-managers
        self.submodules = SubmoduleManager(repo_path)
        self.cherry_pick = CherryPickManager(repo_path)
        self.rebase = RebaseManager(repo_path)

        # Track operations
        self.operation_history: list[Any] = []

    def execute_workflow(self, workflow_config: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a complex Git workflow.

        Args:
            workflow_config: Configuration defining the workflow steps

        Returns:
            Dictionary with workflow execution results
        """
        results = {
            "workflow_id": workflow_config.get("id", "unknown"),
            "steps": [],
            "overall_success": True,
            "errors": [],
        }

        try:
            # Execute workflow steps
            for step in workflow_config.get("steps", []):
                step_result = self._execute_workflow_step(step)
                results["steps"].append(step_result)

                if not step_result.get("success", False):
                    results["overall_success"] = False
                    results["errors"].append(step_result.get("error", "Unknown error"))

                    # Check if workflow should stop on error
                    if step.get("stop_on_error", False):
                        break

        except Exception as e:
            results["overall_success"] = False
            results["errors"].append(str(e))

        return results

    def _execute_workflow_step(self, step: dict[str, Any]) -> dict[str, Any]:
        """Execute a single workflow step."""
        step_type = step.get("type")
        step_result: dict[str, Any] = {
            "step_type": step_type,
            "success": False,
            "message": "",
            "details": {},
        }

        try:
            if step_type == "submodule_init":
                results = self.submodules.initialize_submodules(
                    recursive=step.get("recursive", False)
                )
                step_result["success"] = True
                step_result["message"] = f"Initialized {len(results)} submodules"
                step_result["details"] = results

            elif step_type == "submodule_update":
                results = self.submodules.update_submodules(
                    recursive=step.get("recursive", False)
                )
                step_result["success"] = True
                step_result["message"] = f"Updated {len(results)} submodules"
                step_result["details"] = results

            elif step_type == "cherry_pick":
                operation = self.cherry_pick.cherry_pick_commit(
                    step["commit_sha"], strategy=step.get("strategy", "recursive")
                )
                step_result["success"] = operation.status in ["applied"]
                step_result["message"] = operation.message
                step_result["details"] = {
                    "commit_sha": operation.commit_sha,
                    "status": operation.status,
                    "conflicts": len(operation.conflicts),
                }

            elif step_type == "rebase":
                rebase_op = self.rebase.start_interactive_rebase(
                    step["base_commit"], target_branch=step.get("target_branch")
                )
                step_result["success"] = rebase_op.status == "in_progress"
                step_result["message"] = rebase_op.message
                step_result["details"] = {
                    "base_commit": rebase_op.base_commit,
                    "target_branch": rebase_op.target_branch,
                    "status": rebase_op.status,
                }

            else:
                step_result["message"] = f"Unknown step type: {step_type}"

        except Exception as e:
            step_result["message"] = f"Step execution failed: {e}"

        return step_result

    def get_repository_health(self) -> dict[str, Any]:
        """
        Get comprehensive repository health information.

        Returns:
            Dictionary with repository health metrics
        """
        health_info: dict[str, Any] = {
            "submodules": self.submodules.get_submodule_status(),
            "recent_operations": self.operation_history[-10:],  # Last 10 operations
            "repository_stats": {
                "total_commits": len(list(self.repo.iter_commits())),
                "branches": len(self.repo.heads),
                "remotes": len(self.repo.remotes),
                "stashes": self._count_stashes(),
            },
        }

        # Calculate health score
        health_score = self._calculate_health_score(health_info)
        health_info["health_score"] = health_score

        return health_info

    def _count_stashes(self) -> int:
        """Count stashes, returning 0 when the repository has no stash ref."""
        try:
            return len(list(self.repo.iter_commits("refs/stash")))
        except git.GitCommandError:
            return 0

    def _calculate_health_score(self, health_info: dict[str, Any]) -> float:
        """Calculate overall repository health score (0-100)."""
        score = 100.0

        # Check submodule health
        submodules = health_info.get("submodules", {})
        for submodule_info in submodules.values():
            if submodule_info.get("status") in ["error", "missing"]:
                score -= 10
            elif submodule_info.get("status") == "dirty":
                score -= 5

        # Check for recent failed operations
        recent_ops = health_info.get("recent_operations", [])
        failed_ops = sum(1 for op in recent_ops if not op.get("success", True))
        score -= failed_ops * 5

        return max(0.0, min(100.0, score))


def create_advanced_git_operations(
    repo_path: str | Path,
) -> AdvancedGitOperations:
    """
    Create an AdvancedGitOperations instance for a repository.

    Args:
        repo_path: Path to the Git repository

    Returns:
        AdvancedGitOperations instance

    Raises:
        GitOperationError: If repository is not valid or accessible
    """
    try:
        git.Repo(repo_path)

        # Verify it's a valid Git repository
        if not (Path(repo_path) / ".git").exists():
            raise GitOperationError(f"Path {repo_path} is not a Git repository")

        return AdvancedGitOperations(repo_path)

    except git.InvalidGitRepositoryError as exc:
        raise GitOperationError(f"Invalid Git repository: {repo_path}") from exc
    except Exception as e:
        raise GitOperationError(f"Error accessing repository {repo_path}: {e}") from e
