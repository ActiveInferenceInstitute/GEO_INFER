#!/usr/bin/env python3

"""
Unit tests for advanced Git operations (SubmoduleManager, CherryPickManager,
RebaseManager, AdvancedGitOperations).

All subprocess git work happens inside tmp_path-scoped repositories.
"""

import logging
import subprocess

import pytest

from geo_infer_git.core.advanced_git import (
    CherryPickManager,
    RebaseManager,
    SubmoduleManager,
    create_advanced_git_operations,
)
from geo_infer_git.utils.error_handler import GitOperationError

GIT_IDENTITY = (
    "-c",
    "user.name=Test User",
    "-c",
    "user.email=test@example.com",
)


@pytest.fixture(autouse=True)
def inert_git_editors(monkeypatch):
    """Make rebase/cherry-pick continuation paths non-interactive."""
    monkeypatch.setenv("GIT_EDITOR", "true")
    monkeypatch.setenv("GIT_SEQUENCE_EDITOR", "true")


def git(*args: str, cwd) -> None:
    """Run a git command, raising on failure."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def commit_all(repo_dir, message: str) -> str:
    """Stage everything and commit; return the new HEAD sha."""
    git("add", "-A", cwd=repo_dir)
    git(*GIT_IDENTITY, "commit", "-m", message, cwd=repo_dir)
    return head_sha(repo_dir)


def head_sha(repo_dir) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_branch(repo_dir) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_temporary_repository_disables_inherited_background_monitor(clean_repo):
    """Read Git's effective setting, including command-environment overrides."""
    result = subprocess.run(
        ["git", "config", "--get", "core.fsmonitor"],
        cwd=clean_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "false"


def init_repo(repo_dir, file_name: str = "README.md", content: str = "# Test\n"):
    """Create an isolated git repository with one commit.

    Identity is configured repo-locally: commits created by GitPython
    (cherry-pick, cherry-pick --continue, rebase --continue) inherit the
    process environment, and CI runners have no global git identity. Linux
    git refuses to auto-detect one, which fails every commit-creating path.
    """
    repo_dir.mkdir(parents=True)
    git("init", cwd=repo_dir)
    git("config", "protocol.file.allow", "always", cwd=repo_dir)
    git("config", "user.name", "Test User", cwd=repo_dir)
    git("config", "user.email", "test@example.com", cwd=repo_dir)
    (repo_dir / file_name).write_text(content)
    commit_all(repo_dir, "initial commit")
    return repo_dir


@pytest.fixture
def clean_repo(tmp_path):
    """A single-commit git repository."""
    return init_repo(tmp_path / "clean_repo")


@pytest.fixture
def cherry_pick_conflict_repo(tmp_path):
    """A repository where cherry-picking a feature commit onto the base
    branch produces a content conflict in file.txt."""
    repo_dir = init_repo(tmp_path / "cp_repo", file_name="file.txt", content="base\n")

    git("checkout", "-b", "feature", cwd=repo_dir)
    (repo_dir / "file.txt").write_text("feature line\n")
    commit_all(repo_dir, "feature change")
    feature_sha = head_sha(repo_dir)

    git("checkout", "-", cwd=repo_dir)
    (repo_dir / "file.txt").write_text("main line\n")
    commit_all(repo_dir, "main change")

    return repo_dir, feature_sha


@pytest.fixture
def repo_with_submodule(tmp_path):
    """A parent repository with one committed submodule named 'sub'."""
    child = init_repo(tmp_path / "child", content="# Child\n")
    parent = init_repo(tmp_path / "parent")
    git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "sub",
        cwd=parent,
    )
    git(*GIT_IDENTITY, "commit", "-m", "add submodule", cwd=parent)
    # Submodule-local fetches (update --remote) need the file protocol
    # explicitly allowed in the submodule clone's own config.
    git("config", "protocol.file.allow", "always", cwd=parent / "sub")
    return parent, child


@pytest.fixture
def rebase_conflict_repo(tmp_path):
    """A repository where rebasing the feature branch onto the base-branch
    tip produces a content conflict in file.txt. The fixture leaves HEAD on
    the feature branch, which start_interactive_rebase rebases, and returns
    the base-branch tip as the rebase base."""
    repo_dir = init_repo(tmp_path / "rb_repo", file_name="file.txt", content="base\n")

    git("checkout", "-b", "feature", cwd=repo_dir)
    (repo_dir / "file.txt").write_text("feature line\n")
    commit_all(repo_dir, "feature change")
    feature_branch = current_branch(repo_dir)

    git("checkout", "-", cwd=repo_dir)
    (repo_dir / "file.txt").write_text("main line\n")
    commit_all(repo_dir, "main change")
    base_tip_sha = head_sha(repo_dir)
    git("checkout", "feature", cwd=repo_dir)

    return repo_dir, base_tip_sha, feature_branch


class TestAdvancedGitSubmoduleManager:
    """Test SubmoduleManager load/init/update/status against a real submodule."""

    def test_loads_gitmodules_entries(self, repo_with_submodule):
        parent, child = repo_with_submodule

        manager = SubmoduleManager(parent)

        assert "sub" in manager.submodules
        info = manager.submodules["sub"]
        assert info.path == "sub"
        assert info.url.endswith("/child")

    def test_initialize_submodules_reports_success_and_commit(
        self, repo_with_submodule
    ):
        parent, child = repo_with_submodule

        manager = SubmoduleManager(parent)
        results = manager.initialize_submodules()

        assert results == {"sub": True}
        assert manager.submodules["sub"].status == "initialized"
        assert manager.submodules["sub"].commit == head_sha(child)

    def test_get_submodule_status_after_initialization(self, repo_with_submodule):
        parent, _ = repo_with_submodule

        manager = SubmoduleManager(parent)
        manager.initialize_submodules()

        status = manager.get_submodule_status()
        assert set(status) == {"sub"}
        sub = status["sub"]
        assert sub["path"] == "sub"
        assert sub["exists"] is True
        assert sub["is_git_repo"] is True
        assert sub["is_dirty"] is False
        assert sub["remotes"] == ["origin"]

    def test_repository_without_submodules_is_empty(self, clean_repo):
        manager = SubmoduleManager(clean_repo)

        assert manager.submodules == {}
        assert manager.get_submodule_status() == {}

    def test_update_submodules_marks_updated(self, repo_with_submodule):
        parent, _ = repo_with_submodule

        manager = SubmoduleManager(parent)
        results = manager.update_submodules()

        assert results == {"sub": True}
        assert manager.submodules["sub"].status == "updated"

    def test_sync_submodules_marks_synced(self, repo_with_submodule):
        parent, _ = repo_with_submodule

        manager = SubmoduleManager(parent)
        results = manager.sync_submodules()

        assert results == {"sub": True}
        assert manager.submodules["sub"].status == "synced"

    def test_recursive_dependencies_scanned_from_nested_gitmodules(
        self, repo_with_submodule
    ):
        parent, _ = repo_with_submodule
        # Dependency scanning reads .gitmodules inside the checked-out
        # submodule copy; declare the nested module after initialization so
        # submodule update cannot check the recorded sha back over it.
        sub_checkout = parent / "sub"
        (sub_checkout / ".gitmodules").write_text(
            '[submodule "nested"]\n'
            "\tpath = vendor/nested\n"
            "\turl = https://example.com/nested.git\n"
        )

        manager = SubmoduleManager(parent)
        manager.initialize_submodules()
        manager.submodules["sub"].recursive = True

        status = manager.get_submodule_status()
        assert status["sub"]["dependencies"] == ["vendor/nested"]


class TestAdvancedGitCherryPick:
    """Test CherryPickManager clean and conflicting cherry-picks."""

    def test_clean_cherry_pick_is_applied(self, tmp_path):
        repo_dir = init_repo(tmp_path / "cp_clean", content="base\n")

        git("checkout", "-b", "feature", cwd=repo_dir)
        (repo_dir / "feature.txt").write_text("new file\n")
        commit_all(repo_dir, "add feature file")
        feature_sha = head_sha(repo_dir)
        git("checkout", "-", cwd=repo_dir)

        manager = CherryPickManager(repo_dir)
        operation = manager.cherry_pick_commit(feature_sha)

        assert operation.status == "applied"
        assert (repo_dir / "feature.txt").read_text() == "new file\n"

    def test_conflicting_cherry_pick_detects_conflicts_and_resolves_ours(
        self, cherry_pick_conflict_repo
    ):
        repo_dir, feature_sha = cherry_pick_conflict_repo

        manager = CherryPickManager(repo_dir)
        operation = manager.cherry_pick_commit(feature_sha)

        assert operation.status == "conflicts"
        assert len(operation.conflicts) == 1
        assert operation.conflicts[0].file_path == "file.txt"
        assert operation.conflicts[0].our_content == "main line"
        assert operation.conflicts[0].their_content == "feature line"

        assert manager.resolve_conflicts(0, resolution_strategy="ours") is True
        assert operation.status == "applied"
        assert (repo_dir / "file.txt").read_text() == "main line"

    def test_unknown_commit_sha_fails_gracefully(self, clean_repo):
        manager = CherryPickManager(clean_repo)
        operation = manager.cherry_pick_commit("0" * 40)

        assert operation.status == "failed"
        assert "Cherry-pick failed" in operation.message

    def test_resolve_conflicts_out_of_range_index_raises(self, clean_repo):
        manager = CherryPickManager(clean_repo)

        with pytest.raises(ValueError, match="out of range"):
            manager.resolve_conflicts(5)

    def test_range_pick_applies_single_commit(self, tmp_path):
        repo_dir = init_repo(tmp_path / "range_repo", content="base\n")
        base_sha = head_sha(repo_dir)

        git("checkout", "-b", "feature", cwd=repo_dir)
        (repo_dir / "ranged.txt").write_text("ranged\n")
        commit_all(repo_dir, "ranged change")
        tip_sha = head_sha(repo_dir)
        git("checkout", "-", cwd=repo_dir)

        manager = CherryPickManager(repo_dir)
        operations = manager.cherry_pick_range(base_sha, tip_sha)

        assert len(operations) == 1
        assert operations[0].status == "applied"

    def test_analyze_conflict_missing_file_returns_none_and_warns(
        self, clean_repo, caplog
    ):
        """An unreadable conflict file is surfaced with a warning, not silence."""
        manager = CherryPickManager(clean_repo)

        with caplog.at_level(logging.WARNING):
            conflict = manager._analyze_conflict("missing.txt", "UU")

        assert conflict is None
        assert "missing.txt" in caplog.text


class TestAdvancedGitRebase:
    """Test RebaseManager conflict, continue, and abort paths."""

    def test_conflicting_rebase_reports_conflicts(self, rebase_conflict_repo):
        repo_dir, base_sha, feature_branch = rebase_conflict_repo

        manager = RebaseManager(repo_dir)
        operation = manager.start_interactive_rebase(base_sha)

        assert operation.status == "conflicts"
        assert operation.total_steps == 1
        assert [c.file_path for c in operation.conflicts] == ["file.txt"]
        # During a rebase, HEAD tracks the base branch, so "ours" is the
        # base-branch content and "theirs" is the replayed feature change.
        assert operation.conflicts[0].our_content == "main line"
        assert operation.conflicts[0].their_content == "feature line"

    def test_resolve_then_continue_rebase_completes(self, rebase_conflict_repo):
        repo_dir, base_sha, feature_branch = rebase_conflict_repo

        manager = RebaseManager(repo_dir)
        operation = manager.start_interactive_rebase(base_sha)
        assert operation.status == "conflicts"

        (repo_dir / "file.txt").write_text("merged line\n")
        manager.repo.git.add(".")

        assert manager.continue_rebase() is True
        assert manager.current_rebase is None
        assert operation.status == "completed"

    def test_abort_rebase_restores_feature_branch(self, rebase_conflict_repo):
        repo_dir, base_sha, feature_branch = rebase_conflict_repo

        manager = RebaseManager(repo_dir)
        operation = manager.start_interactive_rebase(base_sha)
        assert operation.status == "conflicts"

        assert manager.abort_rebase() is True
        assert manager.current_rebase is None
        assert operation.status == "aborted"
        assert current_branch(repo_dir) == feature_branch

    def test_continue_without_active_rebase_returns_false(self, clean_repo):
        assert RebaseManager(clean_repo).continue_rebase() is False

    def test_abort_without_active_rebase_returns_false(self, clean_repo):
        assert RebaseManager(clean_repo).abort_rebase() is False


class TestAdvancedGitOperationsFacade:
    """Test the unified AdvancedGitOperations facade."""

    def test_create_returns_wired_instance(self, clean_repo):
        ops = create_advanced_git_operations(clean_repo)

        assert isinstance(ops.submodules, SubmoduleManager)
        assert isinstance(ops.cherry_pick, CherryPickManager)
        assert isinstance(ops.rebase, RebaseManager)

    def test_create_rejects_non_repository(self, tmp_path):
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()

        with pytest.raises(GitOperationError):
            create_advanced_git_operations(plain_dir)

    def test_repository_health_score_is_perfect_for_clean_repo(self, clean_repo):
        ops = create_advanced_git_operations(clean_repo)

        health = ops.get_repository_health()

        assert health["health_score"] == 100.0
        assert health["repository_stats"]["total_commits"] == 1
        assert health["submodules"] == {}

    def test_workflow_unknown_step_type_is_recorded_as_failure(self, clean_repo):
        ops = create_advanced_git_operations(clean_repo)

        results = ops.execute_workflow(
            {"id": "wf-1", "steps": [{"type": "teleport", "stop_on_error": True}]}
        )

        assert results["overall_success"] is False
        assert results["steps"][0]["success"] is False
        assert "Unknown step type: teleport" in results["steps"][0]["message"]

    def test_workflow_submodule_init_step_on_submodule_free_repo(self, clean_repo):
        ops = create_advanced_git_operations(clean_repo)

        results = ops.execute_workflow(
            {"id": "wf-2", "steps": [{"type": "submodule_init"}]}
        )

        assert results["overall_success"] is True
        assert results["steps"][0]["success"] is True
        assert results["steps"][0]["message"] == "Initialized 0 submodules"
