#!/usr/bin/env python3

"""
Unit tests for the geo-infer-git console-script CLI.

Covers load_repo_list (yaml/json dispatch, bad types), command handlers
with a monkeypatched RepoManager, and exit-code semantics (sys.exit(1)
on partial failure or missing input).
"""

import argparse
import json
from typing import Any

import pytest
import yaml

from geo_infer_git import cli


def make_args(**overrides: Any) -> argparse.Namespace:
    """Build an argparse.Namespace matching main()'s parser output."""
    defaults: dict[str, Any] = {
        "verbose": False,
        "config": None,
        "command": None,
        "file": None,
        "url": None,
        "name": None,
        "sequential": False,
        "repos": [],
        "json": False,
        "create": None,
        "checkout": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestLoadRepoList:
    """Test load_repo_list file parsing and validation."""

    def test_loads_yaml_list(self, tmp_path):
        repo_file = tmp_path / "repos.yaml"
        repo_file.write_text(
            yaml.safe_dump([{"url": "https://example.com/a.git", "name": "a"}])
        )

        repos = cli.load_repo_list(str(repo_file))

        assert repos == [{"url": "https://example.com/a.git", "name": "a"}]

    def test_loads_yml_extension(self, tmp_path):
        repo_file = tmp_path / "repos.yml"
        repo_file.write_text(yaml.safe_dump([{"url": "https://example.com/a.git"}]))

        repos = cli.load_repo_list(str(repo_file))

        assert repos[0]["url"] == "https://example.com/a.git"

    def test_loads_json_list(self, tmp_path):
        repo_file = tmp_path / "repos.json"
        repo_file.write_text('[{"url": "https://example.com/b.git"}]')

        repos = cli.load_repo_list(str(repo_file))

        assert repos == [{"url": "https://example.com/b.git"}]

    def test_missing_file_exits_1(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            cli.load_repo_list(str(tmp_path / "missing.yaml"))
        assert excinfo.value.code == 1

    def test_unsupported_extension_exits_1(self, tmp_path):
        repo_file = tmp_path / "repos.txt"
        repo_file.write_text("[{'url': 'https://example.com/a.git'}]")

        with pytest.raises(SystemExit) as excinfo:
            cli.load_repo_list(str(repo_file))
        assert excinfo.value.code == 1

    def test_non_list_document_exits_1(self, tmp_path):
        repo_file = tmp_path / "repos.yaml"
        repo_file.write_text(yaml.safe_dump({"url": "https://example.com/a.git"}))

        with pytest.raises(SystemExit) as excinfo:
            cli.load_repo_list(str(repo_file))
        assert excinfo.value.code == 1

    def test_non_dict_entry_exits_1(self, tmp_path):
        repo_file = tmp_path / "repos.json"
        repo_file.write_text('["not-a-dict"]')

        with pytest.raises(SystemExit) as excinfo:
            cli.load_repo_list(str(repo_file))
        assert excinfo.value.code == 1

    def test_entry_missing_url_exits_1(self, tmp_path):
        repo_file = tmp_path / "repos.json"
        repo_file.write_text('[{"name": "a"}]')

        with pytest.raises(SystemExit) as excinfo:
            cli.load_repo_list(str(repo_file))
        assert excinfo.value.code == 1


class FakeRepoManager:
    """RepoManager test double recording construction and call arguments."""

    def __init__(
        self,
        clone_results: dict[str, bool] | None = None,
        sync_results: dict[str, bool] | None = None,
        status_results: dict[str, Any] | None = None,
        branch_results: dict[str, bool] | None = None,
    ):
        self.config_path: str | None = None
        self.clone_calls: list[dict[str, Any]] = []
        self.sync_calls: list[dict[str, Any]] = []
        self.status_calls: list[dict[str, Any]] = []
        self.branch_calls: list[dict[str, Any]] = []
        self._clone_results = clone_results or {}
        self._sync_results = sync_results or {}
        self._status_results = status_results or {}
        self._branch_results = branch_results or {}

    def clone_repositories(self, repos, parallel=True):
        self.clone_calls.append({"repos": repos, "parallel": parallel})
        return self._clone_results

    def sync_repositories(self, repo_names=None):
        self.sync_calls.append({"repo_names": repo_names})
        return self._sync_results

    def check_repo_status(self, repo_names=None):
        self.status_calls.append({"repo_names": repo_names})
        return self._status_results

    def create_branch(self, branch_name, repo_names=None):
        self.branch_calls.append(
            {"action": "create", "branch": branch_name, "repo_names": repo_names}
        )
        return self._branch_results

    def checkout_branch(self, branch_name, repo_names=None):
        self.branch_calls.append(
            {"action": "checkout", "branch": branch_name, "repo_names": repo_names}
        )
        return self._branch_results


@pytest.fixture
def patch_repo_manager(monkeypatch):
    """Install a FakeRepoManager as cli.RepoManager and return the instance."""

    def _install(manager: FakeRepoManager) -> FakeRepoManager:
        monkeypatch.setattr(cli, "RepoManager", lambda config_path=None: manager)
        return manager

    return _install


class TestCloneCommand:
    """Test clone_command exit codes and RepoManager wiring."""

    def test_all_success_does_not_exit(self, patch_repo_manager):
        manager = patch_repo_manager(
            FakeRepoManager(clone_results={"a": True, "b": True})
        )

        cli.clone_command(make_args(url="https://example.com/a.git", name="a"))

        assert manager.clone_calls[0]["repos"] == [
            {"url": "https://example.com/a.git", "name": "a"}
        ]
        assert manager.clone_calls[0]["parallel"] is True

    def test_partial_failure_exits_1(self, patch_repo_manager):
        patch_repo_manager(FakeRepoManager(clone_results={"a": True, "b": False}))

        with pytest.raises(SystemExit) as excinfo:
            cli.clone_command(make_args(url="https://example.com/a.git", name="a"))
        assert excinfo.value.code == 1

    def test_sequential_flag_disables_parallel(self, patch_repo_manager):
        manager = patch_repo_manager(FakeRepoManager(clone_results={"a": True}))

        cli.clone_command(
            make_args(url="https://example.com/a.git", name="a", sequential=True)
        )

        assert manager.clone_calls[0]["parallel"] is False

    def test_file_source_loads_repo_list(self, tmp_path, patch_repo_manager):
        manager = patch_repo_manager(FakeRepoManager(clone_results={"a": True}))
        repo_file = tmp_path / "repos.json"
        repo_file.write_text('[{"url": "https://example.com/a.git", "name": "a"}]')

        cli.clone_command(make_args(file=str(repo_file)))

        assert manager.clone_calls[0]["repos"] == [
            {"url": "https://example.com/a.git", "name": "a"}
        ]

    def test_no_source_specified_exits_1(self, patch_repo_manager):
        patch_repo_manager(FakeRepoManager(clone_results={}))

        with pytest.raises(SystemExit) as excinfo:
            cli.clone_command(make_args())
        assert excinfo.value.code == 1


class TestSyncCommand:
    """Test sync_command exit codes and repo selection."""

    def test_all_success_does_not_exit(self, patch_repo_manager):
        manager = patch_repo_manager(FakeRepoManager(sync_results={"a": True}))

        cli.sync_command(make_args(repos=["a"]))

        assert manager.sync_calls[0]["repo_names"] == ["a"]

    def test_no_repo_selection_passes_none(self, patch_repo_manager):
        manager = patch_repo_manager(
            FakeRepoManager(sync_results={"a": True, "b": True})
        )

        cli.sync_command(make_args())

        assert manager.sync_calls[0]["repo_names"] is None

    def test_partial_failure_exits_1(self, patch_repo_manager):
        patch_repo_manager(FakeRepoManager(sync_results={"a": True, "b": False}))

        with pytest.raises(SystemExit) as excinfo:
            cli.sync_command(make_args(repos=["a", "b"]))
        assert excinfo.value.code == 1


class TestStatusCommand:
    """Test status_command output formatting."""

    def test_json_output_prints_status(self, capsys, patch_repo_manager):
        patch_repo_manager(
            FakeRepoManager(
                status_results={
                    "a": {
                        "branch": "main",
                        "is_dirty": False,
                        "commits_behind": 0,
                        "commits_ahead": 2,
                    }
                }
            )
        )

        cli.status_command(make_args(repos=["a"], json=True))

        payload = json.loads(capsys.readouterr().out)
        assert payload["a"]["branch"] == "main"
        assert payload["a"]["commits_ahead"] == 2

    def test_error_entry_is_reported_without_crashing(self, patch_repo_manager):
        patch_repo_manager(
            FakeRepoManager(status_results={"a": {"error": "not a repository"}})
        )

        # Must not raise or exit: the error is logged and iteration continues.
        cli.status_command(make_args(repos=["a"]))


class TestBranchCommand:
    """Test branch_command subcommand dispatch and exit codes."""

    def test_create_branch_all_success(self, patch_repo_manager):
        manager = patch_repo_manager(FakeRepoManager(branch_results={"a": True}))

        cli.branch_command(make_args(create="feature", repos=["a"]))

        assert manager.branch_calls[0]["action"] == "create"
        assert manager.branch_calls[0]["branch"] == "feature"

    def test_checkout_branch(self, patch_repo_manager):
        manager = patch_repo_manager(FakeRepoManager(branch_results={"a": True}))

        cli.branch_command(make_args(checkout="feature", repos=["a"]))

        assert manager.branch_calls[0]["action"] == "checkout"
        assert manager.branch_calls[0]["branch"] == "feature"

    def test_partial_checkout_failure_exits_1(self, patch_repo_manager):
        patch_repo_manager(FakeRepoManager(branch_results={"a": True, "b": False}))

        with pytest.raises(SystemExit) as excinfo:
            cli.branch_command(make_args(checkout="feature", repos=["a", "b"]))
        assert excinfo.value.code == 1

    def test_no_operation_selected_does_not_exit(self, patch_repo_manager):
        manager = patch_repo_manager(FakeRepoManager())

        # Neither --create nor --checkout: nothing to do, no failure exit.
        cli.branch_command(make_args(repos=["a"]))

        assert manager.branch_calls == []


class TestMainDispatch:
    """Test main() argument parsing and command dispatch."""

    @pytest.fixture(autouse=True)
    def quiet_logging(self, monkeypatch):
        monkeypatch.setattr(cli, "setup_logging", lambda verbose: None)

    def _run_main(self, monkeypatch, argv, handler_name):
        calls = []

        def handler(args):
            calls.append(args)

        monkeypatch.setattr(cli, handler_name, handler)
        monkeypatch.setattr("sys.argv", ["geo-infer-git", *argv])
        cli.main()
        return calls

    def test_clone_dispatches_to_clone_command(self, monkeypatch):
        calls = self._run_main(
            monkeypatch,
            ["clone", "--url", "https://example.com/a.git"],
            "clone_command",
        )

        assert len(calls) == 1
        assert calls[0].url == "https://example.com/a.git"
        assert calls[0].command == "clone"

    def test_sync_dispatches_to_sync_command(self, monkeypatch):
        calls = self._run_main(monkeypatch, ["sync", "a", "b"], "sync_command")

        assert calls[0].repos == ["a", "b"]

    def test_status_json_flag_parsed(self, monkeypatch):
        calls = self._run_main(monkeypatch, ["status", "--json", "a"], "status_command")

        assert calls[0].json is True

    def test_branch_create_parsed(self, monkeypatch):
        calls = self._run_main(
            monkeypatch, ["branch", "--create", "feature", "a"], "branch_command"
        )

        assert calls[0].create == "feature"

    def test_no_command_prints_help_and_exits_1(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["geo-infer-git"])

        with pytest.raises(SystemExit) as excinfo:
            cli.main()

        assert excinfo.value.code == 1
        assert "GEO-INFER-GIT" in capsys.readouterr().out
