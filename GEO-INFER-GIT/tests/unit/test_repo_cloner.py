#!/usr/bin/env python3

"""
Unit tests for repository cloner.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from geo_infer_git.core.repo_cloner import RepoCloner, CloneProgress
from geo_infer_git.utils.config_loader import CloneConfig
from geo_infer_git.core.github_api import GitHubRepository


class TestCloneProgress:
    """Test CloneProgress class."""

    def test_initialization(self):
        """Test CloneProgress initialization."""
        progress = CloneProgress()

        assert progress.total_repos == 0
        assert progress.completed_repos == 0
        assert progress.failed_repos == 0
        assert progress.skipped_repos == 0
        assert progress.start_time > 0

    def test_increment_methods(self):
        """Test increment methods."""
        progress = CloneProgress()

        progress.increment_completed()
        progress.increment_failed()
        progress.increment_skipped()

        assert progress.completed_repos == 1
        assert progress.failed_repos == 1
        assert progress.skipped_repos == 1

    def test_properties(self):
        """Test computed properties."""
        progress = CloneProgress()
        progress.total_repos = 10
        progress.completed_repos = 7

        assert progress.elapsed_time >= 0
        assert progress.success_rate == 70.0

    def test_get_stats(self):
        """Test get_stats method."""
        progress = CloneProgress()
        progress.total_repos = 10
        progress.completed_repos = 7
        progress.failed_repos = 2
        progress.skipped_repos = 1

        stats = progress.get_stats()

        assert stats["total"] == 10
        assert stats["completed"] == 7
        assert stats["failed"] == 2
        assert stats["skipped"] == 1
        assert "elapsed_time" in stats
        assert "success_rate" in stats


class TestRepoCloner:
    """Test RepoCloner class."""

    def setup_method(self):
        """Set up test method."""
        self.config = CloneConfig(
            output_dir=tempfile.mkdtemp(), github_token="test_token"
        )
        self.cloner = RepoCloner(self.config)

    def teardown_method(self):
        """Clean up after test method."""
        shutil.rmtree(self.config.output_dir, ignore_errors=True)

    def test_initialization(self):
        """Test RepoCloner initialization."""
        assert self.cloner.config == self.config
        assert self.cloner.progress is not None
        assert self.cloner.output_dir == Path(self.config.output_dir)
        assert self.cloner.git_env["GIT_TOKEN"] == "test_token"

    def test_clone_repository_success(self):
        """Test successful repository cloning."""
        with patch("git.Repo.clone_from") as mock_clone:
            mock_repo = Mock()
            mock_clone.return_value = mock_repo

            result = self.cloner.clone_repository("owner", "test-repo")

            assert result is True
            assert self.cloner.progress.completed_repos == 1
            mock_clone.assert_called_once()

    def test_clone_repository_already_exists(self):
        """Test cloning repository that already exists."""
        # Create existing repository directory with .git
        repo_path = self.cloner.output_dir / "owner" / "test-repo"
        repo_path.mkdir(parents=True)
        (repo_path / ".git").mkdir()

        result = self.cloner.clone_repository("owner", "test-repo")

        assert result is True
        assert self.cloner.progress.skipped_repos == 1

    def test_clone_repository_git_error(self):
        """Test repository cloning with Git error."""
        with patch("git.Repo.clone_from") as mock_clone:
            mock_clone.side_effect = Exception("Git error")

            result = self.cloner.clone_repository("owner", "test-repo")

            assert result is False
            assert self.cloner.progress.failed_repos == 1

    def test_clone_repositories_for_user_parallel(self):
        """Test cloning repositories for user in parallel."""
        # Create test repositories
        repos = [
            GitHubRepository(
                name="repo1",
                full_name="owner/repo1",
                owner="owner",
                description="",
                url="",
                clone_url="https://github.com/owner/repo1.git",
                ssh_url="",
                default_branch="main",
                language="",
                stars=0,
                forks=0,
                size=0,
                created_at="",
                updated_at="",
                pushed_at="",
                archived=False,
                private=False,
                fork=False,
                topics=[],
            ),
            GitHubRepository(
                name="repo2",
                full_name="owner/repo2",
                owner="owner",
                description="",
                url="",
                clone_url="https://github.com/owner/repo2.git",
                ssh_url="",
                default_branch="main",
                language="",
                stars=0,
                forks=0,
                size=0,
                created_at="",
                updated_at="",
                pushed_at="",
                archived=False,
                private=False,
                fork=False,
                topics=[],
            ),
        ]

        with patch.object(self.cloner, "_clone_repo_from_github_obj") as mock_clone:
            mock_clone.return_value = True

            success_count, total_count = self.cloner.clone_repositories_for_user(
                "testuser", repos
            )

            assert success_count == 2
            assert total_count == 2
            assert mock_clone.call_count == 2

    def test_clone_repositories_for_user_sequential(self):
        """Test cloning repositories for user sequentially."""
        # Disable concurrency
        self.cloner.config.concurrency_enabled = False

        repos = [
            GitHubRepository(
                name="repo1",
                full_name="owner/repo1",
                owner="owner",
                description="",
                url="",
                clone_url="https://github.com/owner/repo1.git",
                ssh_url="",
                default_branch="main",
                language="",
                stars=0,
                forks=0,
                size=0,
                created_at="",
                updated_at="",
                pushed_at="",
                archived=False,
                private=False,
                fork=False,
                topics=[],
            )
        ]

        with patch.object(self.cloner, "_clone_repo_from_github_obj") as mock_clone:
            mock_clone.return_value = True

            success_count, total_count = self.cloner.clone_repositories_for_user(
                "testuser", repos
            )

            assert success_count == 1
            assert total_count == 1

    def test_clone_repo_from_github_obj(self):
        """Test cloning from GitHub repository object."""
        github_repo = GitHubRepository(
            name="test-repo",
            full_name="owner/test-repo",
            owner="owner",
            description="",
            url="",
            clone_url="https://github.com/owner/test-repo.git",
            ssh_url="",
            default_branch="develop",
            language="",
            stars=0,
            forks=0,
            size=0,
            created_at="",
            updated_at="",
            pushed_at="",
            archived=False,
            private=False,
            fork=False,
            topics=[],
        )

        with patch.object(self.cloner, "clone_repository") as mock_clone:
            mock_clone.return_value = True

            result = self.cloner._clone_repo_from_github_obj(github_repo)

            assert result is True
            mock_clone.assert_called_once_with("owner", "test-repo", "develop")

    def test_setup_lfs(self):
        """Test Git LFS setup."""
        mock_repo = Mock()

        # Test with LFS required
        mock_repo.git.config.return_value = "true"

        self.cloner._setup_lfs(mock_repo)

        mock_repo.git.lfs.assert_called_once_with("pull", "origin")

    def test_setup_lfs_no_lfs(self):
        """Test Git LFS setup when LFS not available."""
        mock_repo = Mock()
        mock_repo.git.config.side_effect = Exception("LFS not available")

        # Should not raise exception
        self.cloner._setup_lfs(mock_repo)

    def test_clone_multiple_repositories_parallel(self):
        """Test cloning multiple repositories in parallel."""
        repositories = [("owner1", "repo1", "main"), ("owner2", "repo2", "develop")]

        with patch.object(self.cloner, "clone_repository") as mock_clone:
            mock_clone.return_value = True

            results = self.cloner.clone_multiple_repositories(repositories)

            assert len(results) == 2
            assert results["owner1/repo1"] is True
            assert results["owner2/repo2"] is True
            assert mock_clone.call_count == 2

    def test_clone_multiple_repositories_sequential(self):
        """Test cloning multiple repositories sequentially."""
        self.cloner.config.concurrency_enabled = False

        repositories = [("owner1", "repo1", "main"), ("owner2", "repo2", "develop")]

        with patch.object(self.cloner, "clone_repository") as mock_clone:
            mock_clone.return_value = True

            results = self.cloner.clone_multiple_repositories(repositories)

            assert len(results) == 2
            assert results["owner1/repo1"] is True
            assert results["owner2/repo2"] is True

    def test_get_disk_usage(self):
        """Test disk usage calculation."""
        # Create some files in output directory
        test_file = self.cloner.output_dir / "test.txt"
        test_file.write_text("test content")

        usage = self.cloner.get_disk_usage()

        assert "total_bytes" in usage
        assert "used_bytes" in usage
        assert "free_bytes" in usage
        assert "used_percent" in usage
        assert usage["total_bytes"] > 0

    def test_cleanup_failed_clones(self):
        """Cleanup removes only owner/repo-shaped dirs lacking .git."""
        # A failed clone attempt: owner/repo shaped without .git
        failed_dir = self.cloner.output_dir / "owner" / "failed-repo"
        failed_dir.mkdir(parents=True)
        (failed_dir / "some_file.txt").write_text("test")

        # An unrelated file and directory at the output root must survive
        (self.cloner.output_dir / "notes.md").write_text("keep me")
        keep_dir = self.cloner.output_dir / "project-notes"
        keep_dir.mkdir()
        (keep_dir / "keep.txt").write_text("keep me too")

        # An existing clone (with .git) must survive
        good_dir = self.cloner.output_dir / "owner" / "good-repo"
        good_dir.mkdir(parents=True)
        (good_dir / ".git").mkdir()

        cleaned_count = self.cloner.cleanup_failed_clones()

        assert cleaned_count == 1
        assert not failed_dir.exists()
        assert good_dir.exists() and (good_dir / ".git").exists()
        assert (self.cloner.output_dir / "notes.md").exists()
        assert (keep_dir / "keep.txt").exists()

    def test_get_clone_stats(self):
        """Test getting clone statistics."""
        self.cloner.progress.total_repos = 5
        self.cloner.progress.completed_repos = 3
        self.cloner.progress.failed_repos = 1
        self.cloner.progress.skipped_repos = 1

        stats = self.cloner.get_clone_stats()

        assert stats["total"] == 5
        assert stats["completed"] == 3
        assert stats["failed"] == 1
        assert stats["skipped"] == 1
        assert stats["success_rate"] == 60.0

    def test_reset_progress(self):
        """Test progress reset."""
        self.cloner.progress.total_repos = 5
        self.cloner.progress.completed_repos = 3

        self.cloner.reset_progress()

        assert self.cloner.progress.total_repos == 0
        assert self.cloner.progress.completed_repos == 0

    def test_estimate_clone_time(self):
        """Test clone time estimation."""
        # Test with concurrency enabled
        estimate = self.cloner.estimate_clone_time(10, 50000)

        assert "parallel_estimate_seconds" in estimate
        assert "sequential_estimate_seconds" in estimate
        assert "recommended_approach" in estimate

        # Test with concurrency disabled
        self.cloner.config.concurrency_enabled = False
        estimate = self.cloner.estimate_clone_time(10, 50000)

        assert "sequential_estimate_seconds" in estimate
        assert "recommended_approach" in estimate

    def test_close(self):
        """Test cloner cleanup resets progress state."""
        self.cloner.progress.total_repos = 3
        self.cloner.close()

        assert self.cloner.progress.total_repos == 0
        assert self.cloner.progress.completed_repos == 0
