#!/usr/bin/env python3

"""
Unit tests for configuration loader utilities.
"""

import tempfile
import json
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch

from geo_infer_git.utils.config_loader import (
    ConfigLoader,
    CloneConfig,
    TargetRepository,
    TargetUser,
    load_clone_config,
    load_target_repos_config,
    load_target_users_config,
)


class TestCloneConfig:
    """Test CloneConfig dataclass."""

    def test_default_initialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default CloneConfig initialization."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        config = CloneConfig()

        assert config.output_dir == "./cloned_repositories"
        assert config.base_dir == "./repos"
        assert config.github_token == ""
        assert config.concurrency_enabled is True
        assert config.max_workers == 4

    def test_validation(self):
        """Test CloneConfig validation."""
        # Valid config should not raise
        config = CloneConfig(max_workers=10)
        assert config.max_workers == 10

        # Invalid max_workers should be corrected
        config = CloneConfig(max_workers=0)
        assert config.max_workers == 1

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test GitHub token from environment."""
        monkeypatch.setenv("GITHUB_TOKEN", "test_token")
        config = CloneConfig()
        assert config.github_token == "test_token"


class TestTargetRepository:
    """Test TargetRepository dataclass."""

    def test_valid_initialization(self):
        """Test valid TargetRepository initialization."""
        repo = TargetRepository(owner="testuser", repo="testrepo")

        assert repo.owner == "testuser"
        assert repo.repo == "testrepo"
        assert repo.branch == "main"
        assert repo.enabled is True

    def test_validation(self):
        """Test TargetRepository validation."""
        # Missing owner should raise
        with pytest.raises(ValueError):
            TargetRepository(owner="", repo="testrepo")

        # Missing repo should raise
        with pytest.raises(ValueError):
            TargetRepository(owner="testuser", repo="")

        # Invalid clone depth should raise
        with pytest.raises(ValueError):
            TargetRepository(owner="testuser", repo="testrepo", clone_depth=-1)


class TestTargetUser:
    """Test TargetUser dataclass."""

    def test_valid_initialization(self):
        """Test valid TargetUser initialization."""
        user = TargetUser(username="testuser")

        assert user.username == "testuser"
        assert user.max_repos == 10
        assert user.enabled is True

    def test_validation(self):
        """Test TargetUser validation."""
        # Missing username should raise
        with pytest.raises(ValueError):
            TargetUser(username="")

        # Invalid max_repos should be corrected
        user = TargetUser(username="testuser", max_repos=0)
        assert user.max_repos == 1


class TestConfigLoader:
    """Test ConfigLoader class."""

    def test_initialization(self):
        """Test ConfigLoader initialization."""
        # Test default initialization (config dir may not exist in test environment)
        loader = ConfigLoader()
        assert loader.config_dir is not None

        # Test custom config directory
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigLoader(temp_dir)
            assert loader.config_dir == Path(temp_dir)

    def test_load_yaml_config(self):
        """Test YAML configuration loading."""
        # Create a temporary config directory with test file
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()

            # Create test YAML file
            test_yaml = config_dir / "test.yaml"
            test_yaml.write_text("""
test:
  key: value
  number: 42
""")

            loader = ConfigLoader(str(config_dir))

            # Test with valid YAML
            config = loader.load_yaml_config("test.yaml")
            assert config == {"test": {"key": "value", "number": 42}}

    def test_load_json_config(self):
        """Test JSON configuration loading."""
        # Create a temporary config directory with test file
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()

            # Create test JSON file
            test_json = config_dir / "test.json"
            test_json.write_text('{"test": {"key": "value", "number": 42}}')

            loader = ConfigLoader(str(config_dir))

            # Test with valid JSON
            config = loader.load_json_config("test.json")
            assert config == {"test": {"key": "value", "number": 42}}

    def test_validate_config(self):
        """Test configuration validation."""
        loader = ConfigLoader()

        # Valid config
        valid_config = {
            "general": {"output_dir": "/tmp/test"},
            "github": {"token": "test_token"},
        }
        errors = loader.validate_config(valid_config, "clone_config")
        assert len(errors) == 0

        # Invalid config
        invalid_config = {
            "general": {"output_dir": ""},  # Empty string should fail
            "github": {"max_retries": 0},  # Zero should fail
        }
        errors = loader.validate_config(invalid_config, "clone_config")
        assert len(errors) > 0

    def test_load_clone_config(self):
        """Test clone configuration loading."""
        loader = ConfigLoader()

        # Mock example.yaml file
        config_data = {
            "general": {"output_dir": "/tmp/test_output"},
            "github": {"token": "test_token"},
            "concurrency": {"max_workers": 8},
        }

        with patch.object(loader, "load_yaml_config") as mock_load:
            mock_load.return_value = config_data
            clone_config = loader.load_clone_config()

            assert clone_config.output_dir == "/tmp/test_output"
            assert clone_config.github_token == "test_token"
            assert clone_config.max_workers == 8

    def test_load_target_repos_config(self):
        """Test target repositories configuration loading."""
        loader = ConfigLoader()

        # Mock target_repos.yaml file
        repos_data = {
            "repositories": [
                {
                    "owner": "testuser",
                    "repo": "testrepo",
                    "branch": "develop",
                    "enabled": True,
                }
            ]
        }

        with patch.object(loader, "load_yaml_config") as mock_load:
            mock_load.return_value = repos_data
            repos = loader.load_target_repos_config()

            assert len(repos) == 1
            assert repos[0].owner == "testuser"
            assert repos[0].repo == "testrepo"
            assert repos[0].branch == "develop"

    def test_load_target_users_config(self):
        """Test target users configuration loading."""
        loader = ConfigLoader()

        # Mock target_users.yaml file
        users_data = {
            "users": [{"username": "testuser", "max_repos": 5, "enabled": True}]
        }

        with patch.object(loader, "load_yaml_config") as mock_load:
            mock_load.return_value = users_data
            users = loader.load_target_users_config()

            assert len(users) == 1
            assert users[0].username == "testuser"
            assert users[0].max_repos == 5

    def test_save_config(self):
        """Test configuration saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigLoader(temp_dir)

            test_config = {"test": {"key": "value"}}
            loader.save_config(test_config, "test.json")

            # Verify file was created and contains correct data
            config_file = Path(temp_dir) / "test.json"
            assert config_file.exists()

            with open(config_file) as f:
                saved_config = json.load(f)
                assert saved_config == test_config


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_clone_config_function(self):
        """Test load_clone_config convenience function."""
        with patch(
            "geo_infer_git.utils.config_loader.ConfigLoader"
        ) as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.load_clone_config.return_value = CloneConfig()

            config = load_clone_config("/tmp/test")

            mock_loader_class.assert_called_once()
            mock_loader.load_clone_config.assert_called_once_with()
            assert isinstance(config, CloneConfig)

    def test_load_target_repos_config_function(self):
        """Test load_target_repos_config convenience function."""
        with patch(
            "geo_infer_git.utils.config_loader.ConfigLoader"
        ) as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.load_target_repos_config.return_value = []

            repos = load_target_repos_config("/tmp/test")

            mock_loader_class.assert_called_once()
            mock_loader.load_target_repos_config.assert_called_once_with()
            assert repos == []

    def test_load_target_users_config_function(self):
        """Test load_target_users_config convenience function."""
        with patch(
            "geo_infer_git.utils.config_loader.ConfigLoader"
        ) as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.load_target_users_config.return_value = []

            users = load_target_users_config("/tmp/test")

            mock_loader_class.assert_called_once()
            mock_loader.load_target_users_config.assert_called_once_with()
            assert users == []


class TestErrorHandling:
    """Test error handling in config loader."""

    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        loader = ConfigLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_yaml_config("nonexistent.yaml")

    def test_invalid_yaml(self):
        """Test handling of invalid YAML."""
        # Create a temporary config directory with invalid YAML file
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()

            # Create invalid YAML file
            test_yaml = config_dir / "test.yaml"
            test_yaml.write_text("""
invalid: yaml: content: [
""")

            loader = ConfigLoader(str(config_dir))

            with pytest.raises(yaml.YAMLError):
                loader.load_yaml_config("test.yaml")

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        # Create a temporary config directory with invalid JSON file
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()

            # Create invalid JSON file
            test_json = config_dir / "test.json"
            test_json.write_text('{"invalid": json}')

            loader = ConfigLoader(str(config_dir))

            with pytest.raises(json.JSONDecodeError):
                loader.load_json_config("test.json")
