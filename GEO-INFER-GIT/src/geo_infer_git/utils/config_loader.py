#!/usr/bin/env python3

"""
Configuration loader utilities for GEO-INFER-GIT.

This module provides functionality to load and validate configuration files
for the GEO-INFER-GIT repository management system.
"""

import os
import importlib.resources
import yaml
import json
import logging
from pathlib import Path
from typing import Any, cast
from dataclasses import dataclass, field
import jsonschema

logger = logging.getLogger(__name__)


@dataclass
class CloneConfig:
    """Configuration for repository cloning operations."""

    # General settings
    output_dir: str = "./cloned_repositories"
    base_dir: str = "./repos"

    # GitHub API settings
    github_token: str = field(
        default_factory=lambda: os.environ.get("GITHUB_TOKEN", "")
    )
    github_api_url: str = "https://api.github.com"
    github_wait_on_rate_limit: bool = True
    github_max_retries: int = 3
    github_retry_delay: float = 1.0

    # Concurrency settings
    concurrency_enabled: bool = True
    max_workers: int = 4

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "json"
    report_format: str = "markdown"

    # Authentication settings
    auth_method: str = "https"  # https, ssh, token

    # Repository settings
    default_branch: str = "main"
    clone_depth: int = 1
    sparse_checkout: bool = False

    # Sync settings
    auto_sync: bool = True
    sync_interval: int = 3600  # seconds

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.github_token and self.auth_method == "token":
            logger.warning("GitHub token not provided but auth_method is 'token'")

        if self.max_workers < 1:
            logger.warning(
                f"Invalid max_workers value {self.max_workers}, setting to 1"
            )
            self.max_workers = 1


@dataclass
class TargetRepository:
    """Configuration for a target repository to clone."""

    owner: str
    repo: str
    branch: str = "main"
    tags: list[str] = field(default_factory=list)
    clone_depth: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate repository configuration."""
        if not self.owner or not self.repo:
            raise ValueError("Repository owner and name are required")

        if self.clone_depth < 0:
            raise ValueError("Clone depth must be non-negative")


@dataclass
class TargetUser:
    """Configuration for repositories from a specific user."""

    username: str
    include_repos: list[str] = field(default_factory=list)
    exclude_repos: list[str] = field(default_factory=list)
    max_repos: int = 10
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate user configuration."""
        if not self.username:
            raise ValueError("Username is required")

        if self.max_repos < 1:
            logger.warning(f"Invalid max_repos value {self.max_repos}, setting to 1")
            self.max_repos = 1


class ConfigLoader:
    """
    Configuration loader and validator for GEO-INFER-GIT.

    Handles loading, validation, and management of configuration files
    for repository cloning operations.
    """

    def __init__(self, config_dir: str | None = None):
        """
        Initialize the configuration loader.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        elif os.environ.get("GEO_INFER_GIT_CONFIG"):
            # Explicit env-var override pointing at a checkout's config directory
            self.config_dir = Path(os.environ["GEO_INFER_GIT_CONFIG"])
        else:
            # Packaged default: config resources ship inside the package tree
            self.config_dir = importlib.resources.files("geo_infer_git") / "config"

        self.schemas = self._load_schemas()
        self.config_cache: dict[str, dict[str, Any]] = {}

    def _load_schemas(self) -> dict[str, dict]:
        """Load JSON schemas for configuration validation."""
        schemas = {}

        # Define schemas for each configuration type
        schemas["clone_config"] = {
            "type": "object",
            "properties": {
                "general": {
                    "type": "object",
                    "properties": {
                        "output_dir": {"type": "string"},
                        "base_dir": {"type": "string"},
                    },
                },
                "github": {
                    "type": "object",
                    "properties": {
                        "token": {"type": "string"},
                        "api_url": {"type": "string"},
                        "wait_on_rate_limit": {"type": "boolean"},
                        "max_retries": {"type": "integer", "minimum": 1},
                        "retry_delay": {"type": "number", "minimum": 0},
                    },
                },
                "concurrency": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "max_workers": {"type": "integer", "minimum": 1},
                    },
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "log_level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        },
                        "log_format": {"type": "string", "enum": ["json", "text"]},
                        "report_format": {
                            "type": "string",
                            "enum": ["markdown", "html", "json"],
                        },
                    },
                },
            },
        }

        schemas["target_repos"] = {
            "type": "object",
            "properties": {
                "repositories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string"},
                            "repo": {"type": "string"},
                            "branch": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "clone_depth": {"type": "integer", "minimum": 0},
                            "enabled": {"type": "boolean"},
                        },
                        "required": ["owner", "repo"],
                    },
                }
            },
        }

        schemas["target_users"] = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "include_repos": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "exclude_repos": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "max_repos": {"type": "integer", "minimum": 1},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "enabled": {"type": "boolean"},
                        },
                        "required": ["username"],
                    },
                }
            },
        }

        return schemas

    def load_yaml_config(self, filename: str) -> dict[str, Any]:
        """
        Load a YAML configuration file with validation.

        Args:
            filename: Name of the configuration file (without path)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            jsonschema.ValidationError: If configuration doesn't match schema
        """
        cache_key = f"yaml_{filename}"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]

        try:
            with importlib.resources.as_file(self.config_dir / filename) as config_path:
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_path}"
                    )

                with config_path.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                if config is None:
                    config = {}

                # Validate against schema if available
                schema_key = filename.replace(".yaml", "").replace(".yml", "")
                if schema_key in self.schemas:
                    jsonschema.validate(config, self.schemas[schema_key])

                self.config_cache[cache_key] = config
                logger.info(f"Loaded configuration from {config_path}")
                return cast(dict[str, Any], config)

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filename}: {e}")
            raise
        except jsonschema.ValidationError as e:
            logger.error(f"Configuration validation failed for {filename}: {e}")
            raise

    def load_json_config(self, filename: str) -> dict[str, Any]:
        """
        Load a JSON configuration file with validation.

        Args:
            filename: Name of the configuration file (without path)

        Returns:
            Configuration dictionary
        """
        cache_key = f"json_{filename}"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        try:
            with importlib.resources.as_file(self.config_dir / filename) as config_path:
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_path}"
                    )

                with config_path.open("r", encoding="utf-8") as f:
                    config = json.load(f)

                self.config_cache[cache_key] = config
                logger.info(f"Loaded configuration from {config_path}")
                return cast(dict[str, Any], config)

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {filename}: {e}")
            raise

    def load_clone_config(self, config_dir: str | None = None) -> CloneConfig:
        """
        Load and merge clone configuration from multiple sources.

        Args:
            config_dir: Optional config directory override

        Returns:
            CloneConfig object with merged configuration
        """
        if config_dir:
            old_dir = self.config_dir
            self.config_dir = Path(config_dir)

        try:
            # Try to load from example.yaml first
            try:
                config_data = self.load_yaml_config("example.yaml")
            except FileNotFoundError:
                # Fall back to default configuration
                config_data = {}

            # Convert to CloneConfig object
            clone_config = CloneConfig()

            # Update from loaded configuration
            if "general" in config_data:
                general = config_data["general"]
                if "output_dir" in general:
                    clone_config.output_dir = general["output_dir"]
                if "base_dir" in general:
                    clone_config.base_dir = general["base_dir"]

            if "github" in config_data:
                github = config_data["github"]
                if "token" in github:
                    clone_config.github_token = github["token"]
                if "api_url" in github:
                    clone_config.github_api_url = github["api_url"]
                if "wait_on_rate_limit" in github:
                    clone_config.github_wait_on_rate_limit = github[
                        "wait_on_rate_limit"
                    ]
                if "max_retries" in github:
                    clone_config.github_max_retries = github["max_retries"]
                if "retry_delay" in github:
                    clone_config.github_retry_delay = github["retry_delay"]

            if "concurrency" in config_data:
                concurrency = config_data["concurrency"]
                if "enabled" in concurrency:
                    clone_config.concurrency_enabled = concurrency["enabled"]
                if "max_workers" in concurrency:
                    clone_config.max_workers = concurrency["max_workers"]

            if "logging" in config_data:
                logging_config = config_data["logging"]
                if "log_level" in logging_config:
                    clone_config.log_level = logging_config["log_level"]
                if "log_format" in logging_config:
                    clone_config.log_format = logging_config["log_format"]
                if "report_format" in logging_config:
                    clone_config.report_format = logging_config["report_format"]

            return clone_config

        finally:
            if config_dir:
                self.config_dir = old_dir

    def load_target_repos_config(
        self, config_dir: str | None = None
    ) -> list[TargetRepository]:
        """
        Load target repositories configuration.

        Args:
            config_dir: Optional config directory override

        Returns:
            List of TargetRepository objects
        """
        if config_dir:
            old_dir = self.config_dir
            self.config_dir = Path(config_dir)

        try:
            try:
                config_data = self.load_yaml_config("target_repos.yaml")
            except FileNotFoundError:
                return []

            repositories = []
            for repo_data in config_data.get("repositories", []):
                try:
                    repo = TargetRepository(**repo_data)
                    if repo.enabled:
                        repositories.append(repo)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Skipping invalid repository configuration: {e}")
                    continue

            return repositories

        finally:
            if config_dir:
                self.config_dir = old_dir

    def load_target_users_config(
        self, config_dir: str | None = None
    ) -> list[TargetUser]:
        """
        Load target users configuration.

        Args:
            config_dir: Optional config directory override

        Returns:
            List of TargetUser objects
        """
        if config_dir:
            old_dir = self.config_dir
            self.config_dir = Path(config_dir)

        try:
            try:
                config_data = self.load_yaml_config("target_users.yaml")
            except FileNotFoundError:
                return []

            users = []
            for user_data in config_data.get("users", []):
                try:
                    user = TargetUser(**user_data)
                    if user.enabled:
                        users.append(user)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Skipping invalid user configuration: {e}")
                    continue

            return users

        finally:
            if config_dir:
                self.config_dir = old_dir

    def save_config(self, config: dict[str, Any], filename: str) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary to save
            filename: Name of the file to save to (without path)
        """
        if not isinstance(self.config_dir, Path):
            raise RuntimeError(
                "save_config requires a filesystem config directory; "
                "the packaged default config is read-only"
            )

        config_path = self.config_dir / filename

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                if filename.endswith(".json"):
                    json.dump(config, f, indent=2)
                else:
                    yaml.dump(config, f, default_flow_style=False, indent=2)

            logger.info(f"Saved configuration to {config_path}")

            # Clear cache for this file
            cache_key = f"{'json' if filename.endswith('.json') else 'yaml'}_{filename}"
            self.config_cache.pop(cache_key, None)

        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise

    def validate_config(self, config: dict[str, Any], schema_name: str) -> list[str]:
        """
        Validate configuration against schema.

        Args:
            config: Configuration to validate
            schema_name: Name of schema to validate against

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if schema_name not in self.schemas:
            errors.append(f"Schema not found: {schema_name}")
            return errors

        try:
            jsonschema.validate(config, self.schemas[schema_name])
            return errors
        except jsonschema.ValidationError as e:
            errors.append(f"Configuration validation failed: {e.message}")
            return errors

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self.config_cache.clear()
        logger.debug("Configuration cache cleared")


# Convenience constructors for the public configuration API
def load_clone_config(config_dir: str | None = None) -> CloneConfig:
    """Load clone configuration."""
    loader = ConfigLoader(config_dir)
    return loader.load_clone_config()


def load_target_repos_config(
    config_dir: str | None = None,
) -> list[TargetRepository]:
    """Load target repositories configuration."""
    loader = ConfigLoader(config_dir)
    return loader.load_target_repos_config()


def load_target_users_config(config_dir: str | None = None) -> list[TargetUser]:
    """Load target users configuration."""
    loader = ConfigLoader(config_dir)
    return loader.load_target_users_config()
