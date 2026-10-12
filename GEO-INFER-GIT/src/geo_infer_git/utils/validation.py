#!/usr/bin/env python3

"""
Validation utilities for GEO-INFER-GIT.

This module provides comprehensive validation functionality for
configuration files, repository data, and user inputs.
"""

import os
import re
import json
from typing import Any
from pathlib import Path
import jsonschema
from urllib.parse import urlparse


class ConfigValidator:
    """
    Configuration file validator with comprehensive validation rules.

    Validates YAML and JSON configuration files against schemas and
    provides detailed error reporting.
    """

    def __init__(self) -> None:
        """Initialize the configuration validator."""
        self.schemas = self._load_default_schemas()
        self.custom_schemas: dict[str, Any] = {}

    def _load_default_schemas(self) -> dict[str, dict[str, Any]]:
        """Load default validation schemas."""
        return {
            "clone_config": {
                "type": "object",
                "properties": {
                    "general": {
                        "type": "object",
                        "properties": {
                            "output_dir": {"type": "string", "minLength": 1},
                            "base_dir": {"type": "string", "minLength": 1},
                        },
                    },
                    "github": {
                        "type": "object",
                        "properties": {
                            "token": {"type": "string"},
                            "api_url": {"type": "string", "format": "uri"},
                            "wait_on_rate_limit": {"type": "boolean"},
                            "max_retries": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "retry_delay": {
                                "type": "number",
                                "minimum": 0.1,
                                "maximum": 60,
                            },
                        },
                    },
                    "concurrency": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "max_workers": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                            },
                        },
                    },
                    "logging": {
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "string",
                                "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                            },
                            "format": {"type": "string", "enum": ["json", "text"]},
                            "file": {"type": "string"},
                        },
                    },
                },
            },
            "target_repositories": {
                "type": "object",
                "properties": {
                    "repositories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "owner": {
                                    "type": "string",
                                    "pattern": "^[a-zA-Z0-9]([a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$",
                                },
                                "repo": {
                                    "type": "string",
                                    "pattern": "^[a-zA-Z0-9._-]+$",
                                },
                                "branch": {"type": "string", "minLength": 1},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "clone_depth": {"type": "integer", "minimum": 0},
                                "enabled": {"type": "boolean"},
                            },
                            "required": ["owner", "repo"],
                        },
                    }
                },
            },
            "target_users": {
                "type": "object",
                "properties": {
                    "users": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "username": {
                                    "type": "string",
                                    "pattern": "^[a-zA-Z0-9]([a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$",
                                },
                                "include_repos": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "exclude_repos": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "max_repos": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "enabled": {"type": "boolean"},
                            },
                            "required": ["username"],
                        },
                    }
                },
            },
        }

    def add_custom_schema(self, name: str, schema: dict[str, Any]) -> None:
        """
        Add a custom validation schema.

        Args:
            name: Schema name
            schema: JSON schema definition
        """
        self.custom_schemas[name] = schema

    def validate_config(self, config: dict[str, Any], schema_name: str) -> list[str]:
        """
        Validate configuration against a schema.

        Args:
            config: Configuration to validate
            schema_name: Name of schema to use

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Get schema
        schema = self.schemas.get(schema_name) or self.custom_schemas.get(schema_name)
        if not schema:
            errors.append(f"Schema '{schema_name}' not found")
            return errors

        # Validate against schema
        try:
            jsonschema.validate(config, schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Invalid schema: {e.message}")

        return errors

    def validate_github_url(self, url: str) -> list[str]:
        """
        Validate GitHub repository URL format.

        Args:
            url: URL to validate

        Returns:
            List of validation error messages
        """
        errors = []

        if not url:
            errors.append("URL cannot be empty")
            return errors

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            errors.append(f"Invalid URL format: {e}")
            return errors

        # Check scheme
        if parsed.scheme not in ["http", "https", "git", "ssh"]:
            errors.append(f"Invalid URL scheme: {parsed.scheme}")

        # Check hostname for GitHub
        if "github.com" not in parsed.netloc:
            errors.append(f"Not a GitHub URL: {parsed.netloc}")

        # Check path format
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(path_parts) < 2:
            errors.append("URL must include owner and repository name")
        elif len(path_parts) > 2:
            errors.append("URL contains too many path components")

        # Validate owner and repo names
        if len(path_parts) >= 2:
            owner, repo = path_parts[0], path_parts[1]
            if not re.match(
                r"^[a-zA-Z0-9]([a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$", owner
            ):
                errors.append(f"Invalid owner name: {owner}")
            if not re.match(r"^[a-zA-Z0-9._-]+$", repo):
                errors.append(f"Invalid repository name: {repo}")

        return errors

    def validate_directory_path(
        self, path: str, must_exist: bool = False, writable: bool = True
    ) -> list[str]:
        """
        Validate directory path.

        Args:
            path: Directory path to validate
            must_exist: Whether directory must exist
            writable: Whether directory must be writable

        Returns:
            List of validation error messages
        """
        errors = []

        if not path:
            errors.append("Directory path cannot be empty")
            return errors

        try:
            path_obj = Path(path).resolve()
        except Exception as e:
            errors.append(f"Invalid path format: {e}")
            return errors

        # Check if path exists
        if must_exist and not path_obj.exists():
            errors.append(f"Directory does not exist: {path}")
        elif path_obj.exists() and not path_obj.is_dir():
            errors.append(f"Path exists but is not a directory: {path}")

        # Check writability
        if writable:
            try:
                # Try to create a test file
                test_file = path_obj / ".test_write"
                test_file.touch()
                test_file.unlink()
            except Exception:
                errors.append(f"Directory is not writable: {path}")

        # Check parent directory exists for non-existent paths
        if not path_obj.exists() and not path_obj.parent.exists():
            errors.append(f"Parent directory does not exist: {path_obj.parent}")

        return errors

    def validate_github_token(self, token: str) -> list[str]:
        """
        Validate GitHub token format.

        Args:
            token: GitHub token to validate

        Returns:
            List of validation error messages
        """
        errors = []

        if not token:
            errors.append("Token cannot be empty")
            return errors

        # GitHub tokens are typically 40 characters for personal access tokens
        # or longer for fine-grained tokens
        if len(token) < 20:
            errors.append("Token appears to be too short")

        # Check for valid characters (GitHub tokens use base64-like encoding)
        if not re.match(r"^[a-zA-Z0-9_-]+$", token):
            errors.append("Token contains invalid characters")

        return errors

    def validate_branch_name(self, branch: str) -> list[str]:
        """
        Validate Git branch name.

        Args:
            branch: Branch name to validate

        Returns:
            List of validation error messages
        """
        errors = []

        if not branch:
            errors.append("Branch name cannot be empty")
            return errors

        # Git branch name rules
        if len(branch) > 255:
            errors.append("Branch name too long (max 255 characters)")

        # Check for invalid characters
        invalid_chars = [" ", "..", "~", "^", ":", "?", "*", "[", "\\"]
        for char in invalid_chars:
            if char in branch:
                errors.append(f"Branch name contains invalid character: {char}")

        # Check for reserved names
        reserved_names = ["HEAD", "refs", "refs/heads", "refs/tags"]
        if branch.lower() in [name.lower() for name in reserved_names]:
            errors.append(f"Branch name '{branch}' is reserved")

        return errors


class RepositoryValidator:
    """
    Repository data validator for ensuring data integrity.
    """

    def __init__(self) -> None:
        """Initialize the repository validator."""
        self.required_fields = ["name", "url"]
        self.validation_errors: list[str] = []

    def validate_repository_data(self, repo_data: dict[str, Any]) -> list[str]:
        """
        Validate repository data structure.

        Args:
            repo_data: Repository data dictionary

        Returns:
            List of validation error messages
        """
        errors = []

        # Required fields
        required_fields = ["name", "owner", "url", "clone_url"]
        for field in required_fields:
            if field not in repo_data or not repo_data[field]:
                errors.append(f"Missing or empty required field: {field}")

        # Validate URLs if present
        url_fields = ["url", "clone_url", "ssh_url"]
        for field in url_fields:
            if field in repo_data and repo_data[field]:
                url_errors = self.validate_github_url(repo_data[field])
                errors.extend([f"{field}: {error}" for error in url_errors])

        # Validate numeric fields
        if "stars" in repo_data and not isinstance(repo_data["stars"], int):
            errors.append("Stars must be an integer")

        if "size" in repo_data and not isinstance(repo_data["size"], int):
            errors.append("Size must be an integer")

        return errors

    def validate_github_url(self, url: str) -> list[str]:
        """Validate GitHub URL format."""
        validator = ConfigValidator()
        return validator.validate_github_url(url)

    def validate_owner_repo_format(self, owner: str, repo: str) -> list[str]:
        """
        Validate owner and repository name format.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            List of validation error messages
        """
        errors = []

        # Owner validation (GitHub username/organization rules)
        if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$", owner):
            errors.append(f"Invalid owner format: {owner}")

        # Repository validation
        if not re.match(r"^[a-zA-Z0-9._-]+$", repo):
            errors.append(f"Invalid repository format: {repo}")

        return errors


class InputValidator:
    """
    Input validation utilities for user inputs and command-line arguments.
    """

    def __init__(self) -> None:
        """Initialize the input validator."""
        self.validation_errors: list[str] = []

    def validate_positive_integer(self, value: Any, field_name: str) -> list[str]:
        """
        Validate that a value is a positive integer.

        Args:
            value: Value to validate
            field_name: Name of the field for error messages

        Returns:
            List of validation error messages
        """
        errors = []

        if value is None:
            errors.append(f"{field_name} cannot be None")
            return errors

        try:
            int_value = int(value)
            if int_value <= 0:
                errors.append(f"{field_name} must be positive")
        except (ValueError, TypeError):
            errors.append(f"{field_name} must be an integer")

        return errors

    def validate_string_length(
        self,
        value: Any,
        field_name: str,
        min_length: int = 0,
        max_length: int | None = None,
    ) -> list[str]:
        """
        Validate string length.

        Args:
            value: String to validate
            field_name: Name of the field for error messages
            min_length: Minimum allowed length
            max_length: Maximum allowed length

        Returns:
            List of validation error messages
        """
        errors = []

        if value is None:
            errors.append(f"{field_name} cannot be None")
            return errors

        if not isinstance(value, str):
            errors.append(f"{field_name} must be a string")
            return errors

        if len(value) < min_length:
            errors.append(f"{field_name} must be at least {min_length} characters")

        if max_length and len(value) > max_length:
            errors.append(f"{field_name} must be at most {max_length} characters")

        return errors

    def validate_enum_value(
        self, value: Any, field_name: str, allowed_values: list[str]
    ) -> list[str]:
        """
        Validate that a value is in a list of allowed values.

        Args:
            value: Value to validate
            field_name: Name of the field for error messages
            allowed_values: List of allowed values

        Returns:
            List of validation error messages
        """
        errors = []

        if value is None:
            errors.append(f"{field_name} cannot be None")
            return errors

        if str(value).lower() not in [str(v).lower() for v in allowed_values]:
            errors.append(f"{field_name} must be one of: {', '.join(allowed_values)}")

        return errors


def validate_config_file(config_path: str) -> tuple[bool, list[str]]:
    """
    Validate a configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check if file exists
    if not os.path.exists(config_path):
        errors.append(f"Configuration file not found: {config_path}")
        return False, errors

    # Check file extension
    if not config_path.endswith((".yaml", ".yml", ".json")):
        errors.append(f"Unsupported file format: {config_path}")
        return False, errors

    try:
        # Load file
        if config_path.endswith(".json"):
            with open(config_path) as f:
                config = json.load(f)
        else:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

        if config is None:
            errors.append("Configuration file is empty")
            return False, errors

        # Determine schema based on filename
        filename = os.path.basename(config_path)
        if "target_repos" in filename or "repositories" in filename:
            schema_name = "target_repositories"
        elif "target_users" in filename or "users" in filename:
            schema_name = "target_users"
        elif "clone" in filename or "config" in filename:
            schema_name = "clone_config"
        else:
            errors.append(f"Cannot determine schema for file: {filename}")
            return False, errors

        # Validate against schema
        validator = ConfigValidator()
        schema_errors = validator.validate_config(config, schema_name)
        errors.extend(schema_errors)

    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON format: {e}")
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML format: {e}")
    except Exception as e:
        errors.append(f"Error reading configuration file: {e}")

    return len(errors) == 0, errors


def validate_github_credentials(
    token: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> list[str]:
    """
    Validate GitHub authentication credentials.

    Args:
        token: GitHub personal access token
        username: GitHub username
        password: GitHub password

    Returns:
        List of validation error messages
    """
    errors = []

    # At least one authentication method must be provided
    if not token and not (username and password):
        errors.append("Either token or username+password must be provided")
        return errors

    # Validate token if provided
    if token:
        validator = ConfigValidator()
        token_errors = validator.validate_github_token(token)
        errors.extend(token_errors)

    # Validate username if provided
    if username:
        input_validator = InputValidator()
        username_errors = input_validator.validate_string_length(
            username, "username", 1, 39
        )
        errors.extend(username_errors)

    # Validate password if provided
    if password and len(password) < 8:
        errors.append("Password must be at least 8 characters")

    return errors
