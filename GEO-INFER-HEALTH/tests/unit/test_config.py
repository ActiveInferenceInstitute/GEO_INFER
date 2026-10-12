"""
Unit tests for configuration utilities.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from geo_infer_health.utils.config import (
    load_yaml_config,
    load_json_config,
    validate_config,
    merge_configs,
    resolve_environment_variables,
    HealthConfig,
    load_config,
    save_config,
    get_config_value,
)


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_yaml_config(self):
        """Test loading YAML configuration."""
        config_data = {
            "module": {"name": "test", "version": "1.0.0"},
            "api": {"host": "localhost", "port": 8000},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.safe_dump(config_data, f)
            temp_path = f.name

        try:
            loaded_config = load_yaml_config(temp_path)
            assert loaded_config["module"]["name"] == "test"
            assert loaded_config["api"]["port"] == 8000
        finally:
            os.unlink(temp_path)

    def test_load_json_config(self):
        """Test loading JSON configuration."""
        config_data = {
            "module": {"name": "test", "version": "1.0.0"},
            "api": {"host": "localhost", "port": 8000},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(config_data, f)
            temp_path = f.name

        try:
            loaded_config = load_json_config(temp_path)
            assert loaded_config["module"]["name"] == "test"
            assert loaded_config["api"]["port"] == 8000
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_config(self):
        """Test loading nonexistent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("nonexistent_file.yaml")


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        config_data = {
            "module": {
                "name": "GEO-INFER-HEALTH",
                "version": "1.0.0",
                "description": "Test module",
            },
            "api": {"host": "127.0.0.1", "port": 8000, "workers": 1},
            "database": {"type": "memory"},
            "logging": {"level": "INFO"},
            "analysis": {
                "disease_surveillance": {
                    "default_scan_radius_km": 1.0,
                    "hotspot_threshold_cases": 5,
                }
            },
        }

        validated_config = validate_config(config_data)
        assert isinstance(validated_config, HealthConfig)
        assert validated_config.module["name"] == "GEO-INFER-HEALTH"

    def test_validate_invalid_config(self):
        """Test validating an invalid configuration."""
        invalid_config = {
            "module": {
                "name": "test",
                "version": "invalid_version",  # Should be semantic version
            }
        }

        with pytest.raises(ValueError, match="semantic version"):
            validate_config(invalid_config)

    def test_health_config_creation(self):
        """Test creating HealthConfig object."""
        config_data = {
            "module": {"name": "test", "version": "1.0.0"},
            "api": {"host": "localhost", "port": 8000},
            "database": {"type": "memory"},
        }

        config = HealthConfig(**config_data)
        assert config.module["name"] == "test"
        assert config.api["port"] == 8000


class TestConfigMerging:
    """Test configuration merging."""

    def test_merge_configs_basic(self):
        """Test basic configuration merging."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        merged = merge_configs(base, override)

        assert merged["a"] == 1
        assert merged["b"] == 3  # Override takes precedence
        assert merged["c"] == 4

    def test_merge_configs_nested(self):
        """Test nested configuration merging."""
        base = {
            "api": {"host": "localhost", "port": 8000},
            "database": {"type": "sqlite"},
        }
        override = {"api": {"port": 9000}, "logging": {"level": "DEBUG"}}

        merged = merge_configs(base, override)

        assert merged["api"]["host"] == "localhost"
        assert merged["api"]["port"] == 9000  # Overridden
        assert merged["database"]["type"] == "sqlite"
        assert merged["logging"]["level"] == "DEBUG"


class TestEnvironmentVariableResolution:
    """Test environment variable resolution."""

    def test_resolve_environment_variables(self):
        """Test resolving environment variables in config."""
        with patch.dict(os.environ, {"TEST_HOST": "testhost", "TEST_PORT": "9000"}):
            config = {
                "api": {
                    "host": "${TEST_HOST}",
                    "port": "${TEST_PORT}",
                    "timeout": "${TEST_TIMEOUT:30}",  # With default
                }
            }

            resolved = resolve_environment_variables(config)

            assert resolved["api"]["host"] == "testhost"
            assert resolved["api"]["port"] == "9000"
            assert resolved["api"]["timeout"] == "30"  # Default value

    def test_resolve_missing_environment_variable(self):
        """Test handling missing environment variables."""
        config = {"api": {"host": "${MISSING_VAR}", "port": 8000}}

        resolved = resolve_environment_variables(config)

        # Should keep original string if variable not found
        assert resolved["api"]["host"] == "${MISSING_VAR}"


class TestConfigFileOperations:
    """Test configuration file save/load operations."""

    def test_save_and_load_yaml_config(self):
        """Test saving and loading YAML configuration."""
        config_data = {
            "module": {"name": "test", "version": "1.0.0"},
            "api": {"host": "localhost", "port": 8000},
        }

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            save_config(config_data, temp_path)
            loaded_config = load_yaml_config(temp_path)

            assert loaded_config["module"]["name"] == "test"
            assert loaded_config["api"]["port"] == 8000
        finally:
            os.unlink(temp_path)

    def test_save_and_load_json_config(self):
        """Test saving and loading JSON configuration."""
        config_data = {
            "module": {"name": "test", "version": "1.0.0"},
            "api": {"host": "localhost", "port": 8000},
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_config(config_data, temp_path)
            loaded_config = load_json_config(temp_path)

            assert loaded_config["module"]["name"] == "test"
            assert loaded_config["api"]["port"] == 8000
        finally:
            os.unlink(temp_path)


class TestConfigValueRetrieval:
    """Test configuration value retrieval."""

    def test_get_config_value_simple(self):
        """Test getting simple configuration values."""
        config = HealthConfig(
            module={"name": "test", "version": "1.0.0"},
            api={"host": "localhost", "port": 8000},
        )

        assert get_config_value(config, "module.name") == "test"
        assert get_config_value(config, "api.port") == 8000

    def test_get_config_value_nested(self):
        """Test getting nested configuration values."""
        config = HealthConfig(
            analysis={
                "disease_surveillance": {
                    "default_scan_radius_km": 1.0,
                    "hotspot_threshold_cases": 5,
                }
            }
        )

        assert (
            get_config_value(
                config, "analysis.disease_surveillance.default_scan_radius_km"
            )
            == 1.0
        )
        assert (
            get_config_value(
                config, "analysis.disease_surveillance.hotspot_threshold_cases"
            )
            == 5
        )

    def test_get_config_value_with_default(self):
        """Test getting configuration values with defaults."""
        config = HealthConfig()

        assert (
            get_config_value(config, "nonexistent.key", "default_value")
            == "default_value"
        )
        assert get_config_value(config, "api.host", "localhost") == "localhost"

    def test_get_config_value_missing_key(self):
        """Test getting missing configuration values."""
        config = HealthConfig()

        assert get_config_value(config, "nonexistent.key") is None
        assert get_config_value(config, "nonexistent.deep.key", "default") == "default"


class TestLoadConfigIntegration:
    """Test integrated configuration loading."""

    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "module": {"name": "test", "version": "1.0.0"},
            "api": {"host": "localhost", "port": 8000},
            "database": {"type": "memory"},
            "analysis": {
                "disease_surveillance": {
                    "default_scan_radius_km": 1.0,
                    "hotspot_threshold_cases": 5,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.safe_dump(config_data, f)
            temp_path = f.name

        try:
            with patch(
                "geo_infer_health.utils.config.get_default_config_path",
                return_value=Path(temp_path),
            ):
                loaded_config = load_config()

                assert isinstance(loaded_config, HealthConfig)
                assert loaded_config.module["name"] == "test"
                assert loaded_config.api["port"] == 8000
        finally:
            os.unlink(temp_path)

    def test_load_config_with_env_vars(self):
        """Test loading configuration with environment variable resolution."""
        config_data = {
            "api": {"host": "${TEST_HOST:default_host}", "port": "${TEST_PORT:8000}"},
            "database": {"connection_string": "${DB_URL:sqlite:///test.db}"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.safe_dump(config_data, f)
            temp_path = f.name

        try:
            with patch.dict(
                os.environ, {"TEST_HOST": "env_host", "DB_URL": "postgresql://test"}
            ):
                with patch(
                    "geo_infer_health.utils.config.get_default_config_path",
                    return_value=Path(temp_path),
                ):
                    loaded_config = load_config()

                    assert loaded_config.api["host"] == "env_host"  # From env
                    assert loaded_config.api["port"] == "8000"  # Default value
                    assert (
                        loaded_config.database["connection_string"]
                        == "postgresql://test"
                    )  # From env
        finally:
            os.unlink(temp_path)


class TestConfigErrorHandling:
    """Test configuration error handling."""

    def test_load_config_invalid_format(self):
        """Test loading configuration with invalid format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("invalid format content")
            temp_path = f.name

        try:
            with patch(
                "geo_infer_health.utils.config.get_default_config_path",
                return_value=Path(temp_path),
            ):
                with pytest.raises(ValueError):
                    load_config()
        finally:
            os.unlink(temp_path)

    def test_save_config_invalid_path(self, tmp_path: Path) -> None:
        """Saving below a regular file fails regardless of user privileges."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(NotADirectoryError):
            save_config({"test": "data"}, blocker / "nested" / "config.yaml")

    def test_validate_config_type_mismatch(self):
        """Test validation with type mismatches."""
        invalid_config = {
            "api": {
                "port": "not_a_number"  # Should be integer
            }
        }

        with pytest.raises(ValueError, match="api.port"):
            validate_config(invalid_config)
