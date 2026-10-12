"""Top-level test fixtures and configuration for the entire GEO-INFER framework."""

import os
import yaml
import pytest
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from collections.abc import Generator
from geo_infer_intra.utils.config import load_default_config

# List of all GEO-INFER modules
GEO_INFER_MODULES = [
    "geo_infer_act",
    "geo_infer_ag",
    "geo_infer_agent",
    "geo_infer_ai",
    "geo_infer_ant",
    "geo_infer_api",
    "geo_infer_app",
    "geo_infer_art",
    "geo_infer_bayes",
    "geo_infer_bio",
    "geo_infer_civ",
    "geo_infer_cog",
    "geo_infer_comms",
    "geo_infer_data",
    "geo_infer_econ",
    "geo_infer_git",
    "geo_infer_intra",
    "geo_infer_log",
    "geo_infer_math",
    "geo_infer_norms",
    "geo_infer_ops",
    "geo_infer_org",
    "geo_infer_pep",
    "geo_infer_req",
    "geo_infer_risk",
    "geo_infer_sec",
    "geo_infer_sim",
    "geo_infer_space",
    "geo_infer_time",
]


# Core utility classes
@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str | None = None


@dataclass
class Config:
    environment: str
    debug: bool
    logging: LoggingConfig
    module: str | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create a Config object from a dictionary."""
        return cls(
            environment=config_dict.get("environment", "development"),
            debug=config_dict.get("debug", False),
            module=config_dict.get("module"),
            logging=LoggingConfig(
                level=config_dict.get("logging", {}).get("level", "INFO"),
                format=config_dict.get("logging", {}).get("format", "text"),
                file=config_dict.get("logging", {}).get("file"),
            ),
        )


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return Config.from_dict(config_dict)


def setup_logging(
    log_level: str, json_format: bool = False, log_file: str | None = None
):
    """Set up logging with the specified configuration."""
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    # Configure format based on json_format flag
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if json_format:
        log_format = '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

    # Set up basic configuration
    logging.basicConfig(
        level=getattr(logging, log_level), format=log_format, handlers=handlers
    )


#############################################################
# Environment & Configuration Fixtures
#############################################################


@pytest.fixture(scope="session")
def test_env() -> Generator[dict[str, str], None, None]:
    """Set up test environment variables for all modules."""
    env_vars = {
        "GEO_INFER_ENV": "test",
        "GEO_INFER_DEBUG": "true",
        "GEO_INFER_LOG_LEVEL": "DEBUG",
    }

    # Add module-specific environment variables
    for module in GEO_INFER_MODULES:
        module_upper = module.upper().replace("-", "_")
        env_vars[f"{module_upper}_ENV"] = "test"
        env_vars[f"{module_upper}_DEBUG"] = "true"
        env_vars[f"{module_upper}_LOG_LEVEL"] = "DEBUG"

    # Store original environment variables
    original_env = {k: os.environ.get(k) for k in env_vars.keys()}

    # Set test environment variables
    for k, v in env_vars.items():
        os.environ[k] = v

    yield env_vars

    # Restore original environment variables
    for k, v in original_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def test_log_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for test logs."""
    log_dir = temp_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


@pytest.fixture(scope="session")
def test_config_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for test configurations."""
    config_dir = temp_dir / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir


@pytest.fixture(scope="session")
def test_data_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for test data."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def test_config_factory():
    """Factory fixture for creating module-specific configurations."""

    def _create_config(module_name: str, log_dir: Path) -> Config:
        return Config(
            environment="test",
            debug=True,
            module=module_name,
            logging=LoggingConfig(
                level="DEBUG", format="json", file=str(log_dir / f"{module_name}.log")
            ),
        )

    return _create_config


@pytest.fixture(scope="session")
def test_config_file_factory(test_config_dir: Path, test_log_dir: Path):
    """Factory fixture to create config files for different modules."""

    def _create_config_file(module_name: str) -> Path:
        config_file = test_config_dir / f"{module_name}_config.yml"
        config = {
            "module": module_name,
            "environment": "test",
            "debug": True,
            "logging": {
                "level": "DEBUG",
                "format": "json",
                "file": str(test_log_dir / f"{module_name}.log"),
            },
        }

        # Add module-specific configuration
        if module_name == "geo_infer_space":
            config["coordinate_systems"] = ["WGS84", "EPSG:3857", "EPSG:4326"]
            config["spatial_index"] = "H3"
        elif module_name == "geo_infer_time":
            config["time_formats"] = ["ISO8601", "RFC3339"]
            config["timezone"] = "UTC"
        elif module_name == "geo_infer_api":
            config["server"] = {"host": "localhost", "port": 9000, "workers": 1}
            config["auth"] = {"enabled": False}

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config_file

    return _create_config_file


#############################################################
# Common Test Data Fixtures
#############################################################


@pytest.fixture(scope="session")
def test_geojson_point():
    """Create a GeoJSON Point for testing."""
    return {"type": "Point", "coordinates": [102.0, 0.5]}


@pytest.fixture(scope="session")
def test_geojson_polygon():
    """Create a GeoJSON Polygon for testing."""
    return {
        "type": "Polygon",
        "coordinates": [
            [[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]
        ],
    }


@pytest.fixture(scope="session")
def test_geojson_feature(test_geojson_polygon):
    """Create a GeoJSON Feature for testing."""
    return {
        "type": "Feature",
        "properties": {"name": "Test Polygon", "value": 42},
        "geometry": test_geojson_polygon,
    }


@pytest.fixture(scope="session")
def test_geojson_feature_collection(test_geojson_feature, test_geojson_point):
    """Create a GeoJSON FeatureCollection for testing."""
    point_feature = {
        "type": "Feature",
        "properties": {"name": "Test Point", "value": 21},
        "geometry": test_geojson_point,
    }

    return {
        "type": "FeatureCollection",
        "features": [test_geojson_feature, point_feature],
    }


@pytest.fixture(scope="session")
def test_h3_indexes():
    """Create a list of H3 indexes for testing."""
    return [
        "8928308280fffff",  # San Francisco
        "891e9a4893fffff",  # London
        "891fb46623fffff",  # New York
    ]


@pytest.fixture(scope="session")
def test_time_series_data():
    """Create sample time series data for testing."""
    return {
        "timestamps": [
            "2023-01-01T00:00:00Z",
            "2023-01-02T00:00:00Z",
            "2023-01-03T00:00:00Z",
            "2023-01-04T00:00:00Z",
            "2023-01-05T00:00:00Z",
        ],
        "values": [10.5, 11.2, 9.8, 12.3, 10.9],
    }


#############################################################
# Module-specific Config Fixtures
#############################################################


@pytest.fixture(scope="session")
def space_config_file(test_config_file_factory):
    """Create test configuration file for geo_infer_space module."""
    return test_config_file_factory("geo_infer_space")


@pytest.fixture(scope="session")
def time_config_file(test_config_file_factory):
    """Create test configuration file for geo_infer_time module."""
    return test_config_file_factory("geo_infer_time")


@pytest.fixture(scope="session")
def api_config_file(test_config_file_factory):
    """Create test configuration file for geo_infer_api module."""
    return test_config_file_factory("geo_infer_api")


@pytest.fixture(scope="session")
def data_config_file(test_config_file_factory):
    """Create test configuration file for geo_infer_data module."""
    return test_config_file_factory("geo_infer_data")


#############################################################
# Test Validation Utilities
#############################################################


@pytest.fixture(scope="session")
def assert_valid_geojson():
    """Fixture for validating GeoJSON objects."""

    def _assert_valid_geojson(geojson_obj: dict[str, Any]) -> bool:
        """
        Validates that a dictionary conforms to GeoJSON standards.

        Returns True if valid, raises AssertionError otherwise.
        """
        # Check type field exists
        assert "type" in geojson_obj, "GeoJSON must have 'type' field"

        # Check based on type
        if geojson_obj["type"] == "Feature":
            assert "geometry" in geojson_obj, "Feature must have 'geometry' field"
            assert "properties" in geojson_obj, "Feature must have 'properties' field"
            assert isinstance(geojson_obj["properties"], dict), (
                "Properties must be an object"
            )

            # Recursive check on geometry
            if geojson_obj["geometry"] is not None:
                _assert_valid_geojson(geojson_obj["geometry"])

        elif geojson_obj["type"] == "FeatureCollection":
            assert "features" in geojson_obj, (
                "FeatureCollection must have 'features' field"
            )
            assert isinstance(geojson_obj["features"], list), (
                "Features must be an array"
            )

            # Recursive check on each feature
            for feature in geojson_obj["features"]:
                _assert_valid_geojson(feature)

        elif geojson_obj["type"] in [
            "Point",
            "LineString",
            "Polygon",
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
            "GeometryCollection",
        ]:
            assert "coordinates" in geojson_obj or "geometries" in geojson_obj, (
                f"{geojson_obj['type']} must have 'coordinates' or 'geometries' field"
            )

        return True

    return _assert_valid_geojson


@pytest.fixture(scope="session")
def test_logger(test_log_dir):
    """Set up test logger."""
    log_file = test_log_dir / "test.log"
    setup_logging(log_level="DEBUG", json_format=True, log_file=str(log_file))
    return logging.getLogger("test")


#############################################################
# Reset Fixtures (Run Before Each Test)
#############################################################


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    yield
    logging.getLogger().handlers = []
    logging.getLogger().level = logging.NOTSET


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before each test."""
    try:
        from prometheus_client import REGISTRY

        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            REGISTRY.unregister(collector)
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def test_config(test_log_dir):
    """Create a default test configuration."""
    config = load_default_config()
    # Override common test settings
    config["general"]["debug_mode"] = True
    config["general"]["log_level"] = "DEBUG"
    if "logging" in config["general"]:
        config["general"]["log_file"] = str(test_log_dir / "test.log")

    return config
