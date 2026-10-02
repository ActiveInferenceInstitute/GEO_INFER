"""
GEO-INFER-TEST: Comprehensive Testing and Quality Assurance Module

This module provides testing, validation, and quality assurance capabilities
for the GEO-INFER framework. It supports unit testing, integration testing,
performance testing, and data quality validation.

Key Features:
- Automated test suite execution
- Data quality validation and monitoring
- Performance benchmarking and regression testing
- Integration testing across modules
- Spatial data validation
- IoT sensor data quality control
- Bayesian inference validation
"""

from importlib import import_module

# Process and execution helpers remain usable in minimal build/probe lanes.
_EXPORT_MODULES = {
    "TestOutcome": "models.types",
    "ValidationRule": "models.types",
    "BaseValidator": "core.validators",
    "DataQualityValidator": "core.validators",
    "PerformanceValidator": "core.validators",
    "SpatialValidator": "core.validators",
    "IoTValidator": "core.validators",
    "BayesianValidator": "core.validators",
    "QualityController": "core.validators",
    "run_full_system_test": "core.validators",
    "GeoInferTestRunner": "core.test_runner",
    "TestConfiguration": "core.test_runner",
    "LocalService": "testing",
    "as_finite_array": "testing",
    "assert_finite": "testing",
    "assert_model_contract": "testing",
    "assert_no_nan_statistics": "testing",
    "assert_probability": "testing",
    "assert_same_finite_values": "testing",
    "assert_seed_replay": "testing",
    "assert_stochastic_matrix": "testing",
    "assert_visualization_manifest": "testing",
}


def __getattr__(name: str):
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))


__version__ = "0.4.0"

__all__ = [
    "GeoInferTestRunner",
    "TestConfiguration",
    "TestOutcome",
    "ValidationRule",
    "BaseValidator",
    "DataQualityValidator",
    "PerformanceValidator",
    "SpatialValidator",
    "IoTValidator",
    "BayesianValidator",
    "QualityController",
    "run_full_system_test",
    "LocalService",
    "as_finite_array",
    "assert_finite",
    "assert_model_contract",
    "assert_no_nan_statistics",
    "assert_probability",
    "assert_same_finite_values",
    "assert_seed_replay",
    "assert_stochastic_matrix",
    "assert_visualization_manifest",
]
