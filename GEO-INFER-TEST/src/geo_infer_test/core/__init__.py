"""Core functionality for the GEO-INFER-TEST module.

This submodule contains the essential components for comprehensive testing
of the GEO-INFER ecosystem, including integration with logging and reporting."""

from importlib import import_module as _import_module

_EXPORTS = {
    "TestConfiguration": (".test_runner", "TestConfiguration"),
    "TestResult": (".test_runner", "TestResult"),
    "GeoInferTestRunner": (".test_runner", "GeoInferTestRunner"),
    "TestDiscoverer": (".test_discoverer", "TestDiscoverer"),
    "TestOrchestrator": (".test_orchestrator", "TestOrchestrator"),
    "TestSuiteManager": (".test_orchestrator", "TestSuiteManager"),
    "LogIntegration": (".log_integration", "LogIntegration"),
    "LoggingTestReporter": (".log_integration", "LoggingTestReporter"),
    "TestLogger": (".log_integration", "TestLogger"),
    "LogAnalyzer": (".log_integration", "LogAnalyzer"),
    "ModuleHealthChecker": (".module_health", "ModuleHealthChecker"),
    "HealthMetrics": (".module_health", "HealthMetrics"),
    "SystemValidator": (".module_health", "SystemValidator"),
    "DependencyChecker": (".module_health", "DependencyChecker"),
    "PerformanceMonitor": (".performance_monitor", "PerformanceMonitor"),
    "BenchmarkRunner": (".performance_monitor", "BenchmarkRunner"),
    "LoadTester": (".performance_monitor", "LoadTester"),
    "MetricsCollector": (".performance_monitor", "MetricsCollector"),
    "PerformanceAnalyzer": (".performance_monitor", "PerformanceAnalyzer"),
}

__all__ = [
    # Test execution framework
    "TestConfiguration",
    "TestResult",
    "GeoInferTestRunner",
    "TestDiscoverer",
    "TestOrchestrator",
    "TestSuiteManager",
    # Logging and monitoring integration
    "LogIntegration",
    "LoggingTestReporter",
    "TestLogger",
    "LogAnalyzer",
    # Health monitoring
    "ModuleHealthChecker",
    "HealthMetrics",
    "SystemValidator",
    "DependencyChecker",
    # Performance monitoring
    "PerformanceMonitor",
    "BenchmarkRunner",
    "LoadTester",
    "MetricsCollector",
    "PerformanceAnalyzer",
]


def __getattr__(name: str):
    if name in {
        "test_runner",
        "test_discoverer",
        "test_orchestrator",
        "log_integration",
        "module_health",
        "performance_monitor",
    }:
        value = _import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, member = _EXPORTS[name]
    value = getattr(_import_module(module, __name__), member)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
