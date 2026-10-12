"""
Configuration Management for GEO-INFER-MATH

This module provides configuration management for various
mathematical operations and backends.
"""

from typing import Any
import logging
import os

logger = logging.getLogger(__name__)


class MathConfig:
    """
    Configuration manager for GEO-INFER-MATH.

    Provides centralized configuration for:
    - Theorem proving backends
    - Information theory parameters
    - Performance settings
    - Numerical precision
    """

    def __init__(self) -> None:
        """Initialize configuration with defaults."""
        self._config: dict[str, dict[str, Any]] = {
            "theorem_proving": {
                "backend": "z3",
                "timeout": 10.0,
                "enable_caching": True,
            },
            "information_theory": {
                "base": 2.0,
                "epsilon": 1e-10,
                "default_bins": 20,
            },
            "performance": {
                "enable_caching": True,
                "cache_size": 256,
                "parallel_processing": False,
                "num_workers": 4,
            },
            "numerical": {
                "precision": "float64",
                "epsilon": 1e-10,
                "max_iterations": 1000,
            },
            "symbolic_math": {
                "backend": "sympy",
                "enable_proof_generation": True,
            },
        }

        # Load from environment variables
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Theorem proving
        tp_backend = os.getenv("GEO_INFER_MATH_TP_BACKEND")
        if tp_backend is not None:
            self._config["theorem_proving"]["backend"] = tp_backend

        tp_timeout = os.getenv("GEO_INFER_MATH_TP_TIMEOUT")
        if tp_timeout is not None:
            self._config["theorem_proving"]["timeout"] = float(tp_timeout)

        # Performance
        caching = os.getenv("GEO_INFER_MATH_ENABLE_CACHING")
        if caching is not None:
            self._config["performance"]["enable_caching"] = caching.lower() == "true"

        parallel = os.getenv("GEO_INFER_MATH_PARALLEL")
        if parallel is not None:
            self._config["performance"]["parallel_processing"] = (
                parallel.lower() == "true"
            )

    def get(self, section: str, key: str | None = None) -> Any:
        """
        Get configuration value.

        Args:
            section: Configuration section
            key: Optional key within section

        Returns:
            Configuration value
        """
        if key is None:
            return self._config.get(section, {})
        return self._config.get(section, {}).get(key)

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            section: Configuration section
            key: Key within section
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
        logger.debug(f"Set config: {section}.{key} = {value}")

    def update(self, section: str, values: dict[str, Any]) -> None:
        """
        Update configuration section.

        Args:
            section: Configuration section
            values: Dictionary of values to update
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section].update(values)

    def to_dict(self) -> dict[str, Any]:
        """
        Get full configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()


# Global configuration instance
_config = MathConfig()


def get_config() -> MathConfig:
    """
    Get global configuration instance.

    Returns:
        Global MathConfig instance
    """
    return _config


def configure(**kwargs: Any) -> MathConfig:
    """
    Configure GEO-INFER-MATH settings.

    Keyword arguments accept two forms:

    - Dotted keys ``"<section>.<option>"``:
      ``configure(**{"theorem_proving.backend": "z3"})``.
    - Dict-of-sections values:
      ``configure(**{"theorem_proving": {"backend": "z3"}})``.

    Keys in any other form, or naming an unknown section, raise
    ``ValueError`` so misconfiguration is surfaced instead of silently
    landing in the wrong bucket. Declared sections: ``theorem_proving``,
    ``information_theory``, ``performance``, ``numerical``,
    ``symbolic_math``.

    Args:
        **kwargs: Configuration options

    Returns:
        Configuration instance

    Example:
        >>> configure(**{"theorem_proving.backend": "z3"})
    """
    known_sections = _config.to_dict()
    for key, value in kwargs.items():
        if "." not in key:
            if not isinstance(value, dict):
                raise ValueError(
                    f"Invalid configuration key {key!r}: expected "
                    "'<section>.<option>' dotted form or a dict-of-sections "
                    "value"
                )
            if key not in known_sections:
                raise ValueError(f"Unknown configuration section: {key!r}")
            _config.update(key, value)
            continue
        section, _, option = key.partition(".")
        if section not in known_sections:
            raise ValueError(f"Unknown configuration section: {section!r}")
        _config.set(section, option, value)

    return _config
