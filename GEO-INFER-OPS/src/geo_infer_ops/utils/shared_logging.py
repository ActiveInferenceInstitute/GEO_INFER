"""
Shared logging configuration for GEO-INFER modules.

This module provides standardized structured logging using structlog that can be
used across all GEO-INFER modules for consistent logging behavior.

``configure_logging`` is the SINGLE documented app-level logging entry for
GEO-INFER-OPS: CLI entrypoints call it once at startup. Library modules must
stay passive — ``logging.getLogger(__name__)`` / ``structlog.get_logger``
only — and never mutate the root logger at import time.
"""

import sys
import logging
import structlog
from typing import Any, cast
from pathlib import Path


# Define log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Global configuration state
_logging_configured = False


def configure_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
    module_name: str | None = None,
    enable_console: bool = True,
) -> None:
    """
    Configure the logging system for GEO-INFER modules.

    This function should be called once at application startup to configure
    structured logging using structlog. It can be called multiple times safely.

    Args:
        log_level: The minimum log level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to output logs as JSON (useful for production)
        log_file: Path to a file for writing logs (None means stdout only)
        module_name: Name of the module for context (optional)
        enable_console: Whether to enable console output
    """
    global _logging_configured

    level = LOG_LEVELS.get(log_level.upper(), logging.INFO)

    # Configure structlog processors
    processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),  # ISO 8601 format
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add module name if provided
    if module_name:
        processors.insert(2, structlog.processors.add_log_level)

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging handlers
    handlers: list[logging.Handler] = []

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        handlers.append(console_handler)

    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if _logging_configured:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    # Set format for standard logging (structlog handles its own formatting)
    formatter = logging.Formatter("%(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)

    _logging_configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    This is the recommended way to get a logger in GEO-INFER modules.
    It returns a structlog BoundLogger that supports structured logging.

    Args:
        name: Name of the logger, typically __name__ or module name.
              If None, uses the calling module's name.

    Returns:
        A structured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data", record_count=100, status="success")
    """
    if name is None:
        # Try to get the calling module's name
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "geo_infer")
        else:
            name = "geo_infer"

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


class LoggingContext:
    """
    Context manager for temporarily adding context to log entries.

    This allows you to add contextual information that will be included
    in all log messages within the context.

    Example:
        >>> with LoggingContext(request_id="123", user="admin"):
        ...     logger.info("Processing request")
        # All logs within this block will include request_id and user
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize logging context.

        Args:
            **kwargs: Key-value pairs to add to log context
        """
        self.temp_context = kwargs
        self.old_context: dict[str, Any] = {}

    def __enter__(self) -> "LoggingContext":
        """Enter context and bind context variables."""
        for key, value in self.temp_context.items():
            # Store old value if it exists
            try:
                context_vars = structlog.contextvars.get_contextvars()
                self.old_context[key] = context_vars.get(key)
            except Exception:
                self.old_context[key] = None

            # Bind new value
            structlog.contextvars.bind_contextvars(**{key: value})
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore previous context variables."""
        for key, old_value in self.old_context.items():
            if old_value is None:
                structlog.contextvars.unbind_contextvars(key)
            else:
                structlog.contextvars.bind_contextvars(**{key: old_value})


def setup_module_logging(
    module_name: str,
    log_level: str | None = None,
    log_file: str | None = None,
) -> structlog.stdlib.BoundLogger:
    """
    Convenience function to set up logging for a GEO-INFER module.

    This function configures logging and returns a logger instance
    ready to use. It's a one-stop function for module initialization.

    Args:
        module_name: Name of the module (e.g., "geo_infer_space")
        log_level: Log level (defaults to INFO if not configured)
        log_file: Optional log file path

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_module_logging("geo_infer_space", log_level="DEBUG")
        >>> logger.info("Module initialized")
    """
    # Configure if not already configured
    if not _logging_configured:
        configure_logging(
            log_level=log_level or "INFO",
            json_format=False,
            log_file=log_file,
            module_name=module_name,
        )

    return get_logger(module_name)


__all__ = [
    "configure_logging",
    "get_logger",
    "LoggingContext",
    "setup_module_logging",
    "LOG_LEVELS",
]
