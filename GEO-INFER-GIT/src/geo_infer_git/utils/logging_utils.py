#!/usr/bin/env python3

"""
Logging utilities for GEO-INFER-GIT.

This module provides comprehensive logging functionality including
structured logging, log formatting, and log management.
"""

import sys
import logging
import logging.handlers
from typing import Any, cast
from collections.abc import Callable
from pathlib import Path
import json
import time


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
                "message",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Enhanced text formatter for human-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable text."""
        # Add color codes for different log levels
        color_codes = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }

        color = color_codes.get(record.levelname, "")
        reset = "\033[0m" if color else ""

        # Format timestamp
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        # Format message with color
        if color:
            formatted_message = (
                f"{color}{record.levelname}{reset}: {record.getMessage()}"
            )
        else:
            formatted_message = f"{record.levelname}: {record.getMessage()}"

        # Add context if available
        if hasattr(record, "repo_name"):
            formatted_message += f" [{record.repo_name}]"

        return f"{timestamp} - {formatted_message}"


class GeoInferGitLogger:
    """
    Enhanced logger for GEO-INFER-GIT with structured logging support.

    Provides:
    - Multiple output formats (JSON, text)
    - File and console logging
    - Log rotation and management
    - Context-aware logging
    - Performance metrics logging
    """

    def __init__(
        self, name: str = "geo_infer_git", config: dict[str, Any] | None = None
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            config: Logging configuration dictionary
        """
        self.logger = logging.getLogger(name)
        self.config = config or {}

        # Prevent duplicate handlers
        if self.logger.handlers:
            return

        self.logger.setLevel(getattr(logging, self.config.get("level", "INFO").upper()))

        # Setup formatters
        self.json_formatter = JSONFormatter()
        self.text_formatter = TextFormatter()

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up logging handlers based on configuration."""
        format_type = self.config.get("format", "text")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if format_type == "json":
            console_handler.setFormatter(self.json_formatter)
        else:
            console_handler.setFormatter(self.text_formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        log_file = self.config.get("file")
        if log_file:
            # Create log directory if needed
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # File handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )

            if format_type == "json":
                file_handler.setFormatter(self.json_formatter)
            else:
                file_handler.setFormatter(self.text_formatter)

            self.logger.addHandler(file_handler)

    def log_repo_operation(self, operation: str, repo_name: str, **kwargs: Any) -> None:
        """
        Log a repository operation with context.

        Args:
            operation: Operation being performed
            repo_name: Repository name
            **kwargs: Additional context data
        """
        extra_data = {"operation": operation, "repo_name": repo_name}
        extra_data.update(kwargs)

        self.logger.info(f"Repository operation: {operation}", extra=extra_data)

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log an API call with details.

        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            **kwargs: Additional context data
        """
        extra_data = {
            "api_endpoint": endpoint,
            "http_method": method,
            "http_status": status_code,
        }
        extra_data.update(kwargs)

        if status_code and status_code >= 400:
            self.logger.error(f"API call failed: {method} {endpoint}", extra=extra_data)
        else:
            self.logger.info(f"API call: {method} {endpoint}", extra=extra_data)

    def log_performance(self, operation: str, duration: float, **kwargs: Any) -> None:
        """
        Log performance metrics.

        Args:
            operation: Operation that was timed
            duration: Duration in seconds
            **kwargs: Additional performance data
        """
        extra_data = {
            "operation": operation,
            "duration_seconds": duration,
            "performance_metric": True,
        }
        extra_data.update(kwargs)

        self.logger.info(
            f"Performance: {operation} completed in {duration:.2f}s", extra=extra_data
        )

    def log_error_with_context(
        self, error: Exception, operation: str | None = None, **kwargs: Any
    ) -> None:
        """
        Log an error with additional context.

        Args:
            error: Exception that occurred
            operation: Operation that failed
            **kwargs: Additional context data
        """
        extra_data = {"error_type": type(error).__name__}
        if operation:
            extra_data["failed_operation"] = operation
        extra_data.update(kwargs)

        self.logger.error(
            f"Error occurred: {str(error)}", extra=extra_data, exc_info=True
        )


def setup_logging(config: dict[str, Any] | None = None) -> GeoInferGitLogger:
    """
    Set up logging for GEO-INFER-GIT.

    Args:
        config: Logging configuration

    Returns:
        Configured logger instance
    """
    return GeoInferGitLogger("geo_infer_git", config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"geo_infer_git.{name}")


class LogContext:
    """
    Context manager for adding temporary context to log records.
    """

    def __init__(self, logger: logging.Logger, **context: Any) -> None:
        """
        Initialize log context.

        Args:
            logger: Logger instance
            **context: Context key-value pairs
        """
        self._logger: Any = logger
        self.context = context
        self.old_factory: Any = None

    def __enter__(self) -> "LogContext":
        """Enter context and set up custom log record factory."""
        old_factory = self._logger.makeRecord

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return cast(logging.LogRecord, record)

        self._logger.makeRecord = cast(Any, record_factory)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore original factory."""
        self._logger.makeRecord = self.old_factory


def log_with_context(
    logger: logging.Logger, level: int, message: str, **context: Any
) -> None:
    """
    Log a message with additional context.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **context: Additional context data
    """
    if context:
        # Create a log record with context
        record = logger.makeRecord(logger.name, level, __file__, 0, message, (), None)
        for key, value in context.items():
            setattr(record, key, value)
        logger.handle(record)
    else:
        logger.log(level, message)


class PerformanceTimer:
    """
    Timer for measuring operation performance.
    """

    def __init__(self, operation: str, logger: logging.Logger | None = None):
        """
        Initialize performance timer.

        Args:
            operation: Name of operation being timed
            logger: Logger instance for reporting
        """
        self.operation = operation
        self.logger = logger
        self.start_time: float | None = None

    def __enter__(self) -> "PerformanceTimer":
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing and log result."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            if self.logger:
                cast(Any, self.logger).log_performance(self.operation, duration)


def time_operation(operation: str, logger: logging.Logger | None = None) -> Callable:
    """
    Decorator for timing function execution.

    Args:
        operation: Name of operation being timed
        logger: Logger instance for reporting

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            timer_logger: Any = logger or get_logger(func.__module__)
            with PerformanceTimer(operation, timer_logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator
