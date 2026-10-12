"""
Logging utilities for GEO-INFER-HEALTH module.

Provides centralized logging configuration and utilities.
"""

import sys
from pathlib import Path
from typing import Any
from collections.abc import Callable

from loguru import logger

from .config import get_global_config


def setup_logging(
    level: str = "INFO",
    format: str | None = None,
    file_path: str | None = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    verbose: bool = False,
) -> None:
    """
    Setup logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format string
        file_path: Path to log file (if None, uses config or default)
        max_bytes: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        verbose: Enable verbose logging
    """
    # Remove default handler
    logger.remove()

    # Load configuration
    try:
        config = get_global_config()
        config_logging = config.logging
        config_file = config_logging.get("file", {})
    except Exception:
        config_logging = {}
        config_file = {}

    # Determine logging level
    if not level:
        level = config_logging.get("level", "INFO")

    # Determine log format
    if not format:
        format = config_logging.get(
            "format",
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>",
        )

    # Determine file path. Imports should never create root-level log files;
    # file logging is opt-in through explicit configuration or the CLI.
    if not file_path and config_file.get("enabled", False):
        file_path = config_file.get("path", "logs/health.log")

    # Create logs directory if needed
    if file_path:
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure file logging
        max_bytes = config_file.get("max_bytes", max_bytes)
        backup_count = config_file.get("backup_count", backup_count)

        logger.add(
            file_path,
            format=format,
            level=level,
            rotation=max_bytes,
            retention=backup_count,
            encoding="utf-8",
        )

    # Configure console logging
    if verbose or level == "DEBUG":
        # Verbose format with more details
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    else:
        console_format = format

    logger.add(sys.stdout, format=console_format, level=level, colorize=True)

    # Log setup completion
    logger.info(f"Logging configured with level: {level}")
    if file_path:
        logger.info(f"Log file: {file_path}")


def get_logger(name: str = "geo_infer_health") -> Any:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(self, operation_name: str, log_threshold: float = 1.0):
        """
        Initialize performance logger.

        Args:
            operation_name: Name of the operation being timed
            log_threshold: Minimum duration in seconds to log (default: 1.0)
        """
        self.operation_name = operation_name
        self.log_threshold = log_threshold
        self.start_time: float | None = None

    def __enter__(self) -> "PerformanceLogger":
        import time

        self.start_time = time.time()
        logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        import time

        if self.start_time is None:
            return

        duration = time.time() - self.start_time

        if duration >= self.log_threshold:
            if exc_type:
                logger.warning(
                    f"Operation '{self.operation_name}' failed after {duration:.2f}s: {exc_val}"
                )
            else:
                logger.info(
                    f"Operation '{self.operation_name}' completed in {duration:.2f}s"
                )
        else:
            logger.debug(
                f"Operation '{self.operation_name}' completed in {duration:.2f}s"
            )


def log_function_call(
    func_name: str | None = None, log_args: bool = False, log_result: bool = False
) -> Callable[..., Any]:
    """
    Decorator to log function calls.

    Args:
        func_name: Override function name in logs
        log_args: Whether to log function arguments
        log_result: Whether to log function return value
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = func_name or f"{func.__module__}.{func.__name__}"

            # Log function call
            if log_args:
                logger.debug(f"Calling {name} with args={args}, kwargs={kwargs}")
            else:
                logger.debug(f"Calling {name}")

            try:
                with PerformanceLogger(name):
                    result = func(*args, **kwargs)

                if log_result:
                    logger.debug(f"{name} returned: {result}")

                return result

            except Exception as e:
                logger.error(f"Error in {name}: {e}")
                raise

        return wrapper

    return decorator


def log_performance(
    operation_name: str, duration: float, metadata: dict[str, Any] | None = None
) -> None:
    """
    Log performance metrics.

    Args:
        operation_name: Name of the operation
        duration: Duration in seconds
        metadata: Additional metadata to log
    """
    metadata_str = ""
    if metadata:
        metadata_str = f" | {metadata}"

    logger.info(f"Performance: {operation_name} took {duration:.3f}s{metadata_str}")


def create_log_context(context_info: dict[str, Any]) -> Any:
    """
    Create a logging context with additional information.

    Args:
        context_info: Dictionary of context information

    Returns:
        Logger instance with context
    """
    return logger.bind(**context_info)


def setup_structured_logging(
    service_name: str = "geo-infer-health",
    version: str = "1.0.0",
    environment: str = "development",
    file_path: str | None = None,
    level: str = "INFO",
) -> None:
    """
    Setup JSON structured logging for production use.

    Replaces loguru's default handler with a serialized (JSON) sink on
    stdout and attaches service metadata to every record via
    ``logger.bind``. File logging is opt-in: pass ``file_path`` to also
    write rotating JSON logs to disk (the parent directory is created).

    Args:
        service_name: Name of the service
        version: Service version
        environment: Deployment environment
        file_path: Optional path for a rotating JSON log file
        level: Minimum console log level
    """
    context = {
        "service": service_name,
        "version": version,
        "environment": environment,
    }
    logger.configure(extra=context)
    logger.remove()
    logger.add(sys.stdout, level=level, serialize=True)

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            file_path,
            level="DEBUG",
            rotation="1 day",
            retention="30 days",
            encoding="utf-8",
            serialize=True,
        )
