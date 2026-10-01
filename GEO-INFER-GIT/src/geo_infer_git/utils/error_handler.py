#!/usr/bin/env python3

"""
Error handling utilities for GEO-INFER-GIT.

This module provides comprehensive error handling functionality including
retry mechanisms, error classification, and graceful failure recovery.
"""

import os
import time
import logging
import functools
from typing import Any, cast
from collections.abc import Callable
from enum import Enum
import requests
import git

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    NETWORK = "network"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    GIT_OPERATION = "git_operation"
    FILESYSTEM = "filesystem"
    CONFIGURATION = "configuration"
    API_LIMIT = "api_limit"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class GeoInferGitError(Exception):
    """
    Base exception class for GEO-INFER-GIT errors.

    Provides structured error information including category, severity,
    and recovery suggestions.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = False,
        suggestions: list[str] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """
        Initialize the error.

        Args:
            message: Error message
            category: Error category
            severity: Error severity level
            recoverable: Whether the error is recoverable
            suggestions: List of recovery suggestions
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.suggestions = suggestions or []
        self.original_error = original_error

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"[{self.category.value.upper()}] {super().__str__()}"]

        if self.severity != ErrorSeverity.MEDIUM:
            parts.append(f"Severity: {self.severity.value}")

        if not self.recoverable:
            parts.append("Non-recoverable")

        if self.suggestions:
            parts.append(f"Suggestions: {', '.join(self.suggestions)}")

        return " | ".join(parts)


class NetworkError(GeoInferGitError):
    """Network-related errors.

    Recoverable by default, matching :func:`classify_error`, so the network
    recovery strategies run for directly constructed errors too.
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class AuthenticationError(GeoInferGitError):
    """Authentication-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class PermissionDeniedError(GeoInferGitError):
    """Permission-related errors (the custom analogue of the builtin
    :class:`PermissionError`, which is deliberately not shadowed)."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class GitOperationError(GeoInferGitError):
    """Git operation errors.

    Recoverable by default, matching :func:`classify_error`, so the Git
    recovery strategies run for directly constructed errors too.
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        super().__init__(
            message,
            category=ErrorCategory.GIT_OPERATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class FilesystemError(GeoInferGitError):
    """Filesystem-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            category=ErrorCategory.FILESYSTEM,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class ConfigurationError(GeoInferGitError):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class APILimitError(GeoInferGitError):
    """API rate limit errors."""

    def __init__(
        self, message: str, reset_time: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(
            message,
            category=ErrorCategory.API_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            **kwargs,
        )
        self.reset_time = reset_time


class ValidationError(GeoInferGitError):
    """Validation-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def classify_error(error: Exception) -> tuple[ErrorCategory, ErrorSeverity, bool]:
    """
    Classify an exception into category, severity, and recoverability.

    Args:
        error: Exception to classify

    Returns:
        Tuple of (category, severity, recoverable)
    """

    # Network errors
    if isinstance(error, requests.RequestException):
        if isinstance(error, requests.ConnectionError):
            return ErrorCategory.NETWORK, ErrorSeverity.HIGH, True
        elif isinstance(error, requests.Timeout):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, True
        elif isinstance(error, requests.HTTPError):
            status_code = error.response.status_code if error.response else 0
            if status_code == 401:
                return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH, False
            elif status_code == 403:
                return ErrorCategory.PERMISSION, ErrorSeverity.HIGH, False
            elif status_code == 429:
                return ErrorCategory.API_LIMIT, ErrorSeverity.MEDIUM, True
            else:
                return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, True

    # Git errors
    elif isinstance(error, git.GitCommandError):
        return ErrorCategory.GIT_OPERATION, ErrorSeverity.MEDIUM, True

    # File system errors
    elif isinstance(error, (FileNotFoundError, PermissionError, OSError)):
        return ErrorCategory.FILESYSTEM, ErrorSeverity.HIGH, False

    # Configuration errors
    elif isinstance(error, (ValueError, TypeError, KeyError)):
        return ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, False

    # Default classification
    return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM, False


def retry_on_error(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_errors: tuple[type[Exception], ...] | None = None,
    logger_instance: logging.Logger | None = None,
) -> Callable:
    """
    Decorator for retrying functions on errors.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_errors: Tuple of exception types to retry on
        logger_instance: Logger instance for logging retry attempts

    Returns:
        Decorated function
    """
    default_retryable_errors = (
        requests.RequestException,
        git.GitCommandError,
        ConnectionError,
        TimeoutError,
    )

    if retryable_errors is None:
        retryable_errors = default_retryable_errors

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as error:
                    last_error = error

                    # Check if error is retryable
                    if not isinstance(error, retryable_errors):
                        if logger_instance:
                            logger_instance.error(
                                f"Non-retryable error in {func.__name__}: {error}"
                            )
                        raise

                    # Don't retry on last attempt
                    if attempt == max_attempts - 1:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    # Add jitter if enabled
                    if jitter:
                        import random

                        delay *= 0.5 + random.random()

                    if logger_instance:
                        logger_instance.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed in {func.__name__}: {error}. "
                            f"Retrying in {delay:.1f}s..."
                        )

                    time.sleep(delay)

            # All retries failed
            if logger_instance:
                logger_instance.error(
                    f"All {max_attempts} attempts failed in {func.__name__}"
                )

            raise cast(BaseException, last_error)

        return wrapper

    return decorator


def handle_error(
    error: Exception,
    operation: str | None = None,
    logger_instance: logging.Logger | None = None,
    reraise: bool = True,
) -> GeoInferGitError | None:
    """
    Handle and classify an error.

    Args:
        error: Exception to handle
        operation: Operation that failed
        logger_instance: Logger instance for logging
        reraise: Whether to re-raise the error

    Returns:
        Classified error if not reraising, None if reraised
    """
    category, severity, recoverable = classify_error(error)

    # Create suggestions based on error type
    suggestions = []

    if category == ErrorCategory.NETWORK:
        suggestions.extend(
            [
                "Check your internet connection",
                "Verify the target repository URL is accessible",
                "Check if firewalls or proxies are blocking the connection",
            ]
        )
    elif category == ErrorCategory.AUTHENTICATION:
        suggestions.extend(
            [
                "Verify your GitHub token is valid and has appropriate permissions",
                "Check that the token hasn't expired",
                "Ensure the token has 'repo' scope for private repositories",
            ]
        )
    elif category == ErrorCategory.PERMISSION:
        suggestions.extend(
            [
                "Check if you have access to the repository",
                "Verify the repository exists and is not private",
                "Ensure your token has the necessary permissions",
            ]
        )
    elif category == ErrorCategory.API_LIMIT:
        suggestions.extend(
            [
                "Wait for rate limit reset",
                "Use a GitHub token to increase rate limits",
                "Reduce the frequency of API calls",
            ]
        )
    elif category == ErrorCategory.GIT_OPERATION:
        suggestions.extend(
            [
                "Check if the repository exists",
                "Verify Git is properly installed",
                "Check disk space and permissions",
                "Ensure the repository is not corrupted",
            ]
        )
    elif category == ErrorCategory.FILESYSTEM:
        suggestions.extend(
            [
                "Check disk space availability",
                "Verify file permissions",
                "Ensure the directory path exists",
            ]
        )
    elif category == ErrorCategory.VALIDATION:
        suggestions.extend(
            [
                "Check configuration file format",
                "Verify all required fields are present",
                "Ensure data types are correct",
            ]
        )

    # Create structured error
    structured_error = GeoInferGitError(
        str(error),
        category=category,
        severity=severity,
        recoverable=recoverable,
        suggestions=suggestions,
        original_error=error,
    )

    # Log the error
    if logger_instance:
        cast(Any, logger_instance).log_error_with_context(error, operation)

    if reraise:
        raise structured_error from error

    return structured_error


class ErrorRecoveryManager:
    """
    Manager for handling error recovery strategies.
    """

    #: Attempts allowed before :meth:`_retry_with_backoff` gives up.
    MAX_RETRY_ATTEMPTS = 3
    #: First backoff delay; doubles per attempt.
    BASE_RETRY_DELAY_SECONDS = 1.0
    #: Ceiling applied to the exponential backoff delay.
    MAX_BACKOFF_SECONDS = 30.0
    #: Timeout applied to every recovery HTTP probe.
    REQUEST_TIMEOUT_SECONDS = 5
    #: An index.lock older than this is treated as abandoned.
    STALE_LOCK_SECONDS = 300.0

    def __init__(self, logger_instance: logging.Logger | None = None) -> None:
        """
        Initialize error recovery manager.

        Args:
            logger_instance: Logger instance for logging recovery actions
        """
        self.logger = logger_instance
        self.recovery_strategies = self._load_recovery_strategies()

    def _load_recovery_strategies(self) -> dict[ErrorCategory, list[Callable]]:
        """Load default recovery strategies for each error category."""
        strategies = {}

        strategies[ErrorCategory.NETWORK] = [
            self._retry_with_backoff,
            self._check_network_connectivity,
        ]

        strategies[ErrorCategory.AUTHENTICATION] = [
            self._check_token_validity,
            self._refresh_token,
        ]

        strategies[ErrorCategory.API_LIMIT] = [
            self._wait_for_rate_limit_reset,
            self._use_alternate_api_endpoint,
        ]

        strategies[ErrorCategory.GIT_OPERATION] = [
            self._retry_git_operation,
            self._clean_git_cache,
        ]

        strategies[ErrorCategory.FILESYSTEM] = [
            self._check_disk_space,
            self._verify_permissions,
        ]

        return strategies

    def attempt_recovery(
        self, error: GeoInferGitError, context: dict[str, Any] | None = None
    ) -> bool:
        """
        Attempt to recover from an error.

        Args:
            error: Error to recover from
            context: Additional context for recovery

        Returns:
            True if recovery was successful, False otherwise
        """
        context = context or {}

        if not error.recoverable:
            if self.logger:
                self.logger.warning(f"Error {error.category.value} is not recoverable")
            return False

        strategies = self.recovery_strategies.get(error.category, [])

        for strategy in strategies:
            try:
                if strategy(error, context):
                    if self.logger:
                        self.logger.info(
                            f"Successfully recovered using {strategy.__name__}"
                        )
                    return True
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"Recovery strategy {strategy.__name__} failed: {e}"
                    )

        return False

    def _retry_with_backoff(
        self, error: GeoInferGitError, context: dict[str, Any]
    ) -> bool:
        """Wait out a transient failure so the caller can retry.

        Sleeps for an exponentially growing delay derived from the attempt
        counter the caller supplies, capped by ``max_backoff_seconds``.
        Returns True when the caller should retry, and False once the
        attempt budget is spent so the error propagates instead of looping.

        Args:
            error: The classified error being recovered from.
            context: Recovery context; honours ``attempt``,
                ``max_retry_attempts``, ``base_delay``, and
                ``max_backoff_seconds``.

        Returns:
            True if a retry should follow, False if the budget is exhausted.
        """
        attempt = int(context.get("attempt", 1))
        max_attempts = int(context.get("max_retry_attempts", self.MAX_RETRY_ATTEMPTS))
        if attempt >= max_attempts:
            if self.logger:
                self.logger.warning(
                    f"Retry budget exhausted after {attempt} attempts for {error.category.value}"
                )
            return False

        base_delay = float(context.get("base_delay", self.BASE_RETRY_DELAY_SECONDS))
        max_delay = float(context.get("max_backoff_seconds", self.MAX_BACKOFF_SECONDS))
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
        if self.logger:
            self.logger.info(f"Backing off {delay:.2f}s before retry {attempt + 1}")
        time.sleep(delay)
        return True

    def _check_network_connectivity(
        self, error: GeoInferGitError, context: dict[str, Any]
    ) -> bool:
        """Check if network connectivity is available."""
        try:
            requests.get("https://api.github.com", timeout=5)
            return True
        except requests.RequestException:
            return False

    def _check_token_validity(
        self, error: GeoInferGitError, context: dict[str, Any]
    ) -> bool:
        """Check if authentication token is valid."""
        token = context.get("token")
        if not token:
            return False

        try:
            response = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {token}"},
                timeout=5,
            )
            return bool(response.status_code == 200)
        except requests.RequestException:
            return False

    def _refresh_token(self, error: GeoInferGitError, context: dict[str, Any]) -> bool:
        """Refresh the authentication token through the configured mechanism.

        Two mechanisms are supported, in order: a ``token_refresh`` callable
        supplied by the caller, or an OAuth refresh-token exchange against
        ``token_url``. A refreshed token is written back into ``context``
        under ``token`` and validated before success is reported. When
        neither mechanism is configured there is nothing to refresh and the
        strategy declines.

        Args:
            error: The classified authentication error.
            context: Recovery context; honours ``token_refresh``,
                ``refresh_token``, ``token_url``, and ``client_id``.

        Returns:
            True if a validated replacement token was obtained.
        """
        refresher = context.get("token_refresh")
        if callable(refresher):
            try:
                new_token = refresher()
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"Token refresh callable failed: {exc}")
                return False
            if not new_token:
                return False
            context["token"] = new_token
            return self._check_token_validity(error, context)

        refresh_token = context.get("refresh_token")
        token_url = context.get("token_url")
        if not refresh_token or not token_url:
            if self.logger:
                self.logger.info(
                    "No token refresh mechanism configured; cannot refresh"
                )
            return False

        payload = {"grant_type": "refresh_token", "refresh_token": refresh_token}
        if context.get("client_id"):
            payload["client_id"] = context["client_id"]

        try:
            response = requests.post(
                token_url,
                data=payload,
                headers={"Accept": "application/json"},
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            if self.logger:
                self.logger.warning(f"Token refresh request failed: {exc}")
            return False

        if response.status_code != 200:
            return False
        try:
            new_token = response.json().get("access_token")
        except ValueError:
            return False
        if not new_token:
            return False

        context["token"] = new_token
        return self._check_token_validity(error, context)

    def _wait_for_rate_limit_reset(
        self, error: GeoInferGitError, context: dict[str, Any]
    ) -> bool:
        """Wait for API rate limit reset."""
        if hasattr(error, "reset_time") and error.reset_time:
            wait_time = max(0, error.reset_time - int(time.time()))
            if self.logger:
                self.logger.info(f"Waiting {wait_time} seconds for rate limit reset")
            time.sleep(wait_time)
            return True
        return False

    def _use_alternate_api_endpoint(
        self, error: GeoInferGitError, context: dict[str, Any]
    ) -> bool:
        """Switch to the first alternate API endpoint that answers healthily.

        Probes each candidate from ``alternate_api_endpoints`` and records
        the first one that responds without a rate-limit or server status
        into ``context['api_base_url']``, so the retried operation targets
        it. Endpoints equal to the failing ``api_base_url`` are skipped.

        Args:
            error: The classified API-limit error.
            context: Recovery context; honours ``alternate_api_endpoints``
                and ``api_base_url``.

        Returns:
            True if a healthy alternate endpoint was selected.
        """
        candidates = context.get("alternate_api_endpoints") or []
        current = context.get("api_base_url")
        headers = {"Accept": "application/vnd.github+json"}
        if context.get("token"):
            headers["Authorization"] = f"token {context['token']}"

        for endpoint in candidates:
            if not endpoint or endpoint == current:
                continue
            try:
                response = requests.get(
                    endpoint,
                    headers=headers,
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                if self.logger:
                    self.logger.debug(
                        f"Alternate endpoint {endpoint} unreachable: {exc}"
                    )
                continue

            # 403/429 mean the alternate is rate limited too; 5xx means unhealthy.
            if response.status_code in (403, 429) or response.status_code >= 500:
                continue

            context["api_base_url"] = endpoint
            if self.logger:
                self.logger.info(f"Switched to alternate API endpoint {endpoint}")
            return True

        if self.logger:
            self.logger.info("No healthy alternate API endpoint available")
        return False

    def _retry_git_operation(
        self, error: GeoInferGitError, context: dict[str, Any]
    ) -> bool:
        """Confirm the working repository is usable, then allow a retry.

        A Git operation is only worth retrying if the repository is still a
        valid Git directory; retrying against a corrupted or partially
        cloned tree just repeats the failure. When ``path`` names a valid
        repository the strategy backs off and reports that a retry should
        follow.

        Args:
            error: The classified Git-operation error.
            context: Recovery context; honours ``path`` and the keys read
                by :meth:`_retry_with_backoff`.

        Returns:
            True if the repository is usable and a retry should follow.
        """
        path = context.get("path")
        if path:
            try:
                git.Repo(path)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"Repository at {path} is not usable: {exc}")
                return False

        return self._retry_with_backoff(error, context)

    def _clean_git_cache(
        self, error: GeoInferGitError, context: dict[str, Any]
    ) -> bool:
        """Clear a stale index lock and repack loose objects.

        The two recoverable causes of a repeated Git failure this handles
        are an ``index.lock`` left behind by a killed process and an
        overgrown loose-object store. A lock file is only removed once it
        is older than ``stale_lock_seconds``, so a lock held by a live
        concurrent operation is never taken away from it.

        Args:
            error: The classified Git-operation error.
            context: Recovery context; requires ``path`` and honours
                ``stale_lock_seconds``.

        Returns:
            True if the repository was cleaned and a retry may succeed.
        """
        path = context.get("path")
        if not path:
            return False

        try:
            repo = git.Repo(path)
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    f"Cannot clean cache; {path} is not a repository: {exc}"
                )
            return False

        cleaned = False
        stale_after = float(context.get("stale_lock_seconds", self.STALE_LOCK_SECONDS))
        lock_path = os.path.join(repo.git_dir, "index.lock")
        if os.path.exists(lock_path):
            age = time.time() - os.path.getmtime(lock_path)
            if age < stale_after:
                if self.logger:
                    self.logger.info(
                        f"index.lock is {age:.0f}s old; a concurrent operation may hold it"
                    )
                return False
            try:
                os.remove(lock_path)
                cleaned = True
                if self.logger:
                    self.logger.info(f"Removed stale index.lock ({age:.0f}s old)")
            except OSError as exc:
                if self.logger:
                    self.logger.warning(f"Could not remove stale index.lock: {exc}")
                return False

        try:
            repo.git.gc("--prune=now", "--quiet")
            cleaned = True
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"git gc failed: {exc}")

        return cleaned

    def _check_disk_space(
        self, error: GeoInferGitError, context: dict[str, Any]
    ) -> bool:
        """Check available disk space."""
        try:
            import shutil

            total, used, free = shutil.disk_usage(context.get("path", "."))
            return free > 1024 * 1024 * 100  # At least 100MB free
        except Exception:
            return False

    def _verify_permissions(
        self, error: GeoInferGitError, context: dict[str, Any]
    ) -> bool:
        """Verify file/directory permissions."""
        path = context.get("path")
        if not path:
            return False

        try:
            # Try to create a test file
            test_file = os.path.join(path, ".test_permissions")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception:
            return False


def with_error_handling(
    operation: str | None = None,
    logger_instance: logging.Logger | None = None,
    max_retries: int = 3,
) -> Callable:
    """
    Decorator for comprehensive error handling.

    Args:
        operation: Name of operation for logging
        logger_instance: Logger instance for logging errors
        max_retries: Maximum number of retry attempts

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            recovery_manager = ErrorRecoveryManager(logger_instance)

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)

                except Exception as error:
                    # Handle and classify the error
                    structured_error = handle_error(
                        error,
                        operation or func.__name__,
                        logger_instance,
                        reraise=False,
                    )
                    assert structured_error is not None

                    # Attempt recovery if possible
                    if structured_error.recoverable and attempt < max_retries - 1:
                        context = {
                            "function": func.__name__,
                            "operation": func,
                            "attempt": attempt + 1,
                            "max_retry_attempts": max_retries,
                            "args": args,
                            "kwargs": kwargs,
                        }
                        # Surface a repository path so Git strategies can act on it.
                        if "path" in kwargs:
                            context["path"] = kwargs["path"]

                        if recovery_manager.attempt_recovery(structured_error, context):
                            continue

                    # No more retries or non-recoverable error
                    if logger_instance:
                        logger_instance.error(
                            f"Final failure in {func.__name__}: {structured_error}"
                        )

                    raise structured_error

            return None  # Should not reach here

        return wrapper

    return decorator
