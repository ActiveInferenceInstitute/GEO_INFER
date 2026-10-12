"""
Health check system for GEO-INFER-OPS.

This module provides comprehensive health monitoring for GEO-INFER
modules, services, and dependencies.
"""

import logging
import asyncio
import time
from typing import Any
from collections.abc import Callable
from datetime import datetime, UTC
from dataclasses import dataclass, field
from enum import StrEnum

import psutil

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Represents a health check result."""

    name: str
    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert health check to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


class HealthChecker:
    """
    Health checker for GEO-INFER modules and services.

    Provides comprehensive health monitoring including system resources,
    service availability, dependency checks, and custom health checks.
    """

    def __init__(
        self,
        check_interval_seconds: int = 60,
        timeout_seconds: int = 5,
        enable_system_checks: bool = True,
    ) -> None:
        """
        Initialize the health checker.

        Args:
            check_interval_seconds: Interval between health checks
            timeout_seconds: Timeout for individual health checks
            enable_system_checks: Whether to enable system resource checks
        """
        self.check_interval_seconds = check_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.enable_system_checks = enable_system_checks

        self.custom_checks: dict[str, dict[str, Any]] = {}
        self.health_history: list[dict[str, Any]] = []

    def register_check(
        self, name: str, check_func: Callable, async_check: bool = False
    ) -> None:
        """
        Register a custom health check.

        Args:
            name: Name of the health check
            check_func: Function that returns health status
            async_check: Whether the check function is async
        """
        self.custom_checks[name] = {
            "func": check_func,
            "async": async_check,
        }
        logger.info(f"Registered health check: {name}")

    async def check_system_resources(self) -> HealthCheck:
        """
        Check system resource availability.

        Returns:
            HealthCheck for system resources
        """
        start_time = time.time()

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            issues = []

            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")

            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated memory usage: {memory.percent:.1f}%")

            if disk.percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Low disk space: {disk.percent:.1f}% used")
            elif disk.percent > 80:
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated disk usage: {disk.percent:.1f}%")

            message = "; ".join(issues) if issues else "System resources healthy"

            duration_ms = (time.time() - start_time) * 1000

            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                },
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"System resource check failed: {e}")
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"System check error: {str(e)}",
                duration_ms=duration_ms,
            )

    async def check_service(
        self, name: str, check_func: Callable | None = None
    ) -> HealthCheck:
        """
        Check health of a service.

        Args:
            name: Service name
            check_func: Optional custom check function

        Returns:
            HealthCheck for the service
        """
        start_time = time.time()

        try:
            if check_func:
                if asyncio.iscoroutinefunction(check_func):
                    result = await asyncio.wait_for(
                        check_func(), timeout=self.timeout_seconds
                    )
                else:
                    result = check_func()
            else:
                # Default: assume healthy if no check function
                result = True

            duration_ms = (time.time() - start_time) * 1000

            if result is True or result is None:
                return HealthCheck(
                    name=f"service_{name}",
                    status=HealthStatus.HEALTHY,
                    message=f"Service {name} is healthy",
                    duration_ms=duration_ms,
                )
            elif isinstance(result, dict):
                status = result.get("status", HealthStatus.HEALTHY)
                message = result.get("message", f"Service {name} check completed")
                details = result.get("details", {})
                return HealthCheck(
                    name=f"service_{name}",
                    status=status,
                    message=message,
                    details=details,
                    duration_ms=duration_ms,
                )
            else:
                return HealthCheck(
                    name=f"service_{name}",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Service {name} check returned unexpected result",
                    duration_ms=duration_ms,
                )

        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name=f"service_{name}",
                status=HealthStatus.UNHEALTHY,
                message=f"Service {name} health check timed out",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Service {name} health check failed: {e}")
            return HealthCheck(
                name=f"service_{name}",
                status=HealthStatus.UNHEALTHY,
                message=f"Service {name} check error: {str(e)}",
                duration_ms=duration_ms,
            )

    async def run_all_checks(self) -> dict[str, Any]:
        """
        Run all registered health checks.

        Returns:
            Dictionary containing all health check results
        """
        logger.info("Running health checks")

        checks: list[HealthCheck] = []

        # System resource checks
        if self.enable_system_checks:
            system_check = await self.check_system_resources()
            checks.append(system_check)

        # Custom checks
        for name, check_info in self.custom_checks.items():
            check_func = check_info["func"]
            is_async = check_info["async"]

            try:
                if is_async or asyncio.iscoroutinefunction(check_func):
                    result = await asyncio.wait_for(
                        check_func(), timeout=self.timeout_seconds
                    )
                else:
                    result = check_func()

                if isinstance(result, HealthCheck):
                    checks.append(result)
                elif isinstance(result, dict):
                    checks.append(
                        HealthCheck(
                            name=name,
                            status=result.get("status", HealthStatus.HEALTHY),
                            message=result.get("message", ""),
                            details=result.get("details", {}),
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            name=name,
                            status=(
                                HealthStatus.HEALTHY
                                if result
                                else HealthStatus.UNHEALTHY
                            ),
                            message=f"Check {name} completed",
                        )
                    )
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                checks.append(
                    HealthCheck(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check error: {str(e)}",
                    )
                )

        # Determine overall status
        statuses = [check.status for check in checks]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        results: dict[str, Any] = {
            "status": overall_status.value,
            "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "checks": [check.to_dict() for check in checks],
            "summary": {
                "total": len(checks),
                "healthy": sum(1 for c in checks if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in checks if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(
                    1 for c in checks if c.status == HealthStatus.UNHEALTHY
                ),
            },
        }

        self.health_history.append(results)

        logger.info(
            f"Health checks completed: {overall_status.value} "
            f"({results['summary']['healthy']} healthy, "
            f"{results['summary']['unhealthy']} unhealthy)"
        )

        return results

    def get_health_status(self) -> dict[str, Any]:
        """
        Get current health status (synchronous wrapper).

        Returns:
            Health status dictionary
        """
        return asyncio.run(self.run_all_checks())

    def get_health_history(
        self, limit: int = 100, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """
        Get health check history.

        Args:
            limit: Maximum number of records to return
            since: Only return records since this time

        Returns:
            List of health check results
        """
        history = self.health_history

        if since:
            history = [
                h for h in history if datetime.fromisoformat(h["timestamp"]) >= since
            ]

        return history[-limit:]
