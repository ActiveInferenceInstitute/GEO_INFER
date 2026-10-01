#!/usr/bin/env python3
"""GEO-INFER-OPS module orchestrator.

Runs one documented end-to-end OPS operation: a health-check sweep with the
real ``HealthChecker`` (system-resource probe plus registered synthetic
service checks) and a dependency-ordered task workflow executed by the real
``Orchestrator`` on synthetic metrics. All work goes through the real
``geo_infer_ops`` public API.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_ops import HealthChecker, HealthCheck, HealthStatus
    from geo_infer_ops import Orchestrator, TaskStatus

    rng = np.random.default_rng(42)

    # Synthetic service metrics: latency and queue depth for two services.
    ingest_latency_ms = float(np.round(rng.uniform(20.0, 40.0), 2))
    storage_queue_depth = int(rng.integers(10, 40))

    def _ingest_pipeline_check() -> HealthCheck:
        healthy = ingest_latency_ms < 100.0
        return HealthCheck(
            name="ingest_pipeline",
            status=HealthStatus.HEALTHY if healthy else HealthStatus.DEGRADED,
            message=f"Ingest latency {ingest_latency_ms} ms",
            details={"latency_ms": ingest_latency_ms, "threshold_ms": 100.0},
        )

    def _storage_queue_check() -> HealthCheck:
        healthy = storage_queue_depth < 50
        return HealthCheck(
            name="storage_queue",
            status=HealthStatus.HEALTHY if healthy else HealthStatus.DEGRADED,
            message=f"Queue depth {storage_queue_depth}",
            details={"queue_depth": storage_queue_depth, "threshold": 50},
        )

    health_checker = HealthChecker(
        check_interval_seconds=60, timeout_seconds=5, enable_system_checks=True
    )
    health_checker.register_check("ingest_pipeline", _ingest_pipeline_check)
    health_checker.register_check("storage_queue", _storage_queue_check)
    health_report = asyncio.run(health_checker.run_all_checks())

    # Dependency-ordered synthetic workflow: ingest -> validate + summarize.
    ingest_rows = int(rng.integers(50, 100))
    values = np.round(rng.normal(50.0, 10.0, ingest_rows), 3)

    def _ingest() -> dict[str, Any]:
        return {"rows": ingest_rows, "mean": float(np.mean(values))}

    def _validate() -> bool:
        return bool(np.all(np.isfinite(values)))

    def _summarize() -> dict[str, Any]:
        return {
            "rows": ingest_rows,
            "mean": float(np.round(np.mean(values), 3)),
            "std": float(np.round(np.std(values), 3)),
            "min": float(np.round(np.min(values), 3)),
            "max": float(np.round(np.max(values), 3)),
        }

    orchestrator = Orchestrator(max_concurrent_tasks=4, enable_monitoring=True)
    orchestrator.add_task("ingest", _ingest, task_id="ingest")
    orchestrator.add_task(
        "validate", _validate, dependencies=["ingest"], task_id="validate"
    )
    orchestrator.add_task(
        "summarize", _summarize, dependencies=["validate"], task_id="summarize"
    )
    workflow = asyncio.run(orchestrator.execute_workflow())

    task_statuses = {
        task_id: info["status"] for task_id, info in workflow["task_results"].items()
    }
    summarize_status = orchestrator.get_task_status("summarize")
    return {
        "operation": "health_check_and_task_orchestration",
        "synthetic_ingest_latency_ms": ingest_latency_ms,
        "synthetic_storage_queue_depth": storage_queue_depth,
        "health_overall_status": health_report["status"],
        "health_check_count": health_report["summary"]["total"],
        "health_summary": health_report["summary"],
        "system_check_details": {
            key: value
            for key, value in health_report["checks"][0].get("details", {}).items()
        },
        "workflow_total_tasks": workflow["total_tasks"],
        "workflow_completed": workflow["completed_tasks"],
        "workflow_failed": workflow["failed_tasks"],
        "workflow_task_statuses": task_statuses,
        "all_tasks_completed": all(
            status == TaskStatus.COMPLETED.value for status in task_statuses.values()
        ),
        "workflow_summarize_result": (
            summarize_status["result"] if summarize_status else None
        ),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("OPS", _operation))
