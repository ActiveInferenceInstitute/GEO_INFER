"""
Orchestration engine for GEO-INFER-OPS.

This module provides workflow orchestration, task scheduling, and
dependency management for GEO-INFER module operations.
"""

import logging
import asyncio
from typing import Any
from collections.abc import Callable
from datetime import datetime, UTC
from dataclasses import dataclass, field
from enum import StrEnum
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(StrEnum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a task in the orchestration workflow."""

    task_id: str
    name: str
    func: Callable
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any | None = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize task ID if not provided."""
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


class Orchestrator:
    """
    Workflow orchestrator for GEO-INFER operations.

    Manages task execution, dependency resolution, error handling,
    and workflow monitoring.
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        retry_delay_seconds: int = 5,
        enable_monitoring: bool = True,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks
            retry_delay_seconds: Delay between retries in seconds
            enable_monitoring: Whether to enable workflow monitoring
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.retry_delay_seconds = retry_delay_seconds
        self.enable_monitoring = enable_monitoring

        self.tasks: dict[str, Task] = {}
        self.task_dependencies: dict[str, set[str]] = {}
        self.execution_history: list[dict[str, Any]] = []

    def add_task(
        self,
        name: str,
        func: Callable,
        dependencies: list[str] | None = None,
        task_id: str | None = None,
        max_retries: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a task to the workflow.

        Args:
            name: Task name
            func: Function to execute
            dependencies: List of task IDs this task depends on
            task_id: Optional task ID (if None, generates one)
            max_retries: Maximum number of retries on failure
            metadata: Optional task metadata

        Returns:
            Task ID
        """
        task_id = task_id or str(uuid.uuid4())

        task = Task(
            task_id=task_id,
            name=name,
            func=func,
            dependencies=dependencies or [],
            max_retries=max_retries,
            metadata=metadata or {},
        )

        self.tasks[task_id] = task
        self.task_dependencies[task_id] = set(task.dependencies)

        logger.info(f"Added task: {name} (ID: {task_id})")
        return task_id

    def _validate_dependencies(self) -> bool:
        """
        Validate that all task dependencies exist.

        Returns:
            True if all dependencies are valid, False otherwise
        """
        for task_id, deps in self.task_dependencies.items():
            for dep_id in deps:
                if dep_id not in self.tasks:
                    logger.error(f"Task {task_id} has invalid dependency: {dep_id}")
                    return False
        return True

    def _get_ready_tasks(self) -> list[Task]:
        """
        Get tasks that are ready to execute (dependencies satisfied).

        Returns:
            List of ready tasks
        """
        ready_tasks = []

        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            deps_satisfied = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if deps_satisfied:
                ready_tasks.append(task)

        return ready_tasks

    async def _execute_task(self, task: Task) -> None:
        """
        Execute a single task.

        Args:
            task: Task to execute
        """
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now(UTC).replace(tzinfo=None)

        logger.info(f"Executing task: {task.name} (ID: {task.task_id})")

        try:
            # Execute the task function
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(**task.metadata)
            else:
                result = task.func(**task.metadata)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now(UTC).replace(tzinfo=None)

            duration = (task.end_time - task.start_time).total_seconds()
            logger.info(
                f"Task completed: {task.name} (ID: {task.task_id}) in {duration:.2f}s"
            )

        except Exception as e:
            task.error = str(e)
            task.end_time = datetime.now(UTC).replace(tzinfo=None)

            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.warning(
                    f"Task failed, retrying: {task.name} "
                    f"(ID: {task.task_id}, attempt {task.retry_count}/{task.max_retries})"
                )
                await asyncio.sleep(self.retry_delay_seconds)
            else:
                task.status = TaskStatus.FAILED
                logger.error(
                    f"Task failed after {task.max_retries} retries: "
                    f"{task.name} (ID: {task.task_id}): {e}"
                )

    async def execute_workflow(self) -> dict[str, Any]:
        """
        Execute the complete workflow.

        Returns:
            Workflow execution results
        """
        logger.info("Starting workflow execution")

        if not self._validate_dependencies():
            raise ValueError("Invalid task dependencies detected")

        start_time = datetime.now(UTC).replace(tzinfo=None)
        completed_tasks = 0
        failed_tasks = 0
        counted_tasks: set[str] = set()

        # Execute tasks until all are completed or failed
        while True:
            ready_tasks = self._get_ready_tasks()

            if not ready_tasks:
                # Check if workflow is complete
                all_complete = all(
                    task.status
                    in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                    for task in self.tasks.values()
                )

                if all_complete:
                    break

                # Check for deadlock (tasks pending but dependencies not satisfied)
                pending_tasks = [
                    task
                    for task in self.tasks.values()
                    if task.status == TaskStatus.PENDING
                ]

                if pending_tasks:
                    logger.warning(
                        f"Workflow deadlock detected: {len(pending_tasks)} tasks pending"
                    )
                    for task in pending_tasks:
                        task.status = TaskStatus.FAILED
                        task.error = "Workflow deadlock: dependencies not satisfied"
                    break

                await asyncio.sleep(0.1)
                continue

            # Execute ready tasks concurrently (up to max_concurrent_tasks)
            tasks_to_execute = ready_tasks[: self.max_concurrent_tasks]
            await asyncio.gather(
                *[self._execute_task(task) for task in tasks_to_execute]
            )

            # Update counters idempotently: each task is counted at most once,
            # regardless of how many loop iterations observe its final status.
            for task in self.tasks.values():
                if task.task_id in counted_tasks:
                    continue
                if task.status == TaskStatus.COMPLETED:
                    completed_tasks += 1
                    counted_tasks.add(task.task_id)
                elif task.status == TaskStatus.FAILED:
                    failed_tasks += 1
                    counted_tasks.add(task.task_id)

        end_time = datetime.now(UTC).replace(tzinfo=None)
        duration = (end_time - start_time).total_seconds()

        # Generate execution summary
        results = {
            "workflow_id": str(uuid.uuid4()),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_tasks": len(self.tasks),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "task_results": {
                task_id: {
                    "name": task.name,
                    "status": task.status.value,
                    "result": str(task.result) if task.result else None,
                    "error": task.error,
                    "duration": (
                        (task.end_time - task.start_time).total_seconds()
                        if task.start_time and task.end_time
                        else None
                    ),
                }
                for task_id, task in self.tasks.items()
            },
        }

        self.execution_history.append(results)

        logger.info(
            f"Workflow execution completed: {completed_tasks} completed, "
            f"{failed_tasks} failed in {duration:.2f}s"
        )

        return results

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """
        Get status of a specific task.

        Args:
            task_id: Task identifier

        Returns:
            Task status dictionary or None if task not found
        """
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "dependencies": task.dependencies,
            "result": str(task.result) if task.result else None,
            "error": task.error,
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None,
        }

    def get_workflow_status(self) -> dict[str, Any]:
        """
        Get overall workflow status.

        Returns:
            Workflow status dictionary
        """
        status_counts: dict[str, int] = {}
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_tasks": len(self.tasks),
            "status_counts": status_counts,
            "tasks": {
                task_id: self.get_task_status(task_id) for task_id in self.tasks.keys()
            },
        }

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled, False otherwise
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task.status = TaskStatus.CANCELLED
            task.end_time = datetime.now(UTC).replace(tzinfo=None)
            logger.info(f"Cancelled task: {task.name} (ID: {task_id})")
            return True

        return False

    def reset_workflow(self) -> None:
        """Reset all tasks to pending status."""
        for task in self.tasks.values():
            task.status = TaskStatus.PENDING
            task.result = None
            task.error = None
            task.start_time = None
            task.end_time = None
            task.retry_count = 0

        logger.info("Workflow reset")
