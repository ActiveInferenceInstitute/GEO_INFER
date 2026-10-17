"""
Orchestration engine for GEO-INFER-OPS.

This module provides workflow orchestration, task scheduling, and
dependency management for GEO-INFER module operations.
"""

import logging
import asyncio
import inspect
import math
import time
from typing import Any
from collections.abc import Callable
from collections import deque
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
        retry_delay_seconds: float = 5,
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
        self._validate_configuration()
        self._running: dict[str, asyncio.Task[None]] = {}
        self._executing = False

        self.tasks: dict[str, Task] = {}
        self.task_dependencies: dict[str, set[str]] = {}
        self.execution_history: list[dict[str, Any]] = []

    def _validate_configuration(self) -> None:
        if type(self.max_concurrent_tasks) is not int or self.max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be a positive integer")
        delay = self.retry_delay_seconds
        if (
            isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not math.isfinite(delay)
            or delay < 0
        ):
            raise ValueError("retry_delay_seconds must be finite and nonnegative")

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
        if self._executing:
            raise RuntimeError("Cannot add tasks while a workflow is executing")
        if not callable(func):
            raise TypeError("Task func must be callable")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if dependencies is not None and (
            not isinstance(dependencies, list)
            or any(not isinstance(dep, str) or not dep for dep in dependencies)
        ):
            raise ValueError("dependencies must be an array of task identifiers")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary of keyword arguments")
        task_id = str(uuid.uuid4()) if task_id is None else task_id
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty string")
        if task_id in self.tasks:
            raise ValueError(f"Task already exists: {task_id}")

        task = Task(
            task_id=task_id,
            name=name,
            func=func,
            dependencies=list(dependencies or []),
            max_retries=max_retries,
            metadata=dict(metadata or {}),
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
        for task_id, task in self.tasks.items():
            deps = task.dependencies
            for dep_id in deps:
                if dep_id not in self.tasks:
                    logger.error(f"Task {task_id} has invalid dependency: {dep_id}")
                    return False
        # Check the whole graph before invoking even independent tasks.
        degrees = {
            name: len(set(task.dependencies)) for name, task in self.tasks.items()
        }
        dependents: dict[str, list[str]] = {name: [] for name in self.tasks}
        for name, task in self.tasks.items():
            for dependency in set(task.dependencies):
                dependents[dependency].append(name)
        ready = deque(name for name, degree in degrees.items() if degree == 0)
        resolved = 0
        while ready:
            name = ready.popleft()
            resolved += 1
            for dependent in dependents[name]:
                degrees[dependent] -= 1
                if degrees[dependent] == 0:
                    ready.append(dependent)
        if resolved != len(self.tasks):
            logger.error("Workflow dependency cycle detected")
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
        try:
            while True:
                try:
                    result = task.func(**task.metadata)
                    if inspect.isawaitable(result):
                        result = await result
                    if task.status == TaskStatus.CANCELLED:
                        raise asyncio.CancelledError
                    task.result = result
                    task.error = None
                    task.status = TaskStatus.COMPLETED
                    break
                except Exception as error:
                    task.error = str(error)
                    if task.retry_count >= task.max_retries:
                        task.status = TaskStatus.FAILED
                        logger.error(
                            "Task failed: %s (%s): %s", task.name, task.task_id, error
                        )
                        break
                    task.retry_count += 1
                    logger.warning(
                        "Retrying task %s (%s), retry %s/%s",
                        task.name,
                        task.task_id,
                        task.retry_count,
                        task.max_retries,
                    )
                    await asyncio.sleep(self.retry_delay_seconds)
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.error = "Task cancelled"
            raise
        finally:
            task.end_time = datetime.now(UTC).replace(tzinfo=None)

    async def execute_workflow(self) -> dict[str, Any]:
        """
        Execute the complete workflow.

        Returns:
            Workflow execution results
        """
        if self._executing:
            raise RuntimeError("Workflow is already executing")
        self._validate_configuration()
        if not self._validate_dependencies():
            raise ValueError(
                "Invalid task dependencies detected (missing task or cycle)"
            )
        self._executing = True
        start_time = datetime.now(UTC).replace(tzinfo=None)
        started = time.perf_counter()
        try:
            while True:
                ready = self._get_ready_tasks()
                for task in ready[: self.max_concurrent_tasks - len(self._running)]:
                    self._running[task.task_id] = asyncio.create_task(
                        self._execute_task(task)
                    )
                if not self._running:
                    for task in self.tasks.values():
                        if task.status == TaskStatus.PENDING:
                            task.status = TaskStatus.FAILED
                            task.error = "Workflow deadlock: dependencies not satisfied"
                            task.end_time = datetime.now(UTC).replace(tzinfo=None)
                    break
                done, _ = await asyncio.wait(
                    self._running.values(), return_when=asyncio.FIRST_COMPLETED
                )
                for name, future in list(self._running.items()):
                    if future in done:
                        del self._running[name]
                        if not future.cancelled():
                            future.result()
        except BaseException:
            for future in self._running.values():
                future.cancel()
            await asyncio.gather(*self._running.values(), return_exceptions=True)
            for task in self.tasks.values():
                if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                    task.status = TaskStatus.CANCELLED
                    task.error = "Workflow cancelled"
                    task.end_time = datetime.now(UTC).replace(tzinfo=None)
            raise
        finally:
            self._running.clear()
            self._executing = False
        end_time = datetime.now(UTC).replace(tzinfo=None)
        duration = time.perf_counter() - started
        completed_tasks = sum(
            task.status == TaskStatus.COMPLETED for task in self.tasks.values()
        )
        failed_tasks = sum(
            task.status == TaskStatus.FAILED for task in self.tasks.values()
        )

        # Generate execution summary
        results = {
            "workflow_id": str(uuid.uuid4()),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_tasks": len(self.tasks),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "cancelled_tasks": sum(
                task.status == TaskStatus.CANCELLED for task in self.tasks.values()
            ),
            "skipped_tasks": sum(
                task.status == TaskStatus.SKIPPED for task in self.tasks.values()
            ),
            "task_results": {
                task_id: {
                    "name": task.name,
                    "status": task.status.value,
                    "result": str(task.result) if task.result is not None else None,
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

    def get_task_result(self, task_id: str) -> Any:
        """Return a completed task's original native value for composition.

        Raises KeyError for an unknown task and RuntimeError before completion.
        Status summaries retain their historical string representation.
        """
        task = self.tasks[task_id]
        if task.status != TaskStatus.COMPLETED:
            raise RuntimeError(
                f"Task is not completed: {task_id} ({task.status.value})"
            )
        return task.result

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
            "dependencies": list(task.dependencies),
            "result": str(task.result) if task.result is not None else None,
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
            task.error = "Task cancelled"
            future = self._running.get(task_id)
            if future is not None:
                future.cancel()
            task.end_time = datetime.now(UTC).replace(tzinfo=None)
            logger.info(f"Cancelled task: {task.name} (ID: {task_id})")
            return True

        return False

    def reset_workflow(self) -> None:
        """Reset all tasks to pending status."""
        if self._executing:
            raise RuntimeError("Cannot reset an executing workflow")
        for task in self.tasks.values():
            task.status = TaskStatus.PENDING
            task.result = None
            task.error = None
            task.start_time = None
            task.end_time = None
            task.retry_count = 0

        logger.info("Workflow reset")
