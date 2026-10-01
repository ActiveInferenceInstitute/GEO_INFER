#!/usr/bin/env python3
"""
GEO-INFER Module Orchestrator

Advanced orchestration system for managing cross-module integrations,
workflow execution, and pattern-based coordination across the GEO-INFER ecosystem.

Execution-engine limitation
---------------------------
``ModuleOrchestrator`` drives every workflow step through :class:`APIConnector`,
which intentionally raises ``RuntimeError``: the local examples package ships no
remote module API servers. Consequently ``execute_workflow`` (and the five
execution strategies behind it) cannot succeed against the bundled
``SAMPLE_WORKFLOWS`` unless you provide a real transport — for example by
monkeypatching or subclassing ``APIConnector`` with an object returning
HTTP-like responses (``status_code``, ``json()``). Module discovery/health
reporting likewise only becomes meaningful once a transport is injected and
modules are declared in an orchestrator config file (``modules`` section).
The workflow *definition* surface (registration, ``list_workflows``,
``get_workflow_definition``, dependency validation, guard-condition
evaluation) is fully functional locally.
"""

import asyncio
import importlib.resources
import ast
import logging
import operator
import os
import time
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pathlib import Path

from ..models.integration_models import (
    WorkflowDefinition,
    IntegrationResult,
    ExecutionContext,
)


class ConfigManager:
    """Small local configuration reader for the self-contained examples package."""

    def __init__(self, config_path: str | None = None):
        self._config: dict[str, Any] = {}
        if config_path:
            with open(config_path, encoding="utf-8") as handle:
                self._config = yaml.safe_load(handle) or {}

    def get_config(self, key: str, default: Any = None) -> Any:
        """Return one configuration section without external service access."""
        return self._config.get(key, default)


def setup_logging(name: str) -> logging.Logger:
    """Return the package logger; CLI entrypoints own handler configuration."""
    return logging.getLogger(name)


class APIConnector:
    """Explicit boundary for optional remote health checks in examples."""

    def get(self, **_: Any) -> Any:
        """Reject undeclared remote calls instead of silently contacting a service."""
        raise RuntimeError("remote API connectors are not configured in local examples")

    async def get_async(self, **_: Any) -> Any:
        """Reject undeclared remote calls instead of silently contacting a service."""
        raise RuntimeError("remote API connectors are not configured in local examples")

    async def post_async(self, **_: Any) -> Any:
        """Reject undeclared remote calls instead of silently contacting a service."""
        raise RuntimeError("remote API connectors are not configured in local examples")


class PerformanceMonitor:
    """Track wall-clock durations for workflow executions and steps."""

    def __init__(self) -> None:
        self._workflow_starts: dict[str, float] = {}
        self._workflow_totals: dict[str, float] = {}
        self._step_starts: dict[str, float] = {}
        self._step_totals: dict[str, dict[str, float]] = {}

    def start_workflow_tracking(self, execution_id: str) -> None:
        self._workflow_starts[execution_id] = time.time()

    def stop_workflow_tracking(self, execution_id: str) -> None:
        started = self._workflow_starts.pop(execution_id, None)
        if started is not None:
            self._workflow_totals[execution_id] = time.time() - started

    def start_step_tracking(self, execution_id: str, step_name: str) -> None:
        self._step_starts[(execution_id, step_name)] = time.time()

    def stop_step_tracking(self, execution_id: str, step_name: str) -> None:
        started = self._step_starts.pop((execution_id, step_name), None)
        if started is None:
            return
        elapsed = time.time() - started
        self._step_totals.setdefault(execution_id, {})[step_name] = elapsed

    def get_workflow_metrics(self, execution_id: str) -> dict[str, Any]:
        return {
            "total_seconds": self._workflow_totals.get(execution_id),
            "step_seconds": dict(self._step_totals.get(execution_id, {})),
        }

    def shutdown(self) -> None:
        self._workflow_starts.clear()
        self._step_starts.clear()


class ExecutionStrategy(Enum):
    """Available execution strategies for workflow orchestration."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    EVENT_DRIVEN = "event_driven"
    FEEDBACK_LOOP = "feedback_loop"


class ModuleStatus(Enum):
    """Module availability and health status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    ERROR = "error"
    INITIALIZING = "initializing"


class _SafeConditionEvaluator(ast.NodeVisitor):
    """Evaluate a small, data-only expression language for workflow guards."""

    _binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _comparison_operators = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda left, right: left in right,
        ast.NotIn: lambda left, right: left not in right,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
    }

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def evaluate(self, condition: str) -> bool:
        if not condition or len(condition) > 1000:
            raise ValueError("condition must contain at most 1000 characters")
        tree = ast.parse(condition, mode="eval")
        return bool(self.visit(tree.body))

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id != "data":
            raise ValueError("only the data name is available")
        return self.data

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith("_"):
            raise ValueError("private attributes are not allowed")
        value = self.visit(node.value)
        if isinstance(value, dict):
            return value[node.attr]
        return getattr(value, node.attr)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if not isinstance(index, (str, int)):
            raise ValueError("only string and integer indexes are allowed")
        return value[index]

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_List(self, node: ast.List) -> list[Any]:
        return [self.visit(element) for element in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self.visit(element) for element in node.elts)

    def visit_Set(self, node: ast.Set) -> set[Any]:
        return {self.visit(element) for element in node.elts}

    def visit_BoolOp(self, node: ast.BoolOp) -> bool:
        if isinstance(node.op, ast.And):
            return all(self.visit(value) for value in node.values)
        if isinstance(node.op, ast.Or):
            return any(self.visit(value) for value in node.values)
        raise ValueError("unsupported boolean operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.UAdd):
            return operator.pos(value)
        if isinstance(node.op, ast.USub):
            return operator.neg(value)
        raise ValueError("unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        operation = self._binary_operators.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported binary operator")
        return operation(self.visit(node.left), self.visit(node.right))

    def visit_Compare(self, node: ast.Compare) -> bool:
        left = self.visit(node.left)
        for operation_node, comparator_node in zip(node.ops, node.comparators):
            operation = self._comparison_operators.get(type(operation_node))
            if operation is None:
                raise ValueError("unsupported comparison operator")
            right = self.visit(comparator_node)
            if not operation(left, right):
                return False
            left = right
        return True

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"unsupported condition syntax: {type(node).__name__}")


@dataclass
class WorkflowExecution:
    """Represents a single workflow execution instance."""

    workflow_id: str
    execution_id: str
    status: str
    start_time: float
    end_time: float | None = None
    results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    module_statuses: dict[str, ModuleStatus] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)


class ModuleOrchestrator:
    """
    Advanced orchestrator for managing cross-module integrations and workflows.

    Key capabilities:
    - Pattern-based workflow execution
    - Module health monitoring and failover
    - Performance optimization and resource management
    - Event-driven coordination
    - Configuration management across modules
    """

    def __init__(
        self,
        config_path: str | None = None,
        monitoring_enabled: bool = True,
        resilience_enabled: bool = True,
    ):
        """
        Initialize the module orchestrator.

        Args:
            config_path: Path to orchestrator configuration file
            monitoring_enabled: Enable performance and health monitoring
            resilience_enabled: Enable automatic failover and recovery
        """
        self.logger = setup_logging(__name__)
        self.config_manager = ConfigManager(config_path)
        self.api_connector = APIConnector()

        # Core components
        self.modules: dict[str, Any] = {}
        self.module_health: dict[str, ModuleStatus] = {}
        self.workflows: dict[str, WorkflowDefinition] = {}
        self.active_executions: dict[str, WorkflowExecution] = {}

        # Monitoring and optimization
        self.monitoring_enabled = monitoring_enabled
        self.resilience_enabled = resilience_enabled
        if monitoring_enabled:
            self.performance_monitor = PerformanceMonitor()

        # Load configuration
        self._load_configuration()
        self._initialize_modules()
        self._register_sample_workflows()

    def _load_configuration(self) -> None:
        """Load orchestrator configuration and workflow definitions."""
        try:
            # Load orchestrator settings
            orchestrator_config = self.config_manager.get_config("orchestrator", {})
            self.max_concurrent_workflows = orchestrator_config.get(
                "max_concurrent_workflows", 5
            )
            self.default_timeout = orchestrator_config.get("default_timeout", 300)
            self.retry_attempts = orchestrator_config.get("retry_attempts", 3)

            # Load packaged workflow definitions (see _workflow_definition_files).
            for workflow_file in self._workflow_definition_files():
                with workflow_file.open("r", encoding="utf-8") as f:
                    workflow_data = yaml.safe_load(f)
                    workflow = WorkflowDefinition.from_dict(workflow_data)
                    self.workflows[workflow.id] = workflow

            self.logger.info(f"Loaded {len(self.workflows)} workflow definitions")

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def _workflow_definition_files(self) -> list[Any]:
        """Return workflow-definition YAML files from the packaged directory.

        Discovery uses ``importlib.resources`` against
        ``geo_infer_examples/workflows`` so installed wheels resolve the
        packaged resources without parent-path climbing. An explicit
        ``GEO_INFER_EXAMPLES_WORKFLOWS`` environment variable overrides the
        packaged location with a filesystem directory (e.g. a repo checkout);
        a missing override directory falls back to the packaged resources
        with a warning.
        """
        override = os.environ.get("GEO_INFER_EXAMPLES_WORKFLOWS")
        if override:
            override_path = Path(override)
            if override_path.is_dir():
                return sorted(override_path.glob("*.yaml"))
            self.logger.warning(
                "GEO_INFER_EXAMPLES_WORKFLOWS is not a directory: %s", override
            )
        workflows_root = importlib.resources.files("geo_infer_examples") / "workflows"
        if not workflows_root.is_dir():
            return []
        return sorted(
            child
            for child in workflows_root.iterdir()
            if child.is_file() and child.name.endswith(".yaml")
        )

    def _initialize_modules(self) -> None:
        """Initialize and health-check available modules."""
        module_configs = self.config_manager.get_config("modules", {})

        for module_name, module_config in module_configs.items():
            try:
                self._initialize_module(module_name, module_config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize module {module_name}: {e}")
                self.module_health[module_name] = ModuleStatus.ERROR

    def _register_sample_workflows(self) -> None:
        """Register the bundled sample workflows so they are listable locally.

        Sample workflows are registered as *definitions only*: unlike
        :meth:`register_workflow`, module availability is not required, because
        the modules they reference live behind remote APIs that are not
        configured in this examples package (see the module docstring).
        """
        for data in SAMPLE_WORKFLOWS.values():
            workflow = WorkflowDefinition.from_dict(data)
            self.workflows[workflow.id] = workflow
        self.logger.info(
            f"Registered {len(SAMPLE_WORKFLOWS)} sample workflow definitions"
        )

    def _initialize_module(self, module_name: str, config: dict[str, Any]) -> None:
        """Initialize a specific module and check its health."""
        self.logger.info(f"Initializing module: {module_name}")
        self.module_health[module_name] = ModuleStatus.INITIALIZING

        try:
            # Health check via API
            health_response = self.api_connector.get(
                module=module_name, endpoint="/health", timeout=10
            )

            if health_response.status_code == 200:
                self.module_health[module_name] = ModuleStatus.AVAILABLE
                self.modules[module_name] = {
                    "config": config,
                    "api_base": config.get("api_base"),
                    "capabilities": health_response.json().get("capabilities", []),
                }
                self.logger.info(f"Module {module_name} is available")
            else:
                self.module_health[module_name] = ModuleStatus.DEGRADED
                self.logger.warning(f"Module {module_name} health check failed")

        except Exception as e:
            self.module_health[module_name] = ModuleStatus.UNAVAILABLE
            self.logger.error(f"Module {module_name} initialization failed: {e}")

    def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """
        Register a new workflow definition.

        Args:
            workflow: Workflow definition to register

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Validate workflow
            if not self._validate_workflow(workflow):
                return False

            self.workflows[workflow.id] = workflow
            self.logger.info(f"Registered workflow: {workflow.id}")
            return True

        except Exception as e:
            self.logger.error(f"Error registering workflow {workflow.id}: {e}")
            return False

    def _validate_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Validate workflow definition and module dependencies."""
        # Check if required modules are available
        for step in workflow.steps:
            module_name = step.module
            if module_name not in self.modules:
                self.logger.error(f"Required module {module_name} not available")
                return False

            if self.module_health[module_name] == ModuleStatus.ERROR:
                self.logger.error(f"Required module {module_name} in error state")
                return False

        # Validate step dependencies
        step_names = {step.name for step in workflow.steps}
        for step in workflow.steps:
            for dependency in step.dependencies:
                if dependency not in step_names:
                    self.logger.error(
                        f"Invalid dependency {dependency} in step {step.name}"
                    )
                    return False

        return True

    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: dict[str, Any],
        execution_context: ExecutionContext | None = None,
    ) -> IntegrationResult:
        """
        Execute a registered workflow with given input data.

        Args:
            workflow_id: ID of workflow to execute
            input_data: Input data for workflow execution
            execution_context: Optional execution context and parameters

        Returns:
            Integration result with outputs and metadata
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]
        execution_id = f"{workflow_id}_{int(time.time())}"

        # Create execution tracking
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status="initializing",
            start_time=time.time(),
        )
        self.active_executions[execution_id] = execution

        try:
            self.logger.info(f"Starting workflow execution: {execution_id}")

            if self.monitoring_enabled:
                self.performance_monitor.start_workflow_tracking(execution_id)

            # Execute based on strategy
            strategy = (
                workflow.execution_strategy.value
                if isinstance(workflow.execution_strategy, ExecutionStrategy)
                else str(workflow.execution_strategy)
            )
            if strategy == ExecutionStrategy.SEQUENTIAL.value:
                result = await self._execute_sequential(workflow, input_data, execution)
            elif strategy == ExecutionStrategy.PARALLEL.value:
                result = await self._execute_parallel(workflow, input_data, execution)
            elif strategy == ExecutionStrategy.CONDITIONAL.value:
                result = await self._execute_conditional(
                    workflow, input_data, execution
                )
            elif strategy == ExecutionStrategy.EVENT_DRIVEN.value:
                result = await self._execute_event_driven(
                    workflow, input_data, execution
                )
            elif strategy == ExecutionStrategy.FEEDBACK_LOOP.value:
                result = await self._execute_feedback_loop(
                    workflow, input_data, execution
                )
            else:
                raise ValueError(
                    f"Unknown execution strategy: {workflow.execution_strategy}"
                )

            execution.status = "completed"
            execution.end_time = time.time()
            execution.results = result.data

            if self.monitoring_enabled:
                execution.performance_metrics = (
                    self.performance_monitor.get_workflow_metrics(execution_id)
                )

            self.logger.info(f"Workflow execution completed: {execution_id}")
            return result

        except Exception as e:
            execution.status = "failed"
            execution.end_time = time.time()
            execution.errors.append(str(e))

            self.logger.error(f"Workflow execution failed: {execution_id} - {e}")

            if self.resilience_enabled:
                # Attempt recovery
                recovery_result = await self._attempt_recovery(
                    workflow, input_data, execution, e
                )
                if recovery_result:
                    return recovery_result

            raise
        finally:
            if self.monitoring_enabled:
                self.performance_monitor.stop_workflow_tracking(execution_id)

    async def _execute_sequential(
        self,
        workflow: WorkflowDefinition,
        input_data: dict[str, Any],
        execution: WorkflowExecution,
    ) -> IntegrationResult:
        """Execute workflow steps sequentially."""
        current_data = input_data.copy()
        results = {}

        for step in workflow.steps:
            self.logger.debug(f"Executing step: {step.name}")

            try:
                # Check module health
                if self.module_health.get(step.module) == ModuleStatus.ERROR:
                    raise Exception(f"Module {step.module} is in error state")

                # Execute step
                step_result = await self._execute_step(step, current_data, execution)
                results[step.name] = step_result

                # Update data for next step
                if step.output_mapping:
                    for key, value in step.output_mapping.items():
                        current_data[key] = step_result.get(value, step_result)
                else:
                    current_data.update(step_result)

                execution.module_statuses[step.module] = ModuleStatus.AVAILABLE

            except Exception as e:
                execution.module_statuses[step.module] = ModuleStatus.ERROR
                if not step.optional:
                    raise
                else:
                    self.logger.warning(f"Optional step {step.name} failed: {e}")
                    results[step.name] = {"error": str(e), "status": "skipped"}

        return IntegrationResult(
            success=True,
            data=results,
            metadata={
                "execution_id": execution.execution_id,
                "workflow_id": workflow.id,
                "execution_time": time.time() - execution.start_time,
            },
        )

    async def _execute_parallel(
        self,
        workflow: WorkflowDefinition,
        input_data: dict[str, Any],
        execution: WorkflowExecution,
    ) -> IntegrationResult:
        """Execute independent workflow steps in parallel."""
        # Group steps by dependencies
        dependency_groups = self._group_by_dependencies(workflow.steps)
        results = {}
        current_data = input_data.copy()

        for group in dependency_groups:
            # Execute group in parallel
            tasks = []
            for step in group:
                task = asyncio.create_task(
                    self._execute_step(step, current_data, execution)
                )
                tasks.append((step.name, step, task))

            # Wait for group completion
            group_results = {}
            for step_name, step, task in tasks:
                try:
                    step_result = await task
                    group_results[step_name] = step_result
                    execution.module_statuses[step.module] = ModuleStatus.AVAILABLE

                except Exception as e:
                    execution.module_statuses[step.module] = ModuleStatus.ERROR
                    if not step.optional:
                        raise
                    group_results[step_name] = {"error": str(e), "status": "skipped"}

            results.update(group_results)

            # Update data with group results
            for step_name, step_result in group_results.items():
                if isinstance(step_result, dict) and "error" not in step_result:
                    current_data.update(step_result)

        return IntegrationResult(
            success=True,
            data=results,
            metadata={
                "execution_id": execution.execution_id,
                "workflow_id": workflow.id,
                "execution_time": time.time() - execution.start_time,
            },
        )

    async def _execute_conditional(
        self,
        workflow: WorkflowDefinition,
        input_data: dict[str, Any],
        execution: WorkflowExecution,
    ) -> IntegrationResult:
        """Execute workflow with conditional step execution."""
        current_data = input_data.copy()
        results = {}

        for step in workflow.steps:
            # Check execution condition
            if step.condition and not self._evaluate_condition(
                step.condition, current_data
            ):
                self.logger.debug(f"Skipping step {step.name} - condition not met")
                results[step.name] = {
                    "status": "skipped",
                    "reason": "condition_not_met",
                }
                continue

            try:
                step_result = await self._execute_step(step, current_data, execution)
                results[step.name] = step_result

                # Update data
                if step.output_mapping:
                    for key, value in step.output_mapping.items():
                        current_data[key] = step_result.get(value, step_result)
                else:
                    current_data.update(step_result)

                execution.module_statuses[step.module] = ModuleStatus.AVAILABLE

            except Exception as e:
                execution.module_statuses[step.module] = ModuleStatus.ERROR
                if not step.optional:
                    raise
                results[step.name] = {"error": str(e), "status": "failed"}

        return IntegrationResult(
            success=True,
            data=results,
            metadata={
                "execution_id": execution.execution_id,
                "workflow_id": workflow.id,
                "execution_time": time.time() - execution.start_time,
            },
        )

    async def _execute_event_driven(
        self,
        workflow: WorkflowDefinition,
        input_data: dict[str, Any],
        execution: WorkflowExecution,
    ) -> IntegrationResult:
        """Execute workflow using event-driven pattern."""
        # Initialize event bus for this execution
        execution_events: dict[str, list[Any]] = {}
        results: dict[str, Any] = {}
        current_data = input_data.copy()

        # Set up event listeners
        for step in workflow.steps:
            if step.trigger_events:
                for event in step.trigger_events:
                    if event not in execution_events:
                        execution_events[event] = []
                    execution_events[event].append(step)

        # Start with steps that have no trigger events
        initial_steps = [step for step in workflow.steps if not step.trigger_events]

        # Execute initial steps
        for step in initial_steps:
            try:
                step_result = await self._execute_step(step, current_data, execution)
                results[step.name] = step_result
                current_data.update(step_result)

                # Trigger events
                if step.emits_events:
                    for event in step.emits_events:
                        await self._trigger_event(
                            event,
                            step_result,
                            execution_events,
                            current_data,
                            execution,
                            results,
                        )

            except Exception as e:
                if not step.optional:
                    raise
                results[step.name] = {"error": str(e), "status": "failed"}

        return IntegrationResult(
            success=True,
            data=results,
            metadata={
                "execution_id": execution.execution_id,
                "workflow_id": workflow.id,
                "execution_time": time.time() - execution.start_time,
            },
        )

    async def _execute_feedback_loop(
        self,
        workflow: WorkflowDefinition,
        input_data: dict[str, Any],
        execution: WorkflowExecution,
    ) -> IntegrationResult:
        """Execute workflow with feedback loops (Active Inference pattern)."""
        current_data = input_data.copy()
        results = {}
        max_iterations = workflow.max_iterations or 10
        convergence_threshold = workflow.convergence_threshold or 0.001

        for iteration in range(max_iterations):
            self.logger.debug(f"Feedback loop iteration {iteration + 1}")
            iteration_results = {}

            # Execute all steps in current iteration
            for step in workflow.steps:
                try:
                    step_result = await self._execute_step(
                        step, current_data, execution
                    )
                    iteration_results[step.name] = step_result

                    # Update beliefs/data
                    if step.feedback_mapping:
                        for key, value in step.feedback_mapping.items():
                            current_data[key] = step_result.get(value, step_result)

                except Exception as e:
                    if not step.optional:
                        raise
                    iteration_results[step.name] = {"error": str(e), "status": "failed"}

            results[f"iteration_{iteration}"] = iteration_results

            # Check convergence
            if iteration > 0 and self._check_convergence(
                results, convergence_threshold
            ):
                self.logger.info(f"Workflow converged at iteration {iteration + 1}")
                break

        return IntegrationResult(
            success=True,
            data=results,
            metadata={
                "execution_id": execution.execution_id,
                "workflow_id": workflow.id,
                "iterations": len(
                    [k for k in results.keys() if k.startswith("iteration_")]
                ),
                "execution_time": time.time() - execution.start_time,
            },
        )

    async def _execute_step(
        self, step: Any, input_data: dict[str, Any], execution: WorkflowExecution
    ) -> dict[str, Any]:
        """Execute a single workflow step."""
        module_name = step.module

        if self.monitoring_enabled:
            self.performance_monitor.start_step_tracking(
                execution.execution_id, step.name
            )

        try:
            # Prepare step input
            step_input = input_data.copy()
            if step.input_mapping:
                step_input = {
                    key: input_data.get(value, value)
                    for key, value in step.input_mapping.items()
                }

            # Execute via API
            response = await self.api_connector.post_async(
                module=module_name,
                endpoint=step.endpoint,
                data=step_input,
                timeout=step.timeout or self.default_timeout,
            )

            if response.status_code == 200:
                res = response.json()
                return res if isinstance(res, dict) else {"result": res}
            else:
                raise Exception(
                    f"Step execution failed: {response.status_code} - {response.text}"
                )

        finally:
            if self.monitoring_enabled:
                self.performance_monitor.stop_step_tracking(
                    execution.execution_id, step.name
                )

    def _group_by_dependencies(self, steps: list[Any]) -> list[list[Any]]:
        """Group workflow steps by their dependency relationships."""
        groups = []
        remaining_steps = steps.copy()
        processed_steps = set()

        while remaining_steps:
            # Find steps with no unprocessed dependencies
            ready_steps = []
            for step in remaining_steps:
                if all(dep in processed_steps for dep in step.dependencies):
                    ready_steps.append(step)

            if not ready_steps:
                # Circular dependency or other issue
                raise Exception("Unable to resolve step dependencies")

            groups.append(ready_steps)
            for step in ready_steps:
                remaining_steps.remove(step)
                processed_steps.add(step.name)

        return groups

    def _evaluate_condition(self, condition: str, data: dict[str, Any]) -> bool:
        """Evaluate a conditional expression against current data.

        A malformed guard expression (``ValueError``/``SyntaxError`` from the
        safe evaluator) logs a warning and evaluates to ``False`` so the step
        is skipped visibly. Other exception types are programming errors and
        propagate.
        """
        try:
            return _SafeConditionEvaluator(data).evaluate(condition)
        except (ValueError, SyntaxError) as exc:
            self.logger.warning(
                f"Guard condition {condition!r} treated as False: {exc}"
            )
            return False

    async def _trigger_event(
        self,
        event_name: str,
        event_data: dict[str, Any],
        execution_events: dict[str, list[Any]],
        current_data: dict[str, Any],
        execution: WorkflowExecution,
        results: dict[str, Any],
    ) -> None:
        """Trigger an event and execute associated steps."""
        if event_name in execution_events:
            for step in execution_events[event_name]:
                if step.name not in results:  # Avoid duplicate execution
                    try:
                        step_result = await self._execute_step(
                            step, current_data, execution
                        )
                        results[step.name] = step_result
                        current_data.update(step_result)

                        # Chain events
                        if step.emits_events:
                            for next_event in step.emits_events:
                                await self._trigger_event(
                                    next_event,
                                    step_result,
                                    execution_events,
                                    current_data,
                                    execution,
                                    results,
                                )
                    except Exception as e:
                        if not step.optional:
                            raise
                        results[step.name] = {"error": str(e), "status": "failed"}

    def _check_convergence(self, results: dict[str, Any], threshold: float) -> bool:
        """Check if feedback loop has converged by comparing last two iterations.

        Fewer than two iterations or no shared numeric leaves mean "not
        converged yet" (``False``). Numeric leaves that cannot be compared —
        e.g. integers beyond float range — are malformed data: they are logged
        and propagated instead of silently reading as "never converged".
        """
        iterations = sorted(k for k in results.keys() if k.startswith("iteration_"))
        if len(iterations) < 2:
            return False

        try:
            last_iter = results[iterations[-1]]
            prev_iter = results[iterations[-2]]

            # Collect all numeric leaf values from both iterations
            def _extract_nums(d: Any, prefix: str = "") -> dict[str, float]:
                vals: dict[str, float] = {}
                if isinstance(d, dict):
                    for k, v in d.items():
                        vals.update(_extract_nums(v, f"{prefix}.{k}"))
                elif isinstance(d, (int, float)):
                    vals[prefix] = float(d)
                return vals

            last_nums = _extract_nums(last_iter)
            prev_nums = _extract_nums(prev_iter)

            common_keys = set(last_nums) & set(prev_nums)
            if not common_keys:
                return False

            max_change = max(
                abs(last_nums[k] - prev_nums[k]) / max(abs(prev_nums[k]), 1e-10)
                for k in common_keys
            )

            return bool(max_change < threshold)

        except (OverflowError, RecursionError) as exc:
            self.logger.error(
                f"Convergence check failed on malformed iteration data: {exc}"
            )
            raise

    async def _attempt_recovery(
        self,
        workflow: WorkflowDefinition,
        input_data: dict[str, Any],
        execution: WorkflowExecution,
        error: Exception,
    ) -> IntegrationResult | None:
        """Attempt to recover from workflow execution failure."""
        self.logger.info(f"Attempting recovery for execution {execution.execution_id}")

        # Implement recovery strategies:
        # 1. Retry with degraded modules
        # 2. Skip optional failing steps
        # 3. Use cached results if available
        # 4. Switch to alternative workflow

        try:
            # Simple retry strategy for now
            await asyncio.sleep(1)  # Brief delay

            # Try again with optional modules marked as skippable
            modified_workflow = self._create_resilient_workflow(workflow)
            return await self.execute_workflow(
                modified_workflow.id, input_data, ExecutionContext(resilience_mode=True)
            )

        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            return None

    def _create_resilient_workflow(
        self, workflow: WorkflowDefinition
    ) -> WorkflowDefinition:
        """Create a modified workflow for resilient execution."""
        # Mark problematic modules as optional
        resilient_workflow = workflow.copy()
        for step in resilient_workflow.steps:
            if self.module_health.get(step.module) in [
                ModuleStatus.ERROR,
                ModuleStatus.UNAVAILABLE,
            ]:
                step.optional = True

        return resilient_workflow

    def get_workflow_status(self, execution_id: str) -> WorkflowExecution | None:
        """Get the status of a workflow execution."""
        return self.active_executions.get(execution_id)

    def get_module_health(self) -> dict[str, ModuleStatus]:
        """Get current health status of all modules."""
        return self.module_health.copy()

    def list_workflows(self) -> list[str]:
        """List all registered workflow IDs."""
        return list(self.workflows.keys())

    def get_workflow_definition(self, workflow_id: str) -> WorkflowDefinition | None:
        """Get workflow definition by ID."""
        return self.workflows.get(workflow_id)

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check of orchestrator and modules."""
        modules_status: dict[str, str] = {}
        health_status: dict[str, Any] = {
            "orchestrator": "healthy",
            "modules": modules_status,
            "active_executions": len(self.active_executions),
            "registered_workflows": len(self.workflows),
            "timestamp": time.time(),
        }

        # Check each module
        for module_name in self.modules:
            try:
                response = await self.api_connector.get_async(
                    module=module_name, endpoint="/health", timeout=5
                )

                if response.status_code == 200:
                    modules_status[module_name] = "healthy"
                    self.module_health[module_name] = ModuleStatus.AVAILABLE
                else:
                    modules_status[module_name] = "degraded"
                    self.module_health[module_name] = ModuleStatus.DEGRADED

            except Exception as e:
                modules_status[module_name] = f"unhealthy: {str(e)}"
                self.module_health[module_name] = ModuleStatus.UNAVAILABLE

        return health_status

    def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        self.logger.info("Shutting down module orchestrator")

        # Cancel active executions
        for execution_id in list(self.active_executions.keys()):
            execution = self.active_executions[execution_id]
            if execution.status in ["initializing", "running"]:
                execution.status = "cancelled"
                execution.end_time = time.time()

        if self.monitoring_enabled:
            self.performance_monitor.shutdown()

        self.logger.info("Module orchestrator shutdown complete")


# Example workflow definitions
SAMPLE_WORKFLOWS = {
    "health_surveillance_basic": {
        "id": "health_surveillance_basic",
        "name": "Basic Health Surveillance",
        "description": "Simple health data analysis workflow",
        "execution_strategy": "sequential",
        "steps": [
            {
                "name": "data_ingestion",
                "module": "DATA",
                "endpoint": "/ingest",
                "dependencies": [],
                "optional": False,
            },
            {
                "name": "spatial_analysis",
                "module": "SPACE",
                "endpoint": "/analyze/spatial",
                "dependencies": ["data_ingestion"],
                "optional": False,
            },
            {
                "name": "health_assessment",
                "module": "HEALTH",
                "endpoint": "/assess/outbreak",
                "dependencies": ["spatial_analysis"],
                "optional": False,
            },
        ],
    }
}


async def main() -> None:
    """Demonstration entrypoint: report health and run the first workflow."""
    orchestrator = ModuleOrchestrator()

    # Health check
    health = await orchestrator.health_check()
    print(f"Health status: {health}")

    # Execute sample workflow
    if orchestrator.workflows:
        workflow_id = list(orchestrator.workflows.keys())[0]
        result = await orchestrator.execute_workflow(
            workflow_id, {"test_data": "sample_input"}
        )
        print(f"Workflow result: {result}")

    orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
