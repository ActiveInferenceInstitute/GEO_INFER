"""
Plan module for BDI agents.

Plans represent procedural knowledge about how to achieve goals. This module provides:
- Plan representation with conditions and actions
- Plan selection based on context and applicability
- Plan execution with success/failure handling
- Plan library management for reusable plans
"""

from typing import Any
import logging
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """Enumeration of possible statuses for a plan."""

    PENDING = "pending"  # The plan has not been started
    RUNNING = "running"  # The plan is currently running
    SUCCEEDED = "succeeded"  # The plan has succeeded
    FAILED = "failed"  # The plan has failed


@dataclass
class Plan:
    """
    A plan representing a way to achieve a goal.

    Attributes:
        name: Unique identifier for the plan
        description: Human-readable description of the plan
        goal: The name of the desire/goal this plan aims to achieve (also aliased as desire_name)
        context_condition: Condition that must be true for the plan to be applicable
        context_conditions: Additional context predicates used for applicability
        precondition: Condition that must be true for the plan to start
        postcondition: Condition that indicates the plan has succeeded
        failure_condition: Condition that indicates the plan has failed
        actions: List of actions or sub-plans to execute
        status: Current status of the plan
        priority: Priority level of the plan (higher values = higher priority)
        metadata: Additional information about this plan
        current_action_index: Index of the current action being executed
        complete: Whether this plan has finished executing
        successful: Whether the plan succeeded (None if not complete)
        execution_record: Log of executed actions and their results
    """

    name: str
    description: str = ""
    goal: str = ""
    desire_name: str = ""
    context_condition: dict[str, Any] = field(default_factory=dict)
    context_conditions: dict[str, Any] = field(default_factory=dict)
    precondition: dict[str, Any] = field(default_factory=dict)
    postcondition: dict[str, Any] = field(default_factory=dict)
    failure_condition: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    priority: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    current_action_index: int = 0
    complete: bool = False
    successful: bool | None = None
    execution_record: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize goal/desire_name and context aliases."""
        # Sync goal and desire_name: whichever is set, copy to the other
        if self.desire_name and not self.goal:
            self.goal = self.desire_name
        elif self.goal and not self.desire_name:
            self.desire_name = self.goal
        # Sync context_conditions and context_condition
        if self.context_conditions and not self.context_condition:
            self.context_condition = self.context_conditions
        elif self.context_condition and not self.context_conditions:
            self.context_conditions = self.context_condition

    def is_applicable(self, belief_values: dict[str, Any]) -> bool:
        """
        Check if this plan is applicable in the current context.

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            True if the plan is applicable, False otherwise
        """
        for key, value in self.context_condition.items():
            if key not in belief_values or belief_values[key] != value:
                return False

        return True

    def can_start(self, belief_values: dict[str, Any]) -> bool:
        """
        Check if this plan can start based on its precondition.

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            True if the plan can start, False otherwise
        """
        if self.status != PlanStatus.PENDING:
            return False

        for key, value in self.precondition.items():
            if key not in belief_values or belief_values[key] != value:
                return False

        return True

    def has_succeeded(self, belief_values: dict[str, Any]) -> bool:
        """
        Check if this plan has succeeded based on its postcondition.

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            True if the plan has succeeded, False otherwise
        """
        for key, value in self.postcondition.items():
            if key not in belief_values or belief_values[key] != value:
                return False

        return True

    def has_failed(self, belief_values: dict[str, Any]) -> bool:
        """
        Check if this plan has failed based on its failure condition.

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            True if the plan has failed, False otherwise
        """
        for key, value in self.failure_condition.items():
            if key in belief_values and belief_values[key] == value:
                return True

        return False

    def start(self) -> None:
        """
        Start executing this plan.

        Raises:
            ValueError: If the plan is not in the PENDING status
        """
        if self.status != PlanStatus.PENDING:
            raise ValueError(
                f"Cannot start plan '{self.name}' with status '{self.status.value}'"
            )

        self.status = PlanStatus.RUNNING
        logger.debug(f"Started plan: {self.name}")

    def succeed(self) -> None:
        """
        Mark this plan as succeeded.

        Raises:
            ValueError: If the plan is not in the RUNNING status
        """
        if self.status != PlanStatus.RUNNING:
            raise ValueError(
                f"Cannot succeed plan '{self.name}' with status '{self.status.value}'"
            )

        self.status = PlanStatus.SUCCEEDED
        logger.debug(f"Plan succeeded: {self.name}")

    def fail(self) -> None:
        """
        Mark this plan as failed.

        Raises:
            ValueError: If the plan is not in the RUNNING status
        """
        if self.status != PlanStatus.RUNNING:
            raise ValueError(
                f"Cannot fail plan '{self.name}' with status '{self.status.value}'"
            )

        self.status = PlanStatus.FAILED
        logger.debug(f"Plan failed: {self.name}")

    def reset(self) -> None:
        """Reset this plan to the PENDING status."""
        self.status = PlanStatus.PENDING
        self.current_action_index = 0
        self.complete = False
        self.successful = None
        self.execution_record = []
        logger.debug(f"Reset plan: {self.name}")

    def next_action(self) -> dict[str, Any] | None:
        """
        Get the current action to execute without advancing.

        Returns:
            The current action dict, or None if all actions are done
        """
        if self.current_action_index >= len(self.actions):
            return None
        return self.actions[self.current_action_index]

    def advance(self) -> bool:
        """Advance to the next action in the plan.

        Returns:
            True while actions remain to execute, False once the plan is
            complete (every action has been consumed).
        """
        self.current_action_index += 1
        if self.current_action_index >= len(self.actions):
            self.complete = True
            return False
        return True

    def mark_complete(self, successful: bool) -> None:
        """
        Mark this plan as complete with a success/failure status.

        Args:
            successful: Whether the plan completed successfully
        """
        self.complete = True
        self.successful = successful
        self.status = PlanStatus.SUCCEEDED if successful else PlanStatus.FAILED

    def record_action_result(
        self, action_index: int, result: dict[str, Any], success: bool
    ) -> None:
        """
        Record the result of an executed action.

        Args:
            action_index: Index of the action that was executed
            result: Result dictionary from the action execution
            success: Whether the action succeeded
        """
        self.execution_record.append(
            {
                "action_index": action_index,
                "result": result,
                "success": success,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this plan to a dictionary representation.

        Returns:
            Dictionary representation of this plan
        """
        return {
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "desire_name": self.desire_name,
            "context_condition": self.context_condition,
            "context_conditions": self.context_conditions,
            "precondition": self.precondition,
            "postcondition": self.postcondition,
            "failure_condition": self.failure_condition,
            "actions": self.actions,
            "status": self.status.value,
            "priority": self.priority,
            "metadata": self.metadata,
            "current_action_index": self.current_action_index,
            "complete": self.complete,
            "successful": self.successful,
            "execution_record": self.execution_record,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        """
        Create a Plan instance from a dictionary.

        Args:
            data: Dictionary representation of a plan

        Returns:
            Plan instance
        """
        d = dict(data)

        # Handle status conversion from string
        if "status" in d and isinstance(d["status"], str):
            d["status"] = PlanStatus(d["status"])

        # Drop extra keys not in the dataclass
        known_fields = {
            "name",
            "description",
            "goal",
            "desire_name",
            "context_condition",
            "context_conditions",
            "precondition",
            "postcondition",
            "failure_condition",
            "actions",
            "status",
            "priority",
            "metadata",
            "current_action_index",
            "complete",
            "successful",
            "execution_record",
        }
        filtered = {k: v for k, v in d.items() if k in known_fields}

        return cls(**filtered)


class PlanLibrary:
    """
    A collection of plans maintained by an agent.

    This class handles:
    - Adding, updating, and removing plans
    - Querying plans by various criteria
    - Plan selection based on goals and context
    """

    def __init__(self) -> None:
        """Initialize an empty plan library."""
        self._plans: dict[str, Plan] = {}

    def add(self, plan: Plan) -> None:
        """
        Add a new plan to the plan library.

        Args:
            plan: Plan to add

        Raises:
            ValueError: If a plan with the same name already exists
        """
        if plan.name in self._plans:
            raise ValueError(f"Plan with name '{plan.name}' already exists")

        self._plans[plan.name] = plan
        logger.debug(f"Added plan: {plan.name}")

    def get(self, name: str) -> Plan:
        """
        Get a plan by name.

        Args:
            name: Name of the plan to get

        Returns:
            The plan with the given name

        Raises:
            KeyError: If no plan with the given name exists
        """
        if name not in self._plans:
            raise KeyError(f"No plan with name '{name}' exists")

        return self._plans[name]

    def remove(self, name: str) -> None:
        """
        Remove a plan from the plan library.

        Args:
            name: Name of the plan to remove

        Raises:
            KeyError: If no plan with the given name exists
        """
        if name not in self._plans:
            raise KeyError(f"No plan with name '{name}' exists")

        del self._plans[name]
        logger.debug(f"Removed plan: {name}")

    def has_plan(self, name: str) -> bool:
        """
        Check if a plan with the given name exists.

        Args:
            name: Name of the plan to check for

        Returns:
            True if a plan with the given name exists, False otherwise
        """
        return name in self._plans

    def get_all(self) -> dict[str, Plan]:
        """
        Get all plans in the plan library.

        Returns:
            Dictionary mapping plan names to Plan objects
        """
        return dict(self._plans)

    def get_by_goal(self, goal: str) -> list[Plan]:
        """
        Get all plans for a specific goal.

        Args:
            goal: Name of the goal to get plans for

        Returns:
            List of plans for the given goal
        """
        return [plan for plan in self._plans.values() if plan.goal == goal]

    def select_plan(self, goal: str, belief_values: dict[str, Any]) -> Plan | None:
        """
        Select the most appropriate plan for a given goal in the current context.

        Args:
            goal: Name of the goal to select a plan for
            belief_values: Dictionary of belief values to check against

        Returns:
            The most appropriate applicable plan, or None if no applicable plan exists
        """
        # Get all plans for this goal
        candidate_plans = self.get_by_goal(goal)

        # Filter out non-applicable plans
        applicable_plans = [
            plan for plan in candidate_plans if plan.is_applicable(belief_values)
        ]

        # If no applicable plans, return None
        if not applicable_plans:
            return None

        # Sort by priority (highest first)
        applicable_plans.sort(key=lambda p: p.priority, reverse=True)

        # Return the highest priority applicable plan
        return applicable_plans[0]

    def create_plan_instance(
        self, template_name: str, instance_name: str | None = None
    ) -> Plan:
        """
        Create a new instance of a plan from a template.

        Args:
            template_name: Name of the template plan
            instance_name: Name for the new plan instance (generated if None)

        Returns:
            A new instance of the template plan

        Raises:
            KeyError: If no template plan with the given name exists
        """
        # Get the template plan
        template = self.get(template_name)

        # Create a copy of the template
        template_dict = template.to_dict()

        # Reset the status to PENDING
        template_dict["status"] = PlanStatus.PENDING.value

        # Generate a unique name if none provided
        if instance_name is None:
            instance_name = f"{template_name}_instance_{id(template_dict)}"

        template_dict["name"] = instance_name

        # Create a new plan from the template
        return Plan.from_dict(template_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the plan library to a dictionary representation.

        Returns:
            Dictionary representation of the plan library
        """
        return {"plans": {name: plan.to_dict() for name, plan in self._plans.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanLibrary":
        """
        Create a PlanLibrary instance from a dictionary.

        Args:
            data: Dictionary representation of a plan library

        Returns:
            PlanLibrary instance
        """
        plan_library = cls()

        # Add plans
        for name, plan_data in data.get("plans", {}).items():
            plan = Plan.from_dict(plan_data)
            plan_library._plans[name] = plan

        return plan_library
