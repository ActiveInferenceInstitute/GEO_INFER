"""
Intention module for BDI agents.

Intentions represent the agent's commitment to achieving goals through plans.
This module provides:
- Intention representation as a commitment to a goal with a selected plan
- Intention management with execution tracking
- Intention reconsideration for dynamic adaptation
- Intention scheduling for execution prioritization
"""

from typing import Any
import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum

from geo_infer_agent.models.bdi.plan import Plan, PlanStatus

# Configure logging
logger = logging.getLogger(__name__)


class IntentionStatus(Enum):
    """Enumeration of possible statuses for an intention."""

    ACTIVE = "active"  # The intention is currently active
    SUSPENDED = "suspended"  # The intention is temporarily suspended
    ACHIEVED = "achieved"  # The intention has been achieved
    FAILED = "failed"  # The intention has failed
    DROPPED = "dropped"  # The intention has been dropped


@dataclass
class Intention:
    """
    An intention representing a commitment to achieving a goal through a plan.

    Attributes:
        id: Unique identifier for the intention
        goal: Name of the goal this intention aims to achieve
        plan: The plan being used to achieve the goal
        status: Current status of the intention
        creation_time: When this intention was created
        deadline: Optional deadline by which the intention must be achieved
        priority: Priority level of the intention (higher values = higher priority)
        progress: Estimated progress towards achieving the goal (0.0-1.0)
        metadata: Additional information about this intention
    """

    id: str
    goal: str
    plan: Plan
    status: IntentionStatus = IntentionStatus.ACTIVE
    creation_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    deadline: datetime.datetime | None = None
    priority: float = 1.0
    progress: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate intention after initialization."""
        if self.progress < 0.0 or self.progress > 1.0:
            raise ValueError(
                f"Progress must be between 0.0 and 1.0, got {self.progress}"
            )

    def is_active(self) -> bool:
        """
        Check if this intention is active.

        Returns:
            True if the intention is active, False otherwise
        """
        return self.status == IntentionStatus.ACTIVE

    def is_achieved(self) -> bool:
        """
        Check if this intention has been achieved.

        Returns:
            True if the intention has been achieved, False otherwise
        """
        return self.status == IntentionStatus.ACHIEVED

    def is_failed(self) -> bool:
        """
        Check if this intention has failed.

        Returns:
            True if the intention has failed, False otherwise
        """
        return self.status == IntentionStatus.FAILED

    def is_dropped(self) -> bool:
        """
        Check if this intention has been dropped.

        Returns:
            True if the intention has been dropped, False otherwise
        """
        return self.status == IntentionStatus.DROPPED

    def is_suspended(self) -> bool:
        """
        Check if this intention is suspended.

        Returns:
            True if the intention is suspended, False otherwise
        """
        return self.status == IntentionStatus.SUSPENDED

    def activate(self) -> None:
        """Activate this intention."""
        if self.status == IntentionStatus.SUSPENDED:
            self.status = IntentionStatus.ACTIVE
            logger.debug(f"Activated intention: {self.id}")

    def suspend(self) -> None:
        """Suspend this intention."""
        if self.status == IntentionStatus.ACTIVE:
            self.status = IntentionStatus.SUSPENDED
            logger.debug(f"Suspended intention: {self.id}")

    def achieve(self) -> None:
        """Mark this intention as achieved."""
        self.status = IntentionStatus.ACHIEVED
        self.progress = 1.0

        # Also mark the plan as succeeded
        if self.plan.status == PlanStatus.RUNNING:
            self.plan.succeed()

        logger.debug(f"Marked intention as achieved: {self.id}")

    def fail(self) -> None:
        """Mark this intention as failed."""
        self.status = IntentionStatus.FAILED

        # Also mark the plan as failed
        if self.plan.status == PlanStatus.RUNNING:
            self.plan.fail()

        logger.debug(f"Marked intention as failed: {self.id}")

    def drop(self) -> None:
        """Drop this intention."""
        self.status = IntentionStatus.DROPPED
        logger.debug(f"Dropped intention: {self.id}")

    def set_progress(self, progress: float) -> None:
        """
        Set the progress of this intention.

        Args:
            progress: Progress value between 0.0 and 1.0

        Raises:
            ValueError: If progress is not between 0.0 and 1.0
        """
        if progress < 0.0 or progress > 1.0:
            raise ValueError(f"Progress must be between 0.0 and 1.0, got {progress}")

        self.progress = progress

    def is_expired(self) -> bool:
        """
        Check if this intention has expired (missed its deadline).

        Returns:
            True if the intention has a deadline and has expired, False otherwise
        """
        if self.deadline is None:
            return False

        return datetime.datetime.now() > self.deadline

    def should_reconsider(self, belief_values: dict[str, Any]) -> bool:
        """
        Check if this intention should be reconsidered based on current beliefs.

        This is a basic implementation that checks if:
        1. The plan has failed
        2. The intention has expired

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            True if the intention should be reconsidered, False otherwise
        """
        # Check if the plan has failed according to its failure condition
        if self.plan.has_failed(belief_values):
            return True

        # Check if the intention has expired
        if self.is_expired():
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this intention to a dictionary representation.

        Returns:
            Dictionary representation of this intention
        """
        return {
            "id": self.id,
            "goal": self.goal,
            "plan": self.plan.to_dict(),
            "status": self.status.value,
            "creation_time": self.creation_time.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "priority": self.priority,
            "progress": self.progress,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intention":
        """
        Create an Intention instance from a dictionary.

        Args:
            data: Dictionary representation of an intention

        Returns:
            Intention instance
        """
        # Handle special fields
        if "status" in data and isinstance(data["status"], str):
            data["status"] = IntentionStatus(data["status"])

        if "creation_time" in data and isinstance(data["creation_time"], str):
            data["creation_time"] = datetime.datetime.fromisoformat(
                data["creation_time"]
            )

        if (
            "deadline" in data
            and isinstance(data["deadline"], str)
            and data["deadline"] is not None
        ):
            data["deadline"] = datetime.datetime.fromisoformat(data["deadline"])

        if "plan" in data and isinstance(data["plan"], dict):
            data["plan"] = Plan.from_dict(data["plan"])

        return cls(**data)


class IntentionStructure:
    """
    A structure for managing intentions.

    This class handles:
    - Adding, updating, and removing intentions
    - Selecting intentions for execution based on priority
    - Tracking intention execution and progress
    - Reconsideration of intentions
    """

    def __init__(self) -> None:
        """Initialize an empty intention structure."""
        self._intentions: dict[str, Intention] = {}

    def add(self, intention: Intention) -> None:
        """
        Add a new intention to the intention structure.

        Args:
            intention: Intention to add

        Raises:
            ValueError: If an intention with the same ID already exists
        """
        if intention.id in self._intentions:
            raise ValueError(f"Intention with ID '{intention.id}' already exists")

        self._intentions[intention.id] = intention
        logger.debug(f"Added intention: {intention.id}")

    def get(self, intention_id: str) -> Intention:
        """
        Get an intention by ID.

        Args:
            intention_id: ID of the intention to get

        Returns:
            The intention with the given ID

        Raises:
            KeyError: If no intention with the given ID exists
        """
        if intention_id not in self._intentions:
            raise KeyError(f"No intention with ID '{intention_id}' exists")

        return self._intentions[intention_id]

    def remove(self, intention_id: str) -> None:
        """
        Remove an intention from the intention structure.

        Args:
            intention_id: ID of the intention to remove

        Raises:
            KeyError: If no intention with the given ID exists
        """
        if intention_id not in self._intentions:
            raise KeyError(f"No intention with ID '{intention_id}' exists")

        del self._intentions[intention_id]
        logger.debug(f"Removed intention: {intention_id}")

    def has_intention(self, intention_id: str) -> bool:
        """
        Check if an intention with the given ID exists.

        Args:
            intention_id: ID of the intention to check for

        Returns:
            True if an intention with the given ID exists, False otherwise
        """
        return intention_id in self._intentions

    def has_intention_for_goal(self, goal: str) -> bool:
        """
        Check if there is an active intention for the given goal.

        Args:
            goal: Name of the goal to check for

        Returns:
            True if there is an active intention for the given goal, False otherwise
        """
        for intention in self._intentions.values():
            if intention.goal == goal and intention.is_active():
                return True

        return False

    def get_all(self) -> dict[str, Intention]:
        """
        Get all intentions in the intention structure.

        Returns:
            Dictionary mapping intention IDs to Intention objects
        """
        return dict(self._intentions)

    def get_active(self) -> dict[str, Intention]:
        """
        Get all active intentions in the intention structure.

        Returns:
            Dictionary mapping intention IDs to active Intention objects
        """
        return {
            id: intention
            for id, intention in self._intentions.items()
            if intention.is_active()
        }

    def get_suspended(self) -> dict[str, Intention]:
        """
        Get all suspended intentions in the intention structure.

        Returns:
            Dictionary mapping intention IDs to suspended Intention objects
        """
        return {
            id: intention
            for id, intention in self._intentions.items()
            if intention.is_suspended()
        }

    def get_achieved(self) -> dict[str, Intention]:
        """
        Get all achieved intentions in the intention structure.

        Returns:
            Dictionary mapping intention IDs to achieved Intention objects
        """
        return {
            id: intention
            for id, intention in self._intentions.items()
            if intention.is_achieved()
        }

    def get_failed(self) -> dict[str, Intention]:
        """
        Get all failed intentions in the intention structure.

        Returns:
            Dictionary mapping intention IDs to failed Intention objects
        """
        return {
            id: intention
            for id, intention in self._intentions.items()
            if intention.is_failed()
        }

    def get_by_goal(self, goal: str) -> list[Intention]:
        """
        Get all intentions for a specific goal.

        Args:
            goal: Name of the goal to get intentions for

        Returns:
            List of intentions for the given goal
        """
        return [
            intention
            for intention in self._intentions.values()
            if intention.goal == goal
        ]

    def select_intention(self, max_count: int | None = None) -> list[Intention]:
        """
        Select intentions for execution based on priority.

        Args:
            max_count: Maximum number of intentions to select

        Returns:
            List of selected intentions, sorted by priority (highest first)
        """
        # Get all active intentions
        active_intentions = list(self.get_active().values())

        # Sort by priority (highest first)
        active_intentions.sort(key=lambda i: i.priority, reverse=True)

        # Limit to max_count if specified
        if max_count is not None and max_count >= 0:
            return active_intentions[:max_count]

        return active_intentions

    def reconsider_intentions(
        self, belief_values: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """
        Reconsider intentions based on current beliefs.

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            Tuple containing two lists:
            - List of IDs of newly achieved intentions
            - List of IDs of newly failed intentions
        """
        achieved = []
        failed = []

        for intention_id, intention in self._intentions.items():
            if intention.is_active():
                # Check if the intention has been achieved
                if intention.plan.has_succeeded(belief_values):
                    intention.achieve()
                    achieved.append(intention_id)

                # Check if the intention has failed
                elif intention.should_reconsider(belief_values):
                    intention.fail()
                    failed.append(intention_id)

        return achieved, failed

    def update_progress(self, intention_id: str, progress: float) -> None:
        """
        Update the progress of an intention.

        Args:
            intention_id: ID of the intention to update
            progress: New progress value

        Raises:
            KeyError: If no intention with the given ID exists
        """
        intention = self.get(intention_id)
        intention.set_progress(progress)

    def clean_completed(
        self, keep_achieved: bool = False, keep_failed: bool = False
    ) -> int:
        """
        Remove completed intentions from the intention structure.

        Args:
            keep_achieved: Whether to keep achieved intentions
            keep_failed: Whether to keep failed intentions

        Returns:
            Number of intentions removed
        """
        to_remove = []

        for intention_id, intention in self._intentions.items():
            if (
                (intention.is_achieved() and not keep_achieved)
                or (intention.is_failed() and not keep_failed)
                or intention.is_dropped()
            ):
                to_remove.append(intention_id)

        for intention_id in to_remove:
            del self._intentions[intention_id]

        return len(to_remove)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the intention structure to a dictionary representation.

        Returns:
            Dictionary representation of the intention structure
        """
        return {
            "intentions": {
                id: intention.to_dict() for id, intention in self._intentions.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentionStructure":
        """
        Create an IntentionStructure instance from a dictionary.

        Args:
            data: Dictionary representation of an intention structure

        Returns:
            IntentionStructure instance
        """
        intention_structure = cls()

        # Add intentions
        for id, intention_data in data.get("intentions", {}).items():
            intention = Intention.from_dict(intention_data)
            intention_structure._intentions[id] = intention

        return intention_structure
