"""
Desire module for BDI agents.

Desires represent goals that the agent wants to achieve. This module provides:
- Basic desire representation with attributes
- Priority-based desire management
- Conditioned desires with activation/deactivation conditions
- Geospatial desires with location-specific goals
"""

from typing import Any
import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class DesireState(Enum):
    """Enumeration of possible states for a desire."""

    ACTIVE = "active"  # The desire is currently active
    INACTIVE = "inactive"  # The desire is currently inactive
    ACHIEVED = "achieved"  # The desire has been achieved
    FAILED = "failed"  # The desire has permanently failed
    SUSPENDED = "suspended"  # The desire is temporarily suspended


@dataclass
class Desire:
    """
    A desire representing a goal the agent wants to achieve.

    Attributes:
        name: Unique identifier for the desire
        description: Human-readable description of the desire
        priority: Priority level of the desire (higher values = higher priority)
        state: Current state of the desire
        deadline: Optional deadline for the desire
        conditions: General conditions for satisfaction
        preconditions: Dictionary of conditions that must be met for the desire to be adopted
        success_conditions: Dictionary of conditions that indicate the desire has been achieved
        failure_conditions: Dictionary of conditions that indicate the desire has failed
        spatial_reference: Optional geospatial reference for location-specific desires
        metadata: Additional information about this desire
        achieved: Whether this desire has been achieved
        achieved_at: Timestamp when the desire was achieved
    """

    name: str
    description: str
    priority: float = 1.0
    state: DesireState = DesireState.ACTIVE
    deadline: datetime.datetime | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    preconditions: dict[str, Any] = field(default_factory=dict)
    success_conditions: dict[str, Any] = field(default_factory=dict)
    failure_conditions: dict[str, Any] = field(default_factory=dict)
    spatial_reference: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    achieved: bool = False
    achieved_at: datetime.datetime | None = None

    def set_achieved(self, achieved: bool) -> None:
        """
        Set the achieved status of this desire.

        Args:
            achieved: Whether the desire is achieved
        """
        self.achieved = achieved
        if achieved:
            self.achieved_at = datetime.datetime.now()
            self.state = DesireState.ACHIEVED
        else:
            self.achieved_at = None
            if self.state == DesireState.ACHIEVED:
                self.state = DesireState.ACTIVE

    def is_expired(self) -> bool:
        """
        Check if this desire has expired based on its deadline.

        Returns:
            True if the desire has a deadline that has passed, False otherwise
        """
        if self.deadline is None:
            return False
        return datetime.datetime.now() > self.deadline

    def is_active(self) -> bool:
        """
        Check if this desire is active.

        Returns:
            True if the desire is active, False otherwise
        """
        return self.state == DesireState.ACTIVE

    def is_achieved(self) -> bool:
        """
        Check if this desire has been achieved.

        Returns:
            True if the desire has been achieved, False otherwise
        """
        return self.state == DesireState.ACHIEVED

    def is_failed(self) -> bool:
        """
        Check if this desire has failed.

        Returns:
            True if the desire has failed, False otherwise
        """
        return self.state == DesireState.FAILED

    def activate(self) -> None:
        """Activate this desire."""
        if self.state != DesireState.ACHIEVED and self.state != DesireState.FAILED:
            self.state = DesireState.ACTIVE
            logger.debug(f"Activated desire: {self.name}")

    def deactivate(self) -> None:
        """Deactivate this desire."""
        if self.state == DesireState.ACTIVE:
            self.state = DesireState.INACTIVE
            logger.debug(f"Deactivated desire: {self.name}")

    def suspend(self) -> None:
        """Suspend this desire."""
        if self.state == DesireState.ACTIVE:
            self.state = DesireState.SUSPENDED
            logger.debug(f"Suspended desire: {self.name}")

    def mark_achieved(self) -> None:
        """Mark this desire as achieved."""
        self.state = DesireState.ACHIEVED
        logger.debug(f"Marked desire as achieved: {self.name}")

    def mark_failed(self) -> None:
        """Mark this desire as failed."""
        self.state = DesireState.FAILED
        logger.debug(f"Marked desire as failed: {self.name}")

    def check_preconditions(self, belief_values: dict[str, Any]) -> bool:
        """
        Check if the preconditions for this desire are met.

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            True if all preconditions are met, False otherwise
        """
        for key, value in self.preconditions.items():
            if key not in belief_values or belief_values[key] != value:
                return False

        return True

    def check_success(self, belief_values: dict[str, Any]) -> bool:
        """
        Check if the success conditions for this desire are met.

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            True if all success conditions are met, False otherwise
        """
        for key, value in self.success_conditions.items():
            if key not in belief_values or belief_values[key] != value:
                return False

        return True

    def check_failure(self, belief_values: dict[str, Any]) -> bool:
        """
        Check if the failure conditions for this desire are met.

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            True if any failure condition is met, False otherwise
        """
        for key, value in self.failure_conditions.items():
            if key in belief_values and belief_values[key] == value:
                return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this desire to a dictionary representation.

        Returns:
            Dictionary representation of this desire
        """
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "state": self.state.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "conditions": self.conditions,
            "preconditions": self.preconditions,
            "success_conditions": self.success_conditions,
            "failure_conditions": self.failure_conditions,
            "spatial_reference": self.spatial_reference,
            "metadata": self.metadata,
            "achieved": self.achieved,
            "achieved_at": self.achieved_at.isoformat() if self.achieved_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Desire":
        """
        Create a Desire instance from a dictionary.

        Args:
            data: Dictionary representation of a desire

        Returns:
            Desire instance
        """
        # Work on a copy to avoid mutating input
        d = dict(data)

        # Handle state conversion from string
        if "state" in d and isinstance(d["state"], str):
            d["state"] = DesireState(d["state"])

        # Handle timestamp conversions
        if "deadline" in d and isinstance(d["deadline"], str):
            d["deadline"] = datetime.datetime.fromisoformat(d["deadline"])

        if "achieved_at" in d and isinstance(d["achieved_at"], str):
            d["achieved_at"] = datetime.datetime.fromisoformat(d["achieved_at"])

        # Drop extra keys not in the dataclass
        known_fields = {
            "name",
            "description",
            "priority",
            "state",
            "deadline",
            "conditions",
            "preconditions",
            "success_conditions",
            "failure_conditions",
            "spatial_reference",
            "metadata",
            "achieved",
            "achieved_at",
        }
        filtered = {k: v for k, v in d.items() if k in known_fields}

        return cls(**filtered)


class DesireSet:
    """
    A collection of desires maintained by an agent.

    This class handles:
    - Adding, updating, and removing desires
    - Querying desires by various criteria
    - Selecting desires based on priority and state
    - Checking desire conditions against beliefs
    """

    def __init__(self) -> None:
        """Initialize an empty desire set."""
        self._desires: dict[str, Desire] = {}

    def add(self, desire: Desire) -> None:
        """
        Add a new desire to the desire set.

        Args:
            desire: Desire to add

        Raises:
            ValueError: If a desire with the same name already exists
        """
        if desire.name in self._desires:
            raise ValueError(f"Desire with name '{desire.name}' already exists")

        self._desires[desire.name] = desire
        logger.debug(f"Added desire: {desire.name}")

    def get(self, name: str) -> Desire:
        """
        Get a desire by name.

        Args:
            name: Name of the desire to get

        Returns:
            The desire with the given name

        Raises:
            KeyError: If no desire with the given name exists
        """
        if name not in self._desires:
            raise KeyError(f"No desire with name '{name}' exists")

        return self._desires[name]

    def remove(self, name: str) -> None:
        """
        Remove a desire from the desire set.

        Args:
            name: Name of the desire to remove

        Raises:
            KeyError: If no desire with the given name exists
        """
        if name not in self._desires:
            raise KeyError(f"No desire with name '{name}' exists")

        del self._desires[name]
        logger.debug(f"Removed desire: {name}")

    def has_desire(self, name: str) -> bool:
        """
        Check if a desire with the given name exists.

        Args:
            name: Name of the desire to check for

        Returns:
            True if a desire with the given name exists, False otherwise
        """
        return name in self._desires

    def get_all(self) -> dict[str, Desire]:
        """
        Get all desires in the desire set.

        Returns:
            Dictionary mapping desire names to Desire objects
        """
        return dict(self._desires)

    def get_active(self) -> dict[str, Desire]:
        """
        Get all active desires in the desire set.

        Returns:
            Dictionary mapping desire names to active Desire objects
        """
        return {
            name: desire for name, desire in self._desires.items() if desire.is_active()
        }

    def get_achieved(self) -> dict[str, Desire]:
        """
        Get all achieved desires in the desire set.

        Returns:
            Dictionary mapping desire names to achieved Desire objects
        """
        return {
            name: desire
            for name, desire in self._desires.items()
            if desire.is_achieved()
        }

    def get_failed(self) -> dict[str, Desire]:
        """
        Get all failed desires in the desire set.

        Returns:
            Dictionary mapping desire names to failed Desire objects
        """
        return {
            name: desire for name, desire in self._desires.items() if desire.is_failed()
        }

    def select_desires(self, max_count: int | None = None) -> list[Desire]:
        """
        Select desires based on priority.

        Args:
            max_count: Maximum number of desires to select

        Returns:
            List of selected desires, sorted by priority (highest first)
        """
        # Get all active desires
        active_desires = list(self.get_active().values())

        # Sort by priority (highest first)
        active_desires.sort(key=lambda d: d.priority, reverse=True)

        # Limit to max_count if specified
        if max_count is not None and max_count >= 0:
            return active_desires[:max_count]

        return active_desires

    def update_states(
        self, belief_values: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """
        Update the states of all desires based on the current beliefs.

        Args:
            belief_values: Dictionary of belief values to check against

        Returns:
            Tuple containing two lists:
            - List of names of newly achieved desires
            - List of names of newly failed desires
        """
        achieved = []
        failed = []

        for name, desire in self._desires.items():
            if desire.is_active():
                # Check for success
                if desire.check_success(belief_values):
                    desire.mark_achieved()
                    achieved.append(name)

                # Check for failure
                elif desire.check_failure(belief_values):
                    desire.mark_failed()
                    failed.append(name)
            elif desire.state == DesireState.INACTIVE:
                # Check if preconditions are met
                if desire.check_preconditions(belief_values):
                    desire.activate()

        return achieved, failed

    def query_spatial(self, center: dict[str, float], radius: float) -> list[Desire]:
        """
        Query desires by spatial reference within a radius of a center point.

        Args:
            center: Center point coordinates
            radius: Radius to search within

        Returns:
            List of desires with spatial references within the radius
        """
        results = []

        for desire in self._desires.values():
            if desire.spatial_reference and self._is_in_radius(
                desire.spatial_reference, center, radius
            ):
                results.append(desire)

        return results

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the desire set to a dictionary representation.

        Returns:
            Dictionary representation of the desire set
        """
        return {
            "desires": {
                name: desire.to_dict() for name, desire in self._desires.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesireSet":
        """
        Create a DesireSet instance from a dictionary.

        Args:
            data: Dictionary representation of a desire set

        Returns:
            DesireSet instance
        """
        desire_set = cls()

        # Add desires
        for name, desire_data in data.get("desires", {}).items():
            desire = Desire.from_dict(desire_data)
            desire_set._desires[name] = desire

        return desire_set

    @staticmethod
    def _is_in_radius(
        location: dict[str, float], center: dict[str, float], radius: float
    ) -> bool:
        """
        Check if a location is within a radius of a center point.

        Uses the haversine formula to compute great-circle distance on Earth.
        ``radius`` is interpreted in kilometres.

        Args:
            location: Location to check (expects "lat" and "lng" keys, degrees)
            center: Center location (expects "lat" and "lng" keys, degrees)
            radius: Maximum distance from center in kilometres

        Returns:
            True if the location is within the radius, False otherwise
        """
        import math

        if not location or not center:
            return False

        lat1 = math.radians(location.get("lat", 0))
        lng1 = math.radians(location.get("lng", 0))
        lat2 = math.radians(center.get("lat", 0))
        lng2 = math.radians(center.get("lng", 0))

        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
        )
        distance_km = 2 * 6371.0 * math.asin(math.sqrt(a))

        return distance_km <= radius
