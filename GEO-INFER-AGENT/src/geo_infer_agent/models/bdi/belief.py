"""
Belief module for BDI agents.

Beliefs represent an agent's information about the world. This module provides:
- Basic belief representation
- Belief base management with consistency checking
- Temporal belief tracking for historical data
- Spatial belief representation for geospatial applications
- Uncertainty handling with confidence levels
"""

from typing import Any
import datetime
import logging
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Belief:
    """
    A single belief representing a piece of information the agent holds about the world.

    Attributes:
        name: Unique identifier for the belief
        value: The content/value of the belief
        source: Where this belief originated (perception, communication, reasoning, etc.)
        timestamp: When this belief was acquired or last updated
        confidence: Confidence level in this belief (0.0-1.0)
        spatial_reference: Optional geospatial reference (e.g., coordinates)
        metadata: Additional information about this belief
        history: List of previous belief states (value, confidence, timestamp, metadata)
    """

    name: str
    value: Any
    source: str = "unknown"
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    confidence: float = 1.0
    spatial_reference: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate belief after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    def update(
        self,
        value: Any,
        source: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update the value, source, and confidence of this belief.

        Stores the previous state in history before updating.

        Args:
            value: New value for the belief
            source: New source of the belief (if different)
            confidence: New confidence level (if different)
            metadata: Updated metadata (if provided)
        """
        # Store current state in history before updating
        self.history.append(
            {
                "value": self.value,
                "confidence": self.confidence,
                "timestamp": self.timestamp,
                "metadata": self.metadata.copy(),
            }
        )

        self.value = value
        self.timestamp = datetime.datetime.now()

        if source is not None:
            self.source = source

        if confidence is not None:
            if confidence < 0.0 or confidence > 1.0:
                raise ValueError(
                    f"Confidence must be between 0.0 and 1.0, got {confidence}"
                )
            self.confidence = confidence

        if metadata is not None:
            self.metadata = metadata

    def is_outdated(self, max_age_seconds: float) -> bool:
        """
        Check if this belief is outdated based on its timestamp.

        Args:
            max_age_seconds: Maximum age in seconds for a belief to be considered current

        Returns:
            True if the belief is older than max_age_seconds, False otherwise
        """
        age = datetime.datetime.now() - self.timestamp
        return age.total_seconds() > max_age_seconds

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this belief to a dictionary representation.

        Returns:
            Dictionary representation of this belief
        """
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "spatial_reference": self.spatial_reference,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Belief":
        """
        Create a Belief instance from a dictionary.

        Args:
            data: Dictionary representation of a belief

        Returns:
            Belief instance
        """
        # Handle timestamp conversion from string
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])

        return cls(**data)


class BeliefBase:
    """
    A collection of beliefs maintained by an agent.

    This class handles:
    - Adding, updating, and removing beliefs
    - Querying beliefs by various criteria
    - Consistency checking between beliefs
    - Historical tracking of belief changes
    """

    def __init__(self) -> None:
        """Initialize an empty belief base."""
        self._beliefs: dict[str, Belief] = {}
        self._history: list[
            tuple[datetime.datetime, str, str, Any, Any]
        ] = []  # (time, operation, name, old_value, new_value)
        self._max_history_size: int = 1000

    def add(self, belief: Belief) -> None:
        """
        Add a new belief to the belief base.

        Args:
            belief: Belief to add

        Raises:
            ValueError: If a belief with the same name already exists
        """
        if belief.name in self._beliefs:
            raise ValueError(f"Belief with name '{belief.name}' already exists")

        self._beliefs[belief.name] = belief
        self._add_history_entry("add", belief.name, None, belief.value)
        logger.debug(f"Added belief: {belief.name}")

    def update(
        self,
        name: str,
        value: Any,
        source: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """
        Update an existing belief.

        Args:
            name: Name of the belief to update
            value: New value for the belief
            source: New source of the belief (if different)
            confidence: New confidence level (if different)

        Raises:
            KeyError: If no belief with the given name exists
        """
        if name not in self._beliefs:
            raise KeyError(f"No belief with name '{name}' exists")

        old_value = self._beliefs[name].value
        self._beliefs[name].update(value, source, confidence)
        self._add_history_entry("update", name, old_value, value)
        logger.debug(f"Updated belief: {name}")

    def get(self, name: str) -> Belief:
        """
        Get a belief by name.

        Args:
            name: Name of the belief to get

        Returns:
            The belief with the given name

        Raises:
            KeyError: If no belief with the given name exists
        """
        if name not in self._beliefs:
            raise KeyError(f"No belief with name '{name}' exists")

        return self._beliefs[name]

    def get_value(self, name: str) -> Any:
        """
        Get the value of a belief by name.

        Args:
            name: Name of the belief to get the value of

        Returns:
            The value of the belief with the given name

        Raises:
            KeyError: If no belief with the given name exists
        """
        return self.get(name).value

    def remove(self, name: str) -> None:
        """
        Remove a belief from the belief base.

        Args:
            name: Name of the belief to remove

        Raises:
            KeyError: If no belief with the given name exists
        """
        if name not in self._beliefs:
            raise KeyError(f"No belief with name '{name}' exists")

        old_value = self._beliefs[name].value
        del self._beliefs[name]
        self._add_history_entry("remove", name, old_value, None)
        logger.debug(f"Removed belief: {name}")

    def has_belief(self, name: str) -> bool:
        """
        Check if a belief with the given name exists.

        Args:
            name: Name of the belief to check for

        Returns:
            True if a belief with the given name exists, False otherwise
        """
        return name in self._beliefs

    def get_all(self) -> dict[str, Belief]:
        """
        Get all beliefs in the belief base.

        Returns:
            Dictionary mapping belief names to Belief objects
        """
        return dict(self._beliefs)

    def get_all_values(self) -> dict[str, Any]:
        """
        Get all belief values in the belief base.

        Returns:
            Dictionary mapping belief names to belief values
        """
        return {name: belief.value for name, belief in self._beliefs.items()}

    def query(self, **kwargs: Any) -> list[Belief]:
        """
        Query beliefs by various criteria.

        Args:
            **kwargs: Criteria to match against beliefs

        Returns:
            List of beliefs matching the criteria
        """
        results = []

        for belief in self._beliefs.values():
            matches = True

            for key, value in kwargs.items():
                if not hasattr(belief, key) or getattr(belief, key) != value:
                    matches = False
                    break

            if matches:
                results.append(belief)

        return results

    def query_spatial(self, center: dict[str, float], radius: float) -> list[Belief]:
        """
        Query beliefs by spatial reference within a radius of a center point.

        Args:
            center: Center point coordinates
            radius: Radius to search within

        Returns:
            List of beliefs with spatial references within the radius
        """
        results = []

        for belief in self._beliefs.values():
            if belief.spatial_reference and self._is_in_radius(
                belief.spatial_reference, center, radius
            ):
                results.append(belief)

        return results

    def check_consistency(self) -> list[tuple[str, str, Any, Any]]:
        """
        Check for consistency issues between beliefs.

        This is a simple implementation that looks for beliefs with the same name
        but different values in nested dictionaries.

        Returns:
            List of tuples describing consistency issues
        """
        issues = []
        flat_beliefs = self._flatten_beliefs()

        # Check for duplicates
        keys = set()
        for key in flat_beliefs:
            if key in keys:
                parent_key, child_key = (
                    key.rsplit(".", 1) if "." in key else (key, None)
                )
                issues.append(("duplicate", key, parent_key, child_key))
            keys.add(key)

        return issues

    def clean_outdated(self, max_age_seconds: float) -> int:
        """
        Remove beliefs that are older than a certain age.

        Args:
            max_age_seconds: Maximum age in seconds for a belief to be kept

        Returns:
            Number of beliefs removed
        """
        to_remove = []

        for name, belief in self._beliefs.items():
            if belief.is_outdated(max_age_seconds):
                to_remove.append(name)

        for name in to_remove:
            self.remove(name)

        return len(to_remove)

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get the history of belief changes.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of history entries as dictionaries
        """
        if limit is None:
            entries = self._history
        else:
            entries = self._history[-limit:]

        return [
            {
                "timestamp": timestamp.isoformat(),
                "operation": operation,
                "name": name,
                "old_value": old_value,
                "new_value": new_value,
            }
            for timestamp, operation, name, old_value, new_value in entries
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the belief base to a dictionary representation.

        Returns:
            Dictionary representation of the belief base
        """
        return {
            "beliefs": {
                name: belief.to_dict() for name, belief in self._beliefs.items()
            },
            "history": self.get_history(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BeliefBase":
        """
        Create a BeliefBase instance from a dictionary.

        Args:
            data: Dictionary representation of a belief base

        Returns:
            BeliefBase instance
        """
        belief_base = cls()

        # Add beliefs
        for name, belief_data in data.get("beliefs", {}).items():
            belief = Belief.from_dict(belief_data)
            belief_base._beliefs[name] = belief

        # We don't restore history as it's meant for runtime tracking

        return belief_base

    def _add_history_entry(
        self, operation: str, name: str, old_value: Any, new_value: Any
    ) -> None:
        """
        Add an entry to the history.

        Args:
            operation: Type of operation ("add", "update", "remove")
            name: Name of the belief
            old_value: Old value of the belief (None for "add")
            new_value: New value of the belief (None for "remove")
        """
        self._history.append(
            (datetime.datetime.now(), operation, name, old_value, new_value)
        )

        # Trim history if it exceeds the maximum size
        if len(self._history) > self._max_history_size:
            self._history = self._history[-self._max_history_size :]

    def _flatten_beliefs(self) -> dict[str, Any]:
        """
        Flatten nested belief values for consistency checking.

        Returns:
            Dictionary mapping flattened keys to values
        """
        flat_beliefs = {}

        for name, belief in self._beliefs.items():
            if isinstance(belief.value, dict):
                flat_beliefs.update(self._flatten_dict(belief.value, prefix=name))
            else:
                flat_beliefs[name] = belief.value

        return flat_beliefs

    @staticmethod
    def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """
        Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            prefix: Prefix for flattened keys

        Returns:
            Flattened dictionary
        """
        flat_dict = {}

        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flat_dict.update(BeliefBase._flatten_dict(value, prefix=new_key))
            else:
                flat_dict[new_key] = value

        return flat_dict

    @staticmethod
    def _is_in_radius(
        location: dict[str, float], center: dict[str, float], radius: float
    ) -> bool:
        """
        Check if a location is within a radius of a center point.

        Uses the haversine formula to compute great-circle distance on Earth.
        ``radius`` is interpreted in the same units as the return value of the
        haversine calculation – kilometres.

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
