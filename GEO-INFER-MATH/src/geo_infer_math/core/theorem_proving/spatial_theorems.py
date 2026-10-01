"""
Spatial Mathematics Theorems Library

This module provides a library of spatial mathematics theorems
including geometric, statistical, and topological theorems.
"""

from typing import Any
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TheoremType(Enum):
    """Type of theorem."""

    GEOMETRIC = "geometric"
    STATISTICAL = "statistical"
    TOPOLOGICAL = "topological"
    ALGEBRAIC = "algebraic"
    ANALYTICAL = "analytical"


@dataclass
class SpatialTheorem:
    """Represents a spatial mathematics theorem."""

    name: str
    statement: str
    theorem_type: TheoremType
    proof: str | None = None
    assumptions: list[str] = field(default_factory=list)
    corollaries: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)


class GeometricTheorem(SpatialTheorem):
    """Geometric theorem for spatial mathematics."""

    def __init__(self, name: str, statement: str, **kwargs: Any) -> None:
        """Initialize geometric theorem."""
        super().__init__(
            name=name, statement=statement, theorem_type=TheoremType.GEOMETRIC, **kwargs
        )


class StatisticalTheorem(SpatialTheorem):
    """Statistical theorem for spatial mathematics."""

    def __init__(self, name: str, statement: str, **kwargs: Any) -> None:
        """Initialize statistical theorem."""
        super().__init__(
            name=name,
            statement=statement,
            theorem_type=TheoremType.STATISTICAL,
            **kwargs,
        )


class TopologicalTheorem(SpatialTheorem):
    """Topological theorem for spatial mathematics."""

    def __init__(self, name: str, statement: str, **kwargs: Any) -> None:
        """Initialize topological theorem."""
        super().__init__(
            name=name,
            statement=statement,
            theorem_type=TheoremType.TOPOLOGICAL,
            **kwargs,
        )


class TheoremDatabase:
    """
    Database of proven spatial mathematics theorems.

    Provides storage and retrieval of theorems for reuse.
    """

    def __init__(self) -> None:
        """Initialize theorem database."""
        self._theorems: dict[str, SpatialTheorem] = {}
        self._initialize_standard_theorems()

    def _initialize_standard_theorems(self) -> None:
        """Initialize standard spatial mathematics theorems."""
        # Geometric theorems
        self.add_theorem(
            GeometricTheorem(
                name="Triangle Inequality",
                statement="For any three points A, B, C: d(A,C) ≤ d(A,B) + d(B,C)",
                proof="Standard geometric proof",
                assumptions=["d is a metric"],
                applications=["Distance calculations", "Path planning"],
            )
        )

        self.add_theorem(
            GeometricTheorem(
                name="Pythagorean Theorem",
                statement="For a right triangle with sides a, b, c (c is hypotenuse): a² + b² = c²",
                proof="Standard geometric proof",
                applications=["Distance calculations", "Coordinate transformations"],
            )
        )

        # Statistical theorems
        self.add_theorem(
            StatisticalTheorem(
                name="Central Limit Theorem (Spatial)",
                statement="Spatial averages converge to normal distribution under certain conditions",
                proof="Statistical proof",
                assumptions=["Independent spatial samples", "Finite variance"],
                applications=["Spatial sampling", "Statistical inference"],
            )
        )

        self.add_theorem(
            StatisticalTheorem(
                name="Moran's I Expectation",
                statement="E[I] = -1/(n-1) for spatial autocorrelation under null hypothesis",
                proof="Statistical derivation",
                assumptions=["Random spatial pattern"],
                applications=["Spatial autocorrelation testing"],
            )
        )

        # Topological theorems
        self.add_theorem(
            TopologicalTheorem(
                name="Jordan Curve Theorem",
                statement="A simple closed curve divides the plane into two regions",
                proof="Topological proof",
                applications=["Polygon containment", "Spatial queries"],
            )
        )

    def add_theorem(self, theorem: SpatialTheorem) -> None:
        """
        Add a theorem to the database.

        Args:
            theorem: Theorem to add
        """
        self._theorems[theorem.name] = theorem
        logger.debug(f"Added theorem: {theorem.name}")

    def get_theorem(self, name: str) -> SpatialTheorem | None:
        """
        Retrieve a theorem by name.

        Args:
            name: Theorem name

        Returns:
            Theorem if found, None otherwise
        """
        return self._theorems.get(name)

    def search_theorems(
        self, theorem_type: TheoremType | None = None, keyword: str | None = None
    ) -> list[SpatialTheorem]:
        """
        Search theorems by type or keyword.

        Args:
            theorem_type: Filter by theorem type
            keyword: Search keyword in name or statement

        Returns:
            List of matching theorems
        """
        results = []

        for theorem in self._theorems.values():
            # Filter by type
            if theorem_type is not None and theorem.theorem_type != theorem_type:
                continue

            # Filter by keyword
            if keyword is not None:
                keyword_lower = keyword.lower()
                if (
                    keyword_lower not in theorem.name.lower()
                    and keyword_lower not in theorem.statement.lower()
                ):
                    continue

            results.append(theorem)

        return results

    def list_theorems(self) -> list[str]:
        """
        List all theorem names.

        Returns:
            List of theorem names
        """
        return list(self._theorems.keys())

    def get_theorems_by_type(self, theorem_type: TheoremType) -> list[SpatialTheorem]:
        """
        Get all theorems of a specific type.

        Args:
            theorem_type: Type of theorems to retrieve

        Returns:
            List of theorems of the specified type
        """
        return [
            theorem
            for theorem in self._theorems.values()
            if theorem.theorem_type == theorem_type
        ]


# Global theorem database instance
_theorem_database = TheoremDatabase()


def get_theorem_database() -> TheoremDatabase:
    """
    Get the global theorem database.

    Returns:
        Global theorem database instance
    """
    return _theorem_database
