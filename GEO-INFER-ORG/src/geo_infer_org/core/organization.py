"""
Organization modeling for GEO-INFER-ORG.

Provides org structure graph construction, role hierarchy analysis,
and resource allocation optimization.
"""

from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrgStructureType(Enum):
    """Types of organizational structures."""

    HIERARCHICAL = "hierarchical"
    MATRIX = "matrix"
    FLAT = "flat"
    NETWORK = "network"
    HYBRID = "hybrid"


class RoleLevel(Enum):
    """Hierarchical role levels."""

    EXECUTIVE = 5
    DIRECTOR = 4
    MANAGER = 3
    LEAD = 2
    INDIVIDUAL = 1


@dataclass
class OrgUnit:
    """Represents an organizational unit (department, team, division)."""

    unit_id: str
    name: str
    parent_id: str | None = None
    head_role: str | None = None
    member_count: int = 0
    budget: float = 0.0
    location: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Role:
    """Represents a role within the organization."""

    role_id: str
    title: str
    level: RoleLevel
    unit_id: str
    reports_to: str | None = None
    responsibilities: list[str] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)


@dataclass
class Resource:
    """Represents an allocatable resource."""

    resource_id: str
    name: str
    capacity: float
    unit_cost: float
    assigned_to: str | None = None
    resource_type: str = "general"


@dataclass
class OrgMetrics:
    """Organizational structure metrics."""

    total_units: int
    total_roles: int
    max_depth: int
    avg_span_of_control: float
    centralization_score: float
    hierarchy_ratio: float


class OrganizationModel:
    """
    Models organizational structures as directed graphs.

    Supports building hierarchical, matrix, and network org structures,
    computing structural metrics, and analyzing reporting relationships.
    """

    def __init__(
        self, structure_type: OrgStructureType = OrgStructureType.HIERARCHICAL
    ) -> None:
        """
        Initialize the organization model.

        Args:
            structure_type: The type of organizational structure.
        """
        self._structure_type = structure_type
        self._units: dict[str, OrgUnit] = {}
        self._roles: dict[str, Role] = {}
        self._adjacency: dict[str, list[str]] = {}  # parent -> children

    def add_unit(self, unit: OrgUnit) -> None:
        """
        Add an organizational unit to the model.

        Args:
            unit: The organizational unit to add.

        Raises:
            ValueError: If unit_id already exists or parent doesn't exist.
        """
        if unit.unit_id in self._units:
            raise ValueError(f"Unit {unit.unit_id} already exists")
        if unit.parent_id and unit.parent_id not in self._units:
            raise ValueError(f"Parent unit {unit.parent_id} does not exist")

        self._units[unit.unit_id] = unit
        self._adjacency.setdefault(unit.unit_id, [])
        if unit.parent_id:
            self._adjacency.setdefault(unit.parent_id, [])
            self._adjacency[unit.parent_id].append(unit.unit_id)
        logger.info("Unit added: %s (parent=%s)", unit.unit_id, unit.parent_id)

    def add_role(self, role: Role) -> None:
        """
        Add a role to the organization model.

        Args:
            role: The role to add.

        Raises:
            ValueError: If role_id already exists or unit doesn't exist.
        """
        if role.role_id in self._roles:
            raise ValueError(f"Role {role.role_id} already exists")
        if role.unit_id not in self._units:
            raise ValueError(f"Unit {role.unit_id} does not exist")

        self._roles[role.role_id] = role

    def get_unit(self, unit_id: str) -> OrgUnit:
        """
        Retrieve an organizational unit.

        Args:
            unit_id: The unit identifier.

        Returns:
            The organizational unit.

        Raises:
            KeyError: If the unit does not exist.
        """
        if unit_id not in self._units:
            raise KeyError(f"Unit {unit_id} not found")
        return self._units[unit_id]

    def get_children(self, unit_id: str) -> list[OrgUnit]:
        """
        Get direct child units of a given unit.

        Args:
            unit_id: The parent unit identifier.

        Returns:
            List of child OrgUnit objects.
        """
        child_ids = self._adjacency.get(unit_id, [])
        return [self._units[cid] for cid in child_ids]

    def get_descendants(self, unit_id: str) -> list[OrgUnit]:
        """
        Get all descendant units (recursive children) of a unit.

        Args:
            unit_id: The root unit identifier.

        Returns:
            List of all descendant OrgUnit objects.
        """
        descendants = []
        stack = list(self._adjacency.get(unit_id, []))
        while stack:
            current = stack.pop()
            descendants.append(self._units[current])
            stack.extend(self._adjacency.get(current, []))
        return descendants

    def get_ancestors(self, unit_id: str) -> list[OrgUnit]:
        """
        Get the chain of ancestors from a unit up to the root.

        Args:
            unit_id: The unit identifier.

        Returns:
            List of ancestor OrgUnit objects, from parent to root.
        """
        ancestors: list[OrgUnit] = []
        current = self._units.get(unit_id)
        if current is None:
            return ancestors
        while current.parent_id:
            parent = self._units.get(current.parent_id)
            if parent is None:
                break
            ancestors.append(parent)
            current = parent
        return ancestors

    def compute_depth(self, unit_id: str) -> int:
        """
        Compute the depth of a unit in the hierarchy (root = 0).

        Args:
            unit_id: The unit identifier.

        Returns:
            Depth as an integer.
        """
        return len(self.get_ancestors(unit_id))

    def compute_metrics(self) -> OrgMetrics:
        """
        Compute organizational structure metrics.

        Returns:
            OrgMetrics with structural analysis values.
        """
        if not self._units:
            return OrgMetrics(
                total_units=0,
                total_roles=0,
                max_depth=0,
                avg_span_of_control=0.0,
                centralization_score=0.0,
                hierarchy_ratio=0.0,
            )

        # Max depth
        max_depth = 0
        for uid in self._units:
            d = self.compute_depth(uid)
            if d > max_depth:
                max_depth = d

        # Average span of control (avg number of direct children for non-leaf nodes)
        spans = []
        for children in self._adjacency.values():
            if children:
                spans.append(len(children))
        avg_span = sum(spans) / len(spans) if spans else 0.0

        # Centralization: proportion of units at top 2 levels
        total = len(self._units)
        top_level_count = sum(1 for uid in self._units if self.compute_depth(uid) <= 1)
        centralization = top_level_count / total if total > 0 else 0.0

        # Hierarchy ratio: managers / individual contributors
        manager_count = sum(
            1 for r in self._roles.values() if r.level.value >= RoleLevel.MANAGER.value
        )
        ic_count = sum(
            1 for r in self._roles.values() if r.level.value < RoleLevel.MANAGER.value
        )
        hierarchy_ratio = manager_count / ic_count if ic_count > 0 else 0.0

        return OrgMetrics(
            total_units=total,
            total_roles=len(self._roles),
            max_depth=max_depth,
            avg_span_of_control=round(avg_span, 2),
            centralization_score=round(centralization, 4),
            hierarchy_ratio=round(hierarchy_ratio, 4),
        )

    def find_reporting_chain(self, role_id: str) -> list[Role]:
        """
        Find the complete reporting chain for a role.

        Args:
            role_id: The role identifier.

        Returns:
            List of Role objects from immediate supervisor to top.

        Raises:
            KeyError: If the role does not exist.
        """
        if role_id not in self._roles:
            raise KeyError(f"Role {role_id} not found")

        chain = []
        current = self._roles[role_id]
        visited: set[str] = {role_id}
        while current.reports_to and current.reports_to not in visited:
            supervisor = self._roles.get(current.reports_to)
            if supervisor is None:
                break
            chain.append(supervisor)
            visited.add(supervisor.role_id)
            current = supervisor
        return chain

    def allocate_budget(
        self,
        total_budget: float,
        strategy: str = "proportional",
    ) -> dict[str, float]:
        """
        Allocate budget across organizational units.

        Strategies:
        - "proportional": Based on member count
        - "equal": Equal distribution
        - "weighted": Based on existing budget weights

        Args:
            total_budget: Total budget to allocate.
            strategy: Allocation strategy name.

        Returns:
            Mapping of unit_id to allocated budget amount.

        Raises:
            ValueError: If total_budget is negative or strategy is unknown.
        """
        if total_budget < 0:
            raise ValueError("total_budget must be non-negative")
        if not self._units:
            return {}

        if strategy == "equal":
            per_unit = total_budget / len(self._units)
            result = {uid: round(per_unit, 2) for uid in self._units}
        elif strategy == "proportional":
            total_members = sum(u.member_count for u in self._units.values())
            if total_members == 0:
                per_unit = total_budget / len(self._units)
                result = {uid: round(per_unit, 2) for uid in self._units}
            else:
                result = {
                    uid: round(total_budget * u.member_count / total_members, 2)
                    for uid, u in self._units.items()
                }
        elif strategy == "weighted":
            total_weight = sum(u.budget for u in self._units.values())
            if total_weight == 0:
                per_unit = total_budget / len(self._units)
                result = {uid: round(per_unit, 2) for uid in self._units}
            else:
                result = {
                    uid: round(total_budget * u.budget / total_weight, 2)
                    for uid, u in self._units.items()
                }
        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")

        logger.info(
            "Budget allocated: strategy=%s total=%.2f units=%d",
            strategy,
            total_budget,
            len(result),
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the org model to a dictionary.

        Returns:
            Dictionary representation of the organization.
        """
        return {
            "structure_type": self._structure_type.value,
            "units": {
                uid: {
                    "unit_id": u.unit_id,
                    "name": u.name,
                    "parent_id": u.parent_id,
                    "member_count": u.member_count,
                    "budget": u.budget,
                }
                for uid, u in self._units.items()
            },
            "roles": {
                rid: {
                    "role_id": r.role_id,
                    "title": r.title,
                    "level": r.level.name,
                    "unit_id": r.unit_id,
                    "reports_to": r.reports_to,
                }
                for rid, r in self._roles.items()
            },
        }
