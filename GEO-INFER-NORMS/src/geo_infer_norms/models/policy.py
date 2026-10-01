"""
Policy models for representing policy frameworks and implementations.

This module provides data models for policies, policy implementations, and
their relationships with jurisdictions and entities.
"""

from dataclasses import dataclass, field
from typing import Any
import datetime
from shapely.geometry import MultiPolygon
import uuid


@dataclass
class Policy:
    """
    A class representing a policy or plan.

    Policies are formal statements of intention that guide decision-making
    and implementation of regulatory frameworks.
    """

    id: str
    name: str
    description: str
    policy_type: str  # e.g., 'land use', 'transportation', 'environmental'
    issuing_authority: str
    jurisdiction_ids: list[str] = field(default_factory=list)
    adoption_date: datetime.date | None = None
    effective_date: datetime.date | None = None
    expiration_date: datetime.date | None = None
    parent_policy_id: str | None = None
    related_policies: list[str] = field(default_factory=list)
    related_regulations: list[str] = field(default_factory=list)
    objectives: list[dict[str, Any]] = field(default_factory=list)
    spatial_extent: MultiPolygon | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    category: str | None = None
    status: str | None = None
    source_url: str | None = None
    tags: list[str] | None = None

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        policy_type: str,
        issuing_authority: str,
        jurisdiction_ids: list[str] | None = None,
        adoption_date: datetime.date | None = None,
        effective_date: datetime.date | None = None,
        expiration_date: datetime.date | None = None,
        parent_policy_id: str | None = None,
        related_policies: list[str] | None = None,
        related_regulations: list[str] | None = None,
        objectives: list[dict[str, Any]] | None = None,
        spatial_extent: MultiPolygon | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> "Policy":
        """
        Create a new Policy with a generated UUID.

        Args:
            name: Name of the policy
            description: Description of the policy
            policy_type: Type of policy
            issuing_authority: Authority that issued the policy
            jurisdiction_ids: List of jurisdiction IDs where the policy applies
            adoption_date: Date when the policy was adopted
            effective_date: Date when the policy became effective
            expiration_date: Date when the policy expires
            parent_policy_id: Optional ID of parent policy
            related_policies: List of related policy IDs
            related_regulations: List of related regulation IDs
            objectives: List of policy objectives
            spatial_extent: Optional spatial extent of the policy
            attributes: Dictionary of additional attributes

        Returns:
            A new Policy instance
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            policy_type=policy_type,
            issuing_authority=issuing_authority,
            jurisdiction_ids=jurisdiction_ids or [],
            adoption_date=adoption_date,
            effective_date=effective_date,
            expiration_date=expiration_date,
            parent_policy_id=parent_policy_id,
            related_policies=related_policies or [],
            related_regulations=related_regulations or [],
            objectives=objectives or [],
            spatial_extent=spatial_extent,
            attributes=attributes or {},
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the policy.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def add_jurisdiction(self, jurisdiction_id: str) -> None:
        """
        Add a jurisdiction to the policy.

        Args:
            jurisdiction_id: ID of the jurisdiction to add
        """
        if jurisdiction_id not in self.jurisdiction_ids:
            self.jurisdiction_ids.append(jurisdiction_id)
            self.updated_at = datetime.datetime.now()

    def add_related_policy(self, policy_id: str) -> None:
        """
        Add a related policy to the policy.

        Args:
            policy_id: ID of the related policy to add
        """
        if policy_id != self.id and policy_id not in self.related_policies:
            self.related_policies.append(policy_id)
            self.updated_at = datetime.datetime.now()

    def add_related_regulation(self, regulation_id: str) -> None:
        """
        Add a related regulation to the policy.

        Args:
            regulation_id: ID of the related regulation to add
        """
        if regulation_id not in self.related_regulations:
            self.related_regulations.append(regulation_id)
            self.updated_at = datetime.datetime.now()

    def add_objective(
        self,
        name: str,
        description: str,
        metrics: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add an objective to the policy.

        Args:
            name: Name of the objective
            description: Description of the objective
            metrics: Optional list of metrics for evaluating the objective
        """
        objective = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "metrics": metrics or [],
        }

        self.objectives.append(objective)
        self.updated_at = datetime.datetime.now()

    def set_spatial_extent(self, extent: MultiPolygon) -> None:
        """
        Set the spatial extent of the policy.

        Args:
            extent: Spatial extent as a MultiPolygon
        """
        self.spatial_extent = extent
        self.updated_at = datetime.datetime.now()

    def is_active(self, reference_date: datetime.date | None = None) -> bool:
        """
        Check if the policy is active as of the reference date.

        Args:
            reference_date: Date to check against (defaults to current date)

        Returns:
            True if the policy is active, False otherwise
        """
        if reference_date is None:
            reference_date = datetime.date.today()

        if self.effective_date is None:
            is_effective = True
        else:
            is_effective = self.effective_date <= reference_date

        is_not_expired = (
            self.expiration_date is None or reference_date <= self.expiration_date
        )

        return is_effective and is_not_expired


@dataclass
class PolicyImplementation:
    """
    A class representing the implementation of a policy.

    Policy implementations track how policies are put into practice, including
    actions, resources, and outcomes.
    """

    id: str
    policy_id: str
    name: str
    description: str
    implementation_status: (
        str  # e.g., 'planned', 'in_progress', 'completed', 'suspended'
    )
    implementing_entity_id: str
    start_date: datetime.date | None = None
    end_date: datetime.date | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    spatial_coverage: MultiPolygon | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    geometry: Any | None = None
    jurisdiction_id: str | None = None
    budget: float | None = None
    status: str | None = None
    metrics: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        policy_id: str,
        name: str,
        description: str,
        implementation_status: str,
        implementing_entity_id: str,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        actions: list[dict[str, Any]] | None = None,
        resources: list[dict[str, Any]] | None = None,
        outcomes: list[dict[str, Any]] | None = None,
        spatial_coverage: MultiPolygon | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> "PolicyImplementation":
        """
        Create a new PolicyImplementation with a generated UUID.

        Args:
            policy_id: ID of the policy being implemented
            name: Name of the implementation
            description: Description of the implementation
            implementation_status: Status of the implementation
            implementing_entity_id: ID of the entity implementing the policy
            start_date: Date when implementation started
            end_date: Date when implementation ended
            actions: List of implementation actions
            resources: List of resources used in implementation
            outcomes: List of implementation outcomes
            spatial_coverage: Optional spatial coverage of the implementation
            attributes: Dictionary of additional attributes

        Returns:
            A new PolicyImplementation instance
        """
        return cls(
            id=str(uuid.uuid4()),
            policy_id=policy_id,
            name=name,
            description=description,
            implementation_status=implementation_status,
            implementing_entity_id=implementing_entity_id,
            start_date=start_date,
            end_date=end_date,
            actions=actions or [],
            resources=resources or [],
            outcomes=outcomes or [],
            spatial_coverage=spatial_coverage,
            attributes=attributes or {},
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the implementation.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def add_action(
        self,
        name: str,
        description: str,
        status: str,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        responsible_party: str | None = None,
    ) -> str:
        """
        Add an action to the implementation.

        Args:
            name: Name of the action
            description: Description of the action
            status: Status of the action
            start_date: Optional start date of the action
            end_date: Optional end date of the action
            responsible_party: Optional party responsible for the action

        Returns:
            ID of the added action
        """
        action_id = str(uuid.uuid4())

        action = {
            "id": action_id,
            "name": name,
            "description": description,
            "status": status,
            "start_date": start_date,
            "end_date": end_date,
            "responsible_party": responsible_party,
        }

        self.actions.append(action)
        self.updated_at = datetime.datetime.now()

        return action_id

    def add_resource(
        self,
        name: str,
        resource_type: str,
        amount: float | None = None,
        unit: str | None = None,
        provider: str | None = None,
    ) -> str:
        """
        Add a resource to the implementation.

        Args:
            name: Name of the resource
            resource_type: Type of resource (e.g., 'financial', 'human', 'technical')
            amount: Optional amount of the resource
            unit: Optional unit of the resource
            provider: Optional provider of the resource

        Returns:
            ID of the added resource
        """
        resource_id = str(uuid.uuid4())

        resource = {
            "id": resource_id,
            "name": name,
            "resource_type": resource_type,
            "amount": amount,
            "unit": unit,
            "provider": provider,
        }

        self.resources.append(resource)
        self.updated_at = datetime.datetime.now()

        return resource_id

    def add_outcome(
        self,
        name: str,
        description: str,
        status: str,
        measurement: dict[str, Any] | None = None,
        achievement_date: datetime.date | None = None,
    ) -> str:
        """
        Add an outcome to the implementation.

        Args:
            name: Name of the outcome
            description: Description of the outcome
            status: Status of the outcome (e.g., 'achieved', 'partial', 'not_achieved')
            measurement: Optional measurement details
            achievement_date: Optional date when the outcome was achieved

        Returns:
            ID of the added outcome
        """
        outcome_id = str(uuid.uuid4())

        outcome = {
            "id": outcome_id,
            "name": name,
            "description": description,
            "status": status,
            "measurement": measurement or {},
            "achievement_date": achievement_date,
        }

        self.outcomes.append(outcome)
        self.updated_at = datetime.datetime.now()

        return outcome_id

    def set_spatial_coverage(self, coverage: MultiPolygon) -> None:
        """
        Set the spatial coverage of the implementation.

        Args:
            coverage: Spatial coverage as a MultiPolygon
        """
        self.spatial_coverage = coverage
        self.updated_at = datetime.datetime.now()

    def update_status(self, status: str) -> None:
        """
        Update the implementation status.

        Args:
            status: New implementation status
        """
        self.implementation_status = status
        self.updated_at = datetime.datetime.now()

        # If completing the implementation, set end date if not already set
        if status == "completed" and self.end_date is None:
            self.end_date = datetime.date.today()

    def is_active(self, reference_date: datetime.date | None = None) -> bool:
        """
        Check if the implementation is active as of the reference date.

        Args:
            reference_date: Date to check against (defaults to current date)

        Returns:
            True if the implementation is active, False otherwise
        """
        if reference_date is None:
            reference_date = datetime.date.today()

        # Implementation isn't active if it hasn't started yet
        if self.start_date is not None and self.start_date > reference_date:
            return False

        # Implementation isn't active if it has ended
        if self.end_date is not None and self.end_date < reference_date:
            return False

        # Implementation isn't active if it's suspended or completed
        if self.implementation_status in ["suspended", "completed"]:
            return False

        return True
