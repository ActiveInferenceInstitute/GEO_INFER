"""
Regulation models for representing legal regulations and frameworks.

This module provides data models for regulations, regulatory frameworks, and
their relationships with jurisdictions and entities.
"""

from dataclasses import dataclass, field
from typing import Any
import datetime
import uuid


@dataclass
class Regulation:
    """
    A class representing a legal regulation.

    Regulations are legal rules that entities must comply with, often issued
    by government agencies or other regulatory bodies.
    """

    id: str
    name: str
    description: str
    regulation_type: str  # e.g., 'environmental', 'zoning', 'safety'
    issuing_authority: str
    effective_date: datetime.date
    applicable_jurisdictions: list[str] = field(default_factory=list)
    parent_regulation_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    expiration_date: datetime.date | None = None
    amendment_date: datetime.date | None = None
    reference_code: str | None = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    code: str | None = None
    category: str | None = None
    superseded_regulation_id: str | None = None
    source_url: str | None = None
    tags: list[str] | None = None

    def __eq__(self, other: object) -> bool:
        """Check equality based on the unique regulation ID."""
        if not isinstance(other, Regulation):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Generate hash based on the unique regulation ID."""
        return hash(self.id)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        regulation_type: str,
        issuing_authority: str,
        effective_date: datetime.date,
        applicable_jurisdictions: list[str] | None = None,
        parent_regulation_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        expiration_date: datetime.date | None = None,
        amendment_date: datetime.date | None = None,
        reference_code: str | None = None,
    ) -> "Regulation":
        """
        Create a new Regulation with a generated UUID.

        Args:
            name: Name of the regulation
            description: Description of the regulation
            regulation_type: Type of regulation
            issuing_authority: Authority that issued the regulation
            effective_date: Date when the regulation became effective
            applicable_jurisdictions: List of jurisdiction IDs where the regulation applies
            parent_regulation_id: Optional ID of parent regulation
            attributes: Dictionary of additional attributes
            expiration_date: Optional date when the regulation expires
            amendment_date: Optional date when the regulation was last amended
            reference_code: Optional reference code for the regulation

        Returns:
            A new Regulation instance
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            regulation_type=regulation_type,
            issuing_authority=issuing_authority,
            effective_date=effective_date,
            applicable_jurisdictions=applicable_jurisdictions or [],
            parent_regulation_id=parent_regulation_id,
            attributes=attributes or {},
            expiration_date=expiration_date,
            amendment_date=amendment_date,
            reference_code=reference_code,
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the regulation.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def add_jurisdiction(self, jurisdiction_id: str) -> None:
        """
        Add an applicable jurisdiction to the regulation.

        Args:
            jurisdiction_id: ID of the jurisdiction to add
        """
        if jurisdiction_id not in self.applicable_jurisdictions:
            self.applicable_jurisdictions.append(jurisdiction_id)
            self.updated_at = datetime.datetime.now()

    def remove_jurisdiction(self, jurisdiction_id: str) -> None:
        """
        Remove an applicable jurisdiction from the regulation.

        Args:
            jurisdiction_id: ID of the jurisdiction to remove
        """
        if jurisdiction_id in self.applicable_jurisdictions:
            self.applicable_jurisdictions.remove(jurisdiction_id)
            self.updated_at = datetime.datetime.now()

    def is_active(self, reference_date: datetime.date | None = None) -> bool:
        """
        Check if the regulation is active as of the reference date.

        Args:
            reference_date: Date to check against (defaults to current date)

        Returns:
            True if the regulation is active, False otherwise
        """
        if reference_date is None:
            reference_date = datetime.date.today()

        is_effective = self.effective_date <= reference_date
        is_not_expired = (
            self.expiration_date is None or reference_date <= self.expiration_date
        )

        return is_effective and is_not_expired

    def amend(
        self, new_description: str, amendment_date: datetime.date | None = None
    ) -> None:
        """
        Amend the regulation with a new description.

        Args:
            new_description: New description for the regulation
            amendment_date: Date of the amendment (defaults to current date)
        """
        self.description = new_description
        self.amendment_date = amendment_date or datetime.date.today()
        self.updated_at = datetime.datetime.now()


@dataclass
class RegulatoryFramework:
    """
    A class representing a collection of related regulations.

    Regulatory frameworks group related regulations under a common purpose,
    typically enforced by the same authority or addressing the same domain.
    """

    id: str
    name: str
    description: str
    domain: str = ""  # e.g., 'environment', 'urban planning', 'finance'
    issuing_authority: str = ""
    regulations: list[str] = field(default_factory=list)  # List of regulation IDs
    attributes: dict[str, Any] = field(default_factory=dict)
    effective_date: datetime.date | None = None
    expiration_date: datetime.date | None = None
    version: str = "1.0"
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    authority: str | None = None
    sector: str | None = None
    regulation_ids: list[str] | None = None

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        domain: str,
        issuing_authority: str,
        regulations: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
        effective_date: datetime.date | None = None,
        expiration_date: datetime.date | None = None,
        version: str = "1.0",
    ) -> "RegulatoryFramework":
        """
        Create a new RegulatoryFramework with a generated UUID.

        Args:
            name: Name of the framework
            description: Description of the framework
            domain: Domain the framework applies to
            issuing_authority: Authority that issued the framework
            regulations: List of regulation IDs in the framework
            attributes: Dictionary of additional attributes
            effective_date: Optional date when the framework became effective
            expiration_date: Optional date when the framework expires
            version: Version of the framework

        Returns:
            A new RegulatoryFramework instance
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            domain=domain,
            issuing_authority=issuing_authority,
            regulations=regulations or [],
            attributes=attributes or {},
            effective_date=effective_date,
            expiration_date=expiration_date,
            version=version,
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the framework.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def add_regulation(self, regulation_id: str) -> None:
        """
        Add a regulation to the framework.

        Args:
            regulation_id: ID of the regulation to add
        """
        if regulation_id not in self.regulations:
            self.regulations.append(regulation_id)
            self.updated_at = datetime.datetime.now()

    def remove_regulation(self, regulation_id: str) -> None:
        """
        Remove a regulation from the framework.

        Args:
            regulation_id: ID of the regulation to remove
        """
        if regulation_id in self.regulations:
            self.regulations.remove(regulation_id)
            self.updated_at = datetime.datetime.now()

    def is_active(self, reference_date: datetime.date | None = None) -> bool:
        """
        Check if the framework is active as of the reference date.

        Args:
            reference_date: Date to check against (defaults to current date)

        Returns:
            True if the framework is active, False otherwise
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

    def update_version(self, new_version: str) -> None:
        """
        Update the framework's version.

        Args:
            new_version: New version string
        """
        self.version = new_version
        self.updated_at = datetime.datetime.now()
