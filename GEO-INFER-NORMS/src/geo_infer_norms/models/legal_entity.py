"""
Legal entity models for representing entities and jurisdictions in regulatory frameworks.

This module provides data models for legal entities, jurisdictions, and their relationships
in the context of legal and regulatory frameworks.
"""

from dataclasses import dataclass, field
from typing import Any
import datetime
from shapely.geometry import Point, Polygon, MultiPolygon
import uuid


@dataclass
class LegalEntity:
    """
    A class representing a legal entity subject to regulations.

    Legal entities can be organizations, facilities, land parcels, or other
    entities that are governed by regulations.
    """

    id: str
    name: str
    entity_type: str  # e.g., 'organization', 'facility', 'parcel'
    jurisdiction_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    geometry: Polygon | None = None
    point_location: Point | None = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)

    @classmethod
    def create(
        cls,
        name: str,
        entity_type: str,
        jurisdiction_ids: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
        parent_id: str | None = None,
        geometry: Polygon | None = None,
        point_location: Point | None = None,
    ) -> "LegalEntity":
        """
        Create a new LegalEntity with a generated UUID.

        Args:
            name: Name of the legal entity
            entity_type: Type of entity (e.g., 'organization', 'facility', 'parcel')
            jurisdiction_ids: List of jurisdiction IDs this entity belongs to
            attributes: Dictionary of additional attributes
            parent_id: Optional ID of parent entity
            geometry: Optional Shapely polygon representing the entity's boundary
            point_location: Optional Shapely point representing the entity's location

        Returns:
            A new LegalEntity instance
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            entity_type=entity_type,
            jurisdiction_ids=jurisdiction_ids or [],
            attributes=attributes or {},
            parent_id=parent_id,
            geometry=geometry,
            point_location=point_location,
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the entity.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def add_jurisdiction(self, jurisdiction_id: str) -> None:
        """
        Add a jurisdiction to the entity.

        Args:
            jurisdiction_id: ID of the jurisdiction to add
        """
        if jurisdiction_id not in self.jurisdiction_ids:
            self.jurisdiction_ids.append(jurisdiction_id)
            self.updated_at = datetime.datetime.now()

    def remove_jurisdiction(self, jurisdiction_id: str) -> None:
        """
        Remove a jurisdiction from the entity.

        Args:
            jurisdiction_id: ID of the jurisdiction to remove
        """
        if jurisdiction_id in self.jurisdiction_ids:
            self.jurisdiction_ids.remove(jurisdiction_id)
            self.updated_at = datetime.datetime.now()

    def set_geometry(self, geometry: Polygon) -> None:
        """
        Set the entity's boundary geometry.

        Args:
            geometry: Shapely polygon representing the entity's boundary
        """
        self.geometry = geometry

        # Update point location to be the centroid of the polygon
        if geometry is not None:
            self.point_location = geometry.centroid

        self.updated_at = datetime.datetime.now()

    def set_point_location(self, point: Point) -> None:
        """
        Set the entity's point location.

        Args:
            point: Shapely point representing the entity's location
        """
        self.point_location = point
        self.updated_at = datetime.datetime.now()


@dataclass
class Jurisdiction:
    """
    A class representing a jurisdiction with legal authority.

    Jurisdictions can be countries, states, counties, cities, or other administrative
    divisions with legal authority to enforce regulations.
    """

    id: str
    name: str
    level: str  # e.g., 'federal', 'state', 'county', 'city'
    code: str | None = None
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    geometry: MultiPolygon | None = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    description: str | None = None

    @classmethod
    def create(
        cls,
        name: str,
        level: str,
        code: str | None = None,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        geometry: MultiPolygon | None = None,
    ) -> "Jurisdiction":
        """
        Create a new Jurisdiction with a generated UUID.

        Args:
            name: Name of the jurisdiction
            level: Level of jurisdiction (e.g., 'federal', 'state', 'county', 'city')
            code: Optional code (e.g., state code, county code)
            parent_id: Optional ID of parent jurisdiction
            attributes: Dictionary of additional attributes
            geometry: Optional Shapely multipolygon representing the jurisdiction's boundary

        Returns:
            A new Jurisdiction instance
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            level=level,
            code=code,
            parent_id=parent_id,
            attributes=attributes or {},
            geometry=geometry,
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the jurisdiction.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def set_geometry(self, geometry: MultiPolygon) -> None:
        """
        Set the jurisdiction's boundary geometry.

        Args:
            geometry: Shapely multipolygon representing the jurisdiction's boundary
        """
        self.geometry = geometry
        self.updated_at = datetime.datetime.now()

    def set_parent(self, parent_id: str) -> None:
        """
        Set the jurisdiction's parent.

        Args:
            parent_id: ID of the parent jurisdiction
        """
        self.parent_id = parent_id
        self.updated_at = datetime.datetime.now()

    def contains_point(self, point: Point) -> bool:
        """
        Check if the jurisdiction contains a point.

        Args:
            point: Shapely point to check

        Returns:
            True if the jurisdiction contains the point, False otherwise
        """
        if self.geometry is None:
            return False

        return bool(self.geometry.contains(point))

    def overlaps_with(self, other_geometry: MultiPolygon) -> bool:
        """
        Check if the jurisdiction overlaps with another geometry.

        Args:
            other_geometry: Shapely multipolygon to check against

        Returns:
            True if the jurisdiction overlaps with the geometry, False otherwise
        """
        if self.geometry is None:
            return False

        return bool(self.geometry.overlaps(other_geometry))
