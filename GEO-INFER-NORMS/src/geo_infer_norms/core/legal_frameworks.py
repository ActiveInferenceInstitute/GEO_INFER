"""
Legal frameworks module for geospatial analysis of laws and regulations.

This module provides classes and functions for representing, analyzing, and
applying legal frameworks in geospatial contexts.
"""

import geopandas as gpd
import logging
from shapely.geometry import Point, Polygon, MultiPolygon

from ..models.legal_entity import Jurisdiction
from ..models.regulation import Regulation

logger = logging.getLogger(__name__)


class LegalFramework:
    """
    A class representing a comprehensive legal framework for geospatial analysis.

    This class manages collections of regulations, jurisdictions, and their
    spatial relationships for legal analysis in geographic contexts.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        jurisdictions: list[Jurisdiction] | None = None,
        regulations: list[Regulation] | None = None,
    ):
        """
        Initialize a LegalFramework instance.

        Args:
            name: Name of the legal framework
            description: Description of the legal framework
            jurisdictions: List of Jurisdiction objects
            regulations: List of Regulation objects
        """
        self.name = name
        self.description = description
        self.jurisdictions = jurisdictions or []
        self.regulations = regulations or []
        self._jurisdiction_index: dict[str, Jurisdiction] = {}
        self._regulation_index: dict[str, Regulation] = {}

        # Initialize indexes
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build spatial and attribute indexes for fast lookups."""
        # Index jurisdictions by ID
        self._jurisdiction_index = {j.id: j for j in self.jurisdictions}

        # Index regulations by ID
        self._regulation_index = {r.id: r for r in self.regulations}

    def add_jurisdiction(self, jurisdiction: Jurisdiction) -> None:
        """
        Add a jurisdiction to the legal framework.

        Args:
            jurisdiction: The Jurisdiction object to add
        """
        self.jurisdictions.append(jurisdiction)
        self._jurisdiction_index[jurisdiction.id] = jurisdiction
        logger.info(f"Added jurisdiction {jurisdiction.name} to framework {self.name}")

    def add_regulation(self, regulation: Regulation) -> None:
        """
        Add a regulation to the legal framework.

        Args:
            regulation: The Regulation object to add
        """
        self.regulations.append(regulation)
        self._regulation_index[regulation.id] = regulation
        logger.info(f"Added regulation {regulation.name} to framework {self.name}")

    def get_regulations_by_jurisdiction(self, jurisdiction_id: str) -> list[Regulation]:
        """
        Get all regulations applicable to a specific jurisdiction.

        Args:
            jurisdiction_id: The ID of the jurisdiction

        Returns:
            A list of applicable Regulation objects
        """
        if jurisdiction_id not in self._jurisdiction_index:
            logger.warning(f"Jurisdiction ID {jurisdiction_id} not found in framework")
            return []

        jurisdiction = self._jurisdiction_index[jurisdiction_id]
        applicable_regs = []

        for reg in self.regulations:
            if jurisdiction_id in reg.applicable_jurisdictions:
                applicable_regs.append(reg)
            elif (
                jurisdiction.parent_id
                and jurisdiction.parent_id in reg.applicable_jurisdictions
            ):
                applicable_regs.append(reg)

        return applicable_regs

    def get_jurisdictions_by_point(self, point: Point) -> list[Jurisdiction]:
        """
        Get all jurisdictions that contain a specific geographic point.

        Args:
            point: A Shapely Point geometry

        Returns:
            A list of Jurisdiction objects that contain the point
        """
        containing_jurisdictions = []

        for jurisdiction in self.jurisdictions:
            if jurisdiction.geometry is not None and jurisdiction.geometry.contains(
                point
            ):
                containing_jurisdictions.append(jurisdiction)

        return containing_jurisdictions

    def get_regulations_by_point(self, point: Point) -> list[Regulation]:
        """
        Get all regulations applicable to a specific geographic point.

        Args:
            point: A Shapely Point geometry

        Returns:
            A list of applicable Regulation objects
        """
        jurisdictions = self.get_jurisdictions_by_point(point)
        applicable_regs = set()

        for jurisdiction in jurisdictions:
            jur_regs = self.get_regulations_by_jurisdiction(jurisdiction.id)
            applicable_regs.update(jur_regs)

        return list(applicable_regs)

    def export_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Export the legal framework's jurisdictions to a GeoDataFrame.

        Returns:
            A GeoDataFrame containing jurisdiction geometries and properties
        """
        data = []

        for jurisdiction in self.jurisdictions:
            if jurisdiction.geometry is not None:
                jur_dict = {
                    "id": jurisdiction.id,
                    "name": jurisdiction.name,
                    "level": jurisdiction.level,
                    "parent_id": jurisdiction.parent_id,
                    "geometry": jurisdiction.geometry,
                    "regulation_count": len(
                        self.get_regulations_by_jurisdiction(jurisdiction.id)
                    ),
                }
                data.append(jur_dict)

        if not data:
            logger.warning(
                "No jurisdictions with geometry found for GeoDataFrame export"
            )
            return gpd.GeoDataFrame()

        return gpd.GeoDataFrame(data, crs="EPSG:4326")

    def __repr__(self) -> str:
        return f"LegalFramework(name='{self.name}', jurisdictions={len(self.jurisdictions)}, regulations={len(self.regulations)})"


class JurisdictionHandler:
    """
    Utility class for handling jurisdictional operations and hierarchies.

    This class provides methods for managing jurisdictional relationships,
    hierarchies, and boundary operations.
    """

    def __init__(self, jurisdictions: list[Jurisdiction] | None = None):
        """
        Initialize a JurisdictionHandler instance.

        Args:
            jurisdictions: List of Jurisdiction objects to manage
        """
        self.jurisdictions = jurisdictions or []
        self._jurisdiction_index = {j.id: j for j in self.jurisdictions}
        self._hierarchy_cache: dict[str, list[Jurisdiction]] = {}

    def add_jurisdiction(self, jurisdiction: Jurisdiction) -> None:
        """
        Add a jurisdiction to the handler.

        Args:
            jurisdiction: The Jurisdiction object to add
        """
        self.jurisdictions.append(jurisdiction)
        self._jurisdiction_index[jurisdiction.id] = jurisdiction
        # Clear hierarchy cache when adding new jurisdictions
        self._hierarchy_cache = {}

    def get_jurisdiction_by_id(self, jurisdiction_id: str) -> Jurisdiction | None:
        """
        Get a jurisdiction by its ID.

        Args:
            jurisdiction_id: The ID of the jurisdiction

        Returns:
            The Jurisdiction object or None if not found
        """
        return self._jurisdiction_index.get(jurisdiction_id)

    def get_jurisdiction_hierarchy(self, jurisdiction_id: str) -> list[Jurisdiction]:
        """
        Get the hierarchical chain of jurisdictions from the given one up to the root.

        Args:
            jurisdiction_id: The ID of the jurisdiction

        Returns:
            A list of Jurisdiction objects representing the hierarchy, starting with the given jurisdiction
        """
        # Check cache first
        if jurisdiction_id in self._hierarchy_cache:
            return self._hierarchy_cache[jurisdiction_id]

        hierarchy = []
        visited: set = set()
        current_id: str | None = jurisdiction_id

        while current_id:
            jurisdiction = self._jurisdiction_index.get(current_id)
            if not jurisdiction:
                break

            if current_id in visited:
                # Malformed data: a parent cycle would loop forever.
                break

            visited.add(current_id)
            hierarchy.append(jurisdiction)
            current_id = jurisdiction.parent_id

        # Cache the result
        self._hierarchy_cache[jurisdiction_id] = hierarchy
        return hierarchy

    def find_jurisdictions_by_name(
        self, name: str, partial_match: bool = False
    ) -> list[Jurisdiction]:
        """
        Find jurisdictions by name.

        Args:
            name: The name to search for
            partial_match: Whether to allow partial matching

        Returns:
            A list of matching Jurisdiction objects
        """
        matches = []

        for jurisdiction in self.jurisdictions:
            if partial_match and name.lower() in jurisdiction.name.lower():
                matches.append(jurisdiction)
            elif jurisdiction.name.lower() == name.lower():
                matches.append(jurisdiction)

        return matches

    def find_jurisdictions_at_level(self, level: str) -> list[Jurisdiction]:
        """
        Find all jurisdictions at a specific level (e.g., 'federal', 'state', 'county', 'city').

        Args:
            level: The level to search for

        Returns:
            A list of matching Jurisdiction objects
        """
        return [j for j in self.jurisdictions if j.level == level]

    def get_overlapping_jurisdictions(
        self, geometry: Point | Polygon | MultiPolygon
    ) -> list[Jurisdiction]:
        """
        Find all jurisdictions that overlap with a given geometry.

        Args:
            geometry: A Shapely geometry to check for overlaps

        Returns:
            A list of overlapping Jurisdiction objects
        """
        overlapping = []

        for jurisdiction in self.jurisdictions:
            if jurisdiction.geometry is None:
                continue

            if isinstance(geometry, Point):
                if jurisdiction.geometry.contains(geometry):
                    overlapping.append(jurisdiction)
            else:
                if jurisdiction.geometry.intersects(geometry):
                    overlapping.append(jurisdiction)

        return overlapping

    def create_jurisdiction_graph(self) -> dict[str, list[str]]:
        """
        Create a graph representation of the jurisdictional hierarchy.

        Returns:
            A dictionary mapping jurisdiction IDs to lists of child jurisdiction IDs
        """
        graph: dict[str, list[str]] = {j.id: [] for j in self.jurisdictions}

        for jurisdiction in self.jurisdictions:
            if jurisdiction.parent_id and jurisdiction.parent_id in graph:
                graph[jurisdiction.parent_id].append(jurisdiction.id)

        return graph

    def export_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Export the handler's jurisdictions to a GeoDataFrame.

        Returns:
            A GeoDataFrame containing jurisdiction geometries and properties
        """
        data = []

        for jurisdiction in self.jurisdictions:
            if jurisdiction.geometry is not None:
                jur_dict = {
                    "id": jurisdiction.id,
                    "name": jurisdiction.name,
                    "level": jurisdiction.level,
                    "parent_id": jurisdiction.parent_id,
                    "geometry": jurisdiction.geometry,
                }
                data.append(jur_dict)

        if not data:
            logger.warning(
                "No jurisdictions with geometry found for GeoDataFrame export"
            )
            return gpd.GeoDataFrame()

        return gpd.GeoDataFrame(data, crs="EPSG:4326")
