"""
Zoning models for representing zoning regulations and land use classifications.

This module provides data models for zoning codes, land use types, and zoning
districts used in urban planning and land use regulation.
"""

from dataclasses import dataclass, field
from typing import Any
import datetime
from shapely.geometry import Polygon
import uuid


@dataclass
class ZoningCode:
    """
    A class representing a zoning code or designation.

    Zoning codes define the allowed uses and development standards for areas
    within a jurisdiction's zoning ordinance.
    """

    code: str  # Primary identifier (e.g., "R-1", "C-2")
    name: str
    description: str
    category: str  # e.g., "residential", "commercial", "industrial", "mixed_use"
    jurisdiction_id: str
    allowed_uses: list[str] = field(default_factory=list)
    conditional_uses: list[str] = field(default_factory=list)
    prohibited_uses: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    max_height: float | None = None
    max_density: float | None = None
    min_lot_size: float | None = None
    max_lot_coverage: float | None = None
    max_floor_area_ratio: float = 0.0
    setbacks: Any = None
    environmental_requirements: Any = None

    @classmethod
    def create(
        cls,
        code: str,
        name: str,
        description: str,
        category: str,
        jurisdiction_id: str,
        allowed_uses: list[str] | None = None,
        conditional_uses: list[str] | None = None,
        prohibited_uses: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> "ZoningCode":
        """
        Create a new ZoningCode.

        Args:
            code: The code identifier (e.g., "R-1", "C-2")
            name: Name of the zoning code
            description: Description of the zoning code
            category: Category of the zoning code
            jurisdiction_id: ID of the jurisdiction where the code applies
            allowed_uses: List of allowed uses
            conditional_uses: List of conditionally allowed uses
            prohibited_uses: List of prohibited uses
            attributes: Dictionary of additional attributes

        Returns:
            A new ZoningCode instance
        """
        return cls(
            code=code,
            name=name,
            description=description,
            category=category,
            jurisdiction_id=jurisdiction_id,
            allowed_uses=allowed_uses or [],
            conditional_uses=conditional_uses or [],
            prohibited_uses=prohibited_uses or [],
            attributes=attributes or {},
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the zoning code.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def add_allowed_use(self, use: str) -> None:
        """
        Add an allowed use to the zoning code.

        Args:
            use: Use to add to allowed uses
        """
        if use not in self.allowed_uses:
            self.allowed_uses.append(use)

            # Remove from other categories if present
            if use in self.conditional_uses:
                self.conditional_uses.remove(use)
            if use in self.prohibited_uses:
                self.prohibited_uses.remove(use)

            self.updated_at = datetime.datetime.now()

    def add_conditional_use(self, use: str) -> None:
        """
        Add a conditional use to the zoning code.

        Args:
            use: Use to add to conditional uses
        """
        if use not in self.conditional_uses:
            self.conditional_uses.append(use)

            # Remove from prohibited if present (allowed takes precedence)
            if use in self.prohibited_uses:
                self.prohibited_uses.remove(use)

            self.updated_at = datetime.datetime.now()

    def add_prohibited_use(self, use: str) -> None:
        """
        Add a prohibited use to the zoning code.

        Args:
            use: Use to add to prohibited uses
        """
        if use not in self.prohibited_uses:
            self.prohibited_uses.append(use)

            # Remove from other categories if present
            if use in self.allowed_uses:
                self.allowed_uses.remove(use)
            if use in self.conditional_uses:
                self.conditional_uses.remove(use)

            self.updated_at = datetime.datetime.now()

    def is_use_allowed(self, use: str) -> bool:
        """
        Check if a use is allowed under this zoning code.

        Args:
            use: Use to check

        Returns:
            True if the use is allowed, False otherwise
        """
        return use in self.allowed_uses

    def is_use_conditional(self, use: str) -> bool:
        """
        Check if a use is conditionally allowed under this zoning code.

        Args:
            use: Use to check

        Returns:
            True if the use is conditionally allowed, False otherwise
        """
        return use in self.conditional_uses

    def is_use_prohibited(self, use: str) -> bool:
        """
        Check if a use is prohibited under this zoning code.

        Args:
            use: Use to check

        Returns:
            True if the use is prohibited, False otherwise
        """
        if use in self.prohibited_uses:
            return True

        # If explicitly allowed or conditional, it's not prohibited
        if use in self.allowed_uses or use in self.conditional_uses:
            return False

        # If the zoning code has explicit prohibited uses and this isn't one, it might be allowed
        # depending on how the jurisdiction interprets unlisted uses
        return len(self.prohibited_uses) > 0


@dataclass
class ZoningDistrict:
    """
    A class representing a geographic area with a specific zoning designation.

    Zoning districts are geographic areas that have been assigned a specific
    zoning code under a jurisdiction's zoning ordinance.
    """

    id: str
    name: str
    zoning_code: str  # Reference to a ZoningCode
    jurisdiction_id: str
    geometry: Polygon | None = None
    overlay_codes: list[str] = field(default_factory=list)  # Additional zoning overlays
    attributes: dict[str, Any] = field(default_factory=dict)
    effective_date: datetime.date | None = None
    expiration_date: datetime.date | None = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    description: str | None = None
    area: float = 0.0
    area_hectares: float = 0.0
    population: float = 0.0
    employment: float = 0.0

    @classmethod
    def create(
        cls,
        name: str,
        zoning_code: str,
        jurisdiction_id: str,
        geometry: Polygon | None = None,
        overlay_codes: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
        effective_date: datetime.date | None = None,
        expiration_date: datetime.date | None = None,
    ) -> "ZoningDistrict":
        """
        Create a new ZoningDistrict with a generated UUID.

        Args:
            name: Name of the zoning district
            zoning_code: Zoning code for the district
            jurisdiction_id: ID of the jurisdiction where the district is located
            geometry: Optional Shapely polygon representing the district's boundary
            overlay_codes: List of overlay zoning codes that apply to the district
            attributes: Dictionary of additional attributes
            effective_date: Optional date when the zoning became effective
            expiration_date: Optional date when the zoning expires

        Returns:
            A new ZoningDistrict instance
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            zoning_code=zoning_code,
            jurisdiction_id=jurisdiction_id,
            geometry=geometry,
            overlay_codes=overlay_codes or [],
            attributes=attributes or {},
            effective_date=effective_date,
            expiration_date=expiration_date,
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the zoning district.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def set_geometry(self, geometry: Polygon) -> None:
        """
        Set the district's boundary geometry.

        Args:
            geometry: Shapely polygon representing the district's boundary
        """
        self.geometry = geometry
        self.updated_at = datetime.datetime.now()

    def add_overlay_code(self, overlay_code: str) -> None:
        """
        Add an overlay zoning code to the district.

        Args:
            overlay_code: Overlay code to add
        """
        if overlay_code not in self.overlay_codes:
            self.overlay_codes.append(overlay_code)
            self.updated_at = datetime.datetime.now()

    def remove_overlay_code(self, overlay_code: str) -> None:
        """
        Remove an overlay zoning code from the district.

        Args:
            overlay_code: Overlay code to remove
        """
        if overlay_code in self.overlay_codes:
            self.overlay_codes.remove(overlay_code)
            self.updated_at = datetime.datetime.now()

    def is_active(self, reference_date: datetime.date | None = None) -> bool:
        """
        Check if the zoning district is active as of the reference date.

        Args:
            reference_date: Date to check against (defaults to current date)

        Returns:
            True if the district is active, False otherwise
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

    def change_zoning(self, new_zoning_code: str) -> None:
        """
        Change the zoning code for the district.

        Args:
            new_zoning_code: New zoning code to apply
        """
        self.zoning_code = new_zoning_code
        self.updated_at = datetime.datetime.now()


@dataclass
class LandUseType:
    """
    A class representing a type of land use.

    Land use types categorize how land is used (e.g., residential, commercial)
    and may have different regulatory requirements.
    """

    id: str
    name: str
    category: str  # e.g., "residential", "commercial", "industrial"
    subcategory: str | None = None
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    compatible_uses: list[str] = field(default_factory=list)
    incompatible_uses: list[str] = field(default_factory=list)
    typical_zoning_codes: list[str] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    intensity: float | None = None

    @classmethod
    def create(
        cls,
        name: str,
        category: str,
        subcategory: str | None = None,
        description: str = "",
        attributes: dict[str, Any] | None = None,
        compatible_uses: list[str] | None = None,
        incompatible_uses: list[str] | None = None,
        typical_zoning_codes: list[str] | None = None,
    ) -> "LandUseType":
        """
        Create a new LandUseType with a generated UUID.

        Args:
            name: Name of the land use type
            category: Category of the land use type
            subcategory: Optional subcategory
            description: Description of the land use type
            attributes: Dictionary of additional attributes
            compatible_uses: List of compatible use IDs
            incompatible_uses: List of incompatible use IDs
            typical_zoning_codes: List of typical zoning codes for this use

        Returns:
            A new LandUseType instance
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            category=category,
            subcategory=subcategory,
            description=description,
            attributes=attributes or {},
            compatible_uses=compatible_uses or [],
            incompatible_uses=incompatible_uses or [],
            typical_zoning_codes=typical_zoning_codes or [],
        )

    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update or add an attribute to the land use type.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
        self.updated_at = datetime.datetime.now()

    def add_compatible_use(self, use_id: str) -> None:
        """
        Add a compatible use to the land use type.

        Args:
            use_id: ID of the compatible use to add
        """
        if use_id not in self.compatible_uses:
            self.compatible_uses.append(use_id)

            # Remove from incompatible if present
            if use_id in self.incompatible_uses:
                self.incompatible_uses.remove(use_id)

            self.updated_at = datetime.datetime.now()

    def add_incompatible_use(self, use_id: str) -> None:
        """
        Add an incompatible use to the land use type.

        Args:
            use_id: ID of the incompatible use to add
        """
        if use_id not in self.incompatible_uses:
            self.incompatible_uses.append(use_id)

            # Remove from compatible if present
            if use_id in self.compatible_uses:
                self.compatible_uses.remove(use_id)

            self.updated_at = datetime.datetime.now()

    def add_typical_zoning_code(self, zoning_code: str) -> None:
        """
        Add a typical zoning code for this land use type.

        Args:
            zoning_code: Zoning code to add
        """
        if zoning_code not in self.typical_zoning_codes:
            self.typical_zoning_codes.append(zoning_code)
            self.updated_at = datetime.datetime.now()

    def is_compatible_with(self, use_id: str) -> bool | None:
        """
        Check if this land use type is compatible with another.

        Args:
            use_id: ID of the land use to check compatibility with

        Returns:
            True if compatible, False if incompatible, None if unspecified
        """
        if use_id in self.compatible_uses:
            return True
        if use_id in self.incompatible_uses:
            return False
        return None  # Compatibility not specified
