"""
Agricultural API Resource Classes

RESTful resource abstractions for fields, crops, and yield data.
Each resource encapsulates domain-specific query, filtering, and
data retrieval logic for agricultural geospatial entities.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResourceResponse:
    """Standard response wrapper for resource operations."""

    data: Any
    count: int
    resource_type: str
    filters_applied: dict[str, Any] = field(default_factory=dict)


class FieldsResource:
    """
    Resource for managing agricultural field entities.

    Provides CRUD-like access to field records including
    boundaries, soil characteristics, and management history.
    """

    def __init__(self) -> None:
        self._fields: dict[str, dict[str, Any]] = {}

    def create(
        self,
        name: str,
        area_hectares: float,
        location: dict[str, float],
        soil_type: str = "loam",
        crop_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Register a new agricultural field.

        Args:
            name: Human-readable field name.
            area_hectares: Field area in hectares (must be > 0).
            location: Dict with 'lat' and 'lon' keys.
            soil_type: Dominant soil type.
            crop_type: Currently planted crop, if any.
            metadata: Arbitrary extra attributes.

        Returns:
            The created field record including generated id.

        Raises:
            ValueError: If area_hectares <= 0 or location missing keys.
        """
        if area_hectares <= 0:
            raise ValueError("area_hectares must be positive")
        if "lat" not in location or "lon" not in location:
            raise ValueError("location must contain 'lat' and 'lon'")

        field_id = str(uuid.uuid4())[:8]
        record: dict[str, Any] = {
            "id": field_id,
            "name": name,
            "area_hectares": area_hectares,
            "location": location,
            "soil_type": soil_type,
            "crop_type": crop_type,
            "metadata": metadata or {},
        }
        self._fields[field_id] = record
        logger.info("Created field %s (%s)", field_id, name)
        return record

    def get(self, field_id: str) -> dict[str, Any] | None:
        """Return a single field by id, or None if not found."""
        return self._fields.get(field_id)

    def list(
        self,
        soil_type: str | None = None,
        crop_type: str | None = None,
        min_area: float | None = None,
    ) -> ResourceResponse:
        """
        List fields with optional filters.

        Args:
            soil_type: Filter by soil type.
            crop_type: Filter by planted crop.
            min_area: Minimum area in hectares.

        Returns:
            ResourceResponse with matching fields.
        """
        results = list(self._fields.values())
        filters: dict[str, Any] = {}

        if soil_type is not None:
            results = [f for f in results if f["soil_type"] == soil_type]
            filters["soil_type"] = soil_type
        if crop_type is not None:
            results = [f for f in results if f["crop_type"] == crop_type]
            filters["crop_type"] = crop_type
        if min_area is not None:
            results = [f for f in results if f["area_hectares"] >= min_area]
            filters["min_area"] = min_area

        return ResourceResponse(
            data=results,
            count=len(results),
            resource_type="fields",
            filters_applied=filters,
        )

    def update(self, field_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """
        Update a field record in place.

        Args:
            field_id: The field to update.
            updates: Key-value pairs to merge into the record.

        Returns:
            Updated record, or None if field_id not found.
        """
        record = self._fields.get(field_id)
        if record is None:
            return None
        for key, value in updates.items():
            if key != "id":
                record[key] = value
        logger.info("Updated field %s", field_id)
        return record

    def delete(self, field_id: str) -> bool:
        """Remove a field. Returns True if deleted, False if not found."""
        if field_id in self._fields:
            del self._fields[field_id]
            logger.info("Deleted field %s", field_id)
            return True
        return False


class CropsResource:
    """
    Resource for crop type reference data and seasonal parameters.

    Stores crop-specific agronomic coefficients used across
    yield prediction, water usage, and carbon models.
    """

    # Built-in crop reference data
    _CROP_DATABASE: dict[str, dict[str, Any]] = {
        "corn": {
            "scientific_name": "Zea mays",
            "growing_season_days": 120,
            "optimal_temp_min": 18.0,
            "optimal_temp_max": 32.0,
            "water_requirement_mm": 500,
            "base_yield_tonnes_ha": 9.5,
            "category": "cereal",
        },
        "wheat": {
            "scientific_name": "Triticum aestivum",
            "growing_season_days": 150,
            "optimal_temp_min": 12.0,
            "optimal_temp_max": 25.0,
            "water_requirement_mm": 450,
            "base_yield_tonnes_ha": 3.5,
            "category": "cereal",
        },
        "rice": {
            "scientific_name": "Oryza sativa",
            "growing_season_days": 130,
            "optimal_temp_min": 22.0,
            "optimal_temp_max": 35.0,
            "water_requirement_mm": 1200,
            "base_yield_tonnes_ha": 4.5,
            "category": "cereal",
        },
        "soybean": {
            "scientific_name": "Glycine max",
            "growing_season_days": 100,
            "optimal_temp_min": 20.0,
            "optimal_temp_max": 30.0,
            "water_requirement_mm": 450,
            "base_yield_tonnes_ha": 2.8,
            "category": "legume",
        },
        "cotton": {
            "scientific_name": "Gossypium hirsutum",
            "growing_season_days": 160,
            "optimal_temp_min": 20.0,
            "optimal_temp_max": 35.0,
            "water_requirement_mm": 700,
            "base_yield_tonnes_ha": 1.8,
            "category": "fiber",
        },
    }

    def __init__(self) -> None:
        self._custom_crops: dict[str, dict[str, Any]] = {}

    def get(self, crop_name: str) -> dict[str, Any] | None:
        """
        Get crop reference data by name (case-insensitive).

        Returns:
            Crop data dict or None if not found.
        """
        key = crop_name.lower()
        if key in self._CROP_DATABASE:
            return {"name": key, **self._CROP_DATABASE[key]}
        if key in self._custom_crops:
            return {"name": key, **self._custom_crops[key]}
        return None

    def list(self, category: str | None = None) -> ResourceResponse:
        """
        List all known crops, optionally filtered by category.

        Args:
            category: Filter crops by category (cereal, legume, fiber, etc.).

        Returns:
            ResourceResponse with crop records.
        """
        all_crops: list[dict[str, Any]] = []
        for name, data in {**self._CROP_DATABASE, **self._custom_crops}.items():
            all_crops.append({"name": name, **data})

        filters: dict[str, Any] = {}
        if category is not None:
            all_crops = [c for c in all_crops if c.get("category") == category]
            filters["category"] = category

        return ResourceResponse(
            data=all_crops,
            count=len(all_crops),
            resource_type="crops",
            filters_applied=filters,
        )

    def register(self, crop_name: str, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Register a custom crop type.

        Args:
            crop_name: Crop identifier (lowercased internally).
            properties: Agronomic properties for the crop.

        Returns:
            The registered crop record.

        Raises:
            ValueError: If crop_name is empty.
        """
        if not crop_name or not crop_name.strip():
            raise ValueError("crop_name must be non-empty")
        key = crop_name.lower().strip()
        self._custom_crops[key] = properties
        logger.info("Registered custom crop: %s", key)
        return {"name": key, **properties}

    def get_water_requirement(self, crop_name: str) -> float | None:
        """Return water requirement in mm for a crop, or None if unknown."""
        record = self.get(crop_name)
        if record is None:
            return None
        return record.get("water_requirement_mm")

    def get_optimal_temperature_range(self, crop_name: str) -> dict[str, float] | None:
        """Return optimal temperature range dict or None if unknown."""
        record = self.get(crop_name)
        if record is None:
            return None
        return {
            "min": record.get("optimal_temp_min", 0.0),
            "max": record.get("optimal_temp_max", 0.0),
        }


class YieldResource:
    """
    Resource for yield estimation and historical yield records.

    Combines soil, weather, and crop parameters to produce
    yield estimates and stores historical observations.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def estimate(
        self,
        crop_name: str,
        area_hectares: float,
        soil_quality: float = 1.0,
        weather_factor: float = 1.0,
    ) -> dict[str, Any]:
        """
        Estimate yield for a crop under given conditions.

        Args:
            crop_name: Crop to estimate for.
            area_hectares: Planted area in hectares.
            soil_quality: Multiplier [0-2] for soil conditions (1.0 = average).
            weather_factor: Multiplier [0-2] for weather conditions (1.0 = average).

        Returns:
            Yield estimate dict with total_tonnes, yield_per_hectare, and confidence.

        Raises:
            ValueError: If area_hectares <= 0 or multipliers out of range.
        """
        if area_hectares <= 0:
            raise ValueError("area_hectares must be positive")
        if not (0.0 <= soil_quality <= 2.0):
            raise ValueError("soil_quality must be between 0 and 2")
        if not (0.0 <= weather_factor <= 2.0):
            raise ValueError("weather_factor must be between 0 and 2")

        crops = CropsResource()
        crop_data = crops.get(crop_name)
        if crop_data is None:
            base_yield = 5.0  # default for unknown crops
        else:
            base_yield = crop_data.get("base_yield_tonnes_ha", 5.0)

        adjusted_yield = base_yield * soil_quality * weather_factor
        total = adjusted_yield * area_hectares

        # Confidence decreases as factors deviate from 1.0
        deviation = abs(soil_quality - 1.0) + abs(weather_factor - 1.0)
        confidence = max(0.3, 1.0 - deviation * 0.25)

        return {
            "crop": crop_name,
            "area_hectares": area_hectares,
            "yield_per_hectare": round(adjusted_yield, 2),
            "total_tonnes": round(total, 2),
            "soil_quality": soil_quality,
            "weather_factor": weather_factor,
            "confidence": round(confidence, 2),
        }

    def record_observation(
        self,
        crop_name: str,
        area_hectares: float,
        actual_yield_tonnes: float,
        season: str,
        year: int,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Record an actual yield observation for historical tracking.

        Args:
            crop_name: Crop harvested.
            area_hectares: Area harvested.
            actual_yield_tonnes: Measured total yield in tonnes.
            season: Growing season identifier.
            year: Harvest year.
            notes: Free-text notes.

        Returns:
            The stored observation record.
        """
        record_id = str(uuid.uuid4())[:8]
        observation: dict[str, Any] = {
            "id": record_id,
            "crop": crop_name,
            "area_hectares": area_hectares,
            "actual_yield_tonnes": actual_yield_tonnes,
            "yield_per_hectare": round(actual_yield_tonnes / area_hectares, 2)
            if area_hectares > 0
            else 0.0,
            "season": season,
            "year": year,
            "notes": notes,
        }
        self._records.append(observation)
        logger.info("Recorded yield observation %s for %s", record_id, crop_name)
        return observation

    def get_history(
        self,
        crop_name: str | None = None,
        year: int | None = None,
    ) -> ResourceResponse:
        """
        Query historical yield observations.

        Args:
            crop_name: Filter by crop.
            year: Filter by harvest year.

        Returns:
            ResourceResponse with matching observations.
        """
        results = list(self._records)
        filters: dict[str, Any] = {}

        if crop_name is not None:
            results = [r for r in results if r["crop"] == crop_name]
            filters["crop"] = crop_name
        if year is not None:
            results = [r for r in results if r["year"] == year]
            filters["year"] = year

        return ResourceResponse(
            data=results,
            count=len(results),
            resource_type="yield_observations",
            filters_applied=filters,
        )

    def average_yield(self, crop_name: str) -> float | None:
        """
        Compute average historical yield per hectare for a crop.

        Returns:
            Average yield in tonnes/hectare, or None if no records exist.
        """
        matching = [r for r in self._records if r["crop"] == crop_name]
        if not matching:
            return None
        return round(
            sum(float(r["yield_per_hectare"]) for r in matching) / len(matching), 2
        )
