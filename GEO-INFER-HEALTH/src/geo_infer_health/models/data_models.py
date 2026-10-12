from pydantic import BaseModel, Field, field_validator
from typing import Any
from datetime import datetime


class Location(BaseModel):
    """Represents a geographic location."""

    latitude: float | None = Field(..., description="Latitude of the location.")
    longitude: float | None = Field(..., description="Longitude of the location.")
    crs: str = Field(default="EPSG:4326", description="Coordinate Reference System.")

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, value: float | None) -> float | None:
        if value is not None and not -90 <= value <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        return value

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, value: float | None) -> float | None:
        if value is not None and not -180 <= value <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees.")
        return value


class HealthFacility(BaseModel):
    """Represents a healthcare facility."""

    facility_id: str = Field(..., description="Unique identifier for the facility.")
    name: str = Field(..., description="Name of the health facility.")
    facility_type: str = Field(
        ..., description="Type of health facility (e.g., hospital, clinic)."
    )
    location: Location = Field(..., description="Geographic location of the facility.")
    capacity: int | None = Field(
        default=None, description="Capacity of the facility (e.g., number of beds)."
    )
    services_offered: list[str] = Field(
        default_factory=list, description="List of services offered."
    )
    operating_hours: str | None = Field(
        default=None, description="Operating hours of the facility."
    )
    contact_info: dict[str, str] | None = Field(
        default=None, description="Contact information (e.g., phone, email)."
    )

    @field_validator("capacity")
    @classmethod
    def validate_capacity(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("Capacity must be non-negative.")
        return value


class DiseaseReport(BaseModel):
    """Represents a report of a disease case or observation."""

    report_id: str = Field(..., description="Unique identifier for the report.")
    disease_code: str = Field(
        ..., description="Standardized code for the disease (e.g., ICD-10)."
    )
    location: Location = Field(
        ..., description="Geographic location of the reported case."
    )
    report_date: datetime = Field(..., description="Date and time of the report.")
    case_count: int = Field(
        default=1, description="Number of cases in this report.", ge=1
    )
    source: str | None = Field(
        default=None, description="Source of the report (e.g., hospital, lab)."
    )
    demographics: dict[str, Any] | None = Field(
        default=None, description="Anonymized demographic data if available."
    )
    notes: str | None = Field(
        default=None, description="Additional notes for the report."
    )


class PopulationData(BaseModel):
    """Represents population data for a given area, potentially a polygon."""

    area_id: str = Field(
        ..., description="Identifier for the administrative or statistical area."
    )
    # geometry: Any # Would typically be a GeoJSON dict or a Shapely geometry object
    population_count: int = Field(..., description="Total population in the area.")
    age_distribution: dict[str, int] | None = Field(
        default=None, description="Population count by age group."
    )
    other_demographics: dict[str, Any] | None = Field(
        default=None, description="Other relevant demographic splits."
    )

    @field_validator("population_count")
    @classmethod
    def validate_population_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Population count must be non-negative.")
        return value


class EnvironmentalData(BaseModel):
    """Represents an environmental data point or raster summary."""

    data_id: str = Field(
        ..., description="Identifier for the environmental data record."
    )
    parameter_name: str = Field(
        ...,
        description="Name of the environmental parameter (e.g., PM2.5, Temperature).",
    )
    value: float = Field(..., description="Value of the environmental parameter.")
    unit: str = Field(..., description="Unit of the environmental parameter.")
    location: Location = Field(
        ..., description="Geographic location of the data point."
    )
    timestamp: datetime = Field(..., description="Timestamp of the data recording.")
    # For raster data, this might include cell_value, raster_id, band_info etc.
    # For vector data, it might be associated with a polygon geometry.
