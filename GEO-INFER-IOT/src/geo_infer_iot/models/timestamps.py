"""Aware UTC timestamps shared by sensor and network models."""

from datetime import UTC, datetime
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, field_validator, ValidationInfo
from geo_infer_time.core.timestamps import normalize_timestamp


def utc_now() -> datetime:
    """Return an aware wall-clock timestamp; durations use monotonic clocks."""
    return datetime.now(UTC)


class TimestampModel(BaseModel):
    """Validate timestamp fields during construction and assignment."""

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("*", mode="before")
    @classmethod
    def normalize_datetime_fields(cls, value: Any, info: ValidationInfo) -> Any:
        annotation = cls.model_fields[info.field_name].annotation
        origin, arguments = get_origin(annotation), get_args(annotation)
        scalar = annotation is datetime or (
            origin in (Union, UnionType) and datetime in arguments
        )
        timestamp_mapping = origin is dict and arguments == (str, datetime)
        if value is None or not (scalar or timestamp_mapping):
            return value
        try:
            if timestamp_mapping and isinstance(value, dict):
                return {key: normalize_timestamp(item) for key, item in value.items()}
            return normalize_timestamp(value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
