"""
Validation utilities for GEO-INFER-COMMS.

This module provides comprehensive validation functions for geospatial data,
message content, and communication parameters to ensure data integrity
and system reliability.
"""

from __future__ import annotations
from typing import Any, Final
import re
from datetime import datetime, UTC


def validate_coordinates(longitude: Any, latitude: Any) -> bool:
    """
    Validate longitude and latitude coordinates.

    Args:
        longitude: Longitude value in degrees
        latitude: Latitude value in degrees

    Returns:
        True if coordinates are valid, False otherwise
    """
    try:
        # Check if numeric
        if not isinstance(longitude, (int, float)) or not isinstance(
            latitude, (int, float)
        ):
            return False

        # Check valid ranges
        if not (-180 <= longitude <= 180):
            return False

        if not (-90 <= latitude <= 90):
            return False

        return True
    except (TypeError, ValueError):
        return False


SUPPORTED_CRS: Final[dict[str, str]] = {
    "WGS84": "EPSG:4326",  # World Geodetic System 1984
    "UTM": "UTM",  # Universal Transverse Mercator
    "WEB_MERCATOR": "EPSG:3857",  # Web Mercator (used by most web maps)
    "LOCAL": "LOCAL",  # Local coordinate system
}
"""Supported coordinate reference systems, keyed by short name.

Single source of truth for CRS validation; ``models.spatial.CoordinateSystem``
derives its constants from this mapping.
"""


def validate_crs(crs: str) -> bool:
    """
    Validate coordinate reference system string.

    Args:
        crs: Coordinate reference system identifier

    Returns:
        True if CRS is valid, False otherwise
    """
    return crs in SUPPORTED_CRS.values()


def validate_email(email: Any) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if email format is valid, False otherwise
    """
    if not isinstance(email, str):
        return False

    # Basic email regex pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    return bool(re.match(pattern, email))


def validate_phone(phone: Any) -> bool:
    """
    Validate phone number format.

    Args:
        phone: Phone number to validate

    Returns:
        True if phone format is valid, False otherwise
    """
    if not isinstance(phone, str):
        return False

    # Remove common separators
    cleaned = re.sub(r"[\s\-\(\)\.]+", "", phone)

    # Check if all digits and reasonable length
    if not cleaned.isdigit():
        return False

    if not (7 <= len(cleaned) <= 15):
        return False

    return True


def validate_message_content(content: Any, max_length: int = 10000) -> bool:
    """
    Validate message content.

    Args:
        content: Message content to validate
        max_length: Maximum allowed length

    Returns:
        True if content is valid, False otherwise
    """
    if not isinstance(content, str):
        return False

    if not content.strip():
        return False

    if len(content) > max_length:
        return False

    return True


def validate_message_priority(priority: str) -> bool:
    """
    Validate message priority level.

    Args:
        priority: Priority level to validate

    Returns:
        True if priority is valid, False otherwise
    """
    valid_priorities = ["low", "normal", "high", "urgent"]
    return priority.lower() in valid_priorities


def validate_message_type(message_type: str) -> bool:
    """
    Validate message type.

    Args:
        message_type: Message type to validate

    Returns:
        True if message type is valid, False otherwise
    """
    valid_types = [
        "text",
        "image",
        "file",
        "location",
        "alert",
        "sensor_data",
        "command",
        "status",
    ]
    return message_type.lower() in valid_types


def validate_user_id(user_id: Any) -> bool:
    """
    Validate user identifier format.

    Args:
        user_id: User ID to validate

    Returns:
        True if user ID is valid, False otherwise
    """
    if not isinstance(user_id, str):
        return False

    if not user_id.strip():
        return False

    # Allow alphanumeric, hyphens, underscores, and dots
    if not re.match(r"^[a-zA-Z0-9._-]+$", user_id):
        return False

    if len(user_id) > 100:
        return False

    return True


def validate_channel_id(channel_id: str) -> bool:
    """
    Validate channel identifier format.

    Args:
        channel_id: Channel ID to validate

    Returns:
        True if channel ID is valid, False otherwise
    """
    return validate_user_id(channel_id)  # Same format as user ID


def validate_spatial_bounds(bounds: Any) -> bool:
    """
    Validate spatial bounds structure.

    Args:
        bounds: Spatial bounds dictionary to validate

    Returns:
        True if bounds are valid, False otherwise
    """
    if not isinstance(bounds, dict):
        return False

    required_fields = ["min_longitude", "min_latitude", "max_longitude", "max_latitude"]
    if not all(field in bounds for field in required_fields):
        return False

    try:
        # Check coordinate values
        min_lon = float(bounds["min_longitude"])
        min_lat = float(bounds["min_latitude"])
        max_lon = float(bounds["max_longitude"])
        max_lat = float(bounds["max_latitude"])

        if not validate_coordinates(min_lon, min_lat):
            return False

        if not validate_coordinates(max_lon, max_lat):
            return False

        # Check bounds logic
        if min_lon >= max_lon:
            return False

        if min_lat >= max_lat:
            return False

        return True
    except (ValueError, TypeError, KeyError):
        return False


def validate_geojson_feature(feature: Any) -> bool:
    """
    Validate GeoJSON Feature structure.

    Args:
        feature: GeoJSON Feature to validate

    Returns:
        True if feature is valid, False otherwise
    """
    if not isinstance(feature, dict):
        return False

    if feature.get("type") != "Feature":
        return False

    # Check for geometry
    if "geometry" not in feature:
        return False

    geometry = feature["geometry"]
    if not validate_geojson_geometry(geometry):
        return False

    # Properties are optional but if present should be dict
    properties = feature.get("properties")
    if properties is not None and not isinstance(properties, dict):
        return False

    return True


def validate_geojson_geometry(geometry: Any) -> bool:
    """
    Validate GeoJSON Geometry structure.

    Args:
        geometry: GeoJSON Geometry to validate

    Returns:
        True if geometry is valid, False otherwise
    """
    if not isinstance(geometry, dict):
        return False

    required_fields = ["type", "coordinates"]
    if not all(field in geometry for field in required_fields):
        return False

    geom_type = geometry["type"]
    valid_types = [
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
        "GeometryCollection",
    ]

    if geom_type not in valid_types:
        return False

    coordinates = geometry["coordinates"]
    if not isinstance(coordinates, list):
        return False

    # Basic structure validation based on geometry type
    if geom_type == "Point":
        return len(coordinates) >= 2 and all(
            isinstance(coord, (int, float)) for coord in coordinates[:2]
        )
    elif geom_type == "LineString":
        return len(coordinates) >= 2 and all(
            isinstance(coord, list) and len(coord) >= 2 for coord in coordinates
        )
    elif geom_type == "Polygon":
        return len(coordinates) >= 1 and all(
            isinstance(ring, list) and len(ring) >= 4 for ring in coordinates
        )
    else:
        # For other types, basic list validation
        return True


def validate_notification_type(notification_type: str) -> bool:
    """
    Validate notification type.

    Args:
        notification_type: Notification type to validate

    Returns:
        True if notification type is valid, False otherwise
    """
    valid_types = ["info", "warning", "error", "success", "reminder"]
    return notification_type.lower() in valid_types


VALID_DELIVERY_METHODS = ["in_app", "email", "sms", "push", "websocket"]

VALID_EVENT_TYPES = [
    "data_update",
    "system_alert",
    "user_action",
    "sensor_trigger",
    "geospatial_change",
]


def validate_delivery_methods(methods: Any) -> bool:
    """
    Validate notification delivery methods.

    Args:
        methods: List of delivery methods to validate

    Returns:
        True if all methods are valid, False otherwise
    """
    if not isinstance(methods, list):
        return False

    valid_methods = VALID_DELIVERY_METHODS
    return all(method in valid_methods for method in methods)


def validate_event_type(event_type: str) -> bool:
    """
    Validate event type.

    Args:
        event_type: Event type to validate

    Returns:
        True if event type is valid, False otherwise
    """
    valid_types = VALID_EVENT_TYPES
    return event_type.lower() in valid_types


def validate_timestamp(timestamp: Any) -> bool:
    """
    Validate timestamp format and value.

    Args:
        timestamp: Timestamp to validate (string or datetime object)

    Returns:
        True if timestamp is valid, False otherwise
    """
    try:
        if isinstance(timestamp, str):
            # Try to parse ISO format
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif isinstance(timestamp, datetime):
            # Check if not too far in future or past
            now = datetime.now(UTC)
            if abs((timestamp - now).total_seconds()) > 31536000:  # 1 year
                return False
        else:
            return False

        return True
    except (ValueError, TypeError):
        return False


def validate_url(url: Any) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if URL format is valid, False otherwise
    """
    if not isinstance(url, str):
        return False

    # Basic URL regex pattern
    pattern = r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?$"

    return bool(re.match(pattern, url))


def validate_file_size(size_bytes: int, max_size_mb: float = 10.0) -> bool:
    """
    Validate file size.

    Args:
        size_bytes: File size in bytes
        max_size_mb: Maximum allowed size in MB

    Returns:
        True if file size is within limits, False otherwise
    """
    if not isinstance(size_bytes, int) or size_bytes < 0:
        return False

    max_bytes = max_size_mb * 1024 * 1024
    return size_bytes <= max_bytes


def validate_message_recipients(recipients: Any) -> bool:
    """
    Validate list of message recipients.

    Args:
        recipients: List of recipient IDs to validate

    Returns:
        True if all recipients are valid, False otherwise
    """
    if not isinstance(recipients, list):
        return False

    if not recipients:
        return False

    if len(recipients) > 1000:  # Reasonable limit
        return False

    return all(validate_user_id(recipient) for recipient in recipients)


def validate_spatial_filter(filter_config: Any) -> bool:
    """
    Validate spatial filter configuration.

    Args:
        filter_config: Spatial filter configuration to validate

    Returns:
        True if filter config is valid, False otherwise
    """
    if not isinstance(filter_config, dict):
        return False

    required_fields = ["filter_type", "parameters"]
    if not all(field in filter_config for field in required_fields):
        return False

    filter_type = filter_config["filter_type"]
    valid_types = ["bounds", "radius", "polygon", "proximity"]
    if filter_type not in valid_types:
        return False

    parameters = filter_config["parameters"]
    if not isinstance(parameters, dict):
        return False

    # Validate based on filter type
    if filter_type == "bounds":
        return validate_spatial_bounds(parameters.get("bounds", {}))
    elif filter_type == "radius":
        return (
            "center" in parameters
            and isinstance(parameters.get("radius_meters"), (int, float))
            and parameters["radius_meters"] > 0
        )
    elif filter_type == "polygon":
        return "polygon" in parameters
    elif filter_type == "proximity":
        return "target_location" in parameters and isinstance(
            parameters.get("max_distance_meters"), (int, float)
        )

    return True


def validate_collaboration_session(session_config: Any) -> bool:
    """
    Validate collaboration session configuration.

    Args:
        session_config: Session configuration to validate

    Returns:
        True if session config is valid, False otherwise
    """
    if not isinstance(session_config, dict):
        return False

    required_fields = ["name", "session_type", "participants"]
    if not all(field in session_config for field in required_fields):
        return False

    # Validate name
    name = session_config["name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        return False

    # Validate session type
    session_type = session_config["session_type"]
    valid_types = ["meeting", "workshop", "planning", "review"]
    if session_type not in valid_types:
        return False

    # Validate participants
    participants = session_config["participants"]
    if not isinstance(participants, list) or not participants:
        return False

    if not all(validate_user_id(participant) for participant in participants):
        return False

    # Validate duration if provided
    duration = session_config.get("duration")
    if duration is not None:
        if not isinstance(duration, int) or not (1 <= duration <= 480):
            return False

    # Validate features if provided
    features = session_config.get("features", [])
    if features:
        if not isinstance(features, list):
            return False
        valid_features = ["screen_share", "whiteboard", "file_share", "voice", "video"]
        if not all(feature in valid_features for feature in features):
            return False

    return True


def validate_stream_config(stream_config: Any) -> bool:
    """
    Validate data stream configuration.

    Args:
        stream_config: Stream configuration to validate

    Returns:
        True if stream config is valid, False otherwise
    """
    if not isinstance(stream_config, dict):
        return False

    required_fields = ["name", "stream_type"]
    if not all(field in stream_config for field in required_fields):
        return False

    # Validate name
    name = stream_config["name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        return False

    # Validate stream type
    stream_type = stream_config["stream_type"]
    valid_types = ["data", "video", "audio", "geospatial", "sensor"]
    if stream_type not in valid_types:
        return False

    # Validate geospatial filter if provided
    geospatial_filter = stream_config.get("geospatial_filter")
    if geospatial_filter is not None:
        if not validate_spatial_filter(geospatial_filter):
            return False

    return True


def sanitize_message_content(content: Any) -> str:
    """
    Sanitize message content to prevent XSS and other issues.

    Args:
        content: Raw message content

    Returns:
        Sanitized content
    """
    if not isinstance(content, str):
        return ""

    # Basic sanitization - remove potentially dangerous characters
    # In production, would use a proper HTML sanitization library
    dangerous_patterns = [
        r"<script[^>]*>.*?</script>",
        r"<iframe[^>]*>.*?</iframe>",
        r"javascript:",
        r"vbscript:",
        r"on\w+\s*=",
    ]

    sanitized = content
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE | re.DOTALL)

    return sanitized.strip()


def validate_and_sanitize_inputs(**kwargs: Any) -> dict[str, Any]:
    """
    Validate and sanitize multiple input parameters.

    Args:
        **kwargs: Input parameters to validate

    Returns:
        Dictionary of validated and sanitized inputs

    Raises:
        ValueError: If any input is invalid
    """
    results = {}

    for key, value in kwargs.items():
        if key.endswith("_content") or key in ["message", "description", "title"]:
            results[key] = sanitize_message_content(value)
        elif key.endswith("_email"):
            if not validate_email(value):
                raise ValueError(f"Invalid email format: {value}")
            results[key] = value.lower().strip()
        elif key.endswith("_phone"):
            if not validate_phone(value):
                raise ValueError(f"Invalid phone format: {value}")
            results[key] = value
        elif key in ["user_id", "channel_id", "participant_id"]:
            if not validate_user_id(value):
                raise ValueError(f"Invalid user/channel ID: {value}")
            results[key] = value
        else:
            results[key] = value

    return results


def validate_configuration(config: Any, required_keys: list[str]) -> bool:
    """
    Validate configuration dictionary against required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required configuration keys

    Returns:
        True if all required keys are present and valid, False otherwise
    """
    if not isinstance(config, dict):
        return False

    return all(key in config for key in required_keys)
