"""
Role-based access control for geospatial data.

This module implements RBAC specifically designed for protecting access
to sensitive geospatial information, including spatial and attribute-based
permissions.
"""

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, shape
import jwt
import datetime
import hashlib


class SpatialPermission:
    """Represents a spatial permission for accessing geographic areas."""

    def __init__(
        self,
        name: str,
        geometry: Polygon | MultiPolygon | None = None,
        wkt: str | None = None,
        geojson: dict | None = None,
        attributes: list[str] | None = None,
        max_resolution: int | None = None,
    ):
        """
        Initialize a spatial permission.

        Args:
            name: Name of the permission
            geometry: Shapely geometry object defining the permitted area
            wkt: WKT string representation of the geometry
            geojson: GeoJSON representation of the geometry
            attributes: List of attributes that can be accessed
            max_resolution: Maximum spatial resolution allowed (e.g., H3 resolution)
        """
        self.name = name
        self.attributes = attributes or []
        self.max_resolution = max_resolution

        # Set geometry from one of the provided formats
        if geometry is not None:
            self.geometry = geometry
        elif wkt is not None:
            from shapely import wkt as shapely_wkt

            self.geometry = shapely_wkt.loads(wkt)
        elif geojson is not None:
            self.geometry = shape(geojson)
        else:
            self.geometry = None

    def __repr__(self) -> str:
        """Return string representation of the permission."""
        return f"SpatialPermission({self.name}, attributes={self.attributes}, max_resolution={self.max_resolution})"

    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point is contained in the permitted area."""
        from shapely.geometry import Point

        if self.geometry is None:
            return True  # No spatial restriction
        point = Point(lon, lat)
        return bool(self.geometry.contains(point))

    def filter_geodataframe(
        self, gdf: gpd.GeoDataFrame, geometry_col: str = "geometry"
    ) -> gpd.GeoDataFrame:
        """Filter a GeoDataFrame to only include geometries within the permitted area."""
        if self.geometry is None:
            return gdf

        # Spatial filter
        filtered = gdf[gdf[geometry_col].within(self.geometry)]

        # Attribute filter if attributes are specified
        if self.attributes:
            cols_to_keep = [
                c for c in filtered.columns if c in self.attributes or c == geometry_col
            ]
            filtered = filtered[cols_to_keep]

        return filtered


class Role:
    """Represents a security role with associated permissions."""

    def __init__(self, name: str, permissions: list[SpatialPermission] | None = None):
        """
        Initialize a role with permissions.

        Args:
            name: Role name
            permissions: List of spatial permissions
        """
        self.name = name
        self.permissions = permissions or []

    def add_permission(self, permission: SpatialPermission) -> None:
        """Add a permission to the role."""
        self.permissions.append(permission)

    def has_permission(self, permission_name: str) -> bool:
        """Check if the role has a specific permission."""
        return any(p.name == permission_name for p in self.permissions)

    def get_accessible_area(self) -> Polygon | MultiPolygon | None:
        """Get the combined area of all spatial permissions."""
        from shapely.ops import unary_union

        if not self.permissions:
            return None

        geometries = [p.geometry for p in self.permissions if p.geometry is not None]
        if not geometries:
            return None

        return unary_union(geometries)


class GeospatialAccessManager:
    """Manages access control for geospatial data."""

    def __init__(self, secret_key: str):
        """
        Initialize the access manager.

        Args:
            secret_key: Secret key for JWT token generation/validation
        """
        self.secret_key = secret_key
        self._jwt_secret = hashlib.sha256(secret_key.encode("utf-8")).digest()
        self.roles: dict[str, Role] = {}
        self.user_roles: dict[str, list[str]] = {}

    def add_role(self, role: Role) -> None:
        """Add a role to the manager."""
        self.roles[role.name] = role

    def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """
        Assign a role to a user.

        Args:
            user_id: User identifier
            role_name: Name of the role to assign

        Returns:
            True if assignment was successful, False otherwise
        """
        if role_name not in self.roles:
            return False

        if user_id not in self.user_roles:
            self.user_roles[user_id] = []

        if role_name not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role_name)

        return True

    def get_user_roles(self, user_id: str) -> list[Role]:
        """Get all roles assigned to a user."""
        role_names = self.user_roles.get(user_id, [])
        return [self.roles[name] for name in role_names if name in self.roles]

    def generate_token(self, user_id: str, expiration_hours: int = 24) -> str:
        """
        Generate a JWT token for a user.

        Args:
            user_id: User identifier
            expiration_hours: Token validity in hours

        Returns:
            JWT token string
        """
        role_names = self.user_roles.get(user_id, [])

        payload = {
            "user_id": user_id,
            "roles": role_names,
            "exp": datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            + datetime.timedelta(hours=expiration_hours),
        }

        return jwt.encode(payload, self._jwt_secret, algorithm="HS256")

    def validate_token(self, token: str) -> dict | None:
        """
        Validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self._jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            # Token has expired
            return None
        except jwt.InvalidTokenError:
            # Invalid token
            return None

    def can_access_location(self, user_id: str, lat: float, lon: float) -> bool:
        """
        Check if a user can access data at a specific location.

        Args:
            user_id: User identifier
            lat: Latitude
            lon: Longitude

        Returns:
            True if user has access, False otherwise
        """
        user_roles = self.get_user_roles(user_id)

        for role in user_roles:
            for permission in role.permissions:
                if permission.contains_point(lat, lon):
                    return True

        return False

    def filter_geodataframe(
        self, user_id: str, gdf: gpd.GeoDataFrame, geometry_col: str = "geometry"
    ) -> gpd.GeoDataFrame:
        """
        Filter a GeoDataFrame based on user's spatial permissions.

        Args:
            user_id: User identifier
            gdf: GeoDataFrame to filter
            geometry_col: Name of the geometry column

        Returns:
            Filtered GeoDataFrame
        """
        user_roles = self.get_user_roles(user_id)

        if not user_roles:
            return gpd.GeoDataFrame(geometry=[])  # Empty GeoDataFrame

        # Get all accessible areas
        accessible_areas = []
        permitted_attributes = set()

        for role in user_roles:
            for permission in role.permissions:
                if permission.geometry is not None:
                    accessible_areas.append(permission.geometry)
                permitted_attributes.update(permission.attributes)

        if not accessible_areas:
            filtered_gdf = gdf.copy()
        else:
            # Combine all accessible areas
            from shapely.ops import unary_union

            combined_area = unary_union(accessible_areas)

            # Filter by area
            filtered_gdf = gdf[gdf[geometry_col].within(combined_area)]

        # Filter attributes if needed
        if permitted_attributes:
            cols_to_keep = [
                c
                for c in filtered_gdf.columns
                if c in permitted_attributes or c == geometry_col
            ]
            filtered_gdf = filtered_gdf[cols_to_keep]

        return filtered_gdf
