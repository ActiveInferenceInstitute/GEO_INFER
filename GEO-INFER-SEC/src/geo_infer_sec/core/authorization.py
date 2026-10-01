"""
Authorization framework for GEO-INFER-SEC.

This module provides RBAC (Role-Based Access Control) and ABAC
(Attribute-Based Access Control) for geospatial data access.
"""

import logging
from typing import Any
from enum import StrEnum

from .access_control import GeospatialAccessManager, Role, SpatialPermission

logger = logging.getLogger(__name__)


class PermissionType(StrEnum):
    """Types of permissions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


class AuthorizationManager:
    """
    Authorization manager implementing RBAC and ABAC.

    Provides role-based and attribute-based access control for
    geospatial resources with spatial and temporal constraints.
    """

    def __init__(
        self,
        access_manager: GeospatialAccessManager | None = None,
        secret_key: str | None = None,
    ) -> None:
        """
        Initialize the authorization manager.

        Args:
            access_manager: Optional GeospatialAccessManager instance
            secret_key: Secret key for access manager. If not provided, will raise error.
                       Must be set via environment variable or configuration.
        """
        import os

        if secret_key is None:
            secret_key = os.getenv("GEO_INFER_SEC_SECRET_KEY")
            if secret_key is None:
                raise ValueError(
                    "Secret key must be provided either as parameter or "
                    "via GEO_INFER_SEC_SECRET_KEY environment variable. "
                    "Never use default secrets in production!"
                )

        self.access_manager = access_manager or GeospatialAccessManager(
            secret_key=secret_key
        )

    def check_permission(
        self,
        user_id: str,
        resource: str,
        permission: PermissionType,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        """
        Check if a user has permission to perform an action on a resource.

        Args:
            user_id: User identifier
            resource: Resource identifier
            permission: Type of permission required
            attributes: Optional attributes for ABAC (e.g., spatial bounds, time)

        Returns:
            True if user has permission, False otherwise
        """
        user_roles = self.access_manager.get_user_roles(user_id)

        # Check role-based permissions
        for role in user_roles:
            if self._role_has_permission(role, resource, permission):
                # Check spatial constraints if provided
                if (
                    attributes
                    and "latitude" in attributes
                    and "longitude" in attributes
                ):
                    lat = attributes["latitude"]
                    lon = attributes["longitude"]
                    if not self.access_manager.can_access_location(user_id, lat, lon):
                        logger.warning(
                            f"User {user_id} denied access to location ({lat}, {lon})"
                        )
                        return False

                logger.info(
                    f"User {user_id} granted {permission.value} permission on {resource}"
                )
                return True

        logger.warning(
            f"User {user_id} denied {permission.value} permission on {resource}"
        )
        return False

    def _role_has_permission(
        self, role: Role, resource: str, permission: PermissionType
    ) -> bool:
        """
        Check if a role has a specific permission.

        Args:
            role: Role to check
            resource: Resource identifier
            permission: Permission type

        Returns:
            True if role has permission, False otherwise
        """
        # Check if role has admin permission (grants all)
        for perm in role.permissions:
            if perm.name == "admin" or perm.name == "*":
                return True

            # Check resource-specific permissions
            if perm.name == f"{resource}:{permission.value}":
                return True
            if perm.name == f"{resource}:*":
                return True

        return False

    def grant_permission(
        self,
        user_id: str,
        resource: str,
        permission: PermissionType,
        spatial_bounds: Any | None = None,
    ) -> bool:
        """
        Grant a permission to a user.

        Args:
            user_id: User identifier
            resource: Resource identifier
            permission: Permission type to grant
            spatial_bounds: Optional spatial bounds for the permission

        Returns:
            True if permission granted successfully, False otherwise
        """
        user_roles = self.access_manager.get_user_roles(user_id)

        if not user_roles:
            # Create a default role for the user
            role_name = f"user_{user_id}_role"
            role = Role(role_name)
            self.access_manager.add_role(role)
            self.access_manager.assign_role_to_user(user_id, role_name)
            user_roles = [role]

        # Add permission to user's primary role
        primary_role = user_roles[0]
        permission_name = f"{resource}:{permission.value}"

        spatial_perm = SpatialPermission(
            name=permission_name,
            geometry=spatial_bounds,
        )

        primary_role.add_permission(spatial_perm)

        logger.info(
            f"Granted {permission.value} permission on {resource} to user {user_id}"
        )
        return True

    def revoke_permission(
        self, user_id: str, resource: str, permission: PermissionType
    ) -> bool:
        """
        Revoke a permission from a user.

        Args:
            user_id: User identifier
            resource: Resource identifier
            permission: Permission type to revoke

        Returns:
            True if permission revoked successfully, False otherwise
        """
        user_roles = self.access_manager.get_user_roles(user_id)

        permission_name = f"{resource}:{permission.value}"

        for role in user_roles:
            role.permissions = [
                p for p in role.permissions if p.name != permission_name
            ]

        logger.info(
            f"Revoked {permission.value} permission on {resource} from user {user_id}"
        )
        return True

    def list_user_permissions(self, user_id: str) -> list[dict[str, Any]]:
        """
        List all permissions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of permission dictionaries
        """
        user_roles = self.access_manager.get_user_roles(user_id)
        permissions = []

        for role in user_roles:
            for perm in role.permissions:
                permissions.append(
                    {
                        "role": role.name,
                        "permission": perm.name,
                        "attributes": perm.attributes,
                        "max_resolution": perm.max_resolution,
                    }
                )

        return permissions
