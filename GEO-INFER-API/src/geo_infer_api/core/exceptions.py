"""
Custom exceptions for the GEO-INFER-API.
"""

from typing import Any

from fastapi import HTTPException, status


class APIError(HTTPException):
    """Base exception for API errors."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str | None = None,
        additional_info: dict[str, Any] | None = None,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code or f"API_{status_code}"
        self.additional_info = additional_info or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.detail,
                "status_code": self.status_code,
                **self.additional_info,
            }
        }


class ValidationError(APIError):
    """Exception raised for validation errors."""

    def __init__(
        self,
        detail: str,
        field: str | None = None,
        value: Any | None = None,
        additional_info: dict[str, Any] | None = None,
    ):
        error_info = {"field": field, "value": str(value)} if field else {}
        if additional_info:
            error_info.update(additional_info)

        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR",
            additional_info=error_info,
        )


class NotFoundError(APIError):
    """Exception raised when a resource is not found."""

    def __init__(
        self,
        resource: str,
        identifier: str | None = None,
        additional_info: dict[str, Any] | None = None,
    ):
        detail = f"{resource} not found"
        if identifier:
            detail += f" with ID: {identifier}"

        error_info = {"resource": resource}
        if identifier:
            error_info["identifier"] = identifier
        if additional_info:
            error_info.update(additional_info)

        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="RESOURCE_NOT_FOUND",
            additional_info=error_info,
        )


class ConflictError(APIError):
    """Exception raised when there's a conflict (e.g., duplicate resource)."""

    def __init__(
        self,
        resource: str,
        conflict_reason: str,
        identifier: str | None = None,
        additional_info: dict[str, Any] | None = None,
    ):
        detail = f"Conflict for {resource}: {conflict_reason}"
        if identifier:
            detail += f" (ID: {identifier})"

        error_info = {"resource": resource, "conflict_reason": conflict_reason}
        if identifier:
            error_info["identifier"] = identifier
        if additional_info:
            error_info.update(additional_info)

        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="RESOURCE_CONFLICT",
            additional_info=error_info,
        )


class GeometryError(APIError):
    """Exception raised for geometry-related errors."""

    def __init__(
        self,
        detail: str,
        geometry_type: str | None = None,
        operation: str | None = None,
        additional_info: dict[str, Any] | None = None,
    ):
        error_info = {}
        if geometry_type:
            error_info["geometry_type"] = geometry_type
        if operation:
            error_info["operation"] = operation
        if additional_info:
            error_info.update(additional_info)

        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="GEOMETRY_ERROR",
            additional_info=error_info,
        )


class ProcessingError(APIError):
    """Exception raised for processing-related errors."""

    def __init__(
        self,
        detail: str,
        operation: str | None = None,
        processing_stage: str | None = None,
        additional_info: dict[str, Any] | None = None,
    ):
        error_info = {}
        if operation:
            error_info["operation"] = operation
        if processing_stage:
            error_info["processing_stage"] = processing_stage
        if additional_info:
            error_info.update(additional_info)

        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="PROCESSING_ERROR",
            additional_info=error_info,
        )


class BadRequestError(APIError):
    """Exception raised for bad request errors."""

    def __init__(
        self,
        detail: str,
        field: str | None = None,
        additional_info: dict[str, Any] | None = None,
    ):
        error_info = {"field": field} if field else {}
        if additional_info:
            error_info.update(additional_info)

        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="BAD_REQUEST",
            additional_info=error_info,
        )
