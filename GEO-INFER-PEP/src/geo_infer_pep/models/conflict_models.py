"""Conflict resolution data models."""

from datetime import datetime
from pydantic import BaseModel, Field


class ConflictCase(BaseModel):
    """A workplace conflict-resolution case."""

    case_id: str = Field(..., description="Unique identifier for the case")
    title: str
    description: str | None = None
    parties: list[str] = Field(default_factory=list)  # Employee IDs involved
    severity: str | None = None  # e.g., "low", "medium", "high"
    status: str = "open"  # e.g., "open", "under_review", "resolved", "dismissed"
    opened_at: datetime = Field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    resolution_notes: str | None = None
