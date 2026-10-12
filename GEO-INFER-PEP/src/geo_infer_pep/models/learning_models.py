"""Learning & Development data models."""

from datetime import datetime
from pydantic import BaseModel, Field


class LearningCourse(BaseModel):
    """A learning & development course offered to employees."""

    course_id: str = Field(..., description="Unique identifier for the course")
    title: str
    description: str | None = None
    category: str | None = None  # e.g., "compliance", "technical", "leadership"
    provider: str | None = None
    duration_hours: float | None = None
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)


class LearningEnrollment(BaseModel):
    """An employee enrollment in a learning course."""

    enrollment_id: str = Field(..., description="Unique identifier for the enrollment")
    employee_id: str  # Employee ID of the enrolled employee
    course_id: str  # LearningCourse ID
    status: str = "enrolled"  # e.g., "enrolled", "completed", "dropped"
    enrolled_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    score: float | None = None
    notes: str | None = None
