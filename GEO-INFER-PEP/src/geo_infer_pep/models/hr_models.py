"""HR specific data models."""

from typing import Any
from datetime import datetime, date
from pydantic import BaseModel, Field
from enum import StrEnum


class EmploymentStatus(StrEnum):
    ACTIVE = "active"
    TERMINATED = "terminated"
    ON_LEAVE = "on_leave"
    PENDING_HIRE = "pending_hire"


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"
    OTHER = "other"


class Compensation(BaseModel):
    salary: float
    currency: str = "USD"
    pay_frequency: str  # e.g., "annual", "monthly", "hourly"
    bonus_potential: float | None = None
    stock_options: int | None = None


class JobHistoryEntry(BaseModel):
    job_title: str
    department: str
    start_date: date
    end_date: date | None = None
    manager_id: str | None = None  # Employee ID of the manager
    is_current: bool = False


class PerformanceReview(BaseModel):
    review_id: str
    review_date: date
    reviewer_id: str  # Employee ID of the reviewer
    overall_rating: float  # e.g., on a scale of 1-5
    comments: str | None = None
    goals_set: list[str] | None = None
    areas_for_improvement: list[str] | None = None


class Employee(BaseModel):
    employee_id: str = Field(..., description="Unique identifier for the employee")
    first_name: str
    last_name: str
    middle_name: str | None = None
    preferred_name: str | None = None
    email: str
    personal_email: str | None = None
    phone_number: str | None = None
    date_of_birth: date | None = None
    gender: Gender | None = None
    nationality: str | None = None

    hire_date: date | None = None
    termination_date: date | None = None
    employment_status: EmploymentStatus = EmploymentStatus.ACTIVE
    job_title: str
    department: str
    manager_id: str | None = None  # Employee ID of the direct manager
    location: str | None = None  # e.g., Office name, "Remote"

    compensation: Compensation | None = None
    job_history: list[JobHistoryEntry] = []
    performance_reviews: list[PerformanceReview] = []

    emergency_contact_name: str | None = None
    emergency_contact_phone: str | None = None

    custom_fields: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
