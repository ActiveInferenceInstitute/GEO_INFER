"""Talent Acquisition and Management specific data models."""

from datetime import datetime, date
from pydantic import BaseModel, Field
from enum import StrEnum


class JobRequisitionStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    ON_HOLD = "on_hold"
    FILLED = "filled"
    CANCELLED = "cancelled"


class CandidateStatus(StrEnum):
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEWING = "interviewing"
    OFFER_EXTENDED = "offer_extended"
    OFFER_ACCEPTED = "offer_accepted"
    OFFER_DECLINED = "offer_declined"
    HIRED = "hired"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class InterviewType(StrEnum):
    PHONE_SCREEN = "phone_screen"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    PANEL = "panel"
    HM_INTERVIEW = "hm_interview"
    FINAL = "final"


class InterviewFeedback(BaseModel):
    interviewer_id: str  # Could be Employee ID
    interviewer_name: str | None = None  # Denormalized for convenience
    rating: float | None = None  # e.g., 1-5 scale
    pros: list[str] | None = None
    cons: list[str] | None = None
    notes: str | None = None
    recommend_hire: bool | None = None
    feedback_submitted_at: datetime = Field(default_factory=datetime.now)


class Interview(BaseModel):
    interview_id: str
    interview_type: InterviewType
    scheduled_at: datetime
    interviewers: list[str]  # List of Employee IDs
    feedback: list[InterviewFeedback] = []
    status: str = "scheduled"  # e.g., scheduled, completed, cancelled


class Offer(BaseModel):
    offer_id: str
    offered_at: date
    expires_at: date | None = None
    salary_offered: float | None = None
    currency: str | None = "USD"
    bonus_offered: float | None = None
    stock_options_offered: int | None = None
    start_date_proposed: date | None = None
    status: str = "pending"  # e.g., pending, accepted, declined, rescinded
    accepted_at: date | None = None
    declined_at: date | None = None


class Candidate(BaseModel):
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    first_name: str
    last_name: str
    email: str
    phone_number: str | None = None
    linkedin_profile: str | None = None
    resume_url: str | None = None  # Or store as blob/file path
    portfolio_url: str | None = None
    source: str | None = None  # e.g., "LinkedIn", "Referral", "Careers Page"
    applied_at: datetime = Field(default_factory=datetime.now)
    status: CandidateStatus = CandidateStatus.APPLIED
    job_requisition_id: str | None = None  # Link to JobRequisition
    current_company: str | None = None
    current_title: str | None = None
    skills: list[str] = []
    interviews: list[Interview] = []
    offer: Offer | None = None
    notes: str | None = None
    tags: list[str] = []
    updated_at: datetime = Field(default_factory=datetime.now)


class JobRequisition(BaseModel):
    requisition_id: str = Field(
        ..., description="Unique identifier for the job requisition"
    )
    job_title: str
    department: str
    location: str | None = None
    description: str | None = None
    responsibilities: list[str] | None = None
    qualifications: list[str] | None = None
    status: JobRequisitionStatus = JobRequisitionStatus.OPEN
    opened_at: date
    closed_at: date | None = None
    hiring_manager_id: str | None = None  # Employee ID
    priority: str | None = "medium"  # e.g., high, medium, low
    salary_min: float | None = None
    salary_max: float | None = None
    currency: str = "USD"
    candidates: list[Candidate] = []  # Candidates associated with this req
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
