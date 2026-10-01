"""Survey and engagement data models."""

from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field


class Survey(BaseModel):
    """An engagement survey issued to employees."""

    survey_id: str = Field(..., description="Unique identifier for the survey")
    title: str
    description: str | None = None
    questions: list[str] = Field(default_factory=list)
    audience: str | None = None  # e.g., "all_employees", "engineering"
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)


class SurveyResponse(BaseModel):
    """A single respondent's answers to a survey."""

    response_id: str = Field(..., description="Unique identifier for the response")
    survey_id: str  # Survey ID this response belongs to
    respondent_id: str | None = None  # Anonymized surveys may omit this
    submitted_at: datetime = Field(default_factory=datetime.now)
    answers: dict[str, Any] = Field(default_factory=dict)
