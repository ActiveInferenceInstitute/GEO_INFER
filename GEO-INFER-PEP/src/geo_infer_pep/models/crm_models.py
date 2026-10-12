"""CRM specific data models."""

from datetime import datetime
from pydantic import BaseModel


class InteractionLog(BaseModel):
    timestamp: datetime = datetime.now()
    channel: str  # e.g., "email", "call", "meeting"
    summary: str
    agent_id: str | None = None


class Address(BaseModel):
    street: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str | None = None


class Customer(BaseModel):
    customer_id: str
    first_name: str | None = None
    last_name: str
    email: str | None = None
    phone_number: str | None = None
    company: str | None = None
    job_title: str | None = None
    address: Address | None = None
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    source: str | None = None  # e.g., "website_form", "referral", "cold_outreach"
    status: str | None = "active"  # e.g., "lead", "active_customer", "churned"
    tags: list[str] = []
    interaction_history: list[InteractionLog] = []
    website: str | None = None
    linkedin_profile: str | None = None
    notes: str | None = None
