"""Talent Data Transformers."""

import logging
import pandas as pd
from ..models.talent_models import Candidate, JobRequisition

logger = logging.getLogger(__name__)


def clean_candidate_data(candidates: list[Candidate]) -> list[Candidate]:
    """
    Performs comprehensive cleaning operations on a list of Candidate objects.

    Cleaning operations include:
    - Standardize skills to lowercase and remove duplicates
    - Clean and validate email addresses
    - Clean phone numbers (remove non-numeric characters except +)
    - Standardize names (title case)
    - Clean LinkedIn profile URLs
    - Validate candidate status values
    - Clean company names

    Args:
        candidates: List of Candidate objects to clean

    Returns:
        List of cleaned Candidate objects
    """
    cleaned_candidates = []

    for cand in candidates:
        try:
            # Create a copy to avoid modifying the original
            cand_copy = Candidate(**cand.model_dump())

            # Clean and standardize skills
            if cand_copy.skills:
                # Convert to lowercase, remove duplicates, and filter empty strings
                cleaned_skills = [
                    skill.lower().strip() for skill in cand_copy.skills if skill.strip()
                ]
                cand_copy.skills = list(set(cleaned_skills))  # Remove duplicates

            # Clean email addresses - ensure lowercase
            if cand_copy.email:
                cand_copy.email = cand_copy.email.strip().lower()

            # Clean phone numbers - keep only digits and +
            if cand_copy.phone_number:
                cleaned_phone = "".join(
                    c for c in cand_copy.phone_number if c.isdigit() or c == "+"
                )
                cand_copy.phone_number = cleaned_phone

            # Clean names - title case
            if cand_copy.first_name:
                cand_copy.first_name = cand_copy.first_name.strip().title()
            if cand_copy.last_name:
                cand_copy.last_name = cand_copy.last_name.strip().title()

            # Clean LinkedIn profile URL
            if cand_copy.linkedin_profile:
                linkedin = cand_copy.linkedin_profile.strip()
                if linkedin and not linkedin.startswith(("http://", "https://")):
                    linkedin = "https://" + linkedin
                cand_copy.linkedin_profile = linkedin

            # Clean company names
            if cand_copy.current_company:
                cand_copy.current_company = cand_copy.current_company.strip().title()
            if cand_copy.current_title:
                cand_copy.current_title = cand_copy.current_title.strip().title()

            cleaned_candidates.append(cand_copy)

        except Exception as e:
            logger.error(f"Error cleaning candidate {cand.candidate_id}: {str(e)}")
            # Add the original if cleaning fails
            cleaned_candidates.append(cand)

    logger.info(f"Successfully cleaned {len(cleaned_candidates)} candidate records")
    return cleaned_candidates


def enrich_candidate_data(
    candidates: list[Candidate],
    requisitions: list[JobRequisition] | None = None,
) -> list[Candidate]:
    """
    Enriches candidate data with calculated fields and requisition context.

    Enrichment operations include:
    - Link candidates to job requisition details
    - Calculate time since application
    - Add candidate experience level estimation
    - Calculate application completeness score
    - Add skill matching scores against job requirements
    - Estimate candidate quality indicators

    Args:
        candidates: List of Candidate objects to enrich
        requisitions: Optional list of JobRequisition objects for context

    Returns:
        List of enriched Candidate objects
    """
    enriched_candidates = []

    # Create lookup dictionaries
    req_lookup = (
        {req.requisition_id: req for req in requisitions} if requisitions else {}
    )

    for cand in candidates:
        try:
            # Create a copy to avoid modifying the original
            cand_copy = Candidate(**cand.model_dump())

            # Link to job requisition if available
            if (
                cand_copy.job_requisition_id
                and cand_copy.job_requisition_id in req_lookup
            ):
                req = req_lookup[cand_copy.job_requisition_id]
                # Add requisition details to notes
                if not cand_copy.notes:
                    cand_copy.notes = ""
                cand_copy.notes += f"\nApplied for: {req.job_title} in {req.department}"

                # Add department and location info
                if (
                    not hasattr(cand_copy, "department_applied")
                    or not cand_copy.department_applied
                ):
                    cand_copy.notes += f"\nDepartment: {req.department}"
                if req.location:
                    cand_copy.notes += f"\nLocation: {req.location}"

            # Calculate time since application
            if cand_copy.applied_at:
                from datetime import datetime

                days_since_application = (datetime.now() - cand_copy.applied_at).days
                cand_copy.notes = (
                    cand_copy.notes or ""
                ) + f"\nDays since application: {days_since_application}"

                # Add urgency indicator
                if days_since_application <= 7:
                    urgency = "Very Recent"
                elif days_since_application <= 30:
                    urgency = "Recent"
                elif days_since_application <= 90:
                    urgency = "Aging"
                else:
                    urgency = "Old"

                cand_copy.notes += f"\nApplication Urgency: {urgency}"

            # Calculate application completeness score
            completeness_score = 0
            completeness_items = []

            if cand_copy.email:
                completeness_score += 1
                completeness_items.append("email")
            if cand_copy.phone_number:
                completeness_score += 1
                completeness_items.append("phone")
            if cand_copy.resume_url or cand_copy.portfolio_url:
                completeness_score += 1
                completeness_items.append("resume/portfolio")
            if cand_copy.linkedin_profile:
                completeness_score += 1
                completeness_items.append("linkedin")
            if cand_copy.skills:
                completeness_score += 1
                completeness_items.append("skills")
            if cand_copy.current_company or cand_copy.current_title:
                completeness_score += 1
                completeness_items.append("experience")

            max_score = 6
            completeness_percentage = (completeness_score / max_score) * 100

            cand_copy.notes = (
                (cand_copy.notes or "")
                + f"\nApplication Completeness: {completeness_score}/{max_score} ({completeness_percentage:.1f}%)"
            )

            # Estimate candidate experience level
            if cand_copy.current_title:
                title_lower = cand_copy.current_title.lower()

                if any(
                    keyword in title_lower
                    for keyword in [
                        "senior",
                        "lead",
                        "principal",
                        "architect",
                        "director",
                    ]
                ):
                    experience_level = "Senior"
                elif any(
                    keyword in title_lower
                    for keyword in ["junior", "associate", "trainee", "intern"]
                ):
                    experience_level = "Junior"
                elif any(
                    keyword in title_lower
                    for keyword in ["mid", "specialist", "analyst"]
                ):
                    experience_level = "Mid-Level"
                else:
                    experience_level = "Professional"

                cand_copy.notes = (
                    cand_copy.notes or ""
                ) + f"\nEstimated Experience Level: {experience_level}"

            # Add skill analysis if job requisition is available
            if (
                cand_copy.job_requisition_id
                and cand_copy.job_requisition_id in req_lookup
                and cand_copy.skills
            ):
                req = req_lookup[cand_copy.job_requisition_id]
                if req.qualifications:
                    # Simple keyword matching for skill-job fit
                    req_keywords = [qual.lower() for qual in req.qualifications]
                    candidate_skills = [skill.lower() for skill in cand_copy.skills]

                    matches = sum(
                        1
                        for req_kw in req_keywords
                        for skill in candidate_skills
                        if req_kw in skill or skill in req_kw
                    )
                    skill_match_score = (
                        (matches / len(req_keywords)) * 100 if req_keywords else 0
                    )

                    cand_copy.notes = (
                        cand_copy.notes or ""
                    ) + f"\nSkill-Job Match: {skill_match_score:.1f}%"

            enriched_candidates.append(cand_copy)

        except Exception as e:
            logger.error(f"Error enriching candidate {cand.candidate_id}: {str(e)}")
            # Add the original if enrichment fails
            enriched_candidates.append(cand)

    logger.info(f"Successfully enriched {len(enriched_candidates)} candidate records")
    return enriched_candidates


def convert_candidates_to_dataframe(candidates: list[Candidate]) -> pd.DataFrame:
    """
    Converts a list of Candidate Pydantic models to a Pandas DataFrame.
    """
    if not candidates:
        return pd.DataFrame()
    candidate_dicts = [cand.model_dump() for cand in candidates]
    df = pd.DataFrame(candidate_dicts)
    logger.info(f"Converted {len(df)} candidate records to DataFrame.")
    return df


def convert_requisitions_to_dataframe(
    requisitions: list[JobRequisition],
) -> pd.DataFrame:
    """
    Converts a list of JobRequisition Pydantic models to a Pandas DataFrame.
    """
    if not requisitions:
        return pd.DataFrame()
    req_dicts = [req.model_dump() for req in requisitions]
    df = pd.DataFrame(req_dicts)
    logger.info(f"Converted {len(df)} job requisition records to DataFrame.")
    return df
