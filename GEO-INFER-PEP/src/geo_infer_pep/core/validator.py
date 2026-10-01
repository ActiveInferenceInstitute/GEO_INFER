"""
PEP Data Validator

This module provides comprehensive validation and integrity checking
for all PEP data types and workflows.
"""

from typing import Any
from collections.abc import Callable
from datetime import datetime, date
import re
import logging
# Email validation will use basic regex validation

from ..models.hr_models import Employee, EmploymentStatus
from ..models.crm_models import Customer
from ..models.talent_models import Candidate, CandidateStatus, JobRequisition

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation operation."""

    def __init__(
        self,
        is_valid: bool,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.validated_at = datetime.now()

    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "validated_at": self.validated_at.isoformat(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


class PEPValidator:
    """
    Comprehensive validator for PEP data and workflows.

    Provides validation for:
    - Data integrity and completeness
    - Business rule compliance
    - Format and type validation
    - Cross-reference validation
    - Workflow-specific validations
    """

    def __init__(self) -> None:
        self.email_regex = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        self.phone_regex = re.compile(r"^[\+]?[1-9][\d]{0,15}$")

    def validate_employee(
        self, employee: Employee, strict: bool = False
    ) -> ValidationResult:
        """
        Comprehensive validation of Employee data.

        Args:
            employee: Employee object to validate
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult with validation status and issues
        """
        result = ValidationResult(True)

        # Required field validation
        if not employee.employee_id or not employee.employee_id.strip():
            result.add_error("Employee ID is required")

        if not employee.first_name or not employee.first_name.strip():
            result.add_error("First name is required")

        if not employee.last_name or not employee.last_name.strip():
            result.add_error("Last name is required")

        if not employee.email or not employee.email.strip():
            result.add_error("Email is required")

        if not employee.job_title or not employee.job_title.strip():
            result.add_error("Job title is required")

        if not employee.department or not employee.department.strip():
            result.add_error("Department is required")

        # Email validation
        if employee.email:
            email_result = self._validate_email(employee.email)
            if not email_result["valid"]:
                result.add_error(f"Invalid email format: {email_result['error']}")

        # Phone validation
        if employee.phone_number:
            phone_result = self._validate_phone(employee.phone_number)
            if not phone_result["valid"]:
                result.add_error(f"Invalid phone format: {phone_result['error']}")

        # Date validations
        if employee.hire_date and employee.hire_date > date.today():
            result.add_error("Hire date cannot be in the future")

        if employee.termination_date and employee.hire_date:
            if employee.termination_date < employee.hire_date:
                result.add_error("Termination date cannot be before hire date")

        if employee.date_of_birth:
            age = (date.today() - employee.date_of_birth).days / 365.25
            if age < 14:
                result.add_error("Employee appears to be underage (under 14)")
            elif age > 100:
                result.add_warning("Employee age seems unusually high (>100 years)")

        # Status validation
        if (
            employee.employment_status == EmploymentStatus.TERMINATED
            and not employee.termination_date
        ):
            result.add_warning("Terminated employee should have termination date")

        if (
            employee.employment_status == EmploymentStatus.ACTIVE
            and employee.termination_date
        ):
            result.add_error("Active employee cannot have termination date")

        # Manager validation
        if employee.manager_id == employee.employee_id:
            result.add_error("Employee cannot be their own manager")

        # Custom fields validation
        if employee.custom_fields:
            for key, value in employee.custom_fields.items():
                if key.startswith("_"):
                    result.add_warning(
                        f"Custom field '{key}' starts with underscore (convention violation)"
                    )

        if strict and result.warnings:
            for warning in result.warnings:
                result.add_error(f"Strict mode: {warning}")

        return result

    def validate_customer(
        self, customer: Customer, strict: bool = False
    ) -> ValidationResult:
        """
        Comprehensive validation of Customer data.

        Args:
            customer: Customer object to validate
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult with validation status and issues
        """
        result = ValidationResult(True)

        # Required field validation
        if not customer.customer_id or not customer.customer_id.strip():
            result.add_error("Customer ID is required")

        if not customer.last_name or not customer.last_name.strip():
            result.add_error("Last name is required")

        # Email validation
        if customer.email:
            email_result = self._validate_email(customer.email)
            if not email_result["valid"]:
                result.add_error(f"Invalid email format: {email_result['error']}")

        # Phone validation
        if customer.phone_number:
            phone_result = self._validate_phone(customer.phone_number)
            if not phone_result["valid"]:
                result.add_error(f"Invalid phone format: {phone_result['error']}")

        # Website validation
        if customer.website:
            if not customer.website.startswith(("http://", "https://")):
                result.add_error("Website URL must start with http:// or https://")

        # LinkedIn validation
        if customer.linkedin_profile:
            if not customer.linkedin_profile.startswith(("http://", "https://")):
                result.add_error(
                    "LinkedIn profile URL must start with http:// or https://"
                )

        # Status validation
        valid_statuses = [
            "active",
            "inactive",
            "lead",
            "prospect",
            "customer",
            "churned",
        ]
        if customer.status and customer.status.lower() not in valid_statuses:
            result.add_error(
                f"Invalid status '{customer.status}'. Must be one of: {valid_statuses}"
            )

        # Date validations
        if customer.created_at and customer.created_at > datetime.now():
            result.add_error("Created date cannot be in the future")

        if customer.updated_at and customer.updated_at < customer.created_at:
            result.add_error("Updated date cannot be before created date")

        # Tag validation
        if customer.tags:
            for tag in customer.tags:
                if len(tag) > 50:
                    result.add_warning(f"Tag '{tag}' is very long (>50 characters)")
                if any(char in tag for char in [",", ";", "|"]):
                    result.add_warning(
                        f"Tag '{tag}' contains special characters that may cause parsing issues"
                    )

        # Interaction history validation
        if customer.interaction_history:
            for interaction in customer.interaction_history:
                if interaction.timestamp > datetime.now():
                    result.add_error("Interaction timestamp cannot be in the future")

                if not interaction.summary or not interaction.summary.strip():
                    result.add_error("Interaction summary cannot be empty")

        if strict and result.warnings:
            for warning in result.warnings:
                result.add_error(f"Strict mode: {warning}")

        return result

    def validate_candidate(
        self, candidate: Candidate, strict: bool = False
    ) -> ValidationResult:
        """
        Comprehensive validation of Candidate data.

        Args:
            candidate: Candidate object to validate
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult with validation status and issues
        """
        result = ValidationResult(True)

        # Required field validation
        if not candidate.candidate_id or not candidate.candidate_id.strip():
            result.add_error("Candidate ID is required")

        if not candidate.first_name or not candidate.first_name.strip():
            result.add_error("First name is required")

        if not candidate.last_name or not candidate.last_name.strip():
            result.add_error("Last name is required")

        if not candidate.email or not candidate.email.strip():
            result.add_error("Email is required")

        # Email validation
        if candidate.email:
            email_result = self._validate_email(candidate.email)
            if not email_result["valid"]:
                result.add_error(f"Invalid email format: {email_result['error']}")

        # Phone validation
        if candidate.phone_number:
            phone_result = self._validate_phone(candidate.phone_number)
            if not phone_result["valid"]:
                result.add_error(f"Invalid phone format: {phone_result['error']}")

        # LinkedIn validation
        if candidate.linkedin_profile:
            if not candidate.linkedin_profile.startswith(("http://", "https://")):
                result.add_error(
                    "LinkedIn profile URL must start with http:// or https://"
                )

        # Resume/portfolio validation
        if candidate.resume_url:
            if not candidate.resume_url.startswith(("http://", "https://")):
                result.add_error("Resume URL must start with http:// or https://")

        if candidate.portfolio_url:
            if not candidate.portfolio_url.startswith(("http://", "https://")):
                result.add_error("Portfolio URL must start with http:// or https://")

        # Date validations
        if candidate.applied_at and candidate.applied_at > datetime.now():
            result.add_error("Application date cannot be in the future")

        # Status validation
        if candidate.status == CandidateStatus.HIRED and not candidate.offer:
            result.add_warning("Hired candidate should have offer information")

        # Offer validation
        if candidate.offer:
            if (
                candidate.offer.accepted_at
                and candidate.offer.accepted_at < candidate.applied_at.date()
            ):
                result.add_error(
                    "Offer acceptance date cannot be before application date"
                )

            if candidate.offer.expires_at and candidate.offer.accepted_at:
                if candidate.offer.expires_at < candidate.offer.accepted_at:
                    result.add_error(
                        "Offer expiration date cannot be before acceptance date"
                    )

        # Interview validation
        if candidate.interviews:
            for interview in candidate.interviews:
                if interview.scheduled_at < candidate.applied_at:
                    result.add_error(
                        "Interview cannot be scheduled before application date"
                    )

                if interview.feedback:
                    for feedback in interview.feedback:
                        if feedback.feedback_submitted_at < interview.scheduled_at:
                            result.add_warning(
                                "Interview feedback submitted before interview date"
                            )

        # Skills validation
        if candidate.skills:
            for skill in candidate.skills:
                if len(skill.strip()) == 0:
                    result.add_error("Skills cannot be empty strings")
                elif len(skill) > 100:
                    result.add_warning(
                        f"Skill '{skill}' is very long (>100 characters)"
                    )

        if strict and result.warnings:
            for warning in result.warnings:
                result.add_error(f"Strict mode: {warning}")

        return result

    def validate_job_requisition(
        self, requisition: JobRequisition, strict: bool = False
    ) -> ValidationResult:
        """
        Comprehensive validation of JobRequisition data.

        Args:
            requisition: JobRequisition object to validate
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult with validation status and issues
        """
        result = ValidationResult(True)

        # Required field validation
        if not requisition.requisition_id or not requisition.requisition_id.strip():
            result.add_error("Requisition ID is required")

        if not requisition.job_title or not requisition.job_title.strip():
            result.add_error("Job title is required")

        if not requisition.department or not requisition.department.strip():
            result.add_error("Department is required")

        # Date validations
        if requisition.opened_at and requisition.opened_at > date.today():
            result.add_error("Opening date cannot be in the future")

        if requisition.closed_at and requisition.opened_at:
            if requisition.closed_at < requisition.opened_at:
                result.add_error("Closing date cannot be before opening date")

        if requisition.closed_at and requisition.closed_at > date.today():
            result.add_error("Closing date cannot be in the future")

        # Status validation
        if requisition.status.value == "closed" and not requisition.closed_at:
            result.add_error("Closed requisition must have closing date")

        if requisition.status.value == "open" and requisition.closed_at:
            result.add_error("Open requisition cannot have closing date")

        # Salary validation
        if requisition.salary_min and requisition.salary_max:
            if requisition.salary_min > requisition.salary_max:
                result.add_error("Minimum salary cannot be greater than maximum salary")

            if requisition.salary_min < 0:
                result.add_error("Minimum salary cannot be negative")

            if requisition.salary_max < 0:
                result.add_error("Maximum salary cannot be negative")

        # Priority validation
        valid_priorities = ["low", "medium", "high", "urgent"]
        if (
            requisition.priority
            and requisition.priority.lower() not in valid_priorities
        ):
            result.add_error(
                f"Invalid priority '{requisition.priority}'. Must be one of: {valid_priorities}"
            )

        if strict and result.warnings:
            for warning in result.warnings:
                result.add_error(f"Strict mode: {warning}")

        return result

    def validate_onboarding_workflow(
        self, candidate_id: str, employees: list[Employee], candidates: list[Candidate]
    ) -> ValidationResult:
        """
        Validate onboarding workflow prerequisites and data integrity.

        Args:
            candidate_id: ID of candidate to onboard
            employees: List of existing employees
            candidates: List of candidates

        Returns:
            ValidationResult with workflow validation status
        """
        result = ValidationResult(True)

        # Find candidate
        candidate = None
        for cand in candidates:
            if cand.candidate_id == candidate_id:
                candidate = cand
                break

        if not candidate:
            result.add_error(f"candidate {candidate_id} not found")
            return result

        # Check candidate status
        if candidate.status != CandidateStatus.OFFER_ACCEPTED:
            result.add_error(
                f"candidate {candidate_id} has status {candidate.status}, expected OFFER_ACCEPTED"
            )

        # Check if employee already exists
        for emp in employees:
            if emp.email == candidate.email:
                result.add_error(
                    f"Employee with email {candidate.email} already exists"
                )

        # Check required candidate data
        if not candidate.email:
            result.add_error("Candidate must have email address")

        if not candidate.first_name or not candidate.last_name:
            result.add_error("Candidate must have complete name")

        # Check offer information
        if not candidate.offer:
            result.add_error("Candidate must have offer information for onboarding")

        if candidate.offer and not candidate.offer.accepted_at:
            result.add_error("Candidate offer must have acceptance date")

        return result

    def validate_data_integrity(
        self,
        employees: list[Employee] | None = None,
        customers: list[Customer] | None = None,
        candidates: list[Candidate] | None = None,
    ) -> dict[str, ValidationResult]:
        """
        Perform comprehensive data integrity validation across all data types.

        Args:
            employees: List of employees to validate
            customers: List of customers to validate
            candidates: List of candidates to validate

        Returns:
            Dictionary mapping data types to validation results
        """
        results = {}

        if employees:
            results["employees"] = self._validate_collection(
                employees, self.validate_employee
            )

        if customers:
            results["customers"] = self._validate_collection(
                customers, self.validate_customer
            )

        if candidates:
            results["candidates"] = self._validate_collection(
                candidates, self.validate_candidate
            )

        # Cross-reference validation
        if employees and candidates:
            results["cross_references"] = self._validate_cross_references(
                employees, candidates
            )

        return results

    def _validate_collection(
        self, items: list[Any], validator_func: Callable[[Any], ValidationResult]
    ) -> ValidationResult:
        """Validate a collection of items and aggregate results."""
        collection_result = ValidationResult(True)
        valid_count = 0
        invalid_count = 0

        for item in items:
            try:
                result = validator_func(item)
                if result.is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    collection_result.add_error(
                        f"Item validation failed: {result.errors}"
                    )
                    for warning in result.warnings:
                        collection_result.add_warning(f"Item warning: {warning}")
            except Exception as e:
                invalid_count += 1
                collection_result.add_error(f"Validation error: {str(e)}")

        collection_result.add_warning(
            f"Validation summary: {valid_count} valid, {invalid_count} invalid items"
        )

        if invalid_count > 0:
            collection_result.is_valid = False

        return collection_result

    def _validate_cross_references(
        self, employees: list[Employee], candidates: list[Candidate]
    ) -> ValidationResult:
        """Validate cross-references between employees and candidates."""
        result = ValidationResult(True)

        # Check for duplicate emails
        employee_emails = {emp.email for emp in employees if emp.email}
        candidate_emails = {cand.email for cand in candidates if cand.email}

        duplicate_emails = employee_emails.intersection(candidate_emails)
        if duplicate_emails:
            result.add_error(
                f"Duplicate emails found between employees and candidates: {duplicate_emails}"
            )

        # Validate manager references
        manager_ids = {emp.employee_id for emp in employees}
        for emp in employees:
            if emp.manager_id and emp.manager_id not in manager_ids:
                result.add_warning(
                    f"Employee {emp.employee_id} references non-existent manager {emp.manager_id}"
                )

        return result

    def _validate_email(self, email: str) -> dict[str, Any]:
        """Validate email address format."""
        if not email or not email.strip():
            return {"valid": False, "error": "Email is empty"}

        # Basic regex validation
        if self.email_regex.match(email):
            return {"valid": True}
        else:
            return {"valid": False, "error": "Invalid email format"}

    def _validate_phone(self, phone: str) -> dict[str, Any]:
        """Validate phone number format."""
        if not phone or not phone.strip():
            return {"valid": False, "error": "Phone is empty"}

        # Remove all non-digit characters except +
        clean_phone = "".join(c for c in phone if c.isdigit() or c == "+")

        if self.phone_regex.match(clean_phone):
            # Check length constraints
            digits_only = "".join(c for c in clean_phone if c.isdigit())
            if len(digits_only) >= 7 and len(digits_only) <= 15:
                return {"valid": True}
            else:
                return {"valid": False, "error": "Phone number must have 7-15 digits"}
        else:
            return {"valid": False, "error": "Invalid phone format"}
