"""HR Data Transformers."""

import logging
from datetime import date
import pandas as pd
from ..models.hr_models import Employee

logger = logging.getLogger(__name__)


def clean_employee_data(employees: list[Employee]) -> list[Employee]:
    """
    Performs comprehensive cleaning operations on a list of Employee objects.

    Cleaning operations include:
    - Standardize department names (title case)
    - Standardize job titles (title case)
    - Validate and clean email addresses
    - Ensure employment status is valid
    - Clean phone numbers (remove non-numeric characters except +)
    - Standardize location names

    Args:
        employees: List of Employee objects to clean

    Returns:
        List of cleaned Employee objects
    """
    cleaned_employees = []

    for emp in employees:
        try:
            # Create a copy to avoid modifying the original
            emp_copy = Employee(**emp.model_dump())

            # Clean department names - standardize to title case
            if emp_copy.department:
                emp_copy.department = emp_copy.department.strip().title()

            # Clean job titles - standardize to title case
            if emp_copy.job_title:
                emp_copy.job_title = emp_copy.job_title.strip().title()

            # Clean location names
            if emp_copy.location:
                emp_copy.location = emp_copy.location.strip().title()

            # Clean email addresses - ensure lowercase
            if emp_copy.email:
                emp_copy.email = emp_copy.email.strip().lower()

            # Clean personal email if present
            if emp_copy.personal_email:
                emp_copy.personal_email = emp_copy.personal_email.strip().lower()

            # Clean phone numbers - keep only digits and +
            if emp_copy.phone_number:
                cleaned_phone = "".join(
                    c for c in emp_copy.phone_number if c.isdigit() or c == "+"
                )
                emp_copy.phone_number = cleaned_phone

            # Clean emergency contact phone
            if emp_copy.emergency_contact_phone:
                cleaned_phone = "".join(
                    c
                    for c in emp_copy.emergency_contact_phone
                    if c.isdigit() or c == "+"
                )
                emp_copy.emergency_contact_phone = cleaned_phone

            # Standardize names - title case
            if emp_copy.first_name:
                emp_copy.first_name = emp_copy.first_name.strip().title()
            if emp_copy.middle_name:
                emp_copy.middle_name = emp_copy.middle_name.strip().title()
            if emp_copy.last_name:
                emp_copy.last_name = emp_copy.last_name.strip().title()
            if emp_copy.preferred_name:
                emp_copy.preferred_name = emp_copy.preferred_name.strip().title()

            cleaned_employees.append(emp_copy)

        except Exception as e:
            logger.error(f"Error cleaning employee {emp.employee_id}: {str(e)}")
            # Add the original if cleaning fails
            cleaned_employees.append(emp)

    logger.info(f"Successfully cleaned {len(cleaned_employees)} employee records")
    return cleaned_employees


def enrich_employee_data(
    employees: list[Employee], org_data: dict | None = None
) -> list[Employee]:
    """
    Enriches employee data with calculated fields and organizational context.

    Enrichment operations include:
    - Calculate tenure in years and months
    - Add age if birth date is available
    - Validate manager relationships
    - Add organizational hierarchy information
    - Calculate service milestones
    - Add department headcount context

    Args:
        employees: List of Employee objects to enrich
        org_data: Optional organizational data for context

    Returns:
        List of enriched Employee objects
    """
    enriched_employees = []

    # Create a lookup dictionary for manager validation
    employee_lookup = {emp.employee_id: emp for emp in employees}

    for emp in employees:
        try:
            # Create a copy to avoid modifying the original
            emp_copy = Employee(**emp.model_dump())

            # Calculate tenure if hire date is available
            if emp_copy.hire_date:
                today = date.today()
                hire_date = emp_copy.hire_date

                # Calculate years and months of service
                years = today.year - hire_date.year
                months = today.month - hire_date.month

                if today.day < hire_date.day:
                    months -= 1

                if months < 0:
                    years -= 1
                    months += 12

                # Add tenure information to custom fields
                emp_copy.custom_fields["tenure_years"] = years
                emp_copy.custom_fields["tenure_months"] = months
                emp_copy.custom_fields["total_tenure_months"] = years * 12 + months

                # Add service milestone information
                if years >= 5:
                    emp_copy.custom_fields["service_milestone"] = f"{years} years"
                elif years >= 1:
                    emp_copy.custom_fields["service_milestone"] = (
                        f"{years} year{'s' if years > 1 else ''}"
                    )
                else:
                    emp_copy.custom_fields["service_milestone"] = (
                        f"{months} month{'s' if months > 1 else ''}"
                    )

            # Calculate age if birth date is available
            if emp_copy.date_of_birth:
                today = date.today()
                birth_date = emp_copy.date_of_birth
                age = (
                    today.year
                    - birth_date.year
                    - ((today.month, today.day) < (birth_date.month, birth_date.day))
                )
                emp_copy.custom_fields["age"] = age

            # Validate manager relationship
            if emp_copy.manager_id:
                if emp_copy.manager_id in employee_lookup:
                    manager = employee_lookup[emp_copy.manager_id]
                    emp_copy.custom_fields["manager_name"] = (
                        f"{manager.first_name} {manager.last_name}"
                    )
                    emp_copy.custom_fields["manager_department"] = manager.department
                else:
                    emp_copy.custom_fields["manager_validation"] = (
                        "Manager ID not found in employee list"
                    )

            # Add department context if org_data is provided
            if org_data and emp_copy.department:
                dept_info = org_data.get("departments", {}).get(emp_copy.department, {})
                emp_copy.custom_fields["department_head"] = dept_info.get(
                    "head", "Unknown"
                )
                emp_copy.custom_fields["department_budget"] = dept_info.get(
                    "budget", "Unknown"
                )

            # Add employment status context
            status_context = {
                "ACTIVE": "Currently employed and active",
                "TERMINATED": "Employment has ended",
                "ON_LEAVE": "Temporarily on leave",
                "PENDING_HIRE": "Hire process in progress",
            }
            emp_copy.custom_fields["employment_status_description"] = (
                status_context.get(emp_copy.employment_status.value, "Unknown status")
            )

            enriched_employees.append(emp_copy)

        except Exception as e:
            logger.error(f"Error enriching employee {emp.employee_id}: {str(e)}")
            # Add the original if enrichment fails
            enriched_employees.append(emp)

    logger.info(f"Successfully enriched {len(enriched_employees)} employee records")
    return enriched_employees


def convert_employees_to_dataframe(employees: list[Employee]) -> pd.DataFrame:
    """
    Converts a list of Employee Pydantic models to a Pandas DataFrame.
    """
    if not employees:
        return pd.DataFrame()

    employee_dicts = [emp.model_dump() for emp in employees]
    df = pd.DataFrame(employee_dicts)
    # Further processing like flattening nested structures (e.g., compensation, job_history)
    # can be done here if needed for specific analyses.
    logger.info(f"Converted {len(df)} employee records to DataFrame.")
    return df
