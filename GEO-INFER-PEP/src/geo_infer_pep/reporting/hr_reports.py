"""HR Reporting functions."""

import logging
from typing import Any
from ..models.hr_models import Employee, EmploymentStatus
from ..hr.transformer import convert_employees_to_dataframe

logger = logging.getLogger(__name__)


def generate_headcount_report(
    employees: list[Employee], group_by: list[str] | None = None
) -> dict[str, Any]:
    """
    Generates a headcount report, optionally grouped by specified fields (e.g., department, location).
    """
    if not employees:
        return {"message": "No employee data for headcount report."}

    df = convert_employees_to_dataframe(employees)
    if df.empty:
        return {"message": "Employee data is empty after conversion to DataFrame."}

    report: dict[str, Any] = {
        "total_headcount": len(df[df["employment_status"] == EmploymentStatus.ACTIVE])
    }

    if group_by:
        active_employees_df = df[df["employment_status"] == EmploymentStatus.ACTIVE]
        for field in group_by:
            if field in active_employees_df.columns:
                report[f"headcount_by_{field}"] = (
                    active_employees_df.groupby(field).size().to_dict()
                )
            else:
                report[f"headcount_by_{field}"] = (
                    f"Field '{field}' not found for grouping."
                )

    logger.info("Generated headcount report.")
    return report


def generate_diversity_report(
    employees: list[Employee], diversity_fields: list[str] | None = None
) -> dict[str, Any]:
    """
    Generates a diversity report based on specified fields (e.g., gender, nationality).
    (This is a simplified example and needs careful consideration of privacy and ethics.)
    """
    if not employees:
        return {"message": "No employee data for diversity report."}

    df = convert_employees_to_dataframe(employees)
    if df.empty:
        return {"message": "Employee data is empty after conversion to DataFrame."}

    active_employees_df = df[df["employment_status"] == EmploymentStatus.ACTIVE]
    report: dict[str, Any] = {
        "total_active_employees_for_diversity_metrics": len(active_employees_df)
    }

    if not diversity_fields:
        diversity_fields = ["gender"]  # Default to gender if no fields specified

    for field in diversity_fields:
        if field in active_employees_df.columns:
            counts = active_employees_df[field].value_counts()
            percentages = active_employees_df[field].value_counts(normalize=True) * 100
            report[f"diversity_by_{field}"] = {
                "counts": counts.to_dict(),
                "percentages": percentages.round(2).to_dict(),
            }
        else:
            report[f"diversity_by_{field}"] = (
                f"Field '{field}' not found for diversity metrics."
            )

    logger.info("Generated diversity report.")
    return report


def get_quarterly_metrics(
    quarter: str, year: int, employees: list[Employee] | None = None
) -> dict[str, Any]:
    """
    Calculates real HR quarterly metrics from employee data.

    Args:
        quarter: Quarter string (e.g., "Q1", "Q2")
        year: Year integer
        employees: List of Employee objects to analyze

    Returns:
        Dictionary containing calculated HR metrics
    """
    logger.info(f"Calculating HR quarterly metrics for {quarter} {year}")

    if not employees:
        return {
            "quarter": quarter,
            "year": year,
            "message": "No employee data available for metrics calculation",
            "headcount_end_of_quarter": 0,
            "attrition_rate_percent": 0.0,
            "new_hires": 0,
            "diversity_snapshot": {},
        }

    # Calculate headcount metrics
    total_employees = len(employees)
    active_employees = len(
        [e for e in employees if e.employment_status.value == "ACTIVE"]
    )

    # Calculate basic diversity metrics
    gender_counts: dict[str, int] = {}
    department_counts: dict[str, int] = {}

    for emp in employees:
        if emp.employment_status.value == "ACTIVE":
            # Gender diversity
            gender = emp.gender.value if emp.gender else "not_specified"
            gender_counts[gender] = gender_counts.get(gender, 0) + 1

            # Department breakdown
            department_counts[emp.department] = (
                department_counts.get(emp.department, 0) + 1
            )

    # Calculate gender percentages
    total_active = sum(gender_counts.values())
    gender_percentages = {}
    if total_active > 0:
        gender_percentages = {
            gender: (count / total_active) * 100
            for gender, count in gender_counts.items()
        }

    # Calculate tenure statistics
    tenure_stats = {"average_tenure_years": 0, "median_tenure_years": 0}
    tenure_values = []

    for emp in employees:
        if (
            emp.employment_status.value == "ACTIVE"
            and emp.hire_date
            and "tenure_years" in emp.custom_fields
        ):
            tenure_years = emp.custom_fields["tenure_years"]
            tenure_values.append(tenure_years)

    if tenure_values:
        tenure_stats["average_tenure_years"] = sum(tenure_values) / len(tenure_values)
        tenure_stats["median_tenure_years"] = sorted(tenure_values)[
            len(tenure_values) // 2
        ]

    # Estimate attrition and new hires (simplified calculation)
    # In a real system, this would compare with previous quarter data
    terminated_employees = len(
        [e for e in employees if e.employment_status.value == "TERMINATED"]
    )
    pending_hires = len(
        [e for e in employees if e.employment_status.value == "PENDING_HIRE"]
    )

    # Simplified attrition rate calculation
    attrition_rate = (
        (terminated_employees / total_employees * 100) if total_employees > 0 else 0
    )

    metrics = {
        "quarter": quarter,
        "year": year,
        "headcount_end_of_quarter": active_employees,
        "total_employees": total_employees,
        "attrition_rate_percent": round(attrition_rate, 2),
        "new_hires": pending_hires,
        "terminated_employees": terminated_employees,
        "diversity_snapshot": {
            "gender_counts": gender_counts,
            "gender_percentages": gender_percentages,
            "department_breakdown": department_counts,
        },
        "tenure_statistics": tenure_stats,
        "data_source": f"Calculated from {len(employees)} employee records",
    }

    logger.info(f"Successfully calculated HR metrics for {quarter} {year}")
    return metrics


# Add more HR-specific reporting functions here, e.g.:
# - Attrition rate report
# - Compensation summary report
# - Performance review completion status
