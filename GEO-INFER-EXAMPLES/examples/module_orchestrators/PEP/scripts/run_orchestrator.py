#!/usr/bin/env python3
"""GEO-INFER-PEP module orchestrator.

Runs one documented end-to-end PEP operation on synthetic data: build a
synthetic workforce of employee records, run the real HR cleaning and
enrichment transformers (title-casing, tenure calculation, manager
validation), then generate headcount and diversity reports. All work goes
through the real ``geo_infer_pep`` public API.
"""

from __future__ import annotations

import sys
from datetime import date
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    from geo_infer_pep.hr.transformer import clean_employee_data, enrich_employee_data
    from geo_infer_pep.models.hr_models import (
        Compensation,
        Employee,
        EmploymentStatus,
        Gender,
    )
    from geo_infer_pep.reporting.hr_reports import (
        generate_diversity_report,
        generate_headcount_report,
    )

    # Synthetic workforce: 14 employees across three departments and sites.
    departments = [" research", " GIS LAB", "outreach"]
    locations = ["Crescent City HQ", "Eureka Field Office", "Remote"]
    genders = [Gender.FEMALE, Gender.MALE, Gender.NON_BINARY, Gender.FEMALE]
    nationalities = ["US", "CA", "MX", "US"]
    titles = ["Research Scientist", "GIS Analyst", "Engagement Coordinator"]

    employees: list[Employee] = []
    for i in range(14):
        employees.append(
            Employee(
                employee_id=f"emp-{i:03d}",
                first_name=f"  first{i}",
                last_name=f"LAST{i}",
                email=f"First{i}.Last{i}@example.org",
                hire_date=date(2019 + (i % 6), 1 + (i % 12), 1 + (i % 27)),
                employment_status=(
                    EmploymentStatus.TERMINATED
                    if i in (4, 11)
                    else EmploymentStatus.ACTIVE
                ),
                job_title=titles[i % len(titles)],
                department=departments[i % len(departments)],
                manager_id="emp-000" if i not in (0,) else None,
                location=locations[i % len(locations)],
                gender=genders[i % len(genders)],
                nationality=nationalities[i % len(nationalities)],
                compensation=Compensation(
                    salary=52_000.0 + 3_400.0 * i,
                    currency="USD",
                    pay_frequency="annual",
                ),
            )
        )

    cleaned = clean_employee_data(employees)
    enriched = enrich_employee_data(cleaned)

    headcount = generate_headcount_report(enriched, group_by=["department", "location"])
    diversity = generate_diversity_report(
        enriched, diversity_fields=["gender", "nationality"]
    )

    tenured = [e for e in enriched if "tenure_years" in e.custom_fields]
    manager_validations = {
        e.employee_id: e.custom_fields.get("manager_name")
        for e in enriched
        if "manager_name" in e.custom_fields
    }

    return {
        "operation": "hr_pipeline_clean_enrich_report",
        "records_input": len(employees),
        "records_cleaned": len(cleaned),
        "records_enriched": len(enriched),
        "sample_cleaning": {
            "employee_id": enriched[0].employee_id,
            "name_after_clean": f"{enriched[0].first_name} {enriched[0].last_name}",
            "email_after_clean": enriched[0].email,
            "department_after_clean": enriched[0].department,
        },
        "enrichment": {
            "records_with_tenure": len(tenured),
            "sample_tenure_years": tenured[0].custom_fields["tenure_years"]
            if tenured
            else None,
            "manager_names_resolved": manager_validations,
        },
        "headcount_report": headcount,
        "diversity_report": diversity,
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("PEP", _operation))
