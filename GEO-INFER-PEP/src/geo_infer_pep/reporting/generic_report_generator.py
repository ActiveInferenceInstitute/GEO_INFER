"""Generic report generation utilities."""

import json
import tempfile
from pathlib import Path


def create_quarterly_overview(
    hr_metrics: dict,
    crm_metrics: dict,
    talent_metrics: dict,
    output_path: str | None = None,
) -> str:
    """Write a structured quarterly overview report to JSON."""
    report_content = {
        "title": "Quarterly People Operations Report",
        "hr_summary": hr_metrics,
        "crm_summary": crm_metrics,
        "talent_summary": talent_metrics,
    }
    if output_path is None:
        with tempfile.NamedTemporaryFile(
            prefix="quarterly_people_report_", suffix=".json", delete=False
        ) as report_file:
            output_path = report_file.name

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as report_file:
        json.dump(report_content, report_file, indent=2, default=str)
    return str(path)
