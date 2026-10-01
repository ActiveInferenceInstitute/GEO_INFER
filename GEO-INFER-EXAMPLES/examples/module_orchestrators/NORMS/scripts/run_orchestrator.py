#!/usr/bin/env python3
"""GEO-INFER-NORMS module orchestrator.

Runs one documented end-to-end NORMS operation on synthetic data: track
compliance of synthetic land entities against synthetic regulations over
time, query per-entity and per-regulation compliance, and generate a summary
compliance report. All work goes through the real ``geo_infer_norms`` public
API.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_norms.core.compliance_tracking import (
        ComplianceReport,
        ComplianceTracker,
    )
    from geo_infer_norms.models.compliance_status import ComplianceStatus

    # Synthetic compliance history: 5 land parcels against 3 regulations,
    # two reporting periods each (fixed timestamps keep this deterministic).
    base = datetime.datetime(2026, 1, 1, 12, 0, 0)
    later = datetime.datetime(2026, 4, 1, 12, 0, 0)
    regulations = ["REG-SHORELINE-SETBACK", "REG-RIPARIAN-BUFFER", "REG-SEPTIC-DENSITY"]
    entity_ids = [f"parcel-{i:02d}" for i in range(5)]

    # Deterministic outcome matrix: rows are periods, columns are entities.
    # First period values, then a second-period remediation for parcel-01.
    outcomes = [
        [
            (True, 0.95),
            (False, 0.40),
            (True, 0.80),
            (False, 0.55),
            (True, 0.90),
        ],
        [
            (True, 0.97),
            (True, 0.85),
            (True, 0.82),
            (False, 0.50),
            (True, 0.92),
        ],
    ]

    statuses: list[ComplianceStatus] = []
    for period_index, (timestamp, rows) in enumerate(zip((base, later), outcomes)):
        for entity_slot, (compliant, level) in enumerate(rows):
            entity_id = entity_ids[entity_slot]
            for regulation_id in regulations:
                statuses.append(
                    ComplianceStatus(
                        id=f"cs-{period_index}-{entity_slot}-{regulation_id}",
                        entity_id=entity_id,
                        regulation_id=regulation_id,
                        is_compliant=compliant,
                        compliance_level=level,
                        timestamp=timestamp,
                        notes="synthetic inspection record",
                        evaluation_method="rule_check",
                    )
                )

    tracker = ComplianceTracker(
        name="synthetic-county-land-compliance",
        description="Synthetic compliance tracking for orchestrator demo",
        compliance_statuses=statuses,
    )

    entity_report = tracker.get_entity_compliance("parcel-01", as_of_date=later)
    regulation_report = tracker.get_regulation_compliance(
        "REG-RIPARIAN-BUFFER", as_of_date=later
    )

    report = ComplianceReport(
        compliance_tracker=tracker,
        title="Synthetic County Compliance Summary",
        description="End-to-end orchestrator run on synthetic records",
    )
    summary = report.generate_summary_report(as_of_date=later)

    return {
        "operation": "compliance_tracking_and_summary_report",
        "statuses_tracked": len(statuses),
        "entities": len(entity_ids),
        "regulations": len(regulations),
        "entity_report_parcel_01": {
            "status": entity_report["status"],
            "compliance_count": entity_report["compliance_count"],
            "non_compliance_count": entity_report["non_compliance_count"],
            "compliance_percentage": round(entity_report["compliance_percentage"], 2),
        },
        "regulation_report_riparian": {
            "entity_count": regulation_report["entity_count"],
            "compliant_count": regulation_report["compliant_count"],
            "non_compliant_count": regulation_report["non_compliant_count"],
            "compliance_percentage": round(
                regulation_report["compliance_percentage"], 2
            ),
        },
        "summary_report": {
            "entity_count": summary["entity_count"],
            "regulation_count": summary["regulation_count"],
            "total_status_count": summary["total_status_count"],
            "latest_status_count": summary["latest_status_count"],
            "compliant_count": summary["compliant_count"],
            "non_compliant_count": summary["non_compliant_count"],
            "compliance_percentage": round(summary["compliance_percentage"], 2),
            "average_compliance_level": round(summary["average_compliance_level"], 4),
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("NORMS", _operation))
