"""
Compliance frameworks for geospatial data.

This module provides tools for ensuring compliance with various
regulatory requirements for geospatial data handling.
"""

from typing import Any
from collections.abc import Callable
from enum import Enum
import datetime
import json
import logging
import geopandas as gpd


logger = logging.getLogger(__name__)


class ComplianceRegime(Enum):
    """Enumeration of supported compliance regimes."""

    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    NATIONAL_SECURITY = "national_security"
    UTILITIES = "utilities"
    TRANSPORTATION = "transportation"


class ComplianceRule:
    """Represents a compliance rule for geospatial data."""

    def __init__(
        self,
        name: str,
        regime: ComplianceRegime,
        description: str,
        validator: Callable[[Any], bool],
        priority: int = 1,
    ):
        """
        Initialize a compliance rule.

        Args:
            name: Rule name
            regime: Compliance regime this rule belongs to
            description: Human-readable description of the rule
            validator: Function that validates compliance with the rule
            priority: Rule priority (higher values indicate higher priority)
        """
        self.name = name
        self.regime = regime
        self.description = description
        self.validator = validator
        self.priority = priority

    def __repr__(self) -> str:
        """Return string representation of the rule."""
        return f"ComplianceRule({self.name}, {self.regime.value}, priority={self.priority})"

    def check(self, data: Any) -> bool:
        """
        Check if data complies with the rule.

        Args:
            data: Data to validate

        Returns:
            True if compliant, False otherwise
        """
        try:
            return self.validator(data)
        except Exception as e:
            logger.error(f"Error checking compliance rule {self.name}: {str(e)}")
            return False


class ComplianceViolation:
    """Represents a compliance violation."""

    def __init__(
        self,
        rule: ComplianceRule,
        data_reference: str,
        timestamp: datetime.datetime | None = None,
        details: str | None = None,
    ):
        """
        Initialize a compliance violation.

        Args:
            rule: The violated rule
            data_reference: Reference to the data that violated the rule
            timestamp: When the violation was detected
            details: Additional details about the violation
        """
        self.rule = rule
        self.data_reference = data_reference
        self.timestamp = timestamp or datetime.datetime.now(datetime.UTC).replace(
            tzinfo=None
        )
        self.details = details

    def __repr__(self) -> str:
        """Return string representation of the violation."""
        return f"ComplianceViolation({self.rule.name}, {self.data_reference}, {self.timestamp})"

    def to_dict(self) -> dict:
        """Convert the violation to a dictionary."""
        return {
            "rule_name": self.rule.name,
            "regime": self.rule.regime.value,
            "data_reference": self.data_reference,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class ComplianceFramework:
    """Framework for managing geospatial data compliance."""

    def __init__(self) -> None:
        """Initialize the compliance framework."""
        self.rules: dict[str, ComplianceRule] = {}
        self.violations: list[ComplianceViolation] = []

    def add_rule(self, rule: ComplianceRule) -> None:
        """
        Add a compliance rule.

        Args:
            rule: Rule to add
        """
        self.rules[rule.name] = rule

    def get_rules_by_regime(self, regime: ComplianceRegime) -> list[ComplianceRule]:
        """
        Get all rules for a specific compliance regime.

        Args:
            regime: The compliance regime

        Returns:
            List of compliance rules
        """
        return [rule for rule in self.rules.values() if rule.regime == regime]

    def check_compliance(
        self,
        data: Any,
        data_reference: str,
        regimes: list[ComplianceRegime] | None = None,
    ) -> list[ComplianceViolation]:
        """
        Check compliance of data against rules.

        Args:
            data: Data to check
            data_reference: Reference to identify the data
            regimes: Optional list of regimes to check against

        Returns:
            List of compliance violations
        """
        violations = []

        # Determine which rules to check
        if regimes:
            rules_to_check = [r for r in self.rules.values() if r.regime in regimes]
        else:
            rules_to_check = list(self.rules.values())

        # Sort rules by priority
        rules_to_check.sort(key=lambda r: r.priority, reverse=True)

        # Check each rule
        for rule in rules_to_check:
            if not rule.check(data):
                violation = ComplianceViolation(
                    rule=rule,
                    data_reference=data_reference,
                    details=f"Failed compliance check for {rule.regime.value}",
                )
                violations.append(violation)
                self.violations.append(violation)

        return violations

    def check_geodataframe_compliance(
        self,
        gdf: gpd.GeoDataFrame,
        data_reference: str,
        regimes: list[ComplianceRegime] | None = None,
    ) -> list[ComplianceViolation]:
        """
        Check compliance of a GeoDataFrame.

        Args:
            gdf: GeoDataFrame to check
            data_reference: Reference to identify the data
            regimes: Optional list of regimes to check against

        Returns:
            List of compliance violations
        """
        return self.check_compliance(gdf, data_reference, regimes)

    def generate_compliance_report(
        self, output_format: str = "json", output_file: str | None = None
    ) -> str | dict:
        """
        Generate a compliance report.

        Args:
            output_format: Format for the report ('json' or 'html')
            output_file: Optional path to save the report

        Returns:
            Report as a string or dictionary
        """
        report: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.UTC)
            .replace(tzinfo=None)
            .isoformat(),
            "total_rules": len(self.rules),
            "total_violations": len(self.violations),
            "violations_by_regime": {},
            "violations": [v.to_dict() for v in self.violations],
        }

        # Count violations by regime
        for violation in self.violations:
            regime = violation.rule.regime.value
            if regime not in report["violations_by_regime"]:
                report["violations_by_regime"][regime] = 0
            report["violations_by_regime"][regime] += 1

        if output_format == "json":
            result = json.dumps(report, indent=2)
            if output_file:
                with open(output_file, "w") as f:
                    f.write(result)
            return result
        elif output_format == "html":
            # Simple HTML report
            html = f"""
            <html>
            <head><title>Compliance Report</title></head>
            <body>
                <h1>Compliance Report</h1>
                <p>Generated: {report["timestamp"]}</p>
                <h2>Summary</h2>
                <p>Total Rules: {report["total_rules"]}</p>
                <p>Total Violations: {report["total_violations"]}</p>

                <h2>Violations by Regime</h2>
                <ul>
            """

            for regime, count in report["violations_by_regime"].items():
                html += f"<li>{regime}: {count}</li>\n"

            html += """
                </ul>

                <h2>All Violations</h2>
                <table border="1">
                <tr>
                    <th>Rule</th>
                    <th>Regime</th>
                    <th>Data Reference</th>
                    <th>Timestamp</th>
                    <th>Details</th>
                </tr>
            """

            for v in report["violations"]:
                html += f"""
                <tr>
                    <td>{v["rule_name"]}</td>
                    <td>{v["regime"]}</td>
                    <td>{v["data_reference"]}</td>
                    <td>{v["timestamp"]}</td>
                    <td>{v["details"] or ""}</td>
                </tr>
                """

            html += """
                </table>
            </body>
            </html>
            """

            if output_file:
                with open(output_file, "w") as f:
                    f.write(html)

            return html
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def clear_violations(self) -> None:
        """Clear all recorded violations."""
        self.violations = []


# Predefined compliance validators
def create_gdpr_validators() -> dict[str, ComplianceRule]:
    """
    Create GDPR compliance validators.

    Returns:
        Dictionary of compliance rules for GDPR
    """
    validators = {}

    # Personal data minimization validator
    def personal_data_minimization(gdf: gpd.GeoDataFrame) -> bool:
        sensitive_columns = ["name", "address", "phone", "email", "id_number", "ssn"]
        found_columns = [
            col
            for col in gdf.columns
            if any(s in col.lower() for s in sensitive_columns)
        ]
        return len(found_columns) <= 2  # Allow at most 2 PII columns

    validators["gdpr_data_minimization"] = ComplianceRule(
        name="gdpr_data_minimization",
        regime=ComplianceRegime.GDPR,
        description="Minimize personal data collection",
        validator=personal_data_minimization,
        priority=10,
    )

    # Location precision validator (reduce precision for personal locations)
    def location_precision(gdf: gpd.GeoDataFrame) -> bool:
        if "precision" in gdf.columns:
            # If dataset tracks precision, check it's not too high
            return all(gdf["precision"] <= 3)  # Limit precision to city level
        return True

    validators["gdpr_location_precision"] = ComplianceRule(
        name="gdpr_location_precision",
        regime=ComplianceRegime.GDPR,
        description="Limit location precision for personal data",
        validator=location_precision,
        priority=8,
    )

    return validators
