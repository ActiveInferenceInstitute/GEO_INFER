"""
Risk assessment models for geospatial security.

This module provides models and utilities for assessing security risks
in geospatial applications and data.
"""

from enum import Enum
import datetime
import json
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon


class RiskSeverity(Enum):
    """Severity levels for security risks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLikelihood(Enum):
    """Likelihood levels for security risks."""

    RARE = "rare"
    UNLIKELY = "unlikely"
    POSSIBLE = "possible"
    LIKELY = "likely"
    CERTAIN = "certain"


class RiskCategory(Enum):
    """Categories of geospatial security risks."""

    PRIVACY = "privacy"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    REGULATORY = "regulatory"
    INFRASTRUCTURE = "infrastructure"
    DATA_QUALITY = "data_quality"
    GEOPOLITICAL = "geopolitical"


class GeospatialSecurityRisk:
    """Model representing a security risk in a geospatial context."""

    def __init__(
        self,
        name: str,
        description: str,
        category: RiskCategory,
        severity: RiskSeverity,
        likelihood: RiskLikelihood,
        affected_asset: str,
        mitigation_strategies: list[str] | None = None,
        spatial_context: Polygon | MultiPolygon | None = None,
    ):
        """
        Initialize a geospatial security risk.

        Args:
            name: Risk name
            description: Risk description
            category: Risk category
            severity: Risk severity
            likelihood: Risk likelihood
            affected_asset: Asset affected by the risk
            mitigation_strategies: List of mitigation strategies
            spatial_context: Geographic area where the risk applies
        """
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity
        self.likelihood = likelihood
        self.affected_asset = affected_asset
        self.mitigation_strategies = mitigation_strategies or []
        self.spatial_context = spatial_context
        self.created_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        self.updated_at = self.created_at

    def __repr__(self) -> str:
        """Return string representation of the risk."""
        return f"GeospatialSecurityRisk({self.name}, {self.category.value}, {self.severity.value})"

    def calculate_risk_score(self) -> int:
        """
        Calculate a numerical risk score.

        Returns:
            Risk score (higher is more severe)
        """
        # Map severity to score
        severity_scores = {
            RiskSeverity.LOW: 1,
            RiskSeverity.MEDIUM: 2,
            RiskSeverity.HIGH: 3,
            RiskSeverity.CRITICAL: 4,
        }

        # Map likelihood to score
        likelihood_scores = {
            RiskLikelihood.RARE: 1,
            RiskLikelihood.UNLIKELY: 2,
            RiskLikelihood.POSSIBLE: 3,
            RiskLikelihood.LIKELY: 4,
            RiskLikelihood.CERTAIN: 5,
        }

        # Risk score = Severity × Likelihood
        return severity_scores[self.severity] * likelihood_scores[self.likelihood]

    def to_dict(self) -> dict:
        """
        Convert the risk to a dictionary.

        Returns:
            Dictionary representation of the risk
        """
        result = {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "likelihood": self.likelihood.value,
            "affected_asset": self.affected_asset,
            "mitigation_strategies": self.mitigation_strategies,
            "risk_score": self.calculate_risk_score(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        # Add spatial context if available
        if self.spatial_context is not None:
            import shapely.geometry

            result["spatial_context"] = shapely.geometry.mapping(self.spatial_context)

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "GeospatialSecurityRisk":
        """
        Create a risk from a dictionary.

        Args:
            data: Dictionary with risk data

        Returns:
            GeospatialSecurityRisk instance
        """
        # Get spatial context if available
        spatial_context = None
        if "spatial_context" in data:
            import shapely.geometry

            spatial_context = shapely.geometry.shape(data["spatial_context"])

        # Create the risk
        risk = cls(
            name=data["name"],
            description=data["description"],
            category=RiskCategory(data["category"]),
            severity=RiskSeverity(data["severity"]),
            likelihood=RiskLikelihood(data["likelihood"]),
            affected_asset=data["affected_asset"],
            mitigation_strategies=data.get("mitigation_strategies", []),
            spatial_context=spatial_context,
        )

        # Set timestamps if available
        if "created_at" in data:
            risk.created_at = datetime.datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            risk.updated_at = datetime.datetime.fromisoformat(data["updated_at"])

        return risk


class RiskAssessment:
    """Assessment of multiple security risks for a geospatial system."""

    def __init__(self, name: str, description: str | None = None):
        """
        Initialize a risk assessment.

        Args:
            name: Assessment name
            description: Assessment description
        """
        self.name = name
        self.description = description
        self.risks: list[GeospatialSecurityRisk] = []
        self.created_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        self.updated_at = self.created_at

    def add_risk(self, risk: GeospatialSecurityRisk) -> None:
        """
        Add a risk to the assessment.

        Args:
            risk: Risk to add
        """
        self.risks.append(risk)
        self.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)

    def remove_risk(self, risk_name: str) -> bool:
        """
        Remove a risk from the assessment.

        Args:
            risk_name: Name of the risk to remove

        Returns:
            True if the risk was removed, False if not found
        """
        for i, risk in enumerate(self.risks):
            if risk.name == risk_name:
                self.risks.pop(i)
                self.updated_at = datetime.datetime.now(datetime.UTC).replace(
                    tzinfo=None
                )
                return True

        return False

    def get_risk_by_name(self, risk_name: str) -> GeospatialSecurityRisk | None:
        """
        Get a risk by its name.

        Args:
            risk_name: Name of the risk

        Returns:
            The risk if found, None otherwise
        """
        for risk in self.risks:
            if risk.name == risk_name:
                return risk

        return None

    def get_risks_by_category(
        self, category: RiskCategory
    ) -> list[GeospatialSecurityRisk]:
        """
        Get all risks in a specific category.

        Args:
            category: Risk category

        Returns:
            List of risks in the category
        """
        return [risk for risk in self.risks if risk.category == category]

    def get_risks_by_severity(
        self, severity: RiskSeverity
    ) -> list[GeospatialSecurityRisk]:
        """
        Get all risks with a specific severity.

        Args:
            severity: Risk severity

        Returns:
            List of risks with the severity
        """
        return [risk for risk in self.risks if risk.severity == severity]

    def get_risks_by_likelihood(
        self, likelihood: RiskLikelihood
    ) -> list[GeospatialSecurityRisk]:
        """
        Get all risks with a specific likelihood.

        Args:
            likelihood: Risk likelihood

        Returns:
            List of risks with the likelihood
        """
        return [risk for risk in self.risks if risk.likelihood == likelihood]

    def calculate_total_risk_score(self) -> int:
        """
        Calculate the total risk score for the assessment.

        Returns:
            Total risk score
        """
        return sum(risk.calculate_risk_score() for risk in self.risks)

    def get_highest_risks(self, count: int = 5) -> list[GeospatialSecurityRisk]:
        """
        Get the highest-scoring risks.

        Args:
            count: Number of risks to return

        Returns:
            List of the highest-scoring risks
        """
        # Sort risks by score in descending order
        sorted_risks = sorted(
            self.risks, key=lambda r: r.calculate_risk_score(), reverse=True
        )

        # Return the top N risks
        return sorted_risks[:count]

    def to_dict(self) -> dict:
        """
        Convert the assessment to a dictionary.

        Returns:
            Dictionary representation of the assessment
        """
        return {
            "name": self.name,
            "description": self.description,
            "risks": [risk.to_dict() for risk in self.risks],
            "total_risk_score": self.calculate_total_risk_score(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RiskAssessment":
        """
        Create an assessment from a dictionary.

        Args:
            data: Dictionary with assessment data

        Returns:
            RiskAssessment instance
        """
        assessment = cls(name=data["name"], description=data.get("description"))

        # Add risks
        for risk_data in data.get("risks", []):
            risk = GeospatialSecurityRisk.from_dict(risk_data)
            assessment.add_risk(risk)

        # Set timestamps if available
        if "created_at" in data:
            assessment.created_at = datetime.datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            assessment.updated_at = datetime.datetime.fromisoformat(data["updated_at"])

        return assessment

    def to_json(self, indent: int = 2) -> str:
        """
        Convert the assessment to a JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "RiskAssessment":
        """
        Create an assessment from a JSON string.

        Args:
            json_str: JSON string with assessment data

        Returns:
            RiskAssessment instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def generate_risk_matrix(self) -> pd.DataFrame:
        """
        Generate a risk matrix showing the distribution of risks.

        Returns:
            DataFrame representing the risk matrix
        """
        # Create a matrix of severity vs. likelihood
        severities = [s.value for s in RiskSeverity]
        likelihoods = [likelihood.value for likelihood in RiskLikelihood]

        # Initialize matrix with zeros
        matrix = pd.DataFrame(0, index=severities, columns=likelihoods)

        # Count risks in each cell
        for risk in self.risks:
            matrix.loc[risk.severity.value, risk.likelihood.value] += 1

        return matrix

    def generate_risk_report(self, format: str = "text") -> str:
        """
        Generate a risk assessment report.

        Args:
            format: Report format ('text' or 'html')

        Returns:
            Report as a string
        """
        if format == "text":
            report = f"Risk Assessment: {self.name}\n"
            report += f"Description: {self.description or 'N/A'}\n"
            report += f"Date: {self.updated_at.strftime('%Y-%m-%d')}\n\n"

            report += f"Total Risk Score: {self.calculate_total_risk_score()}\n"
            report += f"Number of Risks: {len(self.risks)}\n\n"

            report += "Highest Risks:\n"
            for i, risk in enumerate(self.get_highest_risks(5), 1):
                report += f"{i}. {risk.name} (Score: {risk.calculate_risk_score()})\n"
                report += f"   Severity: {risk.severity.value}, Likelihood: {risk.likelihood.value}\n"
                report += f"   Category: {risk.category.value}\n"
                report += f"   Description: {risk.description}\n\n"

            report += "Risk Categories:\n"
            for category in RiskCategory:
                risks = self.get_risks_by_category(category)
                report += f"{category.value}: {len(risks)} risks\n"

            return report

        elif format == "html":
            report = f"""
            <html>
            <head>
                <title>Risk Assessment: {self.name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .high {{ background-color: #ffcccc; }}
                    .medium {{ background-color: #ffffcc; }}
                    .low {{ background-color: #ccffcc; }}
                </style>
            </head>
            <body>
                <h1>Risk Assessment: {self.name}</h1>
                <p><strong>Description:</strong> {self.description or "N/A"}</p>
                <p><strong>Date:</strong> {self.updated_at.strftime("%Y-%m-%d")}</p>
                <p><strong>Total Risk Score:</strong> {self.calculate_total_risk_score()}</p>
                <p><strong>Number of Risks:</strong> {len(self.risks)}</p>

                <h2>Highest Risks</h2>
                <table>
                    <tr>
                        <th>Risk</th>
                        <th>Score</th>
                        <th>Severity</th>
                        <th>Likelihood</th>
                        <th>Category</th>
                        <th>Description</th>
                    </tr>
            """

            for risk in self.get_highest_risks(5):
                css_class = ""
                if (
                    risk.severity == RiskSeverity.HIGH
                    or risk.severity == RiskSeverity.CRITICAL
                ):
                    css_class = "high"
                elif risk.severity == RiskSeverity.MEDIUM:
                    css_class = "medium"
                else:
                    css_class = "low"

                report += f"""
                    <tr class="{css_class}">
                        <td>{risk.name}</td>
                        <td>{risk.calculate_risk_score()}</td>
                        <td>{risk.severity.value}</td>
                        <td>{risk.likelihood.value}</td>
                        <td>{risk.category.value}</td>
                        <td>{risk.description}</td>
                    </tr>
                """

            report += """
                </table>

                <h2>Risk Categories</h2>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Count</th>
                    </tr>
            """

            for category in RiskCategory:
                risks = self.get_risks_by_category(category)
                report += f"""
                    <tr>
                        <td>{category.value}</td>
                        <td>{len(risks)}</td>
                    </tr>
                """

            report += """
                </table>

                <h2>All Risks</h2>
                <table>
                    <tr>
                        <th>Risk</th>
                        <th>Score</th>
                        <th>Severity</th>
                        <th>Likelihood</th>
                        <th>Category</th>
                        <th>Affected Asset</th>
                    </tr>
            """

            for risk in sorted(
                self.risks, key=lambda r: r.calculate_risk_score(), reverse=True
            ):
                css_class = ""
                if (
                    risk.severity == RiskSeverity.HIGH
                    or risk.severity == RiskSeverity.CRITICAL
                ):
                    css_class = "high"
                elif risk.severity == RiskSeverity.MEDIUM:
                    css_class = "medium"
                else:
                    css_class = "low"

                report += f"""
                    <tr class="{css_class}">
                        <td>{risk.name}</td>
                        <td>{risk.calculate_risk_score()}</td>
                        <td>{risk.severity.value}</td>
                        <td>{risk.likelihood.value}</td>
                        <td>{risk.category.value}</td>
                        <td>{risk.affected_asset}</td>
                    </tr>
                """

            report += """
                </table>
            </body>
            </html>
            """

            return report

        else:
            raise ValueError(f"Unsupported format: {format}")


def create_common_geospatial_risks() -> list[GeospatialSecurityRisk]:
    """
    Create a list of common geospatial security risks.

    Returns:
        List of common security risks
    """
    risks = []

    # Privacy risks
    risks.append(
        GeospatialSecurityRisk(
            name="location_tracking",
            description="Unauthorized tracking of individual movements through location data",
            category=RiskCategory.PRIVACY,
            severity=RiskSeverity.HIGH,
            likelihood=RiskLikelihood.POSSIBLE,
            affected_asset="user_location_data",
            mitigation_strategies=[
                "Implement location data anonymization",
                "Limit precision of location data",
                "Enforce strict retention policies",
            ],
        )
    )

    risks.append(
        GeospatialSecurityRisk(
            name="sensitive_location_disclosure",
            description="Disclosure of sensitive locations (homes, workplaces, etc.)",
            category=RiskCategory.PRIVACY,
            severity=RiskSeverity.HIGH,
            likelihood=RiskLikelihood.POSSIBLE,
            affected_asset="point_of_interest_data",
            mitigation_strategies=[
                "Apply geographic masking techniques",
                "Implement access controls for sensitive locations",
                "Use differential privacy for aggregated data",
            ],
        )
    )

    # Data breach risks
    risks.append(
        GeospatialSecurityRisk(
            name="geospatial_data_breach",
            description="Unauthorized access to sensitive geospatial datasets",
            category=RiskCategory.DATA_BREACH,
            severity=RiskSeverity.CRITICAL,
            likelihood=RiskLikelihood.UNLIKELY,
            affected_asset="geospatial_database",
            mitigation_strategies=[
                "Encrypt sensitive geospatial data",
                "Implement strong access controls",
                "Regular security audits of database",
            ],
        )
    )

    # Unauthorized access risks
    risks.append(
        GeospatialSecurityRisk(
            name="api_security_bypass",
            description="Bypassing security controls on geospatial APIs",
            category=RiskCategory.UNAUTHORIZED_ACCESS,
            severity=RiskSeverity.HIGH,
            likelihood=RiskLikelihood.POSSIBLE,
            affected_asset="geospatial_api",
            mitigation_strategies=[
                "Implement robust API authentication",
                "Rate limiting and request validation",
                "Regular penetration testing of API endpoints",
            ],
        )
    )

    # Regulatory risks
    risks.append(
        GeospatialSecurityRisk(
            name="gdpr_noncompliance",
            description="Non-compliance with GDPR for location data of EU citizens",
            category=RiskCategory.REGULATORY,
            severity=RiskSeverity.HIGH,
            likelihood=RiskLikelihood.POSSIBLE,
            affected_asset="user_location_data",
            mitigation_strategies=[
                "Implement consent management",
                "Data minimization practices",
                "Right to be forgotten mechanisms",
            ],
        )
    )

    # Infrastructure risks
    risks.append(
        GeospatialSecurityRisk(
            name="gis_server_vulnerability",
            description="Security vulnerabilities in GIS server software",
            category=RiskCategory.INFRASTRUCTURE,
            severity=RiskSeverity.MEDIUM,
            likelihood=RiskLikelihood.POSSIBLE,
            affected_asset="gis_server",
            mitigation_strategies=[
                "Regular security patching",
                "Vulnerability scanning",
                "Network segregation for GIS infrastructure",
            ],
        )
    )

    # Data quality risks
    risks.append(
        GeospatialSecurityRisk(
            name="coordinate_tampering",
            description="Malicious tampering with coordinates in geospatial data",
            category=RiskCategory.DATA_QUALITY,
            severity=RiskSeverity.MEDIUM,
            likelihood=RiskLikelihood.UNLIKELY,
            affected_asset="vector_datasets",
            mitigation_strategies=[
                "Implement data validation checks",
                "Digital signatures for geospatial data",
                "Audit logging for data modifications",
            ],
        )
    )

    # Geopolitical risks
    risks.append(
        GeospatialSecurityRisk(
            name="border_data_disputes",
            description="Security issues from disputed borders or territories in maps",
            category=RiskCategory.GEOPOLITICAL,
            severity=RiskSeverity.MEDIUM,
            likelihood=RiskLikelihood.POSSIBLE,
            affected_asset="boundary_data",
            mitigation_strategies=[
                "Multiple boundary datasets for different regions",
                "Clear metadata about boundary sources",
                "Regional-specific content delivery",
            ],
        )
    )

    return risks
