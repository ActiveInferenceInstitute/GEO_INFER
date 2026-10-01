"""
Data validation and quality assurance for GEO-INFER-DATA.

This module provides comprehensive data validation capabilities including
geospatial validation, temporal validation, completeness checks, and
quality assessment.

This module hosts ``DataQualityManager``, which drives the
:class:`~geo_infer_data.utils.validation.GeospatialValidator` (configured via
``ValidationConfig`` and ``ValidationRule``) defined in
``geo_infer_data.utils.validation``; import those names from there.
"""

import logging
from typing import Any
from datetime import datetime, timedelta, UTC
from enum import StrEnum

from ..models.schemas import DataQualityReport, DatasetMetadata, QualityStatus
from ..utils.validation import (
    GeospatialValidator,
    ValidationConfig,
    ValidationRule,
)


logger = logging.getLogger(__name__)


class ValidationLevel(StrEnum):
    """Validation strictness levels."""

    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    STRICT = "strict"


class DataQualityManager:
    """
    Comprehensive data quality management and validation.

    This class provides centralized data quality management including validation,
    monitoring, reporting, and improvement recommendations for geospatial datasets.
    It supports multiple validation strategies, quality scoring, trend analysis,
    and automated improvement suggestions.

    Quality dimensions assessed:
    - **Completeness**: Missing values, required fields, data coverage
    - **Accuracy**: Outlier detection, coordinate validity, value ranges
    - **Consistency**: Data type consistency, duplicates, temporal order
    - **Validity**: Format validation, schema compliance, constraint checks
    - **Timeliness**: Temporal consistency, update frequency, data freshness
    - **Uniqueness**: Duplicate detection, primary key validation

    Features:
    - Multiple validation rule sets (basic, standard, comprehensive)
    - Real-time and batch validation modes
    - Quality trend analysis and reporting
    - Automated improvement recommendations
    - Integration with various data formats and sources
    - Configurable quality thresholds and validation rules
    - Historical quality tracking and analysis

    Attributes:
        config: Validation configuration with rules and thresholds
        validator: GeospatialValidator instance for quality checks
        quality_history: List of historical validation reports
        monitoring_enabled: Whether real-time monitoring is active

    Methods:
        validate_dataset(): Validate a specific dataset with comprehensive assessment
        get_improvement_recommendations(): Generate improvement recommendations
        get_quality_trends(): Analyze quality trends over time
        register_dataset(): Register data and metadata for validation.

    Args:
        validation_rules: Validation rules to apply. Can be a string specifying
            rule set ('basic', 'standard', 'comprehensive') or a list of specific
            rule names. Comprehensive rules include all available validations.
        quality_threshold: Minimum quality score threshold (0.0 to 1.0) for
            determining acceptable data quality. Reports below this threshold
            will trigger improvement recommendations.
        real_time_monitoring: Whether to enable real-time quality monitoring
            and continuous validation. When enabled, provides ongoing quality
            assessment for streaming or frequently updated data.

    Raises:
        ConfigurationError: If validation configuration is invalid
        ValidationError: If validation setup fails

    Examples:
        >>> # Initialize with comprehensive validation
        >>> quality_manager = DataQualityManager(
        ...     validation_rules='comprehensive',
        ...     quality_threshold=0.85,
        ...     real_time_monitoring=True
        ... )
        >>>
        >>> # Validate environmental monitoring dataset
        >>> report = await quality_manager.validate_dataset('environmental_sensors_2023')
        >>> print(f"Overall quality score: {report.overall_score:.2f}")
        >>> print(f"Quality status: {report.overall_score >= 0.85 and 'PASS' or 'FAIL'}")
        >>>
        >>> # Analyze quality by dimension
        >>> for check_name, check in report.checks.items():
        ...     status = "✓ PASS" if check.status == 'pass' else "⚠ WARNING" if check.status == 'warning' else "✗ FAIL"
        ...     print(f"{check_name}: {check.score:.2f} ({status})")
        >>>
        >>> # Get improvement recommendations
        >>> if report.overall_score < 0.85:
        ...     recommendations = quality_manager.get_improvement_recommendations(report)
        ...     print("Improvement recommendations:")
        ...     for rec in recommendations:
        ...         print(f"  - {rec}")
        >>>
        >>> # Analyze quality trends
        >>> trends = quality_manager.get_quality_trends(days=30)
        >>> print(f"Quality trend: {trends['score_trend']}")
        >>> print(f"Average score: {trends['average_score']:.2f}")
    """

    def __init__(
        self,
        validation_rules: str = "comprehensive",
        quality_threshold: float = 0.8,
        real_time_monitoring: bool = True,
    ):
        normalized_rules = self._normalize_validation_rules(validation_rules)
        self.config = ValidationConfig(
            validation_rules=normalized_rules,
            quality_threshold=quality_threshold,
            real_time_monitoring=real_time_monitoring,
        )

        self.validator = GeospatialValidator(self.config)
        self.quality_history: list[DataQualityReport] = []
        self._datasets: dict[str, tuple[Any, DatasetMetadata]] = {}
        self.monitoring_enabled = real_time_monitoring

        logger.info(
            f"Initialized DataQualityManager with {validation_rules} validation rules"
        )

    @staticmethod
    def _normalize_validation_rules(
        validation_rules: str | list[str],
    ) -> list[str]:
        """Expand named rule sets and reject unknown validation rules."""
        available = {rule.value for rule in ValidationRule}
        presets = {
            "basic": ["completeness", "validity"],
            "standard": ["completeness", "accuracy", "consistency", "validity"],
            "comprehensive": [rule.value for rule in ValidationRule],
            "strict": [rule.value for rule in ValidationRule],
        }

        if isinstance(validation_rules, str):
            text = validation_rules.strip()
            if text in presets:
                rules = presets[text]
            else:
                rules = [rule.strip() for rule in text.split(",") if rule.strip()]
        else:
            rules = list(validation_rules)

        unknown = sorted(set(rules) - available)
        if unknown:
            raise ValueError(f"Unknown validation rule(s): {', '.join(unknown)}")
        return rules

    def register_dataset(
        self, dataset_id: str, data: Any, metadata: DatasetMetadata
    ) -> None:
        """Register a stored dataset as the source for quality validation."""
        if not dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not isinstance(metadata, DatasetMetadata):
            raise TypeError("metadata must be a DatasetMetadata instance")
        self._datasets[dataset_id] = (data, metadata)

    async def validate_dataset(self, dataset_id: str) -> DataQualityReport:
        """
        Validate a specific dataset with comprehensive quality assessment.

        This method performs comprehensive validation of a dataset by loading the data,
        applying all configured validation rules, and generating a detailed quality report.
        The validation process includes data integrity checks, quality scoring, and
        improvement recommendations.

        The validation process includes:
        1. **Data Loading**: Load dataset from storage using the dataset_id
        2. **Schema Validation**: Validate data structure and format
        3. **Quality Assessment**: Apply all configured validation rules
        4. **Score Calculation**: Compute overall and dimensional quality scores
        5. **Recommendation Generation**: Create improvement recommendations
        6. **History Tracking**: Store validation results for trend analysis

        Each validation dimension provides specific insights:
        - **Completeness**: Identifies missing values and data gaps
        - **Accuracy**: Detects outliers, invalid coordinates, and data anomalies
        - **Consistency**: Finds duplicates, type mismatches, and logical inconsistencies
        - **Validity**: Validates against schemas, constraints, and business rules

        Args:
            dataset_id: Unique identifier for the dataset to validate. This should
                correspond to a dataset stored in the configured storage backends.
                The method will attempt to load the dataset data and metadata using
                this identifier.

        Returns:
            Comprehensive quality assessment report containing:
            {
                'dataset_id': str,  # Dataset identifier
                'overall_score': float,  # 0.0 to 1.0 overall quality score
                'checks': {
                    'dimension_name': {
                        'score': float,  # 0.0 to 1.0 dimensional score
                        'status': str,  # 'pass', 'warning', 'fail'
                        'issues': list,  # List of identified issues
                        'metadata': dict  # Additional validation metadata
                    }
                },
                'recommendations': list,  # Improvement recommendations
                'generated_at': datetime,  # Validation timestamp
                'assessment_method': str,  # Validation methodology used
                'validation_rules': list  # Rules applied during validation
            }

        Raises:
            DatasetNotFoundError: If dataset_id does not exist in storage
            DataLoadingError: If dataset cannot be loaded
            ValidationError: If validation process fails
            StorageError: If storage backend is unavailable

        Examples:
            >>> # Validate environmental monitoring dataset
            >>> report = await quality_manager.validate_dataset('env_sensors_2023')
            >>>
            >>> # Check overall quality
            >>> print(f"Overall quality: {report.overall_score:.2f}")
            >>> print(f"Quality status: {report.overall_score >= 0.8 and 'EXCELLENT' or 'NEEDS_IMPROVEMENT'}")
            >>>
            >>> # Review dimensional scores
            >>> for dimension, check in report.checks.items():
            ...     status_icon = "✅" if check.status == 'pass' else "⚠️" if check.status == 'warning' else "❌"
            ...     print(f"{status_icon} {dimension}: {check.score:.2f}")
            ...
            ...     # Show specific issues
            ...     for issue in check.issues:
            ...         print(f"   - {issue['type']}: {issue['message']} ({issue.get('severity', 'medium')})")
            >>>
            >>> # Get improvement recommendations
            >>> if report.recommendations:
            ...     print("📋 Recommendations:")
            ...     for rec in report.recommendations:
            ...         print(f"   • {rec}")
        """
        logger.info(f"Validating dataset: {dataset_id}")

        try:
            validation_data, validation_metadata = self._datasets[dataset_id]
        except KeyError as exc:
            raise KeyError(
                f"Dataset {dataset_id!r} has not been registered for validation"
            ) from exc

        # Perform validation
        quality_report = await self.validator.build_quality_report(
            validation_data, validation_metadata
        )
        quality_report.dataset_id = dataset_id

        # Store in history
        self.quality_history.append(quality_report)

        logger.info(f"Dataset validation completed: {quality_report.overall_score:.2f}")
        return quality_report

    def get_improvement_recommendations(self, report: DataQualityReport) -> list[str]:
        """Get improvement recommendations based on quality report."""
        recommendations = []

        if report.overall_score < self.config.quality_threshold:
            recommendations.append(
                "Overall quality below threshold - review data collection process"
            )

        for check_name, check in report.checks.items():
            if check.status == QualityStatus.FAIL:
                recommendations.append(f"Fix {check_name} validation failures")
            elif check.status == QualityStatus.WARNING:
                recommendations.append(f"Address {check_name} warnings")

        return recommendations

    def get_quality_trends(self, days: int = 30) -> dict[str, Any]:
        """Get quality trends over time."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        def as_utc(value: datetime) -> datetime:
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value.astimezone(UTC)

        recent_reports = [
            r for r in self.quality_history if as_utc(r.generated_at) >= cutoff_date
        ]

        if not recent_reports:
            return {"message": "No recent quality reports available"}

        scores = [r.overall_score for r in recent_reports]
        avg_score = sum(scores) / len(scores)

        return {
            "average_score": avg_score,
            "reports_count": len(recent_reports),
            "score_trend": "improving" if scores[-1] > scores[0] else "declining",
            "period_days": days,
        }
