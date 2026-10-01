"""
Enhanced disease surveillance with Active Inference.

This module implements advanced disease surveillance using Active Inference
principles for probabilistic reasoning, uncertainty quantification, and
adaptive belief updating.
"""

from typing import Any, cast
from collections import defaultdict
import math
from datetime import timedelta

from geo_infer_health.models import DiseaseReport, PopulationData
from geo_infer_health.core.disease_surveillance import DiseaseHotspotAnalyzer
from geo_infer_health.utils.geospatial_utils import haversine_distance
from geo_infer_health.utils.advanced_geospatial import (
    spatial_clustering,
    calculate_spatial_statistics,
    calculate_spatial_autocorrelation,
    calculate_hotspot_statistics,
)
from geo_infer_health.utils.logging import get_logger

logger = get_logger(__name__)


class ActiveInferenceDiseaseAnalyzer(DiseaseHotspotAnalyzer):
    """
    Advanced disease surveillance using Active Inference principles.

    This analyzer implements probabilistic reasoning for disease surveillance,
    incorporating uncertainty quantification, belief updating, and predictive
    modeling based on Active Inference theory.
    """

    def __init__(
        self,
        reports: list[DiseaseReport],
        population_data: list[PopulationData] | None = None,
    ):
        # Input validation
        if not isinstance(reports, list):
            raise TypeError("Reports must be a list of DiseaseReport objects")

        if population_data is not None and not isinstance(population_data, list):
            raise TypeError("Population data must be a list of PopulationData objects")

        self.reports = sorted(reports, key=lambda r: r.report_date)
        self.population_data = population_data if population_data else []

        # Active Inference parameters
        self.precision_parameter = 1.0  # Precision of beliefs
        self.learning_rate = 0.01  # Learning rate for belief updates
        self.free_energy_threshold = 0.1  # Threshold for belief updating

        # Performance optimization: cache for spatial computations
        self._spatial_cache: dict[Any, Any] = {}
        self._distance_cache: dict[Any, Any] = {}

        # Pre-compute spatial statistics with error handling
        self._spatial_stats = None
        if self.reports:
            try:
                locations = [r.location for r in self.reports]
                self._spatial_stats = calculate_spatial_statistics(locations)
            except Exception as e:
                logger.warning(f"Failed to compute spatial statistics: {e}")
                self._spatial_stats = None

        # Initialize belief states for Active Inference
        self._initialize_belief_states()

        logger.info(
            f"Initialized ActiveInferenceDiseaseAnalyzer with {len(self.reports)} reports"
        )

    def _initialize_belief_states(self) -> None:
        """Initialize belief states for Active Inference."""
        self.belief_states = {
            "disease_activity": 0.5,  # Prior belief about disease activity level
            "transmission_rate": 0.1,  # Belief about transmission rate
            "spatial_clustering": 0.3,  # Belief about spatial clustering
            "temporal_trend": 0.0,  # Belief about temporal trends
            "seasonal_pattern": 0.2,  # Belief about seasonal patterns
            "population_risk": 0.1,  # Belief about population-level risk
        }

        self.belief_precisions = {
            "disease_activity": 1.0,
            "transmission_rate": 1.0,
            "spatial_clustering": 1.0,
            "temporal_trend": 1.0,
            "seasonal_pattern": 1.0,
            "population_risk": 1.0,
        }

        # Historical observations for learning
        self.observation_history: list[dict[str, float]] = []

    def _calculate_free_energy(self, observations: dict[str, float]) -> float:
        """
        Calculate variational free energy for Active Inference.

        Args:
            observations: Dictionary of observed values

        Returns:
            Free energy value
        """
        free_energy = 0.0

        for state_name, observed_value in observations.items():
            if state_name in self.belief_states:
                predicted_value = self.belief_states[state_name]
                precision = self.belief_precisions[state_name]

                # Prediction error
                prediction_error = observed_value - predicted_value

                # Free energy = precision * prediction_error^2 - log(precision)
                free_energy += precision * prediction_error**2 - math.log(precision)

        return free_energy

    def _update_beliefs(self, observations: dict[str, float]) -> None:
        """
        Update belief states using Active Inference.

        Args:
            observations: Dictionary of observed values
        """
        free_energy = self._calculate_free_energy(observations)

        if free_energy > self.free_energy_threshold:
            # Store observation for learning
            self.observation_history.append(observations.copy())

            # Update beliefs based on prediction errors
            for state_name, observed_value in observations.items():
                if state_name in self.belief_states:
                    prediction_error = observed_value - self.belief_states[state_name]
                    precision = self.belief_precisions[state_name]

                    # Update belief using learning rate
                    belief_update = self.learning_rate * precision * prediction_error
                    self.belief_states[state_name] += belief_update

                    # Update precision (increases with consistent observations)
                    precision_update = self.learning_rate * (
                        prediction_error**2 - 1 / precision
                    )
                    self.belief_precisions[state_name] = max(
                        0.1, precision + precision_update
                    )

    def _extract_observations(
        self, reports_subset: list[DiseaseReport]
    ) -> dict[str, float]:
        """
        Extract observational features from disease reports for belief updating.

        Args:
            reports_subset: Subset of reports to analyze

        Returns:
            Dictionary of observational features
        """
        if not reports_subset:
            return {}

        # Calculate spatial clustering
        locations = [r.location for r in reports_subset]
        clusters = spatial_clustering(locations, eps_km=1.0, min_samples=3)
        clustering_ratio = len(clusters) / len(locations) if locations else 0

        # Calculate temporal patterns
        if len(reports_subset) > 1:
            time_span = (
                reports_subset[-1].report_date - reports_subset[0].report_date
            ).days
            if time_span > 0:
                temporal_density = len(reports_subset) / time_span
            else:
                temporal_density = len(reports_subset)
        else:
            temporal_density = 0

        # Calculate case intensity
        total_cases = sum(r.case_count for r in reports_subset)
        case_intensity = total_cases / len(reports_subset) if reports_subset else 0

        # Calculate spatial autocorrelation if enough data
        morans_i = 0.0
        if len(locations) >= 10:
            values = [float(r.case_count) for r in reports_subset]
            autocorr_result = calculate_spatial_autocorrelation(
                locations, values, max_distance_km=5.0
            )
            morans_i = autocorr_result.get("morans_i", 0.0)

        return {
            "disease_activity": min(1.0, case_intensity / 10.0),  # Normalize to [0,1]
            "spatial_clustering": clustering_ratio,
            "temporal_trend": min(1.0, temporal_density / 5.0),  # Normalize to [0,1]
            "transmission_rate": morans_i if morans_i > 0 else 0.0,
            "seasonal_pattern": self._calculate_seasonal_pattern(reports_subset),
            "population_risk": self._calculate_population_risk(reports_subset),
        }

    def _calculate_seasonal_pattern(self, reports: list[DiseaseReport]) -> float:
        """Calculate seasonal pattern strength."""
        if len(reports) < 10:
            return 0.0

        # Simple seasonal analysis based on month distribution
        monthly_counts: dict[int, int] = defaultdict(int)
        for report in reports:
            month = report.report_date.month
            monthly_counts[month] += report.case_count

        if not monthly_counts:
            return 0.0

        # Calculate variance in monthly distribution
        counts = list(monthly_counts.values())
        mean_count = sum(counts) / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)

        # Normalize variance to [0,1] scale
        max_possible_variance = (mean_count * 2) ** 2 if mean_count > 0 else 1
        seasonal_strength = min(1.0, variance / max_possible_variance)

        return seasonal_strength

    def _calculate_population_risk(self, reports: list[DiseaseReport]) -> float:
        """Calculate population-level risk indicators."""
        if not self.population_data or not reports:
            return 0.0

        # Simple risk calculation based on case density relative to population
        total_cases = sum(r.case_count for r in reports)
        total_population = sum(p.population_count for p in self.population_data)

        if total_population == 0:
            return 0.0

        case_rate = total_cases / total_population * 100000  # Per 100k population

        # Normalize to [0,1] scale (assuming 1000 cases per 100k is very high risk)
        risk_level = min(1.0, case_rate / 1000.0)

        return risk_level

    def analyze_with_active_inference(
        self, time_window_days: int | None = None
    ) -> dict[str, Any]:
        """
        Perform comprehensive disease analysis using Active Inference.

        Args:
            time_window_days: Optional time window for analysis

        Returns:
            Dictionary containing analysis results
        """
        from geo_infer_health.utils.logging import PerformanceLogger

        # Input validation happens before any error-swallowing try block so
        # invalid arguments surface to the caller.
        if time_window_days is not None and (
            not isinstance(time_window_days, int) or time_window_days < 0
        ):
            raise ValueError("time_window_days must be a non-negative integer")

        with PerformanceLogger("active_inference_analysis", log_threshold=1.0):
            try:
                # Filter reports by time window if specified
                analysis_reports = self.reports
                if time_window_days and self.reports:
                    try:
                        cutoff_date = self.reports[-1].report_date - timedelta(
                            days=time_window_days
                        )
                        analysis_reports = [
                            r for r in self.reports if r.report_date >= cutoff_date
                        ]
                        logger.debug(
                            f"Filtered to {len(analysis_reports)} reports in time window"
                        )
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Error filtering by time window: {e}")
                        analysis_reports = self.reports

                # Stage-failure ledger: appended by each guarded stage below
                # so callers can distinguish degraded output from real values.
                failed_stages: list[str] = []

                # Extract observations with error handling
                try:
                    observations = self._extract_observations(analysis_reports)
                except Exception as e:
                    logger.error(f"Failed to extract observations: {e}")
                    observations = {}
                    failed_stages.append("extract_observations")

                # Update beliefs using Active Inference
                if observations:
                    try:
                        self._update_beliefs(observations)
                    except Exception as e:
                        logger.error(f"Failed to update beliefs: {e}")
                        failed_stages.append("update_beliefs")

                # Perform analyses with error handling
                results: dict[str, Any] = {
                    "belief_states": self.belief_states.copy(),
                    "belief_precisions": self.belief_precisions.copy(),
                    "observations": observations,
                    "failed_stages": failed_stages,
                }

                try:
                    results["traditional_hotspots"] = (
                        self._traditional_hotspot_analysis(analysis_reports)
                    )
                except Exception as e:
                    logger.error(f"Traditional hotspot analysis failed: {e}")
                    results["traditional_hotspots"] = []
                    failed_stages.append("traditional_hotspots")

                try:
                    results["enhanced_hotspots"] = self._enhanced_hotspot_analysis(
                        analysis_reports
                    )
                except Exception as e:
                    logger.error(f"Enhanced hotspot analysis failed: {e}")
                    results["enhanced_hotspots"] = []
                    failed_stages.append("enhanced_hotspots")

                try:
                    results["predictions"] = self._generate_predictions(
                        analysis_reports
                    )
                except Exception as e:
                    logger.error(f"Prediction generation failed: {e}")
                    results["predictions"] = {"short_term_risk": 0.5, "trend": "error"}
                    failed_stages.append("predictions")

                try:
                    results["confidence_intervals"] = (
                        self._calculate_confidence_intervals(analysis_reports)
                    )
                except Exception as e:
                    logger.error(f"Confidence interval calculation failed: {e}")
                    results["confidence_intervals"] = {
                        "incidence_rate": {"lower": 0, "upper": 0, "confidence": 0}
                    }
                    failed_stages.append("confidence_intervals")

                try:
                    results["risk_assessment"] = self._assess_overall_risk(
                        analysis_reports
                    )
                except Exception as e:
                    logger.error(f"Risk assessment failed: {e}")
                    results["risk_assessment"] = {"risk_level": "unknown", "score": 0.5}
                    failed_stages.append("risk_assessment")

                try:
                    results["recommendations"] = self._generate_recommendations()
                except Exception as e:
                    logger.error(f"Recommendation generation failed: {e}")
                    results["recommendations"] = [
                        "Analysis completed with errors - review logs"
                    ]
                    failed_stages.append("recommendations")

                results["degraded"] = bool(failed_stages)

                logger.info(
                    f"Active Inference analysis completed for {len(analysis_reports)} reports"
                )
                return results

            except Exception as e:
                logger.error(f"Critical error in Active Inference analysis: {e}")
                # Return minimal results on critical failure
                return {
                    "error": str(e),
                    "belief_states": (
                        self.belief_states.copy()
                        if isinstance(self.belief_states, dict)
                        else {}
                    ),
                    "traditional_hotspots": [],
                    "enhanced_hotspots": [],
                    "predictions": {"short_term_risk": 0.5, "trend": "error"},
                    "recommendations": ["Analysis failed - check system logs"],
                    "failed_stages": ["analysis"],
                    "degraded": True,
                }

    def _traditional_hotspot_analysis(self, reports: list[DiseaseReport]) -> list[dict]:
        """Perform traditional hotspot analysis."""
        if not reports:
            return []

        locations = [r.location for r in reports]
        case_counts = [r.case_count for r in reports]

        return cast(
            list[dict[str, Any]],
            calculate_hotspot_statistics(locations, case_counts)["hotspots"],
        )

    def _enhanced_hotspot_analysis(self, reports: list[DiseaseReport]) -> list[dict]:
        """Perform enhanced hotspot analysis with Active Inference."""
        if not reports:
            return []

        # Use belief states to weight the analysis
        clustering_belief = self.belief_states.get("spatial_clustering", 0.5)
        activity_belief = self.belief_states.get("disease_activity", 0.5)

        # Adjust analysis parameters based on beliefs
        scan_radius = 1.0 * (
            1 + clustering_belief
        )  # Increase radius if clustering expected
        threshold = 5 * (
            1 + activity_belief
        )  # Adjust threshold based on activity belief

        hotspots = []
        for i, report in enumerate(reports):
            center_loc = report.location
            nearby_reports = [
                r
                for r in reports
                if haversine_distance(r.location, center_loc) <= scan_radius
            ]

            total_cases = sum(r.case_count for r in nearby_reports)

            if total_cases >= threshold:
                # Calculate confidence based on belief precision
                precision = self.belief_precisions.get("spatial_clustering", 1.0)
                confidence = min(1.0, precision / 2.0)

                hotspots.append(
                    {
                        "location": center_loc,
                        "case_count": total_cases,
                        "radius_km": scan_radius,
                        "confidence": confidence,
                        "belief_weighted": True,
                    }
                )

        return hotspots

    def _generate_predictions(self, reports: list[DiseaseReport]) -> dict[str, Any]:
        """Generate predictions using Active Inference."""
        if len(reports) < 5:
            return {"short_term_risk": 0.5, "trend": "insufficient_data"}

        # Simple prediction based on recent trends and beliefs
        recent_reports = reports[-10:]  # Last 10 reports
        recent_rate = sum(r.case_count for r in recent_reports) / len(recent_reports)

        # Factor in belief states
        activity_trend = self.belief_states.get("temporal_trend", 0.0)
        transmission_belief = self.belief_states.get("transmission_rate", 0.1)

        # Predict next period risk
        predicted_risk = recent_rate * (1 + activity_trend) * (1 + transmission_belief)
        predicted_risk = min(1.0, predicted_risk / 20.0)  # Normalize

        # Determine trend
        if activity_trend > 0.1:
            trend = "increasing"
        elif activity_trend < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "short_term_risk": predicted_risk,
            "trend": trend,
            "confidence": self.belief_precisions.get("temporal_trend", 1.0) / 2.0,
            "factors": {
                "recent_activity": recent_rate,
                "transmission_belief": transmission_belief,
                "activity_trend": activity_trend,
            },
        }

    def _calculate_confidence_intervals(
        self, reports: list[DiseaseReport]
    ) -> dict[str, Any]:
        """Calculate confidence intervals for estimates."""
        if not reports:
            return {"incidence_rate": {"lower": 0, "upper": 0, "confidence": 0}}

        case_counts = [r.case_count for r in reports]
        n = len(reports)

        if n < 2:
            return {"incidence_rate": {"lower": 0, "upper": 0, "confidence": 0.5}}

        # Calculate mean and standard error
        mean_cases = sum(case_counts) / n
        variance = sum((c - mean_cases) ** 2 for c in case_counts) / (n - 1)
        std_error = math.sqrt(variance / n)

        # 95% confidence interval
        z_score = 1.96  # 95% confidence
        margin_error = z_score * std_error

        lower_bound = max(0, mean_cases - margin_error)
        upper_bound = mean_cases + margin_error

        # Calculate confidence based on sample size and belief precision
        sample_confidence = min(1.0, n / 30.0)  # Increases with sample size
        belief_confidence = self.belief_precisions.get("disease_activity", 1.0) / 2.0
        overall_confidence = (sample_confidence + belief_confidence) / 2

        return {
            "incidence_rate": {
                "mean": mean_cases,
                "lower": lower_bound,
                "upper": upper_bound,
                "confidence": overall_confidence,
            },
            "sample_size": n,
            "standard_error": std_error,
        }

    def _assess_overall_risk(self, reports: list[DiseaseReport]) -> dict[str, Any]:
        """Assess overall disease risk."""
        if not reports:
            return {"risk_level": "low", "score": 0.0}

        # Combine multiple risk factors
        activity_risk = self.belief_states.get("disease_activity", 0.5)
        clustering_risk = self.belief_states.get("spatial_clustering", 0.3)
        transmission_risk = self.belief_states.get("transmission_rate", 0.1)

        # Weighted risk score
        risk_score = (
            activity_risk * 0.4 + clustering_risk * 0.3 + transmission_risk * 0.3
        )

        # Determine risk level
        if risk_score > 0.7:
            risk_level = "high"
        elif risk_score > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_level": risk_level,
            "score": risk_score,
            "factors": {
                "activity": activity_risk,
                "clustering": clustering_risk,
                "transmission": transmission_risk,
            },
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on belief states."""
        recommendations = []

        activity_level = self.belief_states.get("disease_activity", 0.5)
        clustering_level = self.belief_states.get("spatial_clustering", 0.3)
        transmission_rate = self.belief_states.get("transmission_rate", 0.1)

        if activity_level > 0.7:
            recommendations.append(
                "High disease activity detected - implement immediate intervention measures"
            )
            recommendations.append("Increase surveillance frequency in affected areas")

        if clustering_level > 0.6:
            recommendations.append(
                "Strong spatial clustering suggests targeted interventions"
            )
            recommendations.append("Focus resources on identified hotspot areas")

        if transmission_rate > 0.3:
            recommendations.append(
                "Elevated transmission rate - review contact tracing effectiveness"
            )
            recommendations.append("Consider community-level preventive measures")

        if len(recommendations) == 0:
            recommendations.append(
                "Disease activity within normal ranges - maintain standard surveillance"
            )

        return recommendations
