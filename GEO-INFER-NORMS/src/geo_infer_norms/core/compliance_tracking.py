"""
Compliance tracking module for monitoring and analyzing regulatory compliance.

This module provides classes and functions for tracking compliance with regulations
across geographic areas and jurisdictions.
"""

import geopandas as gpd
from typing import Any
import logging
import matplotlib.pyplot as plt
import datetime
import uuid

from geo_infer_norms.models.compliance_status import ComplianceStatus, ComplianceMetric
from geo_infer_norms.models.regulation import Regulation
from geo_infer_norms.models.legal_entity import LegalEntity

logger = logging.getLogger(__name__)


class ComplianceTracker:
    """
    A class for tracking compliance with regulations across entities and jurisdictions.

    This class manages compliance statuses, provides analysis tools, and generates
    compliance reports for regulatory frameworks.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        compliance_statuses: list[ComplianceStatus] | None = None,
        compliance_metrics: list[ComplianceMetric] | None = None,
    ):
        """
        Initialize a ComplianceTracker instance.

        Args:
            name: Name of the compliance tracker
            description: Description of the tracking system
            compliance_statuses: List of ComplianceStatus objects
            compliance_metrics: List of ComplianceMetric objects for evaluations
        """
        self.name = name
        self.description = description
        self.compliance_statuses = compliance_statuses or []
        self.compliance_metrics = compliance_metrics or []
        self._status_index: dict[tuple[str, str], list[ComplianceStatus]] = {}
        self._metric_index = {m.id: m for m in self.compliance_metrics}

        # Build index for faster lookups
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build indexes for fast lookups."""
        self._status_index = {}

        for status in self.compliance_statuses:
            # Index by entity ID and regulation ID
            entity_reg_key = (status.entity_id, status.regulation_id)
            if entity_reg_key not in self._status_index:
                self._status_index[entity_reg_key] = []

            self._status_index[entity_reg_key].append(status)

    def add_compliance_status(self, status: ComplianceStatus) -> None:
        """
        Add a compliance status to the tracker.

        Args:
            status: The ComplianceStatus object to add
        """
        self.compliance_statuses.append(status)

        # Update indexes
        entity_reg_key = (status.entity_id, status.regulation_id)
        if entity_reg_key not in self._status_index:
            self._status_index[entity_reg_key] = []

        self._status_index[entity_reg_key].append(status)

        logger.info(
            f"Added compliance status for entity {status.entity_id}, regulation {status.regulation_id}"
        )

    def add_compliance_metric(self, metric: ComplianceMetric) -> None:
        """
        Add a compliance metric to the tracker.

        Args:
            metric: The ComplianceMetric object to add
        """
        self.compliance_metrics.append(metric)
        self._metric_index[metric.id] = metric
        logger.info(f"Added compliance metric: {metric.name}")

    def get_entity_compliance(
        self, entity_id: str, as_of_date: datetime.datetime | None = None
    ) -> dict[str, Any]:
        """
        Get compliance status for all regulations for a specific entity.

        Args:
            entity_id: The ID of the entity
            as_of_date: Optional date to get compliance status as of
                       (defaults to current date/time if None)

        Returns:
            A dictionary containing compliance information
        """
        if as_of_date is None:
            as_of_date = datetime.datetime.now()

        # Find all status entries for this entity
        entity_statuses = []
        for (e_id, _), statuses in self._status_index.items():
            if e_id == entity_id:
                entity_statuses.extend(statuses)

        if not entity_statuses:
            return {
                "entity_id": entity_id,
                "status": "unknown",
                "message": "No compliance records found for entity",
                "compliance_count": 0,
                "non_compliance_count": 0,
                "regulations": [],
            }

        # Filter to most recent status for each regulation as of the given date
        regulation_latest: dict[str, ComplianceStatus] = {}

        for status in entity_statuses:
            if status.timestamp > as_of_date:
                continue

            reg_id = status.regulation_id
            if (
                reg_id not in regulation_latest
                or status.timestamp > regulation_latest[reg_id].timestamp
            ):
                regulation_latest[reg_id] = status

        # Compile results
        compliant_count = sum(1 for s in regulation_latest.values() if s.is_compliant)
        non_compliant_count = len(regulation_latest) - compliant_count

        regulations_status = []
        for reg_id, status in regulation_latest.items():
            regulations_status.append(
                {
                    "regulation_id": reg_id,
                    "is_compliant": status.is_compliant,
                    "compliance_level": status.compliance_level,
                    "notes": status.notes,
                    "timestamp": status.timestamp.isoformat(),
                }
            )

        overall_status = "compliant" if non_compliant_count == 0 else "non_compliant"

        return {
            "entity_id": entity_id,
            "status": overall_status,
            "compliance_count": compliant_count,
            "non_compliance_count": non_compliant_count,
            "compliance_percentage": (compliant_count / len(regulation_latest)) * 100
            if regulation_latest
            else 0,
            "as_of_date": as_of_date.isoformat(),
            "regulations": regulations_status,
        }

    def get_regulation_compliance(
        self, regulation_id: str, as_of_date: datetime.datetime | None = None
    ) -> dict[str, Any]:
        """
        Get compliance status for all entities for a specific regulation.

        Args:
            regulation_id: The ID of the regulation
            as_of_date: Optional date to get compliance status as of
                       (defaults to current date/time if None)

        Returns:
            A dictionary containing compliance information
        """
        if as_of_date is None:
            as_of_date = datetime.datetime.now()

        # Find all status entries for this regulation
        regulation_statuses = []
        for (_, r_id), statuses in self._status_index.items():
            if r_id == regulation_id:
                regulation_statuses.extend(statuses)

        if not regulation_statuses:
            return {
                "regulation_id": regulation_id,
                "status": "unknown",
                "message": "No compliance records found for regulation",
                "entity_count": 0,
                "compliant_count": 0,
                "non_compliant_count": 0,
                "entities": [],
            }

        # Filter to most recent status for each entity as of the given date
        entity_latest: dict[str, ComplianceStatus] = {}

        for status in regulation_statuses:
            if status.timestamp > as_of_date:
                continue

            entity_id = status.entity_id
            if (
                entity_id not in entity_latest
                or status.timestamp > entity_latest[entity_id].timestamp
            ):
                entity_latest[entity_id] = status

        # Compile results
        compliant_count = sum(1 for s in entity_latest.values() if s.is_compliant)
        non_compliant_count = len(entity_latest) - compliant_count

        entities_status = []
        for entity_id, status in entity_latest.items():
            entities_status.append(
                {
                    "entity_id": entity_id,
                    "is_compliant": status.is_compliant,
                    "compliance_level": status.compliance_level,
                    "notes": status.notes,
                    "timestamp": status.timestamp.isoformat(),
                }
            )

        return {
            "regulation_id": regulation_id,
            "entity_count": len(entity_latest),
            "compliant_count": compliant_count,
            "non_compliant_count": non_compliant_count,
            "compliance_percentage": (compliant_count / len(entity_latest)) * 100
            if entity_latest
            else 0,
            "as_of_date": as_of_date.isoformat(),
            "entities": entities_status,
        }

    def evaluate_compliance(
        self,
        entity: LegalEntity,
        regulation: Regulation,
        evaluation_data: dict[str, Any],
    ) -> ComplianceStatus:
        """
        Evaluate compliance of an entity with a regulation based on data.

        Args:
            entity: The LegalEntity to evaluate
            regulation: The Regulation to evaluate against
            evaluation_data: Dictionary of data points for evaluation

        Returns:
            A new ComplianceStatus object with the evaluation results
        """
        # Find applicable metrics for this regulation
        applicable_metrics = [
            m for m in self.compliance_metrics if m.regulation_id == regulation.id
        ]

        if not applicable_metrics:
            logger.warning(
                f"No compliance metrics found for regulation: {regulation.id}"
            )
            return ComplianceStatus(
                id=str(uuid.uuid4()),
                entity_id=entity.id,
                regulation_id=regulation.id,
                is_compliant=False,
                compliance_level=0.0,
                timestamp=datetime.datetime.now(),
                notes="No compliance metrics available for evaluation",
            )

        # Evaluate each metric
        metric_results: list[dict[str, Any]] = []
        for metric in applicable_metrics:
            # A missing primary field is treated as missing data: no verdict is possible.
            if metric.primary_field not in evaluation_data:
                logger.warning(f"Missing primary field for metric: {metric.name}")
                metric_results.append(
                    {
                        "metric_id": metric.id,
                        "name": metric.name,
                        "is_compliant": False,
                        "compliance_level": 0.0,
                        "notes": f"Missing primary field: {metric.primary_field}",
                    }
                )
                continue

            # Check if required data is available
            if not all(field in evaluation_data for field in metric.required_fields):
                logger.warning(f"Missing required data for metric: {metric.name}")
                metric_results.append(
                    {
                        "metric_id": metric.id,
                        "name": metric.name,
                        "is_compliant": False,
                        "compliance_level": 0.0,
                        "notes": f"Missing required data: {', '.join(set(metric.required_fields) - set(evaluation_data.keys()))}",
                    }
                )
                continue

            # Evaluate metric by its declared type (threshold / range / boolean)
            compliance_level = 0.0
            is_compliant = False
            notes = ""

            try:
                if metric.evaluation_type == "threshold":
                    value = evaluation_data.get(metric.primary_field, 0)
                    threshold = metric.threshold_value

                    if metric.comparison == "greater_than":
                        is_compliant = value > threshold
                    elif metric.comparison == "less_than":
                        is_compliant = value < threshold
                    elif metric.comparison == "equal":
                        is_compliant = (
                            abs(value - threshold) < 0.001
                        )  # Approximate equality
                    else:
                        raise ValueError(
                            f"Metric '{metric.name}': unknown comparison "
                            f"'{metric.comparison}' for threshold evaluation"
                        )

                    compliance_level = 1.0 if is_compliant else 0.0
                    notes = f"Value {value} {'meets' if is_compliant else 'does not meet'} threshold {threshold}"

                elif metric.evaluation_type == "range":
                    value = evaluation_data.get(metric.primary_field, 0)
                    min_val = metric.range_min
                    max_val = metric.range_max

                    is_compliant = min_val <= value <= max_val

                    if is_compliant:
                        compliance_level = 1.0
                    else:
                        # Calculate how far outside the range
                        if value < min_val:
                            distance = (min_val - value) / (
                                min_val if min_val != 0 else 1
                            )
                        else:
                            distance = (value - max_val) / (
                                max_val if max_val != 0 else 1
                            )
                        compliance_level = max(0.0, 1.0 - min(distance, 1.0))

                    notes = f"Value {value} is {'within' if is_compliant else 'outside'} range [{min_val}, {max_val}]"

                elif metric.evaluation_type == "boolean":
                    is_compliant = bool(
                        evaluation_data.get(metric.primary_field, False)
                    )
                    compliance_level = 1.0 if is_compliant else 0.0
                    notes = f"Boolean condition is {'satisfied' if is_compliant else 'not satisfied'}"
                else:
                    raise ValueError(
                        f"Metric '{metric.name}': unknown evaluation_type "
                        f"'{metric.evaluation_type}'"
                    )

            except Exception as e:
                logger.error(f"Error evaluating metric {metric.name}: {str(e)}")
                # An evaluator crash is distinct from a genuine verdict: record
                # an error entry that can never be read as is_compliant=False.
                metric_results.append(
                    {
                        "metric_id": metric.id,
                        "name": metric.name,
                        "is_compliant": None,
                        "compliance_level": None,
                        "weight": metric.weight,
                        "notes": f"Evaluation error: {str(e)}",
                        "error": str(e),
                    }
                )
                continue

            metric_results.append(
                {
                    "metric_id": metric.id,
                    "name": metric.name,
                    "is_compliant": is_compliant,
                    "compliance_level": compliance_level,
                    "weight": metric.weight,
                    "notes": notes,
                }
            )

        # Calculate overall compliance. Error entries (is_compliant None) are
        # neither compliant nor non-compliant: they are excluded from scoring,
        # and any evaluation error prevents an overall compliant verdict.
        evaluated_results = [m for m in metric_results if m["is_compliant"] is not None]
        error_results = [m for m in metric_results if m["is_compliant"] is None]
        compliant_metrics = [m for m in evaluated_results if m["is_compliant"]]
        overall_compliant = not error_results and (
            len(compliant_metrics) == len(evaluated_results)
        )

        # Weight the evaluated metrics based on their configured weight
        if evaluated_results:
            total_weight = sum(m.get("weight", 1.0) for m in evaluated_results)
            weighted_sum = sum(
                m["compliance_level"] * m.get("weight", 1.0) for m in evaluated_results
            )
            overall_compliance_level = (
                weighted_sum / total_weight if total_weight > 0 else 0.0
            )
        else:
            overall_compliance_level = 0.0

        status = ComplianceStatus(
            id=str(uuid.uuid4()),
            entity_id=entity.id,
            regulation_id=regulation.id,
            is_compliant=overall_compliant,
            compliance_level=overall_compliance_level,
            timestamp=datetime.datetime.now(),
            notes=(
                f"Evaluated {len(evaluated_results)} metrics. "
                f"{len(compliant_metrics)} compliant."
                + (
                    f" {len(error_results)} metric evaluation(s) failed with errors."
                    if error_results
                    else ""
                )
            ),
            metric_results=metric_results,
        )

        # Add the status to the tracker
        self.add_compliance_status(status)

        return status

    def export_compliance_to_geodataframe(
        self,
        entities: list[LegalEntity],
        regulation_id: str | None = None,
        as_of_date: datetime.datetime | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Export compliance status to a GeoDataFrame for spatial analysis.

        Args:
            entities: List of LegalEntity objects with geometries
            regulation_id: Optional regulation ID to filter by
            as_of_date: Optional date to get compliance status as of

        Returns:
            A GeoDataFrame with compliance status and geometries
        """
        if as_of_date is None:
            as_of_date = datetime.datetime.now()

        data = []

        for entity in entities:
            if entity.geometry is None:
                continue

            # Get compliance for this entity
            if regulation_id:
                # Filter to a specific regulation
                statuses = self._status_index.get((entity.id, regulation_id), [])
                if not statuses:
                    continue

                # Get most recent status as of the given date
                valid_statuses = [s for s in statuses if s.timestamp <= as_of_date]
                if not valid_statuses:
                    continue

                status = max(valid_statuses, key=lambda s: s.timestamp)

                data.append(
                    {
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "regulation_id": regulation_id,
                        "is_compliant": status.is_compliant,
                        "compliance_level": status.compliance_level,
                        "timestamp": status.timestamp,
                        "geometry": entity.geometry,
                    }
                )
            else:
                # Get overall compliance for all regulations
                compliance_info = self.get_entity_compliance(entity.id, as_of_date)

                if not compliance_info.get("regulations"):
                    # No recorded statuses: skip rather than emit an empty row.
                    continue

                data.append(
                    {
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "compliant_count": compliance_info["compliance_count"],
                        "non_compliant_count": compliance_info["non_compliance_count"],
                        "compliance_percentage": compliance_info[
                            "compliance_percentage"
                        ],
                        "overall_status": compliance_info["status"],
                        "geometry": entity.geometry,
                    }
                )

        if not data:
            logger.warning(
                "No compliance data with geometry found for GeoDataFrame export"
            )
            return gpd.GeoDataFrame()

        return gpd.GeoDataFrame(data, crs="EPSG:4326")

    def visualize_compliance(
        self,
        compliance_gdf: gpd.GeoDataFrame,
        column: str = "compliance_level",
        figsize: tuple[int, int] = (12, 8),
        cmap: str = "RdYlGn",
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create a visualization of compliance data.

        Args:
            compliance_gdf: GeoDataFrame with compliance data
            column: Column to visualize
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for visualization
            save_path: Optional path to save the figure

        Returns:
            A matplotlib Figure object
        """
        if compliance_gdf.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No compliance data available for visualization",
                ha="center",
                va="center",
            )
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            return fig

        if column not in compliance_gdf.columns:
            logger.warning(f"Column '{column}' not found in GeoDataFrame")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                f"Column '{column}' not found in data",
                ha="center",
                va="center",
            )
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            return fig

        fig, ax = plt.subplots(figsize=figsize)

        # Determine if we should use categorical or continuous mapping
        if column == "is_compliant" or column == "overall_status":
            # Categorical mapping
            categorical_cmap = {
                "compliant": "#1a9850",
                "non_compliant": "#d73027",
                True: "#1a9850",
                False: "#d73027",
            }

            for _, entity in compliance_gdf.iterrows():
                color = categorical_cmap.get(
                    entity[column], "#808080"
                )  # Grey for unknown values
                ax.plot(*entity.geometry.exterior.xy, color="black", linewidth=0.5)
                ax.fill(*entity.geometry.exterior.xy, color=color, alpha=0.7)

            # Add legend
            if column == "is_compliant":
                patches = [
                    plt.Rectangle((0, 0), 1, 1, fc="#1a9850", alpha=0.7),
                    plt.Rectangle((0, 0), 1, 1, fc="#d73027", alpha=0.7),
                ]
                ax.legend(patches, ["Compliant", "Non-Compliant"], loc="upper right")
            else:
                patches = [
                    plt.Rectangle((0, 0), 1, 1, fc="#1a9850", alpha=0.7),
                    plt.Rectangle((0, 0), 1, 1, fc="#d73027", alpha=0.7),
                    plt.Rectangle((0, 0), 1, 1, fc="#808080", alpha=0.7),
                ]
                ax.legend(
                    patches,
                    ["Compliant", "Non-Compliant", "Unknown"],
                    loc="upper right",
                )
        else:
            # Continuous mapping
            compliance_gdf.plot(
                column=column,
                ax=ax,
                legend=True,
                cmap=cmap,
                legend_kwds={"label": column.replace("_", " ").title()},
            )

        ax.set_title(f"Compliance Status: {column.replace('_', ' ').title()}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def __repr__(self) -> str:
        return f"ComplianceTracker(name='{self.name}', statuses={len(self.compliance_statuses)}, metrics={len(self.compliance_metrics)})"


class ComplianceReport:
    """
    A class for generating detailed compliance reports.

    This class provides methods for generating various types of compliance reports,
    including summary reports, detailed entity reports, and trend analyses.
    """

    def __init__(
        self,
        compliance_tracker: ComplianceTracker,
        title: str = "Compliance Report",
        description: str = "",
    ):
        """
        Initialize a ComplianceReport instance.

        Args:
            compliance_tracker: The ComplianceTracker to generate reports from
            title: Title of the report
            description: Description of the report
        """
        self.compliance_tracker = compliance_tracker
        self.title = title
        self.description = description
        self.generated_at = datetime.datetime.now()

    def generate_summary_report(
        self, as_of_date: datetime.datetime | None = None
    ) -> dict[str, Any]:
        """
        Generate a summary compliance report.

        Args:
            as_of_date: Optional date to generate report as of

        Returns:
            A dictionary containing the summary report
        """
        if as_of_date is None:
            as_of_date = datetime.datetime.now()

        # Count unique entities and regulations
        entity_ids = set()
        regulation_ids = set()

        for (
            entity_id,
            regulation_id,
        ), _ in self.compliance_tracker._status_index.items():
            entity_ids.add(entity_id)
            regulation_ids.add(regulation_id)

        # Calculate overall compliance statistics
        total_statuses = len(self.compliance_tracker.compliance_statuses)
        recent_statuses = [
            s
            for s in self.compliance_tracker.compliance_statuses
            if s.timestamp <= as_of_date
        ]

        if not recent_statuses:
            return {
                "title": self.title,
                "description": self.description,
                "generated_at": self.generated_at.isoformat(),
                "as_of_date": as_of_date.isoformat(),
                "status": "error",
                "message": "No compliance data found for the specified date",
            }

        # Get the most recent status for each entity-regulation pair
        latest_statuses: dict[tuple[str, str], ComplianceStatus] = {}

        for status in recent_statuses:
            key = (status.entity_id, status.regulation_id)
            if (
                key not in latest_statuses
                or status.timestamp > latest_statuses[key].timestamp
            ):
                latest_statuses[key] = status

        # Calculate compliance statistics
        compliant_count = sum(
            1 for status in latest_statuses.values() if status.is_compliant
        )
        non_compliant_count = len(latest_statuses) - compliant_count

        # Calculate average compliance level
        avg_compliance_level = (
            sum(status.compliance_level for status in latest_statuses.values())
            / len(latest_statuses)
            if latest_statuses
            else 0
        )

        return {
            "title": self.title,
            "description": self.description,
            "generated_at": self.generated_at.isoformat(),
            "as_of_date": as_of_date.isoformat(),
            "entity_count": len(entity_ids),
            "regulation_count": len(regulation_ids),
            "total_status_count": total_statuses,
            "latest_status_count": len(latest_statuses),
            "compliant_count": compliant_count,
            "non_compliant_count": non_compliant_count,
            "compliance_percentage": (compliant_count / len(latest_statuses)) * 100
            if latest_statuses
            else 0,
            "average_compliance_level": avg_compliance_level,
            "tracker_name": self.compliance_tracker.name,
        }

    def generate_entity_report(
        self, entity_id: str, as_of_date: datetime.datetime | None = None
    ) -> dict[str, Any]:
        """
        Generate a detailed compliance report for a specific entity.

        Args:
            entity_id: The ID of the entity
            as_of_date: Optional date to generate report as of

        Returns:
            A dictionary containing the entity report
        """
        if as_of_date is None:
            as_of_date = datetime.datetime.now()

        # Get basic compliance info for the entity
        compliance_info = self.compliance_tracker.get_entity_compliance(
            entity_id, as_of_date
        )

        if compliance_info.get("status") == "unknown":
            return {
                "title": f"Entity Compliance Report: {entity_id}",
                "generated_at": self.generated_at.isoformat(),
                "as_of_date": as_of_date.isoformat(),
                "entity_id": entity_id,
                "status": "error",
                "message": "No compliance data found for entity",
            }

        # Enhance report with trend data
        trend_data = self._calculate_entity_trends(entity_id, as_of_date)

        return {
            "title": f"Entity Compliance Report: {entity_id}",
            "generated_at": self.generated_at.isoformat(),
            "as_of_date": as_of_date.isoformat(),
            "entity_id": entity_id,
            "overall_status": compliance_info["status"],
            "compliance_percentage": compliance_info["compliance_percentage"],
            "compliance_count": compliance_info["compliance_count"],
            "non_compliance_count": compliance_info["non_compliance_count"],
            "regulation_details": compliance_info["regulations"],
            "trend_data": trend_data,
        }

    def generate_regulation_report(
        self, regulation_id: str, as_of_date: datetime.datetime | None = None
    ) -> dict[str, Any]:
        """
        Generate a detailed compliance report for a specific regulation.

        Args:
            regulation_id: The ID of the regulation
            as_of_date: Optional date to generate report as of

        Returns:
            A dictionary containing the regulation report
        """
        if as_of_date is None:
            as_of_date = datetime.datetime.now()

        # Get basic compliance info for the regulation
        compliance_info = self.compliance_tracker.get_regulation_compliance(
            regulation_id, as_of_date
        )

        if compliance_info.get("entity_count", 0) == 0:
            return {
                "title": f"Regulation Compliance Report: {regulation_id}",
                "generated_at": self.generated_at.isoformat(),
                "as_of_date": as_of_date.isoformat(),
                "regulation_id": regulation_id,
                "status": "error",
                "message": "No compliance data found for regulation",
            }

        # Enhance report with trend data
        trend_data = self._calculate_regulation_trends(regulation_id, as_of_date)

        return {
            "title": f"Regulation Compliance Report: {regulation_id}",
            "generated_at": self.generated_at.isoformat(),
            "as_of_date": as_of_date.isoformat(),
            "regulation_id": regulation_id,
            "entity_count": compliance_info["entity_count"],
            "compliant_count": compliance_info["compliant_count"],
            "non_compliant_count": compliance_info["non_compliant_count"],
            "compliance_percentage": compliance_info["compliance_percentage"],
            "entity_details": compliance_info["entities"],
            "trend_data": trend_data,
        }

    def _calculate_entity_trends(
        self, entity_id: str, end_date: datetime.datetime, months_back: int = 12
    ) -> dict[str, Any]:
        """
        Calculate compliance trends for an entity over time.

        Args:
            entity_id: The ID of the entity
            end_date: End date for the trend analysis
            months_back: Number of months to look back

        Returns:
            A dictionary containing trend data
        """
        # Create time points for analysis
        time_points = []
        current_date = end_date

        for i in range(months_back):
            # Move back one month
            if current_date.month == 1:
                new_month = 12
                new_year = current_date.year - 1
            else:
                new_month = current_date.month - 1
                new_year = current_date.year

            current_date = datetime.datetime(new_year, new_month, 1)
            time_points.append(current_date)

        # Reverse to get chronological order
        time_points.reverse()
        time_points.append(end_date)

        # Get compliance data at each time point
        trend_data = []

        for date in time_points:
            compliance_info = self.compliance_tracker.get_entity_compliance(
                entity_id, date
            )

            if compliance_info.get("status") != "unknown":
                trend_data.append(
                    {
                        "date": date.isoformat(),
                        "compliance_percentage": compliance_info[
                            "compliance_percentage"
                        ],
                        "compliance_count": compliance_info["compliance_count"],
                        "non_compliance_count": compliance_info["non_compliance_count"],
                    }
                )

        return {
            "timeline": [point["date"] for point in trend_data],
            "compliance_percentage": [
                point["compliance_percentage"] for point in trend_data
            ],
            "compliance_count": [point["compliance_count"] for point in trend_data],
            "non_compliance_count": [
                point["non_compliance_count"] for point in trend_data
            ],
        }

    def _calculate_regulation_trends(
        self, regulation_id: str, end_date: datetime.datetime, months_back: int = 12
    ) -> dict[str, Any]:
        """
        Calculate compliance trends for a regulation over time.

        Args:
            regulation_id: The ID of the regulation
            end_date: End date for the trend analysis
            months_back: Number of months to look back

        Returns:
            A dictionary containing trend data
        """
        # Create time points for analysis
        time_points = []
        current_date = end_date

        for i in range(months_back):
            # Move back one month
            if current_date.month == 1:
                new_month = 12
                new_year = current_date.year - 1
            else:
                new_month = current_date.month - 1
                new_year = current_date.year

            current_date = datetime.datetime(new_year, new_month, 1)
            time_points.append(current_date)

        # Reverse to get chronological order
        time_points.reverse()
        time_points.append(end_date)

        # Get compliance data at each time point
        trend_data = []

        for date in time_points:
            compliance_info = self.compliance_tracker.get_regulation_compliance(
                regulation_id, date
            )

            if compliance_info.get("entity_count", 0) > 0:
                trend_data.append(
                    {
                        "date": date.isoformat(),
                        "entity_count": compliance_info["entity_count"],
                        "compliance_percentage": compliance_info[
                            "compliance_percentage"
                        ],
                        "compliant_count": compliance_info["compliant_count"],
                        "non_compliant_count": compliance_info["non_compliant_count"],
                    }
                )

        return {
            "timeline": [point["date"] for point in trend_data],
            "entity_count": [point["entity_count"] for point in trend_data],
            "compliance_percentage": [
                point["compliance_percentage"] for point in trend_data
            ],
            "compliant_count": [point["compliant_count"] for point in trend_data],
            "non_compliant_count": [
                point["non_compliant_count"] for point in trend_data
            ],
        }

    def export_report_to_html(self, save_path: str) -> str:
        """
        Export a summary report to HTML format.

        Args:
            save_path: Path to save the HTML report

        Returns:
            Path to the saved HTML file
        """
        # This would be a more extensive implementation in reality
        # Here we just create a simple HTML template
        summary = self.generate_summary_report()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{summary["title"]}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .stats {{ display: flex; flex-wrap: wrap; }}
                .stat-box {{ background-color: #ffffff; border: 1px solid #dee2e6; 
                           border-radius: 5px; padding: 10px; margin: 10px; min-width: 200px; }}
                .compliant {{ color: #28a745; }}
                .non-compliant {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <h1>{summary["title"]}</h1>
            <p>{summary["description"]}</p>
            <p><strong>Generated:</strong> {summary["generated_at"]}</p>
            <p><strong>As of:</strong> {summary["as_of_date"]}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>Entities</h3>
                        <p>{summary["entity_count"]}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Regulations</h3>
                        <p>{summary["regulation_count"]}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Compliance Percentage</h3>
                        <p class="{"compliant" if summary["compliance_percentage"] >= 80 else "non-compliant"}">
                            {summary["compliance_percentage"]:.1f}%
                        </p>
                    </div>
                    <div class="stat-box">
                        <h3>Average Compliance Level</h3>
                        <p class="{"compliant" if summary["average_compliance_level"] >= 0.8 else "non-compliant"}">
                            {summary["average_compliance_level"]:.2f}
                        </p>
                    </div>
                </div>
                
                <h3>Compliance Breakdown</h3>
                <p><strong>Compliant:</strong> {summary["compliant_count"]} entities</p>
                <p><strong>Non-Compliant:</strong> {summary["non_compliant_count"]} entities</p>
            </div>
            
            <p>Report generated from compliance tracker: {summary["tracker_name"]}</p>
        </body>
        </html>
        """

        with open(save_path, "w") as f:
            f.write(html_content)

        return save_path

    def __repr__(self) -> str:
        return f"ComplianceReport(title='{self.title}', tracker='{self.compliance_tracker.name}')"
