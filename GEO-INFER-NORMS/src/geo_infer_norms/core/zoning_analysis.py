"""
Zoning analysis module for geospatial analysis of zoning laws and land use regulations.

This module provides classes and functions for analyzing zoning regulations,
land use classifications, and their spatial implications.
"""

import geopandas as gpd
from typing import Any
import pandas as pd
import numpy as np
from shapely.geometry import Point, MultiPolygon, LineString
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx

from ..models.zoning import ZoningCode, LandUseType, ZoningDistrict

logger = logging.getLogger(__name__)


class ZoningAnalyzer:
    """
    A class for analyzing zoning regulations and their spatial implications.

    This class provides methods for analyzing zoning districts, assessing compatibility
    between different zoning types, and evaluating zoning changes.
    """

    def __init__(
        self,
        zoning_districts: list[ZoningDistrict] | None = None,
        zoning_codes: list[ZoningCode] | None = None,
    ):
        """
        Initialize a ZoningAnalyzer instance.

        Args:
            zoning_districts: List of ZoningDistrict objects
            zoning_codes: List of ZoningCode objects defining zoning regulations
        """
        self.zoning_districts = zoning_districts or []
        self.zoning_codes = zoning_codes or []
        self._district_index = {d.id: d for d in self.zoning_districts}
        self._code_index = {c.code: c for c in self.zoning_codes}

        # Create compatibility matrix between zoning types
        self._compatibility_matrix = self._build_compatibility_matrix()

    def _build_compatibility_matrix(self) -> dict[str, dict[str, float]]:
        """
        Build a compatibility matrix between different zoning codes.

        Returns:
            A nested dictionary mapping zoning code pairs to compatibility scores (0-1)
        """
        matrix: dict[str, dict[str, float]] = {}

        for code1 in self.zoning_codes:
            matrix[code1.code] = {}
            for code2 in self.zoning_codes:
                # Default compatibility is 0.5
                compatibility = 0.5

                # Same code is fully compatible
                if code1.code == code2.code:
                    compatibility = 1.0
                # Similar categories are more compatible
                elif code1.category == code2.category:
                    compatibility = 0.8
                # Residential and industrial tend to be incompatible
                elif (
                    code1.category == "residential" and code2.category == "industrial"
                ) or (
                    code1.category == "industrial" and code2.category == "residential"
                ):
                    compatibility = 0.1
                # Commercial can be somewhat compatible with residential
                elif (
                    code1.category == "residential" and code2.category == "commercial"
                ) or (
                    code1.category == "commercial" and code2.category == "residential"
                ):
                    compatibility = 0.6

                matrix[code1.code][code2.code] = compatibility

        return matrix

    def add_zoning_district(self, district: ZoningDistrict) -> None:
        """
        Add a zoning district to the analyzer.

        Args:
            district: The ZoningDistrict object to add
        """
        self.zoning_districts.append(district)
        self._district_index[district.id] = district

    def add_zoning_code(self, code: ZoningCode) -> None:
        """
        Add a zoning code to the analyzer.

        Args:
            code: The ZoningCode object to add
        """
        self.zoning_codes.append(code)
        self._code_index[code.code] = code
        # Rebuild compatibility matrix
        self._compatibility_matrix = self._build_compatibility_matrix()

    def get_district_by_id(self, district_id: str) -> ZoningDistrict | None:
        """
        Get a zoning district by its ID.

        Args:
            district_id: The ID of the district

        Returns:
            The ZoningDistrict object or None if not found
        """
        return self._district_index.get(district_id)

    def get_code_by_id(self, code_id: str) -> ZoningCode | None:
        """
        Get a zoning code by its ID.

        Args:
            code_id: The ID of the code

        Returns:
            The ZoningCode object or None if not found
        """
        return self._code_index.get(code_id)

    def get_zoning_at_point(self, point: Point) -> list[ZoningDistrict]:
        """
        Get all zoning districts that contain a specific point.

        Args:
            point: A Shapely Point geometry

        Returns:
            A list of ZoningDistrict objects that contain the point
        """
        containing_districts = []

        for district in self.zoning_districts:
            if district.geometry is not None and district.geometry.contains(point):
                containing_districts.append(district)

        return containing_districts

    def calculate_compatibility(self, code1: str, code2: str) -> float:
        """
        Calculate the compatibility score between two zoning codes.

        Args:
            code1: The first zoning code
            code2: The second zoning code

        Returns:
            A compatibility score between 0 (incompatible) and 1 (fully compatible)
        """
        if (
            code1 not in self._compatibility_matrix
            or code2 not in self._compatibility_matrix
        ):
            logger.warning(
                f"Zoning code {code1} or {code2} not found in compatibility matrix"
            )
            return 0.5  # Default medium compatibility

        return self._compatibility_matrix[code1][code2]

    def analyze_zoning_boundaries(self) -> dict[str, Any]:
        """
        Analyze zoning district boundaries for potential conflicts.

        Returns:
            A dictionary containing analysis results
        """
        if not self.zoning_districts:
            return {
                "status": "error",
                "message": "No zoning districts available for analysis",
            }

        districts_gdf = self.export_districts_to_geodataframe()

        # Find adjacent districts
        adjacency = []
        compatibility_scores = []
        boundary_lengths = []

        for i, district1 in districts_gdf.iterrows():
            for j, district2 in districts_gdf.iterrows():
                if i >= j:  # Skip self-comparisons and duplicates
                    continue

                if district1.geometry.touches(district2.geometry):
                    # Calculate the boundary length
                    boundary = district1.geometry.intersection(district2.geometry)
                    if isinstance(boundary, (LineString, MultiPolygon)):
                        boundary_length = boundary.length
                    else:
                        boundary_length = 0

                    # Get compatibility score
                    compatibility = self.calculate_compatibility(
                        district1.zoning_code, district2.zoning_code
                    )

                    adjacency.append(
                        {
                            "district1_id": district1.id,
                            "district2_id": district2.id,
                            "district1_code": district1.zoning_code,
                            "district2_code": district2.zoning_code,
                            "boundary_length": boundary_length,
                        }
                    )

                    compatibility_scores.append(compatibility)
                    boundary_lengths.append(boundary_length)

        # Calculate statistics
        if not compatibility_scores:
            return {
                "status": "success",
                "message": "No adjacent zoning districts found",
                "adjacency_count": 0,
            }

        avg_compatibility = np.mean(compatibility_scores)
        potential_conflicts = [
            (adj["district1_id"], adj["district2_id"], c_score)
            for adj, c_score in zip(adjacency, compatibility_scores)
            if c_score < 0.3  # Threshold for potential conflict
        ]

        total_boundary_length = sum(boundary_lengths)
        conflict_boundary_length = sum(
            length
            for length, score in zip(boundary_lengths, compatibility_scores)
            if score < 0.3
        )

        return {
            "status": "success",
            "adjacency_count": len(adjacency),
            "average_compatibility": avg_compatibility,
            "potential_conflicts": potential_conflicts,
            "conflict_percentage": (conflict_boundary_length / total_boundary_length)
            * 100
            if total_boundary_length
            else 0,
            "adjacency_details": adjacency,
        }

    def evaluate_zoning_change(self, district_id: str, new_code: str) -> dict[str, Any]:
        """
        Evaluate the impact of changing a district's zoning code.

        Args:
            district_id: The ID of the district to change
            new_code: The new zoning code to apply

        Returns:
            A dictionary containing the evaluation results

        Raises:
            ValueError: If district_id or new_code is empty
            TypeError: If district_id or new_code is not a string
        """
        # Input validation
        if not district_id:
            raise ValueError("district_id cannot be empty")

        if not new_code:
            raise ValueError("new_code cannot be empty")

        if not isinstance(district_id, str):
            raise TypeError("district_id must be a string")

        if not isinstance(new_code, str):
            raise TypeError("new_code must be a string")

        district = self.get_district_by_id(district_id)
        if not district:
            return {
                "status": "error",
                "message": f"District with ID {district_id} not found",
            }

        if new_code not in self._code_index:
            return {"status": "error", "message": f"Zoning code {new_code} not found"}

        old_code = district.zoning_code

        # Get adjacent districts
        districts_gdf = self.export_districts_to_geodataframe()
        district_row = districts_gdf[districts_gdf.id == district_id].iloc[0]

        adjacent_districts = []
        for _, adjacent in districts_gdf.iterrows():
            if adjacent.id != district_id and district_row.geometry.touches(
                adjacent.geometry
            ):
                adjacent_districts.append(adjacent.id)

        # Calculate current and proposed compatibility with adjacent districts
        current_compatibility = []
        proposed_compatibility = []
        adjacent_details = []

        for adj_id in adjacent_districts:
            adj_district = self.get_district_by_id(adj_id)
            if adj_district:
                current_score = self.calculate_compatibility(
                    old_code, adj_district.zoning_code
                )
                proposed_score = self.calculate_compatibility(
                    new_code, adj_district.zoning_code
                )

                current_compatibility.append(current_score)
                proposed_compatibility.append(proposed_score)

                adjacent_details.append(
                    {
                        "district_id": adj_id,
                        "district_name": adj_district.name,
                        "zoning_code": adj_district.zoning_code,
                        "current_compatibility": current_score,
                        "proposed_compatibility": proposed_score,
                        "compatibility_change": proposed_score - current_score,
                    }
                )

        if not current_compatibility:
            return {
                "status": "success",
                "message": "District has no adjacent districts for comparison",
                "district_id": district_id,
                "current_code": old_code,
                "proposed_code": new_code,
                "adjacent_count": 0,
            }

        avg_current_compatibility = np.mean(current_compatibility)
        avg_proposed_compatibility = np.mean(proposed_compatibility)

        return {
            "status": "success",
            "district_id": district_id,
            "district_name": district.name,
            "current_code": old_code,
            "proposed_code": new_code,
            "adjacent_count": len(adjacent_districts),
            "average_current_compatibility": avg_current_compatibility,
            "average_proposed_compatibility": avg_proposed_compatibility,
            "compatibility_change": avg_proposed_compatibility
            - avg_current_compatibility,
            "adjacent_details": adjacent_details,
        }

    def visualize_zoning(
        self,
        figsize: tuple[int, int] = (12, 8),
        highlight_district: str | None = None,
        highlight_color: str = "red",
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create a visualization of zoning districts.

        Args:
            figsize: Figure size as (width, height) in inches
            highlight_district: Optional ID of a district to highlight
            highlight_color: Color for highlighting the selected district
            save_path: Optional path to save the figure

        Returns:
            A matplotlib Figure object
        """
        if not self.zoning_districts:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No zoning districts available for visualization",
                ha="center",
                va="center",
            )
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            return fig

        districts_gdf = self.export_districts_to_geodataframe()

        # Create a color mapping for zoning categories
        unique_categories = districts_gdf["category"].unique()
        cmap = plt.colormaps["tab10"].resampled(len(unique_categories))
        color_dict = {
            cat: f"#{int(cmap(i)[0] * 255):02x}{int(cmap(i)[1] * 255):02x}{int(cmap(i)[2] * 255):02x}"
            for i, cat in enumerate(unique_categories)
        }

        fig, ax = plt.subplots(figsize=figsize)

        # Plot all districts
        for _, district in districts_gdf.iterrows():
            color = color_dict.get(district.category, "#CCCCCC")
            if highlight_district and district.id == highlight_district:
                districts_gdf[districts_gdf.id == highlight_district].plot(
                    ax=ax, color=highlight_color, edgecolor="black", linewidth=2
                )
            else:
                ax.plot(*district.geometry.exterior.xy, color=color, linewidth=1)
                ax.fill(*district.geometry.exterior.xy, color=color, alpha=0.6)

        # Add legend
        patches = [
            plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.6)
            for color in color_dict.values()
        ]
        ax.legend(
            patches, color_dict.keys(), loc="upper right", title="Zoning Categories"
        )

        ax.set_title("Zoning Districts")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.axis("equal")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def export_districts_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Export zoning districts to a GeoDataFrame.

        Returns:
            A GeoDataFrame containing district geometries and properties
        """
        data = []

        for district in self.zoning_districts:
            if district.geometry is not None:
                code_obj = self._code_index.get(district.zoning_code)
                category = code_obj.category if code_obj else "unknown"

                district_dict = {
                    "id": district.id,
                    "name": district.name,
                    "zoning_code": district.zoning_code,
                    "category": category,
                    "geometry": district.geometry,
                }
                data.append(district_dict)

        if not data:
            logger.warning("No districts with geometry found for GeoDataFrame export")
            return gpd.GeoDataFrame()

        return gpd.GeoDataFrame(data, crs="EPSG:4326")

    def get_zoning_statistics(self) -> dict[str, Any]:
        """
        Calculate comprehensive statistics about the zoning districts.

        Returns:
            Dictionary containing zoning statistics including area, population,
            employment, and zoning code distributions.
        """
        if not self.zoning_districts:
            return {"error": "No zoning districts available for analysis"}

        districts_gdf = self.export_districts_to_geodataframe()

        # Basic counts
        stats = {
            "total_districts": len(self.zoning_districts),
            "total_zoning_codes": len(self.zoning_codes),
            "total_area_hectares": districts_gdf.geometry.area.sum() / 10000,
            "total_population": sum(d.population for d in self.zoning_districts),
            "total_employment": sum(d.employment for d in self.zoning_districts),
        }

        # Zoning code distribution
        code_counts = districts_gdf["zoning_code"].value_counts().to_dict()
        stats["zoning_code_distribution"] = code_counts

        # Category distribution
        category_counts = districts_gdf["category"].value_counts().to_dict()
        stats["category_distribution"] = category_counts

        # Area by category
        area_by_category = {}
        for category in category_counts.keys():
            area = (
                districts_gdf[districts_gdf["category"] == category].geometry.area.sum()
                / 10000
            )
            area_by_category[category] = area
        stats["area_by_category_hectares"] = area_by_category

        # Population density statistics
        if stats["total_population"] > 0 and stats["total_area_hectares"] > 0:
            stats["overall_population_density"] = (
                stats["total_population"] / stats["total_area_hectares"]
            )
        else:
            stats["overall_population_density"] = 0

        # Employment density statistics
        if stats["total_employment"] > 0 and stats["total_area_hectares"] > 0:
            stats["overall_employment_density"] = (
                stats["total_employment"] / stats["total_area_hectares"]
            )
        else:
            stats["overall_employment_density"] = 0

        return stats

    def find_zoning_conflicts(self, threshold: float = 0.3) -> list[dict[str, Any]]:
        """
        Identify zoning districts with potential conflicts based on compatibility.

        Args:
            threshold: Compatibility threshold below which conflicts are reported (0-1)

        Returns:
            List of dictionaries describing potential zoning conflicts
        """
        conflicts = []
        districts_gdf = self.export_districts_to_geodataframe()

        # Find adjacent districts
        for i, district1 in districts_gdf.iterrows():
            for j, district2 in districts_gdf.iterrows():
                if i >= j:  # Skip self-comparisons and duplicates
                    continue

                if district1.geometry.touches(district2.geometry):
                    compatibility = self.calculate_compatibility(
                        district1.zoning_code, district2.zoning_code
                    )

                    if compatibility < threshold:
                        conflicts.append(
                            {
                                "district1_id": district1.id,
                                "district1_name": district1.name,
                                "district1_code": district1.zoning_code,
                                "district1_category": district1.category,
                                "district2_id": district2.id,
                                "district2_name": district2.name,
                                "district2_code": district2.zoning_code,
                                "district2_category": district2.category,
                                "compatibility_score": compatibility,
                                "severity": "high" if compatibility < 0.2 else "medium",
                            }
                        )

        return conflicts

    def optimize_zoning_layout(
        self, target_compatibility: float = 0.7
    ) -> dict[str, Any]:
        """
        Suggest zoning layout optimizations to improve compatibility.

        Args:
            target_compatibility: Target compatibility score to achieve

        Returns:
            Dictionary with optimization suggestions and metrics
        """
        conflicts = self.find_zoning_conflicts()

        optimization_suggestions = []

        for conflict in conflicts:
            if conflict["compatibility_score"] < target_compatibility:
                suggestion: dict[str, Any] = {"conflict": conflict, "suggestions": []}

                # Suggest alternative zoning codes
                current_code = self.get_code_by_id(conflict["district2_code"])
                if current_code:
                    for alt_code in self.zoning_codes:
                        if alt_code.category != current_code.category:
                            compatibility = self.calculate_compatibility(
                                conflict["district1_code"], alt_code.code
                            )
                            if compatibility >= target_compatibility:
                                suggestion["suggestions"].append(
                                    {
                                        "alternative_code": alt_code.code,
                                        "alternative_category": alt_code.category,
                                        "improved_compatibility": compatibility,
                                        "compatibility_improvement": compatibility
                                        - conflict["compatibility_score"],
                                    }
                                )

                if suggestion["suggestions"]:
                    optimization_suggestions.append(suggestion)

        return {
            "total_conflicts": len(conflicts),
            "optimization_opportunities": len(optimization_suggestions),
            "suggestions": optimization_suggestions,
            "current_average_compatibility": self._calculate_average_compatibility(),
            "target_compatibility": target_compatibility,
        }

    def _calculate_average_compatibility(self) -> float:
        """Calculate the average compatibility score across all adjacent districts."""
        districts_gdf = self.export_districts_to_geodataframe()
        compatibilities = []

        for i, district1 in districts_gdf.iterrows():
            for j, district2 in districts_gdf.iterrows():
                if i < j and district1.geometry.touches(district2.geometry):
                    compatibility = self.calculate_compatibility(
                        district1.zoning_code, district2.zoning_code
                    )
                    compatibilities.append(compatibility)

        return float(np.mean(compatibilities)) if compatibilities else 0.0

    def generate_zoning_report(self, output_path: str | None = None) -> str:
        """
        Generate a comprehensive zoning analysis report.

        Args:
            output_path: Optional path to save the report

        Returns:
            Formatted report as a string
        """
        stats = self.get_zoning_statistics()
        conflicts = self.find_zoning_conflicts()
        optimization = self.optimize_zoning_layout()

        report = []
        report.append("=" * 60)
        report.append("GEO-INFER-NORMS ZONING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("1. ZONING OVERVIEW")
        report.append("-" * 20)
        report.append(f"Total Zoning Districts: {stats['total_districts']}")
        report.append(f"Total Zoning Codes: {stats['total_zoning_codes']}")
        report.append(".2f")
        report.append(f"Total Population: {stats['total_population']:,}")
        report.append(f"Total Employment: {stats['total_employment']:,}")
        report.append(".1f")
        report.append(".1f")
        report.append("")

        report.append("2. ZONING CODE DISTRIBUTION")
        report.append("-" * 30)
        for code, count in stats["zoning_code_distribution"].items():
            report.append(f"  {code}: {count} districts")
        report.append("")

        report.append("3. CATEGORY DISTRIBUTION")
        report.append("-" * 25)
        for category, count in stats["category_distribution"].items():
            area = stats["area_by_category_hectares"][category]
            report.append(f"  {category}: {count} districts ({area:.1f} ha)")
        report.append("")

        report.append("4. COMPATIBILITY ANALYSIS")
        report.append("-" * 25)
        report.append(
            f"Average Compatibility: {optimization['current_average_compatibility']:.3f}"
        )
        report.append(f"Number of Conflicts: {len(conflicts)}")
        report.append(
            f"Optimization Opportunities: {optimization['optimization_opportunities']}"
        )
        report.append("")

        if conflicts:
            report.append("5. MAJOR CONFLICTS")
            report.append("-" * 18)
            for i, conflict in enumerate(conflicts[:5]):  # Show top 5 conflicts
                report.append(
                    f"  {i + 1}. {conflict['district1_name']} ({conflict['district1_code']})"
                )
                report.append(
                    f"      ↔ {conflict['district2_name']} ({conflict['district2_code']})"
                )
                report.append(".3f")
                report.append(f"      Severity: {conflict['severity']}")
                report.append("")

        report.append("6. OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 35)
        if optimization["suggestions"]:
            for i, suggestion in enumerate(
                optimization["suggestions"][:3]
            ):  # Show top 3 suggestions
                conflict = suggestion["conflict"]
                report.append(
                    f"  {i + 1}. Change {conflict['district2_name']} from {conflict['district2_code']}"
                )
                for alt in suggestion["suggestions"][:2]:  # Show top 2 alternatives
                    report.append(
                        f"      → {alt['alternative_code']} (compatibility: {alt['improved_compatibility']:.3f})"
                    )
                report.append("")
        else:
            report.append("  No optimization suggestions available.")
            report.append("")

        report.append("=" * 60)
        report.append("Report generated by GEO-INFER-NORMS ZoningAnalyzer")
        report.append("=" * 60)

        final_report = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(final_report)
            report.append(f"\nReport saved to: {output_path}")

        return final_report

    def calculate_development_potential(self, district_id: str) -> dict[str, Any]:
        """
        Calculate development potential for a specific zoning district.

        Args:
            district_id: ID of the district to analyze

        Returns:
            Dictionary with development potential metrics
        """
        district = self.get_district_by_id(district_id)
        if not district:
            return {"error": f"District with ID {district_id} not found"}

        zoning_code = self.get_code_by_id(district.zoning_code)
        if not zoning_code:
            return {"error": f"Zoning code {district.zoning_code} not found"}

        potential: dict[str, Any] = {
            "district_id": district_id,
            "district_name": district.name,
            "zoning_code": district.zoning_code,
            "category": zoning_code.category,
            "area_hectares": district.area_hectares,
            "development_capacity": {},
        }

        # Calculate residential capacity
        if zoning_code.max_density:
            potential["development_capacity"]["residential_units"] = (
                district.area_hectares * zoning_code.max_density
            )

        # Calculate commercial capacity (simplified)
        if zoning_code.max_floor_area_ratio:
            potential["development_capacity"]["floor_area_ratio"] = (
                zoning_code.max_floor_area_ratio
            )
            potential["development_capacity"]["max_floor_area"] = (
                district.area * zoning_code.max_floor_area_ratio
            )

        # Calculate height limitations
        if zoning_code.max_height:
            potential["development_capacity"]["max_height_meters"] = (
                zoning_code.max_height
            )
            potential["development_capacity"]["max_height_feet"] = (
                zoning_code.max_height * 3.28084
            )

        # Calculate setback requirements
        if zoning_code.setbacks:
            potential["development_capacity"]["setbacks"] = zoning_code.setbacks

        # Environmental considerations
        if zoning_code.environmental_requirements:
            potential["environmental_requirements"] = (
                zoning_code.environmental_requirements
            )

        return potential

    def compare_zoning_scenarios(
        self, scenarios: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Compare different zoning scenarios.

        Args:
            scenarios: List of scenario dictionaries with district changes

        Returns:
            Comparison results including compatibility improvements
        """
        results: dict[str, Any] = {
            "scenarios": [],
            "baseline_compatibility": self._calculate_average_compatibility(),
            "comparison": {},
        }

        for i, scenario in enumerate(scenarios):
            scenario_result = {
                "scenario_id": scenario.get("id", f"scenario_{i + 1}"),
                "description": scenario.get("description", ""),
                "changes": [],
            }

            # Create temporary zoning analyzer with scenario changes
            temp_districts = self.zoning_districts.copy()

            for change in scenario.get("changes", []):
                district_id = change["district_id"]
                new_code = change["new_code"]

                # Find and update the district
                for district in temp_districts:
                    if district.id == district_id:
                        district.zoning_code = new_code
                        break

            # Create temporary analyzer
            temp_analyzer = ZoningAnalyzer(
                zoning_districts=temp_districts, zoning_codes=self.zoning_codes
            )

            scenario_result["average_compatibility"] = (
                temp_analyzer._calculate_average_compatibility()
            )
            scenario_result["compatibility_improvement"] = (
                scenario_result["average_compatibility"]
                - results["baseline_compatibility"]
            )
            scenario_result["conflicts"] = len(temp_analyzer.find_zoning_conflicts())

            results["scenarios"].append(scenario_result)

        # Sort scenarios by compatibility improvement
        results["scenarios"].sort(
            key=lambda x: x["compatibility_improvement"], reverse=True
        )

        return results

    def __repr__(self) -> str:
        return f"ZoningAnalyzer(districts={len(self.zoning_districts)}, codes={len(self.zoning_codes)})"


class LandUseClassifier:
    """
    A class for classifying and analyzing land use patterns.

    This class provides methods for classifying land use types, analyzing
    land use patterns, and evaluating land use compatibility.
    """

    def __init__(
        self,
        land_use_types: list[LandUseType] | None = None,
        compatibility_thresholds: dict[str, dict[str, float]] | None = None,
    ):
        """
        Initialize a LandUseClassifier instance.

        Args:
            land_use_types: List of LandUseType objects
            compatibility_thresholds: Optional dictionary defining compatibility thresholds
        """
        self.land_use_types = land_use_types or []
        self._type_index = {t.id: t for t in self.land_use_types}

        # Set default compatibility thresholds if not provided
        self.compatibility_thresholds = compatibility_thresholds or {
            "residential": {
                "residential": 0.9,
                "commercial": 0.6,
                "industrial": 0.2,
                "agricultural": 0.7,
                "recreational": 0.8,
                "institutional": 0.7,
                "mixed_use": 0.8,
            },
            "commercial": {
                "residential": 0.6,
                "commercial": 0.9,
                "industrial": 0.5,
                "agricultural": 0.4,
                "recreational": 0.7,
                "institutional": 0.8,
                "mixed_use": 0.8,
            },
            "industrial": {
                "residential": 0.2,
                "commercial": 0.5,
                "industrial": 0.9,
                "agricultural": 0.4,
                "recreational": 0.3,
                "institutional": 0.4,
                "mixed_use": 0.5,
            },
            "agricultural": {
                "residential": 0.7,
                "commercial": 0.4,
                "industrial": 0.4,
                "agricultural": 0.9,
                "recreational": 0.8,
                "institutional": 0.6,
                "mixed_use": 0.6,
            },
            "recreational": {
                "residential": 0.8,
                "commercial": 0.7,
                "industrial": 0.3,
                "agricultural": 0.8,
                "recreational": 0.9,
                "institutional": 0.8,
                "mixed_use": 0.8,
            },
            "institutional": {
                "residential": 0.7,
                "commercial": 0.8,
                "industrial": 0.4,
                "agricultural": 0.6,
                "recreational": 0.8,
                "institutional": 0.9,
                "mixed_use": 0.8,
            },
            "mixed_use": {
                "residential": 0.8,
                "commercial": 0.8,
                "industrial": 0.5,
                "agricultural": 0.6,
                "recreational": 0.8,
                "institutional": 0.8,
                "mixed_use": 0.9,
            },
        }

    def add_land_use_type(self, land_use_type: LandUseType) -> None:
        """
        Add a land use type to the classifier.

        Args:
            land_use_type: The LandUseType object to add
        """
        self.land_use_types.append(land_use_type)
        self._type_index[land_use_type.id] = land_use_type

    def get_land_use_type_by_id(self, type_id: str) -> LandUseType | None:
        """
        Get a land use type by its ID.

        Args:
            type_id: The ID of the land use type

        Returns:
            The LandUseType object or None if not found
        """
        return self._type_index.get(type_id)

    def calculate_compatibility(
        self, type1_category: str, type2_category: str
    ) -> float:
        """
        Calculate the compatibility score between two land use categories.

        Args:
            type1_category: The first land use category
            type2_category: The second land use category

        Returns:
            A compatibility score between 0 (incompatible) and 1 (fully compatible)
        """
        if (
            type1_category not in self.compatibility_thresholds
            or type2_category
            not in self.compatibility_thresholds.get(type1_category, {})
        ):
            logger.warning(
                f"Land use category {type1_category} or {type2_category} not found in compatibility matrix"
            )
            return 0.5  # Default medium compatibility

        return self.compatibility_thresholds[type1_category][type2_category]

    def analyze_land_use_pattern(
        self, land_use_gdf: gpd.GeoDataFrame, category_column: str = "category"
    ) -> dict[str, Any]:
        """
        Analyze the pattern of land use in a given area.

        Args:
            land_use_gdf: GeoDataFrame containing land use polygons
            category_column: Column name for land use categories

        Returns:
            A dictionary containing analysis results
        """
        results: dict[str, Any] = {}
        if land_use_gdf.empty:
            results["status"] = "error"
            results["message"] = "Empty GeoDataFrame provided"
            logger.error(results["message"])
            return results

        # Ensure the GeoDataFrame has the required column
        if category_column not in land_use_gdf.columns:
            results["status"] = "error"
            results["message"] = (
                f"Category column '{category_column}' not found in GeoDataFrame"
            )
            logger.error(results["message"])
            return results

        # Ensure valid geometries
        land_use_gdf = land_use_gdf[land_use_gdf.geometry.is_valid]
        if land_use_gdf.empty:
            results["status"] = "error"
            results["message"] = "No valid geometries found in GeoDataFrame"
            logger.error(results["message"])
            return results

        # Reproject for accurate area calculation (using EPSG:3857 as a common web mercator)
        try:
            gdf_proj = land_use_gdf.to_crs(epsg=3857)
        except Exception as e:
            logger.error(f"Failed to reproject GeoDataFrame for area calculation: {e}")
            results["status"] = "error"
            results["message"] = f"Failed to reproject GeoDataFrame: {e}"
            return results

        gdf_proj["area"] = gdf_proj.geometry.area
        # land_use_gdf['area'] = land_use_gdf.geometry.area # Original line

        total_area = gdf_proj["area"].sum()
        results["total_area"] = total_area

        # Calculate area and percentage by category
        area_by_category = gdf_proj.groupby(category_column)["area"].sum()
        percentage_by_category = (area_by_category / total_area) * 100
        results["area_by_category"] = area_by_category.to_dict()
        results["percentage_by_category"] = percentage_by_category.to_dict()
        results["category_count"] = land_use_gdf[category_column].nunique()

        # Analyze adjacency and compatibility
        adjacency_matrix: dict[str, dict[str, int]] = {}
        compatibility_scores = []

        # Create a spatial index for more efficient computation
        spatial_index = land_use_gdf.sindex

        for idx, parcel in land_use_gdf.iterrows():
            parcel_category = parcel[category_column]

            if parcel_category not in adjacency_matrix:
                adjacency_matrix[parcel_category] = {}

            # Find adjacent parcels using the spatial index
            possible_matches_index = list(
                spatial_index.intersection(parcel.geometry.bounds)
            )
            possible_matches = land_use_gdf.iloc[possible_matches_index]

            for adj_idx, adj_parcel in possible_matches.iterrows():
                if idx == adj_idx:
                    continue

                adj_category = adj_parcel[category_column]

                if parcel.geometry.touches(adj_parcel.geometry):
                    if adj_category not in adjacency_matrix[parcel_category]:
                        adjacency_matrix[parcel_category][adj_category] = 0

                    adjacency_matrix[parcel_category][adj_category] += 1

                    # Calculate compatibility
                    compatibility = self.calculate_compatibility(
                        parcel_category, adj_category
                    )
                    compatibility_scores.append(compatibility)

        # Build a network representation
        land_use_network = nx.Graph()

        for cat1 in adjacency_matrix:
            for cat2 in adjacency_matrix[cat1]:
                weight = adjacency_matrix[cat1][cat2]
                land_use_network.add_edge(cat1, cat2, weight=weight)

        # Calculate network metrics
        network_metrics = {}
        if land_use_network.nodes():
            try:
                network_metrics = {
                    "density": nx.density(land_use_network),
                    "avg_clustering": nx.average_clustering(land_use_network),
                    "centrality": {
                        node: score
                        for node, score in nx.degree_centrality(
                            land_use_network
                        ).items()
                    },
                }
            except Exception as e:
                network_metrics = {"error": str(e)}

        # Overall compatibility score
        avg_compatibility = np.mean(compatibility_scores) if compatibility_scores else 0

        results["status"] = "success"
        results["adjacency_matrix"] = adjacency_matrix
        results["average_compatibility"] = avg_compatibility
        results["network_metrics"] = network_metrics

        return results

    def classify_land_use(
        self, features_gdf: gpd.GeoDataFrame, feature_columns: list[str]
    ) -> gpd.GeoDataFrame:
        """
        Classify land use based on feature characteristics.

        Args:
            features_gdf: GeoDataFrame containing parcels and their features
            feature_columns: List of column names to use as features

        Returns:
            A copy of the input GeoDataFrame with added land use classification

        Raises:
            ValueError: If features_gdf is None or feature_columns is empty
            TypeError: If features_gdf is not a GeoDataFrame
        """
        # Input validation
        if features_gdf is None:
            raise ValueError("features_gdf cannot be None")

        if not isinstance(features_gdf, gpd.GeoDataFrame):
            raise TypeError("features_gdf must be a GeoDataFrame")

        if not feature_columns:
            raise ValueError("feature_columns cannot be empty")

        if not isinstance(feature_columns, list):
            raise TypeError("feature_columns must be a list")

        if features_gdf.empty:
            logger.warning("Empty GeoDataFrame provided for classification")
            return features_gdf.copy()

        # Check if all required columns exist
        missing_columns = [
            col for col in feature_columns if col not in features_gdf.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")

        # Check for non-numeric columns that should be numeric
        for col in feature_columns:
            if not pd.api.types.is_numeric_dtype(features_gdf[col]):
                logger.warning(
                    f"Column '{col}' is not numeric - classification may not work as expected"
                )

        # Simple rule-based classification
        result_gdf = features_gdf.copy()
        result_gdf["land_use_category"] = "unclassified"
        result_gdf["land_use_confidence"] = 0.0

        # Multi-feature weighted scoring classifier.
        # Each parcel receives a score for every land use category based on
        # the available numeric features.  The category with the highest score wins.
        result_gdf = features_gdf.copy()
        result_gdf["land_use_category"] = "unclassified"
        result_gdf["land_use_confidence"] = 0.0

        # Feature weight tables — weight matrix[category][feature] = weight (positive = supports, negative = inhibits)
        RULES: dict = {
            "residential": {
                "building_count": +0.5,
                "population_density": +0.8,
                "business_count": -0.3,
                "farmland_percentage": -0.7,
                "park_area": -0.1,
                "impervious_surface": +0.2,
            },
            "commercial": {
                "building_count": +0.3,
                "business_count": +0.9,
                "population_density": +0.1,
                "farmland_percentage": -0.8,
                "park_area": -0.2,
                "impervious_surface": +0.4,
            },
            "industrial": {
                "industrial_area": +0.9,
                "impervious_surface": +0.6,
                "business_count": +0.2,
                "population_density": -0.5,
                "farmland_percentage": -0.8,
                "park_area": -0.6,
            },
            "agricultural": {
                "farmland_percentage": +0.95,
                "population_density": -0.6,
                "impervious_surface": -0.7,
                "business_count": -0.5,
                "building_count": -0.3,
            },
            "recreational": {
                "park_area": +0.9,
                "population_density": +0.2,
                "impervious_surface": -0.4,
                "industrial_area": -0.8,
                "farmland_percentage": -0.1,
            },
            "institutional": {
                "building_count": +0.4,
                "population_density": +0.3,
                "business_count": +0.2,
                "park_area": +0.1,
                "farmland_percentage": -0.5,
            },
            "mixed_use": {
                "building_count": +0.4,
                "business_count": +0.4,
                "population_density": +0.4,
                "impervious_surface": +0.2,
                "farmland_percentage": -0.4,
            },
            "open_space": {
                "park_area": +0.5,
                "farmland_percentage": +0.3,
                "impervious_surface": -0.8,
                "building_count": -0.6,
                "population_density": -0.6,
            },
        }

        # Normalisation bounds per feature for rough [0,1] scaling
        NORM: dict = {
            "building_count": 1000.0,
            "population_density": 10000.0,
            "business_count": 200.0,
            "farmland_percentage": 100.0,
            "park_area": 100000.0,
            "impervious_surface": 100.0,
            "industrial_area": 500000.0,
        }

        for idx, parcel in result_gdf.iterrows():
            scores: dict = {cat: 0.0 for cat in RULES}
            for cat, weights in RULES.items():
                for feature, weight in weights.items():
                    if feature in feature_columns and feature in parcel:
                        norm = NORM.get(feature, 1.0)
                        val = max(0.0, float(parcel[feature])) / norm
                        val = min(1.0, val)  # clamp to [0, 1]
                        scores[cat] += weight * val

            best_cat = max(scores, key=lambda c: scores[c])
            best_score = scores[best_cat]
            # Map raw score to [0, 1] confidence: (score + n_features) / (2 * n_features)
            n_feat = max(1, len(feature_columns))
            confidence = max(0.0, min(1.0, (best_score + n_feat) / (2.0 * n_feat)))

            result_gdf.at[idx, "land_use_category"] = best_cat
            result_gdf.at[idx, "land_use_confidence"] = round(confidence, 3)

        return result_gdf

    def visualize_land_use(
        self,
        land_use_gdf: gpd.GeoDataFrame,
        category_column: str = "land_use_category",
        figsize: tuple[int, int] = (12, 8),
        cmap: str | ListedColormap | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create a visualization of land use patterns.

        Args:
            land_use_gdf: GeoDataFrame containing land use polygons
            category_column: Column name for land use categories
            figsize: Figure size as (width, height) in inches
            cmap: Optional colormap for visualization
            save_path: Optional path to save the figure

        Returns:
            A matplotlib Figure object
        """
        if land_use_gdf.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No land use data available for visualization",
                ha="center",
                va="center",
            )
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            return fig

        if category_column not in land_use_gdf.columns:
            logger.warning(
                f"Category column '{category_column}' not found in GeoDataFrame"
            )
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                f"Column '{category_column}' not found in data",
                ha="center",
                va="center",
            )
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            return fig

        fig, ax = plt.subplots(figsize=figsize)

        # Create a categorical plot
        land_use_gdf.plot(
            column=category_column,
            ax=ax,
            legend=True,
            cmap=cmap or "tab10",
            categorical=True,
            legend_kwds={"title": "Land Use Categories", "loc": "upper right"},
        )

        ax.set_title("Land Use Classification")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig
