"""
Core functionalities for analyzing the spatial and social impacts of policies and regulations.

This module provides tools to assess how policy changes might affect various aspects
within a geospatial context, including economic factors, social equity, environmental
conditions, and compliance landscapes. It aims to support decision-making by
quantifying potential outcomes of proposed or implemented policies.
"""

import logging

import pandas as pd
import geopandas as gpd
from shapely.geometry import base

from typing import Any


logger = logging.getLogger(__name__)


class PolicyImpactAnalyzer:
    """Analyzes the potential or actual impacts of a policy across various dimensions.

    This class takes policy definitions, relevant geospatial data, and potentially
    socio-economic datasets to model and quantify the effects of policy interventions.
    It can be used for ex-ante (predictive) or ex-post (evaluative) analysis.

    Attributes:
        policy (Any): The policy object or definition to be analyzed.
        context_data (Dict[str, Any]): Supporting data (e.g., population density,
                                         economic indicators, environmental layers).
        spatial_extent (Optional[base.BaseGeometry]): The geographic area of interest.
    """

    def __init__(
        self,
        policy: Any,
        context_data: dict[str, Any],
        spatial_extent: base.BaseGeometry | None = None,
    ):
        """Initializes the PolicyImpactAnalyzer.

        Args:
            policy: The policy object or mapping to analyze.
            context_data: Dictionary containing datasets relevant for the analysis
                          (e.g., GeoDataFrames, DataFrames).
            spatial_extent: Optional geometry defining the analysis boundary.
        """
        self.policy = policy
        self.context_data = context_data
        self.spatial_extent = spatial_extent
        logger.info("Initialized PolicyImpactAnalyzer for policy: %s", policy)

    def analyze_economic_impact(self) -> pd.DataFrame:
        """Analyzes the economic consequences of the policy.

        Examples: Changes in property values, business activity, employment.

        Returns:
            DataFrame summarizing economic impacts.
        """
        logger.info("Analyzing economic impact")

        # Check if we have the necessary data
        if not self.context_data.get("economic_data"):
            # Create a minimal result if no economic data is available
            return pd.DataFrame(
                {
                    "impact_category": [
                        "property_value",
                        "business_activity",
                        "employment",
                        "tax_revenue",
                    ],
                    "impact_type": ["Not analyzed"] * 4,
                    "impact_value": [0.0] * 4,
                    "confidence": [0.0] * 4,
                    "notes": ["No economic data provided in context_data"] * 4,
                }
            )

        # Extract relevant economic data
        economic_data = self.context_data["economic_data"]

        # Extract policy attributes that may affect economic analysis
        policy_type = getattr(self.policy, "policy_type", "unknown")
        # Initialize results DataFrame
        results = []

        # Analyze property value impacts
        if "property_values" in economic_data:
            property_data = economic_data["property_values"]

            # Simple algorithm to estimate property value changes based on policy type
            # In a real implementation, this would use more sophisticated models
            if policy_type in ["zoning_change", "land_use", "development"]:
                # Zoning/land use policies often have significant property value impacts
                avg_change = 0.05  # Assume 5% baseline change

                # Adjust based on policy details if available
                if policy_type == "zoning_change" and hasattr(
                    self.policy, "zoning_details"
                ):
                    if self.policy.zoning_details.get("upzoning", False):
                        avg_change = (
                            0.12  # Upzoning typically increases property values
                        )
                    elif self.policy.zoning_details.get("downzoning", False):
                        avg_change = -0.08  # Downzoning may decrease property values

                # Calculate baseline total impact
                total_property_value = property_data.get("total_value", 0)
                impact_value = total_property_value * avg_change

                # Determine impact type
                impact_type = (
                    "positive"
                    if impact_value > 0
                    else "negative"
                    if impact_value < 0
                    else "neutral"
                )

                results.append(
                    {
                        "impact_category": "property_value",
                        "impact_type": impact_type,
                        "impact_value": impact_value,
                        "change_percentage": avg_change * 100,
                        "confidence": 0.7,
                        "notes": f"Estimated impact based on policy type: {policy_type}",
                    }
                )
            else:
                # Other policy types have a more modest/unknown effect
                results.append(
                    {
                        "impact_category": "property_value",
                        "impact_type": "uncertain",
                        "impact_value": 0.0,
                        "change_percentage": 0.0,
                        "confidence": 0.4,
                        "notes": f"Policy type {policy_type} has uncertain property value impacts",
                    }
                )
        else:
            # No property value data available
            results.append(
                {
                    "impact_category": "property_value",
                    "impact_type": "not_analyzed",
                    "impact_value": 0.0,
                    "change_percentage": 0.0,
                    "confidence": 0.0,
                    "notes": "No property value data available",
                }
            )

        # Analyze business activity impacts
        if "business_activity" in economic_data:
            business_data = economic_data["business_activity"]

            # Simple algorithm for business impact
            if policy_type in ["economic_development", "tax", "infrastructure"]:
                # These policies often directly affect business activity
                if policy_type == "economic_development":
                    impact_type = "positive"
                    impact_value = (
                        business_data.get("revenue", 0) * 0.03
                    )  # Assume 3% increase
                    confidence = 0.75
                elif policy_type == "tax":
                    # Tax policies could be positive or negative depending on details
                    tax_increase = getattr(self.policy, "tax_increase", False)
                    impact_type = "negative" if tax_increase else "positive"
                    impact_value = business_data.get("revenue", 0) * (
                        -0.02 if tax_increase else 0.01
                    )
                    confidence = 0.8
                else:  # infrastructure
                    impact_type = "positive"
                    impact_value = (
                        business_data.get("revenue", 0) * 0.02
                    )  # Assume 2% increase
                    confidence = 0.65

                results.append(
                    {
                        "impact_category": "business_activity",
                        "impact_type": impact_type,
                        "impact_value": impact_value,
                        "change_percentage": (
                            impact_value / business_data.get("revenue", 1)
                        )
                        * 100
                        if business_data.get("revenue", 0) > 0
                        else 0,
                        "confidence": confidence,
                        "notes": f"Estimated impact based on policy type: {policy_type}",
                    }
                )
            else:
                # Other policy types
                results.append(
                    {
                        "impact_category": "business_activity",
                        "impact_type": "minor",
                        "impact_value": business_data.get("revenue", 0)
                        * 0.005,  # Assume 0.5% change
                        "change_percentage": 0.5,
                        "confidence": 0.5,
                        "notes": f"Policy type {policy_type} has minor business activity impacts",
                    }
                )
        else:
            results.append(
                {
                    "impact_category": "business_activity",
                    "impact_type": "not_analyzed",
                    "impact_value": 0.0,
                    "change_percentage": 0.0,
                    "confidence": 0.0,
                    "notes": "No business activity data available",
                }
            )

        # Analyze employment impacts
        if "employment" in economic_data:
            employment_data = economic_data["employment"]

            # Simple algorithm for employment impact
            if policy_type in [
                "economic_development",
                "labor",
                "infrastructure",
                "zoning_change",
            ]:
                # Calculate baseline job impact
                total_jobs = employment_data.get("total_jobs", 0)

                if policy_type == "economic_development":
                    job_change_pct = 0.02  # 2% job growth
                    impact_type = "positive"
                elif policy_type == "labor":
                    # Could be positive or negative
                    is_restrictive = getattr(self.policy, "is_restrictive", False)
                    job_change_pct = -0.01 if is_restrictive else 0.015
                    impact_type = "negative" if is_restrictive else "positive"
                elif policy_type == "infrastructure":
                    job_change_pct = 0.01  # 1% job growth
                    impact_type = "positive"
                else:  # zoning_change
                    # Depends on the direction
                    if hasattr(
                        self.policy, "zoning_details"
                    ) and self.policy.zoning_details.get("upzoning", False):
                        job_change_pct = 0.03  # More development potential
                        impact_type = "positive"
                    else:
                        job_change_pct = 0.005  # Minimal impact
                        impact_type = "minor"

                job_impact = total_jobs * job_change_pct

                results.append(
                    {
                        "impact_category": "employment",
                        "impact_type": impact_type,
                        "impact_value": job_impact,
                        "change_percentage": job_change_pct * 100,
                        "confidence": 0.6,
                        "notes": f"Estimated job impact: {job_impact:.0f} jobs ({job_change_pct * 100:.1f}%)",
                    }
                )
            else:
                # Minimal job impact for other policy types
                results.append(
                    {
                        "impact_category": "employment",
                        "impact_type": "minor",
                        "impact_value": employment_data.get("total_jobs", 0) * 0.002,
                        "change_percentage": 0.2,
                        "confidence": 0.4,
                        "notes": f"Policy type {policy_type} has minimal employment impacts",
                    }
                )
        else:
            results.append(
                {
                    "impact_category": "employment",
                    "impact_type": "not_analyzed",
                    "impact_value": 0.0,
                    "change_percentage": 0.0,
                    "confidence": 0.0,
                    "notes": "No employment data available",
                }
            )

        # Analyze tax revenue impacts
        if "tax_revenue" in economic_data:
            tax_data = economic_data["tax_revenue"]

            # Simple algorithm for tax revenue impact
            if policy_type in ["tax", "zoning_change", "economic_development"]:
                current_revenue = tax_data.get("annual_revenue", 0)

                if policy_type == "tax":
                    # Direct impact on tax revenue
                    tax_increase = getattr(self.policy, "tax_increase", False)
                    tax_rate_change = getattr(self.policy, "tax_rate_change", 0.0)

                    if tax_rate_change != 0:
                        # Use the specific rate change
                        impact_value = current_revenue * tax_rate_change
                    else:
                        # Use a default estimate
                        impact_value = current_revenue * (
                            0.05 if tax_increase else -0.03
                        )

                    impact_type = "positive" if impact_value > 0 else "negative"
                    confidence = 0.85  # High confidence for direct tax policies
                else:
                    # Indirect impact through property values or economic activity
                    # Find our property value impact
                    prop_val_result = next(
                        (
                            r
                            for r in results
                            if r["impact_category"] == "property_value"
                        ),
                        None,
                    )

                    if prop_val_result and prop_val_result["impact_type"] not in [
                        "not_analyzed",
                        "uncertain",
                    ]:
                        # Estimate tax impact based on property value change
                        property_tax_rate = tax_data.get(
                            "property_tax_rate", 0.01
                        )  # Default 1%
                        impact_value = (
                            prop_val_result["impact_value"] * property_tax_rate
                        )
                        impact_type = (
                            "positive"
                            if impact_value > 0
                            else "negative"
                            if impact_value < 0
                            else "neutral"
                        )
                        confidence = 0.7
                    else:
                        # Default minimal impact
                        impact_value = current_revenue * 0.01
                        impact_type = "minor"
                        confidence = 0.5

                results.append(
                    {
                        "impact_category": "tax_revenue",
                        "impact_type": impact_type,
                        "impact_value": impact_value,
                        "change_percentage": (impact_value / current_revenue * 100)
                        if current_revenue > 0
                        else 0,
                        "confidence": confidence,
                        "notes": f"Estimated tax revenue impact based on policy type: {policy_type}",
                    }
                )
            else:
                # Minimal tax impact for other policy types
                results.append(
                    {
                        "impact_category": "tax_revenue",
                        "impact_type": "minor",
                        "impact_value": tax_data.get("annual_revenue", 0) * 0.005,
                        "change_percentage": 0.5,
                        "confidence": 0.3,
                        "notes": f"Policy type {policy_type} has minimal tax revenue impacts",
                    }
                )
        else:
            results.append(
                {
                    "impact_category": "tax_revenue",
                    "impact_type": "not_analyzed",
                    "impact_value": 0.0,
                    "change_percentage": 0.0,
                    "confidence": 0.0,
                    "notes": "No tax revenue data available",
                }
            )

        # Apply spatial context if available
        if self.spatial_extent and "spatial_economic_data" in self.context_data:
            # This would involve more complex spatial analysis in a full implementation
            # For now, we'll just note that spatial context was considered
            for result in results:
                result["notes"] += "; Spatial context considered in analysis"

        # Create and return the DataFrame
        results_df = pd.DataFrame(results)

        # Calculate a summary score (weighted average of impacts)
        if (
            not results_df.empty
            and "confidence" in results_df.columns
            and "impact_value" in results_df.columns
        ):
            # Normalize the impact values by category for fair comparison
            for category in results_df["impact_category"].unique():
                category_rows = results_df["impact_category"] == category
                max_abs_value = (
                    results_df.loc[category_rows, "impact_value"].abs().max()
                )
                if max_abs_value > 0:
                    results_df.loc[category_rows, "normalized_impact"] = (
                        results_df.loc[category_rows, "impact_value"] / max_abs_value
                    )
                else:
                    results_df.loc[category_rows, "normalized_impact"] = 0

            # Calculate weighted score if we have normalized impacts
            if "normalized_impact" in results_df.columns:
                weighted_impacts = (
                    results_df["normalized_impact"] * results_df["confidence"]
                )
                total_confidence = results_df["confidence"].sum()
                overall_score = (
                    weighted_impacts.sum() / total_confidence
                    if total_confidence > 0
                    else 0
                )

                # Add overall assessment
                if overall_score > 0.3:
                    overall_assessment = "Positive economic impact"
                elif overall_score < -0.3:
                    overall_assessment = "Negative economic impact"
                else:
                    overall_assessment = "Neutral or minimal economic impact"

                # Add these as attributes to the DataFrame for easy access
                results_df.attrs["overall_score"] = overall_score
                results_df.attrs["overall_assessment"] = overall_assessment

        return results_df

    def analyze_social_equity_impact(self) -> pd.DataFrame:
        """Assesses the policy's impact on social equity and justice.

        Examples: Distributional effects across demographic groups, access to
                  resources, displacement risk.

        Returns:
            DataFrame summarizing social equity impacts.
        """
        results = []
        demographic_data = self.context_data.get("demographic_data", {})

        if not demographic_data:
            return pd.DataFrame(
                {
                    "equity_dimension": ["access", "displacement", "distribution"],
                    "impact_type": ["not_analyzed"] * 3,
                    "impact_score": [0.0] * 3,
                    "confidence": [0.0] * 3,
                    "notes": ["No demographic data provided"] * 3,
                }
            )

        policy_type = getattr(self.policy, "policy_type", "unknown")

        # Analyze access equity
        access_data = demographic_data.get("access_metrics", {})
        if access_data:
            groups = access_data.get("groups", [])
            if groups:
                access_scores = [g.get("access_score", 0.5) for g in groups]
                disparity = (
                    max(access_scores) - min(access_scores) if access_scores else 0.0
                )
                impact_score = (
                    -disparity
                    if policy_type in ("zoning_change", "land_use")
                    else disparity * 0.5
                )
            else:
                impact_score = 0.0
                disparity = 0.0
            results.append(
                {
                    "equity_dimension": "access",
                    "impact_type": "negative"
                    if impact_score < -0.1
                    else "positive"
                    if impact_score > 0.1
                    else "neutral",
                    "impact_score": impact_score,
                    "confidence": 0.6,
                    "notes": f"Access disparity: {disparity:.2f}",
                }
            )
        else:
            results.append(
                {
                    "equity_dimension": "access",
                    "impact_type": "not_analyzed",
                    "impact_score": 0.0,
                    "confidence": 0.0,
                    "notes": "No access metrics available",
                }
            )

        # Analyze displacement risk
        housing_data = demographic_data.get("housing", {})
        if housing_data:
            avg_rent_burden = housing_data.get("avg_rent_burden", 0.3)
            displacement_risk = (
                min(1.0, avg_rent_burden * 1.5)
                if policy_type in ("zoning_change", "development")
                else avg_rent_burden
            )
            results.append(
                {
                    "equity_dimension": "displacement",
                    "impact_type": "negative" if displacement_risk > 0.5 else "neutral",
                    "impact_score": -displacement_risk,
                    "confidence": 0.55,
                    "notes": f"Displacement risk score: {displacement_risk:.2f}",
                }
            )
        else:
            results.append(
                {
                    "equity_dimension": "displacement",
                    "impact_type": "not_analyzed",
                    "impact_score": 0.0,
                    "confidence": 0.0,
                    "notes": "No housing data available",
                }
            )

        # Analyze distributional effects
        income_data = demographic_data.get("income_distribution", {})
        if income_data:
            gini = income_data.get("gini_coefficient", 0.4)
            policy_gini_impact = (
                -0.02 if policy_type == "economic_development" else 0.01
            )
            new_gini = min(1.0, max(0.0, gini + policy_gini_impact))
            results.append(
                {
                    "equity_dimension": "distribution",
                    "impact_type": "positive" if new_gini < gini else "negative",
                    "impact_score": gini - new_gini,
                    "confidence": 0.5,
                    "notes": f"Gini change: {gini:.3f} -> {new_gini:.3f}",
                }
            )
        else:
            results.append(
                {
                    "equity_dimension": "distribution",
                    "impact_type": "not_analyzed",
                    "impact_score": 0.0,
                    "confidence": 0.0,
                    "notes": "No income distribution data available",
                }
            )

        return pd.DataFrame(results)

    def analyze_environmental_impact(self) -> gpd.GeoDataFrame:
        """Evaluates the environmental consequences of the policy.

        Examples: Changes in land cover, air/water quality, habitat fragmentation.

        Returns:
            GeoDataFrame visualizing or summarizing environmental impacts.
        """
        env_data = self.context_data.get("environmental_data", {})
        policy_type = getattr(self.policy, "policy_type", "unknown")

        if not env_data:
            return gpd.GeoDataFrame(
                {
                    "impact_category": ["land_cover", "air_quality", "water_quality"],
                    "impact_score": [0.0, 0.0, 0.0],
                    "confidence": [0.0, 0.0, 0.0],
                    "notes": ["No environmental data provided"] * 3,
                }
            )

        results = []

        # Land cover impact
        land_data = env_data.get("land_cover", {})
        if land_data:
            change_pct = land_data.get("projected_change_pct", 0.0)
            if policy_type in ("zoning_change", "development", "land_use"):
                change_pct = change_pct if change_pct != 0 else 5.0
            impact_score = -change_pct / 100.0
            results.append(
                {
                    "impact_category": "land_cover",
                    "impact_score": impact_score,
                    "confidence": 0.65,
                    "notes": f"Projected land cover change: {change_pct:.1f}%",
                }
            )
        else:
            results.append(
                {
                    "impact_category": "land_cover",
                    "impact_score": 0.0,
                    "confidence": 0.0,
                    "notes": "No land cover data",
                }
            )

        # Air quality impact
        air_data = env_data.get("air_quality", {})
        if air_data:
            baseline_aqi = air_data.get("baseline_aqi", 50)
            if policy_type == "environmental":
                projected_change = -5.0
            elif policy_type in ("development", "infrastructure"):
                projected_change = 3.0
            else:
                projected_change = 0.0
            results.append(
                {
                    "impact_category": "air_quality",
                    "impact_score": -projected_change / 100.0,
                    "confidence": 0.55,
                    "notes": f"AQI change: {projected_change:+.1f} from baseline {baseline_aqi}",
                }
            )
        else:
            results.append(
                {
                    "impact_category": "air_quality",
                    "impact_score": 0.0,
                    "confidence": 0.0,
                    "notes": "No air quality data",
                }
            )

        # Water quality impact
        water_data = env_data.get("water_quality", {})
        if water_data:
            baseline_wqi = water_data.get("baseline_wqi", 70)
            if policy_type == "environmental":
                projected_change = 5.0
            elif policy_type in ("development", "industrial"):
                projected_change = -3.0
            else:
                projected_change = 0.0
            results.append(
                {
                    "impact_category": "water_quality",
                    "impact_score": projected_change / 100.0,
                    "confidence": 0.5,
                    "notes": f"WQI change: {projected_change:+.1f} from baseline {baseline_wqi}",
                }
            )
        else:
            results.append(
                {
                    "impact_category": "water_quality",
                    "impact_score": 0.0,
                    "confidence": 0.0,
                    "notes": "No water quality data",
                }
            )

        return gpd.GeoDataFrame(results)

    def generate_impact_report(self) -> dict[str, Any]:
        """Compiles a comprehensive report of all analyzed impacts.

        Returns:
            Dictionary containing summaries of economic, social, and environmental
            impacts.
        """
        economic_df = self.analyze_economic_impact()
        social_df = self.analyze_social_equity_impact()
        environmental_gdf = self.analyze_environmental_impact()

        report = {
            "economic": economic_df,
            "social_equity": social_df,
            "environmental": environmental_gdf,
            "summary": {
                "policy": str(self.policy),
                "spatial_extent": str(self.spatial_extent)
                if self.spatial_extent
                else "None",
                "economic_categories_analyzed": len(economic_df)
                if not economic_df.empty
                else 0,
                "social_dimensions_analyzed": len(social_df)
                if not social_df.empty
                else 0,
                "environmental_categories_analyzed": len(environmental_gdf)
                if not environmental_gdf.empty
                else 0,
            },
        }
        return report

    def visualize_spatial_impact(self) -> Any:
        """Creates a map visualizing the spatial distribution of policy impacts.

        Returns:
            A map object (e.g., matplotlib figure, Folium map). Returns None
            if no spatial extent or environmental data is available.
        """
        if self.spatial_extent is None:
            return None

        environmental_gdf = self.analyze_environmental_impact()
        if environmental_gdf.empty:
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        if "impact_score" in environmental_gdf.columns:
            ax.barh(
                environmental_gdf["impact_category"],
                environmental_gdf["impact_score"],
            )
            ax.set_xlabel("Impact Score")
            ax.set_title(f"Spatial Impact: {self.policy}")
        else:
            ax.text(0.5, 0.5, "No impact data to visualize", ha="center", va="center")

        plt.tight_layout()
        return fig


class RegulatoryImpactAssessment:
    """Performs an assessment of the impacts specifically related to regulations.

    Focuses on compliance costs, administrative burden, market effects, and
    achievement of regulatory goals within a spatial context.

    Attributes:
        regulation (Any): The regulation object or definition being assessed.
        affected_entities (gpd.GeoDataFrame): Geospatial data of entities
                                             (e.g., businesses, properties)
                                             affected by the regulation.
        baseline_data (Dict[str, Any]): Data representing the state before the
                                       regulation or under alternative scenarios.
    """

    def __init__(
        self,
        regulation: Any,
        affected_entities: gpd.GeoDataFrame,
        baseline_data: dict[str, Any],
    ):
        """Initializes the RegulatoryImpactAssessment.

        Args:
            regulation: The regulation object or mapping to assess.
            affected_entities: GeoDataFrame of entities potentially impacted.
            baseline_data: Dictionary of baseline datasets for comparison.
        """
        self.regulation = regulation
        self.affected_entities = affected_entities
        self.baseline_data = baseline_data
        logger.info(
            "Initialized RegulatoryImpactAssessment for regulation: %s", regulation
        )

    def estimate_compliance_costs(self) -> pd.DataFrame:
        """Estimates the costs incurred by affected entities to comply.

        Returns:
            DataFrame detailing estimated compliance costs per entity or category.
        """
        if self.affected_entities.empty:
            return pd.DataFrame(
                columns=["entity_category", "estimated_cost", "confidence", "notes"]
            )

        cost_data = self.baseline_data.get("compliance_costs", {})
        base_cost_per_entity = cost_data.get("base_cost_per_entity", 1000.0)
        cost_multiplier = cost_data.get("complexity_multiplier", 1.0)

        rows = []
        categories = self.affected_entities.get(
            "category", pd.Series(["unknown"] * len(self.affected_entities))
        )
        for category in categories.unique():
            count = int((categories == category).sum())
            estimated_cost = base_cost_per_entity * cost_multiplier * count
            rows.append(
                {
                    "entity_category": category,
                    "entity_count": count,
                    "estimated_cost": estimated_cost,
                    "cost_per_entity": base_cost_per_entity * cost_multiplier,
                    "confidence": 0.6,
                    "notes": f"Based on base cost {base_cost_per_entity} * multiplier {cost_multiplier}",
                }
            )

        return pd.DataFrame(rows)

    def assess_administrative_burden(self) -> dict[str, Any]:
        """Evaluates the administrative effort required by the regulation.

        Considers reporting requirements, permitting processes, monitoring, etc.

        Returns:
            Dictionary summarizing administrative burden components.
        """
        admin_data = self.baseline_data.get("administrative", {})
        entity_count = (
            len(self.affected_entities) if not self.affected_entities.empty else 0
        )

        reporting_hours = admin_data.get("reporting_hours_per_entity", 20)
        permitting_hours = admin_data.get("permitting_hours_per_entity", 10)
        monitoring_hours = admin_data.get("monitoring_hours_per_entity", 5)
        hourly_rate = admin_data.get("admin_hourly_rate", 50.0)

        total_hours = (
            reporting_hours + permitting_hours + monitoring_hours
        ) * entity_count
        total_cost = total_hours * hourly_rate

        return {
            "entity_count": entity_count,
            "reporting": {
                "hours_per_entity": reporting_hours,
                "total_hours": reporting_hours * entity_count,
                "cost": reporting_hours * entity_count * hourly_rate,
            },
            "permitting": {
                "hours_per_entity": permitting_hours,
                "total_hours": permitting_hours * entity_count,
                "cost": permitting_hours * entity_count * hourly_rate,
            },
            "monitoring": {
                "hours_per_entity": monitoring_hours,
                "total_hours": monitoring_hours * entity_count,
                "cost": monitoring_hours * entity_count * hourly_rate,
            },
            "total_hours": total_hours,
            "total_cost": total_cost,
            "hourly_rate": hourly_rate,
        }

    def analyze_market_effects(self) -> pd.DataFrame:
        """Analyzes the regulation's impact on market dynamics.

        Examples: Changes in competition, innovation, prices, market entry/exit.

        Returns:
            DataFrame summarizing market effects.
        """
        market_data = self.baseline_data.get("market", {})

        if not market_data:
            return pd.DataFrame(
                {
                    "market_dimension": [
                        "competition",
                        "innovation",
                        "prices",
                        "entry_exit",
                    ],
                    "impact_type": ["not_analyzed"] * 4,
                    "impact_score": [0.0] * 4,
                    "confidence": [0.0] * 4,
                    "notes": ["No market data available"] * 4,
                }
            )

        results = []
        num_entities = (
            len(self.affected_entities) if not self.affected_entities.empty else 0
        )

        # Competition impact
        num_competitors = market_data.get("num_competitors", 10)
        barrier_increase = market_data.get("barrier_increase_pct", 5.0)
        competition_score = -(barrier_increase / 100.0)
        results.append(
            {
                "market_dimension": "competition",
                "impact_type": "negative" if competition_score < -0.05 else "neutral",
                "impact_score": competition_score,
                "confidence": 0.6,
                "notes": f"Barrier increase: {barrier_increase}%, competitors: {num_competitors}",
            }
        )

        # Innovation impact
        innovation_effect = market_data.get("innovation_effect", 0.0)
        results.append(
            {
                "market_dimension": "innovation",
                "impact_type": "positive"
                if innovation_effect > 0
                else "negative"
                if innovation_effect < 0
                else "neutral",
                "impact_score": innovation_effect,
                "confidence": 0.45,
                "notes": f"Innovation effect score: {innovation_effect}",
            }
        )

        # Price impact
        price_change_pct = market_data.get("price_change_pct", 0.0)
        if num_entities > 0 and price_change_pct == 0.0:
            price_change_pct = barrier_increase * 0.3
        results.append(
            {
                "market_dimension": "prices",
                "impact_type": "negative"
                if price_change_pct > 0
                else "positive"
                if price_change_pct < 0
                else "neutral",
                "impact_score": -price_change_pct / 100.0,
                "confidence": 0.55,
                "notes": f"Estimated price change: {price_change_pct:+.1f}%",
            }
        )

        # Market entry/exit
        exit_rate = market_data.get("projected_exit_rate", 0.0)
        results.append(
            {
                "market_dimension": "entry_exit",
                "impact_type": "negative" if exit_rate > 0.05 else "neutral",
                "impact_score": -exit_rate,
                "confidence": 0.5,
                "notes": f"Projected exit rate: {exit_rate * 100:.1f}%",
            }
        )

        return pd.DataFrame(results)

    def evaluate_goal_achievement(self) -> dict[str, Any]:
        """Assesses the extent to which the regulation achieves its stated goals.

        Compares outcomes against the regulation's objectives using relevant metrics.

        Returns:
            Dictionary summarizing goal achievement metrics.
        """
        goals = self.baseline_data.get("regulation_goals", [])

        if not goals:
            return {
                "goals_evaluated": 0,
                "overall_achievement": 0.0,
                "goal_details": [],
                "notes": "No regulation goals specified in baseline data",
            }

        goal_results = []
        for goal in goals:
            goal_name = goal.get("name", "unnamed")
            target_value = goal.get("target_value", 1.0)
            current_value = goal.get("current_value", 0.0)
            metric_type = goal.get("metric_type", "ratio")

            if metric_type == "ratio" and target_value != 0:
                achievement = min(1.0, current_value / target_value)
            elif metric_type == "boolean":
                achievement = 1.0 if current_value else 0.0
            elif metric_type == "reduction" and target_value != 0:
                baseline_val = goal.get("baseline_value", current_value * 1.5)
                actual_reduction = baseline_val - current_value
                target_reduction = baseline_val - target_value
                achievement = (
                    min(1.0, actual_reduction / target_reduction)
                    if target_reduction != 0
                    else 0.0
                )
            else:
                achievement = 0.5

            goal_results.append(
                {
                    "goal_name": goal_name,
                    "target_value": target_value,
                    "current_value": current_value,
                    "achievement_score": achievement,
                    "status": "achieved"
                    if achievement >= 0.9
                    else "partial"
                    if achievement >= 0.5
                    else "not_achieved",
                }
            )

        avg_achievement = (
            sum(g["achievement_score"] for g in goal_results) / len(goal_results)
            if goal_results
            else 0.0
        )

        return {
            "goals_evaluated": len(goal_results),
            "overall_achievement": avg_achievement,
            "goals_achieved": sum(1 for g in goal_results if g["status"] == "achieved"),
            "goals_partial": sum(1 for g in goal_results if g["status"] == "partial"),
            "goals_not_achieved": sum(
                1 for g in goal_results if g["status"] == "not_achieved"
            ),
            "goal_details": goal_results,
        }

    def generate_assessment_summary(self) -> str:
        """Generates a textual summary of the regulatory impact assessment.

        Returns:
            A string containing the assessment summary.
        """
        logger.info("Generating assessment summary")
        # Baseline implementation
        costs_df = self.estimate_compliance_costs()
        admin = self.assess_administrative_burden()
        goals = self.evaluate_goal_achievement()

        summary = "Regulatory Impact Assessment Summary\n"
        summary += f"- Regulation: {self.regulation}\n"
        summary += f"- Affected entities: {len(self.affected_entities)}\n"
        summary += f"- Total compliance cost: ${admin.get('total_cost', 0):,.2f}\n"
        summary += f"- Total admin hours: {admin.get('total_hours', 0)}\n"
        summary += f"- Goals evaluated: {goals.get('goals_evaluated', 0)}\n"
        summary += (
            f"- Overall goal achievement: {goals.get('overall_achievement', 0):.1%}\n"
        )
        if not costs_df.empty and "estimated_cost" in costs_df.columns:
            total_compliance = costs_df["estimated_cost"].sum()
            summary += f"- Total compliance costs: ${total_compliance:,.2f}\n"
        return summary
