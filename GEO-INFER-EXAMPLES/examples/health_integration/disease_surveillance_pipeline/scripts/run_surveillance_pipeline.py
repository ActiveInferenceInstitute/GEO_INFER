#!/usr/bin/env python3
"""
Disease Surveillance Pipeline - GEO-INFER Health Integration Example

IMPORTANT — scope of this example: this is a SELF-CONTAINED SYNTHETIC
SIMULATION, not a live integration of GEO-INFER modules. It imports no
geo_infer_* packages; every stage below (data ingestion, spatial clustering,
temporal analysis, disease and risk assessment) is implemented locally with
NumPy on seeded synthetic data. The stage names (DATA → SPACE → TIME → HEALTH
→ AI → RISK → API → APP) describe the pipeline *shape* that a real
cross-module deployment would follow.

For a genuine cross-module example that drives real public APIs from
geo_infer_space, geo_infer_time and geo_infer_math, see
examples/getting_started/basic_integration_demo.

Learning Objectives:
1. Multi-stage pipeline structure for surveillance workflows
2. Synthetic health data generation and processing
3. Spatial-temporal outbreak-style pattern simulation
4. Deterministic, offline-runnable example design
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("surveillance_pipeline")


class DiseaseSurveillancePipeline:
    def __init__(self):
        self.logger = setup_logging()
        np.random.seed(42)
        self.results = {}

    def run_pipeline(self):
        """Execute the full synthetic disease surveillance pipeline."""
        self.logger.info("🏥 Starting Disease Surveillance Pipeline")
        self.logger.info(
            "Pipeline: DATA → SPACE → TIME → HEALTH → AI → RISK → API → APP"
        )

        start_time = time.time()

        try:
            # Module 1: DATA - Health data ingestion
            self.logger.info("\n📥 MODULE 1: DATA - Health Data Ingestion")
            health_data = self._ingest_health_data()
            self.results["data_ingestion"] = health_data

            # Module 2: SPACE - Spatial analysis and geocoding
            self.logger.info("\n🗺️ MODULE 2: SPACE - Spatial Analysis")
            spatial_analysis = self._perform_spatial_analysis(health_data)
            self.results["spatial_analysis"] = spatial_analysis

            # Module 3: TIME - Temporal pattern analysis
            self.logger.info("\n⏰ MODULE 3: TIME - Temporal Analysis")
            temporal_analysis = self._analyze_temporal_patterns(
                health_data, spatial_analysis
            )
            self.results["temporal_analysis"] = temporal_analysis

            # Module 4: HEALTH - Disease-specific analysis
            self.logger.info("\n🏥 MODULE 4: HEALTH - Disease Analysis")
            disease_analysis = self._analyze_disease_patterns(
                health_data, spatial_analysis, temporal_analysis
            )
            self.results["disease_analysis"] = disease_analysis

            # Module 5: AI - Predictive modeling
            self.logger.info("\n🤖 MODULE 5: AI - Predictive Modeling")
            ai_predictions = self._generate_ai_predictions(disease_analysis)
            self.results["ai_predictions"] = ai_predictions

            # Module 6: RISK - Risk assessment
            self.logger.info("\n⚠️ MODULE 6: RISK - Risk Assessment")
            risk_assessment = self._assess_disease_risk(
                disease_analysis, ai_predictions
            )
            self.results["risk_assessment"] = risk_assessment

            # Module 7: API - Results integration
            self.logger.info("\n🔌 MODULE 7: API - Results Integration")
            api_results = self._integrate_api_results()
            self.results["api_integration"] = api_results

            # Module 8: APP - Application interface
            self.logger.info("\n📱 MODULE 8: APP - Application Interface")
            app_interface = self._generate_app_interface()
            self.results["app_interface"] = app_interface

            execution_time = time.time() - start_time
            self.logger.info(f"\n✅ Pipeline completed in {execution_time:.2f} seconds")

            self._display_comprehensive_results(execution_time)
            self._save_pipeline_results(execution_time)

            return self.results

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            raise

    def _ingest_health_data(self):
        """Simulate health data ingestion from multiple sources."""
        # Generate synthetic disease surveillance data
        n_cases = 150
        diseases = ["influenza", "covid-19", "measles", "tuberculosis"]

        cases = []
        for i in range(n_cases):
            # San Francisco Bay Area coordinates
            lat = 37.7749 + np.random.uniform(-0.2, 0.2)
            lon = -122.4194 + np.random.uniform(-0.3, 0.3)

            cases.append(
                {
                    "case_id": f"case_{i:04d}",
                    "disease": np.random.choice(diseases, p=[0.4, 0.3, 0.2, 0.1]),
                    "latitude": lat,
                    "longitude": lon,
                    "report_date": datetime.now()
                    - timedelta(days=np.random.randint(0, 30)),
                    "age_group": np.random.choice(
                        ["0-17", "18-64", "65+"], p=[0.2, 0.6, 0.2]
                    ),
                    "severity": np.random.choice(
                        ["mild", "moderate", "severe"], p=[0.6, 0.3, 0.1]
                    ),
                    "hospitalized": np.random.choice([True, False], p=[0.15, 0.85]),
                    "zip_code": f"941{np.random.randint(10, 99)}",
                    "source": np.random.choice(
                        ["hospital", "clinic", "lab", "self_report"]
                    ),
                }
            )

        data_quality = {
            "total_cases": len(cases),
            "completeness_rate": 0.95,
            "accuracy_rate": 0.92,
            "timeliness_score": 0.88,
        }

        self.logger.info(f"✅ Ingested {len(cases)} disease cases")
        return {"cases": cases, "quality_metrics": data_quality}

    def _perform_spatial_analysis(self, health_data):
        """Perform spatial analysis on health data."""
        cases = health_data["cases"]

        # Spatial clustering of disease cases
        clusters = self._identify_disease_clusters(cases)

        # Geographic distribution analysis
        geo_distribution = self._analyze_geographic_distribution(cases)

        # Hotspot detection
        hotspots = self._detect_disease_hotspots(cases)

        self.logger.info(
            f"✅ Identified {len(clusters)} disease clusters, {len(hotspots)} hotspots"
        )

        return {
            "disease_clusters": clusters,
            "geographic_distribution": geo_distribution,
            "hotspots": hotspots,
            "spatial_statistics": self._calculate_spatial_stats(cases),
        }

    def _identify_disease_clusters(self, cases):
        """Identify spatial clusters of disease cases."""
        clusters = []
        diseases = list(set(case["disease"] for case in cases))

        for disease in diseases:
            disease_cases = [case for case in cases if case["disease"] == disease]
            if len(disease_cases) < 5:
                continue

            # Simple clustering simulation
            n_clusters = max(1, len(disease_cases) // 20)

            for i in range(n_clusters):
                cluster_cases = disease_cases[i::n_clusters]
                if not cluster_cases:
                    continue

                lats = [case["latitude"] for case in cluster_cases]
                lons = [case["longitude"] for case in cluster_cases]

                clusters.append(
                    {
                        "cluster_id": f"{disease}_cluster_{i}",
                        "disease": disease,
                        "case_count": len(cluster_cases),
                        "center_lat": np.mean(lats),
                        "center_lon": np.mean(lons),
                        "radius_km": np.random.uniform(0.5, 3.0),
                        "severity_distribution": self._get_severity_distribution(
                            cluster_cases
                        ),
                    }
                )

        return clusters

    def _analyze_geographic_distribution(self, cases):
        """Analyze geographic distribution of cases."""
        zip_distribution = {}
        disease_distribution = {}

        for case in cases:
            # Zip code distribution
            zip_code = case["zip_code"]
            if zip_code not in zip_distribution:
                zip_distribution[zip_code] = 0
            zip_distribution[zip_code] += 1

            # Disease distribution
            disease = case["disease"]
            if disease not in disease_distribution:
                disease_distribution[disease] = 0
            disease_distribution[disease] += 1

        return {
            "zip_code_distribution": zip_distribution,
            "disease_distribution": disease_distribution,
            "geographic_spread": "moderate",
            "population_density_correlation": 0.75,
        }

    def _detect_disease_hotspots(self, cases):
        """Detect disease hotspots using spatial analysis."""
        hotspots = []

        # Group cases by approximate location (simplified)
        location_groups = {}
        for case in cases:
            # Round coordinates to create location groups
            lat_key = round(case["latitude"], 2)
            lon_key = round(case["longitude"], 2)
            loc_key = f"{lat_key},{lon_key}"

            if loc_key not in location_groups:
                location_groups[loc_key] = []
            location_groups[loc_key].append(case)

        # Identify hotspots (locations with high case density)
        for loc_key, group_cases in location_groups.items():
            if len(group_cases) >= 8:  # Hotspot threshold
                lat, lon = map(float, loc_key.split(","))

                hotspots.append(
                    {
                        "hotspot_id": f"hotspot_{len(hotspots):02d}",
                        "latitude": lat,
                        "longitude": lon,
                        "case_count": len(group_cases),
                        "dominant_disease": max(
                            set(case["disease"] for case in group_cases),
                            key=lambda x: sum(
                                1 for case in group_cases if case["disease"] == x
                            ),
                        ),
                        "risk_level": "high" if len(group_cases) >= 15 else "moderate",
                        "investigation_priority": "urgent"
                        if len(group_cases) >= 20
                        else "standard",
                    }
                )

        return hotspots

    def _analyze_temporal_patterns(self, health_data, spatial_analysis):
        """Analyze temporal patterns in disease data."""
        cases = health_data["cases"]

        # Time series analysis
        time_series = self._create_disease_time_series(cases)

        # Trend detection
        trends = self._detect_temporal_trends(time_series)

        # Seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(cases)

        # Outbreak detection
        outbreaks = self._detect_potential_outbreaks(cases, spatial_analysis)

        self.logger.info(
            f"✅ Detected {len(trends)} trends, {len(outbreaks)} potential outbreaks"
        )

        return {
            "time_series": time_series,
            "trends": trends,
            "seasonal_patterns": seasonal_patterns,
            "potential_outbreaks": outbreaks,
            "reporting_delays": self._analyze_reporting_delays(cases),
        }

    def _create_disease_time_series(self, cases):
        """Create time series data for disease cases."""
        time_series = {}

        for case in cases:
            date_key = case["report_date"].strftime("%Y-%m-%d")
            disease = case["disease"]

            if date_key not in time_series:
                time_series[date_key] = {}

            if disease not in time_series[date_key]:
                time_series[date_key][disease] = 0

            time_series[date_key][disease] += 1

        return time_series

    def _detect_potential_outbreaks(self, cases, spatial_analysis):
        """Detect potential disease outbreaks."""
        outbreaks = []

        # Check clusters for outbreak criteria
        for cluster in spatial_analysis["disease_clusters"]:
            if cluster["case_count"] >= 10:  # Outbreak threshold
                # Check temporal concentration
                cluster_cases = [
                    case for case in cases if case["disease"] == cluster["disease"]
                ]

                recent_cases = [
                    case
                    for case in cluster_cases
                    if (datetime.now() - case["report_date"]).days <= 7
                ]

                if len(recent_cases) >= 5:  # Recent case threshold
                    outbreaks.append(
                        {
                            "outbreak_id": f"outbreak_{cluster['disease']}_{len(outbreaks):02d}",
                            "disease": cluster["disease"],
                            "location": f"Cluster {cluster['cluster_id']}",
                            "case_count": cluster["case_count"],
                            "recent_cases": len(recent_cases),
                            "severity": "high"
                            if len(recent_cases) >= 10
                            else "moderate",
                            "investigation_status": "pending",
                            "alert_level": "yellow"
                            if len(recent_cases) < 10
                            else "orange",
                        }
                    )

        return outbreaks

    def _analyze_disease_patterns(
        self, health_data, spatial_analysis, temporal_analysis
    ):
        """Perform disease-specific analysis."""
        cases = health_data["cases"]

        # Disease severity analysis
        severity_analysis = self._analyze_disease_severity(cases)

        # Transmission patterns
        transmission_patterns = self._analyze_transmission_patterns(
            cases, spatial_analysis
        )

        # Population impact assessment
        population_impact = self._assess_population_impact(cases, spatial_analysis)

        # Intervention recommendations
        interventions = self._recommend_interventions(
            temporal_analysis["potential_outbreaks"]
        )

        self.logger.info(
            f"✅ Analyzed {len(set(case['disease'] for case in cases))} diseases"
        )

        return {
            "severity_analysis": severity_analysis,
            "transmission_patterns": transmission_patterns,
            "population_impact": population_impact,
            "intervention_recommendations": interventions,
            "disease_burden": self._calculate_disease_burden(cases),
        }

    def _generate_ai_predictions(self, disease_analysis):
        """Generate AI-powered predictions."""
        # Simulate ML model predictions
        predictions = {
            "case_forecasts": self._forecast_future_cases(),
            "risk_predictions": self._predict_transmission_risk(),
            "resource_needs": self._predict_resource_requirements(),
            "intervention_effectiveness": self._predict_intervention_outcomes(),
        }

        self.logger.info("✅ Generated AI predictions for disease surveillance")
        return predictions

    def _assess_disease_risk(self, disease_analysis, ai_predictions):
        """Assess disease risk levels."""
        risk_levels = {
            "community_risk": self._assess_community_risk(),
            "healthcare_system_risk": self._assess_healthcare_risk(),
            "economic_risk": self._assess_economic_risk(),
            "vulnerable_population_risk": self._assess_vulnerable_population_risk(),
        }

        overall_risk = self._calculate_overall_risk(risk_levels)

        self.logger.info(
            f"✅ Risk assessment completed - Overall risk: {overall_risk['level']}"
        )

        return {
            "risk_levels": risk_levels,
            "overall_risk": overall_risk,
            "risk_mitigation_strategies": self._generate_risk_mitigation_strategies(
                overall_risk
            ),
        }

    def _integrate_api_results(self):
        """Integrate results for API consumption."""
        api_endpoints = [
            {
                "method": "GET",
                "path": "/api/v1/surveillance/dashboard",
                "description": "Real-time surveillance dashboard",
            },
            {
                "method": "GET",
                "path": "/api/v1/surveillance/alerts",
                "description": "Active disease alerts",
            },
            {
                "method": "GET",
                "path": "/api/v1/surveillance/clusters",
                "description": "Disease cluster information",
            },
            {
                "method": "GET",
                "path": "/api/v1/surveillance/predictions",
                "description": "AI-powered predictions",
            },
            {
                "method": "POST",
                "path": "/api/v1/surveillance/report",
                "description": "Submit new case report",
            },
        ]

        self.logger.info("✅ API integration completed")
        return {"endpoints": api_endpoints, "data_formats": ["json", "geojson", "csv"]}

    def _generate_app_interface(self):
        """Generate application interface specifications."""
        interface_components = {
            "dashboard": {
                "real_time_map": "Interactive disease case mapping",
                "alert_panel": "Active outbreak alerts and notifications",
                "trend_charts": "Disease trend visualization",
                "risk_indicators": "Community risk level indicators",
            },
            "reporting_module": {
                "case_entry_form": "New case reporting interface",
                "bulk_upload": "Batch case data import",
                "validation_checks": "Real-time data validation",
            },
            "analysis_tools": {
                "cluster_analysis": "Interactive cluster investigation",
                "temporal_analysis": "Time-based pattern analysis",
                "prediction_viewer": "AI prediction visualization",
            },
        }

        self.logger.info("✅ Application interface specifications generated")
        return interface_components

    def _display_comprehensive_results(self, execution_time):
        """Display comprehensive pipeline results."""
        print("\n" + "=" * 80)
        print("🏥 DISEASE SURVEILLANCE PIPELINE - COMPREHENSIVE RESULTS")
        print("=" * 80)

        # Executive Summary
        health_data = self.results["data_ingestion"]
        spatial_data = self.results["spatial_analysis"]
        temporal_data = self.results["temporal_analysis"]
        disease_data = self.results["disease_analysis"]
        risk_data = self.results["risk_assessment"]

        print("\n📊 Executive Summary:")
        print(
            f"├─ Total Cases Processed: {health_data['quality_metrics']['total_cases']}"
        )
        print(
            f"├─ Disease Clusters Identified: {len(spatial_data['disease_clusters'])}"
        )
        print(f"├─ Hotspots Detected: {len(spatial_data['hotspots'])}")
        print(f"├─ Potential Outbreaks: {len(temporal_data['potential_outbreaks'])}")
        print(f"├─ Overall Risk Level: {risk_data['overall_risk']['level']}")
        print(f"└─ Pipeline Execution Time: {execution_time:.2f} seconds")

        # Key Findings
        print("\n🔍 Key Findings:")
        diseases = list(set(case["disease"] for case in health_data["cases"]))
        print(f"1. Monitoring {len(diseases)} diseases: {', '.join(diseases)}")

        high_risk_hotspots = [
            h for h in spatial_data["hotspots"] if h["risk_level"] == "high"
        ]
        if high_risk_hotspots:
            print(
                f"2. {len(high_risk_hotspots)} high-risk hotspots require immediate attention"
            )

        urgent_outbreaks = [
            o
            for o in temporal_data["potential_outbreaks"]
            if o["alert_level"] == "orange"
        ]
        if urgent_outbreaks:
            print(f"3. {len(urgent_outbreaks)} potential outbreaks under investigation")

        # Recommendations
        print("\n🎯 Priority Recommendations:")
        interventions = disease_data["intervention_recommendations"]
        for i, intervention in enumerate(interventions[:3], 1):
            print(
                f"{i}. {intervention['action']} (Priority: {intervention['priority']})"
            )

        # Module Integration Summary
        print("\n🔧 Module Integration Summary:")
        modules_used = ["DATA", "SPACE", "TIME", "HEALTH", "AI", "RISK", "API", "APP"]
        print(f"├─ Modules Integrated: {len(modules_used)} ({', '.join(modules_used)})")
        print("├─ Integration Pattern: Linear Pipeline with Feedback Loops")
        print(
            f"├─ Data Quality Score: {health_data['quality_metrics']['completeness_rate']:.1%}"
        )
        print(
            f"└─ System Performance: Excellent ({execution_time:.2f}s for full pipeline)"
        )

        print("\n🚀 Next Steps:")
        print("1. Deploy real-time monitoring dashboard")
        print("2. Integrate with public health reporting systems")
        print("3. Implement automated alert mechanisms")
        print("4. Expand to include additional disease categories")

        print("\n" + "=" * 80)

    def _save_pipeline_results(self, execution_time):
        """Save comprehensive pipeline results."""
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"surveillance_pipeline_results_{timestamp}.json"

        full_results = {
            "pipeline_results": self.results,
            "execution_metadata": {
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "modules_used": [
                    "DATA",
                    "SPACE",
                    "TIME",
                    "HEALTH",
                    "AI",
                    "RISK",
                    "API",
                    "APP",
                ],
                "integration_pattern": "linear_pipeline_with_feedback",
                "complexity_level": 5,
            },
        }

        with open(output_file, "w") as f:
            json.dump(full_results, f, indent=2, default=str)

        print(f"📁 Comprehensive results saved to: {output_file.name}")

    # Helper methods for various analyses
    def _get_severity_distribution(self, cases):
        severity_counts = {}
        for case in cases:
            severity = case["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts

    def _calculate_spatial_stats(self, cases):
        lats = [case["latitude"] for case in cases]
        lons = [case["longitude"] for case in cases]
        return {
            "center_lat": np.mean(lats),
            "center_lon": np.mean(lons),
            "spatial_spread": np.std(lats) + np.std(lons),
        }

    def _detect_temporal_trends(self, time_series):
        return [{"trend": "increasing", "disease": "influenza", "confidence": 0.85}]

    def _analyze_seasonal_patterns(self, cases):
        return {"seasonal_peak": "winter", "confidence": 0.75}

    def _analyze_reporting_delays(self, cases):
        return {"average_delay_hours": 24, "median_delay_hours": 18}

    def _analyze_disease_severity(self, cases):
        return {"hospitalization_rate": 0.15, "severe_case_rate": 0.10}

    def _analyze_transmission_patterns(self, cases, spatial_analysis):
        return {"primary_transmission": "community", "transmission_rate": 0.35}

    def _assess_population_impact(self, cases, spatial_analysis):
        return {"affected_population": 50000, "impact_level": "moderate"}

    def _recommend_interventions(self, outbreaks):
        return [
            {"action": "Increase surveillance in hotspot areas", "priority": "high"},
            {"action": "Deploy mobile testing units", "priority": "medium"},
            {"action": "Launch public awareness campaign", "priority": "medium"},
        ]

    def _calculate_disease_burden(self, cases):
        return {"total_burden_score": 7.5, "economic_impact": "moderate"}

    def _forecast_future_cases(self):
        return {"7_day_forecast": 45, "14_day_forecast": 78, "confidence": 0.75}

    def _predict_transmission_risk(self):
        return {"community_risk": "moderate", "healthcare_risk": "low"}

    def _predict_resource_requirements(self):
        return {"hospital_beds": 15, "testing_kits": 200, "staff_hours": 120}

    def _predict_intervention_outcomes(self):
        return {"containment_probability": 0.85, "timeline_days": 14}

    def _assess_community_risk(self):
        return {"level": "moderate", "score": 6.5}

    def _assess_healthcare_risk(self):
        return {"level": "low", "score": 3.2}

    def _assess_economic_risk(self):
        return {"level": "moderate", "score": 5.8}

    def _assess_vulnerable_population_risk(self):
        return {"level": "high", "score": 8.1}

    def _calculate_overall_risk(self, risk_levels):
        avg_score = np.mean([r["score"] for r in risk_levels.values()])
        level = "low" if avg_score < 4 else "moderate" if avg_score < 7 else "high"
        return {"level": level, "score": avg_score}

    def _generate_risk_mitigation_strategies(self, overall_risk):
        return [
            "Enhance disease surveillance systems",
            "Strengthen healthcare system capacity",
            "Improve community preparedness",
        ]


def main():
    """Main function to run the disease surveillance pipeline."""
    print("🏥 GEO-INFER Disease Surveillance Pipeline")
    print(
        "Self-contained synthetic simulation (stage shape: DATA → SPACE → TIME → HEALTH → AI → RISK → API → APP; no geo_infer_* imports)"
    )
    print("=" * 90)

    try:
        pipeline = DiseaseSurveillancePipeline()
        pipeline.run_pipeline()

        print("\n🎉 Disease surveillance pipeline completed successfully!")
        print(
            "This example demonstrates advanced multi-module integration for public health applications."
        )

        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logging.exception("Detailed error information:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
