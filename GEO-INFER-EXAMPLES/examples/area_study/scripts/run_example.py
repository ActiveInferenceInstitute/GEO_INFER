#!/usr/bin/env python3
"""
Comprehensive Area Study Template - GEO-INFER Multi-Disciplinary Analysis

This example demonstrates a comprehensive area study integrating:
Technical Infrastructure + Social Systems + Environmental Factors

Learning Objectives:
1. Multi-source data integration (technical, social, environmental)
2. Community-engaged spatial analysis
3. Cross-domain impact assessment
4. Ethical data governance frameworks
5. Participatory research methods
6. Sustainable area planning
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import zlib


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("area_study")


class ComprehensiveAreaStudy:
    def __init__(self, study_id="demo_area_study_001"):
        self.study_id = study_id
        self.logger = setup_logging()
        np.random.seed(42)
        self.results = {}

    def run_area_study(self):
        """Execute the complete area study analysis."""
        self.logger.info("🏛️ Starting Comprehensive Area Study")
        self.logger.info("Integration: Technical + Social + Environmental Data")

        start_time = time.time()

        try:
            # Phase 1: Study Design and Data Collection
            self.logger.info("\n📋 PHASE 1: Study Design and Data Collection")
            study_design = self._design_study_framework()
            self.results["study_design"] = study_design

            # Phase 2: Multi-Source Data Integration
            self.logger.info("\n🔗 PHASE 2: Multi-Source Data Integration")
            technical_data = self._collect_technical_data()
            social_data = self._collect_social_data()
            environmental_data = self._collect_environmental_data()

            integrated_data = self._integrate_multi_source_data(
                technical_data, social_data, environmental_data
            )
            self.results["integrated_data"] = integrated_data

            # Phase 3: Spatial Analysis
            self.logger.info("\n🗺️ PHASE 3: Multi-Scale Spatial Analysis")
            spatial_analysis = self._perform_spatial_analysis(integrated_data)
            self.results["spatial_analysis"] = spatial_analysis

            # Phase 4: Cross-Domain Impact Assessment
            self.logger.info("\n📊 PHASE 4: Cross-Domain Impact Assessment")
            impact_assessment = self._assess_cross_domain_impacts(spatial_analysis)
            self.results["impact_assessment"] = impact_assessment

            # Phase 5: Community Engagement and Validation
            self.logger.info("\n👥 PHASE 5: Community Engagement and Validation")
            community_engagement = self._conduct_community_engagement(impact_assessment)
            self.results["community_engagement"] = community_engagement

            # Phase 6: Sustainable Planning and Recommendations
            self.logger.info("\n🌱 PHASE 6: Sustainable Planning and Recommendations")
            sustainability_plan = self._develop_sustainability_plan(
                impact_assessment, community_engagement
            )
            self.results["sustainability_plan"] = sustainability_plan

            execution_time = time.time() - start_time
            self.logger.info(".2f")
            self._display_area_study_results(execution_time)
            self._save_area_study_results(execution_time)

            return self.results

        except Exception as e:
            self.logger.error(f"❌ Area study failed: {e}")
            raise

    def _design_study_framework(self):
        """Design the area study framework and methodology."""
        # Define study boundaries (example: neighborhood area)
        study_area = {
            "name": "Downtown Neighborhood Study",
            "area_type": "urban_neighborhood",
            "total_area_hectares": 150,
            "population_estimate": 8500,
            "h3_resolution": 9,  # ~100m hexagons
            "study_period": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "data_collection_period": "3_months",
            },
        }

        # Define research questions
        research_questions = {
            "technical": [
                "What is the digital connectivity infrastructure coverage?",
                "How does physical infrastructure condition vary spatially?",
                "What IoT sensors and smart city infrastructure exist?",
            ],
            "social": [
                "How do community networks function within the area?",
                "What are resident priorities and concerns?",
                "How do different demographic groups experience the area?",
            ],
            "environmental": [
                "What is the environmental health burden distribution?",
                "How does green space access vary across neighborhoods?",
                "What are local climate adaptation needs?",
            ],
        }

        # Establish ethical framework
        ethical_framework = {
            "data_privacy": "community_controlled",
            "community_consent": "required_workshops",
            "information_rights": "community_access_priority",
            "benefit_sharing": "direct_community_returns",
            "cultural_competence": "local_language_support",
        }

        self.logger.info("✅ Study framework designed")
        return {
            "study_area": study_area,
            "research_questions": research_questions,
            "ethical_framework": ethical_framework,
            "methodology_version": "1.0",
        }

    def _collect_technical_data(self):
        """Collect technical infrastructure data."""
        # Simulate IoT sensor network data
        iot_sensors = []
        sensor_types = [
            "environmental",
            "traffic",
            "air_quality",
            "noise",
            "connectivity",
        ]

        for i in range(50):  # 50 sensors across the area
            sensor = {
                "sensor_id": "03d",
                "type": np.random.choice(sensor_types),
                "location": {
                    "latitude": 40.7128 + np.random.uniform(-0.01, 0.01),
                    "longitude": -74.0060 + np.random.uniform(-0.01, 0.01),
                },
                "status": "active",
                "data_streams": {
                    "connectivity": {
                        "download_speed": np.random.uniform(50, 1000),  # Mbps
                        "upload_speed": np.random.uniform(10, 100),
                        "latency": np.random.uniform(5, 50),  # ms
                    },
                    "environmental": {
                        "air_quality_index": np.random.uniform(20, 80),
                        "noise_level": np.random.uniform(40, 90),  # dB
                        "temperature": np.random.uniform(15, 30),
                    },
                },
                "last_update": datetime.now()
                - timedelta(minutes=np.random.randint(0, 60)),
            }
            iot_sensors.append(sensor)

        # Physical infrastructure assessment
        infrastructure = {
            "roads": {"condition_score": 0.75, "coverage_km": 45.2},
            "buildings": {"total_count": 850, "avg_condition": 0.68},
            "utilities": {
                "electricity_coverage": 0.98,
                "water_coverage": 0.96,
                "internet_coverage": 0.82,
            },
            "public_spaces": {"parks": 12, "total_green_space_ha": 23.5},
        }

        self.logger.info(
            f"✅ Collected technical data: {len(iot_sensors)} sensors, infrastructure assessment"
        )
        return {
            "iot_sensors": iot_sensors,
            "infrastructure": infrastructure,
            "connectivity_providers": ["Verizon", "AT&T", "Starlink", "Local Co-op"],
            "smart_city_features": [
                "traffic_sensors",
                "air_quality_monitors",
                "smart_lighting",
            ],
        }

    def _collect_social_data(self):
        """Collect social and community data."""
        # Community demographics
        demographics = {
            "total_population": 8500,
            "households": 3200,
            "age_distribution": {
                "0-18": 0.22,
                "19-34": 0.28,
                "35-54": 0.25,
                "55-74": 0.18,
                "75+": 0.07,
            },
            "ethnic_composition": {
                "White": 0.45,
                "Black": 0.28,
                "Hispanic": 0.15,
                "Asian": 0.08,
                "Other": 0.04,
            },
            "median_household_income": 65000,
            "education_levels": {
                "high_school_or_less": 0.32,
                "some_college": 0.28,
                "bachelors_degree": 0.25,
                "graduate_degree": 0.15,
            },
        }

        # Community organizations and networks
        community_orgs = [
            {
                "name": "Downtown Neighborhood Association",
                "type": "civic",
                "members": 450,
            },
            {"name": "Local Business Alliance", "type": "economic", "members": 120},
            {
                "name": "Community Garden Collective",
                "type": "environmental",
                "members": 85,
            },
            {
                "name": "Youth Development Center",
                "type": "social_services",
                "members": 200,
            },
            {
                "name": "Senior Services Network",
                "type": "social_services",
                "members": 180,
            },
        ]

        # Social network analysis (simplified)
        social_networks = {
            "collaboration_ties": 156,
            "resource_sharing": 89,
            "information_flow": 234,
            "conflict_points": 12,
            "network_density": 0.67,
            "central_actors": ["Neighborhood Association", "Business Alliance"],
        }

        # Community priorities (from surveys)
        community_priorities = [
            {"issue": "Public Safety", "priority": "high", "support": 0.89},
            {"issue": "Digital Access", "priority": "high", "support": 0.76},
            {"issue": "Green Space", "priority": "medium", "support": 0.71},
            {"issue": "Local Economy", "priority": "medium", "support": 0.68},
            {"issue": "Transportation", "priority": "medium", "support": 0.65},
        ]

        self.logger.info(
            f"✅ Collected social data: demographics, {len(community_orgs)} organizations, network analysis"
        )
        return {
            "demographics": demographics,
            "community_organizations": community_orgs,
            "social_networks": social_networks,
            "community_priorities": community_priorities,
            "participation_rate": 0.42,  # 42% survey participation
        }

    def _collect_environmental_data(self):
        """Collect environmental and health data."""
        # Environmental quality metrics
        environmental_quality = {
            "air_quality": {
                "pm2_5_average": 12.5,  # μg/m³
                "ozone_level": 45,
                "air_quality_index": 68,
            },
            "water_quality": {
                "potable_water_compliance": 0.97,
                "surface_water_quality": 0.78,
            },
            "noise_levels": {
                "average_daytime": 62,  # dB
                "average_nighttime": 48,
            },
            "green_space_coverage": 0.18,  # 18% of area
            "urban_heat_island_effect": 3.2,  # °C difference
        }

        # Biodiversity assessment
        biodiversity = {
            "species_richness": {"plants": 156, "birds": 43, "insects": 89},
            "habitat_types": [
                "urban_park",
                "community_garden",
                "street_tree",
                "green_roof",
            ],
            "ecosystem_services": {
                "air_filtration": "medium",
                "carbon_sequestration": "low",
                "temperature_regulation": "medium",
            },
        }

        # Health indicators
        health_indicators = {
            "environmental_health_index": 0.72,
            "respiratory_illness_rate": 0.085,  # 8.5% of population
            "heat_related_illness": 0.032,
            "access_to_healthcare": {
                "primary_care_facilities": 3,
                "emergency_services": 1,
                "mental_health_services": 2,
            },
            "physical_activity_spaces": {
                "parks": 12,
                "recreational_facilities": 8,
                "walking_trails": 25,  # km
            },
        }

        # Climate vulnerability
        climate_vulnerability = {
            "flood_risk_zones": ["Zone_A", "Zone_B"],  # FEMA flood zones
            "heat_vulnerability_index": 0.65,
            "extreme_weather_events": {
                "heat_waves": 8,  # per year
                "heavy_rainfall": 12,
                "strong_winds": 6,
            },
            "adaptation_measures": [
                "Tree planting program",
                "Cool roof initiative",
                "Emergency response planning",
            ],
        }

        self.logger.info(
            "✅ Collected environmental data: air quality, biodiversity, health indicators, climate data"
        )
        return {
            "environmental_quality": environmental_quality,
            "biodiversity": biodiversity,
            "health_indicators": health_indicators,
            "climate_vulnerability": climate_vulnerability,
            "data_collection_date": datetime.now().isoformat(),
        }

    def _integrate_multi_source_data(self, technical, social, environmental):
        """Integrate data from multiple sources."""
        # Create unified spatial framework
        h3_cells = self._generate_h3_grid()

        # Spatial data integration
        integrated_spatial_data = []
        for cell in h3_cells:
            cell_data = {
                "h3_index": cell,
                "technical_metrics": self._extract_technical_cell_data(cell, technical),
                "social_metrics": self._extract_social_cell_data(cell, social),
                "environmental_metrics": self._extract_environmental_cell_data(
                    cell, environmental
                ),
            }
            integrated_spatial_data.append(cell_data)

        # Cross-domain correlation analysis
        correlations = self._analyze_cross_domain_correlations(
            technical, social, environmental
        )

        # Data quality assessment
        quality_metrics = {
            "technical_completeness": 0.89,
            "social_completeness": 0.76,
            "environmental_completeness": 0.94,
            "spatial_coverage": 0.91,
            "temporal_consistency": 0.85,
        }

        self.logger.info(
            f"✅ Integrated multi-source data: {len(integrated_spatial_data)} H3 cells analyzed"
        )
        return {
            "spatial_data": integrated_spatial_data,
            "correlations": correlations,
            "quality_metrics": quality_metrics,
            "integration_timestamp": datetime.now().isoformat(),
        }

    def _perform_spatial_analysis(self, integrated_data):
        """Perform multi-scale spatial analysis."""
        spatial_data = integrated_data["spatial_data"]

        # Multi-scale analysis (neighborhood, district, area levels)
        scales = {
            "neighborhood": self._analyze_neighborhood_scale(spatial_data),
            "district": self._analyze_district_scale(spatial_data),
            "area": self._analyze_area_scale(spatial_data),
        }

        # Hotspot identification
        hotspots = {
            "technical_deficit_zones": self._identify_technical_hotspots(spatial_data),
            "social_vulnerability_zones": self._identify_social_hotspots(spatial_data),
            "environmental_concern_zones": self._identify_environmental_hotspots(
                spatial_data
            ),
        }

        # Accessibility analysis
        accessibility = {
            "connectivity_access": self._analyze_connectivity_access(spatial_data),
            "green_space_access": self._analyze_green_space_access(spatial_data),
            "service_accessibility": self._analyze_service_access(spatial_data),
        }

        # Spatial equity assessment
        equity_metrics = self._assess_spatial_equity(spatial_data)

        self.logger.info(
            "✅ Spatial analysis completed: multi-scale patterns, hotspots, accessibility"
        )
        return {
            "multi_scale_analysis": scales,
            "hotspots": hotspots,
            "accessibility": accessibility,
            "equity_metrics": equity_metrics,
        }

    def _assess_cross_domain_impacts(self, spatial_analysis):
        """Assess cross-domain impacts and interactions."""
        # Technical-Social interactions
        tech_social_impacts = {
            "digital_divide_impact": self._analyze_digital_divide_impact(
                spatial_analysis
            ),
            "infrastructure_social_equity": self._analyze_infrastructure_equity(
                spatial_analysis
            ),
            "connectivity_community_engagement": self._analyze_connectivity_engagement(
                spatial_analysis
            ),
        }

        # Social-Environmental interactions
        social_env_impacts = {
            "community_environmental_stewardship": self._analyze_community_stewardship(
                spatial_analysis
            ),
            "environmental_justice_indicators": self._analyze_environmental_justice(
                spatial_analysis
            ),
            "green_space_social_cohesion": self._analyze_green_space_cohesion(
                spatial_analysis
            ),
        }

        # Technical-Environmental interactions
        tech_env_impacts = {
            "smart_infrastructure_sustainability": self._analyze_smart_sustainability(
                spatial_analysis
            ),
            "iot_environmental_monitoring_impact": self._analyze_iot_monitoring_impact(
                spatial_analysis
            ),
            "infrastructure_resilience": self._analyze_infrastructure_resilience(
                spatial_analysis
            ),
        }

        # Overall impact assessment
        overall_impact = {
            "composite_resilience_score": self._calculate_composite_resilience(
                tech_social_impacts, social_env_impacts, tech_env_impacts
            ),
            "sustainability_index": self._calculate_sustainability_index(
                spatial_analysis
            ),
            "quality_of_life_score": self._calculate_quality_of_life(spatial_analysis),
        }

        self.logger.info("✅ Cross-domain impact assessment completed")
        return {
            "tech_social_impacts": tech_social_impacts,
            "social_env_impacts": social_env_impacts,
            "tech_env_impacts": tech_env_impacts,
            "overall_impact": overall_impact,
        }

    def _conduct_community_engagement(self, impact_assessment):
        """Conduct community engagement and validation."""
        # Simulate community workshops
        workshops = [
            {
                "type": "data_validation_workshop",
                "participants": 45,
                "key_feedback": "Confirmed connectivity gaps in low-income areas",
            },
            {
                "type": "priority_setting_workshop",
                "participants": 52,
                "key_feedback": "Public safety and digital access ranked highest",
            },
            {
                "type": "solution_design_workshop",
                "participants": 38,
                "key_feedback": "Community-led monitoring systems preferred",
            },
        ]

        # Community feedback integration
        community_validation = {
            "findings_accuracy": 0.87,  # 87% agreement with community perceptions
            "priority_alignment": 0.82,
            "solution_acceptance": 0.79,
        }

        # Stakeholder perspectives
        stakeholder_groups = {
            "residents": {"satisfaction": 0.76, "engagement_level": 0.68},
            "business_owners": {"satisfaction": 0.82, "engagement_level": 0.74},
            "community_leaders": {"satisfaction": 0.89, "engagement_level": 0.91},
            "local_government": {"satisfaction": 0.71, "engagement_level": 0.85},
        }

        self.logger.info(
            f"✅ Community engagement completed: {sum(w['participants'] for w in workshops)} total participants"
        )
        return {
            "workshops": workshops,
            "community_validation": community_validation,
            "stakeholder_perspectives": stakeholder_groups,
            "engagement_effectiveness": 0.83,
        }

    def _develop_sustainability_plan(self, impact_assessment, community_engagement):
        """Develop sustainable area planning recommendations."""
        # Prioritized recommendations
        recommendations = [
            {
                "priority": "high",
                "domain": "technical",
                "recommendation": "Deploy community Wi-Fi hotspots in connectivity deserts",
                "implementation_cost": "medium",
                "community_support": 0.91,
                "expected_impact": "high",
            },
            {
                "priority": "high",
                "domain": "social",
                "recommendation": "Establish community-led safety monitoring program",
                "implementation_cost": "low",
                "community_support": 0.88,
                "expected_impact": "high",
            },
            {
                "priority": "medium",
                "domain": "environmental",
                "recommendation": "Expand green infrastructure in heat-vulnerable zones",
                "implementation_cost": "high",
                "community_support": 0.76,
                "expected_impact": "medium",
            },
        ]

        # Implementation timeline
        timeline = {
            "short_term": ["Community Wi-Fi deployment", "Safety monitoring setup"],
            "medium_term": [
                "Green infrastructure expansion",
                "Digital literacy programs",
            ],
            "long_term": [
                "Comprehensive infrastructure upgrade",
                "Climate adaptation master plan",
            ],
        }

        # Monitoring framework
        monitoring = {
            "indicators": [
                "Digital connectivity coverage",
                "Community safety perceptions",
                "Environmental quality metrics",
                "Social cohesion indicators",
            ],
            "frequency": "quarterly",
            "responsibility": "community-governance partnership",
        }

        self.logger.info(
            f"✅ Sustainability plan developed: {len(recommendations)} key recommendations"
        )
        return {
            "recommendations": recommendations,
            "implementation_timeline": timeline,
            "monitoring_framework": monitoring,
            "resource_requirements": self._estimate_resource_needs(recommendations),
        }

    def _display_area_study_results(self, execution_time):
        """Display comprehensive area study results."""
        print("\n" + "=" * 80)
        print("🏛️ COMPREHENSIVE AREA STUDY - MULTI-DISCIPLINARY ANALYSIS")
        print("=" * 80)

        # Study Overview
        study_design = self.results["study_design"]
        print("\n📋 Study Overview:")
        print(f"├─ Area: {study_design['study_area']['name']}")
        print(f"├─ Population: {study_design['study_area']['population_estimate']:,}")
        print(
            f"├─ Area Size: {study_design['study_area']['total_area_hectares']} hectares"
        )
        print("└─ Analysis Type: Multi-disciplinary integration")

        # Key Findings
        impact = self.results["impact_assessment"]["overall_impact"]
        print("\n💡 Key Findings:")
        print(
            f"├─ Composite Resilience Score: {impact.get('composite_resilience_score', 0):.2f}"
        )
        print(f"├─ Sustainability Index: {impact.get('sustainability_index', 0):.2f}")
        print(f"└─ Quality of Life Score: {impact.get('quality_of_life_score', 0):.2f}")
        # Domain-Specific Insights
        print("\n🔍 Domain Insights:")
        technical = self.results["integrated_data"]["quality_metrics"]
        print(f"├─ Technical Completeness: {technical.get('completeness', 0):.1%}")
        social = self.results["community_engagement"]["engagement_effectiveness"]
        print(f"├─ Social Engagement: {social:.1%}")
        environmental = self.results["integrated_data"]["spatial_data"][0][
            "environmental_metrics"
        ]
        print(f"├─ Air Quality Index: {environmental['air_quality_index']}")
        print(f"└─ Green Space Coverage: {environmental['green_space_coverage']}")

        # Community Priorities
        community = self.results["community_engagement"]["community_validation"]
        print("\n👥 Community Engagement:")
        print(f"├─ Accuracy Validation: {community.get('findings_accuracy', 0):.1%}")
        print(f"├─ Priority Alignment: {community.get('priority_alignment', 0):.1%}")
        print(f"└─ Solution Acceptance: {community.get('solution_acceptance', 0):.1%}")
        # Recommendations
        sustainability = self.results["sustainability_plan"]
        high_priority = [
            r for r in sustainability["recommendations"] if r["priority"] == "high"
        ]
        print("\n🎯 Top Recommendations:")
        for i, rec in enumerate(high_priority[:3], 1):
            print(
                f"{i}. {rec['recommendation']} (Support: {rec['community_support']:.0%})"
            )

        # Technical Integration
        print("\n🔧 Integration Performance:")
        modules_used = ["SPACE", "DATA", "PLACE", "PEP", "IOT", "BIO", "HEALTH", "API"]
        print(f"├─ Modules Integrated: {len(modules_used)}")
        print("├─ Integration Pattern: Multi-source spatial fusion")
        print(f"├─ System Performance: {execution_time:.2f} seconds")
        print(
            f"└─ Analysis Resolution: H3 Level {study_design['study_area']['h3_resolution']}"
        )

        print("\n" + "=" * 80)

    def _save_area_study_results(self, execution_time):
        """Save comprehensive area study results."""
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"area_study_results_{timestamp}.json"

        full_results = {
            "area_study_results": self.results,
            "execution_metadata": {
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "modules_used": [
                    "SPACE",
                    "DATA",
                    "PLACE",
                    "PEP",
                    "IOT",
                    "BIO",
                    "HEALTH",
                    "API",
                ],
                "integration_pattern": "multi_source_spatial_fusion",
                "study_id": self.study_id,
                "complexity_level": 5,
            },
        }

        with open(output_file, "w") as f:
            json.dump(full_results, f, indent=2, default=str)

        print(f"📁 Results saved to: {output_file.name}")

    # Helper methods for analysis components
    def _generate_h3_grid(self):
        """Generate H3 grid for area (simplified)."""
        return [f"h3_cell_{i}" for i in range(100)]  # 100 H3 cells

    def _cell_rng(self, cell: str) -> "np.random.Generator":
        """Deterministic per-cell RNG so every run of the study is identical."""
        return np.random.default_rng(zlib.crc32(cell.encode("utf-8")))

    def _extract_technical_cell_data(self, cell, technical):
        """Extract technical data for an H3 cell (deterministic simulation)."""
        rng = self._cell_rng(cell)
        return {
            "connectivity_score": float(rng.uniform(0.3, 0.95)),
            "infrastructure_quality": float(rng.uniform(0.4, 0.9)),
            "iot_sensor_density": int(rng.integers(1, 10)),
        }

    def _extract_social_cell_data(self, cell, social):
        """Extract social data for an H3 cell (deterministic simulation)."""
        rng = self._cell_rng(cell)
        return {
            "community_cohesion": float(rng.uniform(0.2, 0.9)),
            "social_vulnerability": float(rng.uniform(0.1, 0.8)),
            "organizational_density": int(rng.integers(0, 5)),
        }

    def _extract_environmental_cell_data(self, cell, environmental):
        """Extract environmental data for an H3 cell (deterministic simulation)."""
        rng = self._cell_rng(cell)
        return {
            "air_quality_index": float(rng.uniform(30, 90)),
            "green_space_coverage": float(rng.uniform(0.05, 0.4)),
            "noise_level": float(rng.uniform(45, 85)),
        }

    @staticmethod
    def _per_cell_metrics(spatial_data, domain, key):
        """Collect one numeric metric across all cells, ignoring missing data."""
        return [
            float(cell[domain][key])
            for cell in spatial_data
            if domain in cell and key in cell[domain]
        ]

    def _analyze_cross_domain_correlations(self, technical, social, environmental):
        """Analyze correlations between domains."""
        return {
            "connectivity_social_cohesion": 0.45,
            "infrastructure_environmental_quality": 0.38,
            "community_green_space": 0.52,
        }

    def _analyze_neighborhood_scale(self, spatial_data):
        return {"scale": "neighborhood", "cells_analyzed": len(spatial_data)}

    def _analyze_district_scale(self, spatial_data):
        return {"scale": "district", "cells_analyzed": len(spatial_data)}

    def _analyze_area_scale(self, spatial_data):
        return {"scale": "area", "cells_analyzed": len(spatial_data)}

    def _identify_technical_hotspots(self, spatial_data):
        """Cells in the lowest quartile of connectivity score."""
        scores = self._per_cell_metrics(
            spatial_data, "technical_metrics", "connectivity_score"
        )
        if not scores:
            return []
        cutoff = float(np.quantile(scores, 0.25))
        return [
            cell["h3_index"]
            for cell in spatial_data
            if cell["technical_metrics"]["connectivity_score"] <= cutoff
        ]

    def _identify_social_hotspots(self, spatial_data):
        """Cells in the highest quartile of social vulnerability."""
        values = self._per_cell_metrics(
            spatial_data, "social_metrics", "social_vulnerability"
        )
        if not values:
            return []
        cutoff = float(np.quantile(values, 0.75))
        return [
            cell["h3_index"]
            for cell in spatial_data
            if cell["social_metrics"]["social_vulnerability"] >= cutoff
        ]

    def _identify_environmental_hotspots(self, spatial_data):
        """Cells in the highest quartile of air quality index (worst air)."""
        values = self._per_cell_metrics(
            spatial_data, "environmental_metrics", "air_quality_index"
        )
        if not values:
            return []
        cutoff = float(np.quantile(values, 0.75))
        return [
            cell["h3_index"]
            for cell in spatial_data
            if cell["environmental_metrics"]["air_quality_index"] >= cutoff
        ]

    def _analyze_connectivity_access(self, spatial_data):
        scores = self._per_cell_metrics(
            spatial_data, "technical_metrics", "connectivity_score"
        )
        if not scores:
            return {"average_access_score": 0.0, "coverage_percentage": 0.0}
        coverage = float(np.mean(np.array(scores) >= 0.5))
        return {
            "average_access_score": round(float(np.mean(scores)), 3),
            "coverage_percentage": round(coverage, 3),
        }

    def _analyze_green_space_access(self, spatial_data):
        values = self._per_cell_metrics(
            spatial_data, "environmental_metrics", "green_space_coverage"
        )
        if not values:
            return {"average_access_score": 0.0, "coverage_percentage": 0.0}
        return {
            "average_access_score": round(float(np.mean(values)), 3),
            "coverage_percentage": round(float(np.mean(values)), 3),
        }

    def _analyze_service_access(self, spatial_data):
        density = self._per_cell_metrics(
            spatial_data, "social_metrics", "organizational_density"
        )
        if not density:
            return {"healthcare_access": 0.0, "education_access": 0.0}
        mean_density = float(np.mean(density))
        # Organizational density (0-5 per cell) is used as the service proxy.
        access = round(mean_density / 5.0, 3)
        return {"healthcare_access": access, "education_access": access}

    def _assess_spatial_equity(self, spatial_data):
        scores = self._per_cell_metrics(
            spatial_data, "technical_metrics", "connectivity_score"
        )
        if not scores:
            return {"equity_score": 0.0, "disparity_index": 1.0}
        arr = np.array(scores)
        disparity = float((arr.max() - arr.min()) / max(float(arr.mean()), 1e-9))
        disparity = min(disparity, 1.0)
        return {
            "equity_score": round(1.0 - disparity, 3),
            "disparity_index": round(disparity, 3),
        }

    def _analyze_digital_divide_impact(self, spatial_analysis):
        accessibility = spatial_analysis.get("accessibility", {}).get(
            "connectivity_access", {}
        )
        coverage = float(accessibility.get("coverage_percentage", 0.0))
        return {
            "divide_severity": round(1.0 - coverage, 3),
            "affected_population": round(1.0 - coverage, 3),
        }

    def _analyze_infrastructure_equity(self, spatial_analysis):
        equity = float(
            spatial_analysis.get("equity_metrics", {}).get("equity_score", 0.0)
        )
        return {"equity_score": equity, "improvement_potential": round(1.0 - equity, 3)}

    def _analyze_connectivity_engagement(self, spatial_analysis):
        accessibility = spatial_analysis.get("accessibility", {}).get(
            "connectivity_access", {}
        )
        score = float(accessibility.get("average_access_score", 0.0))
        cohesion_access = float(
            spatial_analysis.get("accessibility", {})
            .get("service_accessibility", {})
            .get("healthcare_access", 0.0)
        )
        return {
            "engagement_correlation": round((score + cohesion_access) / 2.0, 3),
            "participation_rate": round(score, 3),
        }

    def _analyze_community_stewardship(self, spatial_analysis):
        green = float(
            spatial_analysis.get("accessibility", {})
            .get("green_space_access", {})
            .get("average_access_score", 0.0)
        )
        return {
            "stewardship_score": round(0.5 + green / 2.0, 3),
            "environmental_awareness": round(0.5 + green / 2.0, 3),
        }

    def _analyze_environmental_justice(self, spatial_analysis):
        equity = spatial_analysis.get("equity_metrics", {})
        return {
            "justice_score": float(equity.get("equity_score", 0.0)),
            "vulnerable_groups": float(equity.get("disparity_index", 1.0)),
        }

    def _analyze_green_space_cohesion(self, spatial_analysis):
        green = float(
            spatial_analysis.get("accessibility", {})
            .get("green_space_access", {})
            .get("average_access_score", 0.0)
        )
        service = float(
            spatial_analysis.get("accessibility", {})
            .get("service_accessibility", {})
            .get("healthcare_access", 0.0)
        )
        return {
            "cohesion_correlation": round((green + service) / 2.0, 3),
            "social_benefits": round(green, 3),
        }

    def _analyze_smart_sustainability(self, spatial_analysis):
        equity = float(
            spatial_analysis.get("equity_metrics", {}).get("equity_score", 0.0)
        )
        return {
            "sustainability_score": round(equity, 3),
            "efficiency_gains": round(max(0.0, equity - 0.5), 3),
        }

    def _analyze_iot_monitoring_impact(self, spatial_analysis):
        accessibility = spatial_analysis.get("accessibility", {}).get(
            "connectivity_access", {}
        )
        score = float(accessibility.get("average_access_score", 0.0))
        return {
            "monitoring_effectiveness": round(score, 3),
            "data_quality": round(min(1.0, score + 0.1), 3),
        }

    def _analyze_infrastructure_resilience(self, spatial_analysis):
        equity = float(
            spatial_analysis.get("equity_metrics", {}).get("equity_score", 0.0)
        )
        return {
            "resilience_score": round(equity, 3),
            "vulnerability_reduction": round(max(0.0, equity - 0.4), 3),
        }

    def _calculate_composite_resilience(self, tech_social, social_env, tech_env):
        def _numeric_leaves(node, out):
            if isinstance(node, dict):
                for value in node.values():
                    _numeric_leaves(value, out)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                out.append(float(node))
            return out

        leaves = (
            _numeric_leaves(tech_social, [])
            + _numeric_leaves(social_env, [])
            + _numeric_leaves(tech_env, [])
        )
        return round(float(np.mean(leaves)), 3) if leaves else 0.0

    def _calculate_sustainability_index(self, spatial_analysis):
        return float(
            spatial_analysis.get("equity_metrics", {}).get("equity_score", 0.0)
        )

    def _calculate_quality_of_life(self, spatial_analysis):
        accessibility = spatial_analysis.get("accessibility", {})
        scores = [
            float(block.get("average_access_score", 0.0))
            for block in accessibility.values()
            if isinstance(block, dict) and "average_access_score" in block
        ]
        return round(float(np.mean(scores)), 3) if scores else 0.0

    def _estimate_resource_needs(self, recommendations):
        # Planning heuristic: each recommendation is budgeted and scheduled
        # proportionally to how many actions the study produced.
        count = max(1, len(recommendations))
        return {"estimated_cost": 50000 * count, "timeline_months": 4 * count}


def main():
    """Main function to run the area study."""
    print("🏛️ GEO-INFER Comprehensive Area Study Template")
    print("Multi-Disciplinary Integration: Technical + Social + Environmental")
    print("=" * 70)

    try:
        study = ComprehensiveAreaStudy()
        results = study.run_area_study()

        print("\n🎉 Area study completed successfully!")
        print(
            "This template demonstrates integrated area analysis for sustainable planning."
        )

        print("\n📊 View Results Options:")
        print("1. Console Results: python scripts/show_results.py")
        print("2. Interactive Dashboard: python scripts/launch_dashboard.py")
        print("   (requires: uv pip install streamlit pandas plotly)")
        return 0

    except Exception as e:
        print(f"\n❌ Area study failed: {e}")
        logging.exception("Detailed error information:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
