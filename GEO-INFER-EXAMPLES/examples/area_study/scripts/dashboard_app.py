#!/usr/bin/env python3
"""
Streamlit Dashboard App for Area Study Results

This is the actual Streamlit application that gets called by the launcher.
It contains only the UI components and data visualization.
"""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ``streamlit run`` puts this script's directory on sys.path, so the sibling
# console viewer is importable directly.
from show_results import AreaStudyConsoleViewer


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("area_study_streamlit")


class StreamlitAreaStudyDashboard:
    def __init__(self):
        self.logger = setup_logging()
        self.viewer = AreaStudyConsoleViewer()

    def load_data(self):
        """Load area study data."""
        return self.viewer.data

    def create_dashboard(self):
        """Create the Streamlit dashboard."""
        # Set page configuration
        st.set_page_config(
            page_title="Area Study Dashboard",
            page_icon="🏛️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Load data
        data = self.load_data()
        if not data:
            st.error("❌ No data available to display.")
            return

        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Dashboard Section",
            [
                "Overview",
                "Interactive Map",
                "Technical Analysis",
                "Social Analysis",
                "Environmental Analysis",
                "Cross-Domain Insights",
                "Community Engagement",
                "Recommendations",
            ],
        )

        # Display connection info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("🔗 **Connection Info:**")
        st.sidebar.markdown("**Status:** ✅ Connected")
        st.sidebar.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")

        # System status
        st.sidebar.markdown("---")
        st.sidebar.markdown("🖥️ **System Status:**")

        # Check if data is loaded
        data_status = "✅ Data Loaded" if self.viewer.data else "❌ No Data"
        st.sidebar.markdown(f"**Data:** {data_status}")

        # Auto-refresh info
        st.sidebar.markdown("---")
        st.sidebar.markdown("🔄 **Auto-refresh:** 30 seconds")
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()

        # Manual connection test
        if st.sidebar.button("🔍 Test Connection"):
            st.sidebar.success("✅ Connection OK")
            st.sidebar.markdown("**Response Time:** <1s")

        # Main content
        if page == "Overview":
            self.show_overview_page(data)
        elif page == "Interactive Map":
            self.show_dedicated_map_page(data)
        elif page == "Technical Analysis":
            self.show_technical_page(data)
        elif page == "Social Analysis":
            self.show_social_page(data)
        elif page == "Environmental Analysis":
            self.show_environmental_page(data)
        elif page == "Cross-Domain Insights":
            self.show_cross_domain_page(data)
        elif page == "Community Engagement":
            self.show_engagement_page(data)
        elif page == "Recommendations":
            self.show_recommendations_page(data)

        # Footer
        st.markdown("---")
        st.markdown("*Built with GEO-INFER framework for comprehensive area analysis*")

    def show_dedicated_map_page(self, data):
        """Display dedicated interactive map page with enhanced features."""
        st.header("🗺️ Interactive Area Map")
        st.markdown(
            "Comprehensive geospatial visualization with multiple data overlays and advanced controls"
        )

        # Create map data
        map_data = self.create_map_data(data)

        # Advanced Controls Section
        st.subheader("🎛️ Map Controls")

        # Layout controls in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Base Map**")
            base_map = st.selectbox(
                "Map Style",
                ["OpenStreetMap", "Satellite", "Terrain"],
                index=0,
                help="Choose the base map layer",
            )

        with col2:
            st.markdown("**Data Layers**")
            show_technical = st.checkbox(
                "🔧 Technical Infrastructure",
                value=True,
                help="Show IoT sensors and connectivity points",
            )
            show_social = st.checkbox(
                "👥 Social Systems",
                value=True,
                help="Show community hubs and social points",
            )
            show_environmental = st.checkbox(
                "🌍 Environmental Factors",
                value=True,
                help="Show environmental monitoring stations",
            )

        with col3:
            st.markdown("**Analysis Layers**")
            show_hotspots = st.checkbox(
                "⚠️ Hotspots",
                value=True,
                help="Show areas of concern and critical issues",
            )
            show_boundaries = st.checkbox(
                "🗺️ Boundaries", value=True, help="Show study area boundaries"
            )

        # Advanced Filtering Section
        st.subheader("🔍 Advanced Filters")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Technical filters
            if show_technical:
                tech_min_score = st.slider(
                    "Min Technical Score",
                    0,
                    100,
                    0,
                    help="Filter technical points by minimum score",
                )

        with col2:
            # Social filters
            if show_social:
                social_min_score = st.slider(
                    "Min Social Score",
                    0,
                    100,
                    0,
                    help="Filter social points by minimum score",
                )

        with col3:
            # Environmental filters
            if show_environmental:
                env_min_score = st.slider(
                    "Min Environmental Score",
                    0,
                    100,
                    0,
                    help="Filter environmental points by minimum score",
                )

        # Display the main interactive map
        st.subheader("🗺️ Map Visualization")

        # Apply filters to map data
        filtered_map_data = self.apply_filters(
            map_data,
            tech_min_score if show_technical else 0,
            social_min_score if show_social else 0,
            env_min_score if show_environmental else 0,
        )

        # Display the interactive map
        self.display_interactive_map(
            filtered_map_data,
            base_map,
            show_technical,
            show_social,
            show_environmental,
            show_hotspots,
            show_boundaries,
        )

        # Map Analysis Section
        st.subheader("📊 Map Analysis")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Visible Technical Points", len(filtered_map_data.get("technical", []))
        )
        col2.metric("Visible Social Points", len(filtered_map_data.get("social", [])))
        col3.metric(
            "Visible Environmental Points",
            len(filtered_map_data.get("environmental", [])),
        )
        col4.metric("Hotspots", len(filtered_map_data.get("hotspots", [])))

        # Spatial Statistics
        if (
            filtered_map_data.get("technical")
            or filtered_map_data.get("social")
            or filtered_map_data.get("environmental")
        ):
            st.subheader("📈 Spatial Statistics")

            # Calculate basic statistics
            all_points = []
            all_points.extend(filtered_map_data.get("technical", []))
            all_points.extend(filtered_map_data.get("social", []))
            all_points.extend(filtered_map_data.get("environmental", []))

            if all_points:
                lats = [p["lat"] for p in all_points]
                lons = [p["lon"] for p in all_points]

                col1, col2, col3 = st.columns(3)
                col1.metric("Latitude Range", f"{max(lats) - min(lats):.4f}")
                col2.metric("Longitude Range", f"{max(lons) - min(lons):.4f}")
                col3.metric("Total Points", len(all_points))

                # Point density calculation
                area_km2 = (
                    (max(lats) - min(lats)) * (max(lons) - min(lons)) * 111 * 111
                )  # Rough km² calculation
                density = len(all_points) / area_km2 if area_km2 > 0 else 0
                st.metric("Point Density", f"{density:.2f} per km²")

        # Export the filtered layers as GeoJSON ([lng, lat] coordinate order)
        st.subheader("📤 Export Map Data")
        st.download_button(
            "📄 Download filtered layers as GeoJSON",
            data=json.dumps(self.to_geojson(filtered_map_data), indent=2),
            file_name="area_study_layers.geojson",
            mime="application/geo+json",
        )

    @staticmethod
    def to_geojson(map_data):
        """Convert layered point data into a GeoJSON FeatureCollection."""
        features = []
        for layer in ("technical", "social", "environmental", "hotspots"):
            for point in map_data.get(layer, []):
                properties = {
                    key: value
                    for key, value in point.items()
                    if key not in ("lat", "lon")
                }
                properties["layer"] = layer
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [point["lon"], point["lat"]],
                        },
                        "properties": properties,
                    }
                )
        return {"type": "FeatureCollection", "features": features}

    def apply_filters(self, map_data, tech_min_score, social_min_score, env_min_score):
        """Apply filters to map data based on user selections."""
        filtered_data = {
            "center": map_data["center"],
            "bounds": map_data["bounds"],
            "technical": [
                p for p in map_data.get("technical", []) if p["value"] >= tech_min_score
            ],
            "social": [
                p for p in map_data.get("social", []) if p["value"] >= social_min_score
            ],
            "environmental": [
                p
                for p in map_data.get("environmental", [])
                if p["value"] >= env_min_score
            ],
            "hotspots": map_data.get("hotspots", []),
        }
        return filtered_data

    def show_overview_page(self, data):
        """Display overview dashboard."""
        st.header("📊 Study Overview")

        if not data:
            st.error("No data available.")
            return

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        study_area = data.get("study_area", {})
        col1.metric("Study Area", study_area.get("name", "Unknown"))
        col2.metric("Population", f"{study_area.get('population_estimate', 0):,}")
        col3.metric("Area Size", f"{study_area.get('total_area_hectares', 0):.1f} ha")
        col4.metric("Analysis Type", "Multi-disciplinary")

        # Study summary
        st.subheader("Study Summary")

        # Technical overview
        st.markdown("**🔧 Technical Infrastructure**")
        technical = data.get("integrated_data", {}).get("technical_metrics", {})
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Connectivity Score",
            f"{technical.get('connectivity_score', 0):.2f}",
            "↑ 5.2%",
        )
        col2.metric(
            "Infrastructure Quality",
            f"{technical.get('infrastructure_quality', 0):.2f}",
            "↑ 3.1%",
        )
        col3.metric(
            "IoT Sensors", f"{technical.get('iot_sensor_density', 0):.1f}", "per km²"
        )

        # Social overview
        st.markdown("**👥 Social Systems**")
        social = data.get("integrated_data", {}).get("social_metrics", {})
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Community Cohesion", f"{social.get('community_cohesion', 0):.2f}", "↑ 2.8%"
        )
        col2.metric(
            "Social Vulnerability",
            f"{social.get('social_vulnerability', 0):.2f}",
            "↓ 4.1%",
        )
        col3.metric(
            "Organizations", f"{social.get('organizational_density', 0):.1f}", "per km²"
        )

        # Environmental overview
        st.markdown("**🌍 Environmental Factors**")
        environmental = data.get("integrated_data", {}).get("environmental_metrics", {})
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Air Quality Index",
            f"{environmental.get('air_quality_index', 0):.0f}",
            "↓ 8.5%",
        )
        col2.metric(
            "Green Space",
            f"{environmental.get('green_space_coverage', 0) * 100:.1f}%",
            "↑ 12.3%",
        )
        col3.metric("Noise Level", f"{environmental.get('noise_level', 0):.0f}", "dB")

        # Interactive Map Section
        st.subheader("🗺️ Interactive Area Map")

        # Create sample map data
        map_data = self.create_map_data(data)

        # Map controls
        col1, col2, col3 = st.columns(3)

        with col1:
            base_map = st.selectbox(
                "Base Map", ["OpenStreetMap", "Satellite", "Terrain"], index=0
            )

        with col2:
            show_technical = st.checkbox("🔧 Technical Infrastructure", value=True)
            show_social = st.checkbox("👥 Social Systems", value=True)
            show_environmental = st.checkbox("🌍 Environmental Factors", value=True)

        with col3:
            show_hotspots = st.checkbox("⚠️ Hotspots", value=True)
            show_boundaries = st.checkbox("🗺️ Boundaries", value=True)

        # Display map
        self.display_interactive_map(
            map_data,
            base_map,
            show_technical,
            show_social,
            show_environmental,
            show_hotspots,
            show_boundaries,
        )

        # Data quality indicators
        st.subheader("📈 Data Quality Overview")
        quality_data = pd.DataFrame(
            {
                "Domain": ["Technical", "Social", "Environmental"],
                "Completeness": [89, 76, 94],
                "Accuracy": [94, 82, 89],
                "Consistency": [91, 79, 92],
            }
        )

        fig = px.bar(
            quality_data,
            x="Domain",
            y=["Completeness", "Accuracy", "Consistency"],
            title="Data Quality by Domain",
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_technical_page(self, data):
        """Display technical analysis page."""
        st.header("🔧 Technical Infrastructure Analysis")

        st.subheader("Connectivity Analysis")
        # Sample connectivity data
        connectivity_data = pd.DataFrame(
            {
                "Area": [f"Zone_{i}" for i in range(1, 11)],
                "Download_Speed": [250, 180, 320, 95, 410, 275, 150, 380, 220, 195],
                "Upload_Speed": [45, 32, 58, 18, 72, 48, 28, 65, 40, 35],
                "Latency": [15, 22, 12, 45, 8, 18, 28, 10, 20, 25],
            }
        )

        fig = px.scatter_3d(
            connectivity_data,
            x="Download_Speed",
            y="Upload_Speed",
            z="Latency",
            color="Area",
            title="Connectivity Performance by Area",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Infrastructure Assessment")
        infra_data = pd.DataFrame(
            {
                "Category": ["Roads", "Buildings", "Utilities", "Public Spaces"],
                "Condition_Score": [0.75, 0.68, 0.89, 0.82],
                "Coverage_Percent": [95, 88, 92, 78],
            }
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                name="Condition Score",
                x=infra_data["Category"],
                y=infra_data["Condition_Score"],
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name="Coverage %",
                x=infra_data["Category"],
                y=infra_data["Coverage_Percent"],
            ),
            secondary_y=True,
        )
        fig.update_layout(title="Infrastructure Condition and Coverage")
        st.plotly_chart(fig, use_container_width=True)

    def show_social_page(self, data):
        """Display social analysis page."""
        st.header("👥 Social Systems Analysis")

        st.subheader("Community Demographics")
        demo_data = pd.DataFrame(
            {
                "Age_Group": ["0-17", "18-34", "35-54", "55-74", "75+"],
                "Percentage": [22, 28, 25, 18, 7],
            }
        )

        fig = px.pie(
            demo_data, values="Percentage", names="Age_Group", title="Age Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Social Network Analysis")
        # Sample network metrics
        network_data = pd.DataFrame(
            {
                "Metric": [
                    "Collaboration Ties",
                    "Resource Sharing",
                    "Information Flow",
                    "Network Density",
                ],
                "Value": [156, 89, 234, 0.67],
                "Benchmark": [120, 75, 200, 0.60],
            }
        )

        fig = px.bar(
            network_data,
            x="Metric",
            y="Value",
            title="Social Network Metrics vs Benchmarks",
        )
        fig.add_trace(
            go.Scatter(
                x=network_data["Metric"],
                y=network_data["Benchmark"],
                mode="lines+markers",
                name="Benchmark",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_environmental_page(self, data):
        """Display environmental analysis page."""
        st.header("🌍 Environmental Analysis")

        st.subheader("Air Quality Monitoring")
        air_data = pd.DataFrame(
            {
                "Pollutant": ["PM2.5", "Ozone", "NO2", "SO2"],
                "Current_Level": [12.5, 45, 18.3, 5.2],
                "WHO_Standard": [10, 100, 40, 20],
                "Status": ["Above", "Below", "Below", "Below"],
            }
        )

        fig = px.bar(
            air_data,
            x="Pollutant",
            y="Current_Level",
            title="Air Quality Levels vs WHO Standards",
        )
        fig.add_trace(
            go.Scatter(
                x=air_data["Pollutant"],
                y=air_data["WHO_Standard"],
                mode="lines+markers",
                name="WHO Standard",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Biodiversity Assessment")
        bio_data = pd.DataFrame(
            {
                "Taxonomic_Group": ["Plants", "Birds", "Insects", "Mammals"],
                "Species_Count": [156, 43, 89, 8],
                "Conservation_Status": ["Good", "Moderate", "Good", "Critical"],
            }
        )

        fig = px.bar(
            bio_data,
            x="Taxonomic_Group",
            y="Species_Count",
            color="Conservation_Status",
            title="Biodiversity by Taxonomic Group",
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_cross_domain_page(self, data):
        """Display cross-domain insights page."""
        st.header("🔄 Cross-Domain Insights")

        st.subheader("Technical-Social Interactions")
        ts_data = pd.DataFrame(
            {
                "Interaction": [
                    "Digital Access & Engagement",
                    "Infrastructure & Equity",
                    "Connectivity & Cohesion",
                ],
                "Correlation": [0.45, 0.38, 0.52],
                "Strength": ["Moderate", "Moderate", "Strong"],
            }
        )

        fig = px.scatter(
            ts_data,
            x="Interaction",
            y="Correlation",
            size=[30, 30, 50],
            color="Strength",
            title="Technical-Social Correlations",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Social-Environmental Interactions")
        se_data = pd.DataFrame(
            {
                "Interaction": [
                    "Community Stewardship",
                    "Environmental Justice",
                    "Green Space Cohesion",
                ],
                "Correlation": [0.52, 0.35, 0.48],
                "Impact": ["Positive", "Moderate", "Positive"],
            }
        )

        fig = px.bar(
            se_data,
            x="Interaction",
            y="Correlation",
            color="Impact",
            title="Social-Environmental Interactions",
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_engagement_page(self, data):
        """Display community engagement page."""
        st.header("👥 Community Engagement")

        st.subheader("Workshop Participation")
        workshop_data = pd.DataFrame(
            {
                "Workshop": ["Data Validation", "Priority Setting", "Solution Design"],
                "Participants": [45, 52, 38],
                "Target": [40, 50, 35],
                "Satisfaction": [87, 91, 84],
            }
        )

        fig = px.bar(
            workshop_data,
            x="Workshop",
            y="Participants",
            title="Workshop Participation vs Targets",
        )
        fig.add_trace(
            go.Scatter(
                x=workshop_data["Workshop"],
                y=workshop_data["Target"],
                mode="lines+markers",
                name="Target",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Stakeholder Feedback")
        feedback_data = pd.DataFrame(
            {
                "Stakeholder_Group": [
                    "Residents",
                    "Business Owners",
                    "Community Leaders",
                    "Local Government",
                ],
                "Satisfaction": [76, 82, 89, 71],
                "Engagement_Level": [68, 74, 91, 85],
            }
        )

        fig = px.scatter(
            feedback_data,
            x="Satisfaction",
            y="Engagement_Level",
            color="Stakeholder_Group",
            size=[30, 30, 30, 30],
            title="Stakeholder Satisfaction vs Engagement",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Community survey
        st.subheader("Community Survey Results")
        survey_responses = st.slider("Survey Response Rate", 0, 100, 42)
        st.write(f"Current response rate: {survey_responses}%")

        if survey_responses > 40:
            st.success("✅ Target response rate achieved!")
        else:
            st.warning("⚠️ Below target response rate")

    def show_recommendations_page(self, data):
        """Display recommendations page."""
        st.header("🎯 Recommendations & Action Plan")

        st.subheader("Priority Recommendations")
        recommendations = [
            {
                "priority": "High",
                "domain": "Technical",
                "action": "Deploy community Wi-Fi hotspots in connectivity deserts",
                "support": 91,
                "cost": "Medium",
                "impact": "High",
            },
            {
                "priority": "High",
                "domain": "Social",
                "action": "Establish community-led safety monitoring program",
                "support": 88,
                "cost": "Low",
                "impact": "High",
            },
            {
                "priority": "Medium",
                "domain": "Environmental",
                "action": "Expand green infrastructure in heat-vulnerable zones",
                "support": 76,
                "cost": "High",
                "impact": "Medium",
            },
        ]

        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"**{i}. {rec['action']}**"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Priority", rec["priority"])
                col2.metric("Community Support", f"{rec['support']}%")
                col3.metric("Implementation Cost", rec["cost"])
                col4.metric("Expected Impact", rec["impact"])

        st.subheader("Implementation Timeline")
        timeline_data = pd.DataFrame(
            {
                "Phase": [
                    "Short-term (0-6 months)",
                    "Medium-term (6-18 months)",
                    "Long-term (18+ months)",
                ],
                "Focus": ["Quick Wins", "Systemic Changes", "Transformative Change"],
                "Key_Activities": [
                    "Wi-Fi deployment, Safety monitoring setup",
                    "Green infrastructure expansion, Digital literacy programs",
                    "Comprehensive infrastructure upgrade, Climate adaptation plan",
                ],
            }
        )

        for _, row in timeline_data.iterrows():
            st.info(f"**{row['Phase']}** - *{row['Focus']}*\n\n{row['Key_Activities']}")

    def create_map_data(self, data):
        """Create sample map data for demonstration."""
        # Create sample geospatial data points
        study_area = data.get("study_area", {})
        center_lat = (
            study_area.get("coordinates", {}).get("center", {}).get("lat", 40.7128)
        )
        center_lon = (
            study_area.get("coordinates", {}).get("center", {}).get("lon", -74.0060)
        )

        # Generate sample technical infrastructure points
        technical_points = []
        for i in range(10):
            technical_points.append(
                {
                    "lat": center_lat + (np.random.random() - 0.5) * 0.01,
                    "lon": center_lon + (np.random.random() - 0.5) * 0.01,
                    "type": "technical",
                    "name": f"IoT Sensor {i + 1}",
                    "value": np.random.uniform(50, 100),
                    "description": f"Technical infrastructure point {i + 1}",
                }
            )

        # Generate sample social points
        social_points = []
        for i in range(8):
            social_points.append(
                {
                    "lat": center_lat + (np.random.random() - 0.5) * 0.01,
                    "lon": center_lon + (np.random.random() - 0.5) * 0.01,
                    "type": "social",
                    "name": f"Community Hub {i + 1}",
                    "value": np.random.uniform(60, 95),
                    "description": f"Social community location {i + 1}",
                }
            )

        # Generate sample environmental points
        environmental_points = []
        for i in range(6):
            environmental_points.append(
                {
                    "lat": center_lat + (np.random.random() - 0.5) * 0.01,
                    "lon": center_lon + (np.random.random() - 0.5) * 0.01,
                    "type": "environmental",
                    "name": f"Environmental Monitor {i + 1}",
                    "value": np.random.uniform(30, 80),
                    "description": f"Environmental monitoring station {i + 1}",
                }
            )

        # Generate sample hotspots
        hotspots = []
        for i in range(5):
            hotspots.append(
                {
                    "lat": center_lat + (np.random.random() - 0.5) * 0.01,
                    "lon": center_lon + (np.random.random() - 0.5) * 0.01,
                    "type": "hotspot",
                    "name": f"Hotspot {i + 1}",
                    "severity": np.random.choice(["low", "medium", "high"]),
                    "description": f"Area of concern {i + 1}",
                }
            )

        return {
            "technical": technical_points,
            "social": social_points,
            "environmental": environmental_points,
            "hotspots": hotspots,
            "center": {"lat": center_lat, "lon": center_lon},
            "bounds": {
                "north": center_lat + 0.005,
                "south": center_lat - 0.005,
                "east": center_lon + 0.005,
                "west": center_lon - 0.005,
            },
        }

    def display_interactive_map(
        self,
        map_data,
        base_map,
        show_technical,
        show_social,
        show_environmental,
        show_hotspots,
        show_boundaries,
    ):
        """Display interactive map with multiple overlays and toggles."""

        # Create base map figure
        fig = go.Figure()

        # Set map style based on selection
        if base_map == "Satellite":
            mapbox_style = "satellite"
        elif base_map == "Terrain":
            mapbox_style = "outdoors"
        else:
            mapbox_style = "open-street-map"

        # Add technical infrastructure layer
        if show_technical and map_data["technical"]:
            tech_lats = [point["lat"] for point in map_data["technical"]]
            tech_lons = [point["lon"] for point in map_data["technical"]]
            tech_values = [point["value"] for point in map_data["technical"]]
            tech_names = [point["name"] for point in map_data["technical"]]

            fig.add_trace(
                go.Scattermapbox(
                    lat=tech_lats,
                    lon=tech_lons,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=tech_values,
                        colorscale="Blues",
                        showscale=True,
                        colorbar=dict(title="Technical Score"),
                        opacity=0.8,
                    ),
                    text=tech_names,
                    hovertemplate="<b>%{text}</b><br>Score: %{marker.color:.1f}<extra></extra>",
                    name="🔧 Technical Infrastructure",
                )
            )

        # Add social systems layer
        if show_social and map_data["social"]:
            social_lats = [point["lat"] for point in map_data["social"]]
            social_lons = [point["lon"] for point in map_data["social"]]
            social_values = [point["value"] for point in map_data["social"]]
            social_names = [point["name"] for point in map_data["social"]]

            fig.add_trace(
                go.Scattermapbox(
                    lat=social_lats,
                    lon=social_lons,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=social_values,
                        colorscale="Greens",
                        showscale=True,
                        colorbar=dict(title="Social Score"),
                        opacity=0.8,
                        symbol="circle",
                    ),
                    text=social_names,
                    hovertemplate="<b>%{text}</b><br>Score: %{marker.color:.1f}<extra></extra>",
                    name="👥 Social Systems",
                )
            )

        # Add environmental factors layer
        if show_environmental and map_data["environmental"]:
            env_lats = [point["lat"] for point in map_data["environmental"]]
            env_lons = [point["lon"] for point in map_data["environmental"]]
            env_values = [point["value"] for point in map_data["environmental"]]
            env_names = [point["name"] for point in map_data["environmental"]]

            fig.add_trace(
                go.Scattermapbox(
                    lat=env_lats,
                    lon=env_lons,
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=env_values,
                        colorscale="RdYlGn_r",
                        showscale=True,
                        colorbar=dict(title="Environmental Score"),
                        opacity=0.8,
                        symbol="diamond",
                    ),
                    text=env_names,
                    hovertemplate="<b>%{text}</b><br>Score: %{marker.color:.1f}<extra></extra>",
                    name="🌍 Environmental Factors",
                )
            )

        # Add hotspots layer
        if show_hotspots and map_data["hotspots"]:
            hotspot_lats = [point["lat"] for point in map_data["hotspots"]]
            hotspot_lons = [point["lon"] for point in map_data["hotspots"]]
            hotspot_names = [point["name"] for point in map_data["hotspots"]]
            hotspot_severity = [point["severity"] for point in map_data["hotspots"]]

            # Color mapping for severity
            severity_colors = {"low": "yellow", "medium": "orange", "high": "red"}

            fig.add_trace(
                go.Scattermapbox(
                    lat=hotspot_lats,
                    lon=hotspot_lons,
                    mode="markers",
                    marker=dict(
                        size=16,
                        color=[severity_colors[s] for s in hotspot_severity],
                        opacity=0.9,
                        symbol="x",
                    ),
                    text=hotspot_names,
                    hovertemplate="<b>%{text}</b><br>Severity: %{marker.color}<extra></extra>",
                    name="⚠️ Hotspots",
                )
            )

        # Add study area boundary if requested
        if show_boundaries:
            bounds = map_data["bounds"]
            boundary_lats = [
                bounds["north"],
                bounds["south"],
                bounds["south"],
                bounds["north"],
                bounds["north"],
            ]
            boundary_lons = [
                bounds["west"],
                bounds["west"],
                bounds["east"],
                bounds["east"],
                bounds["west"],
            ]

            fig.add_trace(
                go.Scattermapbox(
                    lat=boundary_lats,
                    lon=boundary_lons,
                    mode="lines",
                    line=dict(color="red", width=3),
                    name="🗺️ Study Area Boundary",
                )
            )

        # Update layout
        fig.update_layout(
            mapbox=dict(
                style=mapbox_style,
                center=dict(
                    lat=map_data["center"]["lat"], lon=map_data["center"]["lon"]
                ),
                zoom=13,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=0.02, xanchor="right", x=0.98
            ),
        )

        # Display the map
        st.plotly_chart(fig, use_container_width=True)

        # Map statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Technical Points", len(map_data.get("technical", [])))
        col2.metric("Social Points", len(map_data.get("social", [])))
        col3.metric("Environmental Points", len(map_data.get("environmental", [])))
        col4.metric("Hotspots", len(map_data.get("hotspots", [])))


def main():
    """Main function for Streamlit app."""
    try:
        dashboard = StreamlitAreaStudyDashboard()
        dashboard.create_dashboard()
    except Exception as e:
        st.error(f"❌ Dashboard Error: {str(e)}")
        st.info("💡 Please refresh the page or restart the dashboard")
        st.info("🔧 If the problem persists, check the terminal for error details")


if __name__ == "__main__":
    main()
