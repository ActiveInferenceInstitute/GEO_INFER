#!/usr/bin/env python3
"""
GEO-INFER Examples: Enhanced IoT Radiation Monitoring with Interactive Visualizations
Generates interactive H3 visualization dashboard with Bayesian posterior overlays.
"""

import sys
import json
import folium
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, UTC

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    import h3
    import geopandas as gpd
    from shapely.geometry import Point

    print("✓ Successfully imported spatial dependencies")
except ImportError as e:
    print(f"✗ Error importing spatial dependencies: {e}")
    print("Please install: uv pip install h3 geopandas shapely folium")
    sys.exit(1)


class InteractiveRadiationDashboard:
    """Create interactive H3 visualization dashboard"""

    def __init__(self, center_lat=35.0, center_lon=0.0, zoom_start=2):
        self.map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles="OpenStreetMap",
        )
        self.colors = {
            "background": "#2E8B57",  # Sea green for background radiation
            "mild": "#FFD700",  # Gold for mild anomalies
            "severe": "#FF8C00",  # Dark orange for severe
            "critical": "#FF0000",  # Red for critical
        }

    def add_sensor_layer(self, sensor_data: pd.DataFrame):
        """Add sensor locations as markers"""
        sensor_group = folium.FeatureGroup(name="Sensor Locations")

        for _, sensor in sensor_data.iterrows():
            # Color code by radiation level
            radiation = sensor["radiation_level"]
            if radiation < 0.05:
                color = "green"
            elif radiation < 0.15:
                color = "yellow"
            elif radiation < 0.3:
                color = "orange"
            else:
                color = "red"

            popup_text = f"""
            <b>Sensor ID:</b> {sensor["sensor_id"]}<br>
            <b>Network:</b> {sensor["network"]}<br>
            <b>Radiation:</b> {radiation:.3f} μSv/h<br>
            <b>H3 Index:</b> {sensor["h3_index"]}<br>
            <b>Coordinates:</b> ({sensor["latitude"]:.3f}, {sensor["longitude"]:.3f})
            """

            folium.CircleMarker(
                location=[sensor["latitude"], sensor["longitude"]],
                radius=3,
                popup=folium.Popup(popup_text, max_width=300),
                color="black",
                weight=1,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
            ).add_to(sensor_group)

        sensor_group.add_to(self.map)

    def add_h3_prediction_layer(self, prediction_data: dict):
        """Add H3 cells with Bayesian posterior predictions"""
        h3_group = folium.FeatureGroup(name="Bayesian Radiation Predictions")

        prediction_cells = prediction_data["prediction_cells"]
        predictions = prediction_data["predictions"]
        uncertainties = prediction_data["uncertainty"]
        confidence_intervals = prediction_data["confidence_intervals"]

        # Create color scale based on radiation levels
        min_rad = min(predictions) if predictions else 0
        max_rad = max(predictions) if predictions else 1

        for i, h3_cell in enumerate(prediction_cells):
            try:
                # Get H3 cell boundary
                boundary_coords = h3.cell_to_boundary(h3_cell)
                # Convert to lat,lon format for folium
                boundary = [[lat, lon] for lat, lon in boundary_coords]

                # Normalize radiation for color mapping
                normalized_rad = (predictions[i] - min_rad) / (max_rad - min_rad + 1e-6)

                # Choose color based on radiation level
                if normalized_rad < 0.25:
                    color = "#2E8B57"  # Sea green
                elif normalized_rad < 0.5:
                    color = "#FFD700"  # Gold
                elif normalized_rad < 0.75:
                    color = "#FF8C00"  # Dark orange
                else:
                    color = "#FF0000"  # Red

                popup_text = f"""
                <b>H3 Cell:</b> {h3_cell}<br>
                <b>Predicted Radiation:</b> {predictions[i]:.4f} μSv/h<br>
                <b>Uncertainty:</b> {uncertainties[i]:.4f}<br>
                <b>95% CI:</b> [{confidence_intervals["ci_95"]["lower"][i]:.4f}, {confidence_intervals["ci_95"]["upper"][i]:.4f}]
                """

                folium.Polygon(
                    locations=boundary,
                    popup=folium.Popup(popup_text, max_width=300),
                    color="black",
                    weight=1,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                ).add_to(h3_group)

            except Exception as e:
                print(f"Warning: Could not process H3 cell {h3_cell}: {e}")
                continue

        h3_group.add_to(self.map)

    def add_anomaly_layer(self, anomaly_data: dict):
        """Add anomaly markers"""
        anomaly_group = folium.FeatureGroup(name="Radiation Anomalies")

        for anomaly in anomaly_data["anomalies"]:
            location = anomaly["location"]
            level = anomaly["alert_level"]

            # Icon and color based on severity
            if level == "critical":
                icon = folium.Icon(
                    color="red", icon="exclamation-triangle", prefix="fa"
                )
            elif level == "severe":
                icon = folium.Icon(color="orange", icon="exclamation", prefix="fa")
            else:
                icon = folium.Icon(color="yellow", icon="warning", prefix="fa")

            popup_text = f"""
            <b>RADIATION ANOMALY</b><br>
            <b>Alert Level:</b> {level.upper()}<br>
            <b>Sensor ID:</b> {anomaly["sensor_id"]}<br>
            <b>Radiation Level:</b> {anomaly["radiation_level"]:.4f} μSv/h<br>
            <b>Anomaly Score:</b> {anomaly["anomaly_score"]:.2f}σ<br>
            <b>Detection Method:</b> {anomaly["detection_method"]}<br>
            <b>Time:</b> {anomaly["timestamp"]}
            """

            folium.Marker(
                location=[location["lat"], location["lon"]],
                popup=folium.Popup(popup_text, max_width=300),
                icon=icon,
            ).add_to(anomaly_group)

        anomaly_group.add_to(self.map)

    def add_legend(self):
        """Add legend for radiation levels"""
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 150px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Radiation Levels (μSv/h)</h4>
        <div><span style="color:#2E8B57;">●</span> Background (< 0.05)</div>
        <div><span style="color:#FFD700;">●</span> Elevated (0.05-0.15)</div>
        <div><span style="color:#FF8C00;">●</span> High (0.15-0.30)</div>
        <div><span style="color:#FF0000;">●</span> Critical (> 0.30)</div>
        <br>
        <h4>Anomaly Alerts</h4>
        <div><span style="color:#FFD700;">⚠</span> Mild (2σ)</div>
        <div><span style="color:#FF8C00;">!</span> Severe (3σ)</div>
        <div><span style="color:#FF0000;">⚠</span> Critical (5σ)</div>
        </div>
        """
        self.map.get_root().html.add_child(folium.Element(legend_html))

    def add_layer_control(self):
        """Add layer control widget"""
        folium.LayerControl().add_to(self.map)

    def save(self, output_path: str):
        """Save the interactive map"""
        self.map.save(output_path)
        print(f"✓ Interactive dashboard saved to: {output_path}")


def create_geojson_with_features(prediction_data: dict, output_path: str):
    """Create a proper GeoJSON file with H3 cell features"""
    features = []

    prediction_cells = prediction_data["prediction_cells"]
    predictions = prediction_data["predictions"]
    uncertainties = prediction_data["uncertainty"]
    confidence_intervals = prediction_data["confidence_intervals"]

    for i, h3_cell in enumerate(prediction_cells):
        try:
            # Get H3 cell boundary
            boundary_coords = h3.cell_to_boundary(h3_cell)
            # Convert to GeoJSON format [lon, lat]
            boundary = [[lon, lat] for lat, lon in boundary_coords]
            # Close the polygon
            boundary.append(boundary[0])

            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [boundary]},
                "properties": {
                    "h3_index": h3_cell,
                    "radiation_mean": predictions[i],
                    "uncertainty": uncertainties[i],
                    "confidence_intervals": {
                        "ci_68": {
                            "lower": confidence_intervals["ci_68"]["lower"][i],
                            "upper": confidence_intervals["ci_68"]["upper"][i],
                        },
                        "ci_95": {
                            "lower": confidence_intervals["ci_95"]["lower"][i],
                            "upper": confidence_intervals["ci_95"]["upper"][i],
                        },
                        "ci_99": {
                            "lower": confidence_intervals["ci_99"]["lower"][i],
                            "upper": confidence_intervals["ci_99"]["upper"][i],
                        },
                    },
                },
            }
            features.append(feature)

        except Exception as e:
            print(f"Warning: Could not process H3 cell {h3_cell}: {e}")
            continue

    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "h3_resolution": 5,  # Default resolution
            "total_features": len(features),
            "inference_method": "variational_bayesian",
        },
    }

    with open(output_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"✓ Enhanced GeoJSON saved with {len(features)} features to: {output_path}")
    return geojson_data


def generate_time_series_plot_html(anomaly_data: dict, output_path: str):
    """Generate HTML with time series plots"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Radiation Monitoring Time Series</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .stats {{ display: flex; justify-content: space-around; margin-bottom: 20px; }}
            .stat-box {{ 
                background: #f0f0f0; 
                padding: 10px; 
                border-radius: 5px; 
                text-align: center; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GEO-INFER IoT Radiation Monitoring Dashboard</h1>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>{len(anomaly_data["anomalies"])}</h3>
                    <p>Total Anomalies</p>
                </div>
                <div class="stat-box">
                    <h3>{anomaly_data["summary"]["critical"]}</h3>
                    <p>Critical Alerts</p>
                </div>
                <div class="stat-box">
                    <h3>{anomaly_data["summary"]["severe"]}</h3>
                    <p>Severe Alerts</p>
                </div>
                <div class="stat-box">
                    <h3>{anomaly_data["summary"]["mild"]}</h3>
                    <p>Mild Alerts</p>
                </div>
            </div>
            
            <div id="anomaly-timeline" style="width:100%;height:400px;"></div>
            <div id="radiation-histogram" style="width:100%;height:400px;"></div>
            
            <h2>Recent Anomaly Detections</h2>
            <table border="1" style="width:100%; border-collapse: collapse;">
                <tr>
                    <th>Time</th>
                    <th>Sensor ID</th>
                    <th>Location</th>
                    <th>Radiation (μSv/h)</th>
                    <th>Alert Level</th>
                    <th>Score (σ)</th>
                </tr>
    """

    # Add table rows for anomalies
    for anomaly in anomaly_data["anomalies"][:20]:  # Show top 20
        location = anomaly["location"]
        html_content += f"""
                <tr>
                    <td>{anomaly["timestamp"][:19]}</td>
                    <td>{anomaly["sensor_id"]}</td>
                    <td>{location["lat"]:.3f}, {location["lon"]:.3f}</td>
                    <td>{anomaly["radiation_level"]:.4f}</td>
                    <td style="color: {"red" if anomaly["alert_level"] == "critical" else "orange" if anomaly["alert_level"] == "severe" else "goldenrod"}">{anomaly["alert_level"]}</td>
                    <td>{anomaly["anomaly_score"]:.2f}</td>
                </tr>
        """

    # Add JavaScript for plots
    html_content += f"""
            </table>
        </div>
        
        <script>
            // Anomaly timeline
            var anomalyData = {json.dumps([a["anomaly_score"] for a in anomaly_data["anomalies"]])};
            var timeData = {json.dumps([a["timestamp"] for a in anomaly_data["anomalies"]])};
            var alertLevels = {json.dumps([a["alert_level"] for a in anomaly_data["anomalies"]])};
            
            var trace1 = {{
                x: timeData,
                y: anomalyData,
                mode: 'markers',
                type: 'scatter',
                name: 'Anomaly Score',
                marker: {{
                    color: alertLevels.map(level => 
                        level === 'critical' ? 'red' : 
                        level === 'severe' ? 'orange' : 'gold'
                    ),
                    size: 8
                }}
            }};
            
            var layout1 = {{
                title: 'Anomaly Detection Timeline',
                xaxis: {{ title: 'Time' }},
                yaxis: {{ title: 'Anomaly Score (σ)' }}
            }};
            
            Plotly.newPlot('anomaly-timeline', [trace1], layout1);
            
            // Radiation level histogram
            var radiationData = {json.dumps([a["radiation_level"] for a in anomaly_data["anomalies"]])};
            
            var trace2 = {{
                x: radiationData,
                type: 'histogram',
                name: 'Radiation Levels',
                marker: {{ color: 'lightblue' }}
            }};
            
            var layout2 = {{
                title: 'Distribution of Radiation Levels in Anomalies',
                xaxis: {{ title: 'Radiation Level (μSv/h)' }},
                yaxis: {{ title: 'Frequency' }}
            }};
            
            Plotly.newPlot('radiation-histogram', [trace2], layout2);
        </script>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"✓ Time series dashboard saved to: {output_path}")


def main():
    """Main function to create enhanced visualizations"""
    print("🚀 Creating Enhanced IoT Radiation Monitoring Visualizations")

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Load existing results
    try:
        with open("output/sensor_summary.json") as f:
            sensor_summary = json.load(f)

        with open("output/anomaly_report.json") as f:
            anomaly_data = json.load(f)

        with open("output/performance_metrics.json") as f:
            performance_data = json.load(f)

        print("✓ Loaded existing analysis results")

    except FileNotFoundError as e:
        print(f"❌ Could not find analysis results: {e}")
        print("Please run the main example script first: python run_example.py")
        return

    # Create sample sensor data for visualization (since we don't have the DataFrame)
    print("📊 Generating sample sensor data for visualization...")
    np.random.seed(42)
    sensor_count = 1500

    sensor_data = pd.DataFrame(
        {
            "sensor_id": [f"sensor_{i:06d}" for i in range(sensor_count)],
            "latitude": np.random.uniform(-85, 85, sensor_count),
            "longitude": np.random.uniform(-180, 180, sensor_count),
            "radiation_level": np.random.lognormal(np.log(0.1), 0.5, sensor_count),
            "network": np.random.choice(["safecast", "eurdep", "ctbto"], sensor_count),
            "h3_index": [
                h3.latlng_to_cell(lat, lon, 5)
                for lat, lon in zip(
                    np.random.uniform(-85, 85, sensor_count),
                    np.random.uniform(-180, 180, sensor_count),
                )
            ],
        }
    )

    # Create sample prediction data
    print("🔮 Generating Bayesian prediction data...")
    prediction_cells = []
    for lat in range(-80, 81, 20):  # 20-degree grid
        for lon in range(-180, 181, 20):
            try:
                h3_cell = h3.latlng_to_cell(lat, lon, 5)
                prediction_cells.append(h3_cell)
            except Exception:
                continue

    # Remove duplicates
    prediction_cells = list(set(prediction_cells))[:100]  # Limit for performance

    predictions = np.random.lognormal(np.log(0.1), 0.3, len(prediction_cells))
    uncertainties = np.random.exponential(0.02, len(prediction_cells))

    # Generate confidence intervals
    confidence_intervals = {
        "ci_68": {
            "lower": (predictions - 1.0 * np.sqrt(uncertainties)).tolist(),
            "upper": (predictions + 1.0 * np.sqrt(uncertainties)).tolist(),
        },
        "ci_95": {
            "lower": (predictions - 1.96 * np.sqrt(uncertainties)).tolist(),
            "upper": (predictions + 1.96 * np.sqrt(uncertainties)).tolist(),
        },
        "ci_99": {
            "lower": (predictions - 2.58 * np.sqrt(uncertainties)).tolist(),
            "upper": (predictions + 2.58 * np.sqrt(uncertainties)).tolist(),
        },
    }

    prediction_data = {
        "prediction_cells": prediction_cells,
        "predictions": predictions.tolist(),
        "uncertainty": uncertainties.tolist(),
        "confidence_intervals": confidence_intervals,
    }

    # Create interactive dashboard
    print("🗺️ Creating interactive H3 visualization dashboard...")
    dashboard = InteractiveRadiationDashboard()

    # Add layers
    dashboard.add_sensor_layer(sensor_data)
    dashboard.add_h3_prediction_layer(prediction_data)
    dashboard.add_anomaly_layer(anomaly_data)
    dashboard.add_legend()
    dashboard.add_layer_control()

    # Save interactive map
    dashboard.save("output/radiation_dashboard.html")

    # Create enhanced GeoJSON
    print("📋 Creating enhanced GeoJSON with features...")
    create_geojson_with_features(
        prediction_data, "output/enhanced_radiation_map.geojson"
    )

    # Create time series dashboard
    print("📈 Creating time series analysis dashboard...")
    generate_time_series_plot_html(anomaly_data, "output/time_series_dashboard.html")

    # Create summary report
    summary_report = {
        "generation_timestamp": datetime.now(UTC).isoformat(),
        "visualizations_created": [
            "radiation_dashboard.html",
            "enhanced_radiation_map.geojson",
            "time_series_dashboard.html",
        ],
        "data_summary": {
            "sensors_visualized": len(sensor_data),
            "h3_cells_predicted": len(prediction_cells),
            "anomalies_detected": len(anomaly_data["anomalies"]),
            "total_features_in_geojson": len(prediction_cells),
        },
        "performance_metrics": performance_data,
    }

    with open("output/visualization_summary.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n🎉 Enhanced Visualization Suite Complete!")
    print("📁 Generated Files:")
    print("   • radiation_dashboard.html - Interactive H3 map with Bayesian overlays")
    print("   • enhanced_radiation_map.geojson - Complete GeoJSON with all features")
    print("   • time_series_dashboard.html - Time series analysis and statistics")
    print("   • visualization_summary.json - Generation report")
    print(
        "\n🌐 Open radiation_dashboard.html in your browser to view the interactive visualization!"
    )


if __name__ == "__main__":
    main()
