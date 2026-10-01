"""
IoT Visualization Utilities

This module provides visualization utilities for IoT sensor data, spatial
interpolation results, and real-time monitoring displays.
"""

import logging
import math
from typing import Any
from datetime import datetime
from pathlib import Path
import numpy as np
import h3

# Optional imports for enhanced visualization
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    import matplotlib.pyplot as plt

    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

logger = logging.getLogger(__name__)


def _coordinate(latitude: Any, longitude: Any, name: str) -> tuple[float, float]:
    """Validate and normalize a latitude/longitude pair."""
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} coordinates must be numeric") from exc
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError(f"{name} coordinates must be finite")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError(f"{name} coordinates are outside geographic bounds")
    return latitude, longitude


def _ensure_parent(output_file: str) -> None:
    """Create a nested output parent when a path includes one."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)


class IoTVisualization:
    """
    Visualization utilities for IoT sensor networks and spatial data.

    Provides methods for:
    - Interactive web maps with sensor locations and measurements
    - Real-time data visualization with WebSocket updates
    - Spatial interpolation surface plots
    - Time series visualization for sensor data
    - Network health and status dashboards
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.maps_cache: dict[str, Any] = {}
        self.plot_cache: dict[str, Any] = {}

        # Default visualization parameters
        self.default_params: dict[str, Any] = {
            "map_center": [40.7128, -74.0060],  # NYC coordinates
            "map_zoom": 10,
            "color_scheme": "viridis",
            "marker_size": 5,
            "heatmap_radius": 15,
            "heatmap_blur": 10,
            "animation_duration": 1000,  # milliseconds
        }

        if not HAS_VISUALIZATION:
            logger.warning(
                "Visualization libraries not available - map generation disabled"
            )

        logger.info("IoTVisualization initialized")

    def create_sensor_map(
        self,
        sensors: list[dict],
        measurements: list[dict] | None = None,
        output_file: str = "sensor_map.html",
    ) -> dict:
        """
        Create an interactive map showing sensor locations and measurements.

        Args:
            sensors: List of sensor information
            measurements: Optional recent measurements for coloring
            output_file: Output HTML file path

        Returns:
            Dictionary with map metadata and generation status
        """
        if not HAS_VISUALIZATION:
            return {"error": "Visualization libraries not available"}

        try:
            if not isinstance(sensors, list):
                raise ValueError("sensors must be a list of mappings")
            valid_sensors = []
            for index, sensor in enumerate(sensors):
                if not isinstance(sensor, dict):
                    raise ValueError(f"sensors[{index}] must be a mapping")
                if sensor.get("latitude") is None or sensor.get("longitude") is None:
                    continue
                _coordinate(
                    sensor["latitude"], sensor["longitude"], f"sensors[{index}]"
                )
                valid_sensors.append(sensor)
            sensors = valid_sensors
            # Calculate map center and bounds
            if sensors:
                latitudes = [s.get("latitude", 0) for s in sensors if "latitude" in s]
                longitudes = [
                    s.get("longitude", 0) for s in sensors if "longitude" in s
                ]

                if latitudes and longitudes:
                    center_lat = np.mean(latitudes)
                    center_lon = np.mean(longitudes)
                    map_center: list[Any] = [center_lat, center_lon]
                else:
                    map_center = list(self.default_params["map_center"])
            else:
                map_center = list(self.default_params["map_center"])

            # Create base map
            m = folium.Map(
                location=map_center,
                zoom_start=int(self.default_params["map_zoom"]),
                tiles="OpenStreetMap",
            )

            # Add sensor markers
            sensor_cluster = MarkerCluster(name="Sensors")

            for sensor in sensors:
                sensor_id = sensor.get("sensor_id", "unknown")
                lat = sensor.get("latitude")
                lon = sensor.get("longitude")

                if lat is None or lon is None:
                    continue

                # Determine marker color based on sensor status
                status = sensor.get("status", "active")
                if status == "active":
                    color = "green"
                elif status == "maintenance":
                    color = "orange"
                elif status == "error":
                    color = "red"
                else:
                    color = "blue"

                # Create popup content
                popup_content = self._create_sensor_popup(sensor)

                # Add marker
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Sensor {sensor_id}",
                    icon=folium.Icon(color=color, icon="map-marker"),
                ).add_to(sensor_cluster)

            sensor_cluster.add_to(m)

            # Add measurement heatmap if measurements provided
            if measurements:
                heatmap_data = self._prepare_heatmap_data(measurements)
                if heatmap_data:
                    HeatMap(
                        heatmap_data,
                        name="Measurements",
                        radius=self.default_params["heatmap_radius"],
                        blur=self.default_params["heatmap_blur"],
                        gradient={
                            0.2: "blue",
                            0.4: "lime",
                            0.6: "yellow",
                            0.8: "orange",
                            1: "red",
                        },
                    ).add_to(m)

            # Add layer control
            folium.LayerControl().add_to(m)

            # Save map
            _ensure_parent(output_file)
            m.save(output_file)

            return {
                "success": True,
                "map_file": output_file,
                "sensor_count": len(sensors),
                "measurement_count": len(measurements) if measurements else 0,
                "center_coordinates": map_center,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error creating sensor map: {e}")
            return {"error": f"Map creation failed: {str(e)}"}

    def _create_sensor_popup(self, sensor: dict) -> str:
        """Create HTML popup content for a sensor marker."""
        sensor_id = sensor.get("sensor_id", "Unknown")
        sensor_type = sensor.get("sensor_type", "Unknown")
        status = sensor.get("status", "Unknown")
        lat = sensor.get("latitude", "N/A")
        lon = sensor.get("longitude", "N/A")

        if lat == "N/A" or lon == "N/A":
            location_text = "N/A"
        else:
            latitude, longitude = _coordinate(lat, lon, "sensor")
            location_text = f"{latitude:.4f}, {longitude:.4f}"

        popup_html = f"""
        <div style="width: 200px;">
            <h4>Sensor {sensor_id}</h4>
            <p><strong>Type:</strong> {sensor_type}</p>
            <p><strong>Status:</strong> {status}</p>
            <p><strong>Location:</strong> {location_text}</p>
        """

        # Add recent measurements if available
        if "last_measurement" in sensor:
            measurement = sensor["last_measurement"]
            popup_html += f"""
            <p><strong>Last Reading:</strong> {measurement.get("value", "N/A")} {measurement.get("unit", "")}</p>
            <p><strong>Timestamp:</strong> {measurement.get("timestamp", "N/A")}</p>
            """

        popup_html += "</div>"
        return popup_html

    def _prepare_heatmap_data(self, measurements: list[dict]) -> list[list]:
        """Prepare measurement data for heatmap visualization."""
        heatmap_data = []

        for measurement in measurements:
            lat = measurement.get("latitude")
            lon = measurement.get("longitude")
            value = measurement.get("value")

            if lat is not None and lon is not None and value is not None:
                try:
                    lat, lon = _coordinate(lat, lon, "measurement")
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                # Normalize value for heatmap intensity (0-1 scale)
                intensity = min(1.0, max(0.0, value / 100.0))  # Simple normalization
                heatmap_data.append([lat, lon, intensity])

        return heatmap_data

    def create_spatial_interpolation_map(
        self,
        interpolation_data: dict,
        sensors: list[dict] | None = None,
        output_file: str = "interpolation_map.html",
    ) -> dict:
        """
        Create a map showing spatial interpolation results.

        Args:
            interpolation_data: Spatial interpolation results
            sensors: Optional sensor locations for reference
            output_file: Output HTML file path

        Returns:
            Dictionary with map generation results
        """
        if not HAS_VISUALIZATION:
            return {"error": "Visualization libraries not available"}

        try:
            if not isinstance(interpolation_data, dict):
                raise ValueError("interpolation_data must be a mapping")
            # Extract interpolation data
            interpolated_values = interpolation_data.get("interpolated_values", [])
            target_coordinates = interpolation_data.get("target_coordinates", [])
            uncertainty = interpolation_data.get("uncertainty", [])

            if not target_coordinates or not interpolated_values:
                return {"error": "No interpolation data provided"}
            if len(target_coordinates) != len(interpolated_values):
                raise ValueError(
                    "target_coordinates and interpolated_values must align"
                )
            if uncertainty and len(uncertainty) != len(target_coordinates):
                raise ValueError("uncertainty must align with target_coordinates")
            target_coordinates = [
                _coordinate(coord[0], coord[1], f"target_coordinates[{index}]")
                for index, coord in enumerate(target_coordinates)
            ]
            interpolated_values = [float(value) for value in interpolated_values]
            if not all(math.isfinite(value) for value in interpolated_values):
                raise ValueError("interpolated_values must be finite")
            if uncertainty:
                uncertainty = [float(value) for value in uncertainty]
                if not all(
                    math.isfinite(value) and value >= 0 for value in uncertainty
                ):
                    raise ValueError(
                        "uncertainty must contain finite nonnegative values"
                    )

            # Create base map centered on data
            center_lat = np.mean([coord[0] for coord in target_coordinates])
            center_lon = np.mean([coord[1] for coord in target_coordinates])

            m = folium.Map(
                location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap"
            )

            # Create interpolation surface as GeoJSON-like features
            features: list[dict[str, Any]] = []
            for i, (lat, lon) in enumerate(target_coordinates):
                value = interpolated_values[i]
                unc = uncertainty[i] if uncertainty else 0.5

                # Point geometry marks the sampled location; the covering H3
                # cell index travels in properties for downstream boundary work
                feature: dict[str, Any] = {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "interpolated_value": value,
                        "uncertainty": unc,
                        "index": i,
                        "h3_index": h3.latlng_to_cell(lat, lon, 8),
                    },
                }
                features.append(feature)

            # Add interpolation points with color coding
            for feature in features:
                lon, lat = feature["geometry"]["coordinates"]
                value = feature["properties"]["interpolated_value"]
                unc = feature["properties"]["uncertainty"]

                # Color based on interpolated value
                if value < 20:
                    color = "blue"
                elif value < 40:
                    color = "green"
                elif value < 60:
                    color = "yellow"
                elif value < 80:
                    color = "orange"
                else:
                    color = "red"

                # Size based on uncertainty (higher uncertainty = larger marker)
                size = 3 + unc * 7

                popup_text = f"""
                <div style="width: 150px;">
                    <h5>Interpolated Value</h5>
                    <p><strong>Value:</strong> {value:.2f}</p>
                    <p><strong>Uncertainty:</strong> {unc:.2f}</p>
                    <p><strong>Location:</strong> {lat:.4f}, {lon:.4f}</p>
                </div>
                """

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=size,
                    popup=folium.Popup(popup_text, max_width=200),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                ).add_to(m)

            # Add sensor locations if provided
            if sensors:
                for sensor in sensors:
                    lat = sensor.get("latitude")
                    lon = sensor.get("longitude")

                    if lat is not None and lon is not None:
                        folium.Marker(
                            location=[lat, lon],
                            popup=f"Sensor: {sensor.get('sensor_id', 'Unknown')}",
                            icon=folium.Icon(color="black", icon="record"),
                        ).add_to(m)

            # Save map
            _ensure_parent(output_file)
            m.save(output_file)

            return {
                "success": True,
                "map_file": output_file,
                "interpolation_points": len(features),
                "sensor_count": len(sensors) if sensors else 0,
                "center_coordinates": [center_lat, center_lon],
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error creating interpolation map: {e}")
            return {"error": f"Interpolation map creation failed: {str(e)}"}

    def create_time_series_plot(
        self, sensor_data: dict[str, list], output_file: str = "timeseries.png"
    ) -> dict:
        """
        Create time series plots for sensor data.

        Args:
            sensor_data: Dictionary mapping sensor IDs to time series data
            output_file: Output image file path

        Returns:
            Dictionary with plot generation results
        """
        if not HAS_VISUALIZATION:
            return {"error": "Visualization libraries not available"}

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            for sensor_id, data_points in sensor_data.items():
                if len(data_points) < 2:
                    continue

                # Extract timestamps and values
                timestamps = [
                    datetime.fromisoformat(dp["timestamp"]) for dp in data_points
                ]
                values = [dp["value"] for dp in data_points]

                # Plot time series
                ax.plot(
                    [t.timestamp() for t in timestamps],
                    values,
                    marker="o",
                    label=f"Sensor {sensor_id}",
                    linewidth=2,
                )

            ax.set_xlabel("Time")
            ax.set_ylabel("Sensor Value")
            ax.set_title("Sensor Time Series Data")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis for better readability
            import matplotlib.dates as mdates

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            fig.tight_layout()
            _ensure_parent(output_file)
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            return {
                "success": True,
                "plot_file": output_file,
                "sensor_count": len(sensor_data),
                "total_data_points": sum(len(data) for data in sensor_data.values()),
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            return {"error": f"Time series plot creation failed: {str(e)}"}

    def create_network_status_dashboard(
        self, network_data: dict, output_file: str = "dashboard.html"
    ) -> dict:
        """
        Create a network status dashboard.

        Args:
            network_data: Network status and health data
            output_file: Output HTML file path

        Returns:
            Dictionary with dashboard generation results
        """
        if not HAS_VISUALIZATION:
            return {"error": "Visualization libraries not available"}

        try:
            # Create HTML dashboard
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>IoT Network Status Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ background: #f5f5f5; padding: 15px; margin: 10px; border-radius: 5px; }}
                    .status-good {{ color: green; }}
                    .status-warning {{ color: orange; }}
                    .status-critical {{ color: red; }}
                    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                </style>
            </head>
            <body>
                <h1>IoT Network Status Dashboard</h1>
                <p><strong>Generated:</strong> {datetime.now().isoformat()}</p>

                <div class="grid">
            """

            # Network overview metrics
            if "network_summary" in network_data:
                summary = network_data["network_summary"]
                html_content += f"""
                <div class="metric">
                    <h3>Network Overview</h3>
                    <p><strong>Total Sensors:</strong> {summary.get("total_sensors", 0)}</p>
                    <p><strong>Health Score:</strong> {summary.get("average_health_score", 0):.2f}</p>
                    <p><strong>Active Sensors:</strong> {summary.get("status_distribution", {}).get("good", 0)}</p>
                    <p><strong>Sensors Needing Attention:</strong> {summary.get("sensors_needing_maintenance", 0)}</p>
                </div>
                """

            # Sensor health breakdown
            if "sensor_assessments" in network_data:
                assessments = network_data["sensor_assessments"]

                # Count by status
                status_counts: dict[str, int] = {}
                for assessment in assessments.values():
                    status = assessment.get("overall_status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1

                html_content += """
                <div class="metric">
                    <h3>Sensor Status Breakdown</h3>
                """

                for status, count in status_counts.items():
                    status_class = f"status-{status}"
                    html_content += f'<p class="{status_class}"><strong>{status.title()}:</strong> {count} sensors</p>'

                html_content += "</div>"

            # Recent measurements
            if "recent_measurements" in network_data:
                recent = network_data["recent_measurements"]
                html_content += f"""
                <div class="metric">
                    <h3>Recent Activity</h3>
                    <p><strong>Total Measurements:</strong> {len(recent)}</p>
                    <p><strong>Time Range:</strong> {recent[0].get("timestamp", "N/A") if recent else "N/A"}</p>
                </div>
                """

            html_content += """
                </div>

                <h2>Detailed Sensor Status</h2>
                <table border="1" style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #f0f0f0;">
                            <th>Sensor ID</th>
                            <th>Status</th>
                            <th>Health Score</th>
                            <th>Last Communication</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            # Add sensor details
            if "sensor_assessments" in network_data:
                for sensor_id, assessment in network_data["sensor_assessments"].items():
                    status = assessment.get("overall_status", "unknown")
                    score = assessment.get("overall_score", 0)
                    last_comm = "N/A"  # Would need to add this data

                    html_content += f"""
                    <tr>
                        <td>{sensor_id}</td>
                        <td class="status-{status}">{status.title()}</td>
                        <td>{score:.2f}</td>
                        <td>{last_comm}</td>
                    </tr>
                    """

            html_content += """
                    </tbody>
                </table>
            </body>
            </html>
            """

            _ensure_parent(output_file)
            # Save dashboard
            with open(output_file, "w") as f:
                f.write(html_content)

            return {
                "success": True,
                "dashboard_file": output_file,
                "sensor_count": len(network_data.get("sensor_assessments", {})),
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return {"error": f"Dashboard creation failed: {str(e)}"}

    def create_heatmap_overlay(
        self,
        measurements: list[dict],
        bounds: dict[str, float],
        output_file: str = "heatmap.png",
    ) -> dict:
        """
        Create a heatmap overlay for sensor measurements.

        Args:
            measurements: List of measurements with lat/lon and values
            bounds: Geographic bounds for the heatmap
            output_file: Output image file path

        Returns:
            Dictionary with heatmap generation results
        """
        if not HAS_VISUALIZATION:
            return {"error": "Visualization libraries not available"}

        try:
            # Extract data for plotting
            latitudes = []
            longitudes = []
            values = []

            for measurement in measurements:
                lat = measurement.get("latitude")
                lon = measurement.get("longitude")
                value = measurement.get("value")

                if lat is not None and lon is not None and value is not None:
                    latitudes.append(lat)
                    longitudes.append(lon)
                    values.append(value)

            if not values:
                return {"error": "No valid measurement data for heatmap"}

            required_bounds = ("lon_min", "lon_max", "lat_min", "lat_max")
            if not all(key in bounds for key in required_bounds):
                return {
                    "error": "bounds must include lon_min, lon_max, lat_min, and lat_max"
                }
            try:
                bounds_values = np.asarray(
                    [bounds[key] for key in required_bounds], dtype=float
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("bounds must contain numeric values") from exc
            if not np.all(np.isfinite(bounds_values)):
                raise ValueError("bounds must contain finite values")
            if (
                bounds_values[0] >= bounds_values[1]
                or bounds_values[2] >= bounds_values[3]
            ):
                return {"error": "bounds must be ordered min-to-max"}

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create scatter plot with color mapping
            scatter = ax.scatter(
                longitudes,
                latitudes,
                c=values,
                cmap="viridis",
                s=50,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Measurement Value")

            # Set bounds and labels
            ax.set_xlim(bounds.get("lon_min"), bounds.get("lon_max"))
            ax.set_ylim(bounds.get("lat_min"), bounds.get("lat_max"))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Sensor Measurement Heatmap")
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            _ensure_parent(output_file)
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            return {
                "success": True,
                "heatmap_file": output_file,
                "data_points": len(values),
                "bounds": bounds,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return {"error": f"Heatmap creation failed: {str(e)}"}

    def get_visualization_status(self) -> dict:
        """Get status of visualization capabilities."""
        return {
            "visualization_available": HAS_VISUALIZATION,
            "libraries": {
                "folium": HAS_VISUALIZATION,
                "matplotlib": HAS_VISUALIZATION,
                "plotly": HAS_VISUALIZATION,
            },
            "supported_formats": ["html", "png", "svg"] if HAS_VISUALIZATION else [],
            "cache_size": len(self.maps_cache),
            "default_parameters": self.default_params,
            "timestamp": datetime.now().isoformat(),
        }
