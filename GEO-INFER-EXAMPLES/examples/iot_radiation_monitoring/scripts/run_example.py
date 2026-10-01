#!/usr/bin/env python3
"""
GEO-INFER Examples: IoT Radiation Monitoring
Main execution script demonstrating integration of IoT, BAYES, SPACE, LOG, and TEST modules.
"""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import yaml
from geo_infer_bayes.models.spatial_gp import SpatialGP

#: Config covariance names mapped to ``SpatialGP`` (kernel, Matern degree).
_COVARIANCE_KERNELS: dict[str, tuple[str, float]] = {
    "exponential": ("exponential", 0.5),
    "matern_32": ("matern", 1.5),
    "matern_52": ("matern", 2.5),
    "squared_exponential": ("rbf", 2.5),
}

#: Approximate metres per degree of latitude, used to express the configured
#: covariance length scale in the degree units of the GP inputs.
_METRES_PER_DEGREE = 111_000.0


class EnhancedLogger:
    """Enhanced logging class for demonstration"""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.start_time = time.time()

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Initialize log file
        self.log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def log(self, level: str, operation: str, data: dict, module: str = "MAIN"):
        """Log a structured message"""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "module": module,
            "operation": operation,
            "elapsed_seconds": time.time() - self.start_time,
            **data,
        }

        # Console output
        if self.config.get("console", {}).get("enabled", True):
            print(f"[{level}] {module}.{operation}: {json.dumps(data, default=str)}")

        # File output
        if self.config.get("file", {}).get("enabled", True):
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")

    def info(self, operation: str, data: dict, module: str = "MAIN"):
        self.log("INFO", operation, data, module)

    def debug(self, operation: str, data: dict, module: str = "MAIN"):
        self.log("DEBUG", operation, data, module)

    def warning(self, operation: str, data: dict, module: str = "MAIN"):
        self.log("WARNING", operation, data, module)

    def error(self, operation: str, data: dict, module: str = "MAIN"):
        self.log("ERROR", operation, data, module)


class QualityController:
    """Quality control and testing for demonstration"""

    def __init__(self, config: dict):
        self.config = config
        self.test_results = {}

    def validate_sensor_data(self, data: pd.DataFrame) -> dict:
        """Validate sensor data quality"""
        validation_config = self.config.get("sensor_validation", {})

        results = {
            "total_records": len(data),
            "valid_records": 0,
            "validation_errors": [],
            "quality_score": 0.0,
        }

        if len(data) == 0:
            return results

        # Check radiation value ranges
        min_rad = validation_config.get("min_radiation", 0.0)
        max_rad = validation_config.get("max_radiation", 100.0)

        valid_radiation = (data["radiation_level"] >= min_rad) & (
            data["radiation_level"] <= max_rad
        )
        invalid_count = (~valid_radiation).sum()

        if invalid_count > 0:
            results["validation_errors"].append(
                f"{invalid_count} records outside radiation range [{min_rad}, {max_rad}]"
            )

        # Check for required fields
        required_fields = [
            "sensor_id",
            "latitude",
            "longitude",
            "radiation_level",
            "timestamp",
        ]
        for field in required_fields:
            if field not in data.columns:
                results["validation_errors"].append(f"Missing required field: {field}")
                return results

        # Check for null values
        null_counts = data[required_fields].isnull().sum()
        for field, count in null_counts.items():
            if count > 0:
                results["validation_errors"].append(f"{count} null values in {field}")

        # Calculate quality score
        valid_records = valid_radiation.sum()
        results["valid_records"] = int(valid_records)
        results["quality_score"] = valid_records / len(data) if len(data) > 0 else 0.0

        return results

    def test_spatial_operations(self, h3_indices: list[str]) -> dict:
        """Test H3 spatial operations"""
        results = {
            "test_name": "spatial_operations",
            "total_cells": len(h3_indices),
            "valid_cells": 0,
            "invalid_cells": [],
            "neighbor_tests": {"passed": 0, "failed": 0},
        }

        for h3_index in h3_indices:
            try:
                # Test H3 index validity
                lat, lon = h3.cell_to_latlng(h3_index)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    results["valid_cells"] += 1

                    # Test neighbor operations
                    neighbors = h3.grid_disk(h3_index, 1)
                    if len(neighbors) > 0:
                        results["neighbor_tests"]["passed"] += 1
                    else:
                        results["neighbor_tests"]["failed"] += 1
                else:
                    results["invalid_cells"].append(h3_index)

            except Exception as e:
                results["invalid_cells"].append(f"{h3_index}: {str(e)}")

        return results

    def test_bayesian_inference(self, posterior_results: dict) -> dict:
        """Test Bayesian inference quality"""
        results = {
            "test_name": "bayesian_inference",
            "convergence_check": False,
            "uncertainty_check": False,
            "prediction_check": False,
            "overall_passed": False,
        }

        # Check convergence
        if posterior_results.get("converged", False):
            results["convergence_check"] = True

        # Check uncertainty bounds
        if "uncertainty" in posterior_results:
            uncertainty = posterior_results["uncertainty"]
            if isinstance(uncertainty, (list, np.ndarray)) and len(uncertainty) > 0:
                mean_uncertainty = np.mean(uncertainty)
                if 0 < mean_uncertainty < 10:  # Reasonable uncertainty range
                    results["uncertainty_check"] = True

        # Check predictions
        if "predictions" in posterior_results:
            predictions = posterior_results["predictions"]
            if isinstance(predictions, (list, np.ndarray)) and len(predictions) > 0:
                predictions_array = np.array(predictions)
                if np.all(np.isfinite(predictions_array)) and np.all(
                    predictions_array >= 0
                ):
                    results["prediction_check"] = True

        # Overall assessment
        results["overall_passed"] = all(
            [
                results["convergence_check"],
                results["uncertainty_check"],
                results["prediction_check"],
            ]
        )

        return results


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config["radiation_monitoring"]
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def generate_sample_sensor_data(config: dict, logger: EnhancedLogger) -> pd.DataFrame:
    """Generate sample sensor data for demonstration"""
    logger.info("data_generation_start", {"operation": "generate_sample_data"}, "DATA")

    sim_config = config["simulation"]
    sensor_count = sim_config["sensor_count"]
    background_radiation = sim_config["background_radiation"]
    noise_level = sim_config["noise_level"]

    # Generate random sensor locations globally
    np.random.seed(42)  # For reproducible results
    latitudes = np.random.uniform(-85, 85, sensor_count)  # Avoid extreme polar regions
    longitudes = np.random.uniform(-180, 180, sensor_count)

    # Generate radiation measurements with noise
    radiation_levels = np.random.normal(background_radiation, noise_level, sensor_count)
    radiation_levels = np.maximum(radiation_levels, 0)  # Ensure non-negative

    # Add simulated anomalies
    if sim_config["anomalies"]["enable"]:
        for anomaly in sim_config["anomalies"]["locations"]:
            # Find sensors near anomaly locations
            distances = np.sqrt(
                (latitudes - anomaly["lat"]) ** 2 + (longitudes - anomaly["lon"]) ** 2
            )
            near_anomaly = distances < 1.0  # Within ~111km

            # Increase radiation levels near anomalies
            radiation_levels[near_anomaly] *= anomaly["intensity"]

    # Create DataFrame
    data = pd.DataFrame(
        {
            "sensor_id": [f"sensor_{i:06d}" for i in range(sensor_count)],
            "latitude": latitudes,
            "longitude": longitudes,
            "radiation_level": radiation_levels,
            "timestamp": datetime.now(UTC),
            "network": np.random.choice(["safecast", "eurdep", "ctbto"], sensor_count),
            "quality_flag": "ok",
        }
    )

    # Add H3 indices
    data["h3_index"] = [
        h3.latlng_to_cell(lat, lon, config["spatial"]["h3_resolution"])
        for lat, lon in zip(data["latitude"], data["longitude"])
    ]

    logger.info(
        "data_generation_complete",
        {
            "sensor_count": len(data),
            "background_radiation": background_radiation,
            "anomalies_enabled": sim_config["anomalies"]["enable"],
            "h3_resolution": config["spatial"]["h3_resolution"],
        },
        "DATA",
    )

    return data


def perform_spatial_indexing(
    data: pd.DataFrame, config: dict, logger: EnhancedLogger
) -> dict:
    """Perform H3 spatial indexing operations"""
    logger.info("spatial_indexing_start", {"operation": "h3_spatial_indexing"}, "SPACE")

    start_time = time.time()

    # Group data by H3 cells
    h3_groups = (
        data.groupby("h3_index")
        .agg(
            {
                "radiation_level": ["mean", "std", "count"],
                "latitude": "mean",
                "longitude": "mean",
                "sensor_id": "count",
            }
        )
        .reset_index()
    )

    # Flatten column names
    h3_groups.columns = [
        "h3_index",
        "radiation_mean",
        "radiation_std",
        "measurement_count",
        "lat_center",
        "lon_center",
        "sensor_count",
    ]

    # Add neighbor information
    h3_groups["neighbors"] = h3_groups["h3_index"].apply(
        lambda x: list(h3.grid_disk(x, config["spatial"]["neighbor_rings"]))
    )

    processing_time = time.time() - start_time

    logger.info(
        "spatial_indexing_complete",
        {
            "unique_h3_cells": len(h3_groups),
            "avg_sensors_per_cell": h3_groups["sensor_count"].mean(),
            "processing_time_seconds": processing_time,
            "h3_resolution": config["spatial"]["h3_resolution"],
        },
        "SPACE",
    )

    return {
        "h3_aggregated_data": h3_groups,
        "processing_time": processing_time,
        "cell_count": len(h3_groups),
    }


def perform_bayesian_inference(
    h3_data: pd.DataFrame, config: dict, logger: EnhancedLogger
) -> dict:
    """Perform Bayesian spatial inference"""
    logger.info("bayesian_inference_start", {"operation": "spatial_inference"}, "BAYES")

    start_time = time.time()
    bayes_config = config["bayesian_inference"]

    # Prepare input data
    coordinates = h3_data[["lat_center", "lon_center"]].to_numpy(dtype=float)
    observations = h3_data["radiation_mean"].to_numpy(dtype=float)

    # Create prediction grid
    resolution = bayes_config["prediction_grid"]["resolution"]
    bounds = config["spatial"]["global_bounds"]

    # Generate H3 grid for prediction (10-degree lattice)
    prediction_h3_cells = []
    for lat in np.arange(bounds["min_lat"], bounds["max_lat"], 10):
        for lon in np.arange(bounds["min_lon"], bounds["max_lon"], 10):
            prediction_h3_cells.append(h3.latlng_to_cell(lat, lon, resolution))

    # Remove duplicates (sorted for reproducible output) and limit size
    prediction_h3_cells = sorted(set(prediction_h3_cells))
    max_cells = bayes_config["prediction_grid"]["max_cells"]
    prediction_h3_cells = prediction_h3_cells[:max_cells]

    prior_mean = bayes_config["prior"]["mean"]
    covariance = bayes_config["covariance"]
    length_scale = covariance["length_scale"]
    kernel, degree = _COVARIANCE_KERNELS[covariance["function"]]

    # Gaussian-process regression with a constant prior mean (BAYES SpatialGP).
    gp = SpatialGP(
        kernel=kernel,
        lengthscale=length_scale / _METRES_PER_DEGREE,
        variance=covariance["variance"],
        noise=covariance["noise_variance"],
        degree=degree,
        mean_function=lambda x: np.full(len(x), prior_mean),
    )
    gp.fit(coordinates, observations)
    prediction_points = np.array(
        [h3.cell_to_latlng(cell) for cell in prediction_h3_cells], dtype=float
    )
    mean_pred, std_pred = gp.predict(prediction_points, return_std=True)
    predictions = mean_pred.tolist()
    uncertainties = (std_pred**2).tolist()

    processing_time = time.time() - start_time

    # Calculate confidence intervals
    confidence_levels = bayes_config["inference"]["confidence_levels"]
    confidence_intervals = {}
    for level in confidence_levels:
        z_score = {0.68: 1.0, 0.95: 1.96, 0.99: 2.58}.get(level, 1.96)
        lower = np.array(predictions) - z_score * np.sqrt(uncertainties)
        upper = np.array(predictions) + z_score * np.sqrt(uncertainties)
        confidence_intervals[f"ci_{int(level * 100)}"] = {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        }

    logger.info(
        "bayesian_inference_complete",
        {
            "prediction_cells": len(prediction_h3_cells),
            "mean_prediction": np.mean(predictions),
            "mean_uncertainty": np.mean(uncertainties),
            "processing_time_seconds": processing_time,
            "converged": True,
            "method": "gaussian_process",
            "kernel": covariance["function"],
        },
        "BAYES",
    )

    return {
        "prediction_cells": prediction_h3_cells,
        "predictions": predictions,
        "uncertainty": uncertainties,
        "confidence_intervals": confidence_intervals,
        "processing_time": processing_time,
        "converged": True,
        "prior_mean": prior_mean,
        "length_scale": length_scale,
    }


def detect_anomalies(data: pd.DataFrame, config: dict, logger: EnhancedLogger) -> dict:
    """Detect radiation anomalies"""
    logger.info("anomaly_detection_start", {"operation": "anomaly_detection"}, "IOT")

    start_time = time.time()
    anomaly_config = config["anomaly_detection"]

    # Statistical anomaly detection
    stat_config = anomaly_config["statistical"]
    radiation_mean = data["radiation_level"].mean()
    radiation_std = data["radiation_level"].std()

    anomalies = []

    for _, row in data.iterrows():
        z_score = abs(row["radiation_level"] - radiation_mean) / radiation_std

        alert_level = None
        if z_score >= stat_config["threshold_critical"]:
            alert_level = "critical"
        elif z_score >= stat_config["threshold_severe"]:
            alert_level = "severe"
        elif z_score >= stat_config["threshold_mild"]:
            alert_level = "mild"

        if alert_level:
            anomalies.append(
                {
                    "id": f"anomaly_{len(anomalies) + 1:03d}",
                    "sensor_id": row["sensor_id"],
                    "location": {"lat": row["latitude"], "lon": row["longitude"]},
                    "h3_index": row["h3_index"],
                    "radiation_level": row["radiation_level"],
                    "anomaly_score": z_score,
                    "alert_level": alert_level,
                    "detection_method": "statistical",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    processing_time = time.time() - start_time

    # Summary by alert level
    alert_summary = {"mild": 0, "severe": 0, "critical": 0}
    for anomaly in anomalies:
        alert_summary[anomaly["alert_level"]] += 1

    logger.info(
        "anomaly_detection_complete",
        {
            "total_anomalies": len(anomalies),
            "alert_levels": alert_summary,
            "processing_time_seconds": processing_time,
            "detection_methods": anomaly_config["methods"],
        },
        "IOT",
    )

    return {
        "anomalies": anomalies,
        "summary": alert_summary,
        "total_sensors": len(data),
        "anomaly_rate": len(anomalies) / len(data),
        "processing_time": processing_time,
    }


def save_results(
    sensor_data: pd.DataFrame,
    spatial_results: dict,
    inference_results: dict,
    anomaly_results: dict,
    config: dict,
    logger: EnhancedLogger,
):
    """Save all results to output files"""
    logger.info("save_results_start", {"operation": "save_outputs"}, "MAIN")

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 1. Save global radiation map as GeoJSON
    prediction_cells = inference_results["prediction_cells"]
    predictions = inference_results["predictions"]
    uncertainties = inference_results["uncertainty"]

    # Create GeoJSON features for H3 cells
    features = []
    for i, h3_cell in enumerate(prediction_cells):
        # Get H3 cell boundary
        try:
            boundary = h3.cell_to_boundary(h3_cell)
            ring = [[lng, lat] for lat, lng in boundary]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
        except Exception as e:
            logger.warning(
                "h3_boundary_error", {"h3_cell": h3_cell, "error": str(e)}, "MAIN"
            )
            continue

        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "h3_index": h3_cell,
                "radiation_mean": predictions[i],
                "uncertainty": uncertainties[i],
                "confidence_intervals": {
                    level: {
                        "lower": inference_results["confidence_intervals"][
                            f"ci_{int(level * 100)}"
                        ]["lower"][i],
                        "upper": inference_results["confidence_intervals"][
                            f"ci_{int(level * 100)}"
                        ]["upper"][i],
                    }
                    for level in config["bayesian_inference"]["inference"][
                        "confidence_levels"
                    ]
                },
            },
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "h3_resolution": config["spatial"]["h3_resolution"],
            "total_features": len(features),
            "inference_method": config["bayesian_inference"]["inference"]["method"],
        },
    }

    with open(output_dir / "global_radiation_map.geojson", "w") as f:
        json.dump(geojson_data, f, indent=2)

    # 2. Save anomaly report
    anomaly_report = {
        "detection_timestamp": datetime.now(UTC).isoformat(),
        "total_anomalies": len(anomaly_results["anomalies"]),
        "by_severity": anomaly_results["summary"],
        "anomaly_rate": anomaly_results["anomaly_rate"],
        "detection_methods": config["anomaly_detection"]["methods"],
        "anomalies": anomaly_results["anomalies"],
    }

    with open(output_dir / "anomaly_report.json", "w") as f:
        json.dump(anomaly_report, f, indent=2)

    # 3. Save sensor data summary
    sensor_summary = {
        "total_sensors": len(sensor_data),
        "networks": sensor_data["network"].value_counts().to_dict(),
        "radiation_statistics": {
            "mean": sensor_data["radiation_level"].mean(),
            "std": sensor_data["radiation_level"].std(),
            "min": sensor_data["radiation_level"].min(),
            "max": sensor_data["radiation_level"].max(),
            "percentiles": {
                "p25": sensor_data["radiation_level"].quantile(0.25),
                "p50": sensor_data["radiation_level"].quantile(0.50),
                "p75": sensor_data["radiation_level"].quantile(0.75),
                "p95": sensor_data["radiation_level"].quantile(0.95),
                "p99": sensor_data["radiation_level"].quantile(0.99),
            },
        },
        "spatial_coverage": {
            "unique_h3_cells": sensor_data["h3_index"].nunique(),
            "h3_resolution": config["spatial"]["h3_resolution"],
        },
    }

    with open(output_dir / "sensor_summary.json", "w") as f:
        json.dump(sensor_summary, f, indent=2, default=str)

    # 4. Save processing performance metrics
    performance_metrics = {
        "timestamp": datetime.now(UTC).isoformat(),
        "processing_times": {
            "spatial_indexing": spatial_results["processing_time"],
            "bayesian_inference": inference_results["processing_time"],
            "anomaly_detection": anomaly_results["processing_time"],
        },
        "data_volumes": {
            "input_sensors": len(sensor_data),
            "h3_cells": spatial_results["cell_count"],
            "prediction_cells": len(inference_results["prediction_cells"]),
            "anomalies_detected": len(anomaly_results["anomalies"]),
        },
    }

    with open(output_dir / "performance_metrics.json", "w") as f:
        json.dump(performance_metrics, f, indent=2)

    logger.info(
        "save_results_complete",
        {
            "output_files": [
                "global_radiation_map.geojson",
                "anomaly_report.json",
                "sensor_summary.json",
                "performance_metrics.json",
            ],
            "output_directory": str(output_dir),
        },
        "MAIN",
    )


def run_tests(
    sensor_data: pd.DataFrame,
    spatial_results: dict,
    inference_results: dict,
    config: dict,
    logger: EnhancedLogger,
) -> dict:
    """Run comprehensive tests"""
    logger.info("testing_start", {"operation": "comprehensive_testing"}, "TEST")

    qc = QualityController(config["quality_control"])
    test_results = {}

    # Test 1: Data quality validation
    test_results["data_quality"] = qc.validate_sensor_data(sensor_data)
    logger.info("test_complete", test_results["data_quality"], "TEST")

    # Test 2: Spatial operations
    h3_indices = sensor_data["h3_index"].unique().tolist()
    test_results["spatial_operations"] = qc.test_spatial_operations(h3_indices)
    logger.info("test_complete", test_results["spatial_operations"], "TEST")

    # Test 3: Bayesian inference quality
    test_results["bayesian_inference"] = qc.test_bayesian_inference(inference_results)
    logger.info("test_complete", test_results["bayesian_inference"], "TEST")

    # Overall test summary
    all_tests_passed = all(
        [
            test_results["data_quality"]["quality_score"] > 0.9,
            test_results["spatial_operations"]["valid_cells"] > 0,
            test_results["bayesian_inference"]["overall_passed"],
        ]
    )

    test_results["summary"] = {
        "all_tests_passed": all_tests_passed,
        "timestamp": datetime.now(UTC).isoformat(),
        "test_count": len(test_results) - 1,  # Exclude summary itself
    }

    logger.info(
        "testing_complete",
        {
            "all_tests_passed": all_tests_passed,
            "test_categories": list(test_results.keys()),
        },
        "TEST",
    )

    return test_results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="GEO-INFER IoT Radiation Monitoring Example"
    )
    parser.add_argument(
        "--config",
        default="../config/example_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--h3-resolution", type=int, help="Override H3 resolution")
    parser.add_argument("--sensor-count", type=int, help="Override sensor count")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command line overrides
    if args.h3_resolution:
        config["spatial"]["h3_resolution"] = args.h3_resolution
    if args.sensor_count:
        config["simulation"]["sensor_count"] = args.sensor_count
    if args.verbose:
        config["logging"]["level"] = "DEBUG"

    # Initialize logger
    logger = EnhancedLogger("radiation_monitoring", config["logging"])

    logger.info(
        "example_start",
        {
            "config_file": args.config,
            "h3_resolution": config["spatial"]["h3_resolution"],
            "sensor_count": config["simulation"]["sensor_count"],
            "project_name": config["project_name"],
        },
    )

    try:
        # Step 1: Generate sample sensor data
        print("\n🔄 Step 1: Generating sample sensor data...")
        sensor_data = generate_sample_sensor_data(config, logger)
        print(f"✓ Generated {len(sensor_data)} sensor measurements")

        # Step 2: Perform spatial indexing
        print("\n🔄 Step 2: Performing H3 spatial indexing...")
        spatial_results = perform_spatial_indexing(sensor_data, config, logger)
        print(f"✓ Indexed into {spatial_results['cell_count']} H3 cells")

        # Step 3: Bayesian spatial inference
        print("\n🔄 Step 3: Performing Bayesian spatial inference...")
        inference_results = perform_bayesian_inference(
            spatial_results["h3_aggregated_data"], config, logger
        )
        print(
            f"✓ Generated predictions for {len(inference_results['prediction_cells'])} grid cells"
        )

        # Step 4: Anomaly detection
        print("\n🔄 Step 4: Detecting radiation anomalies...")
        anomaly_results = detect_anomalies(sensor_data, config, logger)
        print(f"✓ Detected {len(anomaly_results['anomalies'])} anomalies")

        # Step 5: Run tests
        print("\n🔄 Step 5: Running quality assurance tests...")
        test_results = run_tests(
            sensor_data, spatial_results, inference_results, config, logger
        )
        print(
            f"✓ Completed testing - All tests passed: {test_results['summary']['all_tests_passed']}"
        )

        # Step 6: Save results
        print("\n🔄 Step 6: Saving results...")
        save_results(
            sensor_data,
            spatial_results,
            inference_results,
            anomaly_results,
            config,
            logger,
        )
        print("✓ Results saved to output/ directory")

        # Final summary
        total_time = time.time() - logger.start_time
        print(f"\n🎉 Example completed successfully in {total_time:.2f} seconds!")
        print("📊 Key Results:")
        print(
            f"   • Processed {len(sensor_data)} sensors across {spatial_results['cell_count']} H3 cells"
        )
        print(
            f"   • Generated {len(inference_results['prediction_cells'])} spatial predictions"
        )
        print(f"   • Detected {len(anomaly_results['anomalies'])} radiation anomalies")
        print(
            f"   • All quality tests passed: {test_results['summary']['all_tests_passed']}"
        )
        print("📁 Outputs saved to: output/")
        print(f"📝 Logs saved to: {logger.log_file}")

        logger.info(
            "example_complete",
            {
                "total_time_seconds": total_time,
                "success": True,
                "sensors_processed": len(sensor_data),
                "anomalies_detected": len(anomaly_results["anomalies"]),
                "all_tests_passed": test_results["summary"]["all_tests_passed"],
            },
        )

    except Exception as e:
        logger.error(
            "example_failed", {"error": str(e), "error_type": type(e).__name__}
        )
        print(f"\n❌ Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
