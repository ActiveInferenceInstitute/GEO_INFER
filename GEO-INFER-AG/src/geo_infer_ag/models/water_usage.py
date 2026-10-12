"""
Agricultural water usage modeling functionality.
"""

import io
from typing import Any
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

from geo_infer_ag.models.base import (
    AgricultureModel,
    read_verified_payload,
    write_signed_payload,
)


class WaterUsageModel(AgricultureModel):
    """
    Model for predicting agricultural water usage and requirements.

    This model predicts crop water requirements, irrigation needs, water efficiency,
    and other water-related metrics for agricultural fields.

    Attributes:
        crop_type: Type of crop for water usage prediction
        model_type: Type of underlying model ('reference_et', 'statistical', 'process_based')
        water_balance_components: List of water balance components to model
    """

    def __init__(
        self,
        crop_type: str | None = None,
        model_type: str = "reference_et",
        water_balance_components: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the water usage model.

        Args:
            crop_type: Optional crop type (can be used for crop-specific models)
            model_type: Type of model to use for prediction
            water_balance_components: Optional list of water balance components to model
            config: Optional configuration parameters
        """
        name = "water_usage_model"
        if crop_type:
            name = f"{crop_type}_water_usage_model"

        super().__init__(name=name, config=config)

        self.crop_type = crop_type
        self.model_type = model_type
        self.predictor: RandomForestRegressor | None = None
        self.fitted = False
        self.feature_columns: list[str] | None = None

        # Default water balance components if not provided
        if water_balance_components is None:
            self.water_balance_components = [
                "evapotranspiration",
                "precipitation",
                "irrigation",
                "runoff",
                "drainage",
            ]
        else:
            self.water_balance_components = water_balance_components

        # Define required inputs based on model type
        if model_type == "reference_et":
            self.required_inputs = ["field_data", "weather_data"]
        elif model_type == "statistical":
            self.required_inputs = ["field_data", "weather_data"]
        elif model_type == "process_based":
            self.required_inputs = [
                "field_data",
                "weather_data",
                "soil_data",
                "management_data",
            ]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Update metadata
        self.metadata.update(
            {
                "crop_type": crop_type,
                "model_type": model_type,
                "water_balance_components": self.water_balance_components,
            }
        )

        # Initialize crop coefficients for ET-based methods
        self.crop_coefficients: dict[str, dict[str, Any]] = {
            "corn": {
                "initial": 0.3,
                "mid": 1.2,
                "end": 0.6,
                "length_days": [30, 40, 50, 30],
            },
            "wheat": {
                "initial": 0.3,
                "mid": 1.15,
                "end": 0.4,
                "length_days": [20, 30, 60, 30],
            },
            "rice": {
                "initial": 1.05,
                "mid": 1.2,
                "end": 0.9,
                "length_days": [30, 30, 60, 30],
            },
            "soybean": {
                "initial": 0.4,
                "mid": 1.15,
                "end": 0.5,
                "length_days": [20, 30, 60, 30],
            },
            "cotton": {
                "initial": 0.35,
                "mid": 1.2,
                "end": 0.6,
                "length_days": [30, 50, 60, 50],
            },
            "alfalfa": {
                "initial": 0.4,
                "mid": 0.95,
                "end": 0.9,
                "length_days": [10, 30, 150, 30],
            },
            "generic": {
                "initial": 0.3,
                "mid": 1.0,
                "end": 0.5,
                "length_days": [25, 35, 50, 25],
            },
        }

    def fit(
        self,
        training_data: dict[str, Any],
        target_column: str = "water_usage",
        feature_columns: list[str] | None = None,
    ) -> None:
        """
        Train the water usage prediction model using historical data.

        Args:
            training_data: Dictionary of training data sources
            target_column: Column name containing water usage values
            feature_columns: Optional list of feature columns to use

        Raises:
            ValueError: If required training data is missing
        """
        # Only statistical models require fitting
        if self.model_type not in ["statistical"]:
            self.fitted = True
            return

        # Validate required training data
        if "field_data" not in training_data or "weather_data" not in training_data:
            raise ValueError("Field data and weather data required for training")

        # Combine field and weather data for training
        field_data = training_data["field_data"]

        # This is a simplification - in a real model, you would need to properly join
        # and align the field data with weather data, potentially using time series
        # aggregation and spatial joins

        # For simplicity, assume field_data already has the target_column
        train_df = field_data

        # Ensure target column exists
        if target_column not in train_df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in training data"
            )

        # Default feature columns if not provided
        if feature_columns is None:
            # Use all numeric columns except the target
            numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != target_column]

        # Initialize and train a RandomForest model for statistical prediction
        self.predictor = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )

        # Train the model
        X = train_df[feature_columns]
        y = train_df[target_column]

        self.predictor.fit(X, y)
        self.feature_columns = feature_columns

        self.fitted = True

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Predict water usage metrics using the model.

        Args:
            data: Dictionary of input data sources

        Returns:
            Dictionary containing water usage predictions and metadata

        Raises:
            ValueError: If required inputs are missing
        """
        # Validate required inputs
        self.validate_inputs(data)

        # Get field and weather data
        field_data = data["field_data"]
        weather_data = data["weather_data"]

        # Check if weather data is properly formatted
        required_weather_vars = [
            "temperature",
            "solar_radiation",
            "humidity",
            "wind_speed",
        ]
        missing_vars = [
            var for var in required_weather_vars if var not in weather_data.columns
        ]

        if missing_vars and self.model_type in ["reference_et", "process_based"]:
            raise ValueError(f"Missing required weather variables: {missing_vars}")

        # Generate predictions based on model type
        if self.model_type == "reference_et":
            # Reference ET approach calculates crop water requirements based on
            # reference evapotranspiration and crop coefficients

            # Calculate reference ET (using simplified Penman-Monteith equation)
            # Normally this would use a more complex implementation
            reference_et = self._calculate_reference_et(weather_data)

            # Get crop coefficients for the crop type (or use generic)
            crop_type = self.crop_type or "generic"
            crop_coeffs = self.crop_coefficients.get(
                crop_type.lower(), self.crop_coefficients["generic"]
            )

            # For simplification, use the mid-season coefficient
            # In a real model, you would determine growth stage and use the appropriate coefficient
            kc = float(crop_coeffs["mid"])

            # Calculate crop ET (crop water requirement)
            crop_et = reference_et * kc

            # Calculate irrigation requirement (simplified)
            # In reality, this would account for effective rainfall, soil moisture, etc.
            effective_rainfall = (
                weather_data["precipitation"] * 0.7
            )  # Assume 70% effectiveness
            irrigation_requirement = np.maximum(0, crop_et - effective_rainfall)

            # Calculate water metrics for each field
            num_fields = len(field_data)

            # Aggregate daily values to seasonal totals
            # This is simplistic - a real model would account for crop calendars
            seasonal_et = crop_et.sum() if isinstance(crop_et, pd.Series) else crop_et
            seasonal_rainfall = (
                effective_rainfall.sum()
                if isinstance(effective_rainfall, pd.Series)
                else effective_rainfall
            )
            seasonal_irrigation = (
                irrigation_requirement.sum()
                if isinstance(irrigation_requirement, pd.Series)
                else irrigation_requirement
            )

            # Create field-level predictions
            # For simplicity, we're applying the same values to all fields
            # In practice, you would account for spatial variations
            water_requirement = np.full(num_fields, seasonal_et)
            water_requirement_m3 = (
                water_requirement * field_data["area_ha"] * 10
            )  # mm * ha * 10 = m³

            irrigation_requirement_values = np.full(num_fields, seasonal_irrigation)
            irrigation_m3 = irrigation_requirement_values * field_data["area_ha"] * 10

            effective_rainfall_values = np.full(num_fields, seasonal_rainfall)
            rainfall_m3 = effective_rainfall_values * field_data["area_ha"] * 10

            # Calculate water efficiency metrics
            if "yield" in field_data.columns:
                # If yield data is available, calculate actual water productivity
                water_productivity = field_data["yield"] / water_requirement
            else:
                # Use typical water productivity values (kg/m3 or t/ha per 100mm approx) based on crop
                # A hardcoded 1.0 was a baseline; now defaults are crop-specific
                typical_wp = {
                    "corn": 2.0,
                    "wheat": 1.0,
                    "rice": 0.8,
                    "soybean": 0.6,
                    "cotton": 0.5,
                    "alfalfa": 1.2,
                    "generic": 1.0,
                }
                crop_lower = self.crop_type.lower() if self.crop_type else "generic"
                default_wp = typical_wp.get(crop_lower, 1.0)
                water_productivity = np.full(num_fields, default_wp)

            # Add predictions to results
            result = {
                "water_requirement_mm": water_requirement,
                "water_requirement_m3": water_requirement_m3,
                "irrigation_requirement_mm": irrigation_requirement_values,
                "irrigation_requirement_m3": irrigation_m3,
                "effective_rainfall_mm": effective_rainfall_values,
                "effective_rainfall_m3": rainfall_m3,
                "water_productivity": water_productivity,
                "spatial_results": {
                    "water_requirement_mm": water_requirement,
                    "irrigation_requirement_mm": irrigation_requirement_values,
                    "effective_rainfall_mm": effective_rainfall_values,
                    "water_productivity": water_productivity,
                },
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "model_type": self.model_type,
                    "crop_type": self.crop_type,
                    "reference_et_method": "simplified_penman_monteith",
                },
            }

            # Calculate summary statistics
            result["summary"] = {
                "mean_water_requirement_mm": float(np.mean(water_requirement)),
                "total_water_requirement_m3": float(np.sum(water_requirement_m3)),
                "mean_irrigation_requirement_mm": float(
                    np.mean(irrigation_requirement_values)
                ),
                "total_irrigation_requirement_m3": float(np.sum(irrigation_m3)),
            }

        elif self.model_type == "statistical":
            # Check if model is fitted
            if not self.fitted:
                raise ValueError("Statistical model must be fitted before prediction")

            # Check if all feature columns are available
            feature_cols = self.feature_columns or []
            missing_features = [
                col for col in feature_cols if col not in field_data.columns
            ]
            if missing_features:
                raise ValueError(f"Missing feature columns: {missing_features}")

            # Generate predictions
            if self.predictor is None:
                raise ValueError("Statistical model must be fitted before prediction")
            X = field_data[feature_cols]
            water_usage_predictions = self.predictor.predict(X)

            # Calculate water metrics based on predictions
            # Here we assume water_usage_predictions are in mm of water per growing season
            water_requirement = water_usage_predictions
            water_requirement_m3 = water_requirement * field_data["area_ha"] * 10

            # Simplified estimation of irrigation requirement
            # In practice, this would be more complex and depend on rainfall
            rainfall_factor = 0.3  # Assume 30% of water comes from rainfall
            irrigation_requirement = water_requirement * (1 - rainfall_factor)
            irrigation_m3 = irrigation_requirement * field_data["area_ha"] * 10

            # Add predictions to results
            result = {
                "water_requirement_mm": water_requirement,
                "water_requirement_m3": water_requirement_m3,
                "irrigation_requirement_mm": irrigation_requirement,
                "irrigation_requirement_m3": irrigation_m3,
                "spatial_results": {
                    "water_requirement_mm": water_requirement,
                    "irrigation_requirement_mm": irrigation_requirement,
                },
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "model_type": self.model_type,
                    "crop_type": self.crop_type,
                },
            }

            # Calculate summary statistics
            result["summary"] = {
                "mean_water_requirement_mm": float(np.mean(water_requirement)),
                "total_water_requirement_m3": float(np.sum(water_requirement_m3)),
                "mean_irrigation_requirement_mm": float(
                    np.mean(irrigation_requirement)
                ),
                "total_irrigation_requirement_m3": float(np.sum(irrigation_m3)),
            }

        elif self.model_type == "process_based":
            # Process-based prediction using water balance simulation
            # These models require detailed soil, weather, and crop data
            required_process_data = ["soil_data", "management_data"]
            missing_data = [d for d in required_process_data if d not in data]

            if missing_data:
                raise ValueError(
                    f"Missing data for process-based model: {missing_data}"
                )

            soil_data = data["soil_data"]
            management_data = data["management_data"]

            # Soil-Water-Atmosphere (SWA) balance model.
            # Uses the FAO-56 dual crop coefficient framework (Allen et al. 1998):
            #   ETc = (Kcb * Ks + Ke) * ETo
            # where Ks is the water-stress coefficient and Ke is the evaporation coefficient.

            num_fields = len(field_data)

            # Step 1: Compute reference ET using Penman-Monteith (reuse existing helper)
            ref_et = self._calculate_reference_et(weather_data)  # mm/day series
            season_days = len(weather_data)

            # Step 2: Extract soil properties (defaults from FAO-56 typical loam)
            theta_fc = float(
                soil_data.get("field_capacity", pd.Series([0.30])).mean()
            )  # m³/m³
            theta_pwp = float(
                soil_data.get("wilting_point", pd.Series([0.14])).mean()
            )  # m³/m³
            root_depth = float(
                management_data.get("root_depth_m", 0.6)
                if isinstance(management_data, dict)
                else 0.6
            )
            # Total available water (TAW) in mm
            taw = 1000 * (theta_fc - theta_pwp) * root_depth

            # Step 3: Crop coefficient for mid-season
            crop_type = self.crop_type or "generic"
            crop_coeffs = self.crop_coefficients.get(
                crop_type.lower(), self.crop_coefficients["generic"]
            )
            kcb = float(crop_coeffs["mid"])
            ke = 0.05  # soil evaporation coefficient (simplified)

            # Step 4: Daily water balance loop
            daily_et = (kcb + ke) * ref_et  # mm/day
            daily_precip = (
                weather_data["precipitation"]
                if "precipitation" in weather_data.columns
                else pd.Series(np.zeros(season_days))
            )

            # Management irrigation threshold: irrigate when depletion exceeds 50% TAW (p = 0.5)
            p_threshold = float(
                management_data.get("depletion_fraction", 0.5)
                if isinstance(management_data, dict)
                else 0.5
            )
            rain_effectiveness = 0.75  # fraction of rainfall available to roots

            soil_moisture = theta_fc  # start at field capacity
            cumulative_et = 0.0
            cumulative_irr = 0.0
            cumulative_rain = 0.0

            for day in range(season_days):
                et_d = (
                    float(daily_et.iloc[day])
                    if hasattr(daily_et, "iloc")
                    else float(daily_et)
                )
                rain_d = (
                    float(daily_precip.iloc[day])
                    if hasattr(daily_precip, "iloc")
                    else 0.0
                )

                # Water stress coefficient (FAO-56 Eq. 84)
                depletion = theta_fc - soil_moisture  # m³/m³
                dr_mm = depletion * 1000 * root_depth  # root zone depletion in mm
                ks = (
                    min(1.0, max(0.0, (taw - dr_mm) / ((1 - p_threshold) * taw)))
                    if taw > 0
                    else 1.0
                )

                actual_et = ks * et_d
                effective_rain = rain_d * rain_effectiveness

                # Irrigation if depletion exceeds threshold
                irr_d = 0.0
                if soil_moisture < theta_fc - p_threshold * (theta_fc - theta_pwp):
                    irr_d = (theta_fc - soil_moisture) * 1000 * root_depth  # mm

                # Update soil moisture
                soil_moisture = min(
                    theta_fc,
                    soil_moisture
                    + (effective_rain + irr_d) / (1000 * root_depth)
                    - actual_et / (1000 * root_depth),
                )
                soil_moisture = max(theta_pwp, soil_moisture)

                cumulative_et += actual_et
                cumulative_irr += irr_d
                cumulative_rain += effective_rain

            # Field-level aggregation
            water_requirement = np.full(num_fields, cumulative_et)
            irrigation_vals = np.full(num_fields, cumulative_irr)
            rainfall_vals = np.full(num_fields, cumulative_rain)

            water_requirement_m3 = water_requirement * field_data["area_ha"] * 10
            irrigation_m3 = irrigation_vals * field_data["area_ha"] * 10

            water_balance = {}
            for component in self.water_balance_components:
                if component == "evapotranspiration":
                    water_balance[component] = water_requirement
                elif component == "precipitation":
                    water_balance[component] = rainfall_vals
                elif component == "irrigation":
                    water_balance[component] = irrigation_vals
                elif component == "runoff":
                    # Simplified SCS Curve Number estimate (CN=75 for moderate conditions)
                    cn = float(
                        management_data.get("curve_number", 75)
                        if isinstance(management_data, dict)
                        else 75
                    )
                    s = 25400 / cn - 254  # mm
                    total_rain = (
                        float(daily_precip.sum())
                        if hasattr(daily_precip, "sum")
                        else 0.0
                    )
                    runoff_mm = (
                        max(0, (total_rain - 0.2 * s) ** 2 / (total_rain + 0.8 * s))
                        if total_rain > 0.2 * s
                        else 0.0
                    )
                    water_balance[component] = np.full(num_fields, runoff_mm)
                elif component == "drainage":
                    # Drainage = precipitation + irrigation - ET - runoff (simple balance)
                    runoff_mm = (
                        float(water_balance.get("runoff", np.zeros(1))[0])
                        if "runoff" in water_balance
                        else 0.0
                    )
                    drainage = max(
                        0, cumulative_rain + cumulative_irr - cumulative_et - runoff_mm
                    )
                    water_balance[component] = np.full(num_fields, drainage)

            # Add predictions to results
            result = {
                "water_requirement_mm": water_requirement,
                "water_requirement_m3": water_requirement_m3,
                "water_balance": water_balance,
                "spatial_results": {"water_requirement_mm": water_requirement},
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "model_type": self.model_type,
                    "crop_type": self.crop_type,
                    "water_balance_components": self.water_balance_components,
                },
            }

            # Add water balance components to spatial results
            for component, values in water_balance.items():
                result["spatial_results"][f"{component}_mm"] = values

            # Calculate summary statistics
            result["summary"] = {
                "mean_water_requirement_mm": float(np.mean(water_requirement)),
                "total_water_requirement_m3": float(np.sum(water_requirement_m3)),
            }

            # Add water balance summary
            for component, values in water_balance.items():
                result["summary"][f"mean_{component}_mm"] = float(np.mean(values))

        return result

    def _calculate_reference_et(
        self, weather_data: pd.DataFrame
    ) -> float | np.ndarray | pd.Series:
        """
        Calculate reference evapotranspiration using simplified Penman-Monteith.

        Args:
            weather_data: DataFrame with required weather variables

        Returns:
            Reference ET values

        Note:
            This is a simplified implementation. A real model would use the full
            FAO-56 Penman-Monteith equation and properly handle all required inputs.
        """
        # Extract required variables
        temp = weather_data["temperature"]  # Mean temperature in °C
        solar = weather_data["solar_radiation"]  # Solar radiation in MJ/m²/day
        humidity = weather_data["humidity"]  # Relative humidity in %
        wind = weather_data["wind_speed"]  # Wind speed in m/s

        # Simplified ET calculation
        # In reality, this would be the full FAO-56 Penman-Monteith equation
        # This is a very basic approximation
        reference_et = 0.0023 * (temp + 17.8) * np.sqrt(temp) * solar * 0.408

        # Apply wind and humidity adjustments
        # These are highly simplified approximations
        wind_factor = 0.8 + 0.2 * np.clip(wind / 5.0, 0, 1)
        humidity_factor = 0.6 + 0.4 * (1 - np.clip(humidity / 100.0, 0, 1))

        reference_et = reference_et * wind_factor * humidity_factor

        return reference_et

    def calculate_water_footprint(
        self, result: dict[str, Any], yield_data: pd.Series | None = None
    ) -> dict[str, float | np.ndarray | pd.Series]:
        """
        Calculate water footprint metrics from water usage results.

        Args:
            result: Result dictionary from predict method
            yield_data: Optional crop yield data (t/ha)

        Returns:
            Dictionary of water footprint metrics
        """
        if "water_requirement_mm" not in result:
            raise ValueError("Water requirement data missing from results")

        # Get water requirement components
        water_requirement = result["water_requirement_mm"]

        # Get rainfall and irrigation if available
        blue_water = None
        green_water = None

        if "irrigation_requirement_mm" in result:
            # Blue water is irrigation water
            blue_water = result["irrigation_requirement_mm"]

        if "effective_rainfall_mm" in result:
            # Green water is rainfall
            green_water = result["effective_rainfall_mm"]

        # If components aren't available, estimate them
        if blue_water is None and green_water is None:
            # Assume 60% green water (rainfall), 40% blue water (irrigation)
            # This is a simplification - actual values would depend on climate and crop
            green_water = water_requirement * 0.6
            blue_water = water_requirement * 0.4

        if blue_water is None:
            blue_water = water_requirement * 0.4
        if green_water is None:
            green_water = water_requirement * 0.6

        # Calculate total water footprint (m³/ha)
        water_footprint_volumetric = blue_water + green_water

        # Calculate water footprint per unit yield (m³/t) if yield data is available
        water_footprint_yield = None

        if yield_data is not None:
            # Convert mm to m³/ha (1 mm over 1 ha = 10 m³)
            water_footprint_yield = water_footprint_volumetric * 10 / yield_data

        # Prepare water footprint results
        footprint_results = {
            "water_footprint_mm": water_footprint_volumetric,
            "blue_water_mm": blue_water,
            "green_water_mm": green_water,
            "blue_water_fraction": blue_water / water_footprint_volumetric,
        }

        if water_footprint_yield is not None:
            footprint_results["water_footprint_m3_per_ton"] = water_footprint_yield

        return footprint_results

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        import joblib

        # Create a dictionary with all model components
        model_data = {
            "metadata": self.metadata,
            "crop_type": self.crop_type,
            "model_type": self.model_type,
            "water_balance_components": self.water_balance_components,
            "required_inputs": self.required_inputs,
            "fitted": self.fitted,
            "config": self.config,
            "crop_coefficients": self.crop_coefficients,
        }

        # Add model-type specific components
        if self.model_type == "statistical" and self.fitted:
            model_data["predictor"] = self.predictor
            model_data["feature_columns"] = self.feature_columns

        # Serialize to bytes and write an authenticated GISP1 envelope
        buffer = io.BytesIO()
        joblib.dump(model_data, buffer)
        write_signed_payload(path, buffer.getvalue())

    @classmethod
    def load(cls, path: str) -> "WaterUsageModel":
        """
        Load a model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model instance
        """
        import joblib

        # Verify the GISP1 envelope, then load the model data
        verified = read_verified_payload(path)
        model_data = joblib.load(io.BytesIO(verified))

        # Create a new instance
        model = cls(
            crop_type=model_data["crop_type"],
            model_type=model_data["model_type"],
            water_balance_components=model_data["water_balance_components"],
            config=model_data["config"],
        )

        # Restore model attributes
        model.fitted = model_data["fitted"]
        model.crop_coefficients = model_data["crop_coefficients"]

        # Restore model-type specific components
        if model.model_type == "statistical" and model.fitted:
            model.predictor = model_data["predictor"]
            model.feature_columns = model_data["feature_columns"]

        return model
