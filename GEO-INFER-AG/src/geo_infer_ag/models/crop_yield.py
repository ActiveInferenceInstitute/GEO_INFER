"""
Crop yield modeling and prediction functionality.
"""

import io
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

from geo_infer_ag.models.base import (
    AgricultureModel,
    read_verified_payload,
    write_signed_payload,
)


class CropYieldModel(AgricultureModel):
    """
    Model for predicting crop yields based on environmental and management factors.

    This model predicts crop yields using a combination of remote sensing data,
    weather data, soil properties, and management practices.

    Attributes:
        crop_type: Type of crop for which yield is predicted
        model_type: Type of underlying model ('statistical', 'machine_learning', 'process_based')
        predictor: The underlying prediction model
    """

    def __init__(
        self,
        crop_type: str,
        model_type: str = "machine_learning",
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the crop yield model.

        Args:
            crop_type: Type of crop (e.g., 'corn', 'wheat', 'soybean')
            model_type: Type of model to use for prediction
            config: Optional configuration parameters
        """
        name = f"{crop_type}_yield_model"
        super().__init__(name=name, config=config)

        self.crop_type = crop_type.lower()
        self.model_type = model_type
        self.predictor: RandomForestRegressor | None = None
        self.fitted = False
        self.feature_columns: list[str] | None = None
        self._statistical_params: dict[str, Any] | None = None
        self._process_params: dict[str, Any] | None = None

        # Define required inputs based on model type
        if model_type == "machine_learning":
            self.required_inputs = ["field_data"]
        elif model_type == "process_based":
            self.required_inputs = [
                "field_data",
                "weather_data",
                "soil_data",
                "management_data",
            ]
        elif model_type == "statistical":
            self.required_inputs = ["field_data", "historical_yield_data"]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Update metadata
        self.metadata.update({"crop_type": crop_type, "model_type": model_type})

    def fit(
        self,
        training_data: dict[str, Any],
        target_column: str = "yield",
        feature_columns: list[str] | None = None,
    ) -> None:
        """
        Train the yield prediction model using historical data.

        Args:
            training_data: Dictionary of training data sources
            target_column: Column name containing yield values
            feature_columns: Optional list of feature columns to use

        Raises:
            ValueError: If required training data is missing
        """
        # Validate required training data
        if "field_data" not in training_data:
            raise ValueError("Field data required for training")

        if (
            "historical_yield_data" not in training_data
            and self.model_type != "machine_learning"
        ):
            raise ValueError(
                "Historical yield data required for statistical and process-based models"
            )

        # Use field data as base training dataset
        train_df = training_data["field_data"]

        # Ensure target column exists
        if target_column not in train_df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in training data"
            )

        # Default feature columns if not provided
        if feature_columns is None:
            # Use all numeric columns except yield and geometry
            numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != target_column]

        # Handle different model types
        if self.model_type == "machine_learning":
            # Initialize and train a RandomForest model
            self.predictor = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )

            # Train the model
            X = train_df[feature_columns]
            y = train_df[target_column]

            self.predictor.fit(X, y)
            self.feature_columns = feature_columns

        elif self.model_type == "statistical":
            # Statistical model: compute historical averages and trend regression
            if target_column in train_df.columns:
                self._statistical_params = {
                    "mean_yield": float(train_df[target_column].mean()),
                    "std_yield": float(train_df[target_column].std()),
                    "trend": None,
                }
                if "year" in train_df.columns and len(train_df) > 1:
                    from numpy.polynomial.polynomial import polyfit

                    coeffs = polyfit(
                        train_df["year"].values, train_df[target_column].values, 1
                    )
                    self._statistical_params["trend"] = float(coeffs[1])

        elif self.model_type == "process_based":
            # Simplified process-based model using growing-degree-day accumulation
            self._process_params = {
                "base_temperature": 10.0,
                "optimal_temperature": 25.0,
                "max_yield_potential": float(train_df[target_column].max())
                if target_column in train_df.columns
                else 10.0,
                "water_stress_coeff": 0.8,
            }

        self.fitted = True

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Predict crop yields using the model.

        Args:
            data: Dictionary of input data sources

        Returns:
            Dictionary containing yield predictions and metadata

        Raises:
            ValueError: If model is not fitted or required inputs are missing
        """
        # Check if model is fitted
        if not self.fitted and self.model_type == "machine_learning":
            raise ValueError("Model must be fitted before prediction")

        # Validate required inputs
        self.validate_inputs(data)

        # Get field data as the base for predictions
        field_data = data["field_data"]

        # Generate predictions based on model type
        if self.model_type == "machine_learning":
            if self.predictor is None:
                raise ValueError("Model must be fitted before prediction")
            feature_cols = self.feature_columns or []
            # Check if all feature columns are available
            missing_features = [
                col for col in feature_cols if col not in field_data.columns
            ]
            if missing_features:
                raise ValueError(f"Missing feature columns: {missing_features}")

            # Generate predictions
            X = field_data[feature_cols]
            predictions = self.predictor.predict(X)

            # Add predictions to results
            result = {
                "predictions": predictions,
                "spatial_results": {"predicted_yield": predictions},
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "crop_type": self.crop_type,
                    "model_type": self.model_type,
                },
            }

            # Calculate summary statistics
            result["summary"] = {
                "mean_yield": float(np.mean(predictions)),
                "min_yield": float(np.min(predictions)),
                "max_yield": float(np.max(predictions)),
                "std_yield": float(np.std(predictions)),
            }

        elif self.model_type == "statistical":
            # Statistical prediction from historical data and fitted parameters
            if "historical_yield_data" not in data:
                raise ValueError(
                    "Historical yield data required for statistical prediction"
                )

            historical_data = data["historical_yield_data"]

            # Compute baseline from fitted params if available, otherwise from historical data
            statistical_params = self._statistical_params
            if statistical_params is not None:
                baseline = statistical_params["mean_yield"]
                trend = statistical_params.get("trend")
            else:
                # Derive baseline from any numeric column in historical data, excluding 'year'
                numeric_cols = historical_data.select_dtypes(
                    include="number"
                ).columns.tolist()
                yield_cols = [c for c in numeric_cols if c != "year"]
                baseline = (
                    float(historical_data[yield_cols].values.mean())
                    if yield_cols
                    else 5.0
                )
                trend = None

            predictions = np.full(len(field_data), baseline)

            # Apply linear trend if year column available in field data
            if trend is not None and "year" in field_data.columns:
                ref_year = (
                    float(historical_data["year"].min())
                    if "year" in historical_data.columns
                    else 2000.0
                )
                year_delta = field_data["year"].values.astype(float) - ref_year
                predictions = predictions + trend * year_delta

            predictions = np.maximum(predictions, 0.0)

            result = {
                "predictions": predictions,
                "spatial_results": {"predicted_yield": predictions},
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "crop_type": self.crop_type,
                    "model_type": self.model_type,
                },
            }

            # Calculate summary statistics
            result["summary"] = {
                "mean_yield": float(np.mean(predictions)),
                "min_yield": float(np.min(predictions)),
                "max_yield": float(np.max(predictions)),
                "std_yield": float(np.std(predictions)),
            }

        elif self.model_type == "process_based":
            # Process-based prediction using Growing Degree Day accumulation and stress factors
            required_process_data = ["weather_data", "soil_data", "management_data"]
            missing_data = [d for d in required_process_data if d not in data]

            if missing_data:
                raise ValueError(
                    f"Missing data for process-based model: {missing_data}"
                )

            process_params = self._process_params
            params = (
                process_params
                if process_params is not None
                else {
                    "base_temperature": 10.0,
                    "optimal_temperature": 25.0,
                    "max_yield_potential": 10.0,
                    "water_stress_coeff": 0.8,
                }
            )

            weather_data = data["weather_data"]
            soil_data = data["soil_data"]
            management_data = data["management_data"]

            base_temp = params["base_temperature"]
            optimal_temp = params["optimal_temperature"]
            max_yield = params["max_yield_potential"]

            # Growing Degree Days accumulation
            if "temperature" in weather_data.columns:
                gdd = float(
                    np.maximum(weather_data["temperature"].values - base_temp, 0).sum()
                )
            else:
                gdd = 1500.0  # Seasonal default

            # Harvest index as sigmoid of GDD relative to ~1800 GDD threshold
            harvest_index = 1.0 / (1 + np.exp(-(gdd - 1800.0) / 300.0))

            # Water stress from total precipitation vs crop requirement
            if "precipitation" in weather_data.columns:
                total_precip = float(weather_data["precipitation"].sum())
                water_satisfaction = float(np.clip(total_precip / 550.0, 0.3, 1.2))
            else:
                water_satisfaction = float(params["water_stress_coeff"])

            # Temperature stress: penalty for deviation from optimal
            if "temperature" in weather_data.columns:
                mean_temp_dev = float(
                    np.abs(weather_data["temperature"].values - optimal_temp).mean()
                )
                temp_stress = float(
                    np.clip(1.0 - mean_temp_dev / optimal_temp, 0.6, 1.0)
                )
            else:
                temp_stress = 1.0

            # Soil quality modifier from organic matter
            soil_modifier = 1.0
            if "organic_matter" in soil_data.columns:
                om_mean = float(soil_data["organic_matter"].mean())
                soil_modifier = float(np.clip(0.7 + om_mean * 0.1, 0.7, 1.3))

            # Management modifier from practice column if present
            management_modifier = 1.0
            if "practice" in management_data.columns:
                practice_boosts = {
                    "no_till": 1.05,
                    "cover_crops": 1.07,
                    "precision": 1.1,
                }
                for practice in management_data["practice"].values:
                    management_modifier *= practice_boosts.get(str(practice), 1.0)

            base_prediction = (
                max_yield
                * harvest_index
                * water_satisfaction
                * soil_modifier
                * temp_stress
                * management_modifier
            )
            predictions = np.maximum(
                np.full(len(field_data), float(base_prediction)), 0.0
            )

            result = {
                "predictions": predictions,
                "spatial_results": {"predicted_yield": predictions},
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "crop_type": self.crop_type,
                    "model_type": self.model_type,
                    "gdd": gdd,
                    "harvest_index": float(harvest_index),
                    "water_satisfaction": float(water_satisfaction),
                },
            }

            # Calculate summary statistics
            result["summary"] = {
                "mean_yield": float(np.mean(predictions)),
                "min_yield": float(np.min(predictions)),
                "max_yield": float(np.max(predictions)),
                "std_yield": float(np.std(predictions)),
            }

        return result

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance for machine learning models.

        Returns:
            Dictionary of feature importances

        Raises:
            ValueError: If model type doesn't support feature importance
        """
        if self.model_type != "machine_learning" or not self.fitted:
            raise ValueError(
                "Feature importance only available for fitted machine learning models"
            )

        if self.predictor is None or not hasattr(
            self.predictor, "feature_importances_"
        ):
            raise ValueError("Current model doesn't provide feature importances")

        # Get feature importances from the model
        importances = self.predictor.feature_importances_

        # Create dictionary mapping features to importance values
        feats = self.feature_columns or []
        importance_dict = dict(zip(feats, importances))

        # Sort by importance (descending)
        sorted_importances = {
            k: v
            for k, v in sorted(
                importance_dict.items(), key=lambda item: item[1], reverse=True
            )
        }

        return sorted_importances

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
            "required_inputs": self.required_inputs,
            "fitted": self.fitted,
            "config": self.config,
        }

        # Add model-type specific components
        if self.model_type == "machine_learning" and self.fitted:
            model_data["predictor"] = self.predictor
            model_data["feature_columns"] = self.feature_columns
        # Serialize to bytes and write an authenticated GISP1 envelope
        buffer = io.BytesIO()
        joblib.dump(model_data, buffer)
        write_signed_payload(path, buffer.getvalue())

    @classmethod
    def load(cls, path: str) -> "CropYieldModel":
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
            config=model_data["config"],
        )

        # Restore model-type specific components
        model.fitted = model_data["fitted"]

        if model.model_type == "machine_learning" and model.fitted:
            model.predictor = model_data["predictor"]
            model.feature_columns = model_data["feature_columns"]

        return model
