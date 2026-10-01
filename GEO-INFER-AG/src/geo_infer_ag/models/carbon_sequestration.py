"""
Carbon sequestration modeling for agricultural lands.
"""

import io
from typing import Any
import logging
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

from geo_infer_ag.models.base import (
    AgricultureModel,
    read_verified_payload,
    write_signed_payload,
)

logger = logging.getLogger(__name__)


class CarbonSequestrationModel(AgricultureModel):
    """
    Model for predicting carbon sequestration in agricultural soils and biomass.

    This model estimates carbon sequestration potential of agricultural fields
    based on crop types, management practices, soil properties, and climate conditions.

    Attributes:
        model_type: Type of underlying model ('tier1', 'tier2', 'process_based')
        time_horizon: Time horizon in years for sequestration projections
        carbon_pools: List of carbon pools to model (e.g., soil, biomass)
    """

    def __init__(
        self,
        model_type: str = "tier1",
        time_horizon: int = 20,
        carbon_pools: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the carbon sequestration model.

        Args:
            model_type: Type of modeling approach ('tier1', 'tier2', 'process_based')
            time_horizon: Years to project carbon sequestration
            carbon_pools: Optional list of carbon pools to model
            config: Optional configuration parameters
        """
        name = f"carbon_sequestration_{model_type}"
        super().__init__(name=name, config=config)

        self.model_type = model_type
        self.time_horizon = time_horizon
        self.predictor = None
        self.fitted = False

        # Default carbon pools if not provided
        if carbon_pools is None:
            self.carbon_pools = ["soil_carbon", "biomass_carbon"]
        else:
            self.carbon_pools = carbon_pools

        # Define required inputs based on model type
        if model_type == "tier1":
            self.required_inputs = ["field_data"]
        elif model_type == "tier2":
            self.required_inputs = ["field_data", "soil_data", "management_data"]
        elif model_type == "process_based":
            self.required_inputs = [
                "field_data",
                "soil_data",
                "weather_data",
                "management_data",
            ]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Update metadata
        self.metadata.update(
            {
                "model_type": model_type,
                "time_horizon": time_horizon,
                "carbon_pools": self.carbon_pools,
            }
        )

        # Initialize default sequestration rates by crop type (tonnes C/ha/year)
        # These are example values - actual rates would depend on many factors
        self.default_rates = {
            "corn": {"soil_carbon": 0.2, "biomass_carbon": 0.3},
            "wheat": {"soil_carbon": 0.15, "biomass_carbon": 0.2},
            "rice": {"soil_carbon": 0.1, "biomass_carbon": 0.3},
            "soybean": {"soil_carbon": 0.3, "biomass_carbon": 0.2},
            "cotton": {"soil_carbon": 0.1, "biomass_carbon": 0.2},
            "alfalfa": {"soil_carbon": 0.4, "biomass_carbon": 0.3},
            "grassland": {"soil_carbon": 0.5, "biomass_carbon": 0.2},
            "forest": {"soil_carbon": 0.7, "biomass_carbon": 1.2},
            "cover_crop": {"soil_carbon": 0.4, "biomass_carbon": 0.2},
        }

        # Initialize management practice modifiers
        self.practice_modifiers = {
            "no_till": {"soil_carbon": 1.3, "biomass_carbon": 1.0},
            "reduced_till": {"soil_carbon": 1.15, "biomass_carbon": 1.0},
            "conventional_till": {"soil_carbon": 0.9, "biomass_carbon": 1.0},
            "cover_crops": {"soil_carbon": 1.25, "biomass_carbon": 1.1},
            "crop_rotation": {"soil_carbon": 1.1, "biomass_carbon": 1.05},
            "organic_fertilizer": {"soil_carbon": 1.15, "biomass_carbon": 1.1},
            "inorganic_fertilizer": {"soil_carbon": 1.05, "biomass_carbon": 1.15},
            "residue_retention": {"soil_carbon": 1.2, "biomass_carbon": 1.0},
            "residue_removal": {"soil_carbon": 0.8, "biomass_carbon": 1.0},
            "irrigation": {"soil_carbon": 1.05, "biomass_carbon": 1.2},
            "agroforestry": {"soil_carbon": 1.3, "biomass_carbon": 1.5},
        }

        # Initialize soil type modifiers
        self.soil_modifiers = {
            "sandy": {"soil_carbon": 0.8, "biomass_carbon": 1.0},
            "loam": {"soil_carbon": 1.0, "biomass_carbon": 1.0},
            "clay": {"soil_carbon": 1.2, "biomass_carbon": 1.0},
            "organic": {"soil_carbon": 1.3, "biomass_carbon": 1.0},
        }

    def fit(
        self,
        training_data: dict[str, Any],
        target_columns: dict[str, str] | None = None,
        feature_columns: list[str] | None = None,
    ) -> None:
        """
        Train the carbon sequestration model using historical data.

        Args:
            training_data: Dictionary of training data sources
            target_columns: Optional dictionary mapping carbon pools to column names
            feature_columns: Optional list of feature columns to use

        Raises:
            ValueError: If required training data is missing
        """
        # Only tier2 models with statistical components need fitting
        if self.model_type != "tier2":
            self.fitted = True
            return

        # Validate required training data
        if "field_data" not in training_data:
            raise ValueError("Field data required for training")

        # Get training data
        field_data = training_data["field_data"]

        # Default target columns if not provided
        if target_columns is None:
            target_columns = {
                "soil_carbon": "soil_carbon_sequestration",
                "biomass_carbon": "biomass_carbon_sequestration",
            }

        # Check if the required target columns exist
        missing_targets = []
        for pool in self.carbon_pools:
            if (
                pool not in target_columns
                or target_columns[pool] not in field_data.columns
            ):
                missing_targets.append(pool)

        if missing_targets:
            raise ValueError(
                f"Missing target columns for carbon pools: {missing_targets}"
            )

        # Default feature columns if not provided
        if feature_columns is None:
            # Use all numeric columns except target columns
            numeric_cols = field_data.select_dtypes(include=["number"]).columns.tolist()
            target_cols = list(target_columns.values())
            feature_columns = [col for col in numeric_cols if col not in target_cols]

        # Train a separate model for each carbon pool
        self.predictors = {}
        skipped_pools: dict[str, int] = {}

        for pool in self.carbon_pools:
            target_col = target_columns[pool]

            # Initialize a RandomForest model
            predictor = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )

            # Train the model
            X = field_data[feature_columns]
            y = field_data[target_col]

            # Skip pools with insufficient data, recording the skip
            if len(y.dropna()) < 10:
                skipped_pools[pool] = int(len(y.dropna()))
                continue

            predictor.fit(X.loc[y.notna()], y.dropna())

            # Store the trained model
            self.predictors[pool] = {
                "model": predictor,
                "feature_columns": feature_columns,
            }

        if skipped_pools:
            self.metadata["skipped_pools"] = skipped_pools
            for skipped_pool, usable_rows in skipped_pools.items():
                logger.warning(
                    "Tier 2 fit skipped carbon pool '%s': only %d usable "
                    "training rows (< 10 required)",
                    skipped_pool,
                    usable_rows,
                )
        self.fitted = True

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Predict carbon sequestration potential using the model.

        Args:
            data: Dictionary of input data sources

        Returns:
            Dictionary containing carbon sequestration predictions and metadata

        Raises:
            ValueError: If required inputs are missing
        """
        # Validate required inputs
        self.validate_inputs(data)

        # Get field data
        field_data = data["field_data"]

        # Check if crop type column exists
        if "crop_type" not in field_data.columns:
            raise ValueError("Field data must include 'crop_type' column")

        # Generate predictions based on model type
        if self.model_type == "tier1":
            # Tier 1 uses default values based on lookup tables

            # Initialize carbon sequestration arrays
            num_fields = len(field_data)
            sequestration_rates = {
                pool: np.zeros(num_fields) for pool in self.carbon_pools
            }

            # Apply default sequestration rates by crop type
            for i, crop in enumerate(field_data["crop_type"]):
                crop_lower = crop.lower() if isinstance(crop, str) else "grassland"

                # Use default rates or fallback to grassland rates if crop not found
                crop_rates = self.default_rates.get(
                    crop_lower, self.default_rates["grassland"]
                )

                for pool in self.carbon_pools:
                    if pool in crop_rates:
                        sequestration_rates[pool][i] = crop_rates[pool]

            # Apply management practice modifiers if available
            if "management_data" in data:
                management_data = data["management_data"]

                # Check if management data has practices column
                if "practices" in management_data.columns:
                    for i, practices in enumerate(management_data["practices"]):
                        if not practices:
                            continue

                        # Apply modifier for each practice
                        for practice in practices:
                            if practice in self.practice_modifiers:
                                for pool in self.carbon_pools:
                                    if pool in self.practice_modifiers[practice]:
                                        sequestration_rates[pool][i] *= (
                                            self.practice_modifiers[practice][pool]
                                        )

            # Apply soil type modifiers if available
            if "soil_data" in data:
                soil_data = data["soil_data"]

                # Check if soil data has soil type column
                if "soil_type" in soil_data.columns:
                    for i, soil_type in enumerate(soil_data["soil_type"]):
                        if soil_type in self.soil_modifiers:
                            for pool in self.carbon_pools:
                                if pool in self.soil_modifiers[soil_type]:
                                    sequestration_rates[pool][i] *= self.soil_modifiers[
                                        soil_type
                                    ][pool]

            # Calculate total carbon sequestration (tonnes C/ha/year)
            total_sequestration_rate = np.zeros(num_fields)
            for pool in self.carbon_pools:
                total_sequestration_rate += sequestration_rates[pool]

            # Calculate area-based total (tonnes C/year)
            total_sequestration = total_sequestration_rate * field_data["area_ha"]

            # Calculate time horizon projections (tonnes C)
            time_horizon_sequestration = total_sequestration * self.time_horizon

            # Convert to CO2 equivalent (tonnes CO2e)
            # 1 tonne C = 3.67 tonnes CO2e
            co2e_factor = 3.67
            co2e_sequestration = time_horizon_sequestration * co2e_factor

            # Add predictions to results
            result = {
                "sequestration_rates": sequestration_rates,
                "total_sequestration_rate": total_sequestration_rate,
                "total_sequestration": total_sequestration,
                "time_horizon_sequestration": time_horizon_sequestration,
                "co2e_sequestration": co2e_sequestration,
                "spatial_results": {
                    "total_sequestration_rate": total_sequestration_rate,
                    "time_horizon_sequestration": time_horizon_sequestration,
                    "co2e_sequestration": co2e_sequestration,
                },
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "model_type": self.model_type,
                    "carbon_pools": self.carbon_pools,
                    "time_horizon": self.time_horizon,
                },
            }

            # Add individual pool results to spatial results
            for pool in self.carbon_pools:
                result["spatial_results"][f"{pool}_rate"] = sequestration_rates[pool]

            # Calculate summary statistics
            result["summary"] = {
                "mean_sequestration_rate": float(np.mean(total_sequestration_rate)),
                "total_annual_sequestration": float(np.sum(total_sequestration)),
                "total_time_horizon_sequestration": float(
                    np.sum(time_horizon_sequestration)
                ),
                "total_co2e_sequestration": float(np.sum(co2e_sequestration)),
            }

            # Add pool-specific summary statistics
            for pool in self.carbon_pools:
                result["summary"][f"mean_{pool}_rate"] = float(
                    np.mean(sequestration_rates[pool])
                )
                result["summary"][f"total_{pool}_annual"] = float(
                    np.sum(sequestration_rates[pool] * field_data["area_ha"])
                )

        elif self.model_type == "tier2":
            # Tier 2 models use statistical models with more detailed inputs

            # Check if model is fitted
            if (
                not self.fitted
                or not hasattr(self, "predictors")
                or not self.predictors
            ):
                raise ValueError("Tier 2 model must be fitted before prediction")

            # Make predictions for each carbon pool
            sequestration_rates = {}

            for pool, predictor_info in self.predictors.items():
                predictor = predictor_info["model"]
                feature_cols = predictor_info["feature_columns"]

                # Check if all feature columns are available
                missing_features = [
                    col for col in feature_cols if col not in field_data.columns
                ]
                if missing_features:
                    # Fall back to tier 1 method for this pool
                    sequestration_rates[pool] = np.zeros(len(field_data))

                    for i, crop in enumerate(field_data["crop_type"]):
                        crop_lower = (
                            crop.lower() if isinstance(crop, str) else "grassland"
                        )
                        crop_rates = self.default_rates.get(
                            crop_lower, self.default_rates["grassland"]
                        )
                        sequestration_rates[pool][i] = crop_rates.get(pool, 0.0)
                else:
                    # Generate predictions using the trained model
                    X = field_data[feature_cols]
                    sequestration_rates[pool] = predictor.predict(X)

            # Calculate total carbon sequestration (tonnes C/ha/year)
            total_sequestration_rate = np.zeros(len(field_data))
            for pool in self.carbon_pools:
                if pool in sequestration_rates:
                    total_sequestration_rate += sequestration_rates[pool]

            # Calculate area-based total (tonnes C/year)
            total_sequestration = total_sequestration_rate * field_data["area_ha"]

            # Calculate time horizon projections (tonnes C)
            time_horizon_sequestration = total_sequestration * self.time_horizon

            # Convert to CO2 equivalent (tonnes CO2e)
            co2e_factor = 3.67
            co2e_sequestration = time_horizon_sequestration * co2e_factor

            # Add predictions to results
            result = {
                "sequestration_rates": sequestration_rates,
                "total_sequestration_rate": total_sequestration_rate,
                "total_sequestration": total_sequestration,
                "time_horizon_sequestration": time_horizon_sequestration,
                "co2e_sequestration": co2e_sequestration,
                "spatial_results": {
                    "total_sequestration_rate": total_sequestration_rate,
                    "time_horizon_sequestration": time_horizon_sequestration,
                    "co2e_sequestration": co2e_sequestration,
                },
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "model_type": self.model_type,
                    "carbon_pools": list(sequestration_rates.keys()),
                    "time_horizon": self.time_horizon,
                    "skipped_pools": dict(self.metadata.get("skipped_pools", {})),
                },
            }

            # Add individual pool results to spatial results
            for pool in sequestration_rates:
                result["spatial_results"][f"{pool}_rate"] = sequestration_rates[pool]

            # Calculate summary statistics
            result["summary"] = {
                "mean_sequestration_rate": float(np.mean(total_sequestration_rate)),
                "total_annual_sequestration": float(np.sum(total_sequestration)),
                "total_time_horizon_sequestration": float(
                    np.sum(time_horizon_sequestration)
                ),
                "total_co2e_sequestration": float(np.sum(co2e_sequestration)),
            }

            # Add pool-specific summary statistics
            for pool in sequestration_rates:
                result["summary"][f"mean_{pool}_rate"] = float(
                    np.mean(sequestration_rates[pool])
                )
                result["summary"][f"total_{pool}_annual"] = float(
                    np.sum(sequestration_rates[pool] * field_data["area_ha"])
                )

        elif self.model_type == "process_based":
            # Process-based models use detailed biophysical simulations

            # Check for additional required data
            required_process_data = ["soil_data", "weather_data", "management_data"]
            missing_data = [d for d in required_process_data if d not in data]

            if missing_data:
                raise ValueError(
                    f"Missing data for process-based model: {missing_data}"
                )

            soil_data = data["soil_data"]
            weather_data = data["weather_data"]
            management_data = data["management_data"]

            # A real process-based model would implement complex biophysical processes
            # This is a simplified baseline implementation

            # Initialize carbon sequestration arrays
            num_fields = len(field_data)
            sequestration_rates = {
                pool: np.zeros(num_fields) for pool in self.carbon_pools
            }

            # Implement process-based logical approximation based on active inference principles
            # where tier 1 parameters are adapted by dynamic soil and climate states

            # For this mechanistic approximation, we use tier1 approach with dynamic environmental modifiers
            for i, crop in enumerate(field_data["crop_type"]):
                crop_lower = crop.lower() if isinstance(crop, str) else "grassland"
                crop_rates = self.default_rates.get(
                    crop_lower, self.default_rates["grassland"]
                )

                for pool in self.carbon_pools:
                    if pool in crop_rates:
                        # Start with base rate
                        rate = crop_rates[pool]

                        # Apply soil factor (e.g., based on clay content)
                        if "clay_content" in soil_data.columns:
                            clay_factor = (
                                1.0 + (soil_data["clay_content"].iloc[i] - 20) / 100
                            )
                            rate *= np.clip(clay_factor, 0.8, 1.3)

                        # Apply climate factor (e.g., based on temperature and precipitation)
                        if (
                            "temperature" in weather_data.columns
                            and "precipitation" in weather_data.columns
                        ):
                            # Simplified climate impact factor
                            temp = weather_data["temperature"].mean()
                            precip = weather_data["precipitation"].sum()

                            # Optimal conditions: temp around 15°C, precip around 800mm
                            temp_factor = 1.0 - abs(temp - 15) / 30
                            precip_factor = 1.0 - abs(precip - 800) / 1000

                            climate_factor = (
                                0.8 + 0.4 * (temp_factor + precip_factor) / 2
                            )
                            rate *= climate_factor

                        # Apply management factors
                        if "practices" in management_data.columns:
                            practices = management_data["practices"].iloc[i]
                            for practice in practices:
                                if (
                                    practice in self.practice_modifiers
                                    and pool in self.practice_modifiers[practice]
                                ):
                                    rate *= self.practice_modifiers[practice][pool]

                        # Store the calculated rate
                        sequestration_rates[pool][i] = rate

            # Calculate total carbon sequestration (tonnes C/ha/year)
            total_sequestration_rate = np.zeros(num_fields)
            for pool in self.carbon_pools:
                total_sequestration_rate += sequestration_rates[pool]

            # Calculate area-based total (tonnes C/year)
            total_sequestration = total_sequestration_rate * field_data["area_ha"]

            # Calculate time horizon projections (tonnes C)
            # Process models would typically use non-linear accumulation curves
            # This is a simplified linear projection
            time_horizon_sequestration = total_sequestration * self.time_horizon

            # Convert to CO2 equivalent (tonnes CO2e)
            co2e_factor = 3.67
            co2e_sequestration = time_horizon_sequestration * co2e_factor

            # Add predictions to results
            result = {
                "sequestration_rates": sequestration_rates,
                "total_sequestration_rate": total_sequestration_rate,
                "total_sequestration": total_sequestration,
                "time_horizon_sequestration": time_horizon_sequestration,
                "co2e_sequestration": co2e_sequestration,
                "spatial_results": {
                    "total_sequestration_rate": total_sequestration_rate,
                    "time_horizon_sequestration": time_horizon_sequestration,
                    "co2e_sequestration": co2e_sequestration,
                },
                "metadata": {
                    "prediction_time": datetime.now().isoformat(),
                    "model_type": self.model_type,
                    "carbon_pools": self.carbon_pools,
                    "time_horizon": self.time_horizon,
                },
            }

            # Add individual pool results to spatial results
            for pool in self.carbon_pools:
                result["spatial_results"][f"{pool}_rate"] = sequestration_rates[pool]

            # Calculate summary statistics
            result["summary"] = {
                "mean_sequestration_rate": float(np.mean(total_sequestration_rate)),
                "total_annual_sequestration": float(np.sum(total_sequestration)),
                "total_time_horizon_sequestration": float(
                    np.sum(time_horizon_sequestration)
                ),
                "total_co2e_sequestration": float(np.sum(co2e_sequestration)),
            }

            # Add pool-specific summary statistics
            for pool in self.carbon_pools:
                result["summary"][f"mean_{pool}_rate"] = float(
                    np.mean(sequestration_rates[pool])
                )
                result["summary"][f"total_{pool}_annual"] = float(
                    np.sum(sequestration_rates[pool] * field_data["area_ha"])
                )

        return result

    def calculate_carbon_value(
        self, result: dict[str, Any], carbon_price: float = 25.0
    ) -> dict[str, float | np.ndarray]:
        """
        Calculate monetary value of carbon sequestration.

        Args:
            result: Result dictionary from predict method
            carbon_price: Price per tonne of CO2e (default: $25/tCO2e)

        Returns:
            Dictionary with carbon value metrics
        """
        if "co2e_sequestration" not in result:
            raise ValueError("CO2e sequestration data missing from results")

        # Get CO2e sequestration
        co2e_sequestration = result["co2e_sequestration"]

        # Calculate carbon value
        carbon_value = co2e_sequestration * carbon_price

        # Prepare carbon value results
        value_results = {
            "carbon_value": carbon_value,
            "carbon_value_per_ha": carbon_value
            / result["total_sequestration"]
            * result["total_sequestration_rate"],
            "total_carbon_value": float(np.sum(carbon_value)),
            "carbon_price": carbon_price,
        }

        return value_results

    def set_time_horizon(self, years: int) -> None:
        """
        Set the time horizon for carbon sequestration projections.

        Args:
            years: Number of years for projection

        Raises:
            ValueError: If years is not positive
        """
        if years <= 0:
            raise ValueError("Time horizon must be positive")

        self.time_horizon = years
        self.metadata["time_horizon"] = years

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
            "model_type": self.model_type,
            "time_horizon": self.time_horizon,
            "carbon_pools": self.carbon_pools,
            "required_inputs": self.required_inputs,
            "fitted": self.fitted,
            "config": self.config,
            "default_rates": self.default_rates,
            "practice_modifiers": self.practice_modifiers,
            "soil_modifiers": self.soil_modifiers,
        }

        # Add model-type specific components
        if self.model_type == "tier2" and self.fitted and hasattr(self, "predictors"):
            model_data["predictors"] = self.predictors

        # Serialize to bytes and write an authenticated GISP1 envelope
        buffer = io.BytesIO()
        joblib.dump(model_data, buffer)
        write_signed_payload(path, buffer.getvalue())

    @classmethod
    def load(cls, path: str) -> "CarbonSequestrationModel":
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
            model_type=model_data["model_type"],
            time_horizon=model_data["time_horizon"],
            carbon_pools=model_data["carbon_pools"],
            config=model_data["config"],
        )

        # Restore model attributes
        model.fitted = model_data["fitted"]
        model.default_rates = model_data["default_rates"]
        model.practice_modifiers = model_data["practice_modifiers"]
        model.soil_modifiers = model_data["soil_modifiers"]

        # Restore model-type specific components
        if model.model_type == "tier2" and model.fitted and "predictors" in model_data:
            model.predictors = model_data["predictors"]

        return model
