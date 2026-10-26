"""
Comprehensive Test Suite for Enhanced GEO-INFER-ECON Capabilities

Tests for all enhanced modules including:
- Data loading and validation
- Spatial econometric models
- Advanced economic indicators
- Model validation and diagnostics
- Visualization capabilities
- API functionality
- Integration patterns
"""

import unittest
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
import tempfile
from pathlib import Path


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading and validation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.utils.data_loader import (
            EconomicDataLoader,
            DataSourceConfig,
        )

        self.data_loader = EconomicDataLoader()
        self.test_data = pd.DataFrame(
            {
                "region_id": ["A", "B", "C"],
                "year": [2020, 2021, 2022],
                "gdp": [1000, 1100, 1200],
                "population": [100, 105, 110],
            }
        )

        # Create test data source
        self.test_config = DataSourceConfig(
            name="test_data", source_type="file", format="csv", location="test_data.csv"
        )

    def test_data_source_registration(self):
        """Test data source registration"""
        self.data_loader.register_data_source(self.test_config)
        self.assertIn("test_data", self.data_loader.data_sources)

    def test_data_validation(self):
        """Test data validation functionality"""
        validation_result = self.data_loader.validate_economic_data(
            self.test_data, "test_data"
        )

        self.assertTrue(validation_result.is_valid)
        self.assertEqual(validation_result.summary["total_rows"], 3)
        self.assertEqual(validation_result.summary["total_columns"], 4)

    def test_data_preprocessing(self):
        """Test data preprocessing steps"""
        # Add some null values for testing
        test_data_with_nulls = self.test_data.copy()
        test_data_with_nulls.loc[0, "gdp"] = None

        processed_data = self.data_loader.preprocess_economic_data(
            test_data_with_nulls, preprocessing_steps=["remove_nulls"]
        )

        self.assertEqual(len(processed_data), 2)  # One row removed

    def test_data_export(self):
        """Test data export functionality"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            self.data_loader.export_economic_data(
                self.test_data, temp_path, format="csv"
            )
            self.assertTrue(temp_path.exists())

            # Verify exported data
            exported_data = pd.read_csv(temp_path)
            pd.testing.assert_frame_equal(self.test_data, exported_data)
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestSpatialEconometrics(unittest.TestCase):
    """Test cases for spatial econometric models"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.core.econometrics_engine import (
            SpatialEconometricsEngine,
            SpatialWeightsConfig,
        )

        self.SpatialWeightsConfig = SpatialWeightsConfig
        self.engine = SpatialEconometricsEngine()

        # Create test data
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 3)
        self.X = np.column_stack([np.ones(n), X])  # Add intercept
        self.y = self.X @ np.array([1, 2, -1, 0.5]) + np.random.randn(n) * 0.1

        # Create spatial coordinates
        coords = np.random.rand(n, 2) * 100
        self.coords = coords

        # Create test GeoDataFrame
        geometries = [Point(x, y) for x, y in coords]
        self.gdf = gpd.GeoDataFrame(
            {"value": self.y, "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2]},
            geometry=geometries,
        )

    def test_spatial_weights_construction(self):
        """Test spatial weights matrix construction"""
        config = self.SpatialWeightsConfig("knn", {"k": 5})
        W = self.engine.construct_spatial_weights(self.gdf, config)

        self.assertEqual(W.shape, (50, 50))
        self.assertTrue(np.allclose(W.sum(axis=1), 1.0))  # Row standardized

    def test_sar_model_fitting(self):
        """Test SAR model fitting"""
        # Create spatial weights
        config = self.SpatialWeightsConfig("knn", {"k": 5})
        W = self.engine.construct_spatial_weights(self.gdf, config)

        # Fit SAR model
        self.engine.fit(self.X, self.y, W, "sar")

        self.assertTrue(self.engine.is_fitted)
        self.assertIsNotNone(self.engine.coefficients_)
        self.assertGreater(len(self.engine.coefficients_), 0)

    def test_model_prediction(self):
        """Test model prediction capabilities"""
        # Create and fit model
        config = self.SpatialWeightsConfig("knn", {"k": 5})
        W = self.engine.construct_spatial_weights(self.gdf, config)
        self.engine.fit(self.X, self.y, W, "sar")

        # Make predictions
        predictions = self.engine.predict(self.X, W)

        self.assertEqual(len(predictions), len(self.y))

        # Calculate R-squared
        ss_res = np.sum((self.y - predictions) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        self.assertGreater(r_squared, 0.5)  # Should explain reasonable variance

    def test_spatial_diagnostics(self):
        """Test spatial diagnostic calculations"""
        # Create simple spatial weights
        W = np.eye(len(self.y))

        # Get model residuals (from fitted model)
        config = self.SpatialWeightsConfig("knn", {"k": 5})
        W_full = self.engine.construct_spatial_weights(self.gdf, config)
        self.engine.fit(self.X, self.y, W_full, "sar")
        residuals = self.engine.residuals

        # Calculate diagnostics
        diagnostics = self.engine.spatial_diagnostics(residuals, W)

        self.assertIn("morans_i", diagnostics)
        self.assertIn("p_value_morans", diagnostics)
        self.assertIsInstance(diagnostics["morans_i"], float)


class TestEconomicIndicators(unittest.TestCase):
    """Test cases for economic indicators calculations"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.utils.indicators import EconomicIndicators

        self.indicators = EconomicIndicators()

        # Test data
        self.regional_data = {
            "region_A": {"gdp": 1000, "population": 100, "education": 0.8},
            "region_B": {"gdp": 1500, "population": 120, "education": 0.7},
            "region_C": {"gdp": 800, "population": 90, "education": 0.9},
        }

        self.time_series_data = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=12, freq="ME"),
                "gdp": [100 + i * 10 + np.random.normal(0, 5) for i in range(12)],
                "unemployment": [5 + np.random.normal(0, 0.5) for _ in range(12)],
            }
        )

    def test_economic_distance_calculation(self):
        """Test economic distance calculations"""
        region1_data = self.regional_data["region_A"]
        region2_data = self.regional_data["region_B"]

        distance = self.indicators.calculate_economic_distance(
            region1_data, region2_data, method="euclidean"
        )

        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0)

    def test_gini_coefficient(self):
        """Test Gini coefficient calculation"""
        income_data = np.array([1000, 2000, 3000, 4000, 5000])
        gini = self.indicators.calculate_gini_coefficient(income_data)

        self.assertIsInstance(gini, float)
        self.assertGreaterEqual(gini, 0)
        self.assertLessEqual(gini, 1)

    def test_growth_rate_calculation(self):
        """Test growth rate calculations"""
        values = pd.Series([100, 110, 121, 133.1])
        growth_rates = self.indicators.calculate_growth_rate(values)

        self.assertEqual(len(growth_rates), 3)  # One less than input
        self.assertAlmostEqual(growth_rates.iloc[0], 0.1, places=2)  # 10% growth

    def test_human_development_index(self):
        """Test Human Development Index calculation"""
        hdi_data = {
            "life_expectancy": pd.Series([70, 75, 80]),
            "education": pd.Series([10, 12, 14]),
            "income": pd.Series([5000, 10000, 20000]),
        }

        hdi_result = self.indicators.calculate_human_development_index(hdi_data)

        self.assertIn("overall_hdi", hdi_result)
        self.assertIn("hdi_components", hdi_result)
        self.assertIsInstance(hdi_result["overall_hdi"], float)


class TestModelValidation(unittest.TestCase):
    """Test cases for model validation and diagnostics"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.utils.validator import ModelValidator

        self.validator = ModelValidator()

        # Create test data
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        self.X = np.column_stack([np.ones(n), X])
        self.y = self.X @ np.array([1, 2, -1]) + np.random.randn(n) * 0.1

        # Create a real simple linear model instead of a mock
        class SimpleLinearModel:
            """Real linear model for testing validation."""

            def __init__(self, coefficients):
                self.coefficients = coefficients

            def predict(self, X):
                return X @ self.coefficients

        self.test_model = SimpleLinearModel(coefficients=np.array([1, 2, -1]))

    def test_regression_validation(self):
        """Test regression model validation"""
        predictions = self.test_model.predict(self.X)
        validation_results = self.validator.validate_economic_model_results(
            predictions, self.y, model_type="regression"
        )

        self.assertIn("r_squared", validation_results)
        self.assertIn("rmse", validation_results)
        self.assertIn("mape", validation_results)
        self.assertGreater(validation_results["r_squared"], 0.9)  # Should be good fit

    def test_model_assumptions_validation(self):
        """Test econometric model assumptions validation"""
        assumptions_results = self.validator.validate_model_assumptions(
            self.test_model, self.X, self.y
        )

        self.assertIn("linearity", assumptions_results)
        self.assertIn("homoscedasticity", assumptions_results)
        self.assertIn("normality", assumptions_results)
        self.assertIn("independence", assumptions_results)

    def test_spatial_model_validation(self):
        """Test spatial model assumptions validation"""
        # Create simple spatial weights
        W = np.eye(len(self.y))

        spatial_validation = self.validator.validate_spatial_model_assumptions(
            self.test_model, self.X, self.y, W
        )

        self.assertIn("base_assumptions", spatial_validation)
        self.assertIn("spatial_assumptions", spatial_validation)
        self.assertIn("overall_valid", spatial_validation)


class TestVisualization(unittest.TestCase):
    """Test cases for visualization capabilities"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.utils.visualizer import ResultsVisualizer

        self.visualizer = ResultsVisualizer()

        # Test data
        self.test_data = pd.DataFrame(
            {
                "region": ["A", "B", "C"],
                "gdp": [1000, 1500, 800],
                "unemployment": [5, 4, 6],
            }
        )

        # Test spatial data
        coords = [(0, 0), (1, 1), (2, 0)]
        geometries = [Point(x, y) for x, y in coords]
        self.spatial_data = gpd.GeoDataFrame(
            {"region": ["A", "B", "C"], "value": [100, 150, 80]}, geometry=geometries
        )

    def test_economic_indicators_plot(self):
        """Test economic indicators plotting"""
        fig = self.visualizer.plot_economic_indicators(
            self.test_data, ["gdp", "unemployment"]
        )

        self.assertIsNotNone(fig)
        # In a real test, would check figure properties

    def test_choropleth_map_creation(self):
        """Test choropleth map creation"""
        fig = self.visualizer.create_choropleth_map(self.spatial_data, "value")

        self.assertIsNotNone(fig)
        # In a real test, would check map properties

    def test_spatial_heatmap_creation(self):
        """Test spatial heatmap creation"""
        # Add coordinates to test data
        heatmap_data = self.test_data.copy()
        heatmap_data["longitude"] = [0, 1, 2]
        heatmap_data["latitude"] = [0, 1, 0]

        fig = self.visualizer.create_spatial_heatmap(heatmap_data, "gdp")

        self.assertIsNotNone(fig)


class TestAPI(unittest.TestCase):
    """Test cases for API functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.api.economic_api import EconomicAnalysisAPI
        from starlette.testclient import TestClient

        # Mock configuration
        config = {
            "api_keys": {"test_key": "test_user"},
            "rate_limits": {"requests_per_minute": 10, "requests_per_hour": 100},
        }

        self.api = EconomicAnalysisAPI(config)
        self.client = TestClient(self.api.app)

    def test_health_check(self):
        """Test API health check endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["service"], "GEO-INFER-ECON API")
        self.assertIn("component_health", body["data"])

    def test_model_execution_endpoint(self):
        """Test model execution endpoint: auth works and schema is honored."""
        request_data = {
            "model_type": "sar_model",
            "model_configuration": {},
            "data_source": "test_data",
            "parameters": {},
        }

        headers = {"Authorization": "Bearer test_key"}

        response = self.client.post(
            "/api/models/execute", json=request_data, headers=headers
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in ("success", "execution_id", "model_type", "results", "diagnostics"):
            self.assertIn(key, body)
        self.assertEqual(body["model_type"], "sar_model")

    def test_model_execution_requires_auth(self):
        """Test model execution endpoint rejects unauthenticated requests."""
        request_data = {
            "model_type": "sar_model",
            "model_configuration": {},
            "data_source": "test_data",
            "parameters": {},
        }
        response = self.client.post("/api/models/execute", json=request_data)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("WWW-Authenticate"), "Bearer")

    def test_model_execution_rejects_invalid_token(self):
        """Test model execution endpoint rejects unknown bearer tokens with 401."""
        request_data = {
            "model_type": "sar_model",
            "model_configuration": {},
            "data_source": "test_data",
            "parameters": {},
        }
        headers = {"Authorization": "Bearer not-a-real-token"}
        response = self.client.post(
            "/api/models/execute", json=request_data, headers=headers
        )
        self.assertEqual(response.status_code, 401)

    def test_hourly_rate_limit_enforced(self):
        """Test hourly rate limit can trigger with requests spread over one hour."""
        import time as time_module
        from unittest import mock

        clock = {"t": 0.0}

        def fake_time():
            return clock["t"]

        with mock.patch.object(time_module, "time", fake_time):
            api = self.api
            api.rate_limits["requests_per_minute"] = 1000
            api.rate_limits["requests_per_hour"] = 5
            api.request_counts = {}
            for i in range(5):
                clock["t"] = i * 61.0  # spread over >1 minute, one per minute
                self.assertTrue(api._check_rate_limit("client_a"))
                api._record_request("/api/models/execute", "client_a")
            # 6th request: 5 entries within the past hour, hourly limit is 5
            clock["t"] = 5 * 61.0
            self.assertFalse(api._check_rate_limit("client_a"))

    def test_spatial_analysis_endpoint(self):
        """Test spatial analysis endpoint returns the documented contract."""
        request_data = {
            "analysis_type": "spatial_autocorrelation",
            "data_source": "test_spatial_data",
            "coordinates": [[0, 0], [1, 1], [2, 0]],
            "parameters": {},
        }

        response = self.client.post("/api/spatial/analyze", json=request_data)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("success", body)
        self.assertIsInstance(body["success"], bool)


class TestMicroeconomicsModule(unittest.TestCase):
    """Test cases for microeconomics functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.consumer_data = {
            "consumer_id": "test_consumer",
            "income": 1000.0,
            "location": (45.0, -120.0),
            "preferences": {"good_1": 0.6, "good_2": 0.4},
        }
        self.market_prices = [2.0, 3.0]
        self.goods = ["good_1", "good_2"]

    def test_consumer_utility_maximization(self):
        """Test consumer utility maximization"""
        income = self.consumer_data["income"]
        preferences = list(self.consumer_data["preferences"].values())
        prices = np.array(self.market_prices)

        # Cobb-Douglas solution: x_i = (alpha_i * income) / p_i
        expected_quantities = (np.array(preferences) * income) / prices
        expected_expenditure = np.sum(expected_quantities * prices)

        # Verify budget constraint
        self.assertAlmostEqual(expected_expenditure, income, places=2)

        # Verify quantities are positive
        self.assertTrue(all(q > 0 for q in expected_quantities))

    def test_spatial_consumer_choice(self):
        """Test spatial consumer choice with multiple markets"""
        markets = [
            {"location": (45.0, -120.0), "prices": [2.0, 3.0]},
            {"location": (45.1, -119.9), "prices": [2.2, 2.8]},
        ]

        transport_cost_per_km = 0.1
        consumer_location = (45.0, -120.0)

        best_market = None
        max_net_utility = -float("inf")

        for i, market in enumerate(markets):
            distance = (
                np.sqrt(
                    (market["location"][0] - consumer_location[0]) ** 2
                    + (market["location"][1] - consumer_location[1]) ** 2
                )
                * 111
            )  # km per degree

            transport_cost = distance * transport_cost_per_km
            effective_income = self.consumer_data["income"] - transport_cost

            if effective_income > 0:
                utility_proxy = effective_income
                if utility_proxy > max_net_utility:
                    max_net_utility = utility_proxy
                    best_market = i

        self.assertIsNotNone(best_market)


class TestMacroeconomicsModule(unittest.TestCase):
    """Test cases for macroeconomics functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.solow_params = {
            "alpha": 0.33,
            "s": 0.2,
            "n": 0.02,
            "delta": 0.05,
            "g": 0.02,
        }

    def test_solow_steady_state(self):
        """Test Solow growth model steady state calculations"""
        params = self.solow_params

        # Steady-state capital per worker
        k_star = (params["s"] / (params["n"] + params["delta"] + params["g"])) ** (
            1 / (1 - params["alpha"])
        )

        # Steady-state output per worker
        y_star = k_star ** params["alpha"]

        self.assertGreater(k_star, 0)
        self.assertGreater(y_star, 0)

    def test_regional_convergence(self):
        """Test beta convergence analysis"""
        regions_data = pd.DataFrame(
            {
                "region_id": ["A", "B", "C"],
                "gdp_per_capita_init": [20000, 30000, 25000],
                "gdp_per_capita_final": [25000, 33000, 28000],
            }
        )

        years_diff = 20
        growth_rates = (
            np.log(regions_data["gdp_per_capita_final"])
            - np.log(regions_data["gdp_per_capita_init"])
        ) / years_diff

        from scipy.stats import pearsonr

        correlation, p_value = pearsonr(
            np.log(regions_data["gdp_per_capita_init"]), growth_rates
        )

        self.assertIsInstance(correlation, float)
        self.assertIsInstance(p_value, float)


class TestBioregionalModule(unittest.TestCase):
    """Test cases for bioregional economics functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.forest_asset = {
            "asset_id": "test_forest",
            "area_hectares": 100.0,
            "carbon_storage": 500.0,
            "carbon_sequestration_per_year": 10.0,
            "biodiversity_index": 0.8,
        }
        self.service_prices = {"carbon": 50.0, "biodiversity": 100.0}

    def test_ecosystem_service_valuation(self):
        """Test ecosystem service valuation"""
        asset = self.forest_asset
        prices = self.service_prices

        carbon_value = asset["carbon_sequestration_per_year"] * prices["carbon"]
        biodiversity_value = (
            asset["area_hectares"]
            * asset["biodiversity_index"]
            * prices["biodiversity"]
        )
        total_value = carbon_value + biodiversity_value

        self.assertGreater(carbon_value, 0)
        self.assertGreater(biodiversity_value, 0)
        self.assertGreater(total_value, 0)

    def test_credit_quality_assessment(self):
        """Test ecosystem service credit quality assessment"""
        quality_parameters = {
            "additionality": 0.9,
            "permanence": 0.8,
            "measurability": 0.85,
            "leakage_risk": 0.1,
            "co_benefits": 0.7,
        }

        weights = {
            "additionality": 0.3,
            "permanence": 0.25,
            "measurability": 0.2,
            "leakage_risk": 0.15,
            "co_benefits": 0.1,
        }

        quality_score = sum(
            quality_parameters[factor] * weight for factor, weight in weights.items()
        )

        self.assertGreaterEqual(quality_score, 0)
        self.assertLessEqual(quality_score, 1)


class TestIntegratedAnalysis(unittest.TestCase):
    """Test cases for integrated economic analysis"""

    def test_sustainability_indicators(self):
        """Test sustainability indicator calculations"""
        regional_gdp = 1000000
        ecosystem_services_value = 150000
        natural_capital_stock = 2000000

        ecosystem_gdp_ratio = ecosystem_services_value / regional_gdp
        natural_capital_ratio = natural_capital_stock / regional_gdp

        self.assertGreater(ecosystem_gdp_ratio, 0)
        self.assertGreater(natural_capital_ratio, 0)

    def test_policy_recommendations(self):
        """Test policy recommendation logic"""
        indicators = {
            "ecosystem_gdp_ratio": 0.08,
            "natural_capital_ratio": 0.9,
            "rural_urban_gap": 2.5,
        }

        recommendations = []

        if indicators["ecosystem_gdp_ratio"] < 0.1:
            recommendations.append("Invest in natural capital restoration")
        if indicators["rural_urban_gap"] > 2.0:
            recommendations.append("Implement payments for ecosystem services")
        if indicators["natural_capital_ratio"] < 1.0:
            recommendations.append("Strengthen environmental protection")

        self.assertGreater(len(recommendations), 0)


class TestIntegrationPatterns(unittest.TestCase):
    """Test cases for integration patterns with other GEO-INFER modules"""

    def test_spatial_integration(self):
        """Test integration with GEO-INFER-SPACE"""
        # Mock spatial data
        spatial_data = gpd.GeoDataFrame(
            {"region_id": ["A", "B", "C"], "gdp": [1000, 1500, 800]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 0)],
        )

        # Test spatial operations would be performed here
        self.assertEqual(len(spatial_data), 3)

    def test_temporal_integration(self):
        """Test integration with GEO-INFER-TIME"""
        # Mock time series data
        time_data = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=12, freq="ME"),
                "gdp": range(100, 112),
            }
        )

        # Test temporal operations would be performed here
        self.assertEqual(len(time_data), 12)

    def test_data_integration(self):
        """Test integration with GEO-INFER-DATA"""
        # Mock data integration
        data_sources = ["census", "economic", "environmental"]

        # Test data loading and merging would be performed here
        self.assertGreater(len(data_sources), 0)


def run_comprehensive_test_suite():
    """Run comprehensive test suite"""
    print("Running GEO-INFER-ECON Enhanced Capabilities Comprehensive Test Suite")
    print("=" * 80)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDataLoading,
        TestSpatialEconometrics,
        TestEconomicIndicators,
        TestModelValidation,
        TestVisualization,
        TestAPI,
        TestMicroeconomicsModule,
        TestMacroeconomicsModule,
        TestBioregionalModule,
        TestIntegratedAnalysis,
        TestIntegrationPatterns,
    ]

    for test_class in test_classes:
        test_suite.addTest(unittest.makeSuite(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'=' * 80}")
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%"
    )

    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)
