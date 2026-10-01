"""Unit tests for the base AgricultureModel class."""

import pytest
from typing import Any

from geo_infer_ag.models.base import AgricultureModel


class ConcreteAgricultureModel(AgricultureModel):
    """Concrete implementation of AgricultureModel for testing."""

    def __init__(
        self,
        name: str = "test_model",
        version: str = "0.1.0",
        config: dict[str, Any] = None,
    ):
        """Initialize the concrete model."""
        super().__init__(name=name, version=version, config=config)
        self.required_inputs = ["field_data", "weather_data"]

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Implement the predict method for testing."""
        self.validate_inputs(data)
        return {"result": "test_prediction", "inputs": list(data.keys())}


class TestAgricultureModel:
    """Test suite for AgricultureModel class."""

    def test_initialization(self):
        """Test initialization of AgricultureModel."""
        # Create a concrete instance
        model = ConcreteAgricultureModel(name="test_model", version="1.0.0")

        assert model.name == "test_model"
        assert model.version == "1.0.0"
        assert isinstance(model.config, dict)
        assert model.required_inputs == ["field_data", "weather_data"]

        # Check metadata
        assert model.metadata["name"] == "test_model"
        assert model.metadata["version"] == "1.0.0"
        assert model.metadata["type"] == "ConcreteAgricultureModel"

    def test_validate_inputs(self):
        """Test input validation."""
        model = ConcreteAgricultureModel()

        # Valid inputs
        valid_data = {"field_data": "dummy", "weather_data": "dummy"}
        assert model.validate_inputs(valid_data) is True

        # Missing inputs
        with pytest.raises(ValueError):
            model.validate_inputs({"field_data": "dummy"})

        with pytest.raises(ValueError):
            model.validate_inputs({"weather_data": "dummy"})

        with pytest.raises(ValueError):
            model.validate_inputs({})

    def test_predict(self):
        """Test predict method."""
        model = ConcreteAgricultureModel()

        # Valid prediction
        result = model.predict({"field_data": "dummy", "weather_data": "dummy"})
        assert "result" in result
        assert result["result"] == "test_prediction"
        assert "inputs" in result
        assert set(result["inputs"]) == {"field_data", "weather_data"}

        # Invalid prediction (missing inputs)
        with pytest.raises(ValueError):
            model.predict({"field_data": "dummy"})

    def test_info(self):
        """Test info property."""
        config = {"param1": "value1", "param2": 123}
        model = ConcreteAgricultureModel(name="custom_model", config=config)

        info = model.info
        assert info["name"] == "custom_model"
        assert info["version"] == "0.1.0"
        assert info["type"] == "ConcreteAgricultureModel"
        assert info["required_inputs"] == ["field_data", "weather_data"]
        assert info["config"] == config

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save and load persist a concrete model instance."""
        model = ConcreteAgricultureModel()
        model_path = tmp_path / "model.pkl"

        model.save(str(model_path))
        loaded = ConcreteAgricultureModel.load(str(model_path))

        assert isinstance(loaded, ConcreteAgricultureModel)
        assert loaded.name == model.name
        assert loaded.version == model.version
        assert loaded.required_inputs == model.required_inputs
