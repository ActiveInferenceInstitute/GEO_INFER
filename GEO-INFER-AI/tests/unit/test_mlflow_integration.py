"""
Unit tests for the MLflow pipeline (mlflow is a declared geo-infer-ai
dependency, so no availability guard is needed).

MLflow runs against a local sqlite tracking store created per test via
``tmp_path``, so the suite stays deterministic, isolated, and offline.
"""

import warnings

import numpy as np
import pytest

from geo_infer_ai.pipelines.mlflow_integration import MLflowPipeline


@pytest.fixture(autouse=True)
def _quiet_mlflow_chatter():
    """MLflow emits deprecation/maintenance chatter that the repo's
    ``filterwarnings = ["error"]`` would promote into failures; tracking
    behavior is what these tests assert, not warning hygiene."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def tracking_dir(tmp_path):
    """Dedicated local sqlite MLflow tracking store per test."""
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture
def fitted_model():
    """Minimal fitted sklearn model for log/load round-trips."""
    from sklearn.linear_model import LinearRegression

    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 3.0 * X.ravel() + 1.0
    return LinearRegression().fit(X, y)


class TestMLflowPipelineEnabled:
    """Pipeline behavior when MLflow is available and enabled."""

    def test_enabled_with_local_tracking(self, tracking_dir) -> None:
        pipeline = MLflowPipeline(
            experiment_name="unit-local",
            tracking_uri=tracking_dir,
        )
        assert pipeline.enabled is True
        assert pipeline.current_run is None

    def test_full_lifecycle_logs_params_metrics_artifacts(
        self, tracking_dir, tmp_path
    ) -> None:
        pipeline = MLflowPipeline(
            experiment_name="unit-lifecycle",
            tracking_uri=tracking_dir,
        )

        pipeline.start_run(run_name="lifecycle", tags={"suite": "unit"})
        assert pipeline.current_run is not None
        run_id = pipeline.current_run.info.run_id

        pipeline.log_params({"batch_size": 16, "learning_rate": 0.1})
        pipeline.log_metrics({"r2": 0.75, "mse": 0.25}, step=3)

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "notes.txt").write_text("hello", encoding="utf-8")
        pipeline.log_artifacts(artifacts, artifact_path="notes")

        pipeline.end_run()
        assert pipeline.current_run is None

        experiment = __import__("mlflow").get_experiment_by_name("unit-lifecycle")
        assert experiment is not None
        run = __import__("mlflow").get_run(run_id)
        assert run.data.params["batch_size"] == "16"
        assert run.data.metrics["r2"] == 0.75
        assert run.data.metrics["mse"] == 0.25

    def test_sklearn_model_log_and_load_round_trip(
        self, tracking_dir, fitted_model
    ) -> None:
        pipeline = MLflowPipeline(
            experiment_name="unit-model", tracking_uri=tracking_dir
        )
        pipeline.start_run(run_name="model")

        pipeline.log_model(fitted_model, artifact_path="model")
        run_id = pipeline.current_run.info.run_id
        pipeline.end_run()

        model_uri = f"runs:/{run_id}/model"
        loaded = pipeline.load_model(model_uri)
        assert np.allclose(
            loaded.predict(np.array([[2.0]])),
            fitted_model.predict(np.array([[2.0]])),
        )

    def test_log_model_unsupported_type_is_noop(self, tracking_dir) -> None:
        """Models without fit/predict are rejected with a warning, not an error."""
        pipeline = MLflowPipeline(
            experiment_name="unit-nomodel", tracking_uri=tracking_dir
        )
        pipeline.start_run(run_name="nomodel")
        # Should not raise for an object without fit/predict
        pipeline.log_model({"not": "a model"}, artifact_path="junk")
        pipeline.end_run()

    def test_load_model_failure_raises(self, tracking_dir) -> None:
        """A bogus model URI propagates the load failure to the caller."""
        pipeline = MLflowPipeline(
            experiment_name="unit-load-fail", tracking_uri=tracking_dir
        )
        pipeline.start_run(run_name="load-fail")
        from mlflow.exceptions import MlflowException

        with pytest.raises(MlflowException, match="not found"):
            pipeline.load_model("runs:/00000000000000000000000000000000/bogus")
        pipeline.end_run()

    def test_broken_tracking_uri_disables_pipeline(self) -> None:
        """An unusable tracking URI degrades gracefully to disabled mode."""
        pipeline = MLflowPipeline(
            experiment_name="unit-broken", tracking_uri="unknown-scheme://bogus"
        )
        assert pipeline.enabled is False

    def test_setup_skips_when_mlflow_unavailable(
        self, tracking_dir, monkeypatch
    ) -> None:
        """_setup_mlflow flips to disabled when the import guard failed."""
        pipeline = MLflowPipeline(
            experiment_name="unit-guard", tracking_uri=tracking_dir
        )
        import geo_infer_ai.pipelines.mlflow_integration as module

        monkeypatch.setattr(module, "MLFLOW_AVAILABLE", False)
        pipeline._setup_mlflow()
        assert pipeline.enabled is False


class TestMLflowPipelineDisabled:
    """Every no-op path fires when tracking is disabled."""

    @pytest.fixture
    def disabled_pipeline(self) -> MLflowPipeline:
        return MLflowPipeline(enabled=False)

    def test_start_run_is_noop(self, disabled_pipeline: MLflowPipeline) -> None:
        disabled_pipeline.start_run(run_name="never")
        assert disabled_pipeline.current_run is None

    def test_end_run_is_noop(self, disabled_pipeline: MLflowPipeline) -> None:
        disabled_pipeline.end_run()
        assert disabled_pipeline.current_run is None

    def test_log_params_is_noop(self, disabled_pipeline: MLflowPipeline) -> None:
        disabled_pipeline.log_params({"p": 1})

    def test_log_metrics_is_noop(self, disabled_pipeline: MLflowPipeline) -> None:
        disabled_pipeline.log_metrics({"m": 1.0}, step=1)

    def test_log_model_is_noop(self, disabled_pipeline: MLflowPipeline) -> None:
        disabled_pipeline.log_model(object(), artifact_path="x")

    def test_log_artifacts_is_noop(
        self, disabled_pipeline: MLflowPipeline, tmp_path
    ) -> None:
        directory = tmp_path / "none"
        directory.mkdir()
        disabled_pipeline.log_artifacts(directory)

    def test_load_model_raises_when_disabled(
        self, disabled_pipeline: MLflowPipeline
    ) -> None:
        with pytest.raises(ValueError, match="MLflow not enabled"):
            disabled_pipeline.load_model("models:/x/1")
