"""Tests for API configuration settings."""


def test_lazy_application_is_cached_on_actual_submodule(monkeypatch):
    import importlib

    module = importlib.import_module("geo_infer_api.app")
    monkeypatch.delitem(module.__dict__, "main_app", raising=False)
    first = module.main_app
    marker = object()
    first.state.cache_marker = marker
    assert module.main_app is first
    assert module.__dict__["main_app"] is first
    assert module.main_app.state.cache_marker is marker


class TestSettings:
    def test_app_version_tracks_distribution(self):
        """app_version matches the installed distribution, so it survives any release bump."""
        from importlib.metadata import PackageNotFoundError, version

        from geo_infer_api.core.config import get_settings

        settings = get_settings()
        assert settings.app_name == "GEO-INFER-API"
        try:
            assert settings.app_version == version("geo-infer-api")
        except PackageNotFoundError:
            # Source checkout without an installed distribution: a nonempty
            # static fallback is the documented behavior.
            assert settings.app_version
        assert settings.api_prefix == "/api/v1"

    def test_cors_origins_default(self):
        from geo_infer_api.core.config import get_settings

        settings = get_settings()
        assert isinstance(settings.cors_origins, list)

    def test_cors_origins_json_string(self):
        """A JSON-array CORS_ORIGINS string parses into a real list."""
        from geo_infer_api.core.config import Settings

        settings = Settings(
            secret_key="test-secret",
            cors_origins='["https://a.example.com", "https://b.example.com"]',
            _env_file=None,
        )
        assert settings.cors_origins == [
            "https://a.example.com",
            "https://b.example.com",
        ]

    def test_cors_origins_comma_string(self):
        """A comma-separated CORS_ORIGINS string parses into a list."""
        from geo_infer_api.core.config import Settings

        settings = Settings(
            secret_key="test-secret",
            cors_origins="https://a.example.com, https://b.example.com",
            _env_file=None,
        )
        assert settings.cors_origins == [
            "https://a.example.com",
            "https://b.example.com",
        ]

    def test_ogc_features_enabled(self):
        from geo_infer_api.core.config import get_settings

        settings = get_settings()
        assert settings.ogc_api_features_enabled is True
        assert settings.ogc_api_processes_enabled is True
