"""Package export contract for ``geo_infer_risk``.

Every name advertised in ``__all__`` must resolve to a real, non-None object.
The historical failure mode this guards against is the silent-None export: an
optional try/except import that swallows an ImportError, leaves the name bound
to ``None``, and still lists it in ``__all__`` — so ``geo_infer_risk.RiskAPI``
"exists" while being unusable.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import geo_infer_risk
import pytest

# Names once advertised through silent-None try/except imports. The backing
# packages (geo_infer_risk.models, geo_infer_risk.api) never existed and no
# module, test, or example referenced them, so they were removed rather than
# stubbed. The package must not re-introduce them as None-valued exports.
REMOVED_SILENT_NONE_EXPORTS = [
    "FloodModel",
    "EarthquakeModel",
    "HurricaneModel",
    "WildfireModel",
    "DroughtModel",
    "MultiHazardModel",
    "RiskAPI",
    "ModelRegistry",
    "ResultsFormatter",
]


def test_every_all_name_resolves_to_a_real_object() -> None:
    """Each ``__all__`` entry resolves via getattr and is not None."""
    for name in geo_infer_risk.__all__:
        assert hasattr(geo_infer_risk, name), (
            f"__all__ advertises {name!r} but the attribute does not exist"
        )
        assert getattr(geo_infer_risk, name) is not None, (
            f"__all__ advertises {name!r} but it is bound to None"
        )


def test_silent_none_exports_were_removed() -> None:
    """Removed None-valued exports stay removed (no silent reintroduction)."""
    for name in REMOVED_SILENT_NONE_EXPORTS:
        assert name not in geo_infer_risk.__all__, (
            f"{name!r} returned to __all__ without a real implementation"
        )
        assert not hasattr(geo_infer_risk, name), (
            f"{name!r} is bound on the package but not implemented"
        )


def test_enhanced_exposure_export_surface() -> None:
    """EnhancedExposureModel and its subclasses/factories export from core."""
    from geo_infer_risk.core import (
        EnhancedExposureModel,
        EnhancedInfrastructureExposureModel,
        EnhancedPropertyExposureModel,
        EnhancedPopulationExposureModel,
        create_enhanced_infrastructure_exposure_model,
        create_enhanced_property_exposure_model,
        create_enhanced_population_exposure_model,
    )

    from geo_infer_risk import EnhancedExposureModel as package_level

    assert package_level is EnhancedExposureModel
    for cls in (
        EnhancedPropertyExposureModel,
        EnhancedPopulationExposureModel,
        EnhancedInfrastructureExposureModel,
    ):
        assert issubclass(cls, EnhancedExposureModel)
    assert callable(create_enhanced_property_exposure_model)
    assert callable(create_enhanced_population_exposure_model)
    assert callable(create_enhanced_infrastructure_exposure_model)


@pytest.mark.parametrize(
    "failure", ["dependency", "export", "nested", "installed_root"]
)
def test_broken_installed_bayes_import_is_not_reported_as_an_absent_extra(
    failure, tmp_path
):
    """Only a positively absent root package can activate civic fallback."""
    package = tmp_path / "geo_infer_bayes"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    if failure == "installed_root":
        (package / "__init__.py").write_text(
            "raise ModuleNotFoundError('installed BAYES root failed', name='geo_infer_bayes')\n",
            encoding="utf-8",
        )
    elif failure == "dependency":
        (package / "civic_intel.py").write_text(
            "import civic_boundary_missing_dependency\n", encoding="utf-8"
        )
    elif failure == "export":
        (package / "civic_intel.py").write_text(
            "raise ImportError('broken BAYES export')\n", encoding="utf-8"
        )
    code = textwrap.dedent("""
        import importlib.util
        from pathlib import Path
        import sys

        failure, temporary = sys.argv[1:]
        assert "geo_infer_risk" not in sys.modules and "geo_infer_bayes" not in sys.modules
        sys.path.insert(0, temporary)
        spec = importlib.util.find_spec("geo_infer_bayes")
        assert spec is not None and Path(spec.origin).is_relative_to(Path(temporary))
        try:
            import geo_infer_risk
        except ModuleNotFoundError as exc:
            assert failure in {"dependency", "nested", "installed_root"}
            expected = ("civic_boundary_missing_dependency" if failure == "dependency"
                        else "geo_infer_bayes" if failure == "installed_root"
                        else "geo_infer_bayes.civic_intel")
            assert exc.name == expected
        except ImportError as exc:
            assert failure == "export" and str(exc) == "broken BAYES export"
        else:
            raise AssertionError("installed package failure was hidden")
    """)
    subprocess.run(
        [sys.executable, "-I", "-Werror", "-c", code, failure, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
    )
