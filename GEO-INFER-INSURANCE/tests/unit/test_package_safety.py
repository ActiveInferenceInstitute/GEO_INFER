"""Packaging-safety tests: every Python package directory ships in wheels.

setuptools' ``find_packages`` only includes directories containing an
``__init__.py``. A package directory missing that file is silently dropped
from built wheels (it still imports in the dev tree as an implicit namespace
package). These tests pin the invariant for geo_infer_insurance.
"""

import importlib
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2] / "src" / "geo_infer_insurance"


def _python_dirs(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir())


def test_every_directory_with_modules_is_a_package() -> None:
    for rel in _python_dirs(PKG_ROOT):
        directory = PKG_ROOT / rel
        has_modules = any(directory.glob("*.py"))
        if has_modules:
            assert (directory / "__init__.py").is_file(), (
                f"{rel} contains Python modules but has no __init__.py; "
                "find_packages would exclude it from built wheels"
            )


def test_underwriting_subpackages_importable() -> None:
    for subpackage in ("core", "utils", "models"):
        module = importlib.import_module(
            f"geo_infer_insurance.underwriting.{subpackage}"
        )
        assert module.__name__.endswith(subpackage)
    import geo_infer_insurance.underwriting as uw_pkg

    for name in ("UnderwritingEngine", "create_underwriting_engine"):
        assert hasattr(uw_pkg, name), f"geo_infer_insurance.underwriting missing {name}"


def test_root_package_exports_public_api() -> None:
    import geo_infer_insurance

    for name in (
        "PolicyManager",
        "ClaimsProcessor",
        "PricingEngine",
        "create_underwriting_system",
        "underwrite_insurance_policy",
        "process_insurance_claim",
    ):
        assert hasattr(geo_infer_insurance, name), f"geo_infer_insurance missing {name}"
