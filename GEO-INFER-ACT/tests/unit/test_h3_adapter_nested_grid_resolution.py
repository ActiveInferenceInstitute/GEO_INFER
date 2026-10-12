"""Regression tests for ``get_nested_h3_grid_class`` resolution.

The ACT H3 adapter resolves ``NestedH3Grid`` only from the installed
``geo_infer_space`` package (the ACT ``space`` extra). It never extends
``sys.path`` or climbs to a sibling checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Iterator

import pytest

from geo_infer_act.utils.h3_adapter import get_nested_h3_grid_class


def _purge_space_modules(restore: dict) -> None:
    """Drop every cached ``geo_infer_space`` module, then optionally restore."""
    for name in [
        name
        for name in sys.modules
        if name == "geo_infer_space" or name.startswith("geo_infer_space.")
    ]:
        del sys.modules[name]
    sys.modules.update(restore)


@pytest.fixture()
def isolated_space_imports() -> Iterator[list[str]]:
    """Freeze ``sys.path`` and the ``geo_infer_space`` module cache."""
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "geo_infer_space" or name.startswith("geo_infer_space.")
    }
    saved_path = list(sys.path)
    _purge_space_modules({})
    yield saved_path
    _purge_space_modules({})
    sys.modules.update(saved_modules)
    sys.path[:] = saved_path


def _block_space_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the next ``geo_infer_space`` import fail as if uninstalled."""
    monkeypatch.setitem(sys.modules, "geo_infer_space", None)
    monkeypatch.setitem(sys.modules, "geo_infer_space.nested", None)


def test_default_resolution_works_from_empty_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_space_imports: list[str],
) -> None:
    """Resolution returns the packaged NestedH3Grid with no cwd help."""
    monkeypatch.chdir(tmp_path)

    nested_h3_grid = get_nested_h3_grid_class()

    assert nested_h3_grid.__name__ == "NestedH3Grid"
    assert nested_h3_grid.__module__.startswith("geo_infer_space.nested")
    assert sys.path == isolated_space_imports


def test_missing_space_package_raises_import_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_space_imports: list[str],
) -> None:
    """A missing ``geo_infer_space`` raises and leaves ``sys.path`` untouched."""
    monkeypatch.chdir(tmp_path)
    _block_space_import(monkeypatch)

    with pytest.raises(ImportError, match=r"geo-infer-act\[space\]"):
        get_nested_h3_grid_class()

    assert sys.path == isolated_space_imports


def test_repo_checkout_environment_variables_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_space_imports: list[str],
) -> None:
    """The removed repo-path fallback variables no longer extend ``sys.path``."""
    monkeypatch.chdir(tmp_path)
    space_pkg = tmp_path / "GEO-INFER-SPACE" / "src" / "geo_infer_space"
    space_pkg.mkdir(parents=True)
    (space_pkg / "nested.py").write_text("class NestedH3Grid:\n    pass\n")
    monkeypatch.setenv("GEO_INFER_ACT_ALLOW_REPO_PATH_FALLBACK", "1")
    monkeypatch.setenv("GEO_INFER_ACT_REPO_ROOT", str(tmp_path))
    _block_space_import(monkeypatch)

    with pytest.raises(ImportError):
        get_nested_h3_grid_class()

    assert sys.path == isolated_space_imports
