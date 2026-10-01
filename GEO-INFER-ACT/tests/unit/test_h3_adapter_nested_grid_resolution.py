"""Regression tests for GS-023: the ACT H3 adapter must not discover the
sibling GEO-INFER-SPACE checkout via ``Path(__file__)`` parent climbs.

Default resolution relies only on the installed ``geo_infer_space`` package;
repo-checkout access requires BOTH the opt-in flag
``GEO_INFER_ACT_ALLOW_REPO_PATH_FALLBACK`` and an explicit checkout root in
``GEO_INFER_ACT_REPO_ROOT``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
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
def isolated_space_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[str]]:
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
    """Default resolution returns the packaged NestedH3Grid with no cwd help."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEO_INFER_ACT_ALLOW_REPO_PATH_FALLBACK", raising=False)
    monkeypatch.delenv("GEO_INFER_ACT_REPO_ROOT", raising=False)

    nested_h3_grid = get_nested_h3_grid_class()

    assert nested_h3_grid.__name__ == "NestedH3Grid"
    assert nested_h3_grid.__module__.startswith("geo_infer_space.nested")
    assert sys.path == isolated_space_imports


def test_no_silent_repo_fallback_without_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_space_imports: list[str],
) -> None:
    """Without the opt-in flag a missing package raises, sys.path untouched."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEO_INFER_ACT_ALLOW_REPO_PATH_FALLBACK", raising=False)
    monkeypatch.delenv("GEO_INFER_ACT_REPO_ROOT", raising=False)
    _block_space_import(monkeypatch)

    with pytest.raises(ImportError):
        get_nested_h3_grid_class()

    assert sys.path == isolated_space_imports


def test_fallback_requires_explicit_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_space_imports: list[str],
) -> None:
    """Opt-in without an explicit checkout root is rejected, not climbed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEO_INFER_ACT_ALLOW_REPO_PATH_FALLBACK", "1")
    monkeypatch.delenv("GEO_INFER_ACT_REPO_ROOT", raising=False)
    _block_space_import(monkeypatch)

    with pytest.raises(RuntimeError, match="GEO_INFER_ACT_REPO_ROOT"):
        get_nested_h3_grid_class()

    assert sys.path == isolated_space_imports


def test_fallback_honors_explicit_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_space_imports: list[str],
) -> None:
    """With flag plus explicit root, the sibling checkout is used verbatim."""
    monkeypatch.chdir(tmp_path)
    space_src = tmp_path / "GEO-INFER-SPACE" / "src"
    stub_pkg = space_src / "geo_infer_space"
    stub_pkg.mkdir(parents=True)
    (stub_pkg / "nested.py").write_text(
        "class NestedH3Grid:\n    stub_marker = 'gs023'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEO_INFER_ACT_ALLOW_REPO_PATH_FALLBACK", "true")
    monkeypatch.setenv("GEO_INFER_ACT_REPO_ROOT", str(tmp_path))
    # Hide the editable GEO-INFER-SPACE install: a regular package beats a
    # namespace-package candidate regardless of sys.path order, so the
    # fallback must be exercised against an environment where only the
    # explicitly supplied checkout can satisfy the import.
    sys.path[:] = [p for p in sys.path if "GEO-INFER-SPACE" not in p]

    class _OnceBlocker:
        """Fail exactly one ``geo_infer_space`` import, then stand down."""

        def __init__(self) -> None:
            self.armed = True

        def find_spec(
            self,
            fullname: str,
            path: Any | None = None,
            target: Any | None = None,
        ) -> None:
            if self.armed and (
                fullname == "geo_infer_space" or fullname.startswith("geo_infer_space.")
            ):
                self.armed = False
                raise ImportError(f"blocked: {fullname}")

    # Reproduce a truly uninstalled environment: only the blocker and the
    # plain PathFinder stay on the meta path, so the editable-install finder
    # cannot out-rank the sys.path entry the fallback inserts.
    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            _OnceBlocker(),
            sys.modules["_frozen_importlib_external"].PathFinder,
        ],
    )

    nested_h3_grid = get_nested_h3_grid_class()

    assert nested_h3_grid.__name__ == "NestedH3Grid"
    assert getattr(nested_h3_grid, "stub_marker", None) == "gs023"
    assert nested_h3_grid.__module__ == "geo_infer_space.nested"
    assert str(space_src) in sys.path
