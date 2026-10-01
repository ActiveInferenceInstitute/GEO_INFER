"""Regression tests for the clone_repos.py thin wrapper (GS-217)."""

import logging
import runpy
import sys
import types

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "clone_repos.py"


def _install_stub_main(monkeypatch, impl):
    """Swap geo_infer_git.main for a stub module whose main() runs impl."""
    main_module = types.ModuleType("geo_infer_git.main")
    main_module.main = impl
    fake_pkg = types.ModuleType("geo_infer_git")
    fake_pkg.main = main_module
    real_pkg = sys.modules.get("geo_infer_git")
    real_main = sys.modules.get("geo_infer_git.main")
    sys.modules["geo_infer_git"] = fake_pkg
    sys.modules["geo_infer_git.main"] = main_module
    monkeypatch.setattr(logging, "basicConfig", lambda *args, **kwargs: None)
    return real_pkg, real_main


def _restore_modules(real_pkg, real_main):
    if real_pkg is None:
        sys.modules.pop("geo_infer_git", None)
    else:
        sys.modules["geo_infer_git"] = real_pkg
    if real_main is None:
        sys.modules.pop("geo_infer_git.main", None)
    else:
        sys.modules["geo_infer_git.main"] = real_main


def test_wrapper_delegates_to_package_main(monkeypatch):
    """The standalone wrapper must delegate to geo_infer_git.main.main."""
    called = []
    real_pkg, real_main = _install_stub_main(monkeypatch, lambda: called.append("main"))
    try:
        runpy.run_path(str(WRAPPER), run_name="__main__")
    finally:
        _restore_modules(real_pkg, real_main)
    assert called == ["main"]


def test_wrapper_exits_nonzero_on_interrupt(monkeypatch):
    """A KeyboardInterrupt from the package pipeline must exit with code 1."""

    def boom():
        raise KeyboardInterrupt

    real_pkg, real_main = _install_stub_main(monkeypatch, boom)
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path(str(WRAPPER), run_name="__main__")
    finally:
        _restore_modules(real_pkg, real_main)
    assert excinfo.value.code == 1


def test_duplicate_clone_script_removed():
    """clone_script.py (subprocess duplicate of the main pipeline) must be gone."""
    assert not (REPO_ROOT / "clone_script.py").exists()
