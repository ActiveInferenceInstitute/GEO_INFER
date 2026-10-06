"""Public package metadata must agree with its build metadata without imports."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULES = sorted(REPO_ROOT.glob("GEO-INFER-*/pyproject.toml"))


def test_lock_receipt_bytes_survive_windows_style_git_checkout(tmp_path):
    """A real autocrlf checkout preserves lock bytes while converting its control."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }

    def git(*args):
        subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.autocrlf=true",
                "-c",
                "core.safecrlf=false",
                *args,
            ],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            timeout=20,
        )

    git("init", "--quiet")
    lock = (REPO_ROOT / "uv.lock").read_bytes()
    (tmp_path / ".gitattributes").write_bytes(
        (REPO_ROOT / ".gitattributes").read_bytes()
    )
    (tmp_path / "uv.lock").write_bytes(lock)
    (tmp_path / "control.lock").write_bytes(b"public\ncontrol\n")
    git("add", ".gitattributes", "uv.lock", "control.lock")
    (tmp_path / "uv.lock").unlink()
    (tmp_path / "control.lock").unlink()
    git("checkout-index", "--force", "uv.lock", "control.lock")
    assert (
        hashlib.sha256((tmp_path / "uv.lock").read_bytes()).digest()
        == hashlib.sha256(lock).digest()
    )
    assert (tmp_path / "control.lock").read_bytes() == b"public\r\ncontrol\r\n"


def _literals(path: Path) -> dict[str, str]:
    """Read public metadata literals without loading optional runtime libraries."""
    values: dict[str, str] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {
                "__version__",
                "__license__",
            }:
                value = ast.literal_eval(node.value)
                assert isinstance(value, str), f"{path}: {target.id} must be a string"
                assert target.id not in values, f"{path}: duplicate {target.id}"
                values[target.id] = value
    return values


@pytest.mark.parametrize("manifest", MODULES, ids=lambda path: path.parent.name)
def test_runtime_metadata_matches_package_manifest(manifest: Path) -> None:
    """Every package version and any declared runtime license match its wheel."""
    project = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]
    package = project["name"].replace("-", "_")
    path = manifest.parent / "src" / package / "__init__.py"
    values = _literals(path)
    assert values.get("__version__") == project["version"], (
        f"{path}: runtime version {values.get('__version__')!r} differs from "
        f"project.version {project['version']!r}"
    )
    if "__license__" in values:
        license_value = project["license"]
        if isinstance(license_value, dict):
            license_value = license_value["text"]
        assert values["__license__"] == license_value, (
            f"{path}: runtime license {values['__license__']!r} differs from "
            f"project.license {license_value!r}"
        )
