"""Module discovery helpers for navigating a GEO-INFER monorepo checkout.

These utilities locate sibling ``GEO-INFER-*`` module directories and import
their packages by file path. They are used by INTRA's documentation tooling
and by the test suite.
"""

import fnmatch
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def collect_test_modules(root_dir: Path) -> dict[str, Path]:
    """Collect all GEO-INFER modules in the given root directory.

    Args:
        root_dir: Directory containing ``GEO-INFER-*`` module folders.

    Returns:
        Mapping of snake_case module package names to their module directories.
    """
    modules: dict[str, Path] = {}
    if not root_dir.exists():
        return modules

    for item in root_dir.iterdir():
        if item.is_dir() and item.name.startswith("GEO-INFER-"):
            module_name_snake = item.name.lower().replace("-", "_")
            if (item / "src" / module_name_snake).is_dir():
                modules[module_name_snake] = item
    return modules


def import_module_by_path(path: str, name: str) -> ModuleType | None:
    """Import a module from a file path.

    Args:
        path: Path to the Python file to import.
        name: Fully-qualified name to register the module under.

    Returns:
        The imported module, or ``None`` if a spec could not be created.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return None


def find_modules_by_name(root_dir: Path, pattern: str) -> list[Path]:
    """Find GEO-INFER module directories matching a glob pattern.

    Args:
        root_dir: Directory containing ``GEO-INFER-*`` module folders.
        pattern: Glob pattern applied to directory names (e.g. ``GEO-INFER-SP*``).

    Returns:
        Sorted list of matching module directories.
    """
    modules: list[Path] = []
    if not root_dir.exists():
        return modules

    for item in root_dir.iterdir():
        if item.is_dir() and fnmatch.fnmatch(item.name, pattern):
            module_name_snake = item.name.lower().replace("-", "_")
            if (item / "src" / module_name_snake).is_dir():
                modules.append(item)

    return sorted(modules)
