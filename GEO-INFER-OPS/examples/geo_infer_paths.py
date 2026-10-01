#!/usr/bin/env python3
"""
GEO-INFER Path Management Utility

This module provides utilities for managing Python paths and imports
across the GEO-INFER framework modules.
"""

import sys
from pathlib import Path
import logging
import importlib

logger = logging.getLogger(__name__)


class GEOINFERPathManager:
    """Manages Python paths for GEO-INFER framework modules."""

    def __init__(self, framework_root: Path | None = None):
        """
        Initialize the path manager.

        Args:
            framework_root: Root directory of the GEO-INFER framework.
                          If None, will try to auto-detect.
        """
        if framework_root is None:
            framework_root = self._auto_detect_framework_root()

        self.framework_root = framework_root
        self.module_paths = self._discover_module_paths()
        self.installed_modules = self._discover_installed_modules()

    def _auto_detect_framework_root(self) -> Path:
        """Auto-detect the framework root directory."""
        # Start from current directory and look for GEO-INFER modules
        current = Path.cwd()

        # Look for GEO-INFER modules in current directory
        geo_infer_dirs = [
            d
            for d in current.iterdir()
            if d.is_dir() and d.name.startswith("GEO-INFER-")
        ]

        if geo_infer_dirs:
            return current

        # Look in parent directories
        for parent in current.parents:
            geo_infer_dirs = [
                d
                for d in parent.iterdir()
                if d.is_dir() and d.name.startswith("GEO-INFER-")
            ]
            if geo_infer_dirs:
                return parent

        # Fallback to current directory
        logger.warning("Could not auto-detect framework root, using current directory")
        return current

    def _discover_module_paths(self) -> dict[str, Path]:
        """Discover all GEO-INFER module paths for development."""
        module_paths = {}

        for item in self.framework_root.iterdir():
            if item.is_dir() and item.name.startswith("GEO-INFER-"):
                src_path = item / "src"
                if src_path.exists():
                    # Convert GEO-INFER-MODULE to geo_infer_module
                    module_name = item.name.lower().replace("-", "_")
                    module_paths[module_name] = src_path
                    logger.debug(
                        f"Found development module: {module_name} at {src_path}"
                    )

        return module_paths

    def _discover_installed_modules(self) -> dict[str, str]:
        """Discover installed GEO-INFER modules."""
        installed_modules = {}

        # Try to import installed modules
        module_names = [
            "geo_infer_space",
            "geo_infer_place",
            "geo_infer_iot",
            "geo_infer_bayes",
            "geo_infer_act",
            "geo_infer_agent",
            "geo_infer_art",
            "geo_infer_bio",
            "geo_infer_sec",
            "geo_infer_test",
            "geo_infer_api",
            "geo_infer_norms",
            "geo_infer_ops",
            "geo_infer_examples",
            "geo_infer_git",
            "geo_infer_intra",
        ]

        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
                installed_modules[module_name] = module.__file__
                logger.debug(f"Found installed module: {module_name}")
            except ImportError:
                logger.debug(f"Module not installed: {module_name}")

        return installed_modules

    def add_module_paths(self, modules: list[str] | None = None) -> list[str]:
        """
        Add module paths to sys.path.

        Args:
            modules: List of module names to add. If None, adds all modules.

        Returns:
            List of paths that were added.
        """
        added_paths = []

        modules_to_add = self.module_paths.keys() if modules is None else modules

        for module_name in modules_to_add:
            if module_name in self.module_paths:
                path = str(self.module_paths[module_name])
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)
                    logger.debug(f"Added development path: {path}")

        return added_paths

    def add_all_paths(self) -> list[str]:
        """Add all discovered module paths to sys.path."""
        return self.add_module_paths()

    def get_module_path(self, module_name: str) -> Path | None:
        """Get the path for a specific module."""
        return self.module_paths.get(module_name)

    def list_available_modules(self) -> list[str]:
        """List all available modules (both installed and development)."""
        all_modules = set(self.module_paths.keys())
        all_modules.update(self.installed_modules.keys())
        return list(all_modules)

    def is_module_installed(self, module_name: str) -> bool:
        """Check if a module is installed."""
        return module_name in self.installed_modules

    def is_module_development(self, module_name: str) -> bool:
        """Check if a module is available in development mode."""
        return module_name in self.module_paths

    def import_module(
        self, module_name: str, ensure_path: bool = True
    ) -> object | None:
        """
        Import a module, ensuring its path is available.

        Args:
            module_name: Name of the module to import
            ensure_path: Whether to ensure the module path is in sys.path

        Returns:
            The imported module or None if import fails
        """
        # First try to import as installed module
        if module_name in self.installed_modules:
            try:
                module = importlib.import_module(module_name)
                logger.debug(f"Successfully imported installed module: {module_name}")
                return module
            except ImportError as e:
                logger.debug(f"Failed to import installed module {module_name}: {e}")

        # If not installed, try development path
        if ensure_path and module_name in self.module_paths:
            self.add_module_paths([module_name])

        try:
            module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported development module: {module_name}")
            return module
        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            return None

    def import_from_module(
        self, module_name: str, item_name: str, ensure_path: bool = True
    ):
        """
        Import a specific item from a module.

        Args:
            module_name: Name of the module
            item_name: Name of the item to import
            ensure_path: Whether to ensure the module path is in sys.path

        Returns:
            The imported item or None if import fails
        """
        module = self.import_module(module_name, ensure_path)
        if module is None:
            return None

        try:
            item = getattr(module, item_name)
            logger.debug(f"Successfully imported {item_name} from {module_name}")
            return item
        except AttributeError as e:
            logger.error(f"Failed to import {item_name} from {module_name}: {e}")
            return None


# Global instance for convenience
_path_manager = None


def get_path_manager() -> GEOINFERPathManager:
    """Get the global path manager instance."""
    global _path_manager
    if _path_manager is None:
        _path_manager = GEOINFERPathManager()
    return _path_manager


def add_module_paths(modules: list[str] | None = None) -> list[str]:
    """Convenience function to add module paths."""
    return get_path_manager().add_module_paths(modules)


def add_all_paths() -> list[str]:
    """Convenience function to add all module paths."""
    return get_path_manager().add_all_paths()


def import_module(module_name: str) -> object | None:
    """Convenience function to import a module."""
    return get_path_manager().import_module(module_name)


def import_from_module(module_name: str, item_name: str):
    """Convenience function to import an item from a module."""
    return get_path_manager().import_from_module(module_name, item_name)


def list_available_modules() -> list[str]:
    """Convenience function to list available modules."""
    return get_path_manager().list_available_modules()


def is_module_installed(module_name: str) -> bool:
    """Convenience function to check if a module is installed."""
    return get_path_manager().is_module_installed(module_name)


# Auto-add all paths when module is imported
def _auto_setup():
    """Automatically set up paths when this module is imported."""
    try:
        manager = get_path_manager()
        added_paths = manager.add_all_paths()
        if added_paths:
            logger.info(
                f"Auto-added {len(added_paths)} development module paths to sys.path"
            )

        installed_count = len(manager.installed_modules)
        if installed_count > 0:
            logger.info(f"Found {installed_count} installed modules")

    except Exception as e:
        logger.warning(f"Auto-setup failed: {e}")


# Run auto-setup
_auto_setup()
