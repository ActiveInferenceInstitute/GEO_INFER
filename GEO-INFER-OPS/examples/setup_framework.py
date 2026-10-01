#!/usr/bin/env python3
"""
GEO-INFER Framework Unified Installation Script

This script installs all GEO-INFER modules in development mode, making them
discoverable by Python and enabling cross-module imports.

Usage:
    python GEO-INFER-OPS/examples/setup_framework.py [--install-deps] [--modules MODULE1,MODULE2]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GEOINFERInstaller:
    """Unified installer for the GEO-INFER framework."""

    def __init__(self, framework_root: Path):
        self.framework_root = framework_root
        self.modules = self._discover_modules()

    def _discover_modules(self) -> dict[str, Path]:
        """Discover all GEO-INFER modules in the framework root."""
        modules = {}
        for item in self.framework_root.iterdir():
            if item.is_dir() and item.name.startswith("GEO-INFER-"):
                module_name = item.name
                setup_py = item / "setup.py"
                if setup_py.exists():
                    modules[module_name] = item
                    logger.info(f"Found module: {module_name}")
        return modules

    def install_module(self, module_path: Path, module_name: str) -> bool:
        """Install a single module in development mode."""
        try:
            logger.info(f"Installing {module_name}...")

            # Change to module directory
            original_cwd = os.getcwd()
            os.chdir(module_path)

            # Install in development mode using uv
            result = subprocess.run(
                ["uv", "pip", "install", "-e", "."],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info(f"✓ Successfully installed {module_name}")
            os.chdir(original_cwd)
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install {module_name}: {e}")
            logger.error(f"Error output: {e.stderr}")
            os.chdir(original_cwd)
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error installing {module_name}: {e}")
            os.chdir(original_cwd)
            return False

    def install_dependencies(self) -> bool:
        """Install core dependencies for the framework."""
        try:
            logger.info("Installing core dependencies...")

            # Core dependencies that all modules need
            core_deps = [
                "numpy>=1.20.0",
                "pandas>=1.3.0",
                "geopandas>=0.10.0",
                "shapely>=1.8.0",
                "h3>=4.5.0,<5",
                "pyproj>=3.3.0",
                "requests>=2.26.0",
                "pyyaml>=5.4.0",
                "folium>=0.12.0",
                "matplotlib>=3.4.0",
                "plotly>=5.3.0",
                "scipy>=1.7.0",
                "scikit-learn>=1.0.0",
            ]

            for dep in core_deps:
                logger.info(f"Installing {dep}...")
                subprocess.run(
                    ["uv", "pip", "install", dep], check=True, capture_output=True
                )

            logger.info("✓ Core dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install dependencies: {e}")
            return False

    def install_all_modules(
        self, target_modules: list[str] | None = None
    ) -> dict[str, bool]:
        """Install all modules or specified modules."""
        results = {}

        modules_to_install = self.modules
        if target_modules:
            modules_to_install = {
                name: path
                for name, path in self.modules.items()
                if name in target_modules
            }

        logger.info(f"Installing {len(modules_to_install)} modules...")

        for module_name, module_path in modules_to_install.items():
            success = self.install_module(module_path, module_name)
            results[module_name] = success

        return results

    def verify_installation(self) -> dict[str, bool]:
        """Verify that all modules can be imported."""
        results = {}

        for module_name in self.modules.keys():
            try:
                # Convert module name to import name
                import_name = module_name.lower().replace("-", "_")
                __import__(import_name)
                logger.info(f"✓ {module_name} imports successfully")
                results[module_name] = True
            except ImportError as e:
                logger.error(f"✗ {module_name} import failed: {e}")
                results[module_name] = False

        return results

    def test_cross_module_imports(self) -> dict[str, bool]:
        """Test cross-module imports that are known to be problematic."""
        cross_import_tests = [
            ("geo_infer_place", "PlaceAnalyzer"),
            ("geo_infer_space", "setup_osc_geo"),
            ("geo_infer_iot", "IoTDataIngestion"),
            ("geo_infer_bayes", "GaussianProcess"),
        ]

        results = {}
        for module_name, class_name in cross_import_tests:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                logger.info(f"✓ {module_name}.{class_name} imports successfully")
                results[f"{module_name}.{class_name}"] = True
            except (ImportError, AttributeError) as e:
                logger.error(f"✗ {module_name}.{class_name} import failed: {e}")
                results[f"{module_name}.{class_name}"] = False

        return results


def main():
    parser = argparse.ArgumentParser(description="Install GEO-INFER framework modules")
    parser.add_argument(
        "--install-deps", action="store_true", help="Install core dependencies"
    )
    parser.add_argument(
        "--modules",
        type=str,
        help="Comma-separated list of specific modules to install",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify installation after installing"
    )
    parser.add_argument(
        "--test-imports", action="store_true", help="Test cross-module imports"
    )

    args = parser.parse_args()

    # Get framework root (repo root)
    framework_root = Path(__file__).resolve().parents[2]
    installer = GEOINFERInstaller(framework_root)

    logger.info("=== GEO-INFER Framework Installation ===")
    logger.info(f"Framework root: {framework_root}")
    logger.info(f"Discovered {len(installer.modules)} modules")

    # Install dependencies if requested
    if args.install_deps:
        if not installer.install_dependencies():
            logger.error("Failed to install dependencies. Exiting.")
            sys.exit(1)

    # Determine which modules to install
    target_modules = None
    if args.modules:
        target_modules = [f"GEO-INFER-{mod.upper()}" for mod in args.modules.split(",")]

    # Install modules
    results = installer.install_all_modules(target_modules)

    # Report results
    successful = sum(results.values())
    total = len(results)
    logger.info("\n=== Installation Results ===")
    logger.info(f"Successfully installed: {successful}/{total} modules")

    if successful < total:
        logger.warning("Some modules failed to install:")
        for module, success in results.items():
            if not success:
                logger.warning(f"  - {module}")

    # Verify installation if requested
    if args.verify:
        logger.info("\n=== Verifying Installation ===")
        verify_results = installer.verify_installation()
        successful_verify = sum(verify_results.values())
        logger.info(
            f"Verification: {successful_verify}/{len(verify_results)} modules import successfully"
        )

    # Test cross-module imports if requested
    if args.test_imports:
        logger.info("\n=== Testing Cross-Module Imports ===")
        import_results = installer.test_cross_module_imports()
        successful_imports = sum(import_results.values())
        logger.info(
            f"Cross-module imports: {successful_imports}/{len(import_results)} successful"
        )

    logger.info("\n=== Installation Complete ===")
    if successful == total:
        logger.info("🎉 All modules installed successfully!")
        logger.info("You can now import GEO-INFER modules in your Python code.")
    else:
        logger.warning("⚠️  Some modules failed to install. Check the logs above.")


if __name__ == "__main__":
    main()
