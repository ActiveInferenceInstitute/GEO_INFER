#!/usr/bin/env python3
"""
Data Cleanup and Reorganization Utility for Cascadia Framework

This module provides utilities to:
- Clean up old run data from output directory
- Move module-specific data to appropriate module directories
- Maintain run-specific summaries in output directory
- Organize data storage for better performance and clarity
"""

import shutil
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCleanupManager:
    """
    Manages data cleanup and reorganization for the Cascadia framework.

    Features:
    - Clean up old run data
    - Move module data to appropriate directories
    - Maintain run-specific summaries
    - Organize data storage efficiently
    """

    def __init__(self, base_dir: Path):
        """
        Initialize the data cleanup manager.

        Args:
            base_dir: Base directory for the Cascadia framework
        """
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "output"
        self.modules_dir = self.base_dir

        # Define module directories
        self.module_dirs = {
            "zoning": self.base_dir / "zoning" / "data",
            "current_use": self.base_dir / "current_use" / "data",
            "ownership": self.base_dir / "ownership" / "data",
            "improvements": self.base_dir / "improvements" / "data",
            "water_rights": self.base_dir / "water_rights" / "data",
            "ground_water": self.base_dir / "ground_water" / "data",
            "surface_water": self.base_dir / "surface_water" / "data",
            "power_source": self.base_dir / "power_source" / "data",
            "mortgage_debt": self.base_dir / "mortgage_debt" / "data",
        }

        # Create module data directories
        for module_dir in self.module_dirs.values():
            module_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_old_runs(self, keep_recent_runs: int = 3):
        """
        Clean up old run data from output directory.

        Args:
            keep_recent_runs: Number of recent runs to keep
        """
        logger.info("🧹 Cleaning up old run data...")

        # Find all run-specific files
        run_files = []

        # Look for timestamped files
        for pattern in [
            "cascadia_unified_data_*.geojson",
            "cascadia_redevelopment_scores_*.json",
            "cascadia_summary_*.json",
            "cascadia_analysis_report_*.md",
        ]:
            run_files.extend(self.output_dir.glob(pattern))

        # Group files by timestamp
        run_groups = {}
        for file_path in run_files:
            # Extract timestamp from filename
            timestamp = self._extract_timestamp(file_path.name)
            if timestamp:
                if timestamp not in run_groups:
                    run_groups[timestamp] = []
                run_groups[timestamp].append(file_path)

        # Sort runs by timestamp (newest first)
        sorted_runs = sorted(run_groups.keys(), reverse=True)

        # Keep only the most recent runs
        runs_to_keep = sorted_runs[:keep_recent_runs]
        runs_to_delete = sorted_runs[keep_recent_runs:]

        # Delete old run files
        deleted_count = 0
        for run_timestamp in runs_to_delete:
            for file_path in run_groups[run_timestamp]:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted old run file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")

        logger.info(f"✅ Cleaned up {deleted_count} old run files")
        logger.info(f"📁 Kept {len(runs_to_keep)} recent runs")

    def move_module_data_to_modules(self):
        """
        Move module-specific data from output/data to appropriate module directories.
        """
        logger.info("📦 Moving module data to appropriate directories...")

        output_data_dir = self.output_dir / "data"
        if not output_data_dir.exists():
            logger.info("No output/data directory found")
            return

        # Move empirical data
        empirical_dir = output_data_dir / "empirical"
        if empirical_dir.exists():
            for file_path in empirical_dir.glob("empirical_*_data.geojson"):
                module_name = file_path.stem.replace("empirical_", "").replace("_data", "")
                if module_name in self.module_dirs:
                    target_dir = self.module_dirs[module_name] / "empirical"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / file_path.name

                    try:
                        shutil.move(str(file_path), str(target_path))
                        logger.info(f"📁 Moved {file_path.name} to {target_path}")
                    except Exception as e:
                        logger.warning(f"Failed to move {file_path}: {e}")

        # Move cache data
        cache_dir = output_data_dir / "cache"
        if cache_dir.exists():
            for file_path in cache_dir.glob("*_h3_res*.json"):
                # Extract module name from filename
                module_name = file_path.stem.split("_h3_res")[0]
                if module_name in self.module_dirs:
                    target_dir = self.module_dirs[module_name] / "cache"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / file_path.name

                    try:
                        shutil.move(str(file_path), str(target_path))
                        logger.info(f"📁 Moved {file_path.name} to {target_path}")
                    except Exception as e:
                        logger.warning(f"Failed to move {file_path}: {e}")

        # Move processed data
        processed_dir = output_data_dir / "processed"
        if processed_dir.exists():
            for file_path in processed_dir.glob("processed_*_data.geojson"):
                module_name = file_path.stem.replace("processed_", "").replace("_data", "")
                if module_name in self.module_dirs:
                    target_dir = self.module_dirs[module_name] / "processed"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / file_path.name

                    try:
                        shutil.move(str(file_path), str(target_path))
                        logger.info(f"📁 Moved {file_path.name} to {target_path}")
                    except Exception as e:
                        logger.warning(f"Failed to move {file_path}: {e}")

        # Move module-specific directories
        for module_name in ["zoning", "current_use", "ownership", "improvements"]:
            module_data_dir = output_data_dir / module_name
            if module_data_dir.exists():
                target_dir = self.module_dirs[module_name]
                try:
                    # Move contents of module directory
                    for item in module_data_dir.iterdir():
                        target_path = target_dir / item.name
                        if target_path.exists():
                            target_path.unlink()  # Remove existing file
                        shutil.move(str(item), str(target_path))
                        logger.info(f"📁 Moved {item.name} to {target_path}")

                    # Remove empty module directory
                    module_data_dir.rmdir()
                    logger.info(f"🗑️ Removed empty directory: {module_data_dir}")
                except Exception as e:
                    logger.warning(f"Failed to move {module_data_dir}: {e}")

    def cleanup_output_data_directory(self):
        """
        Clean up the output/data directory after moving data to modules.
        """
        output_data_dir = self.output_dir / "data"
        if not output_data_dir.exists():
            return

        # Check if directory is empty or contains only empty subdirectories
        has_content = False
        for item in output_data_dir.rglob("*"):
            if item.is_file():
                has_content = True
                break

        if not has_content:
            try:
                shutil.rmtree(output_data_dir)
                logger.info("🗑️ Removed empty output/data directory")
            except Exception as e:
                logger.warning(f"Failed to remove output/data directory: {e}")
        else:
            logger.info("📁 Output/data directory still contains files, keeping it")

    def create_module_data_structure(self):
        """
        Create standardized data structure for each module.
        """
        logger.info("📁 Creating standardized module data structure...")

        for module_name, module_dir in self.module_dirs.items():
            # Create subdirectories
            subdirs = ["empirical", "cache", "processed", "raw"]
            for subdir in subdirs:
                (module_dir / subdir).mkdir(parents=True, exist_ok=True)

            # Create metadata file
            metadata = {
                "module_name": module_name,
                "created": datetime.now().isoformat(),
                "data_structure": {
                    "empirical": "Real acquired data",
                    "cache": "H3-processed cached data",
                    "processed": "Final processed outputs",
                    "raw": "Unprocessed source data",
                },
            }

            metadata_path = module_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"📁 Created data structure for {module_name}")

    def update_data_manager_paths(self):
        """
        Update the enhanced data manager to use module-specific paths.
        """
        logger.info("🔄 Updating data manager configuration...")

        # Create a new configuration that points to module directories
        config = {
            "module_data_paths": {},
            "output_summary_path": str(self.output_dir),
            "cache_cleanup_enabled": True,
            "keep_recent_runs": 3,
        }

        for module_name, module_dir in self.module_dirs.items():
            config["module_data_paths"][module_name] = str(module_dir)

        config_path = self.base_dir / "config" / "data_cleanup_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"✅ Updated data manager configuration: {config_path}")

    def _extract_timestamp(self, filename: str) -> str | None:
        """Extract timestamp from filename."""
        import re

        # Look for timestamp pattern: YYYYMMDD_HHMMSS
        pattern = r"(\d{8}_\d{6})"
        match = re.search(pattern, filename)
        return match.group(1) if match else None

    def run_full_cleanup(self, keep_recent_runs: int = 3):
        """
        Run complete cleanup and reorganization.

        Args:
            keep_recent_runs: Number of recent runs to keep
        """
        logger.info("🚀 Starting full data cleanup and reorganization...")

        # Step 1: Create module data structure
        self.create_module_data_structure()

        # Step 2: Move module data to appropriate directories
        self.move_module_data_to_modules()

        # Step 3: Clean up old runs
        self.cleanup_old_runs(keep_recent_runs)

        # Step 4: Clean up output data directory
        self.cleanup_output_data_directory()

        # Step 5: Update data manager configuration
        self.update_data_manager_paths()

        logger.info("✅ Full data cleanup and reorganization completed!")


def create_data_cleanup_manager(base_dir: Path) -> DataCleanupManager:
    """
    Create a data cleanup manager instance.

    Args:
        base_dir: Base directory for the Cascadia framework

    Returns:
        DataCleanupManager instance
    """
    return DataCleanupManager(base_dir)
