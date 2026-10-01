#!/usr/bin/env python3
"""
Data Cleanup Script for Cascadia Framework

This script reorganizes the data structure to:
- Move generated files to generated/ directory
- Clean up old run data
- Maintain run-specific summaries
"""

import logging
import sys
from pathlib import Path
import shutil


def setup_logging():
    """Set up logging for the cleanup script."""
    log_dir = Path(__file__).parent / "generated" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "data_cleanup.log"),
        ],
    )


def cleanup_old_logs(base_dir: Path, keep_recent: int = 3):
    """Move scattered log files to generated/logs/."""
    logger = logging.getLogger(__name__)
    log_dir = base_dir / "generated" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Find log files at root level
    log_files = list(base_dir.glob("*.log"))
    for log_file in log_files:
        dest = log_dir / log_file.name
        shutil.move(str(log_file), str(dest))
        logger.info(f"Moved {log_file.name} to generated/logs/")


def cleanup_pycache(base_dir: Path):
    """Remove __pycache__ directories."""
    logger = logging.getLogger(__name__)
    for pycache in base_dir.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            logger.info(f"Removed {pycache}")


def main():
    """Main cleanup function."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Cascadia data cleanup...")

    base_dir = Path(__file__).parent

    try:
        # Move log files to generated/logs
        cleanup_old_logs(base_dir)

        # Clean up __pycache__ directories
        cleanup_pycache(base_dir)

        logger.info("✅ Data cleanup completed successfully!")
        logger.info("📁 Structure:")
        logger.info("   - src/data_modules/  : Data acquisition modules")
        logger.info("   - src/core/          : Processing utilities")
        logger.info("   - tests/             : Test files")
        logger.info("   - generated/         : All outputs, cache, logs")

    except Exception as e:
        logger.error(f"❌ Data cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
