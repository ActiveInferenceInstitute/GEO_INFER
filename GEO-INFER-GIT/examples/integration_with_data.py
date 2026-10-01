#!/usr/bin/env python3

"""
Integration example: GEO-INFER-GIT with GEO-INFER-DATA

This example demonstrates how GEO-INFER-GIT can be used to automatically
clone and manage geospatial datasets and data processing repositories
for integration with the GEO-INFER-DATA module.

Requires network access to the GitHub API. Set GITHUB_TOKEN to raise rate
limits (optional).
"""

import logging
from pathlib import Path

from geo_infer_git.core.github_api import GitHubAPI
from geo_infer_git.core.repo_cloner import RepoCloner
from geo_infer_git.utils.config_loader import CloneConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_geospatial_data_integration() -> tuple[GitHubAPI, RepoCloner]:
    """
    Set up integration between GEO-INFER-GIT and GEO-INFER-DATA for
    automated geospatial dataset acquisition and management.
    """
    # GitHub API client (uses GITHUB_TOKEN from the environment when set)
    api_client = GitHubAPI()

    # Configuration for cloning operations
    clone_config = CloneConfig(
        output_dir="./geospatial_datasets",
        concurrency_enabled=True,
        max_workers=4,
        default_branch="main",
        clone_depth=1,
    )

    # Initialize repository cloner
    cloner = RepoCloner(clone_config)

    return api_client, cloner


def discover_geospatial_datasets(
    api_client: GitHubAPI, cloner: RepoCloner
) -> dict[str, bool]:
    """
    Discover and clone geospatial datasets from various sources.

    This function demonstrates how to use GEO-INFER-GIT to find and
    clone repositories containing geospatial datasets for use with
    GEO-INFER-DATA.
    """
    logger.info("Discovering geospatial datasets...")

    # List geospatial data repositories published by an organization
    search_results = api_client.get_organization_repositories("USGS", max_repos=50)

    # Filter for repositories likely to contain geospatial data
    geospatial_repos = api_client.filter_repositories(
        search_results,
        min_stars=10,
        languages=["Python", "R", "JavaScript"],
        exclude_forks=True,
        has_topics=["geospatial", "gis", "data", "dataset"],
    )

    logger.info(f"Found {len(geospatial_repos)} geospatial repositories")

    # Clone selected repositories
    logger.info("Cloning geospatial datasets...")
    repositories_to_clone: list[tuple[str, str, str]] = []

    for repo in geospatial_repos[:10]:  # Clone top 10
        repositories_to_clone.append((repo.owner, repo.name, repo.default_branch))

    # Clone repositories
    clone_results = cloner.clone_multiple_repositories(repositories_to_clone)

    # Report results
    success_count = sum(1 for success in clone_results.values() if success)
    total_count = len(clone_results)

    logger.info(f"Cloned {success_count}/{total_count} geospatial datasets")

    return clone_results


def integrate_with_data_module(clone_results: dict[str, bool]) -> dict[str, object]:
    """
    Demonstrate integration with GEO-INFER-DATA module.

    This shows how cloned geospatial datasets can be integrated
    with data processing workflows in GEO-INFER-DATA.
    """
    logger.info("Integrating with GEO-INFER-DATA workflows...")

    datasets_dir = Path("./geospatial_datasets")

    # Scan cloned repositories for data files
    data_files = []
    for repo_path in datasets_dir.rglob("*"):
        if repo_path.is_file():
            # Look for common geospatial data formats
            if repo_path.suffix.lower() in [".geojson", ".shp", ".tif", ".nc", ".csv"]:
                data_files.append(repo_path)

    logger.info(f"Found {len(data_files)} geospatial data files")

    # Example integration points with GEO-INFER-DATA:
    # 1. Register datasets in data catalog
    # 2. Extract metadata and create data schemas
    # 3. Set up data validation pipelines
    # 4. Configure data transformation workflows

    integration_report = {
        "total_datasets": len(clone_results),
        "successful_clones": sum(1 for success in clone_results.values() if success),
        "data_files_discovered": len(data_files),
        "datasets_directory": str(datasets_dir),
        "integration_ready": True,
    }

    return integration_report


def cleanup_and_maintenance(cloner: RepoCloner) -> dict[str, object]:
    """
    Demonstrate maintenance operations for cloned datasets.

    Shows how to manage, update, and clean up cloned repositories
    over time.
    """
    logger.info("Performing maintenance operations...")

    # Get current clone statistics
    stats = cloner.get_clone_stats()
    logger.info(f"Current stats: {stats}")

    # Clean up failed clones
    cleaned_count = cloner.cleanup_failed_clones()
    logger.info(f"Cleaned up {cleaned_count} failed clone attempts")

    # Get disk usage information
    disk_usage = cloner.get_disk_usage()
    logger.info(f"Disk usage: {disk_usage}")

    return {
        "cleanup_performed": cleaned_count > 0,
        "disk_usage": disk_usage,
        "repository_count": stats.get("total", 0),
    }


def main() -> dict[str, object] | None:
    """
    Main integration example demonstrating GEO-INFER-GIT with GEO-INFER-DATA.
    """
    logger.info("Starting GEO-INFER-GIT and GEO-INFER-DATA integration example")

    try:
        # Set up integration components
        api_client, cloner = setup_geospatial_data_integration()

        # Discover and clone geospatial datasets
        clone_results = discover_geospatial_datasets(api_client, cloner)

        # Integrate with data processing workflows
        integration_report = integrate_with_data_module(clone_results)

        # Perform maintenance operations
        maintenance_report = cleanup_and_maintenance(cloner)

        # Generate final report
        final_report = {
            "integration": integration_report,
            "maintenance": maintenance_report,
            "clone_results": clone_results,
        }

        logger.info("Integration example completed successfully")
        logger.info(f"Final report: {final_report}")

        return final_report

    except Exception as e:
        logger.error(f"Integration example failed: {e}")
        raise
    finally:
        # Clean up resources
        if "cloner" in locals():
            cloner.close()
        if "api_client" in locals():
            api_client.close()


if __name__ == "__main__":
    main()
