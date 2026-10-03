"""
Pytest fixtures for GEO-INFER-GIT tests.

Provides sample git repository paths, commit metadata lists,
git configurations, and standard spatial fixtures.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from typing import Any
import subprocess
import os


@pytest.fixture(autouse=True)
def isolated_git_monitor_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep temporary repositories from starting user-configured Git daemons."""
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", str(count + 1))
    monkeypatch.setenv(f"GIT_CONFIG_KEY_{count}", "core.fsmonitor")
    monkeypatch.setenv(f"GIT_CONFIG_VALUE_{count}", "false")


@pytest.fixture(scope="session")
def sample_coordinates() -> list[tuple[float, float]]:
    """Standard (lat, lng) coordinate pairs for spatial tests."""
    return [
        (47.6062, -122.3321),
        (37.7749, -122.4194),
        (40.7128, -74.0060),
        (51.5074, -0.1278),
        (35.6762, 139.6503),
    ]


@pytest.fixture(scope="function")
def sample_geodataframe() -> gpd.GeoDataFrame:
    """Standard GeoDataFrame with EPSG:4326 for spatial tests."""
    return gpd.GeoDataFrame(
        {"id": range(5), "value": np.random.uniform(0, 100, 5)},
        geometry=[Point(-122.33 + i * 0.01, 47.61 + i * 0.01) for i in range(5)],
        crs="EPSG:4326",
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for test output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_repo_path(tmp_path: Path) -> Path:
    """Temporary directory initialized as a git repository.

    Creates a bare git repo with an initial commit containing a
    README.md file, suitable for testing git operations.
    """
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    readme = repo_dir / "README.md"
    readme.write_text("# Test Repository\n\nInitial content.\n")

    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    return repo_dir


@pytest.fixture
def commit_metadata_list() -> list[dict[str, str]]:
    """List of commit metadata dicts for git analysis tests.

    Contains 5 synthetic commits with author, date, message, and
    files_changed fields representing a realistic commit history.
    """
    return [
        {
            "sha": "a1b2c3d4e5f6",
            "author": "alice@example.com",
            "date": "2024-01-15T10:30:00Z",
            "message": "Add spatial indexing module",
            "files_changed": 3,
        },
        {
            "sha": "b2c3d4e5f6a1",
            "author": "bob@example.com",
            "date": "2024-01-16T14:00:00Z",
            "message": "Fix H3 cell resolution handling",
            "files_changed": 1,
        },
        {
            "sha": "c3d4e5f6a1b2",
            "author": "alice@example.com",
            "date": "2024-01-17T09:15:00Z",
            "message": "Add unit tests for spatial queries",
            "files_changed": 2,
        },
        {
            "sha": "d4e5f6a1b2c3",
            "author": "charlie@example.com",
            "date": "2024-01-18T16:45:00Z",
            "message": "Update documentation for API endpoints",
            "files_changed": 4,
        },
        {
            "sha": "e5f6a1b2c3d4",
            "author": "bob@example.com",
            "date": "2024-01-19T11:00:00Z",
            "message": "Refactor data pipeline for performance",
            "files_changed": 6,
        },
    ]


@pytest.fixture
def git_config() -> dict[str, Any]:
    """Configuration dict for git analysis operations.

    Specifies analysis parameters such as date ranges, file filters,
    and metrics to compute from repository history.
    """
    return {
        "analysis_period": {
            "start": "2024-01-01",
            "end": "2024-12-31",
        },
        "file_filters": ["*.py", "*.ts", "*.md"],
        "ignore_patterns": ["*.pyc", "__pycache__", "node_modules"],
        "metrics": ["commit_frequency", "author_distribution", "churn_rate"],
        "branch": "main",
    }
