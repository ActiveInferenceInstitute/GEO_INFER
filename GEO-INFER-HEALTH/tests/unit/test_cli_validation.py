"""Exit-code contract for the ``geo-infer-health validate`` CLI command.

Covers the documented contract that a failed validation makes the CLI exit
non-zero (via :func:`geo_infer_health.cli.main`) instead of silently
succeeding, and that the dead ``--schema``/``--type`` flags are gone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from pyogrio.errors import DataSourceError
from shapely.geometry import Point

from geo_infer_health.cli import main, run_validation, setup_cli

MODULE_ROOT = Path(__file__).resolve().parents[2]
MODULE_CONFIG = MODULE_ROOT / "config" / "health_config.yaml"


@pytest.fixture
def valid_input_path(tmp_path: Path) -> Path:
    """A single-feature GeoJSON file that should validate cleanly."""
    gdf = gpd.GeoDataFrame(
        {"report_id": ["r1"], "case_count": [5]},
        geometry=[Point(0.0, 0.0)],
        crs="EPSG:4326",
    )
    path = tmp_path / "reports.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture
def empty_input_path(tmp_path: Path) -> Path:
    """A structurally valid GeoJSON file containing zero features."""
    path = tmp_path / "empty.geojson"
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def cli_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run ``main`` with the given CLI argv (without the program name).

    The working directory is a temporary path so the configured relative
    ``logs/health.log`` file sink never lands in the repository.
    """

    def _run(*argv: str) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["geo-infer-health", *argv])
        main()

    return _run


class TestRunValidationFailure:
    """run_validation must raise on every failure mode, never return silently."""

    def test_missing_input_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.geojson"
        args = __import__("argparse").Namespace(input=str(missing))

        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            run_validation(args, config=None)

    def test_unreadable_input_raises(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.geojson"
        corrupt.write_text("{not geojson at all", encoding="utf-8")
        args = __import__("argparse").Namespace(input=str(corrupt))

        with pytest.raises(DataSourceError):
            run_validation(args, config=None)

    def test_empty_file_raises_value_error(self, empty_input_path: Path) -> None:
        args = __import__("argparse").Namespace(input=str(empty_input_path))

        with pytest.raises(ValueError, match="no data"):
            run_validation(args, config=None)


class TestRunValidationSuccess:
    def test_valid_file_passes(self, valid_input_path: Path) -> None:
        args = __import__("argparse").Namespace(input=str(valid_input_path))

        run_validation(args, config=None)


class TestValidateExitCode:
    """The validate subcommand must exit non-zero on failure via main()."""

    def test_nonexistent_input_exits_nonzero(self, cli_argv, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.geojson"

        with pytest.raises(SystemExit) as excinfo:
            cli_argv(
                "--config",
                str(MODULE_CONFIG),
                "validate",
                "--input",
                str(missing),
            )

        assert excinfo.value.code != 0

    def test_valid_input_exits_zero(self, cli_argv, valid_input_path: Path) -> None:
        cli_argv(
            "--config",
            str(MODULE_CONFIG),
            "validate",
            "--input",
            str(valid_input_path),
        )

    def test_empty_input_exits_nonzero(self, cli_argv, empty_input_path: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli_argv(
                "--config",
                str(MODULE_CONFIG),
                "validate",
                "--input",
                str(empty_input_path),
            )

        assert excinfo.value.code != 0


class TestValidateParserSurface:
    """--schema/--type were parsed but never used; they must be rejected."""

    def test_schema_flag_is_rejected(self) -> None:
        parser = setup_cli()

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(
                ["validate", "--input", "x.geojson", "--schema", "s.json"]
            )

        assert excinfo.value.code == 2

    def test_type_flag_is_rejected(self) -> None:
        parser = setup_cli()

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["validate", "--input", "x.geojson", "--type", "disease"])

        assert excinfo.value.code == 2
