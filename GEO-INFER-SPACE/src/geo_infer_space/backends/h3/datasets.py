"""H3 dataset loading, validation, and export utilities."""

import logging
from typing import Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

from .core import H3Grid, H3Cell


class H3Dataset:
    """
    Container for H3 grid datasets with metadata and utilities.

    Provides methods for managing H3 grid data with associated metadata,
    validation, and export capabilities.
    """

    def __init__(self, grid: H3Grid, metadata: dict[str, Any] | None = None) -> None:
        """
        Initialize H3Dataset with grid and metadata.

        Args:
            grid: H3Grid instance
            metadata: Optional metadata dictionary
        """
        self.grid = grid
        self.metadata = metadata or {}
        self.created_at = datetime.now()

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata entry."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata entry."""
        return self.metadata.get(key, default)

    def validate(self) -> dict[str, Any]:
        """
        Validate dataset integrity.

        Returns:
            Validation report dictionary
        """
        report: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        # Check grid
        if not self.grid or not self.grid.cells:
            report["valid"] = False
            report["errors"].append("Empty or missing grid")
            return report

        # Check data consistency
        property_keys: set[str] = set()
        for cell in self.grid.cells:
            property_keys.update(cell.properties.keys())

        # Check for missing properties
        for cell in self.grid.cells:
            missing_keys = property_keys - set(cell.properties.keys())
            if missing_keys:
                report["warnings"].append(
                    f"Cell {cell.index} missing properties: {missing_keys}"
                )

        # Statistics
        report["statistics"] = {
            "cell_count": len(self.grid.cells),
            "property_keys": list(property_keys),
            "metadata_keys": list(self.metadata.keys()),
        }

        return report

    def export_json(self, filepath: str) -> None:
        """Export dataset to JSON file."""
        data = {
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "grid": self.grid.to_geojson(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Dataset exported to {filepath}")


class H3DataLoader:
    """
    Utilities for loading H3 datasets from various sources.

    Provides methods for loading H3 grid data from files,
    databases, and other data sources.
    """

    @staticmethod
    def from_geojson(filepath: str, name: str = "Loaded_Grid") -> H3Dataset:
        """
        Load H3Dataset from GeoJSON file.

        Args:
            filepath: Path to GeoJSON file
            name: Name for the loaded grid

        Returns:
            H3Dataset instance
        """
        try:
            with open(filepath) as f:
                geojson_data = json.load(f)

            # Extract cells from GeoJSON features
            cells = []
            for feature in geojson_data.get("features", []):
                properties = feature.get("properties", {})
                h3_index = properties.get("h3_index")
                resolution = properties.get("resolution")

                if h3_index and resolution is not None:
                    cell = H3Cell(index=h3_index, resolution=resolution)
                    # Add properties (excluding H3-specific ones)
                    for key, value in properties.items():
                        if key not in [
                            "h3_index",
                            "resolution",
                            "latitude",
                            "longitude",
                            "area_km2",
                        ]:
                            cell.properties[key] = value
                    cells.append(cell)

            # Create grid and dataset
            grid = H3Grid(cells=cells, name=name)

            # Extract metadata from GeoJSON properties
            metadata = geojson_data.get("properties", {})

            return H3Dataset(grid=grid, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to load GeoJSON from {filepath}: {e}")
            return H3Dataset(grid=H3Grid(name=name))

    @staticmethod
    def from_csv(
        filepath: str,
        h3_column: str = "h3_index",
        resolution_column: str = "resolution",
        name: str = "CSV_Grid",
    ) -> H3Dataset:
        """
        Load H3Dataset from CSV file.

        Args:
            filepath: Path to CSV file
            h3_column: Column name containing H3 indices
            resolution_column: Column name containing resolutions
            name: Name for the loaded grid

        Returns:
            H3Dataset instance
        """
        try:
            # Simple CSV parsing without pandas
            cells = []
            with open(filepath) as f:
                lines = f.readlines()

                if not lines:
                    return H3Dataset(grid=H3Grid(name=name))

                # Parse header
                header = [col.strip() for col in lines[0].split(",")]

                if h3_column not in header or resolution_column not in header:
                    logger.error(
                        f"Required columns not found: {h3_column}, {resolution_column}"
                    )
                    return H3Dataset(grid=H3Grid(name=name))

                h3_idx = header.index(h3_column)
                res_idx = header.index(resolution_column)

                # Parse data rows
                for line in lines[1:]:
                    values = [val.strip() for val in line.split(",")]

                    if len(values) <= max(h3_idx, res_idx):
                        continue

                    h3_index = values[h3_idx]
                    try:
                        resolution = int(values[res_idx])
                    except ValueError:
                        continue

                    cell = H3Cell(index=h3_index, resolution=resolution)

                    # Add other columns as properties
                    for i, col_name in enumerate(header):
                        if i < len(values) and col_name not in [
                            h3_column,
                            resolution_column,
                        ]:
                            value: float | str
                            try:
                                # Try to convert to number
                                value = float(values[i])
                                if value.is_integer():
                                    value = int(value)
                            except ValueError:
                                # Keep as string
                                value = values[i]

                            cell.properties[col_name] = value

                    cells.append(cell)

            grid = H3Grid(cells=cells, name=name)
            return H3Dataset(grid=grid)

        except Exception as e:
            logger.error(f"Failed to load CSV from {filepath}: {e}")
            return H3Dataset(grid=H3Grid(name=name))


class H3DataExporter:
    """
    Utilities for exporting H3 datasets to various formats.

    Provides methods for exporting H3 grid data to files
    and other data formats.
    """

    @staticmethod
    def to_geojson(dataset: H3Dataset, filepath: str) -> None:
        """Export dataset to GeoJSON file."""
        dataset.export_json(filepath)

    @staticmethod
    def to_csv(dataset: H3Dataset, filepath: str) -> None:
        """
        Export dataset to CSV file.

        Args:
            dataset: H3Dataset to export
            filepath: Output CSV file path
        """
        try:
            # Collect all property keys
            all_keys: set[str] = set()
            for cell in dataset.grid.cells:
                all_keys.update(cell.properties.keys())

            # Create header
            header = ["h3_index", "resolution", "latitude", "longitude", "area_km2"]
            header.extend(sorted(all_keys))

            # Write CSV
            with open(filepath, "w") as f:
                # Write header
                f.write(",".join(header) + "\n")

                # Write data rows
                for cell in dataset.grid.cells:
                    row = [
                        cell.index,
                        str(cell.resolution),
                        str(cell.latitude),
                        str(cell.longitude),
                        str(cell.area_km2),
                    ]

                    # Add property values
                    for key in sorted(all_keys):
                        value = cell.properties.get(key, "")
                        row.append(str(value))

                    f.write(",".join(row) + "\n")

            logger.info(f"Dataset exported to CSV: {filepath}")

        except Exception as e:
            logger.error(f"Failed to export CSV to {filepath}: {e}")
