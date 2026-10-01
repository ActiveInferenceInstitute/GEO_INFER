"""
API endpoints for SPM analysis

This module provides REST API endpoints for performing SPM analysis
through web services, enabling integration with web applications
and distributed computing environments.
"""

from typing import Any
import numpy as np

from ..models.data_models import SPMData, SPMResult, DesignMatrix
from ..core.glm import fit_glm
from ..core.contrasts import contrast
from ..core.rft import compute_spm


class SPMAPI:
    """
    REST API interface for SPM analysis.

    Provides endpoints for data upload, model fitting, statistical testing,
    and result retrieval in a web service format.
    """

    def __init__(self) -> None:
        self.datasets: dict[str, Any] = {}  # Store uploaded datasets
        self.results: dict[str, Any] = {}  # Store analysis results
        self.next_id = 1

    def upload_data(self, data: dict[str, Any], format: str = "json") -> dict[str, Any]:
        """
        Upload geospatial data for analysis.

        Args:
            data: Data dictionary containing coordinates and values
            format: Data format ('json', 'csv', 'geojson')

        Returns:
            Response with dataset ID
        """
        try:
            dataset_id = f"dataset_{self.next_id}"
            self.next_id += 1

            # Convert to SPMData
            if format == "json":
                spm_data = self._json_to_spmdata(data)
            elif format == "csv":
                spm_data = self._csv_to_spmdata(data)
            elif format == "geojson":
                spm_data = self._geojson_to_spmdata(data)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.datasets[dataset_id] = spm_data

            return {
                "status": "success",
                "dataset_id": dataset_id,
                "n_points": spm_data.n_points,
                "has_temporal": spm_data.has_temporal,
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def fit_model(
        self, dataset_id: str, design_spec: dict[str, Any], method: str = "OLS"
    ) -> dict[str, Any]:
        """
        Fit GLM to uploaded dataset.

        Args:
            dataset_id: ID of uploaded dataset
            design_spec: Design matrix specification
            method: Fitting method

        Returns:
            Response with model fit results
        """
        try:
            if dataset_id not in self.datasets:
                raise ValueError(f"Dataset {dataset_id} not found")

            data = self.datasets[dataset_id]

            # Create design matrix from specification
            design_matrix = self._create_design_from_spec(design_spec, data)

            # Fit model
            result = fit_glm(data, design_matrix, method=method)

            result_id = f"result_{self.next_id}"
            self.next_id += 1
            self.results[result_id] = result

            return {
                "status": "success",
                "result_id": result_id,
                "r_squared": result.model_diagnostics.get("r_squared", 0),
                "n_regressors": design_matrix.n_regressors,
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run_contrast(
        self,
        result_id: str,
        contrast_spec: dict[str, Any],
        correction: str = "uncorrected",
    ) -> dict[str, Any]:
        """
        Run statistical contrast on fitted model.

        Args:
            result_id: ID of fitted model results
            contrast_spec: Contrast specification
            correction: Multiple comparison correction method

        Returns:
            Response with contrast results
        """
        try:
            if result_id not in self.results:
                raise ValueError(f"Result {result_id} not found")

            model_result = self.results[result_id]

            # Create contrast
            if "vector" in contrast_spec:
                contrast_obj = contrast(model_result, contrast_spec["vector"])
            elif "string" in contrast_spec:
                contrast_obj = contrast(model_result, contrast_spec["string"])
            else:
                raise ValueError(
                    "Contrast specification must include 'vector' or 'string'"
                )

            # Apply correction
            spm_result = compute_spm(model_result, contrast_obj, correction=correction)

            # Record the contrast on the stored model result so that
            # get_results(summary|full) report it.
            model_result.contrasts.append(spm_result)
            self.results[result_id] = model_result

            return {
                "status": "success",
                "result_id": result_id,
                "contrast_name": (
                    contrast_obj.name if hasattr(contrast_obj, "name") else "unnamed"
                ),
                "correction_method": correction,
                "n_significant": spm_result.n_significant,
                "threshold": spm_result.threshold,
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_results(self, result_id: str, format: str = "summary") -> dict[str, Any]:
        """
        Retrieve analysis results.

        Args:
            result_id: ID of analysis results
            format: Result format ('summary', 'full', 'visualization'). Note
                that 'visualization' returns a raw JSON-ready payload
                (coordinates, beta map, residuals) for client-side plotting,
                not a rendered figure object.

        Returns:
            Response with results
        """
        try:
            if result_id not in self.results:
                raise ValueError(f"Result {result_id} not found")

            result = self.results[result_id]

            if format == "summary":
                return self._format_summary(result)
            elif format == "full":
                return self._format_full(result)
            elif format == "visualization":
                return self._format_visualization(result)
            else:
                raise ValueError(f"Unknown format: {format}")

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_datasets(self) -> dict[str, Any]:
        """List all uploaded datasets."""
        return {
            "status": "success",
            "datasets": list(self.datasets.keys()),
            "count": len(self.datasets),
        }

    def list_results(self) -> dict[str, Any]:
        """List all analysis results."""
        return {
            "status": "success",
            "results": list(self.results.keys()),
            "count": len(self.results),
        }

    def _json_to_spmdata(self, data: dict[str, Any]) -> SPMData:
        """Convert JSON data to SPMData object."""
        from ..models.data_models import SPMData

        coordinates = np.array(data["coordinates"], dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError(
                "json payload 'coordinates' must be a (n_points, >=2) array of "
                "[lon, lat] pairs"
            )
        data_values = np.array(data.get("data", []))

        return SPMData(
            data=data_values,
            coordinates=coordinates,
            time=data.get("time"),
            covariates=data.get("covariates", {}),
            metadata=data.get("metadata", {}),
            crs=data.get("crs", "EPSG:4326"),
        )

    def _csv_to_spmdata(self, data: dict[str, Any]) -> SPMData:
        """Convert CSV-like data to SPMData object.

        Expects data dict with either:
          - 'rows': list of dicts, each with coordinate and value keys
          - 'columns': dict mapping column names to lists of values

        Coordinate columns are identified by keys containing 'lat', 'lon',
        'x', 'y', or 'easting'/'northing'. Remaining numeric columns become
        the data values.

        Args:
            data: CSV-like data dictionary

        Returns:
            SPMData object
        """
        import numpy as np

        coord_keys_lat = {"lat", "latitude", "y", "northing"}
        coord_keys_lon = {"lon", "lng", "longitude", "x", "easting"}

        rows = data.get("rows", [])
        columns = data.get("columns", {})

        # Convert column-oriented to row-oriented if needed
        if columns and not rows:
            col_names = list(columns.keys())
            n_rows = len(next(iter(columns.values())))
            rows = [{c: columns[c][i] for c in col_names} for i in range(n_rows)]

        if not rows:
            raise ValueError("CSV data must contain 'rows' or 'columns'")

        # Identify coordinate and value columns
        all_keys = set(rows[0].keys())
        lat_col = next((k for k in all_keys if k.lower() in coord_keys_lat), None)
        lon_col = next((k for k in all_keys if k.lower() in coord_keys_lon), None)

        if not lat_col or not lon_col:
            raise ValueError(
                f"Could not identify coordinate columns from keys: {all_keys}. "
                "Expected columns matching lat/lon/x/y/easting/northing."
            )

        value_cols = sorted(all_keys - {lat_col, lon_col})
        # Filter to numeric-looking value columns
        numeric_cols = []
        for col in value_cols:
            try:
                float(rows[0][col])
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue  # Skip non-numeric columns

        # Coordinates follow the SPMData (lon, lat) convention, matching the
        # geojson helper and interactive-map rendering.
        coordinates = np.array([[float(r[lon_col]), float(r[lat_col])] for r in rows])
        data_values = np.array(
            [[float(r.get(c, 0)) for c in numeric_cols] for r in rows]
        )

        return SPMData(
            data=data_values,
            coordinates=coordinates,
            time=data.get("time"),
            covariates=data.get("covariates", {}),
            metadata={
                "source_format": "csv",
                "coordinate_columns": [lon_col, lat_col],
                "value_columns": numeric_cols,
                **(data.get("metadata", {})),
            },
            crs=data.get("crs", "EPSG:4326"),
        )

    def _geojson_to_spmdata(self, data: dict[str, Any]) -> SPMData:
        """Convert GeoJSON FeatureCollection data to SPMData object.

        Expects a GeoJSON ``FeatureCollection`` whose features carry Point
        geometries. Coordinates are read as [lon, lat] per the GeoJSON spec.
        Per-feature numeric values are taken from the ``value`` property when
        present, otherwise from the first numeric property encountered.
        """
        if data.get("type") != "FeatureCollection" or not data.get("features"):
            raise ValueError(
                "GeoJSON data must be a FeatureCollection with a non-empty "
                "'features' list"
            )

        coordinates = []
        data_values = []
        for feature in data["features"]:
            geometry = feature.get("geometry") or {}
            if geometry.get("type") != "Point":
                raise ValueError(
                    f"Unsupported GeoJSON geometry type: {geometry.get('type')!r}; "
                    "only 'Point' features are supported"
                )
            lonlat = geometry.get("coordinates")
            if not isinstance(lonlat, (list, tuple)) or len(lonlat) < 2:
                raise ValueError(
                    "GeoJSON Point coordinates must contain at least [lon, lat]"
                )
            coordinates.append([float(lonlat[0]), float(lonlat[1])])

            properties = feature.get("properties") or {}
            if "value" in properties:
                data_values.append(float(properties["value"]))
            else:
                for prop in properties.values():
                    try:
                        data_values.append(float(prop))
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    raise ValueError(
                        "GeoJSON feature has no 'value' property and no numeric "
                        "property to use as data value"
                    )

        return SPMData(
            data=np.array(data_values),
            coordinates=np.array(coordinates),
            covariates=data.get("covariates", {}),
            metadata={"source_format": "geojson", **(data.get("metadata", {}))},
            crs=data.get("crs", "EPSG:4326"),
        )

    def _create_design_from_spec(
        self, design_spec: dict[str, Any], data: SPMData
    ) -> DesignMatrix:
        """Create design matrix from API specification."""
        from ..utils.helpers import create_design_matrix

        return create_design_matrix(
            data,
            covariates=design_spec.get("covariates", []),
            factors=design_spec.get("factors", {}),
            intercept=design_spec.get("intercept", True),
        )

    def _format_summary(self, result: SPMResult) -> dict[str, Any]:
        """Format results as summary."""
        return {
            "status": "success",
            "result_type": "SPMResult",
            "n_points": result.spm_data.n_points,
            "n_regressors": result.design_matrix.n_regressors,
            "r_squared": result.r_squared,
            "log_likelihood": result.log_likelihood,
            "n_contrasts": len(result.contrasts),
            "significant_contrasts": sum(
                1 for c in result.contrasts if c.n_significant > 0
            ),
        }

    def _format_full(self, result: SPMResult) -> dict[str, Any]:
        """Format full results."""
        return {
            "status": "success",
            "result": {
                "beta_coefficients": result.beta_coefficients.tolist(),
                "residuals": result.residuals.tolist(),
                "model_diagnostics": result.model_diagnostics,
                "processing_metadata": result.processing_metadata,
                "contrasts": [
                    {
                        "name": getattr(c, "name", "unnamed"),
                        "n_significant": c.n_significant,
                        "correction_method": c.correction_method,
                    }
                    for c in result.contrasts
                ],
            },
        }

    def _format_visualization(self, result: SPMResult) -> dict[str, Any]:
        """Format results for visualization."""
        # This would create visualization data structures
        # For now, return basic structure
        return {
            "status": "success",
            "visualization_data": {
                "coordinates": result.spm_data.coordinates.tolist(),
                "beta_map": (
                    result.beta_coefficients.tolist()
                    if result.beta_coefficients.ndim == 1
                    else result.beta_coefficients[:, 0].tolist()
                ),
                "residuals": result.residuals.tolist(),
            },
        }
