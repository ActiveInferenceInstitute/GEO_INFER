#!/usr/bin/env python3
"""GEO-INFER-SPM module orchestrator.

Runs one documented end-to-end SPM operation on synthetic data: generate a
100-point synthetic geospatial field with a north-south trend on a regular
grid, fit a general linear model with elevation and temperature covariates,
evaluate a contrast, and threshold the resulting statistical parametric map
with random-field-theory correction. All work goes through the real
``geo_infer_spm`` public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    import numpy as np

    from geo_infer_spm import (
        SPMData,
        compute_spm,
        contrast,
        create_design_matrix,
        fit_glm,
        generate_synthetic_data,
    )

    # Synthetic 10x10 regular grid over a fictional study region.
    lons = np.linspace(-124.10, -124.00, 10)
    lats = np.linspace(41.50, 41.60, 10)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    coordinates = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    spm_data: SPMData = generate_synthetic_data(
        coordinates,
        effects={"intercept": 10.0, "trend": "north_south"},
        noise_level=0.5,
        random_seed=11,
    )

    design = create_design_matrix(spm_data, covariates=["elevation", "temperature"])
    model_result = fit_glm(spm_data, design, method="OLS")
    contrast_result = contrast(model_result, np.array([0.0, 1.0, 0.0]))
    spm_map = compute_spm(model_result, contrast_result, correction="RFT")

    beta = np.asarray(model_result.beta_coefficients, dtype=float).reshape(-1)
    t_values = np.asarray(contrast_result.t_statistic, dtype=float).reshape(-1)
    p_values = np.asarray(contrast_result.p_values, dtype=float).reshape(-1)
    corrected = (
        None
        if spm_map.corrected_p_values is None
        else np.asarray(spm_map.corrected_p_values, dtype=float).reshape(-1)
    )
    sig_mask = (
        None
        if spm_map.significance_mask is None
        else np.asarray(spm_map.significance_mask, dtype=bool).reshape(-1)
    )
    return {
        "operation": "glm_fit_contrast_and_rft_map",
        "n_points": int(spm_data.n_points),
        "n_regressors": int(beta.size),
        "beta_estimates": [float(b) for b in beta],
        "contrast": {
            "weights": [0.0, 1.0, 0.0],
            "t_statistic_max": float(np.max(np.abs(t_values))),
            "p_value_min": float(np.min(p_values)),
        },
        "spm_map": {
            "correction": str(spm_map.correction_method),
            "threshold": (
                float(spm_map.threshold) if spm_map.threshold is not None else None
            ),
            "n_significant_points": (
                int(np.sum(sig_mask)) if sig_mask is not None else None
            ),
            "corrected_p_value_min": (
                float(np.min(corrected)) if corrected is not None else None
            ),
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("SPM", _operation))
