"""GEO-INFER module integration bridge for Cascadia bioregion analysis.

Each class wraps a GEO-INFER module with graceful degradation — if the module is
not installed, methods return {"available": False, "reason": "..."} instead of
raising exceptions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spatial statistics — GEO-INFER-MATH
# ---------------------------------------------------------------------------
try:
    from geo_infer_math.core.spatial_statistics import MoranI  # type: ignore[import]
    from geo_infer_math.core.interpolation import KrigingInterpolator  # type: ignore[import]

    _MATH_AVAILABLE = True
except ImportError as _e:
    _MATH_AVAILABLE = False
    _MATH_REASON = str(_e)


class CascadiaSpatialStats:
    """Moran's I autocorrelation and Kriging interpolation via GEO-INFER-MATH."""

    def compute_spatial_autocorrelation(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """Compute Moran's I spatial autocorrelation on H3 hexagon scores."""
        if not _MATH_AVAILABLE:
            return {"available": False, "reason": _MATH_REASON}
        try:
            import h3 as h3lib

            scores = {
                k: float(v.get("score", 0.0)) for k, v in h3_data.items() if isinstance(v, dict)
            }
            if len(scores) < 4:
                return {
                    "available": True,
                    "moran_i": None,
                    "p_value": None,
                    "n": len(scores),
                    "note": "Insufficient hexagons for autocorrelation",
                }
            cells = list(scores.keys())
            values = np.array([scores[c] for c in cells])
            coords = np.array([h3lib.cell_to_latlng(c) for c in cells])
            analyzer = MoranI()
            result = analyzer.compute(values, coords)
            return {
                "available": True,
                "moran_i": result.get("I"),
                "p_value": result.get("p_value"),
                "n": len(scores),
                "interpretation": result.get("interpretation", ""),
            }
        except Exception as exc:
            logger.warning("Moran's I computation failed: %s", exc)
            return {"available": True, "error": str(exc)}

    def interpolate_sparse_data(self, known_h3: dict[str, Any], resolution: int) -> dict[str, Any]:
        """Kriging interpolation from sparse H3 observations to full resolution grid."""
        if not _MATH_AVAILABLE:
            return {"available": False, "reason": _MATH_REASON}
        try:
            interpolator = KrigingInterpolator(resolution=resolution)
            result = interpolator.interpolate(known_h3)
            return {"available": True, "interpolated_cells": len(result), "data": result}
        except Exception as exc:
            logger.warning("Kriging interpolation failed: %s", exc)
            return {"available": True, "error": str(exc)}


# ---------------------------------------------------------------------------
# Bayesian uncertainty quantification — GEO-INFER-BAYES
# ---------------------------------------------------------------------------
try:
    from geo_infer_bayes import GaussianProcess  # type: ignore[import]

    _BAYES_AVAILABLE = True
except ImportError as _e:
    _BAYES_AVAILABLE = False
    _BAYES_REASON = str(_e)


class CascadiaBayesianAnalysis:
    """Gaussian Process interpolation and uncertainty quantification via GEO-INFER-BAYES."""

    def estimate_ecological_uncertainty(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """GP posterior mean and std for ecological scores across hexagons."""
        if not _BAYES_AVAILABLE:
            return {"available": False, "reason": _BAYES_REASON}
        try:
            import h3 as h3lib

            scores = {
                k: float(v.get("score", 0.0)) for k, v in h3_data.items() if isinstance(v, dict)
            }
            if not scores:
                return {"available": True, "posterior_mean": {}, "posterior_std": {}, "n": 0}
            cells = list(scores.keys())
            values = np.array([scores[c] for c in cells])
            coords = np.array([h3lib.cell_to_latlng(c) for c in cells])
            gp = GaussianProcess()
            gp.fit(coords, values)
            prediction = gp.predict(coords, return_std=True)
            if isinstance(prediction, tuple):
                posterior_mean_arr, posterior_std_arr = prediction
            else:
                posterior_mean_arr = prediction
                posterior_std_arr = np.zeros_like(posterior_mean_arr)
            posterior_mean = {cells[i]: float(posterior_mean_arr[i]) for i in range(len(cells))}
            posterior_std = {cells[i]: float(posterior_std_arr[i]) for i in range(len(cells))}
            return {
                "available": True,
                "posterior_mean": posterior_mean,
                "posterior_std": posterior_std,
                "n": len(scores),
            }
        except Exception as exc:
            logger.warning("GP estimation failed: %s", exc)
            return {"available": True, "error": str(exc)}


# ---------------------------------------------------------------------------
# Seismic hazard — GEO-INFER-RISK
# ---------------------------------------------------------------------------
try:
    from geo_infer_risk.core.hazard_model import HazardModel  # type: ignore[import]

    _RISK_AVAILABLE = True
except ImportError as _e:
    _RISK_AVAILABLE = False
    _RISK_REASON = str(_e)


class CascadiaSeismicRisk:
    """CSZ seismic hazard per hexagon via GEO-INFER-RISK."""

    def compute_csz_hazard(self, h3_cells: list[str], csz_geojson_path: Path) -> dict[str, Any]:
        """Compute Cascadia Subduction Zone hazard score for each H3 cell."""
        if not _RISK_AVAILABLE:
            return {"available": False, "reason": _RISK_REASON}
        try:
            import h3 as h3lib

            model = HazardModel(
                hazard_type="seismic",
                params={"hazard_type": "ground_shaking", "magnitude": 9.0, "region": "cascadia"},
            )
            events = model.generate_events(1)
            if not events:
                return {
                    "available": True,
                    "error": "No seismic events generated",
                    "n_cells": len(h3_cells),
                }
            event = events[0]
            hazard_scores: dict[str, float] = {}
            for cell in h3_cells:
                lat, lon = h3lib.cell_to_latlng(cell)
                intensity = model.get_intensity_at_location(event, lat, lon)
                hazard_scores[cell] = float(intensity)
            return {"available": True, "hazard_scores": hazard_scores, "n_cells": len(h3_cells)}
        except Exception as exc:
            logger.warning("CSZ hazard computation failed: %s", exc)
            return {"available": True, "error": str(exc)}


# ---------------------------------------------------------------------------
# Forest health — GEO-INFER-FOREST
# ---------------------------------------------------------------------------
try:
    from geo_infer_forest.core.forest_health import ForestHealthAssessor  # type: ignore[import]

    _FOREST_AVAILABLE = True
except ImportError as _e:
    _FOREST_AVAILABLE = False
    _FOREST_REASON = str(_e)


class CascadiaForestHealth:
    """Forest health analysis via GEO-INFER-FOREST."""

    def assess_forest_health(
        self, h3_data: dict[str, Any], ecoregion_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess forest health per hexagon using ecoregion context."""
        if not _FOREST_AVAILABLE:
            return {"available": False, "reason": _FOREST_REASON}
        try:
            assessor = ForestHealthAssessor(ecoregion_data=ecoregion_data)
            results = assessor.assess_health(h3_data, ecoregion_context=ecoregion_data)
            return {"available": True, "results": results, "n_cells": len(h3_data)}
        except Exception as exc:
            logger.warning("Forest health assessment failed: %s", exc)
            return {"available": True, "error": str(exc)}


# ---------------------------------------------------------------------------
# Coastal / salmon habitat — GEO-INFER-MARINE
# ---------------------------------------------------------------------------
try:
    from geo_infer_marine.core.coastal_analyzer import CoastalAnalyzer  # type: ignore[import]

    _MARINE_AVAILABLE = True
except ImportError as _e:
    _MARINE_AVAILABLE = False
    _MARINE_REASON = str(_e)


class CascadiaCoastalAnalysis:
    """Coastal resilience and salmon habitat assessment via GEO-INFER-MARINE."""

    def assess_coastal_resilience(self, h3_data: dict[str, Any]) -> dict[str, Any]:
        """Coastal resilience score per hexagon (derived from vulnerability inversion)."""
        if not _MARINE_AVAILABLE:
            return {"available": False, "reason": _MARINE_REASON}
        try:
            analyzer = CoastalAnalyzer()
            vulnerability = analyzer.assess_coastal_vulnerability(h3_data)
            # Derive resilience as complement of vulnerability
            if isinstance(vulnerability, dict):
                results = {cell: 1.0 - float(score) for cell, score in vulnerability.items()}
            else:
                results = vulnerability
            return {"available": True, "results": results, "n_cells": len(h3_data)}
        except Exception as exc:
            logger.warning("Coastal resilience assessment failed: %s", exc)
            return {"available": True, "error": str(exc)}


# ---------------------------------------------------------------------------
# Ecosystem services valuation — GEO-INFER-ECON
# ---------------------------------------------------------------------------
try:
    from geo_infer_econ.bioregional.bioregional_markets import (  # type: ignore[import]
        BioregionalAsset,
        BioregionalMarketDesign,
    )

    _ECON_AVAILABLE = True
except ImportError as _e:
    _ECON_AVAILABLE = False
    _ECON_REASON = str(_e)


class CascadiaEcosystemServices:
    """Natural capital accounting via GEO-INFER-ECON."""

    def value_ecosystem_services(
        self, h3_data: dict[str, Any], ecoregion_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Estimate ecosystem service values per hexagon using biodiversity market context."""
        if not _ECON_AVAILABLE:
            return {"available": False, "reason": _ECON_REASON}
        try:
            import h3 as h3lib
            import geopandas as gpd
            from shapely.geometry import box

            cascadia_bbox = box(-124.8, 41.8, -114.0, 54.0)
            boundary_gdf = gpd.GeoDataFrame(
                {"name": ["Cascadia"]}, geometry=[cascadia_bbox], crs="EPSG:4326"
            )
            design = BioregionalMarketDesign(bioregion_boundary=boundary_gdf)
            credit_types = ["carbon", "biodiversity", "water"]
            credits_created: list[dict] = []
            for cell_id in list(h3_data.keys())[:5]:  # sample up to 5 cells
                props = h3_data[cell_id] if isinstance(h3_data[cell_id], dict) else {}
                score = float(props.get("score", 0.5))
                lat, lon = h3lib.cell_to_latlng(cell_id)
                asset = BioregionalAsset(
                    asset_id=cell_id,
                    asset_type="natural_land",
                    location=(lat, lon),
                    area_hectares=430.0,  # approx h3 res-7 cell area
                    ecological_attributes={"biodiversity": score, "carbon": score * 0.8},
                    economic_attributes={"value_usd": score * 10000},
                    ownership_type="community",
                    management_regime="conservation",
                    ecosystem_services={st: score for st in credit_types},
                )
                design.register_asset(asset)
                for stype in credit_types:
                    design.create_ecosystem_service_credit(
                        asset_id=cell_id,
                        service_type=stype,
                        quantity=score,
                        quality_parameters={"source": "cascadia_analysis"},
                    )
                    credits_created.append({"cell": cell_id, "type": stype, "quantity": score})
            return {
                "available": True,
                "credits_created": len(credits_created),
                "n_cells": len(h3_data),
                "credit_types": credit_types,
            }
        except Exception as exc:
            logger.warning("Ecosystem services valuation failed: %s", exc)
            return {"available": True, "error": str(exc)}


# ---------------------------------------------------------------------------
# Data quality validation — GEO-INFER-DATA
# ---------------------------------------------------------------------------
try:
    from geo_infer_data.core.validation import DataValidator  # type: ignore[import]

    _DATA_AVAILABLE = True
except ImportError as _e:
    _DATA_AVAILABLE = False
    _DATA_REASON = str(_e)


class CascadiaDataQuality:
    """Data validation and quality scoring via GEO-INFER-DATA."""

    def validate_module_outputs(self, modules_data: dict[str, Any]) -> dict[str, Any]:
        """Quality scores per module output — completeness, consistency, accuracy."""
        if not _DATA_AVAILABLE:
            return {"available": False, "reason": _DATA_REASON}
        try:
            import asyncio

            validator = DataValidator()
            quality_scores = {}
            for module_name, data in modules_data.items():
                # Use sync method if available, otherwise run async
                if hasattr(validator, "validate"):
                    quality_scores[module_name] = validator.validate(data)
                elif hasattr(validator, "validate_async"):
                    quality_scores[module_name] = asyncio.run(validator.validate_async(data))
                else:
                    quality_scores[module_name] = {
                        "score": None,
                        "note": "No validate method found",
                    }
            return {"available": True, "quality_scores": quality_scores}
        except Exception as exc:
            logger.warning("Data quality validation failed: %s", exc)
            return {"available": True, "error": str(exc)}


# ---------------------------------------------------------------------------
# Climate zone analysis — GEO-INFER-CLIMATE
# ---------------------------------------------------------------------------
try:
    from geo_infer_climate.core.climate_processor import ClimateDataProcessor  # type: ignore[import]

    _CLIMATE_AVAILABLE = True
except ImportError as _e:
    _CLIMATE_AVAILABLE = False
    _CLIMATE_REASON = str(_e)


class CascadiaClimateAnalysis:
    """Climate zone overlay and analysis via GEO-INFER-CLIMATE."""

    def assign_climate_zones(
        self, h3_data: dict[str, Any], climate_yaml_path: Path
    ) -> dict[str, Any]:
        """Assign climate zone classifications to each hexagon."""
        if not _CLIMATE_AVAILABLE:
            return {"available": False, "reason": _CLIMATE_REASON}
        if not climate_yaml_path.exists():
            return {"available": True, "error": f"Climate YAML not found: {climate_yaml_path}"}
        try:
            processor = ClimateDataProcessor(config_path=str(climate_yaml_path))
            # Try classify_cells first (expected h3-native method), fall back to assign_zones
            if hasattr(processor, "classify_cells"):
                results = processor.classify_cells(h3_data)
            elif hasattr(processor, "assign_zones"):
                results = processor.assign_zones(h3_data)
            else:
                return {
                    "available": True,
                    "error": "ClimateDataProcessor has no classify_cells or assign_zones method",
                }
            return {"available": True, "results": results, "n_cells": len(h3_data)}
        except Exception as exc:
            logger.warning("Climate zone assignment failed: %s", exc)
            return {"available": True, "error": str(exc)}


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_integration_suite() -> dict[str, Any]:
    """Return all integration wrappers keyed by domain name."""
    return {
        "spatial_stats": CascadiaSpatialStats(),
        "bayesian": CascadiaBayesianAnalysis(),
        "seismic_risk": CascadiaSeismicRisk(),
        "forest_health": CascadiaForestHealth(),
        "coastal": CascadiaCoastalAnalysis(),
        "ecosystem_services": CascadiaEcosystemServices(),
        "data_quality": CascadiaDataQuality(),
        "climate": CascadiaClimateAnalysis(),
    }


def get_availability_report() -> dict[str, bool]:
    """Return which GEO-INFER modules are available."""
    return {
        "geo_infer_math": _MATH_AVAILABLE,
        "geo_infer_bayes": _BAYES_AVAILABLE,
        "geo_infer_risk": _RISK_AVAILABLE,
        "geo_infer_forest": _FOREST_AVAILABLE,
        "geo_infer_marine": _MARINE_AVAILABLE,
        "geo_infer_econ": _ECON_AVAILABLE,
        "geo_infer_data": _DATA_AVAILABLE,
        "geo_infer_climate": _CLIMATE_AVAILABLE,
    }
