# Getting Started with GEO-INFER-FOREST

This guide covers installation, core concepts, and first working examples for forest inventory, carbon modeling, and canopy analysis.

## Installation

Install the module in editable mode using `uv`:

```bash
uv pip install -e ./GEO-INFER-FOREST
```

Verify the installation:

```python
import geo_infer_forest
print(geo_infer_forest.__version__)
# 0.3.0
```

## Core Concepts

### Spatial Forest Inventory

GEO-INFER-FOREST treats forests as spatially explicit datasets. Every analysis -- from biomass estimation to deforestation detection -- operates on georeferenced rasters (`xarray.DataArray`); vector stand polygons are supported via the optional `vector` extra (`uv sync --package geo-infer-forest --extra vector` in this workspace; `pip install geo-infer-forest[vector]` is not available yet, since the distribution is not published to PyPI).

The standard unit of measurement is tons per hectare (t/ha) for biomass and tons of carbon per hectare (tC/ha) for carbon stock.

### Vegetation Index Foundations

Many forest analyses begin with spectral vegetation indices computed from satellite imagery:

| Index | Formula | Use |
|-------|---------|-----|
| NDVI | (NIR - Red) / (NIR + Red) | General vegetation health, canopy cover |
| EVI | G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L) | Reduced saturation in dense forests |

The `CanopyAnalyzer` computes both indices and derives higher-order products such as fractional vegetation cover, leaf area index, and canopy gap maps.

### Biomass-Carbon Relationship

The standard pipeline from remote sensing to carbon credits follows these steps:

1. **Canopy cover** from NDVI via the linear FVC model.
2. **Biomass** from canopy cover and tree density (default: 100 t/ha for mature forest at 100% cover).
3. **Carbon stock** from biomass at 50% carbon fraction.
4. **CO2 equivalent** from carbon at 3.67x molecular weight ratio.
5. **Carbon credit value** from CO2-eq at configurable market price (default: $50/tCO2).

### Change Detection

The `DeforestationDetector` supports two approaches:

- **Two-date comparison**: Subtracts vegetation index between two dates. Flags pixels where NDVI drops exceed a threshold (default 0.15) and the initial NDVI was above 0.3.
- **Time-series analysis**: Computes z-scores against a rolling baseline mean. Flags statistically significant decreases at the configured confidence level (default 0.95).

## First Example: Biomass Estimation

Estimate above-ground biomass from forest cover and tree density data.

```python
import numpy as np
import xarray as xr
from geo_infer_forest.core.forest_inventory import ForestInventory

# Initialize inventory
inventory = ForestInventory()

# Create sample forest cover data (percent, 20x20 grid)
lat = np.linspace(42.0, 42.5, 20)
lon = np.linspace(-124.0, -123.5, 20)

forest_cover = xr.DataArray(
    np.random.uniform(20, 95, (20, 20)),
    dims=["lat", "lon"],
    coords={"lat": lat, "lon": lon},
    attrs={"units": "percent"},
)

# Optional tree density data (trees per hectare)
tree_density = xr.DataArray(
    np.random.uniform(100, 800, (20, 20)),
    dims=["lat", "lon"],
    coords={"lat": lat, "lon": lon},
)

# Estimate biomass
biomass = inventory.estimate_biomass(forest_cover, tree_density)
print(f"Mean biomass: {float(biomass.mean()):.1f} t/ha")
print(f"Range: {float(biomass.min()):.1f} - {float(biomass.max()):.1f} t/ha")

# Calculate total forest area
forest_area = inventory.calculate_forest_area(forest_cover)
print(f"Total forest area: {float(forest_area.sum()):.2f} km2")
```

## Second Example: Carbon Stock and Credits

Convert biomass to carbon stock and estimate carbon credit value.

```python
from geo_infer_forest.core.carbon_sequestration import CarbonSequestrationModeler

modeler = CarbonSequestrationModeler()

# Calculate carbon stock from biomass
carbon_stock = modeler.calculate_carbon_stock(biomass)
print(f"Mean carbon stock: {float(carbon_stock.mean()):.1f} tC/ha")

# Estimate sequestration rate from annual biomass growth
# Assume 2-5% annual growth
biomass_growth = biomass * np.random.uniform(0.02, 0.05, biomass.shape)
sequestration_rate = modeler.estimate_sequestration_rate(biomass_growth)
print(f"Mean sequestration rate: {float(sequestration_rate.mean()):.2f} tC/ha/year")

# Calculate carbon credit value
area_ha = xr.DataArray(
    np.full((20, 20), 10.0),  # 10 hectares per cell
    dims=["lat", "lon"],
    coords={"lat": lat, "lon": lon},
)

credit_value = modeler.calculate_carbon_credits(
    carbon_sequestration=sequestration_rate,
    area=area_ha,
    price_per_ton=50.0,  # $50/tCO2
)
print(f"Total annual credit value: ${float(credit_value.sum()):,.0f}")
```

## Third Example: NDVI and Canopy Cover

Compute vegetation indices from satellite band data and derive canopy cover.

```python
from geo_infer_forest.core.canopy_analysis import CanopyAnalyzer

analyzer = CanopyAnalyzer()

# Simulate satellite bands (scaled 0-1 reflectance)
red = xr.DataArray(
    np.random.uniform(0.02, 0.12, (20, 20)),
    dims=["lat", "lon"],
    coords={"lat": lat, "lon": lon},
)
nir = xr.DataArray(
    np.random.uniform(0.20, 0.55, (20, 20)),
    dims=["lat", "lon"],
    coords={"lat": lat, "lon": lon},
)

# Calculate NDVI
ndvi = analyzer.calculate_ndvi(red, nir)
print(f"NDVI range: {float(ndvi.min()):.3f} to {float(ndvi.max()):.3f}")

# Estimate canopy cover
canopy_cover = analyzer.estimate_canopy_cover(ndvi, method="linear")
print(f"Mean canopy cover: {float(canopy_cover.mean()):.1f}%")

# Estimate Leaf Area Index
lai = analyzer.estimate_leaf_area_index(ndvi)
print(f"Mean LAI: {float(lai.mean()):.2f} m2/m2")

# Detect canopy gaps
gaps = analyzer.detect_canopy_gaps(ndvi)
print(f"Gap fraction: {gaps.attrs['gap_fraction']:.3f}")

# Classify canopy density (0-4 categories)
density_class = analyzer.classify_canopy_density(ndvi)
print(f"Dense canopy pixels (class 3+): {int((density_class >= 3).sum())}")
```

## Fourth Example: Deforestation Detection

Detect forest loss using two-date change analysis.

```python
from geo_infer_forest.core.deforestation import DeforestationDetector

detector = DeforestationDetector()

# NDVI before and after (simulate clearing in southeast corner)
ndvi_before = xr.DataArray(
    np.random.uniform(0.5, 0.85, (20, 20)),
    dims=["lat", "lon"],
    coords={"lat": lat, "lon": lon},
)
ndvi_after = ndvi_before.copy()
ndvi_after[15:, 15:] -= np.random.uniform(0.2, 0.4, (5, 5))

# Detect change
change_result = detector.detect_change_two_date(ndvi_before, ndvi_after)

print(f"Deforestation rate: {change_result.attrs['deforestation_rate']:.3f}")
print(f"Deforested pixels: {change_result.attrs['deforested_pixel_count']}")
```

## Next Steps

- Read the [API Reference](api_reference.md) for the full method catalog.
- Try the [Carbon Stock Example](examples/basic_example.md) for an end-to-end workflow.
- See the [Habitat Connectivity Example](examples/advanced_example.md) for fragmentation and corridor analysis.
