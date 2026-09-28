# Microeconomics Guide

## Introduction

This guide covers microeconomic analysis capabilities in GEO-INFER-ECON, focusing on individual market behavior and spatial microeconomics.

## Location Theory

### Central Place Theory

```python
from geo_infer_econ.microeconomics.market_structure import SpatialMarketAnalysis

# Delineate geographic market hierarchy from price integration
analyzer = SpatialMarketAnalysis()

hierarchy = analyzer.delineate_geographic_markets(
    price_data=price_panel,
    locations=city_ids,
)
```

### Bid-Rent Model

The package does not implement a bid-rent curve model. For land-value
market structure, use the spatial market analysis surface:

```python
from geo_infer_econ.microeconomics.market_structure import SpatialMarketAnalysis

model = SpatialMarketAnalysis()

land_markets = model.delineate_geographic_markets(
    price_data=parcel_price_panel,
    locations=parcel_ids,
)
```

## Consumer Behavior

### Trade Area Analysis

```python
from geo_infer_econ.microeconomics import DemandFunctions

# Cobb-Douglas demand estimation over retail catchments
demand = DemandFunctions(utility_function="cobb_douglas")
```

### Demand Estimation

```python
# Estimate consumer demand (Marshallian, Cobb-Douglas utility)
import numpy as np

quantities = demand.marshallian_demand_cobb_douglas(
    income=income_data,
    prices=np.asarray(prices),
    alpha=np.asarray(preference_shares),
)
```

## Market Analysis

### Competition Analysis

```python
from geo_infer_econ.microeconomics.market_structure import CompetitionAnalysis

comp = CompetitionAnalysis()

# Price-correlation based market definition
correlations = comp.calculate_price_correlation_matrix(
    price_data=competitor_prices,
)
gaps = comp.analyze_entry_barriers(
    industry_data=competitor_profiles,
)
```

## Key Concepts

| Concept | Application |
|---------|-------------|
| Trade Areas | Retail catchment |
| Bid-Rent | Land values |
| Agglomeration | Clustering effects |
| Accessibility | Service coverage |


