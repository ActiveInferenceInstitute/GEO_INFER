# Bioregional Economics Guide

## Introduction

Bioregional economics focuses on economic analysis that respects ecological boundaries and promotes sustainable regional development.

## Key Concepts

### Bioregions

A bioregion is an area defined by natural boundaries (watersheds, ecosystems) rather than political ones.

```python
from geo_infer_econ import EcologicalEconomicsEngine

# The engine runs equilibrium, emergy, footprint, and carrying-capacity
# analyses over a bioregion's data
bioregion = EcologicalEconomicsEngine()
```

### Local Multipliers

Economic activity within a bioregion has multiplier effects:

```python
# Emergy analysis quantifies how much total ecological work backs each
# sector's flows (the engine's multiplier-style lens)
flows = bioregion.run_analysis(
    analysis_type="emergy",
    data={"sectors": ["local_food"], "region": "watershed_boundary"},
)
```

## Analysis Types

### Resource Flow Analysis

```python
# Equilibrium analysis tracks biophysical resource flows
flows = bioregion.run_analysis(
    analysis_type="equilibrium",
    data={"resources": ["water", "energy", "materials"]},
)
```

### Carrying Capacity

```python
# Assess ecological carrying capacity
capacity = bioregion.run_analysis(
    analysis_type="carrying_capacity",
    data={"population": current_pop, "consumption": consumption_patterns},
)
```

## Use Cases

| Use Case | Application |
|----------|-------------|
| Food Systems | Local food economy |
| Energy | Regional renewable potential |
| Water | Watershed-based planning |
| Tourism | Eco-tourism development |


