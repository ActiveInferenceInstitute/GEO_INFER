# World Systems Modeling

## Introduction

World systems modeling in Active Inference refers to the construction of generative models that capture the dynamics of complex socio-ecological systems. This document describes how GEO-INFER-ACT enables modeling of world systems.

## What is a World System?

A world system is an interconnected network of:

- **Physical systems**: Climate, hydrology, geology
- **Ecological systems**: Ecosystems, biodiversity, resource flows
- **Social systems**: Economies, governance, culture
- **Technological systems**: Infrastructure, networks, energy

## Modeling Approach

### Hierarchical Generative Models

World systems are naturally hierarchical:

```
Global Level
├── Continental Level
│   ├── National Level
│   │   ├── Regional Level
│   │   │   ├── Local Level
│   │   │   │   └── Site Level
```

Each level has its own state space and dynamics, with information flowing up and down the hierarchy.

### Multi-Scale State Spaces

```python
from geo_infer_act import ClimateModel, EcologicalModel

# Each subsystem is an active-inference model with its own state space.
# Compose scales by feeding one model's predictions into the next one's
# observations (see geo_infer_act.core.belief_updating and
# geo_infer_act.core.policy_selection).
climate_model = ClimateModel(random_seed=42)
ecological_model = EcologicalModel(random_seed=42)
```

## System Components

### Physical Subsystem

```python
physical_states = {
    "temperature": ContinuousState(range=(-50, 50)),
    "precipitation": ContinuousState(range=(0, 500)),
    "land_cover": CategoricalState(categories=land_types),
    "elevation": StaticState(source=dem_data)
}
```

### Ecological Subsystem

```python
ecological_states = {
    "biomass": ContinuousState(dynamics=growth_model),
    "species_richness": DiscreteState(range=(0, 1000)),
    "habitat_quality": ContinuousState(range=(0, 1)),
    "connectivity": NetworkState(graph=habitat_network)
}
```

### Social Subsystem

```python
social_states = {
    "population": ContinuousState(dynamics=demographic_model),
    "land_use": CategoricalState(categories=use_types),
    "governance": InstitutionalState(rules=governance_rules),
    "economy": EconomicState(model=economic_model)
}
```

## Active Inference for World Systems

### Perception

Agents monitoring world systems update beliefs about system states:

```python
# Multi-source observation integration
observations = {
    "satellite": satellite_imagery,
    "sensors": iot_data,
    "surveys": social_surveys,
    "reports": administrative_data
}

beliefs = agent.perceive(observations)
```

### Action

Interventions in world systems:

```python
actions = {
    "policy": implement_policy,
    "infrastructure": build_infrastructure,
    "conservation": protect_area,
    "restoration": restore_ecosystem
}

# Select action minimizing expected free energy
best_action = agent.act(preferences=sustainable_development_goals)
```

### Learning

Adaptation through model updating:

```python
# Learn from outcomes
agent.learn(
    predicted=expected_outcomes,
    observed=actual_outcomes
)
```

## Applications

### Climate Adaptation Planning

```python
climate_agent = WorldSystemAgent(
    model=coupled_climate_society_model,
    preferences=adaptation_goals
)

# Identify optimal adaptation strategies
strategies = climate_agent.plan(
    scenarios=climate_scenarios,
    horizon=2050
)
```

### Sustainable Development

```python
sdg_agent = WorldSystemAgent(
    model=sdg_system_model,
    preferences=sdg_targets
)

# Balance competing goals
development_plan = sdg_agent.optimize(
    constraints=resource_limits,
    trade_offs=sdg_interactions
)
```

## Further Reading

- [Geospatial Applications](./geospatial_applications.md)
- [Mathematical Framework](./mathematical_framework.md)


