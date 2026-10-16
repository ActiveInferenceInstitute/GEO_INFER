# GEO-INFER-ANT Documentation

GEO-INFER-ANT implements swarm intelligence and complex adaptive systems for geospatial analysis. The module draws on ant colony optimization (ACO), stigmergic communication, and Active Inference to model collective behavior in spatial environments.

## Module Overview

GEO-INFER-ANT provides three layers of swarm intelligence functionality:

1. **Individual Agent Behavior** -- SwarmAgent instances that perceive, decide, and act within spatial environments using Active Inference for decision-making.
2. **Collective Coordination** -- Pheromone-based stigmergy (PheromoneSystem) and digital stigmergy (DigitalStigmergy) for indirect agent coordination through shared environmental signals.
3. **Population Dynamics** -- AgentPopulation management for simulating large-scale swarm behavior with configurable spatial distributions, behavioral heterogeneity, and emergent pattern analysis.

## Core Capabilities

- **Pheromone-based communication**: Multi-type pheromone fields (trail, food, alarm, nest) with H3-indexed spatial concentrations, environmental sensitivity (wind, temperature, humidity), and configurable evaporation/diffusion rates.
- **Digital stigmergy**: Modern indirect coordination through digital traces with credibility scoring, temporal persistence, spatial indexing, and access control.
- **Swarm agent framework**: Individual agents with Active Inference decision-making, spatial navigation, multi-modal sensory processing, and energy management.
- **Population simulation**: Configurable population dynamics with parallel agent updates, environmental interaction, resource foraging, and emergent pattern detection.
- **Optimization algorithms**: Ant Colony Optimization (ACO), Particle Swarm Optimization (PSO), and Artificial Bee Colony (ABC) for geospatial optimization problems.

## Integration Points

GEO-INFER-ANT integrates with the following GEO-INFER modules:

| Module | Integration |
|--------|------------|
| GEO-INFER-ACT | Active Inference models for agent decision-making and free energy minimization |
| GEO-INFER-SPACE | H3 spatial indexing for pheromone fields, agent positions, and neighbor queries |
| GEO-INFER-AGENT | Agent lifecycle management, state tracking, and coordination protocols |
| GEO-INFER-MATH | Mathematical foundations for optimization algorithms |
| GEO-INFER-TIME | Temporal dynamics for simulation scheduling and pheromone decay |

## Documentation Contents

- [Getting Started](getting_started.md) -- Installation, core concepts, and your first swarm simulation
- [API Reference](api_reference.md) -- Class and method documentation for all core components
- [Basic Example: Route Optimization on H3 Grid](examples/basic_example.md) -- Step-by-step ACO route optimization
- [Advanced Example: Multi-Objective Swarm Coverage](examples/advanced_example.md) -- Coverage planning with spatial constraints

## Architecture

```
geo_infer_ant/
  core/
    agent_base.py      -- SwarmAgent, SensoryInput, ActionDecision
    population.py      -- AgentPopulation, PopulationConfig, SimulationResults
    stigmergy.py       -- PheromoneSystem, PheromoneField, PheromoneType
    digital_stigmergy.py -- DigitalStigmergy, DigitalTrace, InformationQuery
  algorithms/
    aco.py             -- AntColonyOptimization
    pso.py             -- ParticleSwarmOptimization
    abc.py             -- ArtificialBeeColony
  applications/
    environmental.py   -- EnvironmentalMonitoringSwarm
    disaster.py        -- DisasterResponseSwarm
    urban.py           -- UrbanTrafficSwarm
  analysis/
    patterns.py        -- SwarmPatternAnalyzer
    metrics.py         -- SwarmPerformanceMetrics
```

## Quick Start

```python
import numpy as np
from geo_infer_ant.core.agent_base import SwarmAgent
from geo_infer_ant.core.population import AgentPopulation
from geo_infer_ant.core.stigmergy import PheromoneSystem

# Create a pheromone system for spatial coordination
pheromone = PheromoneSystem(
    spatial_resolution='h3_r8',
    pheromone_types=['trail', 'food', 'alarm'],
)

# Create a population of 100 agents
population = AgentPopulation(
    population_size=100,
    agent_types=['worker', 'scout'],
    spatial_distribution='clustered',
    spatial_bounds={
        'min_lat': 47.5, 'max_lat': 47.7,
        'min_lng': -122.4, 'max_lng': -122.2,
    },
)

# Run simulation
import asyncio
results = asyncio.run(population.run_simulation(time_steps=500))
print(f"Steps completed: {results.time_steps}")
print(f"Active agents: {results.performance_metrics.get('active_agents', 0)}")
```

## Key Concepts

**Stigmergy** is indirect coordination between agents through environmental modifications. In biological systems, ants deposit pheromones that influence the behavior of other ants. GEO-INFER-ANT implements both classical pheromone-based stigmergy and digital stigmergy for modern coordination scenarios.

**Active Inference** provides a principled framework for agent decision-making by minimizing expected free energy. Agents select actions that reduce uncertainty about their environment while achieving preferred states.

**Emergent behavior** arises from simple individual rules producing complex collective patterns. GEO-INFER-ANT tracks and analyzes these emergent patterns through spatial clustering, information flow analysis, and temporal trend detection.


## Native population memory measurement

The integration RSS probe constructs the same 100-agent population and environment
in a fresh Python child, with a 200 MiB incremental-RSS ceiling and a 60-second
process deadline. It excludes inherited pytest-cov/coverage autostart only from
that measurement child and rejects active trace/profile functions. Parent-suite
coverage and module coverage floors remain unchanged. Ordinary module tests
continue to exercise population construction under coverage.

The probe records import/allocation durations and captures delayed Python stacks
on timeout. Earlier instrumented child timeouts remain retained failures; scoped
Linux reruns did not reproduce their exact cause. A passing native-memory probe
does not establish the cause of an earlier instrumented timeout.
