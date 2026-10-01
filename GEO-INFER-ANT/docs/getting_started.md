# Getting Started with GEO-INFER-ANT

This guide walks through installation, core concepts, and building your first swarm simulation.

## Installation

GEO-INFER-ANT is a member of the root uv workspace. From the repository root:

```bash
uv sync --package geo-infer-ant
```

For full integration with spatial indexing and Active Inference, include the
`integrations` extra (SPACE, ACT, AGENT, MATH):

```bash
uv sync --package geo-infer-ant --extra integrations
```

Run the module's tests through the unified runner:

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ANT
```

### Dependencies

GEO-INFER-ANT requires Python 3.11+ (the workspace target; see `.python-version`) and depends on:

- `numpy` -- Array operations and mathematical computations
- `scipy` -- Spatial distance calculations and optimization
- `scikit-learn` -- Clustering analysis for emergent pattern detection

Optional dependencies for full integration:

- `geo_infer_space` -- H3 spatial indexing (v4 API)
- `geo_infer_act` -- Active Inference decision-making
- `geo_infer_agent` -- Agent lifecycle management
- `geo_infer_time` -- Temporal dynamics

The module operates in standalone mode when optional dependencies are unavailable, falling back to distance-based calculations instead of H3 indexing and rule-based decisions instead of Active Inference.

## Core Concepts

### Stigmergic Communication

Stigmergy is the mechanism by which agents coordinate indirectly through modifications to their shared environment. GEO-INFER-ANT implements two forms:

**Pheromone-based stigmergy** (`PheromoneSystem`): Agents deposit pheromone concentrations at spatial locations. Other agents sense these concentrations and adjust their behavior accordingly. Pheromones evaporate over time and diffuse through space, creating dynamic concentration gradients that guide collective behavior.

The module supports four default pheromone types:

| Type | Evaporation Rate | Diffusion Rate | Use Case |
|------|-----------------|----------------|----------|
| `trail` | 0.10 | 0.05 | Path marking between resources and nest |
| `food` | 0.05 | 0.10 | Resource location signaling |
| `alarm` | 0.20 | 0.20 | Danger or threat warnings |
| `nest` | 0.02 | 0.02 | Home location markers |

**Digital stigmergy** (`DigitalStigmergy`): Modern agents leave digital traces (sensor readings, status updates, hazard reports) in shared information spaces. Traces have credibility scores, visibility scopes, and temporal persistence.

### Swarm Agents

Each `SwarmAgent` follows a perception-decision-action loop:

1. **Perceive**: Gather spatial context, environmental signals, social signals from nearby agents, and stigmergic signals (pheromone concentrations or digital traces).
2. **Decide**: Use Active Inference (or fallback rule-based logic) to select an action that minimizes expected free energy.
3. **Act**: Execute the chosen action (move, deposit pheromone, forage, communicate, rest, monitor).

Agents maintain an energy level that depletes through actions and recovers through foraging or resting. When energy reaches zero, the agent becomes inactive.

### Population Dynamics

`AgentPopulation` manages collections of agents with configurable:

- **Population size**: Number of agents (default 1000)
- **Agent types**: Roles with different parameters (worker, scout, soldier)
- **Spatial distribution**: Initial placement strategy (random, clustered, uniform)
- **Behavioral heterogeneity**: Stochastic variation in agent parameters

For populations larger than ten, `parallel_processing=True` uses the active
asyncio event loop and bounds in-flight updates with `PopulationConfig.max_workers`.
Social context is calculated from active agents' in-memory positions, so the
reported value is an agent count rather than a count of H3 cells.

## First Example: Swarm Foraging Simulation

This example creates a population of 50 agents that forage for resources in a bounded spatial environment.

```python
import numpy as np
import asyncio
from geo_infer_ant.core.population import AgentPopulation

# Create a population with two agent types
population = AgentPopulation(
    population_size=50,
    agent_types=['worker', 'scout'],
    spatial_distribution='clustered',
    spatial_bounds={
        'min_lat': -5.0, 'max_lat': 5.0,
        'min_lng': -5.0, 'max_lng': 5.0,
    },
)

# Initialize the environment with resource patches
environment = population.initialize_environment(
    resource_distribution={
        'food': {
            'type': 'spatial_field',
            'centers': [
                np.array([3.0, 3.0]),
                np.array([-2.0, 4.0]),
                np.array([1.0, -3.0]),
            ],
            'max_density': 1.0,
            'decay_rate': 0.1,
            'regeneration_rate': 0.05,
        },
    },
    environmental_factors={
        'temperature': 22.0,
        'humidity': 55.0,
        'wind_speed': 3.0,
    },
)

# Configure behavioral rules
population.set_behavioral_rules(
    foraging_rules={
        'search_radius': 50.0,
        'return_threshold': 0.8,
    },
    communication_rules={
        'broadcast_range': 30.0,
        'message_types': ['food_found', 'danger'],
    },
)

# Run simulation for 200 time steps
async def run():
    results = await population.run_simulation(
        time_steps=200,
        data_collection=['trajectories', 'interactions', 'emergent_patterns'],
        progress_callback=lambda step, info: print(
            f"Step {step}: {info['agents_alive']} agents active"
        ) if step % 50 == 0 else None,
    )
    return results

results = asyncio.run(run())

# Inspect results
print(f"Simulation completed: {results.time_steps} steps")
print(f"Active agents: {results.performance_metrics.get('active_agents', 0)}")
print(f"Average energy: {results.performance_metrics.get('average_energy', 0):.3f}")

# Check for emergent spatial clusters
if 'spatial_clusters' in results.emergent_patterns:
    clusters = results.emergent_patterns['spatial_clusters']
    print(f"Detected {clusters['n_clusters']} spatial clusters")
```

## Working with Pheromone Systems

Create and manipulate pheromone fields directly:

```python
import numpy as np
import asyncio
from geo_infer_ant.core.stigmergy import PheromoneSystem

# Initialize pheromone system
pheromone = PheromoneSystem(
    spatial_resolution='h3_r8',
    pheromone_types=['trail', 'food'],
    environmental_factors={
        'temperature': 25.0,
        'humidity': 60.0,
        'wind_speed': 2.0,
    },
)

async def demo():
    # Agent deposits trail pheromone
    location = np.array([47.6062, -122.3321])
    success = await pheromone.deposit_pheromone(
        agent_id='scout_001',
        pheromone_type='trail',
        location=location,
        intensity=1.5,
    )
    print(f"Deposit success: {success}")

    # Another agent senses pheromones at a nearby location
    nearby = np.array([47.6065, -122.3325])
    concentrations = await pheromone.sense_pheromones(
        location=nearby,
        sensory_range=200.0,
        sensitivity_threshold=0.01,
    )
    print(f"Sensed concentrations: {concentrations}")

    # Advance pheromone diffusion by 60 seconds
    summary = await pheromone.diffuse_pheromones(time_step=60.0)
    print(f"Diffusion summary: {summary}")

    # Get gradient for navigation
    magnitude, direction = pheromone.get_pheromone_gradient(
        location=nearby,
        pheromone_type='trail',
        radius=50.0,
    )
    print(f"Gradient magnitude: {magnitude:.4f}, direction: {direction}")

asyncio.run(demo())
```

## Working with Digital Stigmergy

For modern coordination scenarios using digital information traces:

```python
import numpy as np
import asyncio
from geo_infer_ant.core.digital_stigmergy import DigitalStigmergy

stigmergy = DigitalStigmergy(
    communication_medium='iot_network',
    information_types=[
        'resource_discovery', 'hazard_warning',
        'traffic_info', 'environmental_data',
    ],
    persistence_model='temporal_decay',
)

async def demo():
    # Agent contributes environmental sensor data
    trace_id = await stigmergy.contribute_information(
        agent_id='sensor_042',
        information_type='environmental_data',
        content={
            'temperature': 28.5,
            'air_quality_index': 42,
            'noise_level_db': 55,
        },
        location=np.array([47.6062, -122.3321]),
        persistence_duration=7200.0,
    )
    print(f"Created trace: {trace_id}")

    # Another agent queries for nearby environmental data
    results = await stigmergy.query_stigmergy(
        agent_id='analyst_007',
        query_type='environmental_data',
        temporal_window='hour',
        credibility_threshold=0.3,
        max_results=10,
    )
    print(f"Query returned {len(results)} traces")

    # Extract emergent patterns from accumulated data
    patterns = await stigmergy.extract_patterns(
        pattern_types=['clusters', 'flows', 'anomalies'],
    )
    print(f"Detected pattern types: {list(patterns.keys())}")

asyncio.run(demo())
```

## Next Steps

- Read the [API Reference](api_reference.md) for complete class and method documentation
- Follow the [Basic Example](examples/basic_example.md) for route optimization on an H3 grid
- Explore the [Advanced Example](examples/advanced_example.md) for multi-objective coverage planning
