---
name: geo-infer-ant
description: Ant Colony Optimization and swarm intelligence for geospatial problems. Use when solving spatial optimization with ACO, PSO, ABC algorithms, implementing stigmergic coordination, or optimizing geographic routing and resource allocation with bio-inspired methods.
prerequisites:
  required: []
  recommended:
    - geo-infer-space
    - geo-infer-act
    - geo-infer-agent
difficulty: advanced
estimated_time: 60min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-ANT

## Instructions

### Core capabilities

- **ACO**: seeded Ant System, Ant Colony System, and Max-Min Ant System for
  matrix-backed routing, VRP-style constraints, and multi-objective paths.
- **PSO**: bounded continuous optimization with obstacle handling, local or
  global neighborhoods, adaptive parameters, and swarm coordination.
- **ABC**: employed, onlooker, and scout phases for bounded continuous search.
- **Stigmergy**: H3-indexed pheromone fields, diffusion, evaporation, deposit
  auditing, and digital trace coordination.
- **Applications**: environmental monitoring, disaster response, and urban
  traffic routing consume real supplied observations and graph/network inputs.
- **Analysis**: spatial patterns, interaction networks, emergence measures, and
  performance statistics are available from importable package APIs.

### Key imports

```python
import numpy as np

from geo_infer_ant.algorithms import (
    AntColonyOptimization,
    ParticleSwarmOptimization,
    ArtificialBeeColony,
)
from geo_infer_ant.core import PheromoneSystem, SwarmAgent, AgentPopulation
```

## Examples

### Matrix-backed ACO

```python
distance_matrix = np.array([
    [0.0, 1.0, 2.0],
    [1.0, 0.0, 1.0],
    [2.0, 1.0, 0.0],
])
optimizer = AntColonyOptimization(
    number_of_ants=20,
    max_iterations=50,
    variant="ACS",
    random_seed=42,
)
optimizer.initialize_problem(
    nodes=np.arange(3),
    distance_matrix=distance_matrix,
)
result = optimizer.solve()
print(result.best_solution, result.best_fitness)
```

## Guidelines

Use explicit bounds and finite numeric objective functions. Seed every
optimization or application run when reproducibility matters, and initialize
ACO with its problem matrices before solving. Treat empty observations or an
absent graph as an input condition and inspect the returned status rather than
assuming an improvement.

### Validation

Run the module gate from the repository root:

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ANT
```

For focused development, run the affected tests first:

```bash
uv run python -m pytest -c pyproject.toml -W error \
  GEO-INFER-ANT/tests/unit/test_deep_contracts.py
```

The fresh-process population RSS probe measures 100 real agents with the existing
200 MiB ceiling and 60-second deadline. Its child excludes inherited coverage
autostart and rejects active tracing/profiling; the parent suite remains covered.
Import/allocation timings and delayed stack traces support timeout diagnosis.

All optimizers accept `random_seed`/`seed` for reproducible local validation.
State serialization is JSON-compatible, H3 resolutions accept integer or
`h3_rN` forms, and malformed spatial bounds or matrices fail at the boundary.

### Integrations

ANT integrates with GEO-INFER-SPACE when installed and keeps deterministic
local behavior available when optional spatial adapters are absent. Population
agents can share a PheromoneSystem and can receive an injected Active Inference
model from GEO-INFER-ACT.
