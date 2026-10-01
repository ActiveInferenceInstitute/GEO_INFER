# Module Integration Guidelines

## Data Flow Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| **Linear Pipeline** | Sequential processing | DATA → SPACE → TIME → ANALYSIS |
| **Hub and Spoke** | Central coordination | API as central hub |
| **Event-Driven** | Real-time responsive | IOT → processing → response |
| **Feedback Loop** | Active inference cycle | observation → belief update → action |

## Module Dependency Matrix

### Foundation Layer

```
MATH ──→ ACT ──→ AGENT
  │        │        │
  ├──→ BAYES    ├──→ SIM
  ├──→ SPM      └──→ ANT
  └──→ AI
```

### Data Layer

```
DATA ──→ SPACE ──→ PLACE
  │        │
  └──→ TIME ──→ IOT
```

### Domain Layer (all depend on SPACE, TIME, DATA)

```
AG, HEALTH, ECON, RISK, LOG, BIO,
CLIMATE, ENERGY, WATER, TRANSPORT,
FOREST, MARINE, EMERGENCY, EDU
```

### Governance Layer

```
SEC, NORMS, REQ, METAGOV ──→ All modules
```

### Application Layer

```
APP, ART, CIV, PEP, ORG, COMMS ──→ Consume from all other layers
```

### Operations Layer

```
OPS, INTRA, GIT, TEST, EXAMPLES ──→ Support all modules
```

## H3 v4 API (Mandatory)

All spatial operations must use H3 v4:

| v4 Method | Removed v3 method (do not use) |
|-----------|---------------------|
| `h3.latlng_to_cell()` | ~~`h3.geo_to_h3()`~~ |
| `h3.cell_to_latlng()` | ~~`h3.h3_to_geo()`~~ |
| `h3.grid_disk()` | ~~`h3.k_ring()`~~ |
| `h3.geo_to_cells()` | ~~`h3.polyfill()`~~ |

## Integration Examples

### ACT + AGENT

```python
from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_agent.core.agent_base import BaseAgent

class InferenceAgent(BaseAgent):
    async def initialize(self):
        self.model = ActiveInferenceModel(state_dim=10, obs_dim=5, action_dim=3)

    async def perceive(self) -> dict:
        return {"observations": self._read_sensors()}

    def update_beliefs(self, perception: dict):
        self.model.update(perception["observations"])
```

### SPACE + DATA

```python
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
from geo_infer_data import MultiSourceDataIngestion

spatial = SpatialIndexingInterface(backend='h3')
cell = spatial.latlng_to_cell(37.7749, -122.4194, 9)

ingestion = MultiSourceDataIngestion()
result = await ingestion.ingest_multi_source(
    satellite={'bbox': [-122.5, 37.7, -122.3, 37.9]}
)
```

### BAYES + RISK

```python
from geo_infer_bayes import BayesianInference
from geo_infer_risk.core.risk_engine import RiskEngine

inference = BayesianInference()
posterior = inference.update_beliefs(observations=loss_data, prior=prior_dist)

engine = RiskEngine()
risk = engine.run_analysis(region=region_data, analysis_type="catastrophe")
```

### TIME + CLIMATE

```python
from geo_infer_time import TemporalAnalyzer
from geo_infer_climate.core.climate_engine import ClimateEngine

analyzer = TemporalAnalyzer()
forecast = analyzer.forecast(temperature_series, horizon=365, method='arima')

climate = ClimateEngine()
projections = climate.project(scenario='rcp8.5', target_year=2050)
```

## Cross-Module Communication

- Use standardised data models (Pydantic) at module boundaries
- Implement proper API versioning for REST endpoints
- Support both sync and async communication patterns
- Handle errors gracefully — never let one module crash another
- Use consistent data formats: GeoJSON for spatial, ISO 8601 for temporal

## Error Propagation

```python
from geo_infer_math.utils.exceptions import MathError

try:
    result = external_module.process(data)
except MathError:
    raise  # Re-raise framework errors
except Exception as e:
    logger.error("Integration error with %s: %s", module_name, e)
    raise RuntimeError(f"Failed to process via {module_name}") from e
```

## Best Practices

1. **Check Dependencies**: Verify module availability before integration
2. **Use Standard Interfaces**: Prefer established interface classes
3. **Handle Errors Gracefully**: Catch and log across module boundaries
4. **Respect Data Flow**: Follow the layered architecture above
5. **Version Compatibility**: Check `pyproject.toml` version constraints
6. **H3 v4 Compliance**: Always use H3 v4 API methods for spatial data
