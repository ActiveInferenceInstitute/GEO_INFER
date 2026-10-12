# GEO-INFER-ANT API Reference

Complete class and method reference for the GEO-INFER-ANT swarm intelligence module.

---

## core.stigmergy

### PheromoneType

Dataclass configuring a specific pheromone type.

```python
@dataclass
class PheromoneType:
    name: str
    evaporation_rate: float = 0.1
    diffusion_rate: float = 0.05
    deposition_amount: float = 1.0
    persistence_time: float = 300.0
    max_intensity: float = 2.0
    min_intensity: float = 0.01
    wind_sensitivity: float = 0.5
    temperature_sensitivity: float = 0.3
    humidity_sensitivity: float = 0.2
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Identifier for the pheromone type |
| `evaporation_rate` | `float` | 0.1 | Decay rate per time unit (must be in (0, 1]) |
| `diffusion_rate` | `float` | 0.05 | Spatial diffusion rate per time unit (must be in [0, 1]) |
| `deposition_amount` | `float` | 1.0 | Default intensity deposited by agents |
| `persistence_time` | `float` | 300.0 | Maximum persistence in seconds |
| `max_intensity` | `float` | 2.0 | Ceiling for concentration at any cell |
| `min_intensity` | `float` | 0.01 | Threshold below which pheromone is removed |
| `wind_sensitivity` | `float` | 0.5 | Sensitivity to wind effects on diffusion |
| `temperature_sensitivity` | `float` | 0.3 | Sensitivity to temperature effects on evaporation |
| `humidity_sensitivity` | `float` | 0.2 | Sensitivity to humidity effects on persistence |

Raises `ValueError` if `evaporation_rate` or `diffusion_rate` is out of range.

---

### PheromoneDeposit

Record of a single pheromone deposit event.

```python
@dataclass
class PheromoneDeposit:
    agent_id: str
    pheromone_type: str
    intensity: float
    location: np.ndarray     # [lat, lng]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Raises `ValueError` if `intensity <= 0`.

---

### PheromoneField

Spatial field representing pheromone concentrations across H3 cells.

```python
@dataclass
class PheromoneField:
    pheromone_type: str
    spatial_resolution: str
    bounds: Dict[str, float]
    concentrations: Dict[str, float]   # h3_cell_id -> concentration
    deposits: List[PheromoneDeposit]
    last_update: datetime
    update_count: int
```

**Methods:**

#### `get_concentration(location: np.ndarray) -> float`

Returns the pheromone concentration at the given `[lat, lng]` location. Uses H3 spatial indexing when available, otherwise falls back to distance-weighted interpolation from nearby deposits with exponential decay (100m decay constant, 1000m max range).

---

### PheromoneSystem

Main pheromone-based stigmergic communication system.

```python
class PheromoneSystem:
    def __init__(
        self,
        spatial_resolution: str = 'h3_r8',
        pheromone_types: Optional[List[str]] = None,
        bounds: Optional[Dict[str, float]] = None,
        environmental_factors: Optional[Dict[str, Any]] = None,
        spatial_backend: str = 'h3',
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spatial_resolution` | `str` | `'h3_r8'` | H3 resolution for spatial indexing |
| `pheromone_types` | `List[str]` | `['trail', 'food', 'alarm', 'nest']` | Pheromone types to initialize |
| `bounds` | `Dict[str, float]` | Global bounds | Spatial bounds (`min_lat`, `max_lat`, `min_lng`, `max_lng`) |
| `environmental_factors` | `Dict[str, Any]` | `{}` | Initial environmental conditions |
| `spatial_backend` | `str` | `'h3'` | Backend for spatial operations (`'h3'`, `'srai'`, `'geopandas'`) |

**Methods:**

#### `async deposit_pheromone(agent_id, pheromone_type, location, intensity=None, metadata=None) -> bool`

Deposit pheromone at a location. If `intensity` is None, uses the default deposition amount for the pheromone type. Intensity is clamped to `[0.01, max_intensity]`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_id` | `str` | Depositing agent identifier |
| `pheromone_type` | `str` | Must match a configured pheromone type |
| `location` | `np.ndarray` | `[lat, lng]` coordinates |
| `intensity` | `Optional[float]` | Override deposition amount |
| `metadata` | `Optional[Dict]` | Additional deposit metadata |

Returns `True` on success, `False` if the pheromone type is unknown.

#### `async sense_pheromones(location, sensory_range, pheromone_types=None, sensitivity_threshold=0.01) -> Dict[str, float]`

Sense pheromone concentrations at a location with environmental modifications applied.

Returns a dictionary mapping pheromone type names to concentrations. Concentrations below `sensitivity_threshold` are reported as 0.0.

#### `async diffuse_pheromones(time_step, environmental_conditions=None, spatial_barriers=None) -> Dict[str, Dict[str, Any]]`

Update pheromone fields by applying evaporation and spatial diffusion across all pheromone types. Returns per-type summary with `cells_updated`, `total_deposits`, `max_concentration`, and `avg_concentration`.

#### `get_pheromone_intensity(location: np.ndarray, pheromone_type: str) -> float`

Synchronous query for pheromone intensity at a specific location.

#### `get_pheromone_gradient(location, pheromone_type, radius=100.0) -> Tuple[float, np.ndarray]`

Compute the pheromone gradient at a location by sampling 8 points around the given radius. Returns `(gradient_magnitude, gradient_direction_vector)` where the direction points toward highest concentration.

#### `find_strongest_trail(start_location, pheromone_type='trail', search_radius=1000.0) -> Optional[Dict[str, Any]]`

Find the strongest pheromone trail within search radius. Returns trail info (`location`, `intensity`, `distance`, `timestamp`, `agent_id`) or None.

#### `get_field_statistics(pheromone_type: str) -> Dict[str, Any]`

Statistical summary of a pheromone field: deposit count, active cells, min/max/avg/std concentrations.

#### `clear_pheromone_field(pheromone_type: str) -> bool`

Clear all concentrations and deposits for a pheromone type.

#### `save_pheromone_fields(filepath: str) -> bool` / `load_pheromone_fields(filepath: str) -> bool`

Serialize/deserialize pheromone state to/from JSON.

---

## core.agent_base

### SensoryInput

Structured sensory input integrating spatial, environmental, social, and stigmergic signals.

```python
@dataclass
class SensoryInput:
    spatial_context: Dict[str, Any]
    environmental_signals: Dict[str, Any]
    social_signals: Dict[str, Any]
    stigmergic_signals: Dict[str, Any]
    temporal_context: Dict[str, Any]
```

**Methods:**

- `process() -> Dict[str, Any]`: Integrate all sensory inputs into a flat dictionary with prefixed keys (`env_`, `social_`, `stigmergic_`).
- `to_dict() -> Dict[str, Any]`: Dictionary representation including processed data.

### ActionDecision

Output of agent decision-making.

```python
@dataclass
class ActionDecision:
    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 0.0
    expected_outcome: Dict[str, Any]
    alternative_actions: List[Dict[str, Any]]
```

`execution_priority` is auto-calculated from confidence and action type urgency (emergency_response: 2.0x, resource_acquisition: 1.5x, communication: 1.2x, movement: 1.0x, monitoring: 0.8x).

### SwarmAgent

Base class for swarm intelligence agents with Active Inference integration.

```python
class SwarmAgent:
    def __init__(
        self,
        agent_id: str,
        position: np.ndarray,
        sensory_range: float = 100.0,
        movement_speed: float = 1.5,
        active_inference_enabled: bool = False,
        spatial_backend: str = "h3",
        **kwargs,
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | required | Unique agent identifier |
| `position` | `np.ndarray` | required | Initial `[lat, lng]` position |
| `sensory_range` | `float` | 100.0 | Maximum perception distance in meters |
| `movement_speed` | `float` | 1.5 | Maximum speed in m/s |
| `active_inference_enabled` | `bool` | `False` | Enable an explicitly configured ACT model |
| `spatial_backend` | `str` | `"h3"` | Spatial backend |

**Key kwargs:** `initial_energy` (float, default 1.0), `memory_capacity` (int, default 50). Enabling ACT also requires `active_inference_model` (an `ActiveInferenceModel` with its `GenerativeModel` configured), `act_observation_encoder` (a callable converting processed sensory data into the model's observation), and `act_actions` (ordered action dictionaries containing `action_type`). Their order must match the model's transition operators. The agent owns a copy of the action dictionaries.

The current bridge accepts flat categorical models. Execute each selected action before requesting another decision. A successful execution advances the prior through the selected transition once and records `act_prediction_history`; failed execution does not advance it. The next decision conditions the next real observation. No observation is synthesized from the action's expected outcome.
Execution uses the agent's owned action snapshot, so mutating a returned decision cannot retarget its physical action or transition. Concurrent execution of that decision raises. If model prediction fails after physical success, retrying the same decision retries prediction without repeating the physical action.

**Methods:**

#### `async perceive_environment(spatial_context, environmental_signals, social_signals, stigmergic_signals, temporal_context) -> SensoryInput`

Gather and integrate multi-modal sensory information. ACT inference runs when `make_decision` consumes this input.

#### `make_decision(sensory_input, internal_motivations=None, behavioral_rules=None) -> ActionDecision`

With ACT enabled, encode the observation and call the configured model's `step` once. Return its chosen action and policy probability as confidence. Encoding and inference errors propagate. The model's configured preferences govern this policy. With ACT disabled, use the rule policy; its default motivations are energy_conservation=0.8, task_completion=0.9, social_coordination=0.7, exploration=0.5.

#### `async execute_action(decision: ActionDecision) -> Dict[str, Any]`

Execute the chosen action. Routes to type-specific handlers for movement, stigmergic, communication, foraging, rest, and monitoring actions. Returns execution result with `success`, `energy_cost`, and `actual_outcome`.

#### `to_dict() -> Dict[str, Any]`

Serialize agent state including position, energy, task memory, and performance history.

---

## core.population

### PopulationConfig

Configuration dataclass for population dynamics.

```python
@dataclass
class PopulationConfig:
    population_size: int = 1000
    agent_types: List[str] = ['worker', 'scout', 'soldier']
    spatial_distribution: str = 'random'
    behavioral_heterogeneity: str = 'stochastic'
    spatial_bounds: Optional[Dict[str, float]] = None
    time_step: float = 1.0
    parallel_processing: bool = True
    max_workers: int = 4
```

### AgentPopulation

Management system for swarm agent populations.

```python
class AgentPopulation:
    def __init__(
        self,
        population_size: int = 1000,
        agent_types: List[str] = None,
        spatial_distribution: str = 'random',
        behavioral_heterogeneity: str = 'stochastic',
        spatial_bounds: Optional[Dict[str, float]] = None,
        **kwargs,
    )
```

**Methods:**

#### `set_behavioral_rules(foraging_rules=None, communication_rules=None, adaptation_rules=None) -> None`

Configure population-wide behavioral rules for foraging, communication, and adaptation.

#### `initialize_environment(spatial_bounds, resource_distribution, obstacle_map, pheromone_diffusion, environmental_factors) -> EnvironmentalState`

Set up the spatial environment with resources, obstacles, and environmental conditions.

#### `create_agents() -> List[SwarmAgent]`

Create and initialize all agents with type-specific configurations and spatial positions. Agent types receive different base parameters (scout: higher sensory range and speed; soldier: higher energy).

#### `async run_simulation(time_steps, environmental_changes=None, data_collection=None, progress_callback=None) -> SimulationResults`

Run the full simulation loop: update environment, update agents (parallel or sequential), collect data, analyze emergent patterns. Environmental changes can be scheduled with `start_time`, `end_time`, and `factors`. Parallel updates use the current asyncio loop and bound in-flight work with `PopulationConfig.max_workers`; social context is derived from active agents' positions and reports nearby-agent counts directly rather than treating spatial-index cells as agents.

#### `get_agent_by_id(agent_id: str) -> Optional[SwarmAgent]`
#### `get_agents_by_type(agent_type: str) -> List[SwarmAgent]`
#### `get_agents_in_region(center: np.ndarray, radius: float) -> List[SwarmAgent]`

Query agents by identifier, type, or spatial proximity.

#### `save_simulation_results(filepath: str) -> None` / `load_simulation_results(filepath: str) -> SimulationResults`

Persist simulation results to/from JSON.

### SimulationResults

```python
@dataclass
class SimulationResults:
    trajectories: List[np.ndarray]
    interactions: List[Dict[str, Any]]
    emergent_patterns: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    simulation_time: float
    time_steps: int
    population_size: int
```

---

## core.digital_stigmergy

### DigitalTrace

Information trace left by agents in the digital environment.

```python
@dataclass
class DigitalTrace:
    trace_id: str
    agent_id: str
    information_type: str
    content: Dict[str, Any]
    location: Optional[np.ndarray] = None
    timestamp: datetime
    visibility_scope: str = 'public'
    persistence_duration: float = 3600.0
    credibility_score: float = 1.0
    access_count: int = 0
    metadata: Dict[str, Any]
```

**Methods:**

- `is_expired() -> bool`: Check if trace has exceeded its persistence duration.
- `get_credibility_weight() -> float`: Credibility score with 24-hour time decay applied.

### DigitalStigmergy

Digital indirect coordination system.

```python
class DigitalStigmergy:
    def __init__(
        self,
        communication_medium: str = 'iot_network',
        information_types: Optional[List[str]] = None,
        persistence_model: str = 'temporal_decay',
        access_control: str = 'public',
        spatial_backend: str = 'h3',
    )
```

Default information types: `resource_discovery`, `hazard_warning`, `traffic_info`, `environmental_data`, `social_coordination`, `task_status`.

**Methods:**

#### `async contribute_information(agent_id, information_type, content, location=None, visibility_scope='public', persistence_duration=3600.0, credibility_score=None, metadata=None) -> str`

Add a digital trace. Returns the trace ID. Credibility is auto-calculated from agent reputation, information type reliability, and content completeness if not provided.

#### `async query_stigmergy(agent_id, query_type, spatial_bounds=None, temporal_window='recent', information_types=None, credibility_threshold=0.5, max_results=10) -> List[DigitalTrace]`

Query traces filtered by type, spatial bounds, temporal window (`recent`, `hour`, `day`, `week`, `all`), and credibility threshold. Results sorted by credibility weight.

#### `async extract_patterns(information_contributions=None, pattern_types=None, temporal_analysis='recent') -> Dict[str, Any]`

Extract emergent patterns: spatial clusters, information flows, anomalies, and temporal trends.

#### `get_system_statistics() -> Dict[str, Any]`

System-wide statistics: trace counts, query totals, information type distribution, agent participation, spatial coverage.
