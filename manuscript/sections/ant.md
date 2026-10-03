## GEO-INFER-ANT — Swarm Intelligence and Complex Adaptive Systems

**Purpose.** GEO-INFER-ANT models swarm intelligence and complex adaptive systems using Active Inference principles for emergent collective behavior in geospatial contexts. Its README casts the module as the home of population-level coordination — ants, bees, and particles as metaphors for distributed spatial decision-making — with an `IMPLEMENTATION_STATUS.md` tracking component maturity and a module-level `run_tests.py` driver.

**Public API.** The `geo_infer_ant` package exports a four-layer surface. Core: `SwarmAgent`, `AgentPopulation`, `PheromoneSystem`, and `DigitalStigmergy` — the stigmergic communication substrate. Algorithms: `AntColonyOptimization`, `ParticleSwarmOptimization`, and `ArtificialBeeColony`. Applications: `EnvironmentalMonitoringSwarm`, `DisasterResponseSwarm`, and `UrbanTrafficSwarm`, instantiating the algorithms on concrete geospatial missions. Analysis: `SwarmPatternAnalyzer` and `SwarmPerformanceMetrics` for evaluating emergent behavior, with configuration utilities (`load_config`, `validate_config`, `config_to_dict`) and an `IntegrationManager` in `utils/`.

**Verification status.** The `tests/` directory is present with 11 test files. The shared repository runner includes ANT in its declared category selections and retains execution receipts; a file census alone does not establish passing behavior.

**Theme role.** Agents: ANT is the collective/many-agent counterpart to GEO-INFER-AGENT's individual-agent runtimes, demonstrating emergent coordination as an active inference phenomenon at population scale.
