# GEO-INFER-ACT Method, Output, and Visualization Inventory

This inventory documents the public Active Inference method surface in
`src/geo_infer_act`, the package runner outputs, and the visualization helpers
that make ACT runs inspectable. It is source-grounded against the current
package layout, not an aspirational API list.

## Canonical Package Exports

`geo_infer_act.__all__` exports:

- `ActiveInferenceModel`
- `ActiveInferenceStepResult`
- `FreeEnergyBreakdown`
- `H3BeliefUpdateResult`
- `H3CellDiagnostics`
- `H3EdgeDiagnostics`
- `H3GridInferenceResult`
- `H3LevelDiagnostics`
- `H3SpatialConsistency`
- `PolicyEvaluation`
- `SpatialInferenceTrace`
- `FreeEnergyCalculator`
- `GenerativeModel`
- `BayesianBeliefUpdate`
- `PolicySelector`
- `VariationalInference`
- `DynamicCausalModel`
- `SpatialActiveInferenceAgent`
- `ClimateModel`
- `IntegrationUtils`

## Core Active Inference Methods

### `core.active_inference.ActiveInferenceModel`

- `set_generative_model(model)`: attach the `GenerativeModel`, align model type,
  initialize current beliefs, and inherit model preferences when needed.
- `perceive(observation)`: update posterior beliefs from one observation vector.
- `act(available_actions=None)`: for supported categorical models, select an
  action through the `inferactively-pymdp==1.0.3` adapter and expose posterior,
  negative-EFE, and VFE metadata. Gaussian policy selection requires the
  explicit control model in `ContinuousPOMDPActiveInference` or Gaussian GNN.
- `update_observations(observations)`: own a copy of the current observation store.
- `predict_beliefs(action_index=None)`: advance one explicit flat categorical
  interval using B in `(next, current[, action])` order; action-conditioned B
  requires the executed action index. No observation or history step is invented.
- `update_preferences(preferences)`: replace the current prior preferences.
- `update_with_outcome(decision, outcome)`: record a decision/outcome pair and
  close the perception-action loop when the outcome contains an observation.
- `generate_policies(available_actions)`: return candidate policy/action
  dictionaries for selection.
- `select_policy(policies)`: evaluate supplied policy dictionaries and return
  the selected policy.
- `compute_expected_free_energy(policy)`: compute EFE for one policy from the
  model's current beliefs and preferences.
- `step(observation, available_actions=None, return_result=False)`: run one
  perceive-act cycle and optionally return `ActiveInferenceStepResult`.
- `compute_free_energy()`: compute the current variational free energy.
- `reset()`: clear current observation, action, policy diagnostics, and history.
- `get_history()`: return a defensive copy of interaction history.
- `get_current_state()`: return current beliefs, observations, action, free
  energy, and model type.
- `apply_to_h3(h3_obs, return_result=False)`: update an H3-enabled generative
  model from cell-indexed observations.
- `infer_over_h3_grid(h3_grid, return_result=False)`: score real H3 4.5
  observations cell-by-cell through pymdp while preserving the agent's current
  state.
- `trace_over_h3_grid(h3_grid, ...)`: return `SpatialInferenceTrace` with
  per-cell posterior, VFE, negative-EFE, action posterior, policy entropy,
  local coherence, belief flux, edge diagnostics, and level summaries.
- `trace_over_nested_h3_grid(h3_grid, ...)`: return the same trace contract for
  nested H3 plus parent aggregate cells and cross-level consistency residuals.
- `set_preferences(preferences)`: compatibility setter for current preferences.

### `core.generative_model`

- `MarkovBlanket.check_conditional_independence(state_idx, all_states)`: test
  whether a state is approximately screened off by sensory and active states.
- `GenerativeModel.update_beliefs(observations)`: dispatch to hierarchical or
  single-level belief updates.
- `GenerativeModel.compute_free_energy()`: compute model VFE for current
  categorical beliefs.
- `GenerativeModel.add_nested_level(child_model)`: append a nested child model.
- `GenerativeModel.update_nested_beliefs(observations)`: update this model and
  recursively propagate observations into nested models.
- `GenerativeModel.enable_spatial_navigation(grid_size)`: switch the model into
  grid-based spatial navigation mode.
- `GenerativeModel.enable_h3_spatial(h3_resolution, boundary)`: build a real
  `h3>=4.5.0,<5` cell set and neighbor graph for the supplied boundary.
- `GenerativeModel.integrate_rxinfer(model_specification, data, backend="rxinfer")`:
  run the supplied Julia model; `backend="local_gaussian"` explicitly chooses
  the declared scalar conjugate model and does not execute Julia source.
- `GenerativeModel.integrate_bayeux(log_density_fn, test_point,
  backend="numpy_metropolis", n_samples=1000, warmup=100, transform_fn=None)`:
  run the declared NumPy Metropolis chain or real Bayeux NumPyro NUTS. The
  density accepts keyword parameters; Bayeux requires a JAX-traceable scalar
  density. Transforms require Bayeux. Backend/caller failures propagate.
- `GenerativeModel.diffuse_beliefs(beliefs, diffusion_rate=0.1)`: convexly blend
  each posterior with the equal-weight mean of its observed graph neighbors.
- `GenerativeModel.aggregate_beliefs_to_resolution(beliefs, target_resolution)`:
  compute equal-cell posterior means under H3 parents; reject invalid cells,
  mixed input resolutions, and refinement.
- `GenerativeModel.set_preferences(preferences)`: replace model preferences.
- `GenerativeModel.get_model_summary()`: summarize dimensions, mode flags,
  beliefs, and integration state.
- `GenerativeModel.update_h3_beliefs(h3_obs, return_result=False)`: validate H3
  observations, update per-cell beliefs through pymdp 1.0.3, and optionally
  return `H3BeliefUpdateResult`.
- `GenerativeModel.compute_h3_cell_diagnostics(...)`: compute
  `SpatialInferenceTrace` diagnostics from flat or nested H3 cell results.

### `core.free_energy.FreeEnergyCalculator`

- `compute_categorical_free_energy(...)`: compute categorical VFE and optional
  `FreeEnergyBreakdown`.
- `compute_gaussian_free_energy(...)`: compute Gaussian free energy from means,
  precision matrices, observations, and optional priors.
- `compute_expected_free_energy(...)`: compute policy EFE from beliefs,
  predicted observations/beliefs, preferences, risk, ambiguity, and exploration
  terms.
- `compute(...)`: compatibility dispatcher for free-energy calculations.

### `core.policy_selection.PolicySelector`

- `select_policy(beliefs, policies, preferences=None)`: evaluate policies and
  select deterministically or by seeded sampling.
- `compute_expected_free_energy(...)`: compute EFE for one policy and optional
  diagnostic breakdown.
- `compute_policy_precision(expected_free_energies, baseline_precision=1.0)`:
  adapt policy precision from EFE separation.
- `evaluate_policy_set(beliefs, policies, preferences=None)`: return arrays and
  `PolicyEvaluation` objects for all candidates without selecting.
- `select_action(beliefs, available_actions, generative_model=None)`: convert
  actions to policies and return the selected action.

### `core.spatial_agent.SpatialActiveInferenceAgent`

- `spatial_perception(observations, propagate_beliefs=True)`: validate
  cell-indexed observations, update beliefs, propagate to neighbors, and record
  spatial VFE.
- `spatial_action()`: compute action-level spatial EFE and return selected
  action diagnostics.
- `step(observations, propagate_beliefs=True, return_result=False)`: run a full
  spatial perception-action cycle and optionally return `H3GridInferenceResult`.
- `trace_step(observations, ...)`: build a `SpatialInferenceTrace` from a
  spatial-agent step without re-running inference when supplied a grid result.
- `step_nested(observations, return_result=False)`: run a nested H3
  perception-action cycle and optionally return `NestedH3GridInferenceResult`.
- `trace_nested_step(observations, ...)`: build nested parent/child trace
  diagnostics from a spatial-agent nested step.
- `set_preferences(preferences)`: set preferred observations per H3 cell.
- `set_observation_model(cell_id, A)`: replace one cell observation model.
- `set_transition_model(cell_id, B)`: replace one cell transition model.
- `update_precision(cell_id, precision)`: update one cell's precision.
- `get_diagnostics()`: return agent, belief, free-energy, coherence, and action
  diagnostics.
- `export_results(filepath)`: write diagnostics, histories, cells, and final
  beliefs to JSON.
- `reset()`: restore uniform beliefs and clear histories.

### Other Core Methods

- `BayesianBeliefUpdate.update_categorical(...)`: categorical Bayes update.
- `BayesianBeliefUpdate.update_gaussian(...)`: Gaussian/Kalman belief update.
- `BayesianBeliefUpdate.compute_prediction_error(...)`: precision-weighted
  prediction error.
- `BayesianBeliefUpdate.compute_surprise(...)`: negative log-probability
  surprise.
- `BayesianBeliefUpdate.update_beliefs(...)`: dispatch categorical or Gaussian
  belief updates by input shape.
- `VariationalInference.mean_field_update(...)`: conjugate categorical or
  Gaussian mean-field update.
- `VariationalInference.mean_field_update_categorical(...)`: categorical
  mean-field convenience method.
- `VariationalInference.mean_field_update_gaussian(...)`: Gaussian mean-field
  convenience method returning the posterior mean.
- `VariationalInference.structured_update(...)`: run structured VI by belief
  propagation or structured mean field.
- `VariationalInference.importance_sampling_update(...)`: approximate posterior
  statistics by importance sampling.
- `VariationalInference.compute_elbo(...)`: compute the evidence lower bound.
- `DynamicCausalModel.state_equation(...)`: continuous-time state derivative.
- `DynamicCausalModel.observation_equation(...)`: observation generation from
  hidden state.
- `DynamicCausalModel.integrate_dynamics(...)`: Euler integration with process
  noise.
- `DynamicCausalModel.generate_observations(...)`: simulate observations from a
  trajectory.
- `DynamicCausalModel.estimate_parameters(observations, inputs, time_points)`:
  fit continuous-time derivatives on the supplied increasing time axis; these
  A/B coefficients are derivative parameters, not discrete transition matrices.
- `DynamicCausalModel.set_parameters(A, B, C)`: replace dynamics matrices.
- `DynamicCausalModel.set_noise_parameters(Q, R)`: replace noise matrices.
- `MarkovDecisionProcess.get_transition_prob(...)`: transition distribution for
  state/action.
- `MarkovDecisionProcess.get_observation_prob(...)`: observation distribution
  for a state.
- `MarkovDecisionProcess.transition(...)`: sample the next state.
- `MarkovDecisionProcess.observe(...)`: sample an observation.
- `MarkovDecisionProcess.simulate(...)`: simulate a state/observation trajectory.
- `MarkovDecisionProcess.get_predictive_state(...)`: predict next-state beliefs.
- `MarkovDecisionProcess.get_predictive_observation(...)`: predict observation
  distribution.
- `MarkovDecisionProcess.update_belief(...)`: Bayesian discrete belief update.
- `MarkovDecisionProcess.set_transition_matrix(...)`: set one state/action
  transition distribution.
- `MarkovDecisionProcess.set_observation_matrix(...)`: set one state
  observation distribution.

## Domain Model Methods

- `models.base.BaseActiveInferenceModel.step(actions=None)`: return a fresh copy
  of the base configuration; concrete models implement their own inference step.
- `models.base.BaseActiveInferenceModel.reset()`: base reset interface.
- `CategoricalModel.set_preferences(...)`: set discrete preference distribution.
- `CategoricalModel.set_transition_matrix(...)`: set normalized transition
  matrix.
- `CategoricalModel.set_likelihood_matrix(...)`: set normalized likelihood
  matrix.
- `CategoricalModel.update_beliefs(...)`: categorical posterior update.
- `CategoricalModel.step(action=None)`: transition beliefs one step.
- `CategoricalModel.reset()`: restore uniform categorical beliefs.
- `CategoricalModel.compute_free_energy()`: categorical KL free energy.
- `GaussianModel.set_preferences(...)`: set continuous preference mean and
  covariance.
- `GaussianModel.set_transition_model(...)`: set state transition parameters.
- `GaussianModel.set_observation_model(...)`: set observation parameters.
- `GaussianModel.update_beliefs(...)`: Kalman belief update.
- `GaussianModel.step(action=None)`: Gaussian prediction step.
- `GaussianModel.reset()`: restore zero mean and identity covariance.
- `ClimateModel.step(observation)`: condition explicit integer modality indices,
  evaluate the joint factored policy posterior, and propagate its selected controls.
- `EcologicalModel.step(observation)`: perform the same factored inference
  contract with ecological modality and factor definitions. Missing observations
  require an explicit supported prediction path, not a fabricated sensor reading.
- `MultiAgentModel.step(actions=None)`: advance multi-agent observable state.
- `MultiAgentModel.enable_h3_spatial(...)`: instantiate one categorical agent per
  H3 cell.
- `MultiAgentModel.simulate_h3_lattice(...)`: run H3 lattice perception,
  free-energy logging, and coordination.
- `MultiAgentModel.coordinate_agents(...)`: coordinate agent beliefs over
  spatial neighbors or environmental stigmergy.
- `MultiAgentModel.get_agent_messages()`: return current coordination messages.
- `ResourceModel.step(actions=None)`: advance resource dynamics and
  free-energy-inspired scoring.
- `ResourceModel.reset()`: reset resources and history.
- `ResourceModel.get_allocation_scores()`: compute allocation priority scores.
- `UrbanModel.step(input_actions=None)`: advance urban agents and environment.
  `planning_horizon` selects an exact bounded sequence of stationary policies;
  the default is two, horizon lies in 1..8, and the factored artifact work
  budget limits joint inference. Default policies hold one joint control for
  the horizon; changing controls mid-policy requires explicit policy sequences.
- `UrbanModel.run_simulation(n_steps=10)`: run repeated urban-planning steps.

## Output and Visualization Methods

### Runners

- `runners.scenarios.run_scenario(config)`: run one scenario and write manifest,
  data, analysis, logs, and visualizations.
- `runners.scenarios.run_all_scenarios(config)`: run the configured scenario
  suite and write a suite manifest.
- `runners.io.write_run_manifest(...)`: write scenario manifest with generated
  file metadata.
- `runners.io.write_suite_manifest(...)`: write suite manifest for multi-scenario
  example runs.
- `runners.io.validate_generated_outputs(...)`: validate required output files
  and visualization sidecars.
- `runners.io.save_matplotlib_figure_artifact(...)`: save PNG visualization,
  embedded ACT metadata, metadata sidecar, data sidecar, and manifest entry.
- `runners.io.write_html_figure_artifact(...)`: save HTML visualization with
  embedded structured metadata, metadata sidecar, data sidecar, and manifest
  entry.

### Analysis and Visualization Helpers

- `ActiveInferenceAnalyzer.record_step(...)`: record beliefs, observations,
  actions, policies, free energy, and metrics.
- `ActiveInferenceAnalyzer.export_full_history(...)`: write the full step
  history JSON file.
- `ActiveInferenceAnalyzer.save_traces_to_csv(...)`: write step traces as CSV.
- `ActiveInferenceAnalyzer.generate_comprehensive_report(...)`: write analyzer
  summaries.
- `create_shared_visualizations(...)`: build a standard visualization set.
- `create_belief_heatmap(...)`: visualize belief trajectories.
- `create_free_energy_plots(...)`: visualize free-energy dynamics.
- `create_policy_plots(...)`: visualize policy/action diagnostics.
- `create_correlation_analysis(...)`: visualize correlations across recorded
  metrics.
- `plot_belief_update(...)`: plot prior, observation, and posterior.
- `plot_free_energy(...)`: plot free-energy trace.
- `plot_policies(...)`: plot policy probabilities.
- `plot_perception_analysis(...)`: plot perception diagnostics.
- `plot_action_analysis(...)`: plot action diagnostics.
- `create_interpretability_dashboard(...)`: build a combined interpretability
  dashboard.
- `plot_hierarchical_beliefs(...)`: visualize hierarchical beliefs from raw
  arrays or model-style belief payloads.
- `plot_markov_blanket(...)`: visualize dictionary or canonical
  `MarkovBlanket` structures.
- `plot_h3_grid_static(...)`: static H3 grid visualization.
- `create_h3_gif(...)`: animated H3 evolution artifact.
- `create_interactive_h3_slider(...)`: interactive H3 visualization.
- `BeliefVisualizer.plot_belief_evolution(...)`: class-based belief evolution
  plot.
- `BeliefVisualizer.plot_free_energy_trace(...)`: class-based free-energy trace
  plot.

## Runner Output Contract

Configured ACT scenarios are expected to write:

- `manifest.json`
- `data/full_history.json`
- `data/step_metrics.csv`
- `analysis/*.json`
- `logs/*`
- `visualizations/*` when visualization is enabled
- `visualizations/*.metadata.json`
- `visualizations/*.data.csv` or `visualizations/*.data.json`

`h3` and `spatial` scenarios also write:

- `data/h3_cells.csv`
- `data/h3_cells.geojson`
- `data/h3_diagnostics.json`
- `data/spatial_inference_trace.json`
- `data/spatial_research_statistics.json`
- `data/h3_cell_diagnostics.csv`
- `data/h3_edge_diagnostics.csv`
- `data/pymdp_h3_diagnostics.json`
- `data/pymdp_policy_posteriors.csv`
- static H3 PNG visualizations
- `visualizations/interactive_h3_map.html`
- `visualizations/pymdp_policy_free_energy.html`
- `visualizations/h3_belief_flux_map.html`
- `visualizations/h3_policy_surface.html`
- `visualizations/h3_policy_transitions.html`
- `visualizations/h3_spatial_autocorrelation.html`
- `visualizations/h3_entropy_free_energy_phase.html`
- `visualizations/spatial_inference_research_report.html`

Nested H3 mode also writes:

- `data/h3_hierarchy.csv`
- `data/nested_h3_diagnostics.json`
- `data/nested_h3_cell_diagnostics.csv`
- `data/nested_h3_parent_child_diagnostics.csv`
- `data/nested_h3_level_diagnostics.csv`
- `visualizations/nested_h3_level_map.html`
- `visualizations/nested_h3_hierarchy_map.html`
- `visualizations/nested_h3_parent_child_residuals.html`

Research-profile H3 runs are opt-in through
`RunConfig.parameters["research_profile"] = True` or
`geo-infer-act-run --research-profile`. They keep public method names and the
real H3/pymdp runtime, but use deterministic likelihood, preference, and
action-conditioned transition matrices for informative trace diagnostics.
Distinct selected actions are not guaranteed: nearby observations can share
one optimal action. The H3 validator checks policy diversity only on an explicit
seven-cell categorical fixture, with independently calculated posteriors,
one-step negative EFE, policy probabilities, and selected actions. Gallery runs
retain their environmental inputs and check those actual policy quantities.

Pymdp metadata retains `policy_beliefs`, the local posterior used to score
future actions. Flat diagnostic tables expose `policy_belief_<state>` columns.
Spatial diffusion or nested blending can subsequently change the displayed
cell belief; those blended values are distinct from the recorded policy input.

The deterministic gallery command writes flat H3, nested H3, flat spatial-agent,
and nested spatial-agent runs with linked visualization sidecars:

```bash
uv run python GEO-INFER-ACT/examples/spatial_active_inference_gallery.py
```

Each visualization should be manifest-referenced and include artifact type,
MIME type, SHA-256 digest, sidecar paths, source data files, plotted metrics,
description, alt text, and dimensions when available.

## Verification Commands

Run these from the repository root after ACT method, runner, or visualization
changes:

```bash
uv run python GEO-INFER-TEST/validate_active_inference_contract.py
uv run python GEO-INFER-TEST/validate_act_script_orchestration.py
uv run python GEO-INFER-TEST/validate_act_geospatial_contract.py
uv run --package geo-infer-act --extra dev python GEO-INFER-ACT/verify_comprehensive.py \
  --output-dir output/act_audit
uv run --package geo-infer-act --extra dev python -m pytest GEO-INFER-ACT/tests -q
```

For the full source scan, precise proof levels, backend limits, and migration
changes, see [the 0.4.0 method review](0_4_method_review.md).

The comprehensive audit regenerates an ignored evidence bundle in
`output/act_audit/`. It exercises representative methods from the
families listed in this inventory, the scenario suite, visualization helpers,
README/local-link checks, and Mermaid render checks.

For ad hoc output and visualization confirmation:

```bash
uv run --package geo-infer-act --extra dev geo-infer-act-examples \
  --output-dir /tmp/geo-infer-act-suite

uv run --package geo-infer-act --extra dev geo-infer-act-run \
  --scenario h3 \
  --config GEO-INFER-ACT/config/active_inference_run.yaml \
  --output-dir /tmp/geo-infer-act-h3 \
  --seed 42 \
  --timesteps 8
```
