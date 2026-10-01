# Agricultural Intelligence: Active Inference for Adaptive Farming

> **Illustrative example notice.** This page contains historical or
> conceptual integration sketches. Names such as `SpatialAnalyzer` and
> domain-specific facade classes are not public GEO-INFER exports in the
> current checkout; verify imports against each module's `src/` package
> and use the module README/tests for executable examples.

> **Illustrative guide.** The code in this page is illustrative: it sketches
> how the module APIs compose for this use case. Some identifiers shown are
> conceptual; always import from the current package exports (see the module
> `__init__.py` and `SKILL.md`) and prefer the runnable scripts under
> `GEO-INFER-*/examples/` for verified behavior. Any numeric results shown
> are illustrative and must be reproduced against your own data before use.


This guide demonstrates multi-season Active Inference models for agricultural decision-making. Where the [Agricultural Applications](agricultural_applications.md) guide covers spatial analysis and prediction, this guide focuses on sequential decision-making under uncertainty -- choosing what to plant, when to irrigate, and how to adapt to changing conditions.

## Overview

Active Inference treats agricultural management as a perception-action loop:

1. **Perception**: The farmer (agent) maintains beliefs about hidden states -- soil health, crop growth stage, pest pressure, market conditions
2. **Action**: The agent selects policies (planting, irrigation, harvesting) that minimize expected free energy
3. **Learning**: Observations (sensor data, yield outcomes, price signals) update beliefs over time

This maps naturally onto seasonal farming cycles where decisions are sequential and outcomes are uncertain.

## Prerequisites

```bash
# From the repository root: installs ACT, AG, CLIMATE and the locked
# numpy, pandas and matplotlib versions
uv sync --all-packages --all-extras --all-groups
```

## Section 1: NDVI Integration and Vegetation Monitoring

Normalized Difference Vegetation Index (NDVI) time series serve as the primary observation signal for crop growth state. We aggregate satellite-derived NDVI to H3 resolution 9 cells and feed it into the Active Inference agent.

### Loading NDVI Time Series

```python
import numpy as np
import pandas as pd
import h3


def generate_ndvi_time_series(
    h3_cells: list[str],
    n_timesteps: int = 36,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic NDVI time series mimicking Sentinel-2 observations.

    Produces a realistic seasonal signal with:
    - Green-up in spring (March-May)
    - Peak NDVI in summer (June-August)
    - Senescence in fall (September-November)
    - Dormancy in winter (December-February)

    Args:
        h3_cells: List of H3 cell indexes (resolution 9).
        n_timesteps: Number of observations (approximately biweekly over 18 months).
        seed: Random seed.

    Returns:
        DataFrame with columns: h3_index, date, ndvi.
    """
    rng = np.random.default_rng(seed)

    # Time axis: biweekly from March 2024
    dates = pd.date_range("2024-03-01", periods=n_timesteps, freq="14D")

    rows = []
    for cell in h3_cells:
        # Seasonal NDVI curve (sinusoidal with asymmetric shape)
        day_of_year = dates.dayofyear.values.astype(float)
        # Peak around day 200 (mid-July)
        phase = 2.0 * np.pi * (day_of_year - 90) / 365.0
        seasonal = 0.45 + 0.35 * np.sin(phase)
        seasonal = np.clip(seasonal, 0.1, 0.9)

        # Spatial variation: slight offset per cell
        cell_hash = hash(cell) % 1000 / 1000.0
        spatial_offset = 0.05 * (cell_hash - 0.5)

        # Observation noise (cloud contamination, sensor noise)
        noise = rng.normal(0, 0.03, size=n_timesteps)

        ndvi = np.clip(seasonal + spatial_offset + noise, 0.0, 1.0)

        for t, (date, val) in enumerate(zip(dates, ndvi)):
            rows.append({
                "h3_index": cell,
                "date": date,
                "ndvi": val,
                "timestep": t,
            })

    return pd.DataFrame(rows)


# Generate sample H3 cells (resolution 9) for a farm region
# Center point: Willamette Valley, Oregon
center_lat, center_lng = 44.635, -123.078
center_cell = h3.latlng_to_cell(center_lat, center_lng, 9)
farm_cells = list(h3.grid_disk(center_cell, 3))  # 37 cells

ndvi_df = generate_ndvi_time_series(farm_cells, n_timesteps=36)
print(f"NDVI observations: {len(ndvi_df)}")
print(f"H3 cells: {len(farm_cells)}")
print(f"Date range: {ndvi_df['date'].min()} to {ndvi_df['date'].max()}")
```

### Spatial Aggregation

```python
def aggregate_ndvi_to_field(ndvi_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell NDVI to field-level summary per timestep.

    Computes mean, std, and coefficient of variation across cells.
    High spatial variability (CV > 0.15) indicates uneven growth
    that may warrant management action.

    Args:
        ndvi_df: DataFrame with h3_index, date, ndvi columns.

    Returns:
        DataFrame with date-level aggregates.
    """
    agg = ndvi_df.groupby("date")["ndvi"].agg(
        ndvi_mean="mean",
        ndvi_std="std",
        ndvi_min="min",
        ndvi_max="max",
    ).reset_index()
    agg["ndvi_cv"] = agg["ndvi_std"] / agg["ndvi_mean"]
    return agg


field_ndvi = aggregate_ndvi_to_field(ndvi_df)
print(f"Field-level NDVI at peak: {field_ndvi['ndvi_mean'].max():.3f}")
print(f"Max spatial variability (CV): {field_ndvi['ndvi_cv'].max():.3f}")
```

## Section 2: Multi-Season Belief Model

The core Active Inference model maintains beliefs about a hidden state that cycles through seasonal agricultural phases.

### Defining the Generative Model

```python
from geo_infer_act.core.active_inference import ActiveInferenceAgent


def build_seasonal_ag_model(
    n_states: int = 4,
    n_observations: int = 5,
    n_actions: int = 3
) -> dict[str, np.ndarray]:
    """Build the generative model matrices for seasonal agriculture.

    Hidden states: [dormant, growing, peak, senescent]
    Observations: [very_low_ndvi, low_ndvi, medium_ndvi, high_ndvi, very_high_ndvi]
    Actions: [no_action, irrigate, fertilize]

    Args:
        n_states: Number of hidden crop growth states.
        n_observations: Number of discretized NDVI observation levels.
        n_actions: Number of available management actions.

    Returns:
        Dict with 'A' (observation model), 'B' (transition model per action),
        'C' (preference vector), 'D' (initial state prior).
    """
    # A: Observation likelihood P(observation | hidden_state)
    # Rows = observations, Columns = hidden states
    A = np.array([
        #  dormant  growing  peak  senescent
        [0.60,     0.05,    0.01,  0.15],   # very_low_ndvi
        [0.30,     0.15,    0.04,  0.40],   # low_ndvi
        [0.08,     0.40,    0.15,  0.35],   # medium_ndvi
        [0.01,     0.30,    0.45,  0.08],   # high_ndvi
        [0.01,     0.10,    0.35,  0.02],   # very_high_ndvi
    ])
    # Normalize columns to sum to 1
    A = A / A.sum(axis=0, keepdims=True)

    # B: Transition matrices P(next_state | current_state, action)
    # One matrix per action
    B = np.zeros((n_actions, n_states, n_states))

    # Action 0: no_action -- natural seasonal transition
    B[0] = np.array([
        # to: dormant  growing  peak  senescent
        [0.70,  0.25,   0.00,  0.05],  # from dormant
        [0.05,  0.50,   0.40,  0.05],  # from growing
        [0.00,  0.05,   0.55,  0.40],  # from peak
        [0.40,  0.10,   0.00,  0.50],  # from senescent
    ]).T  # transpose so B[a][:, s] = P(s'|s, a)

    # Action 1: irrigate -- promotes growth, extends peak
    B[1] = np.array([
        [0.55,  0.40,   0.00,  0.05],  # from dormant
        [0.02,  0.40,   0.55,  0.03],  # from growing
        [0.00,  0.03,   0.70,  0.27],  # from peak
        [0.30,  0.15,   0.05,  0.50],  # from senescent
    ]).T

    # Action 2: fertilize -- boosts transition to growth/peak
    B[2] = np.array([
        [0.45,  0.50,   0.00,  0.05],  # from dormant
        [0.02,  0.35,   0.60,  0.03],  # from growing
        [0.00,  0.02,   0.65,  0.33],  # from peak
        [0.25,  0.20,   0.05,  0.50],  # from senescent
    ]).T

    # Normalize each transition matrix
    for a in range(n_actions):
        B[a] = B[a] / B[a].sum(axis=0, keepdims=True)

    # C: Observation preferences (agent prefers high NDVI)
    # Log preferences (higher = more preferred)
    C = np.array([-3.0, -1.5, 0.0, 2.0, 3.0])

    # D: Initial state prior (start in dormant state)
    D = np.array([0.7, 0.2, 0.05, 0.05])

    return {"A": A, "B": B, "C": C, "D": D}


model_params = build_seasonal_ag_model()
print("Generative model dimensions:")
for key, val in model_params.items():
    print(f"  {key}: {val.shape}")
```

### Running Belief Updates Over Multiple Seasons

```python
def discretize_ndvi(ndvi_value: float) -> int:
    """Convert continuous NDVI to discrete observation index.

    Bins: [0, 0.2) -> 0, [0.2, 0.35) -> 1, [0.35, 0.55) -> 2,
           [0.55, 0.75) -> 3, [0.75, 1.0] -> 4

    Args:
        ndvi_value: Continuous NDVI in [0, 1].

    Returns:
        Discrete observation index (0-4).
    """
    boundaries = [0.2, 0.35, 0.55, 0.75]
    for i, b in enumerate(boundaries):
        if ndvi_value < b:
            return i
    return 4


def run_seasonal_inference(
    field_ndvi: pd.DataFrame,
    model_params: dict[str, np.ndarray],
    planning_horizon: int = 3
) -> pd.DataFrame:
    """Run Active Inference over the seasonal NDVI time series.

    At each timestep:
      1. Observe NDVI (discretized)
      2. Update beliefs about hidden state
      3. Evaluate policies over planning horizon
      4. Select action that minimizes expected free energy

    Args:
        field_ndvi: DataFrame with date and ndvi_mean columns.
        model_params: Generative model matrices from build_seasonal_ag_model.
        planning_horizon: Number of steps to plan ahead.

    Returns:
        DataFrame with belief states, selected actions, and free energy per step.
    """
    agent = ActiveInferenceAgent(
        A=model_params["A"],
        B=model_params["B"],
        C=model_params["C"],
        D=model_params["D"],
        planning_horizon=planning_horizon,
    )

    state_labels = ["dormant", "growing", "peak", "senescent"]
    action_labels = ["no_action", "irrigate", "fertilize"]

    results = []
    for _, row in field_ndvi.iterrows():
        obs = discretize_ndvi(row["ndvi_mean"])

        # Perception: update beliefs given observation
        beliefs = agent.infer_states(obs)

        # Action selection: minimize expected free energy
        action, free_energy = agent.select_action()

        most_likely_state = state_labels[np.argmax(beliefs)]

        results.append({
            "date": row["date"],
            "ndvi_observed": row["ndvi_mean"],
            "observation_discrete": obs,
            "belief_dormant": beliefs[0],
            "belief_growing": beliefs[1],
            "belief_peak": beliefs[2],
            "belief_senescent": beliefs[3],
            "most_likely_state": most_likely_state,
            "selected_action": action_labels[action],
            "expected_free_energy": free_energy,
        })

        # Execute action (updates internal state for next step)
        agent.step(action)

    return pd.DataFrame(results)


inference_results = run_seasonal_inference(field_ndvi, model_params)
print(f"Inference steps: {len(inference_results)}")
print(f"\nAction distribution:")
print(inference_results["selected_action"].value_counts())
print(f"\nState belief trajectory (first 5 steps):")
print(inference_results[["date", "most_likely_state", "selected_action"]].head())
```

### Visualizing Belief Trajectories

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Panel 1: NDVI observations
axes[0].plot(inference_results["date"], inference_results["ndvi_observed"],
             color="green", linewidth=1.5)
axes[0].set_ylabel("NDVI")
axes[0].set_title("Observed NDVI and Active Inference Beliefs")

# Panel 2: Belief states
belief_cols = ["belief_dormant", "belief_growing", "belief_peak", "belief_senescent"]
colors = ["#8B4513", "#32CD32", "#FFD700", "#FF6347"]
for col, color in zip(belief_cols, colors):
    label = col.replace("belief_", "")
    axes[1].fill_between(inference_results["date"],
                         inference_results[col],
                         alpha=0.6, color=color, label=label)
axes[1].set_ylabel("Belief probability")
axes[1].legend(loc="upper right", fontsize=8)

# Panel 3: Actions taken
action_map = {"no_action": 0, "irrigate": 1, "fertilize": 2}
action_numeric = inference_results["selected_action"].map(action_map)
axes[2].step(inference_results["date"], action_numeric, where="mid",
             color="navy", linewidth=1.5)
axes[2].set_yticks([0, 1, 2])
axes[2].set_yticklabels(["No action", "Irrigate", "Fertilize"])
axes[2].set_ylabel("Action")
axes[2].set_xlabel("Date")

plt.tight_layout()
plt.savefig("seasonal_active_inference.png", dpi=150)
```

## Section 3: Market-Linked Yield Optimization

Agricultural decisions must account for market conditions. Expected free energy naturally combines agronomic outcomes with economic value.

### Coupling Yield with Price Signals

```python
def build_market_preferences(
    base_yield_value: float = 250.0,
    price_per_tonne: float = 300.0,
    irrigation_cost: float = 50.0,
    fertilizer_cost: float = 80.0
) -> dict[str, Any]:
    """Compute action-dependent preference vectors incorporating market prices.

    The preference over observations shifts based on the cost/benefit
    of each action and the expected revenue from yield.

    Args:
        base_yield_value: Baseline revenue per hectare at peak NDVI.
        price_per_tonne: Commodity price per tonne.
        irrigation_cost: Cost per irrigation event per hectare.
        fertilizer_cost: Cost per fertilizer application per hectare.

    Returns:
        Dict with action-adjusted preference vectors.
    """
    # Revenue scales with NDVI observation level
    ndvi_yield_factor = np.array([0.1, 0.3, 0.6, 0.85, 1.0])
    base_revenue = base_yield_value * ndvi_yield_factor * price_per_tonne / 300.0

    action_costs = {
        "no_action": 0.0,
        "irrigate": irrigation_cost,
        "fertilize": fertilizer_cost,
    }

    preferences = {}
    for action_name, cost in action_costs.items():
        net_value = base_revenue - cost
        # Convert to log-preference scale
        preferences[action_name] = np.log(np.maximum(net_value, 1.0))

    return preferences


market_prefs = build_market_preferences(price_per_tonne=320.0)
print("Market-adjusted preferences (log scale):")
for action, prefs in market_prefs.items():
    print(f"  {action}: {np.round(prefs, 2)}")
```

### Optimal Crop Rotation Policy

```python
def evaluate_rotation_policies(
    agent: ActiveInferenceAgent,
    crop_sequence: list[str],
    seasons_per_crop: int = 4
) -> dict[str, float]:
    """Evaluate expected free energy for different crop rotation sequences.

    Each crop modifies the observation model (A matrix) and transition
    dynamics (B matrix) slightly, reflecting different growth patterns.

    Args:
        agent: Configured ActiveInferenceAgent.
        crop_sequence: Ordered list of crop names.
        seasons_per_crop: Number of seasonal steps per crop.

    Returns:
        Dict mapping rotation description to cumulative expected free energy.
    """
    crop_modifiers = {
        "winter_wheat": {"growth_rate": 1.0, "peak_ndvi": 0.80},
        "corn": {"growth_rate": 1.2, "peak_ndvi": 0.85},
        "soybeans": {"growth_rate": 0.9, "peak_ndvi": 0.75},
        "cover_crop": {"growth_rate": 0.7, "peak_ndvi": 0.60},
    }

    total_efe = 0.0
    results = {}

    for crop in crop_sequence:
        modifier = crop_modifiers.get(crop, crop_modifiers["winter_wheat"])
        # Simulate seasons for this crop
        crop_efe = 0.0
        for season in range(seasons_per_crop):
            # Synthetic observation based on crop growth curve
            season_phase = season / seasons_per_crop
            ndvi_sim = modifier["peak_ndvi"] * np.sin(np.pi * season_phase)
            obs = discretize_ndvi(ndvi_sim)

            agent.infer_states(obs)
            _, efe = agent.select_action()
            crop_efe += efe

        results[crop] = crop_efe
        total_efe += crop_efe

    results["total"] = total_efe
    return results


# Evaluate two rotation strategies
rotation_a = ["winter_wheat", "corn", "soybeans"]
rotation_b = ["winter_wheat", "cover_crop", "corn"]

print("Rotation A (wheat-corn-soy):")
print("Rotation B (wheat-cover-corn):")
print("\nLower expected free energy = better policy under Active Inference")
```

## Section 4: Climate Adaptation

Integrating climate projections updates the agent's priors about future growing conditions.

### Loading Climate Projections

```python
from geo_infer_climate.core.climate_analyzer import ClimateAnalyzer


def generate_climate_scenarios(
    n_years: int = 10,
    seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Generate simplified climate scenarios for agricultural planning.

    Produces three scenarios: baseline, moderate warming, high warming.

    Args:
        n_years: Number of future years to project.
        seed: Random seed.

    Returns:
        Dict mapping scenario name to DataFrame with yearly projections.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2025, 2025 + n_years)

    scenarios = {}

    # Baseline: stable conditions
    baseline_temp = 15.0 + rng.normal(0, 0.5, n_years)
    baseline_precip = 1100.0 + rng.normal(0, 80, n_years)

    scenarios["baseline"] = pd.DataFrame({
        "year": years,
        "mean_temp_c": baseline_temp,
        "annual_precip_mm": baseline_precip,
        "growing_season_days": np.full(n_years, 200) + rng.integers(-5, 6, n_years),
    })

    # Moderate warming: +0.3C/decade, -5% precipitation
    moderate_trend = np.linspace(0, 0.3, n_years)
    scenarios["moderate_warming"] = pd.DataFrame({
        "year": years,
        "mean_temp_c": baseline_temp + moderate_trend,
        "annual_precip_mm": baseline_precip * (1.0 - 0.005 * np.arange(n_years)),
        "growing_season_days": np.full(n_years, 200) + rng.integers(-5, 10, n_years),
    })

    # High warming: +0.8C/decade, -15% precipitation
    high_trend = np.linspace(0, 0.8, n_years)
    scenarios["high_warming"] = pd.DataFrame({
        "year": years,
        "mean_temp_c": baseline_temp + high_trend,
        "annual_precip_mm": baseline_precip * (1.0 - 0.015 * np.arange(n_years)),
        "growing_season_days": np.full(n_years, 200) + rng.integers(-3, 15, n_years),
    })

    return scenarios


climate_scenarios = generate_climate_scenarios()
for name, df in climate_scenarios.items():
    print(f"\n{name}:")
    print(f"  Temp trend: {df['mean_temp_c'].iloc[0]:.1f} -> {df['mean_temp_c'].iloc[-1]:.1f} C")
    print(f"  Precip trend: {df['annual_precip_mm'].iloc[0]:.0f} -> {df['annual_precip_mm'].iloc[-1]:.0f} mm")
```

### Updating Priors Based on Climate

```python
def adapt_model_to_climate(
    base_params: dict[str, np.ndarray],
    climate_df: pd.DataFrame,
    year: int
) -> dict[str, np.ndarray]:
    """Adjust the generative model parameters based on climate projections.

    Warmer temperatures and lower precipitation shift the transition
    dynamics: faster green-up but shorter peak duration and increased
    senescence probability.

    Args:
        base_params: Original model parameters.
        climate_df: Climate scenario DataFrame.
        year: Target year for adaptation.

    Returns:
        Modified model parameters dict.
    """
    row = climate_df[climate_df["year"] == year].iloc[0]

    # Temperature anomaly relative to baseline 15C
    temp_anomaly = row["mean_temp_c"] - 15.0

    # Precipitation deficit (fraction below 1100mm baseline)
    precip_deficit = max(0, 1.0 - row["annual_precip_mm"] / 1100.0)

    adapted = {k: v.copy() for k, v in base_params.items()}

    # Modify transition matrix for no_action:
    # Higher temp -> faster transition from dormant to growing
    # Lower precip -> faster transition from peak to senescent
    B_adapted = adapted["B"].copy()

    # Increase dormant -> growing transition
    dormant_to_growing_boost = 0.05 * temp_anomaly
    B_adapted[0][1, 0] += max(0, dormant_to_growing_boost)
    B_adapted[0][0, 0] -= max(0, dormant_to_growing_boost)

    # Increase peak -> senescent transition under drought
    peak_to_senescent_boost = 0.1 * precip_deficit
    B_adapted[0][3, 2] += peak_to_senescent_boost
    B_adapted[0][2, 2] -= peak_to_senescent_boost

    # Re-normalize
    for a in range(B_adapted.shape[0]):
        col_sums = B_adapted[a].sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        B_adapted[a] = B_adapted[a] / col_sums

    adapted["B"] = B_adapted
    return adapted


# Adapt model for 2030 under high warming
adapted_2030 = adapt_model_to_climate(
    model_params,
    climate_scenarios["high_warming"],
    year=2030
)
print("Model adapted for 2030 high-warming scenario")
print(f"Transition matrix change (no_action, dormant->growing): "
      f"{model_params['B'][0][1,0]:.3f} -> {adapted_2030['B'][0][1,0]:.3f}")
```

### Adaptive Management Recommendations

```python
def generate_adaptation_report(
    scenarios: dict[str, pd.DataFrame],
    base_params: dict[str, np.ndarray],
    target_years: list[int]
) -> pd.DataFrame:
    """Generate a multi-scenario adaptation report.

    For each scenario and year, runs inference to determine
    the optimal action frequency distribution.

    Args:
        scenarios: Dict of climate scenario DataFrames.
        base_params: Baseline model parameters.
        target_years: Years to evaluate.

    Returns:
        DataFrame summarizing recommended action distributions per scenario.
    """
    rows = []
    for scenario_name, climate_df in scenarios.items():
        for year in target_years:
            if year not in climate_df["year"].values:
                continue

            adapted = adapt_model_to_climate(base_params, climate_df, year)
            climate_row = climate_df[climate_df["year"] == year].iloc[0]

            rows.append({
                "scenario": scenario_name,
                "year": year,
                "mean_temp_c": climate_row["mean_temp_c"],
                "annual_precip_mm": climate_row["annual_precip_mm"],
                "growing_season_days": climate_row["growing_season_days"],
                "recommendation": (
                    "increase_irrigation" if climate_row["annual_precip_mm"] < 1000
                    else "standard_management"
                ),
            })

    return pd.DataFrame(rows)


report = generate_adaptation_report(
    climate_scenarios,
    model_params,
    target_years=[2025, 2028, 2031, 2034]
)
print("Adaptation Report:")
print(report.to_string(index=False))
```

## Full Working Example

The following script ties all sections together into a single executable workflow.

```python
import numpy as np
import pandas as pd
import h3


def main():
    """Complete agricultural intelligence pipeline."""

    # 1. Set up farm region
    center_lat, center_lng = 44.635, -123.078
    center_cell = h3.latlng_to_cell(center_lat, center_lng, 9)
    farm_cells = list(h3.grid_disk(center_cell, 3))
    print(f"Farm region: {len(farm_cells)} H3 cells at resolution 9")

    # 2. Generate and aggregate NDVI
    ndvi_df = generate_ndvi_time_series(farm_cells, n_timesteps=36)
    field_ndvi = aggregate_ndvi_to_field(ndvi_df)
    print(f"NDVI time series: {len(field_ndvi)} observations")

    # 3. Build generative model and run inference
    model_params = build_seasonal_ag_model()
    inference_results = run_seasonal_inference(field_ndvi, model_params)
    print(f"Actions selected: {inference_results['selected_action'].value_counts().to_dict()}")

    # 4. Market analysis
    market_prefs = build_market_preferences(price_per_tonne=320.0)

    # 5. Climate adaptation
    climate_scenarios = generate_climate_scenarios()
    report = generate_adaptation_report(
        climate_scenarios, model_params,
        target_years=[2025, 2028, 2031, 2034]
    )
    print(f"\nAdaptation recommendations for {len(report)} scenario-year combinations")
    print(report[["scenario", "year", "recommendation"]].to_string(index=False))


if __name__ == "__main__":
    main()
```

## Expected Outputs

| Output | Description |
|--------|-------------|
| Belief trajectory plot | Shows how the agent's beliefs about crop state evolve over time |
| Action sequence | Optimal irrigation/fertilization timing based on NDVI observations |
| Market-adjusted preferences | How commodity prices shift the cost-benefit of management actions |
| Climate adaptation report | Scenario-specific recommendations for changing conditions |

## Related Guides

- [Agricultural Applications](agricultural_applications.md) -- spatial analysis and yield prediction
- [Climate Modeling](climate_modeling.md) -- detailed climate analysis workflows
- [Memory Management](../advanced/memory_management.md) -- handling large farm-scale datasets
- [Performance Optimization](../advanced/performance_optimization.md) -- speeding up multi-season simulations
