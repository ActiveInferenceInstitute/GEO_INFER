# Custom Models

> **Illustrative example notice.** This page contains historical or
> conceptual integration sketches. Names such as `SpatialAnalyzer` and
> domain-specific facade classes are not public GEO-INFER exports in the
> current checkout; verify imports against each module's `src/` package
> and use the module README/tests for executable examples.

This guide explains how to build, validate, and integrate custom models within the GEO-INFER framework. It covers Active Inference models via `geo_infer_act`, Bayesian models via `geo_infer_bayes`, and domain-specific spatial models that combine both.

## Active Inference Models with GEO-INFER-ACT

GEO-INFER-ACT implements Active Inference using variational free energy minimization. Custom models extend the generative model to define how hidden states produce observations and how policies are evaluated.

### Model Components

An Active Inference model requires:

1. **Generative model** -- defines joint probability P(observations, states)
2. **Variational posterior** -- approximate beliefs Q(states) over hidden states
3. **Free energy functional** -- the objective to minimize
4. **Policy space** -- candidate action sequences

### Subclassing the Generative Model

```python
import numpy as np
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.free_energy import FreeEnergyCalculator

class SoilMoistureModel(GenerativeModel):
    """
    Generative model for soil moisture dynamics.

    Hidden states: [moisture_level, drainage_rate, evaporation_rate]
    Observations: sensor readings (noisy moisture measurements)
    """

    def __init__(self, n_states: int = 3, observation_noise: float = 0.1):
        super().__init__(n_states=n_states)
        self.observation_noise = observation_noise
        self.transition_matrix = self._build_transition_matrix()
        self.observation_matrix = self._build_observation_matrix()

    def _build_transition_matrix(self) -> np.ndarray:
        """State dynamics: moisture decays via drainage and evaporation."""
        A = np.array([
            [0.85, 0.05, 0.10],
            [0.10, 0.80, 0.10],
            [0.05, 0.15, 0.80],
        ])
        return A

    def _build_observation_matrix(self) -> np.ndarray:
        """Sensor readings are a noisy linear transform of states."""
        B = np.eye(self.n_states) + np.random.default_rng(42).normal(
            0, self.observation_noise, (self.n_states, self.n_states)
        )
        return B

    def predict_observation(self, beliefs: np.ndarray) -> np.ndarray:
        """Predict expected observation given current beliefs."""
        return self.observation_matrix @ beliefs

    def update_beliefs(self, beliefs: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Update beliefs given new observation using free energy gradient descent."""
        predicted = self.predict_observation(beliefs)
        prediction_error = observation - predicted
        # Gradient step on free energy
        lr = 0.1
        updated = beliefs + lr * (self.observation_matrix.T @ prediction_error)
        # Normalize to valid probability
        updated = np.clip(updated, 1e-8, None)
        return updated / updated.sum()
```

### Computing Free Energy

The `FreeEnergyCalculator` in `geo_infer_act.core.free_energy` supports both categorical and Gaussian models.

```python
from geo_infer_act.core.free_energy import FreeEnergyCalculator

calculator = FreeEnergyCalculator()

# Categorical free energy
beliefs = np.array([0.6, 0.3, 0.1])
observations = np.array([0.8, 0.15, 0.05])
preferences = np.log(np.array([0.5, 0.3, 0.2]))  # log prior preferences

fe = calculator.compute_categorical_free_energy(
    beliefs=beliefs,
    observations=observations,
    preferences=preferences,
)
print(f"Free energy: {fe:.4f}")
```

Free energy decomposes as: F = Complexity - Accuracy, where Complexity is D_KL[Q(s) || P(s)] and Accuracy is E_Q[log P(o|s)].

## Bayesian Models with GEO-INFER-BAYES

### Gaussian Process Regression

GEO-INFER-BAYES provides a Gaussian Process implementation using Cholesky decomposition for numerical stability. The implementation falls back to NumPy/SciPy when TensorFlow Probability is not installed.

```python
from geo_infer_bayes.api.tfp_interface import TFPInterface
import numpy as np

# Training data: spatial locations and observations
X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
y_train = np.array([1.2, 2.1, 1.8, 3.0, 2.4])

# Fit GP model (uses Cholesky decomposition internally)
gp = TFPInterface(model_config={"lengthscale": 0.5, "variance": 1.0, "noise": 0.01})
summary = gp.create_spatial_gp_model(X_train, y_train)
print(summary)
```

The `create_spatial_gp_model` method computes:
1. Kernel matrix K using the squared-exponential kernel
2. Cholesky factor L = cholesky(K + noise * I)
3. Weight vector alpha = cho_solve(L, y)
4. Log-marginal-likelihood for model assessment

### Custom Kernel Functions

To use a different kernel, wrap the GP interface:

```python
import numpy as np
from scipy import linalg

def matern_32_kernel(X1: np.ndarray, X2: np.ndarray,
                      lengthscale: float, variance: float) -> np.ndarray:
    """Matern 3/2 kernel for spatial correlation."""
    diff = X1[:, None, :] - X2[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1)) / lengthscale
    sqrt3_r = np.sqrt(3.0) * r
    return variance * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)


def fit_gp_custom_kernel(X: np.ndarray, y: np.ndarray,
                          lengthscale: float = 1.0,
                          variance: float = 1.0,
                          noise: float = 0.01) -> dict:
    """Fit a GP with Matern 3/2 kernel using Cholesky decomposition."""
    K = matern_32_kernel(X, X, lengthscale, variance)
    K += noise * np.eye(len(X))

    L = linalg.cholesky(K, lower=True)
    alpha = linalg.cho_solve((L, True), y)

    log_ml = (
        -0.5 * y @ alpha
        - np.sum(np.log(np.diag(L)))
        - 0.5 * len(y) * np.log(2 * np.pi)
    )

    return {"L": L, "alpha": alpha, "log_marginal_likelihood": log_ml}
```

### Posterior Sampling

The `TFPInterface.sample()` method runs Metropolis-Hastings on the GP hyperparameters in log-space:

```python
# Sample hyperparameter posteriors
traces = gp.sample(n_samples=2000, n_warmup=500, seed=42)

# Posterior summaries
for param, values in traces.items():
    print(f"{param}: mean={values.mean():.4f}, std={values.std():.4f}")
```

## Model Comparison

GEO-INFER-BAYES implements real model comparison metrics. These are not stubs -- each computes a proper numerical estimate.

### LOO-CV (Leave-One-Out Cross-Validation)

```python
def loo_cv_score(X: np.ndarray, y: np.ndarray,
                  kernel_fn, lengthscale: float,
                  variance: float, noise: float) -> float:
    """Compute LOO-CV log predictive density analytically for a GP."""
    K = kernel_fn(X, X, lengthscale, variance) + noise * np.eye(len(X))
    L = linalg.cholesky(K, lower=True)
    alpha = linalg.cho_solve((L, True), y)
    K_inv = linalg.cho_solve((L, True), np.eye(len(X)))

    # LOO predictive mean and variance
    loo_mean = y - alpha / np.diag(K_inv)
    loo_var = 1.0 / np.diag(K_inv)

    # LOO log predictive density
    loo_lpd = -0.5 * np.log(2 * np.pi * loo_var) - 0.5 * (y - loo_mean) ** 2 / loo_var
    return float(np.sum(loo_lpd))
```

### WAIC (Widely Applicable Information Criterion)

```python
def compute_waic(log_likelihoods: np.ndarray) -> dict:
    """
    Compute WAIC from a matrix of log-likelihoods.

    Args:
        log_likelihoods: shape (n_samples, n_observations)
    """
    # Log pointwise predictive density
    lppd = np.sum(
        np.log(np.mean(np.exp(log_likelihoods), axis=0))
    )
    # Effective number of parameters
    p_waic = np.sum(np.var(log_likelihoods, axis=0))
    waic = -2 * (lppd - p_waic)
    return {"waic": waic, "lppd": lppd, "p_waic": p_waic}
```

### DIC (Deviance Information Criterion)

```python
def compute_dic(log_likelihoods: np.ndarray) -> dict:
    """Compute DIC from posterior samples of log-likelihood."""
    mean_deviance = -2 * np.mean(np.sum(log_likelihoods, axis=1))
    deviance_at_mean = -2 * np.sum(np.mean(log_likelihoods, axis=0))
    p_dic = mean_deviance - deviance_at_mean
    dic = deviance_at_mean + 2 * p_dic
    return {"dic": dic, "p_dic": p_dic}
```

### Comparing Models

```python
# Compare GP models with different kernels
models = {
    "RBF_short": {"lengthscale": 0.3, "variance": 1.0},
    "RBF_long": {"lengthscale": 2.0, "variance": 1.0},
    "RBF_high_var": {"lengthscale": 1.0, "variance": 5.0},
}

for name, params in models.items():
    gp = TFPInterface(model_config={**params, "noise": 0.01})
    summary = gp.create_spatial_gp_model(X_train, y_train)
    # Extract log-marginal-likelihood from summary or recompute
    print(f"{name}: {summary}")
```

Use LOO when you have few observations (< 200). Use WAIC or DIC when you have posterior samples from MCMC. BIC and AIC are available as simpler alternatives for quick screening.

## Domain-Specific Spatial Models

### Example: Wildfire Risk Model

```python
import numpy as np
from geo_infer_act.core.free_energy import FreeEnergyCalculator
from geo_infer_bayes.api.tfp_interface import TFPInterface

class WildfireRiskModel:
    """
    Combines Active Inference for decision-making with GP spatial
    interpolation for risk surface estimation.
    """

    def __init__(self, gp_config: dict | None = None):
        self.gp = TFPInterface(model_config=gp_config or {
            "lengthscale": 5.0, "variance": 2.0, "noise": 0.05
        })
        self.fe_calc = FreeEnergyCalculator()
        self._risk_surface = None

    def fit_risk_surface(self, locations: np.ndarray, fire_counts: np.ndarray):
        """Fit GP to historical fire occurrence data."""
        self.gp.create_spatial_gp_model(locations, fire_counts)

    def evaluate_policy(self, beliefs: np.ndarray,
                         observations: np.ndarray) -> float:
        """Evaluate a resource allocation policy using free energy."""
        return self.fe_calc.compute_categorical_free_energy(
            beliefs=beliefs, observations=observations
        )
```

## Integration into the GEO-INFER Pipeline

Custom models integrate through the standard module import pattern:

```python
# In your analysis script
from geo_infer_space import SpatialAnalyzer
from geo_infer_time.core.analysis import TemporalAnalyzer
from geo_infer_act.core.free_energy import FreeEnergyCalculator
from geo_infer_bayes.api.tfp_interface import TFPInterface

# 1. Spatial preprocessing
spatial = SpatialAnalyzer()
features = spatial.extract_features(raw_data)

# 2. Temporal pattern extraction
temporal = TemporalAnalyzer()
trends = temporal.detect_trend(time_series, method="linear")

# 3. Bayesian spatial model
gp = TFPInterface(model_config={"lengthscale": 1.0})
gp.create_spatial_gp_model(locations, observations)

# 4. Active Inference decision layer
fe = FreeEnergyCalculator()
energy = fe.compute_categorical_free_energy(beliefs, obs)
```

## Validation Checklist

Before deploying a custom model:

1. **Prior predictive check** -- sample from the prior and verify outputs are plausible
2. **Posterior predictive check** -- compare model predictions against held-out data
3. **Convergence diagnostics** -- for MCMC, check R-hat < 1.01 and effective sample size
4. **Sensitivity analysis** -- vary prior parameters and check result stability
5. **Cross-validation** -- use LOO or k-fold to estimate out-of-sample performance

## See Also

- [Bayesian Inference Guide](../bayesian_inference_guide.md) -- deeper coverage of Bayesian methods
- [Performance Optimization](performance_optimization.md) -- optimizing model computation
- [Active Inference Guide](../active_inference_guide.md) -- foundational Active Inference concepts
