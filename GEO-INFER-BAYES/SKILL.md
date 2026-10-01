---
name: geo-infer-bayes
description: Bayesian inference and probabilistic modeling for geospatial data. Use when building hierarchical models, computing posteriors with PyMC, performing variational inference, model comparison (LOO/WAIC/DIC), or spatial Gaussian processes.
prerequisites:
  required:
    - geo-infer-math
  recommended:
    - geo-infer-space
    - geo-infer-data
difficulty: advanced
estimated_time: 60min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-BAYES

## Instructions

### Core Capabilities

- **Bayesian inference**: Full posterior computation via MCMC and variational methods
- **Model comparison**: LOO-CV, WAIC, DIC, BIC, AIC (all real implementations)
- **Gaussian processes**: Exact Cholesky and batched inducing-point variational GPs
- **ELBO computation**: Real evidence lower bound (not placeholder)
- Out of scope: partial pooling via Cholesky LKJ decomposition, and
  Jeffreys / reference / unit-information priors. Implemented priors live in
  `geo_infer_bayes.utils.priors` (ICAR, BYM, Leroux, AR1, RW1, RW2, GP).

### Key Imports

```python
from geo_infer_bayes.core.inference import BayesianInference
from geo_infer_bayes.core.model_comparison import ModelComparison
from geo_infer_bayes.models.spatial_gp import SparseSpatialGP, SpatialGP
from geo_infer_bayes.core.variational import VariationalInference
from geo_infer_bayes.utils.diagnostics import mcmc_diagnostics
from geo_infer_bayes.api import PyMCInterface
```

## Examples

Every snippet below runs against the current API.

Fit a spatial Gaussian process by MCMC. `BayesianInference.run` hands the data
back to the model afterwards, so the returned posterior can predict without a
separate `fit` call:

```python
from geo_infer_bayes.core.inference import BayesianInference
from geo_infer_bayes.models.spatial_gp import SpatialGP

model = SpatialGP(kernel="rbf", lengthscale=1.0, variance=1.0, noise=0.5)
inference = BayesianInference(
    model=model, method="mcmc", sampler_config={"n_chains": 4, "random_seed": 0}
)
posterior = inference.run(
    data={"X": X, "y": y}, n_samples=2000, n_warmup=1000, progress_bar=False
)
print(posterior.summary()[["mean", "sd", "r_hat"]])
```

Fit a large spatial dataset without constructing an observation-by-observation
covariance matrix. Supplied inducing locations are fixed; omit them to select a
deterministic maximin subset. `fit` optimizes the collapsed variational ELBO by
default:

```python
from geo_infer_bayes.models import SparseSpatialGP

model = SparseSpatialGP(
    inducing_points=inducing_locations,
    kernel="matern",
    degree=1.5,
    batch_size=2048,
)
model.fit(X, y)
mean, std = model.predict(X_new, return_std=True)
print(model.elbo_)
```

Convergence diagnostics. Samplers return draws pooled across chains, and R-hat
is a between-chain statistic, so split the chain axis back out first:

```python
from geo_infer_bayes.utils.diagnostics import mcmc_diagnostics

diagnostics = mcmc_diagnostics(posterior.chain_samples())
worst_r_hat = max(stats["r_hat"] for stats in diagnostics.values())
```

Predict with calibrated uncertainty. The returned `std` is the total predictive
standard deviation of the latent function -- the mean conditional GP variance
across posterior draws plus the variance of the per-draw means. It excludes
observation noise, which belongs to `posterior_predictive`:

```python
mean, std = model.predict(X_new, posterior=posterior, samples=200, return_std=True)
draws = model.posterior_predictive(posterior, X=X_new, samples=500, random_seed=0)
```

Model comparison by LOO. A GP likelihood is joint, so
`pointwise_log_likelihood` supplies the per-observation terms LOO needs, via the
ordered-conditional decomposition of the marginal likelihood:

```python
import numpy as np
from geo_infer_bayes.core.model_comparison import ModelComparison

names = ["lengthscale", "variance", "noise"]
draws = np.linspace(0, len(posterior.samples["noise"]) - 1, 100, dtype=int)
matrix = np.asarray([
    model.pointwise_log_likelihood(
        {name: float(posterior.samples[name][i]) for name in names},
        {"X": X, "y": y},
    )
    for i in draws
])
results = ModelComparison([model]).compare_models(
    {"log_likelihood_matrix": matrix}, method="loo", random_seed=0
)
```

## Reproducibility

Every sampler and predictive method takes a `random_seed` routed through
`geo_infer_bayes.utils.rng.resolve_rng`, which accepts `None`, an `int`, a
`SeedSequence`, a `BitGenerator`, or a `numpy.random.Generator`, and always
returns a `Generator`; `numpy.random.RandomState` and the `numpy.random` module
raise `TypeError`. Consequences worth knowing:

- Passing an `int` makes a chain replayable; `0` is a valid seed.
- Passing a `Generator` threads one stream through a whole pipeline.
- `None` means OS entropy, so results are *not* replayable. Calling
  `np.random.seed(...)` does not make them so: this module never reads the
  process-wide singleton, and never advances it either.
- For independent parallel chains use
  `geo_infer_bayes.utils.rng.spawn_rng(seed, n)` rather than `seed`, `seed + 1`,
  ... which carries no independence guarantee.
- At boundaries that accept only an `int` seed, such as scikit-learn's
  `random_state`, use `geo_infer_bayes.utils.rng.derive_int_seed`.

## Guidelines

- GP uses actual Cholesky decomposition
- Sparse GP fit uses batched `N` by `M` kernel blocks and `M` by `M`
  sufficient statistics; it never builds a dense `N` by `N` covariance
- TFP interface: real GP + Metropolis-Hastings sampling
- PyMC interface: posterior predictive sampling for predictions
- Variational: real ELBO computation with KL divergence
- Posterior predictions average over draws spread across the chain, not the
  first N draws; the count comes from the draw axis, not from the number of
  parameter names
- LOO and WAIC need a *posterior* pointwise log-likelihood matrix. Passing data
  without one falls back to prior draws, which is not cross-validation and whose
  elpd is not comparable with a posterior-based one
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module BAYES`

### Integrations

- **ACT** → Active Inference belief updating and free energy
- **MATH** → Spatial statistics feeding Bayesian models
- **SPM** → Bayesian GLM fitting for parametric maps
- **AI** → Bayesian hyperparameter optimization
- **RISK** → Bayesian uncertainty quantification for risk
