# Communication rates and spatial entropy

Use `geo_infer_math.core.information_theory.channel_capacity` for communication
models. A discrete transition matrix contains probabilities, a MIMO matrix
contains real or complex channel amplitudes, and waterfilling takes non-negative
**power gains**. These inputs represent different quantities.

`mimo_channel_capacity(H, noise_power, power_constraint=None, base=2)` evaluates
independent Gaussian streams with equal power at every transmitter. For `H`
with `n_t` columns and an explicit total power `P`, the transmit covariance is
`(P / n_t) I`, and the rate is

\[
C = \log_b \det\left(I + \frac{P}{n_t N} H H^H\right).
\]

This equal-power convention follows the singular-value expression in
[Tse and Viswanath, chapter 8, equations 8.15–8.16](https://stanford.edu/~dntse/Chapters_PDF/Fundamentals_Wireless_Communication_chapter8.pdf).
It describes a fixed isotropic transmit covariance; optimizing a covariance
using channel knowledge is a separate problem. The implementation evaluates
singular-mode log-rates without constructing a Gram matrix or determinant.
Ordinary floating-point SVD precision and conditioning limits still apply.

`power_constraint=None` retains unit power **per transmitter**, for a total of
`n_t`. An explicit constraint is now honored; earlier versions ignored it.
For an identity matrix with two transmitters and noise power one, the default
rate remains two bits, total power one gives `2 * log2(1.5)` bits, and total power
zero gives zero.

```python
import numpy as np
from geo_infer_math.core.information_theory.channel_capacity import (
    mimo_channel_capacity,
    waterfilling_power_allocation,
)

equal_power_rate = mimo_channel_capacity(
    np.eye(2), noise_power=1.0, power_constraint=1.0
)  # 1.1699250014423124 bits
assert np.isclose(equal_power_rate, 2 * np.log2(1.5))

# Parallel-channel power gains, in the original subchannel order.
powers, optimized_rate = waterfilling_power_allocation(
    np.array([0.25, 4.0, 1.0]), noise_power=1.0, total_power=2.0
)
# powers: [0.0, 1.375, 0.625]
np.testing.assert_allclose(powers, [0.0, 1.375, 0.625])
assert np.isclose(optimized_rate, np.log2(6.5) + np.log2(1.625))
```

`waterfilling_power_allocation` allocates a total budget across independent
parallel subchannels. It returns allocations in the input order and their
achieved rate. A channel with zero gain receives zero power. If every gain is
zero, the budget remains unused and the rate is zero. Equal positive gains
share the budget equally. Noise power must be finite and positive; powers,
budgets, bandwidths, and gains must be finite and non-negative. Logarithm bases
for communication rates must be finite and greater than one. Invalid physical
parameters raise `ValueError` rather than yielding misleading rates.

`awgn_channel_capacity` accepts zero signal power or bandwidth and returns zero
in either case. Log-domain evaluation preserves small non-zero SNR contributions
and avoids forming overflowing power ratios. `spatial_channel_capacity` uses
transmitter distances from the origin with the existing minimum distance of
`1e-6` in coordinate units. Zero signal power contributes exactly zero; it is
not replaced with an artificial positive SNR.

For information measures, import `spatial_entropy` from
`geo_infer_math.core.information_theory.entropy`; the separate
`core.spatial_statistics.spatial_entropy` is a different function. Spatial
Shannon and Rényi calculations preserve the requested logarithm base through
both the direct function and `EntropyCalculator`. Use `base=2` for bits,
`base=np.e` for nats, and `base=10` for decimal units. This fixes earlier spatial
calls that always returned the default-base result. Existing Tsallis conventions
are unchanged.

The same public targets are discoverable with `geo-infer catalog MATH` and
`geo-infer describe`, and can be called through the shared script, CLI, and MCP
interfaces. Verify the numerical contracts with:

```bash
uv run python -m pytest GEO-INFER-MATH/tests/unit/test_channel_capacity_contracts.py GEO-INFER-MATH/tests/unit/test_information_theory.py
```
