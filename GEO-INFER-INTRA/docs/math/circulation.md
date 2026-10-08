# Finite probability currents and graph decomposition

The owning API is `geo_infer_math.core.circulation`. These are finite real
numerical operations, with input validation and no implicit normalization.
They provide implemented anchors for [issue #58](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/58);
external notation-bridge integration and formal proof acceptance are separate.

`skew_part(A)` returns `(A - A.T)/2`. Its trace and real quadratic form vanish;
adding the symmetric part reconstructs the input. Scaling before subtraction
preserves finite opposite-sign entries near the floating-point limit.

`probability_current(K, pi, kind="transition")` accepts a nonnegative
row-stochastic transition matrix and a normalized nonnegative probability vector.
`kind="rate"` instead requires nonnegative off-diagonals, nonpositive diagonals
and zero row sums. State order is preserved. The vector need not be stationary.
The returned current is `J[i,j] = pi[i] K[i,j] - pi[j] K[j,i]`.
`circulation_part` returns half that current; `satisfies_detailed_balance`
checks each current entry against an explicit absolute tolerance.

`current_divergence(J)[j] = sum_i J[i,j]` uses the incoming convention.
For transition kernels this equals `pi @ K - pi`; for rate kernels it equals
`pi @ K`. Thus stationarity gives zero divergence but does not imply detailed
balance: a directed three-state cycle has a nonzero divergence-free current.
Dropping the circulation loses that current.

```python
import numpy as np
from geo_infer_math.core.circulation import (
    probability_current, current_divergence, satisfies_detailed_balance,
    graph_hodge_decomposition,
)

P = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
pi = np.full(3, 1/3)
J = probability_current(P, pi)
assert np.allclose(current_divergence(J), 0)
assert not satisfies_detailed_balance(P, pi)

B = np.array([[-1., 0., 1.], [1., -1., 0.], [0., 1., -1.]])
parts = graph_hodge_decomposition(B, [2., -1., 5.], np.ones((3, 1)))
assert np.allclose(parts.gradient, [0., -3., 3.])
assert np.allclose(parts.solenoidal, [2., 2., 2.])
assert np.allclose(parts.harmonic, 0)
```

The unweighted discrete graph contract uses incidence `B` (nodes by edges), with
one -1 source and +1 target per edge. Optional face boundary `C` (edges by faces)
must satisfy `B @ C = 0`. Least-squares projections return gradient in
`range(B.T)`, face-curl (`solenoidal`) in `range(C)` and the explicit harmonic
remainder. The remainder lies in `ker(B)` and `ker(C.T)` up to numerical error.
Node and face potentials use the minimum-norm gauge. Disconnected and edgeless
graphs are supported. Without faces, the face-curl is zero and cycle flow is
reported as harmonic. The total divergence-free component is the sum of
`solenoidal` and `harmonic`; the names depend on the supplied finite complex.
This API does not solve continuum PDE boundary conditions or infer faces.

Verify with:

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --paths GEO-INFER-MATH/tests/unit/test_circulation.py
```
