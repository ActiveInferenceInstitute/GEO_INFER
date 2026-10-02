# Native PyMC sampling profiles

The `bayesian` extra installs the real PyMC backend. `BayesianSPM.fit_bayesian_glm`
uses two independent chains and one worker by default. Callers can explicitly set
`chains` and `cores`; cores cannot exceed chains. Split R-hat requires at least
four retained draws per chain. These choices keep native sampling composable with
parallel module processes. The empirical-Bayes mode remains an explicitly named
approximation with separate diagnostics.

## macOS 27 and Apple Clang

[PyTensor issue 2268](https://github.com/pymc-devs/pytensor/issues/2268) documents
an obsolete `-ld64` argument injected into C compilation on macOS 27. GEO-INFER
provides a named, opt-in repository compiler adapter. It validates macOS 27 or
newer, removes only this exact argument, and executes `/usr/bin/clang++` with all
remaining arguments unchanged. It never disables native compilation or selects
another inference algorithm. No user or global PyTensor configuration is changed.

From the repository root, run:

```bash
PYTENSOR_FLAGS="cxx=$PWD/GEO-INFER-SPM/pytensor_clang++_macos27.py" \
  uv run python GEO-INFER-TEST/run_unified_tests.py --module SPM --category unit
```

The adapter must be executable. A non-macOS platform or macOS before version 27
fails explicitly. Linux CI uses the ordinary compiler configuration on Python
3.11 and 3.12; this adapter is an additional local profile.

Verify actual C compilation separately from sampler success:

```python
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode

x = pt.dvector("x")
compiled = pytensor.function([x], x * x + 3 * x, mode=Mode(linker="c", optimizer="fast_run"))
np.testing.assert_array_equal(compiled(np.array([-2.0, 0.0, 4.0])), [-2.0, 0.0, 28.0])
assert type(compiled.maker.linker).__name__ == "CLinker"
```
