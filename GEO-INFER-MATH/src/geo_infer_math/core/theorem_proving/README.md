# GEO-INFER-MATH/src/geo_infer_math/core/theorem_proving

Theorem Proving workspace within `GEO-INFER-MATH`.

## Contents

- `__init__.py`
- `integration.py`
- `proof_strategies.py`
- `proof_verification.py`
- `prover.py`
- `spatial_theorems.py`

## Public Interface

- `integration.py:integrate_with_symbolic_math` (function)
- `integration.py:generate_proof_from_symbolic` (function)
- `integration.py:verify_symbolic_operation` (function)
- `integration.py:SymbolicProofIntegrator` (class)
- `proof_strategies.py:ProofStrategy` (class)
- `proof_strategies.py:GeometricProofStrategy` (class)
- `proof_strategies.py:StatisticalProofStrategy` (class)
- `proof_strategies.py:DirectProofStrategy` (class)
- `proof_strategies.py:ContradictionProofStrategy` (class)
- `proof_strategies.py:InductionProofStrategy` (class)
- `proof_strategies.py:ProofStrategySelector` (class)
- `proof_verification.py:verify_proof` (function)
- `proof_verification.py:ProofVerifier` (class)
- `prover.py:ProofStatus` (class)
- `prover.py:ProofResult` (class)
- `prover.py:TheoremProver` (class)
- `prover.py:create_prover` (function)
- `spatial_theorems.py:TheoremType` (class)
- `spatial_theorems.py:SpatialTheorem` (class)
- `spatial_theorems.py:GeometricTheorem` (class)

## Module Metadata

- Module: `GEO-INFER-MATH`
- Package: `geo_infer_math`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-math`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `sympy>=1.9.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
