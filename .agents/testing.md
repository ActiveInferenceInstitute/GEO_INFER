# Testing Requirements

## Core Principles

- Test all public methods and functions
- Use real data fixtures, never mock internal logic
- Test mathematical correctness of algorithms
- Keep each module at or above its recorded coverage floor
- Run tests with `uv run python -m pytest` or the unified runner

## Test Organisation

```
tests/
├── conftest.py          # Shared fixtures and test utilities
├── unit/                # Fast, isolated, no I/O
│   ├── test_core.py
│   └── test_utils.py
├── integration/         # Cross-module, may use I/O
│   └── test_pipeline.py
└── performance/         # Benchmarks, scalability
    └── test_benchmarks.py
```

### Naming Conventions

- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<method>_<scenario>_<expected>`
- Fixtures: descriptive nouns (`sample_geodata`, `risk_config`)

## Unit Testing

- Test all public methods with representative inputs
- Include edge cases: empty inputs, boundary values, None/NaN
- Test error paths: invalid inputs, missing data, type mismatches
- Verify mathematical correctness against known values
- Keep unit tests fast (< 1s each)

```python
import pytest
import numpy as np

def test_moran_i_positive_autocorrelation():
    """Clustered values should produce positive Moran's I."""
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([1.0, 1.0, 0.0, 0.0])
    result = compute_morans_i(coords, values)
    assert result > 0.0

def test_moran_i_insufficient_data():
    """< 3 points should return 0.0 gracefully."""
    result = compute_morans_i(np.array([[0, 0]]), np.array([1.0]))
    assert result == 0.0
```

## Integration Testing

- Test cross-module data flow
- Validate API endpoints with realistic payloads
- Test configuration loading from YAML
- Verify error propagation across module boundaries
- Test with real data samples (small fixtures checked into repo)

## Property-Based Testing

Use Hypothesis for algorithm correctness:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1))
def test_normalise_preserves_length(values):
    """Normalisation should preserve list length."""
    result = normalise(values)
    assert len(result) == len(values)
```

Target ≥10 modules with property-based tests, prioritising:

- MATH (numerical operations)
- BAYES (probability distributions)
- SPACE (coordinate transformations)
- ACT (belief updates)

## Coverage Requirements

Each module has a line-coverage floor in
`GEO-INFER-TEST/coverage_baseline.json` (measured coverage rounded down to the
nearest 5%). CI re-measures every module whose `src/` or `tests/` changed and
fails when it drops below its floor:

```bash
uv run --with pytest-cov python GEO-INFER-TEST/check_coverage_floor.py \
    --base main --head HEAD
```

There is no single repository-wide percentage threshold.

## Fixtures and Test Data

Use `conftest.py` for shared fixtures:

```python
import pytest

@pytest.fixture
def sample_geodata():
    """GeoJSON feature collection for testing."""
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-122.4, 37.8]},
             "properties": {"value": 42.0}}
        ]
    }

@pytest.fixture
def risk_config():
    """Minimal risk engine configuration."""
    return {"hazard_types": ["earthquake", "flood"], "return_periods": [100, 250, 500]}
```

## Configuration

Pytest and coverage are configured once in the root `pyproject.toml`
(`[tool.pytest.ini_options]`, `[tool.coverage.run]`, `[tool.coverage.report]`);
modules do not carry their own copies. Key settings:

```toml
[tool.pytest.ini_options]
addopts = ["-ra", "-q", "--strict-markers", "--strict-config",
           "--import-mode=importlib", "-W", "error"]
testpaths = ["GEO-INFER-*/tests", "tests", ...]
markers = ["slow: ...", "integration: ...", "unit: ...", ...]
filterwarnings = ["error"]

[tool.coverage.run]
source = ["GEO-INFER-*/src"]
```

Warnings are errors, and every marker must be declared in the root list.

## Running Tests

```bash
# One module
uv run python -m pytest GEO-INFER-MODULE/tests/

# Specific test file
uv run python -m pytest GEO-INFER-MODULE/tests/unit/test_core.py

# With coverage
uv run python -m pytest GEO-INFER-MODULE/tests/ \
    --cov=GEO-INFER-MODULE/src --cov-report=html

# By marker (markers are declared in the root pyproject.toml)
uv run python -m pytest GEO-INFER-MODULE/tests/ -m "unit and not slow"

# Unified runner: one module or one category across all modules
uv run python GEO-INFER-TEST/run_unified_tests.py --module MODULE
uv run python GEO-INFER-TEST/run_unified_tests.py --category performance
```

## CI Integration

Tests run automatically on every PR via GitHub Actions (`.github/workflows/ci.yml`). The CI pipeline:

1. Runs `uv run python GEO-INFER-TEST/run_unified_tests.py` per category
   (unit, integration, performance, system, H3) on Python 3.11 and 3.12
2. Re-measures coverage for touched modules with
   `GEO-INFER-TEST/check_coverage_floor.py` against `coverage_baseline.json`
3. Runs `ruff check` + `ruff format --check` on changed files and
   `ruff check .` over the whole tree under the root `[tool.ruff.lint]` contract
4. Runs `GEO-INFER-TEST/validate_test_contracts.py --strict`

See `workflow.md` for the full gate table.
