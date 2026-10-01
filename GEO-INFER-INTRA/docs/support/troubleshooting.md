# Troubleshooting

This guide provides a systematic approach to diagnosing and resolving issues in the GEO-INFER framework. It covers common error categories, step-by-step diagnosis workflows, and guidance on when to file a GitHub issue.

## Systematic Debugging Approach

Follow this sequence for any issue:

1. **Read the full traceback** -- the bottom of the traceback has the actual error
2. **Identify the error category** (import, CRS, memory, H3, dependency)
3. **Reproduce with minimal code** -- strip away everything not needed
4. **Check the known issues** for your error category below
5. **Search existing GitHub issues** before filing a new one

## Common Error Categories

### Import Errors

**Symptom:** `ModuleNotFoundError` or `ImportError` when importing a GEO-INFER module.

**Diagnosis:**

```bash
# Check if the module is installed
uv pip list | grep geo-infer

# Check if you are in the correct virtual environment
which python

# Try importing the base package
uv run python -c "import geo_infer_space; print(geo_infer_space.__file__)"
```

**Common causes and fixes:**

| Error Message | Cause | Fix |
|--------------|-------|-----|
| `No module named 'geo_infer_space'` | Module not installed, or command run outside the workspace environment | `uv sync --all-packages --all-extras`, then run through `uv run` |
| `No module named 'h3'` | Environment not synced | `uv sync --all-packages --all-extras` |
| `ImportError: cannot import name 'latlng_to_cell'` | H3 v3 installed instead of the locked v4 | `uv sync --all-packages --all-extras` (restores `h3>=4.5.0,<5` from `uv.lock`) |
| `No module named 'tensorflow_probability'` | Optional dependency | Install TFP or use NumPy fallback |

GEO-INFER modules use graceful degradation for optional dependencies. If you see a log message like `"TensorFlow Probability not installed; using NumPy/SciPy GP backend."`, the module is working correctly with the fallback.

### CRS Mismatch Errors

**Symptom:** Spatial operations produce wrong results, geometries appear in the wrong location, or you get explicit CRS warnings.

**Diagnosis:**

```python
import geopandas as gpd

gdf = gpd.read_file("data.geojson")
print(f"CRS: {gdf.crs}")
print(f"Bounds: {gdf.total_bounds}")

# Check if bounds make sense for the expected CRS
# WGS84 (EPSG:4326): lon in [-180, 180], lat in [-90, 90]
# UTM zones: coordinates in meters, typically 100000-900000
```

**Common causes and fixes:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Points appear near (0, 0) | Missing CRS assignment | `gdf = gdf.set_crs("EPSG:4326")` |
| Coordinates in millions | Data in projected CRS, treated as geographic | Check source CRS and reproject |
| Buffer produces tiny results | Geographic CRS used for metric buffer | Reproject to local UTM first |
| Spatial join returns no matches | CRS mismatch between layers | `gdf2 = gdf2.to_crs(gdf1.crs)` |

### Memory Errors

**Symptom:** `MemoryError`, process killed by OOM killer, or system becomes unresponsive.

**Diagnosis:**

```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_info = process.memory_info()
print(f"RSS: {mem_info.rss / 1e9:.2f} GB")
print(f"System available: {psutil.virtual_memory().available / 1e9:.2f} GB")
```

**Common causes and fixes:**

| Scenario | Cause | Fix |
|----------|-------|-----|
| Loading large raster | Full raster loaded into memory | Use `rasterio` windowed reads |
| GP model fitting | n^2 kernel matrix | Reduce n or use sparse approximations |
| H3 at high resolution | Too many cells generated | Lower H3 resolution |
| Pandas DataFrame explosion | Spatial join producing cartesian product | Check join keys, add spatial filter first |

**Quick memory reduction techniques:**

```python
# Reduce DataFrame memory by downcasting types
def reduce_memory(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        if df[col].min() >= 0 and df[col].max() < 65535:
            df[col] = df[col].astype("uint16")
    return df
```

### H3 API Compatibility

GEO-INFER-SPACE uses H3 v4. The v3 to v4 API changed function names.

| v3 Function | v4 Function |
|-------------|-------------|
| `h3.geo_to_h3(lat, lng, res)` | `h3.latlng_to_cell(lat, lng, res)` |
| `h3.h3_to_geo(cell)` | `h3.cell_to_latlng(cell)` |
| `h3.h3_to_geo_boundary(cell)` | `h3.cell_to_boundary(cell)` |
| `h3.k_ring(cell, k)` | `h3.grid_disk(cell, k)` |
| `h3.h3_get_resolution(cell)` | `h3.get_resolution(cell)` |
| `h3.h3_to_parent(cell, res)` | `h3.cell_to_parent(cell, res)` |
| `h3.h3_to_children(cell, res)` | `h3.cell_to_children(cell, res)` |

If you see `AttributeError: module 'h3' has no attribute 'geo_to_h3'`, you have H3 v4 installed but are calling v3 functions.

### Dependency Conflicts

**Symptom:** `pip`/`uv` fails to resolve dependencies, or runtime errors from version mismatches.

**Diagnosis:**

```bash
# Check for conflicting versions
uv pip check

# See what version of a package is installed
uv pip show numpy

# See dependency tree
uv pip list --format=columns
```

**Common conflicts:**

| Conflict | Root Cause | Fix |
|----------|-----------|-----|
| numpy version mismatch | Module A needs numpy>=1.24, B pins <1.24 | Update both module `pyproject.toml` files, then `uv lock` |
| GDAL binding mismatch | System GDAL differs from Python binding | `uv run --with "GDAL==$(gdal-config --version)" ...` (GDAL is not a workspace dependency) |
| shapely 1.x vs 2.x | API changed between major versions | Use `shapely>=2.0` |

## Step-by-Step Diagnosis Workflow

### For Runtime Errors

```
1. Copy the FULL traceback
2. Identify the last line (the actual exception)
3. Identify which GEO-INFER module raised it (look at file paths)
4. Check if the error is from GEO-INFER code or a dependency
5. If dependency: check version compatibility
6. If GEO-INFER code: look at the line that raised it
7. Reproduce with minimal input
```

### For Wrong Results

```
1. Check input data: dtypes, CRS, null values, bounds
2. Check intermediate results at each pipeline step
3. Compare against a known-good result (even a small manual calculation)
4. Visualize spatial data to catch obvious errors
5. Check units (meters vs degrees, UTC vs local time)
```

### For Performance Issues

See [Performance Issues](performance_issues.md) for detailed guidance.

## Reading Python Tracebacks for Geospatial Errors

Geospatial tracebacks often pass through multiple libraries. Here is how to read them:

```
Traceback (most recent call last):
  File "my_script.py", line 15, in main          <-- YOUR CODE (start here)
    result = analyzer.cluster_points(data)
  File ".../geo_infer_space/core/clustering.py"   <-- GEO-INFER MODULE
    labels = _run_dbscan(coords, eps)
  File ".../sklearn/cluster/_dbscan.py"           <-- DEPENDENCY
    ...
  File ".../scipy/spatial/distance.py"            <-- LOW-LEVEL DEPENDENCY
    raise ValueError("Input contains NaN")
ValueError: Input contains NaN                    <-- ACTUAL ERROR
```

Read from bottom to top. The fix is almost always at the boundary between your code and the GEO-INFER module -- in this case, clean NaN values from your input data.

## Using Pytest Markers

GEO-INFER uses pytest markers to categorize tests. Use them to run targeted test suites when diagnosing issues.

```bash
# Run only unit tests for a module
uv run python -m pytest GEO-INFER-SPACE/tests/ -m unit -v

# Run integration tests
uv run python -m pytest GEO-INFER-SPACE/tests/ -m integration -v

# Run geospatial-specific tests
uv run python -m pytest GEO-INFER-SPACE/tests/ -m geospatial -v

# Run fast tests only (skip slow ones)
uv run python -m pytest GEO-INFER-SPACE/tests/ -m fast -v

# Run API tests
uv run python -m pytest GEO-INFER-API/tests/ -m api -v

# Run performance benchmarks
uv run python -m pytest GEO-INFER-SPACE/tests/ -m performance -v
```

Available markers: `unit`, `integration`, `system`, `performance`, `geospatial`, `api`, `slow`, `fast`.

## Log Analysis

Enable debug logging for detailed diagnostics:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or target a specific module
logging.getLogger("geo_infer_bayes").setLevel(logging.DEBUG)
logging.getLogger("geo_infer_act").setLevel(logging.DEBUG)
```

Look for these log patterns:
- `"TensorFlow Probability not installed"` -- fallback backend in use (not an error)
- `"GP model fitted"` -- successful model fitting with parameters
- Warnings about numerical instability in Cholesky decomposition

## When to File a GitHub Issue

File an issue at [github.com/ActiveInferenceInstitute/GEO_INFER/issues](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues) when:

- You can reproduce a bug with a minimal example
- Documentation is incorrect or misleading
- A module behaves differently than its docstring describes
- You encounter a segfault or corruption (always report these)

**Include in your issue:**

1. GEO-INFER module and version
2. Python version (`python --version`)
3. Operating system
4. Minimal reproduction script
5. Full traceback
6. Expected vs actual behavior

## See Also

- [Installation Issues](installation_issues.md) -- setup and dependency problems
- [Performance Issues](performance_issues.md) -- slow runs and memory problems
- [FAQ](faq.md) -- frequently asked questions
