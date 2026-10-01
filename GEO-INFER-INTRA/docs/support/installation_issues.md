# Installation Issues

This guide covers common installation problems for the GEO-INFER framework, including Python version requirements, system dependencies, and platform-specific fixes.

## Prerequisites

### Python Version

GEO-INFER requires Python 3.11+ (the workspace target). Check your version:

```bash
python3 --version
```

If you need to install a specific Python version:

```bash
# macOS (Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev

# Windows (winget)
winget install Python.Python.3.11
```

### uv Package Manager

GEO-INFER uses `uv` as its package manager. Install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

**Common uv issues:**

| Issue | Fix |
|-------|-----|
| `command not found: uv` | Add `~/.cargo/bin` to PATH, or restart your shell |
| `uv sync` reports a stale lock | Run `uv lock` after editing a `pyproject.toml`, then `uv sync --all-packages --all-extras` |
| Old version of uv | `uv self update` |

## Installing GEO-INFER Modules

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/GEO_INFER.git
cd GEO_INFER

# Create .venv and install every workspace module (editable) with all extras
uv sync --all-packages --all-extras
```

`uv sync` creates `.venv/` itself; use `uv run <command>` instead of activating
it.

### Installing Selected Modules

```bash
# One module and its workspace dependencies
uv sync --package geo-infer-space

# One module with a named optional extra
uv sync --package geo-infer-ai --extra dev --extra docs
```

Workspace siblings resolve through each module's `[tool.uv.sources]`
(`{ workspace = true }`), so run uv from anywhere inside the checkout; it
always uses the root `pyproject.toml` and `uv.lock`. Modules have no
`setup.py` or `requirements.txt`; dependencies come only from `pyproject.toml`.

### Editable Install Issues

**Problem:** Editable install succeeds but imports fail.

**Fix:** Check that the `src` layout is correct. GEO-INFER uses `src/geo_infer_module/` layout:

```bash
# Verify the package is findable
uv run python -c "import geo_infer_space; print(geo_infer_space.__file__)"
```

If this prints `None` or a path outside your repo, the environment holds a stale build. Reinstall the workspace member:

```bash
uv sync --package geo-infer-space --reinstall-package geo-infer-space
```

## GDAL System Dependency

Several GEO-INFER modules depend on GDAL through `rasterio`, `fiona`, or `geopandas`. GDAL requires system-level libraries.

### macOS

```bash
# Install GDAL via Homebrew
brew install gdal

# Verify
gdal-config --version

# Run with Python bindings matching the system version
# (GDAL is not a workspace dependency; rasterio/fiona ship their own GDAL)
uv run --with "GDAL==$(gdal-config --version)" python -c "from osgeo import gdal; print(gdal.__version__)"
```

**Common macOS issues:**

| Error | Fix |
|-------|-----|
| `gdal-config: command not found` | `brew install gdal` |
| `ld: library not found for -lgdal` | `export CFLAGS="-I$(brew --prefix gdal)/include" LDFLAGS="-L$(brew --prefix gdal)/lib"` |
| Architecture mismatch (arm64 vs x86_64) | Ensure Homebrew and Python are both arm64 or both x86_64 |

### Linux (Ubuntu/Debian)

```bash
# Install GDAL and development headers
sudo apt update
sudo apt install gdal-bin libgdal-dev python3-gdal

# Point the binding build at the system headers
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

uv run --with "GDAL==$(gdal-config --version)" python -c "from osgeo import gdal; print(gdal.__version__)"
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install gdal gdal-devel python3-gdal
```

### Windows

`rasterio`, `fiona` and `geopandas` are locked workspace dependencies and ship
binary wheels for Windows, so `uv sync --all-packages --all-extras` installs
them without a system GDAL. Use WSL2 if a native build is still required.

## H3 Library Installation

H3 v4 is required (`h3>=4.5.0,<5`, pinned in `uv.lock`). The Python `h3` package includes pre-built wheels for most platforms, and `uv sync` installs it.

```bash
uv sync --all-packages --all-extras

# Verify
python -c "import h3; print(h3.versions())"
```

**If wheels are not available for your platform** (rare), you need the H3 C library:

```bash
# macOS
brew install h3

# Ubuntu
sudo apt install cmake
uv sync --all-packages --all-extras --no-binary-package h3  # builds from source
```

**H3 v3 vs v4 check:**

```python
import h3

# This works on v4 only
try:
    h3.latlng_to_cell(37.7749, -122.4194, 7)
    print("H3 v4 installed correctly")
except AttributeError:
    print("ERROR: H3 v3 installed. Re-sync the locked environment: uv sync --all-packages --all-extras")
```

## Common uv Error Messages

| Error | Meaning | Fix |
|-------|---------|-----|
| `No solution found when resolving dependencies` | Conflicting version requirements across module `pyproject.toml` files | Read the conflict chain uv prints, align the floors (see `.agents/standards.md` Dependency Floor Policy), then `uv lock` |
| `ERROR: No matching distribution` | Package not available for your Python/OS | Check PyPI for available platforms; consider building from source |
| `subprocess-exited-with-error` during install | C extension build failed | Install system dev libraries (gcc, python3-dev, libffi-dev) |
| `externally-managed-environment` | System Python refuses package installs | Use the workspace environment: `uv sync --all-packages --all-extras` creates `.venv/` |
| `The lockfile at uv.lock needs to be updated` | A `pyproject.toml` changed without re-locking (`--locked` mode) | Run `uv lock` and commit the updated `uv.lock` |

## Virtual Environment Conflicts

### Multiple Virtual Environments

If you have multiple environments, ensure you activate the correct one:

```bash
# Check which Python is active
which python
python -c "import sys; print(sys.prefix)"

# List installed GEO-INFER modules in the workspace environment
uv pip list --python .venv | grep geo-infer
```

### Conda + uv Interaction

If conda provides system libraries (GDAL, PROJ, GEOS), keep it to C libraries
only and let uv own every Python package:

```bash
conda create -n geoinfer-sys gdal proj geos -c conda-forge
conda activate geoinfer-sys

# uv still builds the workspace environment from uv.lock
uv sync --all-packages --all-extras
```

Do not install Python packages with conda into the uv environment; `uv sync`
removes packages that are not in `uv.lock`.

## Platform-Specific Notes

### macOS (Apple Silicon)

- Use the arm64 Homebrew (`/opt/homebrew/bin/brew`)
- Ensure Python is arm64: `python -c "import platform; print(platform.machine())"`
- Some packages may require Rosetta 2 for x86_64 emulation
- If `numpy` or `scipy` attempts a source build, confirm the interpreter is arm64 and re-run `uv sync --all-packages --all-extras` (arm64 wheels are available)

### Windows

- Use PowerShell or Windows Terminal, not Command Prompt
- Long path issues: enable long paths in Group Policy or registry
- Use WSL2 for the most compatible experience with geospatial libraries
- Visual Studio Build Tools may be needed for packages without wheels

### Linux

- Install `python3-dev` (or `python3-devel` on Fedora) for C extension compilation
- Install `libspatialindex-dev` for `rtree` / shapely spatial indexing
- Ensure `proj` and `geos` development headers are installed for shapely/pyproj

## Verification After Installation

Run these checks to verify your installation:

```bash
# 1. Check Python version
uv run python --version  # Should be 3.11+

# 2. Check core imports
uv run python -c "
import geo_infer_math; print('MATH OK')
import geo_infer_space; print('SPACE OK')
import geo_infer_act; print('ACT OK')
"

# 3. Check H3 version
uv run python -c "import h3; print(f'H3 v4: {hasattr(h3, \"latlng_to_cell\")}')"

# 4. Check GDAL (if needed)
uv run --with "GDAL==$(gdal-config --version)" python -c "from osgeo import gdal; print(f'GDAL {gdal.VersionInfo()}')"

# 5. Run module tests
uv run python -m pytest GEO-INFER-MATH/tests/ -v --tb=short -q
```

## Getting Further Help

If the above steps do not resolve your issue:

1. Search [GitHub Issues](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues)
2. Check [Troubleshooting](troubleshooting.md) for runtime errors
3. File a new issue with your OS, Python version, full error output, and the commands you ran

## See Also

- [Troubleshooting](troubleshooting.md) -- runtime error diagnosis
- [Performance Issues](performance_issues.md) -- slow execution and memory problems
- [FAQ](faq.md) -- frequently asked questions
