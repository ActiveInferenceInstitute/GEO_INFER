# Environment Setup

This document covers Python environment configuration, package management, environment variables, configuration files, code quality tooling, and CI/CD setup for GEO-INFER.

## Python Requirements

GEO-INFER requires Python 3.11+ (pinned for the workspace in `.python-version`).
The framework uses `uv` as its package manager (not pip, conda, or poetry).

```bash
# Verify Python version
python3 --version  # Must be 3.11+

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify uv
uv --version
```

## Workspace Setup

The repository is a single uv workspace. The root `pyproject.toml` lists every
`GEO-INFER-*` directory as a member, and the root `uv.lock` pins one resolution
for all of them. `uv sync` creates `.venv/` and installs every workspace member
in editable mode, so source changes take effect without reinstalling.

```bash
# Navigate to the repository root
cd /path/to/GEO_INFER

# Install every module with all optional extras (what CI uses)
uv sync --all-packages --all-extras

# Or install a single module and its workspace dependencies
uv sync --package geo-infer-math

# Run commands inside the environment without activating it
uv run python -c "import geo_infer_math, geo_infer_space; print('ok')"
```

CI adds `--locked` so a stale `uv.lock` fails instead of being rewritten, and
skips native-only extras (`cupy`, `mayavi`, `vaex`) that cannot build on its
CPU runners; see `.github/workflows/ci.yml`.

## Dependency Changes

Dependencies are declared only in each module's `pyproject.toml`. Modules have
no `setup.py`, `setup.cfg` or `requirements.txt`; the repository contract
validator rejects them.

```bash
# Add a dependency to a module and refresh uv.lock
uv add --package geo-infer-math "numpy>=1.20.0"

# Add an optional dependency to a named extra
uv add --package geo-infer-math --optional spatial "geopandas>=0.13.0"

# Re-resolve after editing a pyproject.toml by hand
uv lock
uv sync --all-packages --all-extras
```

### Production Builds

Build wheels from the workspace instead of installing source trees:

```bash
# One module
uv build --package geo-infer-math --out-dir dist/

# Every module (the CI build-smoke job)
python GEO-INFER-TEST/build_package_wheels.py --outdir dist/
```

## pyproject.toml Configuration

Each module's `pyproject.toml` defines its metadata, dependencies and optional
extras. Lint, format and pytest configuration live once in the root
`pyproject.toml` (`[tool.ruff]`, `[tool.ruff.lint]`,
`[tool.pytest.ini_options]`).

```toml
# Excerpt: GEO-INFER-MATH/pyproject.toml
[build-system]
requires = ["setuptools>=77.0"]
build-backend = "setuptools.build_meta"

[project]
name = "geo-infer-math"
version = "0.3.0"
license = "CC-BY-NC-SA-4.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "sympy>=1.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "ruff>=0.15.6,<0.16",
    "mypy>=0.910",
]
```

## Environment Variables

GEO-INFER uses environment variables for configuration that varies between environments (development, staging, production). Never hardcode credentials or connection strings.

### Core Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GEO_INFER_ENV` | Deployment environment | `development`, `staging`, `production` |
| `GEO_INFER_LOG_LEVEL` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `GEO_INFER_DATA_DIR` | Base path for local data storage | `/data/geo-infer` |
| `GEO_INFER_CACHE_DIR` | Cache directory for intermediate results | `/tmp/geo-infer-cache` |

### Database Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GEO_INFER_DB_HOST` | PostgreSQL/PostGIS host | `localhost` |
| `GEO_INFER_DB_PORT` | PostgreSQL port | `5432` |
| `GEO_INFER_DB_NAME` | Database name | `geo_infer_db` |
| `GEO_INFER_DB_USER` | Database user | `geo_infer` |
| `GEO_INFER_DB_PASSWORD` | Database password | (set securely) |
| `GEO_INFER_REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |

### External API Keys

| Variable | Description | Source |
|----------|-------------|--------|
| `NOAA_API_TOKEN` | NOAA Climate Data Online | ncdc.noaa.gov |
| `COPERNICUS_UID` | Copernicus CDS user ID | cds.climate.copernicus.eu |
| `COPERNICUS_API_KEY` | Copernicus CDS API key | cds.climate.copernicus.eu |
| `PLANET_API_KEY` | Planet Labs satellite imagery | planet.com |
| `MAPBOX_TOKEN` | Mapbox tile services | mapbox.com |
| `USGS_API_KEY` | USGS data services | usgs.gov |

### Cloud Storage Variables

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_DEFAULT_REGION` | AWS region (e.g., `us-west-2`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob connection string |

## Configuration File Patterns

GEO-INFER supports three configuration patterns. Use the one that fits your deployment.

### YAML Configuration

```yaml
# config/geo_infer.yaml
environment: development

database:
  host: localhost
  port: 5432
  name: geo_infer_db
  pool_size: 5
  max_overflow: 10

spatial:
  default_crs: "EPSG:4326"
  h3_resolution: 7
  tile_cache_size_mb: 512

logging:
  level: INFO
  format: "%(asctime)s %(name)s %(levelname)s %(message)s"
```

Loading YAML configuration:

```python
from pathlib import Path
import yaml

def load_config(config_path: str = "config/geo_infer.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)
```

### Environment File (.env)

```bash
# .env (git-ignored)
GEO_INFER_ENV=development
GEO_INFER_DB_HOST=localhost
GEO_INFER_DB_PORT=5432
GEO_INFER_DB_NAME=geo_infer_db
GEO_INFER_DB_USER=geo_infer
GEO_INFER_DB_PASSWORD=local_dev_password
GEO_INFER_REDIS_URL=redis://localhost:6379/0
NOAA_API_TOKEN=your_token_here
```

Loading with `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Reads .env file

db_host = os.environ.get("GEO_INFER_DB_HOST", "localhost")
db_port = int(os.environ.get("GEO_INFER_DB_PORT", "5432"))
```

### JSON Configuration

```json
{
  "environment": "production",
  "database": {
    "host": "db.internal",
    "port": 5432,
    "name": "geo_infer_db",
    "pool_size": 20
  },
  "spatial": {
    "default_crs": "EPSG:4326",
    "h3_resolution": 7
  }
}
```

## Secrets Management

### Development

Use `.env` files (excluded from git via `.gitignore`):

```bash
echo ".env" >> .gitignore
```

### Staging and Production

Use a secrets manager rather than environment files on disk.

**AWS Secrets Manager:**

```python
import boto3
import json

def get_secret(secret_name: str, region: str = "us-west-2") -> dict:
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

db_creds = get_secret("geo-infer/database")
db_password = db_creds["password"]
```

**Kubernetes Secrets:**

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: geo-infer-secrets
type: Opaque
stringData:
  db-password: "production_password"
  redis-url: "redis://redis:6379/0"
  noaa-token: "production_noaa_token"
```

Mount into pods:

```yaml
env:
  - name: GEO_INFER_DB_PASSWORD
    valueFrom:
      secretKeyRef:
        name: geo-infer-secrets
        key: db-password
```

## Environment-Specific Configuration

### Development

```yaml
environment: development
database:
  host: localhost
  pool_size: 5
logging:
  level: DEBUG
spatial:
  tile_cache_size_mb: 256
```

### Staging

```yaml
environment: staging
database:
  host: staging-db.internal
  pool_size: 10
logging:
  level: INFO
spatial:
  tile_cache_size_mb: 1024
```

### Production

```yaml
environment: production
database:
  host: prod-db.internal
  pool_size: 30
  max_overflow: 20
logging:
  level: WARNING
spatial:
  tile_cache_size_mb: 4096
```

Select the configuration at startup:

```python
import os

env = os.environ.get("GEO_INFER_ENV", "development")
config = load_config(f"config/{env}.yaml")
```

## Code Quality Tools

Ruff is the only lint and format tool. It is configured once in the root
`pyproject.toml`: `[tool.ruff]` targets `py311` with line length 88, and
`[tool.ruff.lint]` selects `E4`, `E7`, `E9`, `F`, `UP`, `B` and `NPY`
(`NPY002` is allowed in tests, examples and scripts). Ruff runs as an
ephemeral tool through `uv run --with`, pinned to the repository range.

```bash
# Format a module
uv run --with 'ruff>=0.15.6,<0.16' ruff format GEO-INFER-MATH/

# Check formatting without changing files
uv run --with 'ruff>=0.15.6,<0.16' ruff format --check GEO-INFER-MATH/

# Lint a module (add --fix to apply safe fixes)
uv run --with 'ruff>=0.15.6,<0.16' ruff check GEO-INFER-MATH/
```

### mypy (Optional Type Checking)

The root `pyproject.toml` carries a strict `[tool.mypy]` configuration for
local use; mypy is not a CI gate.

```bash
uv run mypy GEO-INFER-MATH/src/
```

### CI Quality Gates

`.github/workflows/ci.yml` runs, among other gates:

```bash
# Changed Python files only
uv run --with 'ruff>=0.15.6,<0.16' ruff check <files>
uv run --with 'ruff>=0.15.6,<0.16' ruff format --check <files>

# Repository-wide lint contract (root pyproject [tool.ruff.lint])
uv run --with 'ruff>=0.15.6,<0.16' ruff check .
```

The complete command list is in root `AGENTS.md` "Standard Commands".

## CI/CD Environment Setup

### GitHub Actions

The repository's workflow is `.github/workflows/ci.yml`. A downstream project
that needs PostGIS for integration tests can follow the same uv pattern:

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    services:
      postgres:
        image: postgis/postgis:15-3.3
        env:
          POSTGRES_USER: geo_infer
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: geo_infer_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Set up uv
        uses: astral-sh/setup-uv@v6

      - name: Install locked dependencies
        run: uv sync --locked --python "$(command -v python)" --all-packages --all-extras

      - name: Run tests
        env:
          GEO_INFER_DB_HOST: localhost
          GEO_INFER_DB_PORT: 5432
          GEO_INFER_DB_NAME: geo_infer_test
          GEO_INFER_DB_USER: geo_infer
          GEO_INFER_DB_PASSWORD: test_password
        run: uv run python GEO-INFER-TEST/run_unified_tests.py --category unit

      - name: Code quality
        run: uv run --with 'ruff>=0.15.6,<0.16' ruff format --check .
```

### Docker Development Environment

```dockerfile
# Dockerfile.dev
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . .

RUN uv sync --locked --all-packages --all-extras \
    --no-install-package cupy --no-install-package mayavi \
    --no-install-package vaex --no-install-package vaex-core

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "GEO-INFER-TEST/run_unified_tests.py", "--category", "unit"]
```

Build and run:

```bash
docker build -f Dockerfile.dev -t geo-infer-dev .
docker run --rm geo-infer-dev
```
