# Setting Up GEO-INFER-INTRA

This tutorial sets up GEO-INFER-INTRA from the repository checkout for
first-time use.

## Prerequisites

- Python 3.11 or newer
- `uv` (Python package manager)
- Git
- A text editor

## Step 1: Install GEO-INFER-INTRA

Install from the canonical monorepo; similarly named PyPI projects are not
official releases.

```bash
git clone https://github.com/ActiveInferenceInstitute/GEO_INFER.git
cd GEO_INFER

# Creates .venv/ and installs GEO-INFER-INTRA with its workspace dependencies
uv sync --package geo-infer-intra
```

Use `uv sync --all-packages --all-extras` instead when you will work across
modules.

## Step 2: Create a Configuration File

```bash
mkdir -p ~/.geo-infer/
cp GEO-INFER-INTRA/config/example.yaml ~/.geo-infer/config.yaml
```

## Step 3: Edit the Configuration File

Open `~/.geo-infer/config.yaml` and adjust the sections you need. The example
file documents each setting; the top-level sections are:

```yaml
general:
  debug_mode: false
  log_level: INFO
  log_file: ~/.geo-infer/logs/intra.log

documentation:
  server:
    host: 127.0.0.1
    port: 8000
```

## Step 4: Verify the Installation

```bash
uv run python -c "import geo_infer_intra; print(geo_infer_intra.__version__)"
uv run python -m pytest GEO-INFER-INTRA/tests -q
```

## Step 5: Generate Module Previews

`geo_infer_intra` exports reproducible documentation-preview utilities:

```bash
uv run python -c "
from geo_infer_intra import generate_module_preview_suite
artifacts = generate_module_preview_suite('SPACE', '/tmp/geo-infer-previews')
print(artifacts)
"
```

## Next Steps

- [Tutorials index](../index.md)
- [Your First Analysis](../../getting_started/first_analysis.md)
- [Knowledge Base Usage](../../user_guide/knowledge_base_usage.md)

## Troubleshooting

- Confirm prerequisites with `uv run python --version` and `uv --version`.
- Verify the configuration file is valid YAML.
- See [Installation Issues](../../support/installation_issues.md) and
  [Troubleshooting](../../support/troubleshooting.md).

If problems persist, [open an issue](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues).
