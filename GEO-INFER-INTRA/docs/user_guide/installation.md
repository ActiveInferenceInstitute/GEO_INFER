# Installation Guide

GEO-INFER-INTRA is a member of the GEO-INFER uv workspace. Install it from the
repository checkout; similarly named PyPI projects are not official releases.
The canonical, workspace-wide instructions are in the
[Installation Guide](../getting_started/installation_guide.md).

## Prerequisites

- Python 3.11 or newer
- `uv` on `PATH`
- Git

## Install

```bash
git clone https://github.com/ActiveInferenceInstitute/GEO_INFER.git
cd GEO_INFER

# GEO-INFER-INTRA and its workspace dependencies
uv sync --package geo-infer-intra

# Or the whole workspace with every optional extra
uv sync --all-packages --all-extras --all-groups
```

Dependencies are declared only in `GEO-INFER-INTRA/pyproject.toml` and pinned
by the root `uv.lock`; there is no `setup.py` or `requirements.txt`.

## Configuration

`GEO-INFER-INTRA/config/example.yaml` is the reference configuration. Copy it
before editing:

```bash
cp GEO-INFER-INTRA/config/example.yaml GEO-INFER-INTRA/config/local.yaml
```

## Verify the Installation

```bash
uv run python -c "import geo_infer_intra; print(geo_infer_intra.__version__)"
uv run python -m pytest GEO-INFER-INTRA/tests -q
```

## Troubleshooting

See [Installation Issues](../support/installation_issues.md) for uv, GDAL and
H3 problems, and [Troubleshooting](../support/troubleshooting.md) for runtime
errors.

## Next Steps

- [Your First Analysis](../getting_started/first_analysis.md)
- [Knowledge Base Usage](knowledge_base_usage.md)
