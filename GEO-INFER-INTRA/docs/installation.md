# Installation

Use the [current installation guide](getting_started/installation_guide.md).

GEO-INFER is a Python 3.11+ uv workspace. The supported repository setup is:

`bash
uv sync --all-packages --all-extras --all-groups
uv run python -c "import geo_infer_space, geo_infer_act; print('workspace ready')"
`

The root pyproject.toml, uv.lock, and .python-version define the
environment. Do not substitute a global interpreter or an unrelated package
installation when reproducing a repository result.
