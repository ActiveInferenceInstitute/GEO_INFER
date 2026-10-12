# Installation Guide

GEO-INFER is developed as a repository workspace rather than as a single
monolithic import. Install it with `uv` from the checkout so workspace package
metadata, the shared lockfile, and module extras stay aligned.

## Requirements

- Python 3.11 or newer.
- `uv` installed and available on `PATH`.
- Git for obtaining the repository and inspecting generated documentation.
- A compiler and native geospatial libraries only when an optional dependency
  has no compatible wheel for your platform.

## Reproducible workspace setup

```bash
git clone https://github.com/ActiveInferenceInstitute/GEO_INFER.git
cd GEO_INFER
uv sync --all-packages --all-extras --all-groups
uv run python -c "import geo_infer_space, geo_infer_act; print('GEO-INFER ready')"
```

The shared workspace contract is defined by `pyproject.toml`, `uv.lock`, and
`.python-version`. Keep those files together when reproducing an environment.

## Focused module setup

When working on one module, syncing the package is usually faster:

```bash
uv sync --package geo-infer-space
uv sync --package geo-infer-act
uv sync --package geo-infer-ant
```

Use the module directory's `pyproject.toml`, `README.md`, and `SKILL.md` for
module-specific extras and commands. The module name passed to `uv sync
--package` is the project name from that file, not necessarily the import name.

## Optional dependency profiles

The root project defines grouped extras for AI, Bayesian inference, simulation,
bioinformatics, health, climate, performance, quality, documentation, web, and
IoT work. The broadest local setup is:

```bash
uv sync --all-packages --all-extras --all-groups
```

For a smaller environment, sync the workspace first and add only the extra
needed by the module under development. Optional backends should fail with an
actionable availability error or a documented warning; they must not silently
change the meaning of a successful result.

## Verify the installation

```bash
uv run python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```

For a complete pre-merge check, use the command list in the root
[README.md](../../../README.md) and the CI workflow at
[.github/workflows/ci.yml](../../../.github/workflows/ci.yml).

## Platform notes

- CI runs Python 3.11 and 3.12 on CPU-only Ubuntu runners.
- CUDA-only, source-build, or otherwise native-heavy optional packages are
  intentionally excluded from the hosted CI install; this does not remove the
  extras from the local workspace contract.
- If H3 imports fail, verify the installed version satisfies `h3>=4.5.0,<5`.
- If an import resolves from an unexpected interpreter, use `uv run python` and
  inspect `python -c "import sys; print(sys.executable)"`.

## Environment ownership

Install workspace packages from their `pyproject.toml` metadata and retain the
root `uv.lock`. Run examples through `uv run` from the checkout. A similarly named
package from a registry does not establish that this workspace is installed.

## Next step

Continue with [Your First Analysis](first_analysis.md), or review the
[developer testing guide](../developer_guide/testing_guide.md) if you are
changing source or tests.
