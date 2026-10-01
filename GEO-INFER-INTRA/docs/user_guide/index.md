# User Guide

This guide provides information for users of GEO-INFER, covering installation,
configuration, and everyday usage.

## Contents

- [Installation](installation.md) — workspace setup and single-package sync.
- [Active Inference Principles](active_inference_principles.md) — core
  concepts for working with ACT models.
- [Cookbook](cookbook.md) — common recipes for spatial, temporal, and
  inference tasks.
- [Knowledge Base Usage](knowledge_base_usage.md) — how to use the
  documentation hub and best practices.

## Quick Start

For those who want to get up and running quickly:

1. **Install** — clone and sync the workspace:

```bash
git clone https://github.com/ActiveInferenceInstitute/GEO_INFER.git
cd GEO_INFER
uv sync --all-packages --all-extras --all-groups
```

2. **Verify** — run the documentation and syntax gates:

```bash
python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
uv run python GEO-INFER-TEST/validate_documentation.py --strict
```

3. **Run an example** — pick a module example under `GEO-INFER-*/examples/`
   or follow the [First Analysis](../getting_started/first_analysis.md)
   tutorial.

See the [Getting Started hub](../getting_started/index.md) for the full
onboarding path and the [Installation Guide](../getting_started/installation_guide.md)
for details.

## Everyday Usage

- **Spatial analysis** — see the [SPACE module page](../modules/geo-infer-space.md)
  and the [H3 guide](../geospatial/data_formats/h3/index.md).
- **Active Inference** — see the [ACT module page](../modules/geo-infer-act.md)
  and the [Active Inference guide](../active_inference_guide.md).
- **Temporal analysis** — see the [TIME module page](../modules/geo-infer-time.md)
  and the [Temporal analysis guide](../temporal_analysis_guide.md).
- **Domain modules** — see the [module catalog](../modules/index.md).

## Configuration

Module configuration lives in each module's `config/` directory and is read at
runtime through the module packages. See the module READMEs for the
configuration reference of each package.

## Troubleshooting

- [Support hub](../support/index.md) — FAQ, installation issues, and error
  triage.
- [Troubleshooting](../support/troubleshooting.md) — problem-solving guides.

## Related Resources

- [Repository README](../../README.md) — repository facts and validation
  commands.
- [Developer Guide](../developer_guide/index.md) — for contributors.
- [Examples gallery](../examples_gallery.md) — runnable examples.
