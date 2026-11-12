# scripts

Scripts workspace within GEO-INFER.

## Contents

- `_manuscript_fig_common.py`
- `_render_pdf_override.py`
- `gate.py`
- `manuscript_fig_framework_data_flow.py`
- `manuscript_fig_module_theme_map.py`
- `manuscript_fig_verification_landscape.py`
- `render_manuscript_pdf.py`
- `z_generate_manuscript_variables.py`

## Public Interface

- `gate.py:Arm` (class)
- `gate.py:main` (function)
- `manuscript_fig_framework_data_flow.py:main` (function)
- `manuscript_fig_module_theme_map.py:main` (function)
- `manuscript_fig_verification_landscape.py:main` (function)
- `render_manuscript_pdf.py:RenderError` (class)
- `render_manuscript_pdf.py:main` (function)
- `z_generate_manuscript_variables.py:main` (function)


## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Manuscript Tooling

- These seven tracked helpers build the manuscript figures and render the PDF;
  the manuscript source and artifact generator live in `manuscript/`.
- The `manuscript_fig_*.py` scripts export the framework, theme, and verification
  figures into generated `output/figures/`.
- `z_generate_manuscript_variables.py` and `render_manuscript_pdf.py` hydrate the
  template variables and render the PDF; regenerate artifacts rather than
  hand-editing the produced files.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
