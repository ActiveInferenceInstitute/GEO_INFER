# Agent Instructions: scripts

## Scope

- Owning module: none (repository-level directory outside the GEO-INFER-* module fleet)
- Directory role: Scripts workspace within GEO-INFER.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through the repository's module-level public contracts; this directory defines no Python package.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `_manuscript_fig_common.py`
- `_render_pdf_override.py`
- `manuscript_fig_framework_data_flow.py`
- `manuscript_fig_module_theme_map.py`
- `manuscript_fig_verification_landscape.py`
- `render_manuscript_pdf.py`
- `z_generate_manuscript_variables.py`

## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Manuscript Tooling Rules

- Keep these scripts as thin orchestration surfaces over tracked data.
- Regenerate with `uv run python manuscript/generate_research_artifacts.py`;
  never hand-edit generated figures or the resolved manuscript.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
