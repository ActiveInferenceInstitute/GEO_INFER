# manuscript

Manuscript workspace within GEO-INFER.

## Contents

- `sections/`
- `generate_research_artifacts.py`
- `00_abstract.md`
- `01_introduction.md`
- `02_system_context.md`
- `03_methods.md`
- `04_artifacts_and_evidence.md`
- `05_reproducibility.md`
- `06_limitations_and_next_steps.md`
- `98_symbols_glossary.md`
- `99_references.md`
- `S01_source_surface.md`
- `S02_module_catalog.md`
- `SYNTAX.md`
- `config.yaml`
- `preamble.md`
- `references.bib`

## Public Interface

- `generate_research_artifacts.py:ModuleMetrics` (class)
- `generate_research_artifacts.py:FigureSpec` (class)
- `generate_research_artifacts.py:RepositoryInventory` (class)
- `generate_research_artifacts.py:VerificationResult` (class)
- `generate_research_artifacts.py:VerificationRecord` (class)
- `generate_research_artifacts.py:collect_inventory` (function)
- `generate_research_artifacts.py:generate_figures` (function)
- `generate_research_artifacts.py:write_figure_registry` (function)
- `generate_research_artifacts.py:load_matching_verification` (function)
- `generate_research_artifacts.py:resolve_verification` (function)
- `generate_research_artifacts.py:run_verification` (function)
- `generate_research_artifacts.py:defined_command_groups` (function)
- `generate_research_artifacts.py:build_variables` (function)
- `generate_research_artifacts.py:substitute_manuscript_text` (function)
- `generate_research_artifacts.py:config_metadata_values` (function)
- `generate_research_artifacts.py:refresh_config_metadata` (function)
- `generate_research_artifacts.py:write_resolved_manuscript` (function)
- `generate_research_artifacts.py:bibliography_policy` (function)
- `generate_research_artifacts.py:audit_bibliography` (function)
- `generate_research_artifacts.py:audit_module_census` (function)


## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Research Verification

- `uv run python manuscript/generate_research_artifacts.py --verify` runs the
  producer's verification groups and retains immutable execution receipts.
- Ordinary failures and timeouts continue the diagnostic sweep. An interruption
  retains its failed attempt and stops admission; later groups remain visibly
  not run. An incomplete sweep cannot establish acceptance.
- Add `--full-validation` for the unit, integration, performance and H3 groups.
  Run ROOT manuscript and manuscript-render profiles separately; the render
  profile requires the actual generated PDF.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
