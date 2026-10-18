# Agent Instructions: manuscript

## Scope

- Owning module: none (repository-level directory outside the GEO-INFER-* module fleet)
- Directory role: Manuscript workspace within GEO-INFER.

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

## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Research Verification

- After committing source changes, run `uv run python manuscript/generate_research_artifacts.py --refresh-config`
  and commit only `manuscript/config.yaml` separately. Config-only commits do
  not move the measured source date. Clean generation rejects stale metadata
  before any write; never restore old config after producing hash-bound output.
- `uv run python manuscript/generate_research_artifacts.py --verify` runs the
  producer's verification groups and retains immutable execution receipts.
- Static bibliography errors fail before verification or generated-output writes;
  the resolved manuscript is checked again after substitution.
- Ordinary failures and timeouts continue the diagnostic sweep. An interruption
  retains its failed attempt and stops admission; later groups remain visibly
  not run. An incomplete sweep cannot establish acceptance.
- Add `--full-validation` for the unit, integration, performance and H3 groups.
  Run ROOT manuscript and manuscript-render profiles separately; the render
  profile requires the actual generated PDF.
- Each figure exports PNG, text-preserving SVG, typed JSON and source-stamped
  CSV with per-file SHA-256. `--check` also verifies the complete delivered
  byte set, including resolved manuscript files and the figure registry.
  See [research artifact contracts](../GEO-INFER-INTRA/docs/integration/research_artifacts.md).

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
