# Agent Instructions: GEO-INFER-EXAMPLES

## Scope

- Owning module: `GEO-INFER-EXAMPLES`
- Python package: `geo_infer_examples`
- Directory role: Comprehensive collection of working examples and tutorials demonstrating cross-module integration patterns and real-world applications.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_examples` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `assessment_results/`
- `config/`
- `docs/`
- `examples/`
- `logs/`
- `scripts/`
- `src/`
- `tests/`
- `.gitignore`
- `SKILL.md`
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES
```


## Workflow Guard Contracts

- Conditional workflow expressions use the constrained data-only evaluator;
  function calls, imports, private attributes, and executable expressions are
  rejected.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
