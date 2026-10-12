# Guide for Autonomous Agent Coders

Start with the repository and applicable module `AGENTS.md`, inspect the current
checkout/diff, and read the owning source/tests before editing. Preserve concurrent
work and keep changes in their owner. Root `pyproject.toml`, `uv.lock`, and
`.python-version` define the shared uv workspace; scripts remain thin orchestration.

## Orientation and verifiable contracts

Read the root module index, `TODO.md`, module `SKILL.md`, and connected callers.
Define the successful result and meaningful failure cases before implementation.
If a code index is unavailable, inspect actual imports and callers directly rather
than claiming an impact analysis from a missing index.

The small test below exercises a real timestamp public export. Offset equivalence
is an independent boundary invariant; a constructor mock would bypass it.

```python
from datetime import datetime, UTC
from geo_infer_time import normalize_timestamp

expected = datetime(2024, 1, 1, tzinfo=UTC)
assert normalize_timestamp("2023-12-31T16:00:00-08:00") == expected
assert normalize_timestamp("2024-01-01T00:00:00Z") == expected
try:
    normalize_timestamp("2024-01-01")
except ValueError:
    pass
else:
    raise AssertionError("ambiguous source time was accepted")
```

## Implementation, verification, and handoff

Keep behavior under the owning `src/geo_infer_*/` package and cross-module guidance
in INTRA. Libraries use module loggers; CLI entry points configure handlers.
Declare new package dependencies and migrate direct callers when a contract changes.
Coordinate file ownership before shared edits and inspect focused diffs afterward.

Run the narrowest meaningful pytest selection first, then module suites and the
repository's strict contract validators. Test subprocess deadlines with real
children, storage with a fresh reader, transport with local endpoints, and inference
with analytical oracles. Skips, empty selections, and unaccounted deselection are
not successful execution. Regenerate module README/AGENTS signposts after tracked
surfaces change; use the root documented commands for release gates.

A handoff records changed paths, exact commands/results, dependency or migration
implications, and deferred checks. Separate local tests from installed-wheel,
hosted-CI, source-custody, and live-service evidence. Use fresh independent review
for security and shared infrastructure. Do not commit, publish, deploy, communicate
externally, or alter unrelated state beyond the user's authorized scope.

See [repository instructions](../../../AGENTS.md),
[testing guide](testing_guide.md), [contributing](contributing.md), and
[cross-module contracts](../architecture/cross_module_interaction.md).
