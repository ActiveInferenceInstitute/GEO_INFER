# GEO-INFER documentation guide

Documentation must describe the installed APIs and the evidence behind each claim.
Keep conceptual cross-module guidance in `GEO-INFER-INTRA/docs/`, reusable behavior
in the owning package under `src/`, and executable orchestration in examples or
scripts. Planned interfaces belong in a roadmap or tracked issue.

## Generated inventories and authored guidance

Module and directory `README.md` and `AGENTS.md` files are generated signposts.
Regenerate them after changing tracked source, public exports, tests, dependencies,
or validation commands, then review the diff against the intended change.
Do not replace generated operating contracts with hand-written capability or
communication templates.

```bash
uv run --no-sync python GEO-INFER-TEST/rewrite_readme_agents.py
uv run --no-sync python GEO-INFER-TEST/rewrite_readme_agents.py --check
```

Authored module overviews should explain purpose, actual import paths, inputs and
outputs, declared dependencies/extras, configuration, failure cases, and owning
verification commands. Link to source/tests for detailed signatures rather than
maintaining an invented facade. `SKILL.md` should give action-oriented instructions
using real APIs and explain recovery from concrete failures. Keep source-grounded
facts distinct from conceptual relationships and future plans.

## Executable examples

Use tiny deterministic local data, complete imports, and acceptance assertions.
State the dependency profile, units, spatial coordinate order, time convention,
missing-data policy, and expected result where relevant. Keep remote credentials,
private records, hardware setup, and paid/licensed services out of a self-contained
example. Document service-backed examples with explicit prerequisites and direct
acceptance evidence separately.

This example exercises TIME's public contract and compares a numerical result to
an independently specified expectation. It creates no files and makes no network
requests.

```python
import numpy as np
import pandas as pd
from geo_infer_time import TemporalAnalyzer, TimeSeries

axis = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
series = TimeSeries(np.array([-6.0, -4.0, -2.0, 0.0]), timestamps=axis)
result = TemporalAnalyzer().detect_trend(series)
assert str(series.timestamps.tz) == "UTC"
assert result["trend_direction"] == "increasing"
assert np.isclose(result["slope_per_sample"], 2.0)
assert np.isclose(result["r_squared"], 1.0)
np.testing.assert_allclose(result["trend_values"], [-6.0, -4.0, -2.0, 0.0], atol=1e-12, rtol=0)
```

An assertion of non-null imports proves an import contract; it does not prove a
model's scientific correctness, a deployment, or an optional backend's operation.
Prefer behavioral examples with analytical expectations for usage guidance.
Describe omitted evidence explicitly instead of implying success from an import.

The maintained page inventory is
[`GEO-INFER-TEST/doc_examples.json`](../../GEO-INFER-TEST/doc_examples.json).
Its gate rejects empty inventories, duplicate/outside paths, obsolete exemption
banners, absent Python examples, and examples without an acceptance assertion.
It validates the complete manifest before running each page in a fresh isolated
process and temporary directory with a finite deadline and attempt receipt.
Fenced Python blocks on a page execute together in order; avoid collisions between
blocks or make each block self-contained. Unexpected exceptions must propagate.

```bash
uv run --no-sync python GEO-INFER-TEST/validate_doc_examples.py
```

A conceptual pseudocode sketch should use a `text` fence and identify itself as a
conceptual sketch. Do not advertise it as a public API or hide broken imports behind
an exemption banner. For maintained Python usage examples, execute the example
before accepting an interface or dependency change.

## Writing and linking conventions

Use plain, precise language and present-tense behavior. Explain active inference
where an operation actually implements it. Avoid unsupported speed claims,
coverage badges, component versions, or universal domain-validity claims. Define
acronyms, label units, and give observable failure conditions.

Use one H1 title, descriptive headings, and fenced blocks with language tags.
Use relative links from the document's directory for repository material and full
URLs for external references. Check links instead of guessing anchor names.
Code paths, exports, and install commands must match the owning package and root
uv environment. Keep test categories distinct: unit, integration, performance,
system, slow, and H3 each cover declared selections; installed wheels and hosted
checks supply additional evidence.

Google-style docstrings should describe parameters, return values, relevant
exceptions, shapes/order, ownership of mutable inputs, and assumptions. Only add
mathematical or performance claims supported by implementation and independent
reference tests. An AST-extracted signature or source hash identifies bytes; it
alone does not establish runtime behavior or equivalence to a formal proof.

## Verify and retain evidence

From the repository root, run generated-signpost, strict documentation, import,
example, and skill checks after changing guidance:

```bash
uv run --no-sync python GEO-INFER-TEST/rewrite_readme_agents.py --check
uv run --no-sync python GEO-INFER-TEST/validate_documentation.py --strict
uv run --no-sync python GEO-INFER-TEST/validate_doc_imports.py
uv run --no-sync python GEO-INFER-TEST/validate_doc_examples.py
uv run --no-sync python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal
```

Retain revision, dirty state/source custody, interpreter, lock, command, selection,
result counts, logs, and artifact hashes with verification receipts. Separate local
execution from hosted CI, installed-package behavior, live services, and hardware
acceptance. Review generated files and explain deferred verification with a
concrete follow-up. See the [migration guide](releases/0.4.0_migration.md) and
[integration guide](../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md) for current
SPACE/TIME contracts.
