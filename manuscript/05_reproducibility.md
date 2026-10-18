# Reproducibility {#sec:reproducibility}

## Local Generation

After committing authored source changes, refresh the generator-owned metadata
explicitly:

```bash
uv run python manuscript/generate_research_artifacts.py --refresh-config
```

Review and commit the resulting `manuscript/config.yaml` change as a separate
config-only commit. Then generate the inventory, figures, captions, variables,
verification record, and resolved manuscript copies from the clean checkout:

```bash
uv run python manuscript/generate_research_artifacts.py
uv run python manuscript/generate_research_artifacts.py --verify
uv run python manuscript/generate_research_artifacts.py --full-validation
uv run python manuscript/generate_research_artifacts.py \
    --full-validation --publication
uv run python manuscript/generate_research_artifacts.py --check
```

The first command performs the source inventory and artifact generation. The
second records strict repository and research-model checks. The third adds the
full unit, integration, performance, and H3 suites. Generation requires a clean
working tree and current tracked metadata before it writes outputs. An ordinary
development build may opt into `--allow-dirty`, which can refresh metadata and
records the actual working-tree state. Publication explicitly rejects
`--allow-dirty`; it cannot certify uncommitted source. Publication also requires
verification mode and refuses an empty or failing evidence record. A failed
verification result remains available for diagnosis rather than being erased.

The final `--check` command writes nothing. It checks the published evidence
against the measured checkout and validates the registered artifact closure,
including content hashes, so a render cannot accept a token map while ignoring
changed figure bytes. The generated JSON record remains the only source for
the manuscript's verification summary; a successful freshness check is not a
replacement for executing tests.

## Reproducibility Contract

This contract operationalizes computational reproducibility through recorded
commands, inputs, software provenance, and retained intermediate evidence
[@sandve_reproducible_2013]. It does not make the independent observations
needed to assess a scientific model. The hash of an artifact demonstrates
byte identity against a recorded digest; it does not establish the truth of
the artifact's contents.

- Do not cite results that cannot be regenerated or directly traced.
- Keep generated outputs under `output/` and authored manuscript source under
  `manuscript/`, with the single declared exception recorded below.
- Keep private data, credentials, and unpublished sensitive details out of the manuscript.
- Record exact verification commands, commit, source hash, and figure registry
  before marking this manuscript publication-ready.
- Treat `passed`, `failed`, and `not run` as distinct states; never convert an
  absent verification record into a passing claim.
- Never delete an executed-command record a build did not re-measure; carry it
  forward with its own commit and source hash and publish the gap instead.
  A build that does re-measure is widened to cover every group the stored
  record holds, so replacing the record cannot narrow the measured command
  population or omit a previously recorded failure without re-measuring it.
- Publish every verification count against the tier the record was measured
  at, and check that the counts sum to it. The tier is a property of the
  record, not of the build republishing it. Both the numerator and denominator
  must refer to that same command population.

## Generator-Owned Tracked Source

One tracked file under `manuscript/` is written by the generator rather than
by an author: `manuscript/config.yaml`. The renderer copies the tracked
`config.yaml` over the token-resolved copy in `output/manuscript/` before
rendering, so a variable token left in that file would reach the title page
verbatim instead of being substituted. Four fields are therefore resolved into
the tracked file by explicit metadata refresh and marked `# generator-owned` in place: the
project version, the title-page date, the publication year, and the licence.
Everything else in `config.yaml`, and every other file under `manuscript/`,
is authored.

The title-page date is derived from the last commit that touched anything
other than `config.yaml` itself. That choice makes the value a fixed point:
recording the refreshed file creates a newer commit, and a date derived from
`HEAD` could never settle. A clean generation performs a dry-run comparison
first and rejects stale metadata before modifying tracked source or outputs.
Only `--refresh-config`, or a declared dirty development build, writes the
resolved fields. The separate source and config-only commits make the source
date stable while preserving a clean publication tree. The `--check` mode compares the tracked
`config.yaml` against the published token map, so a title page that describes
a different build than the evidence bundle beside it fails rather than
shipping.

## Reusable Figure Evidence

Each registered figure retains the PNG consumed by the existing manuscript
layout, a vector SVG companion, and machine-readable JSON and CSV datasets.
The companions make the encoding and its underlying measurements inspectable
without extracting values from pixels. They do not add new observations or
turn repository counts into scientific outcome measurements. Registry hashes
bind each representation to the published bytes; the inventory and source
fingerprint establish which checkout supplied the data.

These formats support reuse, but durable public deposition, identifiers,
licensing, and retention remain separate responsibilities. The presence of
structured metadata alone is not a FAIR compliance assessment
[@wilkinson_fair_2016].

## Recorded Build

This source manuscript is resolved at commit `{{RESEARCH_COMMIT}}` on branch
`{{RESEARCH_BRANCH}}`, commit date {{RESEARCH_COMMIT_DATE}}, source hash
`{{RESEARCH_SOURCE_HASH}}`, Python `{{PYTHON_VERSION}}`. The uncommitted-entry
count at measurement time was `{{RESEARCH_TREE_DIRTY_FILE_COUNT}}`; anything
other than zero means the values describe files on disk rather than the named
commit, which the commit stamp's `-dirty` or `-unverified` suffix records. The
recorded verification summary is `{{VERIFICATION_STATUS}}`; the record's own
provenance stamps are published with it in the Artifacts and Evidence
section.
