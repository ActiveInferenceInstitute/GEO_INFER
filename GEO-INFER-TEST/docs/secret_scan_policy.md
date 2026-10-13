# Secret-Scan Policy (SEC-02)

The repository runs gitleaks over the **full git history** on every pull
request and every push/pull request targeting `main` **or** `develop` (the
"Install pinned gitleaks" and "Scan full history for secrets" steps; the
binary is a sha256-pinned 8.30.1 release tarball). The committed
`.gitleaks.toml` extends gitleaks' default rule set with file-path- and
rule-scoped allowlists; the scan is fail-closed (any non-allowlisted
finding fails the job).

## Cadence

- Every pull request and every push to `main` or `develop`: full-history scan.
- The autoresearch harness (`GEO-INFER-TEST/secret_scan_metric.py`)
  measures `secret_scan_findings` on demand; the 2026-09-08 baseline was
  29 findings (default rules, 426 commits, 1.22 GB).

## Local replication

CI installs a sha256-pinned linux x64 tarball, so its binary is not what a
local machine runs; replicate the scan locally with the same committed
policy and the same gitleaks version before pushing:

```bash
# macOS (darwin/arm64): Homebrew carries the same 8.30.x release.
brew install gitleaks
gitleaks version  # expect 8.30.1, matching the CI pin

# The CI invocation, verbatim, from the repository root (full history):
gitleaks detect --source . --config .gitleaks.toml --redact --verbose
```

On darwin/arm64 the Homebrew bottle is the supported install path; a
`gitleaks_8.30.1_darwin_arm64.tar.gz` release tarball also exists for
environments without Homebrew (verify its checksum against the GitHub
release's checksums file before use). Linux x64 users can replay the CI
pin exactly: download `gitleaks_8.30.1_linux_x64.tar.gz` from the same
release URL as `ci.yml`, verify the committed sha256
`551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`, and
run the same command. A local scan that reports a non-allowlisted finding
must be remediated before the push — CI detects the same leak only after
the secret is already in the remote history.

## 2026-09-08 audit of the default-rule findings

All 29 findings were audited site-by-site before any allowlist decision.
None is a live credential:

| Site (rule, count) | Classification | Justification |
| --- | --- | --- |
| `GEO-INFER-INTRA/docs/api/index.md` (curl-auth-header, 8) | Documentation | The page teaches how to call module APIs with curl; every Authorization header carries a placeholder token. |
| `GEO-INFER-INTRA/docs/geospatial/data_formats/h3/h3_programming_interfaces.md` (generic-api-key, 2) | Documentation | Example requests in the H3 interface guide use placeholder keys. |
| `GEO-INFER-EXAMPLES/docs/API_INTEGRATION_GUIDE.md` (generic-api-key, 4) | Documentation | Integration-guide examples use placeholder keys. |
| `GEO-INFER-DATA/README.md` (generic-api-key, 2), `GEO-INFER-REQ/README.md` (generic-api-key, 1) | Documentation | README usage examples use placeholder keys. |
| `GEO-INFER-COMMS/tests/unit/test_rest_api.py` (4), `GEO-INFER-DATA/tests/unit/test_compression.py` (2), `GEO-INFER-OPS/tests/conftest.py` (1), `GEO-INFER-OPS/tests/test_config.py` (1), `GEO-INFER-OPS/tests/test_security.py` (1), `GEO-INFER-SEC/tests/conftest.py` (1) | Test fixtures | Synthetic credentials (`test-secret-...`, `test_key_abc123...`, monkeypatched JWT secrets) that the tests feed to the code under test. |
| `GEO-INFER-DATA/src/geo_infer_data.egg-info/PKG-INFO` (2) | Historical build artifact | The egg-info tree was tracked in pre-2026-03 commits and is untracked at HEAD; the findings exist only in history. |

## Allowlist policy

- Entries are scoped to exact file paths **and** rule IDs; nothing is
  allowlisted glob-wide, so a real credential in a new file (or in an
  unlisted rule) still fails the gate.
- A site enters the allowlist only after a site-by-site audit confirms it
  is synthetic, with the justification recorded here and in the
  `.gitleaks.toml` entry description.
- A real credential is **never** allowlisted: it is revoked with the
  provider first, then removed from the tree, then the scan is re-run.

## 2026-10-12 Cascadia manifest findings

The full-history scan of `4660d3cfe1752e9a5f1dda2eee29a29fed7ea9bb`
reported four `generic-api-key` findings at lines 138, 157, 387 and 406 of
`GEO-INFER-ACT/examples/cascadia/data/cascadia_seed_study_20261012_manifest.json`.
The rule reads API-related filenames followed by high-entropy SHA-256 values.
Each value occurs in the bound-files map and the source-owner map:

| Public source owner | Independently recomputed SHA-256 |
| --- | --- |
| `GEO-INFER-SPACE/src/geo_infer_space/api/rest_api.py` | `bc217359dea513d9dbf4b82094ef3ca4ec485913cd59d912844e26565e0f0ec9` |
| `GEO-INFER-SPACE/src/geo_infer_space/core/api_clients.py` | `7e3f6bba60139bdcbcf1882e86d185d0391ef5594ee35cb0adc56c0a95af936c` |

These values match the exact public file bytes, not credentials. The new
exception requires **both** one of the two named edition-manifest paths and
one of these two exact extracted digests, and applies only to
`generic-api-key`. It excludes no archive, whole manifest or rule. A different
credential in the same manifest, or either value in a different path, remains
subject to detection. The configuration uses the pinned version's
[documented AND matching](https://github.com/gitleaks/gitleaks/blob/v8.30.1/README.md#configuration).

The checksum-verified Darwin/arm64 8.30.1 binary was invoked on three real
temporary Git repositories. The exact named-path/digest fixture exits 0 with
no findings; a different 64-character value in the same manifest exits 1, and
the allowed value in `notes.json` exits 1. Both rejected cases report
`generic-api-key`. A full-history scan reports zero findings across 1,043
commits and approximately 2.45 GB. Logs and redacted JSON receipts remain
under `GEO-INFER-ACT/examples/outputs/gitleaks_controls_20261012` and
`cascadia_gitleaks_history_20261012.*`. Later commits require a new scan.

## Pre-rewrite objects

The 2026-09-07 published-history rewrite means GitHub may still serve
pre-rewrite commit or blob objects when addressed by their exact SHA,
even though the rewritten history no longer references them. The scan
runs against the rewritten local history and cannot reach such objects.
Policy for a surfaced pre-rewrite object: treat it as historical, assume
its content is public, verify no credential in it is still live (revoke
first if one is), and record the object and the decision in this file
with a date. Pre-rewrite SHAs are historical identifiers under the
ledger's history note.
