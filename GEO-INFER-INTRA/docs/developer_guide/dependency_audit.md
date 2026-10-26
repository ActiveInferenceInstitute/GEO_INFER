# Frozen workspace dependency audit

The standalone audit checks every registry node in the universal
`uv.lock`. It includes all locked Python/platform forks, extras and dependency
groups, even packages excluded from CPU installation such as accelerator extras.
Only local workspace/root nodes whose name, version and source match committed
project metadata are excluded. One hash-bound official Torch CUDA12 variant
retains its complete source/version/wheel identity and maps explicitly to the
public Torch 2.13.0 advisory query. Every other unknown registry, VCS source, missing package
response or skipped advisory query fails; there is no advisory ignore list.

The audit uses exact uv 0.13.1 to check lock freshness and exact pip-audit 2.10.1
against the PyPI advisory service. Fully pinned requirements are split only when
a package has multiple advisory-query versions. The receipt retains every full
locked node separately, even when ordinary PyPI Torch and the reviewed CUDA12
variant share one public-release query. Every queried pair must appear exactly once in
the tool's JSON responses. A fresh temporary advisory cache is used for each
attempt. The tool receives
`--strict --no-deps --disable-pip`: it neither
resolves nor installs the target scientific packages. The separate audit tool
environment does not qualify the GEO scientific runtime.

The controller uses the shared process-tree cleanup and runtime-custody helpers,
which require psutil. From the repository root, provision the minimal isolated
controller with exact uv 0.13.1 (no GEO scientific packages are installed):

```bash
uv venv /absolute/path/to/audit-tools --python 3.12
uv pip install --python /absolute/path/to/audit-tools/bin/python \
  'pip-audit==2.10.1' 'psutil==7.0.0'
/absolute/path/to/audit-tools/bin/python GEO-INFER-TEST/audit_dependencies.py \
  --auditor /absolute/path/to/audit-tools/bin/pip-audit
```

The script runs with Python 3.11+ and the controller prerequisites above, without
installing the scientific workspace. Missing controller imports retain a failed
receipt; they cannot become acceptance. `--uv` selects the exact uv binary; `--output` selects the receipt
parent and `--timeout` bounds the complete attempt. Existing attempts are never
overwritten. The default output is
`.geo-infer-test-results/dependency-audit/<time>-<id>/`.

Each v2 receipt retains the complete locked-node-to-query map, queried pair list, explicit local exclusions,
source/lock/configuration hashes and bookends, tool versions, per-batch fully
pinned requirements, raw JSON/stdout/stderr, advisory findings and errors. A
network/tool failure, malformed/incomplete response, changed source, or any
advisory makes the command fail. Findings without a published fix also fail.
The scheduled/PR/main/tag workflow retains these files on success or failure.

The workspace development group declares `httpx2>=2.12.0` for Starlette
1.3.1 TestClient. Its earlier HTTPX fallback emits a deprecation warning and
therefore fails the strict test profile. The selected HTTPX2 floor includes
its [upstream security fix](https://github.com/pydantic/httpx2/security/advisories/GHSA-8xx6-hgc6-gc2m).
Use the documented full-workspace sync with `--all-groups`; existing HTTPX
consumers and member runtime requirements remain in place. The universal
audit includes the new client and every locked platform branch.

This reports known Python-package advisories at the query time. It does not
certify application exploitability, absence of unknown vulnerabilities, or the
security of native libraries bundled inside a wheel. Actual scientific/native,
installed-wheel, service and release destination checks remain separate.

The release lock updates Brotli 1.2.0 to 1.2.1 for CVE-2026-111497. That targeted
change does not imply that a subsequent full audit is clean; the actual receipt
is the authority. Source: [upstream security release](https://github.com/google/brotli/releases/tag/v1.2.1)
and [PyPA advisory](https://github.com/pypa/advisory-database/blob/main/vulns/brotli/PYSEC-2026-4322.yaml).

Audit behavior follows the [pip-audit security model](https://github.com/pypa/pip-audit#security-model)
and [uv universal-lock semantics](https://docs.astral.sh/uv/concepts/projects/layout/).
The OSV batch API was considered; its empty result means no matching published
advisories, rather than proving that a package/version exists. The selected PyPI
service's skipped/unknown rows are rejected along with any coverage mismatch.


## Linux Torch CUDA library ownership

On Linux x86_64, the frozen source selects official Torch 2.13.0+cu126 from its
explicit PyTorch index. Other platforms retain their PyPI source. This keeps
Torch and XGBoost 3.0.3 on one CUDA12 NCCL distribution instead of installing
different CUDA12/CUDA13 binaries at the same nvidia/nccl/lib/libnccl.so.2 path.
Project Python requirements and the Python 3.11/3.12 native profiles remain.

The geo_infer_test/torch_variant.py owner and its hash-bound release record
authenticate the one source/version and each locked wheel URL/digest. The released
CUDA12 wheel and PyPI Torch 2.13.0 wheel report the same upstream Git revision,
while their CUDA flavors differ. Only this explicit release mapping queries public
Torch 2.13.0 advisories; other registries or local versions are refused. The
record does not claim binary equivalence, absence of upstream unsafe PT2 APIs,
native ABI acceptance, or physical GPU execution.

The isolated wheel validator carries a separately retained, hash-bound direct
Torch wheel constraint selected from the lock for its actual target interpreter.
It does not add Torch to minimal environments that did not require it. This
preserves the locked source when installation runs outside the workspace;
adding a global index-priority fallback would not provide that guarantee.
