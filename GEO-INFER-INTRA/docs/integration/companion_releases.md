# GEO-INFER 0.4.0 companion compatibility

The release identities in `.github/companion-releases.json` connect GEO-INFER's
0.4.0 candidate to published GNN 5.0.1 and FEP_Formal 2.1.0. They identify
sources and declared pins. New GEO source, numerical behavior, manuscript
output and hosted results require their own acceptance.

| Project | Published tag | Exact source commit | Exact annotated tag object | Automatic source DOI |
| --- | --- | --- | --- | --- |
| GNN | `v5.0.1` | `e54813b4197df36f7ae0054d5e79f3ad2d5db0ff` | `35e71d1a6f109f6e0d08fda2acd5a94376482cf4` | [10.5281/zenodo.23631909](https://doi.org/10.5281/zenodo.23631909) |
| FEP_Formal | `v2.1.0` | `96bf714d91c088e79b9dea54af0f9151f70de64f` | `6bc61ffa58428190055245781328820c7c86abac` | [10.5281/zenodo.23631895](https://doi.org/10.5281/zenodo.23631895) |

The published release pages are [GNN 5.0.1](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v5.0.1)
and [FEP_Formal 2.1.0](https://github.com/ActiveInferenceInstitute/fep_formal/releases/tag/v2.1.0).

The DOI links identify automatic software source archives. Verifying the
public metadata, downloaded source bytes, separate distribution assets and
manuscript publication remains a separate release procedure. The offline
identity check performs no network requests or DOI resolution.

## Direction of the pins

GEO's `.github/gnn-pair.json` selects the exact GNN source above. That GNN
release's `.github/fep-lean-pair.json` selects the exact FEP_Formal source above.
GNN 5.0.1's own GEO pin remains the accepted historical GEO commit
`21b65a26dd15288d6e578baef86a8bcfd6a7b5db`. A new GEO candidate must retain
that commit as an ancestor and obtain its own paired interchange evidence.
GNN's already published tag does not pin a later GEO release.

The supported interchange schemas remain `gnn-geo-infer/1`,
`gnn-geo-infer/2` and `gnn-geo-infer/factored/1`. The existing
[two-environment interchange procedure](../../../GEO-INFER-TEST/docs/gnn_space_time_2026_09.md#gnn-pair-pin-bump-procedure)
and [ACT interchange guide](../../../GEO-INFER-ACT/docs/gnn_interchange.md)
own exporter/consumer execution. Source identity checks do not execute
categorical, H3, Gaussian or factored models and do not establish numerical
equivalence.

## Run the offline identity check

Use three clean, separate local checkouts. The companion checkouts must be at
the exact released commits and contain their actual annotated tag objects;
the GEO checkout must contain the committed manifest and current pin.
No scientific dependencies, environment synchronization or native proof build
is needed for this stdlib-only command on Python 3.11 or newer:

```bash
PYTHONPATH=GEO-INFER-TEST/src python -B GEO-INFER-TEST/validate_companion_releases.py \
  --geo-root /absolute/path/to/GEO_INFER \
  --gnn-root /absolute/path/to/Generalized_Notation_Notation \
  --fep-root /absolute/path/to/fep_formal \
  --expected-geo-revision FULL_ACTUAL_GEO_COMMIT
```

The result binds exact HEAD/tree/tag identities, project and citation versions,
all three pin files, manifest bytes, GEO baseline ancestry and unchanged
before/after inputs. A dirty checkout, lightweight or moved companion tag,
wrong version, disconnected GEO history or inconsistent pin fails the check.
The local check trusts the reviewed manifest and supplied Git objects; it does
not authenticate a remote origin or establish publication acceptance.

The standalone `companion-releases.yml` workflow checks out the actual GEO
event SHA and the two literal companion SHAs. It retains the resulting JSON
disposition. This complements the scientific two-environment workflow; it
does not replace it. Its remote checkouts provide the hosted source context
that is absent from a purely local invocation.

## Formal and historical evidence

FEP_Formal 2.1.0's catalogue contains 168 topics (`fep-001` through `fep-168`)
in 22 declared families. Topic statements, assumptions, semantic maturity and
native compilation receipts retain the scope of their actual FEP source.
The [notation bridge](../../../GEO-INFER-ACT/docs/fep_lean_notation_bridge.md)
is a correspondence guide; it does not prove GEO code or extend an instance
proof to later GEO source.

The retained Cascadia reference fixture provenance names historical GEO
`2e20fe2e86d19c1f5dfb6184ae88b5f224731a9c`. Its slice-local native receipt
uses Lean 4.34.1. These source and toolchain identities remain historical even
when the files are included in FEP_Formal 2.1.0. New source, variants or
sampled trajectories cannot inherit that receipt. See the existing
[Cascadia guide](../../../GEO-INFER-ACT/docs/cascadia_river_homing.md) for the
finite-model and implementation boundaries.

## Current GEO release evidence

Follow [0.4.0 readiness](../releases/0.4.0_readiness.md) and
[research artifact contracts](research_artifacts.md). Freeze source, refresh
generator-owned config in a separate commit, then collect current local and
hosted evidence. The current wheel-operation registry defines 26 independent
profiles; derive the required roster from that registry rather than reusing a
historical receipt's 20 or 24 profiles. Keep hardware, configured-advisor and
historical-cause limitations explicit.
