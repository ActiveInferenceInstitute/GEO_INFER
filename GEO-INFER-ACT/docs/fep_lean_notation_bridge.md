# fep_lean Notation Bridge

## Introduction

This page maps the Active Inference constructs implemented in GEO-INFER-ACT
onto the corresponding topics of the `fep_lean` catalogue, a separate
repository of 155 Lean 4 theorem-proxies spanning FEP, Active Inference,
Bayesian Mechanics, Information Geometry and Thermodynamics
(`fep_lean` topics `fep-001` through `fep-155` in 20 families).

The mapping is **documentation-level**: it states correspondence of symbols
and constructs between two independent artifacts. It does not state, and
must never be read as stating, that any GEO-INFER implementation is
verified or proved by `fep_lean`, or that any `fep_lean` proof is
validated by GEO-INFER numerical behavior.

## Evidence boundary

The two repositories are distinct evidence planes, mirroring the evidence
firewall of the `fep_lean` bridge contract at
`fep_lean/docs/design/gnn-bridge/bridge-contract.md` (referenced by prose
path only; cross-repository references are never Markdown links):

- **`fep_lean` is the formalization surface.** Native Lean compilation
  (`fep-lean verify`) establishes that the named theorem body compiles
  without warnings; the catalogue's own semantic-review layer
  (`fep_lean/config/theorem_maturity.yaml`) bounds how far each theorem
  proxy reaches toward its topic label. Neither establishes numerical
  behavior of any implementation.
- **GEO-INFER-ACT is the numerical implementation surface.** Its unit and
  contract tests establish executable behavior of the implemented
  estimators. Neither establishes mathematical proof.
- The mapping below is correspondence of constructs. A correspondence
  row is exactly as strong as its weakest side, and it never promotes a
  Lean proof claim into this module's documentation or a simulation
  statistic into a proved property.

The canonical machine-readable mapping artifact lives in the `fep_lean`
checkout at `fep_lean/specs/geo-infer-notation-bridge/data/notation-map.yaml`
(prose path; it is owned and versioned by the `fep_lean` slice). This page
is the human-readable GEO-INFER-side view of that mapping's machine-checked
core subset: the YAML carries the rows that have landed in the `fep_lean`
slice so far and is not exhaustive of the tables below, which cover the
implemented constructs this module actually uses.

## Notation correspondence

The symbols of [Mathematical Framework](./mathematical_framework.md)
correspond to `fep_lean` constructs as follows:

| ACT symbol | ACT reading | `fep_lean` counterpart |
| --- | --- | --- |
| `q(·)` | Approximate posterior | Approximate posterior law of the variational bound topics (`fep-001`, `fep-002`); posterior kernel reconstruction (`fep-017`) (canonical machine-checked row: fep-090) |
| `p(·)` | Generative model | Generative measures/kernels over the finite and Gaussian carrier families |
| `F` | Variational free energy, `F = complexity − accuracy` | Variational free energy as surprisal plus KL (`fep-002`), the variational upper bound (`fep-001`), surprise/self-information (`fep-011`), KL laws (`fep-014`) |
| `G` | Expected free energy | EFE as pragmatic cost minus epistemic value in `fep-021`; discounted pragmatic cost input (`fep-003`); expected information gain via measure KL (`fep-041`) |
| `γ` | Precision / inverse temperature in `p(π) = σ(−γ·G(π))` | Boltzmann–Gibbs inverse-temperature weighting (`fep-031`); entropy-regularized policy cost (`fep-012`) |
| `π` | Policy (action sequence) | Finite policy set with deterministic argmin layer (`fep-008`) and stochastic softmax layer (`fep-028`) |

## Implemented-construct mapping

| GEO-INFER-ACT construct | Implementation | `fep_lean` formal analogue | FEP literature |
| --- | --- | --- | --- |
| Categorical VFE, `F = complexity − accuracy`, returning `FreeEnergyBreakdown` | [`core/free_energy.py`](../src/geo_infer_act/core/free_energy.py) `FreeEnergyCalculator.compute_categorical_free_energy()` | `fep-001` (variational free energy bound), `fep-002` (variational evidence bound via KL divergence), `fep-011` (surprise and self-information), `fep-014` (KL nonnegativity, identity, chain rule) | Friston 2010; Parr, Pezzulo & Friston 2022 (ch. 4) |
| Gaussian VFE with precision matrices | [`core/free_energy.py`](../src/geo_infer_act/core/free_energy.py) `FreeEnergyCalculator.compute_gaussian_free_energy()` | `fep-002` (evidence bound), `fep-014` (KL laws), with `fep-016` (exact scalar quadratic Laplace kernel) as the finite Gaussian analogue | Parr, Pezzulo & Friston 2022 (ch. 4); Friston 2010 |
| EFE with epistemic/pragmatic (risk/ambiguity) decomposition | [`core/free_energy.py`](../src/geo_infer_act/core/free_energy.py) `FreeEnergyCalculator.compute_expected_free_energy()` | `fep-021` (EFE epistemic–pragmatic balance), `fep-003` (discounted pragmatic cost), `fep-041` (expected information gain via measure KL) | Parr, Pezzulo & Friston 2022 (ch. 5); Da Costa et al. 2020 |
| Policy-level EFE and softmax/deterministic selection, `p(π) = σ(−γ·G(π))` | [`core/policy_selection.py`](../src/geo_infer_act/core/policy_selection.py) `PolicySelector` | `fep-028` (support-aware finite softmax policy), `fep-008` (finite policy objective minimizer), `fep-012` (entropy-regularized policy cost), `fep-031` (finite Boltzmann–Gibbs weights) | Friston, FitzGerald, Rigoli, Schwartenbeck & Pezzulo 2017; Parr, Pezzulo & Friston 2022 (ch. 5) |
| Markov blanket four-block partition with conditional-independence check | [`core/generative_model.py`](../src/geo_infer_act/core/generative_model.py) `MarkovBlanket` | `fep-005` (four-block state partition) and `fep-009` (conditional independence laws) as the exact finite analogues; `fep-017` (posterior kernel reconstruction) for the belief-update side. The correspondence is of the partition and conditional-independence constructs only: `fep-005` deliberately does not identify any block as a learned Markov blanket. Canonical machine-checked row for this symbol: `fep-079`. The canonical yaml additionally anchors `HierarchicalLevel` as `fep-081`, which has no page-level row yet (page-level correspondences pending yaml growth). | Friston 2010; Parr, Pezzulo & Friston 2022 (ch. 3) |
| Precision as inverse covariance / inverse temperature | [`core/free_energy.py`](../src/geo_infer_act/core/free_energy.py) Gaussian precision arguments; `PolicySelector.temperature` | `fep-031` (finite Boltzmann–Gibbs weights: positive exponential weights with an attained Gibbs optimizer), `fep-012` (temperature in the entropy-regularized objective) | Friston 2010 |

## Explicit non-claims

1. No GEO-INFER-ACT implementation is verified, proved or certified by any
   `fep_lean` topic.
2. No `fep_lean` theorem is validated, corroborated or exercised by any
   GEO-INFER-ACT test, simulation or trace.
3. Topic ids above name correspondence of constructs at the granularity of
   the `fep_lean` catalogue's own semantic-review boundaries; each topic's
   assumption scope (see `fep_lean/config/topics.yaml`) bounds the
   correspondence, and several rows deliberately cover only a fragment of
   the ACT behavior (for example, the policy-selection row covers argmin
   and softmax layers, not exploration bonuses or habit priors).
4. Numerical agreement or disagreement between a GEO-INFER trace and a
   `fep_lean` witness is a finding to be reported through the bridge
   contract's certificate mechanism, never averaged away and never
   reclassified as proof or validation on either side.

## Further Reading

- [Mathematical Framework](./mathematical_framework.md) — the ACT-side notation table this page maps from
- [Free Energy Principle](./free_energy_principle.md) — conceptual background
- [GNN and GEO-INFER interchange](./gnn_interchange.md) — the executable artifact-level collaboration with GNN, distinct from this notation mapping
- [References](./references.md) — full citations for the literature above
- `fep_lean/specs/geo-infer-notation-bridge/data/notation-map.yaml` — canonical mapping artifact (in the `fep_lean` checkout, prose path)
- `fep_lean/docs/design/gnn-bridge/bridge-contract.md` — bridge contract including the evidence firewall (in the `fep_lean` checkout, prose path)
