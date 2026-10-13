# Cascadia recorded-run diagnostics

The [workflow](cascadia_workflow.md) qualifies a named seed study before the
package projects its trajectories. [study_diagnostics.py](../examples/cascadia/study_diagnostics.py)
is a thin entry point into `geo_infer_act.runners.study_diagnostics` and
`study_visualization`; the same values appear in the interactive explorer.

## Timing and formulas

At step $t$, the process samples cue $o_t$ at recorded state $s_t$, the agent
updates to $q_t$, chooses an action, and the process transitions. Every score
below uses $(q_t,s_t)$ **before that transition**. The explorer derives elapsed
seconds from timezone-aware recorded timestamps and checks them against the
artifact's explicit `step_seconds`; its horizontal axis displays hours.

| Field | Definition | Interpretation |
| --- | --- | --- |
| `posterior_entropy_nats` | $-\sum_i q_i\log q_i$ | State uncertainty after this cue |
| `policy_entropy_nats` | $-\sum_a q(a)\log q(a)$ | Uncertainty across the one-step policies |
| `true_state_probability` | $q(s_t)$ | Mass assigned to the recorded generating state |
| `true_state_surprisal_nats` | $-\log q(s_t)$ | Null means infinite surprisal at zero mass |
| `brier_score` | $\sum_i(q_i-\mathbf{1}_{i=s_t})^2$ | Full-carrier squared error, without averaging over states |

Zero-probability entropy terms contribute zero. Probability vectors must be
finite, nonnegative, at most one, and normalized within **absolute** tolerance
`1e-8`; the tool never clips or renormalizes them. JSON outputs contain no NaN
or Infinity. A null surprisal is explicitly displayed as infinite, not as zero
or an omitted successful sample.

The Brier score lies between zero and two on an exact finite probability
simplex. The fep_formal active slice
`specs/cascadia-experiments/DiagnosticBounds.lean` states this theorem and its
hypotheses, plus a binary wrong-point-mass witness attaining two. The Python
boundary admits rounding tolerance; it is not a formal refinement proof that
floating-point evaluation implements that Lean definition exactly.

## Export and inspect

Use GitVault scratch and cache directories outside the source checkout, as
described in the workflow guide. Select a named, retained edition:

```bash
uv run python GEO-INFER-ACT/examples/cascadia/study_diagnostics.py \
  GEO-INFER-ACT/examples/outputs/STUDY output/study_diagnostics_new.json
uv run python GEO-INFER-ACT/examples/cascadia/study_explorer.py \
  GEO-INFER-ACT/examples/outputs/STUDY output/study_explorer_new.html
```

Output parents must exist; output files must be new. Qualification failure
creates neither report nor explorer. The portable diagnostic schema is
`geo-infer-act/cascadia-diagnostics/1`, with the study-manifest digest and
ordered condition/seed runs. Each run retains every observed step, including
steps in censored runs. `native_claim_ready` remains false.

The scores describe posterior beliefs in a simulated process using the
condition's own A/B tables. They do not estimate held-out prediction skill,
ecological calibration, or policy performance under a shared fixed process.
Inspect every seed and condition before selecting a teaching trajectory.

## Verification

```bash
uv run pytest GEO-INFER-ACT/tests/integration/test_cascadia_diagnostics.py \
  GEO-INFER-ACT/tests/integration/test_cascadia_explorer.py -q --no-cov
```

Tests check independent two-state values, zero true-state mass, malformed
vectors, a real archived first posterior, observation-time alignment,
timezone/clock failures, and create-only report output. The native theorem
receipt and numerical execution/diagnostic receipts remain separate evidence.
