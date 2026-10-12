# GEO-INFER for spatial and temporal materiality assessment

Environmental, social, and governance (ESG) materiality assessment connects an
organisation's activities, affected places and people, observed impacts, and
explicit prioritisation criteria. GEO-INFER can support its data processing and
analysis. It does not supply a universal assessment facade or certify an assessment
against a reporting standard. The examples below use synthetic observations and
illustrative scoring rules; an assessment owner must justify real indicators,
stakeholder participation, units, uncertainty, and decisions.

## 1. Establish organisational context

Record sites, operational boundaries, supply-chain locations, activity periods,
and relevant stakeholders. DATA owns ingestion, storage, metadata, and lineage;
SPACE supplies spatial operations and H3 state ordering; TIME supplies aware
instants and temporal analysis. ORG, CIV, PEP, and COMMS contain specialised
organisational or engagement components: choose a concrete exported component and
its documented inputs rather than assuming a shared stakeholder-analysis facade.

Keep original coordinates and a CRS alongside derived H3 cells. Use a consistent
resolution for a state space, and preserve site identities independently of cell
IDs: several sites can occupy one cell. Record units, measurement method,
observation time, source, and whether an impact is actual, potential, or inferred.
Do not aggregate different indicators merely because they share a location.

## 2. Build an impact inventory with provenance

An inventory may combine monitoring records, operational incidents, surveys,
public datasets, and document review. Each source has its own access rights and
quality limits. DATA's `DatasetMetadata`, `DataLineage`, and `TemporalExtent` can
record origin, transformations, and coverage; these fields do not validate source
truth or establish compliance. Document review and stakeholder interpretation
remain explicit assessment activities unless a concrete, evaluated model is used.

The example aligns a synthetic impact indicator to two neighbouring H3 cells and
three aware instants. One observation is absent and another is measured zero.
The exact axis, values, and temporal coverage are acceptance assertions.

```python
import h3
import numpy as np
import pandas as pd
from geo_infer_data.models.schemas import DataLineage, DatasetMetadata, TemporalExtent
from geo_infer_space import H3StateSpace, align_h3_observations

center = h3.latlng_to_cell(45.5231, -122.6765, 8)
neighbor = sorted(set(h3.grid_disk(center, 1)) - {center})[0]
space = H3StateSpace([neighbor, center])
axis = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
frame = pd.DataFrame([
    {"cell": center, "timestamp": axis[0], "value": 2.0},
    {"cell": center, "timestamp": axis[1], "value": 4.0},
    {"cell": center, "timestamp": axis[2], "value": 6.0},
    {"cell": neighbor, "timestamp": axis[0], "value": 0.0},
    {"cell": neighbor, "timestamp": axis[2], "value": 1.0},
])
metadata = DatasetMetadata(title="Synthetic impact indicator", keywords=["illustrative"],
    temporal=TemporalExtent(start=axis[0].to_pydatetime(), end=axis[-1].to_pydatetime()),
    lineage=DataLineage(source="synthetic document fixture", process="UTC/H3 alignment",
                        created_by="assessment example"))
aligned = align_h3_observations(frame, state_space=space, timestamps=axis, max_entries=6)
assert tuple(aligned.data.columns) == (neighbor, center)
assert aligned.data.loc[axis[0], neighbor] == 0.0
assert np.isnan(aligned.data.loc[axis[1], neighbor])
np.testing.assert_array_equal(aligned.data[center], [2.0, 4.0, 6.0])
assert metadata.temporal.end - metadata.temporal.start == pd.Timedelta(days=2)
```

For real storage and transport acceptance, extend this with DATA's concrete
connector and storage APIs, then verify source IDs, values, metadata, and timestamp
custody across the round-trip. See the
[cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md).

## 3. Assess significance and uncertainty

Define criteria before scoring: scale, affected extent, duration, reversibility,
and likelihood may be relevant, with distinct treatment for actual and potential
impacts. Keep the direction of each indicator explicit. A decline in one metric
can indicate improvement and a decline in another deterioration.

Use TIME's temporal analysis for a declared series and interpret its sample-based
slope in the context of observation intervals. SPACE's weighted interpolation
estimates values from declared inputs; its name does not imply kriging uncertainty.
Use a concrete BAYES model when likelihood and prior assumptions are justified,
recording centroid coordinate order and elapsed seconds for spatiotemporal models.
Domain modules such as BIO, HEALTH, AG, ECON, and RISK require their own data and
model assumptions; their presence does not establish causal attribution or
transfer numerical outputs into a common significance scale.

Missing observations should remain missing until an explicit policy is selected.
Report measured data separately from interpolated, modelled, and elicited data.
Sensitivity analyses should vary both scoring weights and uncertain observations.

## 4. Prioritise with an auditable decision rule

A weighted score is useful only when criteria, scales, weights, and ownership are
visible. This small example calculates a declared rule and its sensitivity to
stakeholder weighting. It makes no universal materiality threshold claim.

```python
import numpy as np

# Rows are two synthetic impacts; columns are normalised extent and severity.
criteria = np.array([[0.8, 0.2], [0.4, 0.9]])
weights = np.array([0.25, 0.75])
assert np.all((criteria >= 0) & (criteria <= 1))
assert np.all(weights >= 0) and np.isclose(weights.sum(), 1.0)
scores = criteria @ weights
np.testing.assert_allclose(scores, [0.35, 0.775])
assert int(np.argmax(scores)) == 1
alternative = criteria @ np.array([0.9, 0.1])
np.testing.assert_allclose(alternative, [0.74, 0.45])
assert int(np.argmax(alternative)) == 0
```

The ranking reversal shows why the assessment must retain criterion values and
weights, not only a final rank. Record stakeholder disagreements, exclusions,
threshold rationale, decision date, and the person or body owning the decision.
API and APP may expose results through their concrete implemented interfaces;
access control and deployed availability require separate operational acceptance.

## Reproducibility and continued monitoring

Keep the source manifest, schemas, CRS, UTC conversion choices, state order,
versioned model configuration, random seeds, dependency lock, and generated
artifacts together. Verify aggregation and interpolation against independently
calculated tiny fixtures before scaling. Bound H3/time allocation and partition
large datasets without changing join keys or missing-data semantics.

A rapid assessment can use a small documented indicator set. A wider assessment
adds domain-specific models only when data and validation support them. Continued
monitoring needs declared sampling intervals, late-arrival policy, finite service
deadlines, and ownership of alerts. These are deployment choices with evidence
requirements, rather than guaranteed consequences of importing modules.

From the repository root, execute this maintained example and the relevant
composition regressions:

```bash
uv run --no-sync python GEO-INFER-TEST/validate_doc_examples.py
uv run --no-sync python -m pytest GEO-INFER-TEST/tests/integration/test_space_time_composition_contract.py
```

Consult the [module catalogue](../modules/index.md) for current components and the
[0.4.0 migration guide](../releases/0.4.0_migration.md) for changed interfaces.
