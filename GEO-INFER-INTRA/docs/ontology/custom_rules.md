# Custom Rules and Data Constraints

Rules belong to the application or to an explicitly selected external rule
engine. GEO-INFER-INTRA does not ship the previously described RuleManager,
SWRL executor, or rule-registration CLI. This guide explains how to define and
test a rule without implying that such a service is installed.

## Separate three contracts

- A **validation constraint** accepts or rejects supplied data, such as a
  finite coordinate range.
- A **derivation rule** adds a statement under declared assumptions, such as
  assigning an application category from measured attributes.
- A **decision rule** chooses an action and may have operational side effects.

Do not report a validation success as an inferred scientific fact or execute a
decision as a side effect of checking data. Record the rule identifier, version,
inputs, assumptions, outcome, and any uncertainty policy.

## Executable application-level constraint

This standalone Python example validates an explicit WGS84 latitude/longitude
pair. It is example application code, not an INTRA API or a replacement for
an owning module's validation:

```python
import math


def validate_latlng(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Expected a latitude/longitude pair")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError("Coordinates must be real numeric values")
    latitude, longitude = value
    if not all(math.isfinite(x) for x in value):
        raise ValueError("Coordinates must be finite")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Coordinates outside WGS84 ranges")
    return latitude, longitude


assert validate_latlng([37.7749, -122.4194]) == (37.7749, -122.4194)
for invalid in ([True, 0], [float("nan"), 0], [91, 0], [0]):
    try:
        validate_latlng(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Accepted invalid coordinate pair: {invalid!r}")
```

A coordinate range check does not establish CRS correctness, observation quality,
polygon validity, or suitability for a geodesic calculation. GeoJSON uses
longitude/latitude order, while H3 indexing uses latitude/longitude.

## External graph constraints

[SHACL](https://www.w3.org/TR/shacl/) defines validation of an RDF data graph
against a shapes graph and describes validation reports. Using it requires an
explicit implementation, shapes, dependencies, and execution path. A successful
SHACL report is bounded by those shapes and the supplied graph. It does not
show that an external service was contacted or that measured values are true.

Keep graph constraints and Python data contracts aligned through explicit field
mappings and regression examples. Pin any external validator and record its
actual result; do not label an unexecuted shape as passed.

## Testing and failure handling

For every rule, test a valid case, a boundary case, missing or unknown data,
and a counterexample. Include nonfinite numeric values, incompatible units,
cycles, and contradictory statements when relevant. Define whether evaluation
raises an exception, returns a diagnostic, or preserves an unknown result.
Never coerce an unknown result to false silently.

Run the consuming package's tests through the
[testing guide](../developer_guide/testing_guide.md). Use
[reasoning](reasoning.md) for logical entailment and
[integration](integration.md) for cross-module mappings.
