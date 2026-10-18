# Ontology and Module Integration

A semantic mapping explains how one application's terms correspond to another
module's inputs and outputs. GEO-INFER does not supply an implicit ontology
merger, knowledge-base synchronization service, or automatically generated API.
Implement and test mappings in the package that owns the consuming behavior.

## Define the handoff

| Boundary detail | Example question |
| --- | --- |
| Identity | Is this a stable feature identifier or only a display label? |
| Geometry | Which geometry version and CRS define the location? |
| Units | Does a value represent metres, degrees, a count, or a rate? |
| Support | Is it a point observation, cell aggregate, interval, or model state? |
| Time | What timezone, interval convention, and missing-time policy apply? |
| Semantics | Is a statement asserted, derived, uncertain, or unknown? |
| Compatibility | Which vocabulary and API revisions were mapped and tested? |

Do not merge two concepts solely because their labels match. Preserve unmapped
or conflicting terms and require an explicit resolution. A conversion that
loses spatial support, uncertainty, or provenance can change the analysis even
when both records pass a structural schema check.

## Discover real interfaces

```bash
uv run --frozen geo-infer modules
uv run --frozen geo-infer catalog SPACE --query latlng_to_cell --limit 20
uv run --frozen geo-infer describe geo_infer_space:latlng_to_cell
uv run --frozen geo-infer schema
```

The shared adapter calls existing public Python APIs; it does not infer a
semantic mapping from an ontology. Runtime descriptions import the selected
package and require its dependencies. Catalog discovery alone establishes
neither backend availability nor a scientific qualification.

## Executable spatial handoff

This script explicitly maps a GeoJSON point's longitude/latitude order to H3's
latitude/longitude call. The application keeps its own feature identifier:

```python
from geo_infer_api import AgentSession

feature = {
    "id": "example-observation-1",
    "geometry": {"type": "Point", "coordinates": [-122.4194, 37.7749]},
}
longitude, latitude = feature["geometry"]["coordinates"]
with AgentSession() as session:
    result = session.execute({
        "op": "call",
        "target": "geo_infer_space:latlng_to_cell",
        "args": [latitude, longitude, 9],
    })
    assert result["ok"], result
    linked = {"feature_id": feature["id"], "h3_cell": result["result"]}
    assert isinstance(linked["h3_cell"], str)
```

This small example assumes its explicit input is a valid EPSG:4326 point. It
performs indexing, not vocabulary alignment, coordinate reprojection, or
knowledge-base insertion. Validate those additional boundaries in the consuming
application when they are required.

## External semantic systems

RDF/OWL stores, GeoSPARQL services, ontology editors, and mapping engines are
separately selected systems. Record dependencies, endpoint configuration,
credentials, authorization, query semantics, availability, and actual responses
for any integration. A repository example or installed skill grants no external
write permission. Keep deterministic local validation available when a remote
system is unavailable.

Retain mapping versions and both original and transformed data. Test round-trip
behavior only where the mapping is intended to be reversible; explicitly
record lossy transformations. See the
[shared execution contract](../integration/agent_interfaces.md),
[cross-module integration guide](../integration/index.md), and
[external systems guidance](../integration/external_systems.md).
