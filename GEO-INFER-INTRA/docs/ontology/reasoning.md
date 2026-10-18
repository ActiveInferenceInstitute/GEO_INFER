# Ontology Reasoning Concepts

Ontology reasoning derives consequences from declared axioms and asserted facts.
This is conceptual guidance for applications that select a reasoner; INTRA has
no bundled OWL reasoner, rule service, or automatic knowledge-base integration.
The numerical inference APIs in ACT, BAYES, and SPM have their own model and
data contracts and are not interchangeable with logical entailment.

## Entailment, validation, and uncertainty

| Question | Required mechanism |
| --- | --- |
| Does a statement follow from these axioms? | A declared logic and a reasoner for that logic |
| Does this supplied record satisfy a constraint? | The owning data validator or explicit graph shapes |
| How probable is a latent state given observations? | A probabilistic model with defined likelihood and prior |
| Is the conclusion useful or empirically accurate? | Domain evidence and an appropriate evaluation design |

The [OWL 2 Primer](https://www.w3.org/TR/owl2-primer/) explains that missing
information under its open-world assumption does not imply a negative fact.
An inconsistent set of assertions also requires explicit handling; a successful
Python import does not prove logical consistency.

## Executable illustration of a limited rule

This complete Python example demonstrates one application-defined subclass
closure. It is not an OWL implementation: it does not handle property axioms,
negation, identity, datatypes, or consistency checking.

```python
parents = {
    "Street": {"TransportFeature"},
    "TransportFeature": {"SpatialFeature"},
}


def ancestors(name):
    found = set()
    pending = list(parents.get(name, ()))
    while pending:
        parent = pending.pop()
        if parent not in found:
            found.add(parent)
            pending.extend(parents.get(parent, ()))
    return found


assert ancestors("Street") == {"TransportFeature", "SpatialFeature"}
assert ancestors("UnknownFeature") == set()
```

The empty set means this local mapping supplies no ancestors. It does not prove
that an unknown feature has no classes in the world. Cyclic class definitions
also need a chosen equivalence/consistency policy before applying this pattern
to a real vocabulary.

## Geospatial interpretation

A named relation such as `contains` needs a defined geometry predicate,
coordinate system, geometry revision, and boundary policy. A graph edge is not
proof that the exact spatial predicate was evaluated. Likewise, an H3 cell
path does not prove road connectivity, accessibility, or travel time.
[GeoSPARQL 1.1](https://docs.ogc.org/is/22-047r1/22-047r1.html) specifies
geospatial RDF vocabulary and query behavior for implementations that support
it. Its existence does not make INTRA a GeoSPARQL endpoint.

## Evidence to retain

Record asserted and derived statements separately, plus vocabulary revision,
reasoner/version, selected logic/profile, input hashes, command or API request,
return status, warnings, and time/resource limits. Preserve incomplete,
interrupted, and inconsistent outcomes explicitly. A timeout is not a negative
answer, and a finite example does not establish complete reasoning over every
possible input.

See [modeling](ontology_modeling.md), [constraints](custom_rules.md), and
[actual tooling](toolkit.md) before choosing an execution surface.
