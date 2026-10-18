# Ontology Modeling Guide

This page describes vocabulary design for applications using GEO-INFER. It is a
conceptual guide: INTRA ships documentation previews and configuration utilities,
not an ontology manager, RDF store, or graphical ontology editor. Use the
[tooling guide](toolkit.md) to distinguish installed interfaces from external
semantic-web tools.

## Define the question and the evidence

Start with questions the application must answer. For example: which feature
was observed, which geometry version locates it, and what spatial and temporal
support does a measurement represent? A name shared by two modules does not
prove that their concepts or measurement supports are interchangeable.

Record each concept's stable identifier, definition, scope, examples,
counterexamples, owner, and revision. Distinguish a feature from its geometry,
a measurement from the feature it describes, and an observation from a model
estimate. Keep units, CRS, timestamp convention, aggregation support, and
missing-value meaning at the data boundary.

## Representation choices

| Layer | Purpose | What it does not establish |
| --- | --- | --- |
| Controlled vocabulary | Stable terms and definitions | An executable inference engine |
| Application schema | Accepted record structure and values | Scientific validity of an observation |
| RDF graph | Identified resources and their relationships | Completeness of the represented world |
| OWL ontology | Explicit logical axioms | Probability estimates or empirical calibration |
| Validation shapes | Constraints on supplied graph data | Truth of every conforming statement |

[RDF 1.1](https://www.w3.org/TR/rdf11-concepts/) defines graphs using
subject–predicate–object triples. This representation is an optional external
interchange choice, not the native serialization of every GEO-INFER module.
The [OWL 2 Primer](https://www.w3.org/TR/owl2-primer/) explains ontology axioms
and the open-world treatment of incomplete information.

## A small conceptual vocabulary

The following JSON is an illustrative design record, not a shipped INTRA
schema, API request, or RDF document:

```json
{
  "vocabulary_id": "example:observation-vocabulary-v1",
  "concepts": [
    {"id": "example:Feature", "definition": "The entity being described."},
    {"id": "example:Observation", "definition": "A recorded measurement event."}
  ],
  "relations": [
    {"id": "example:observes", "source": "example:Observation", "target": "example:Feature"}
  ]
}
```

Application owners must define identifier resolution, schema validation, and
storage before treating such a record as a protocol. Keep asserted statements
separate from derived statements, and retain derivation provenance.

## Review and change control

1. Test representative records, missing information, conflicting units, and
   ambiguous labels with a domain reviewer.
2. Map each term to the actual consuming package's field or argument. Mark
   unmapped terms explicitly; do not invent a matching API.
3. Treat changed definitions, units, and cardinality as potential compatibility
   changes even if the serialized field name stays the same.
4. Preserve the vocabulary revision, source data, mapping version, and validation
   report with results that depend on them.

For geometric relation semantics, see [spatial concepts](spatial_concepts.md).
For executable checks and external shape languages, see [custom rules](custom_rules.md).
For data handoffs, see [integration](integration.md).
