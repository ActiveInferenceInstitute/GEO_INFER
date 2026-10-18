# References {#sec:references}

References supply theoretical context, methodological guidance, or an external
interface definition. A resolved citation does not verify the repository's
implementation; software claims also require inspectable source and execution
evidence. Books and articles retain verified publication metadata and DOI
links where available. Mutable documentation cites a version and access date
without inventing a publication year.

Bibliography lives in `manuscript/references.bib` and is read by Pandoc during
PDF render; citations in the manuscript must resolve to real entries in that
file, and repository-derived counts and captions are not substitutes for
external scientific references. The generator audits that relationship on every
run: it parses `references.bib`, scans the resolved manuscript for citation
keys — excluding cross-reference labels such as figure and table references,
which share the `@` sigil — and fails the build when a citation has no entry.
An entry that no section cites is reported by name on every run, and can be
promoted to a build failure through the bibliography policy in
`manuscript/config.yaml`.
