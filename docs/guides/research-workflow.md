# Reproducing the GEO-INFER paper

Start with the [manuscript source](../manuscript/README.md), then use the [research artifact contract](../../GEO-INFER-INTRA/docs/integration/research_artifacts.md) to identify what each verification group measures. The [module index](../index.md#modules-by-theme) points to each module's own documentation and implementation.

Run the verification, convergence, and rendering arms from a clean checkout with the workspace dependencies and TeX tools installed. The [gate guide](../../GEO-INFER-TEST/docs/gate.md) covers the required commands and receipts. Use [the renderer guide](../../GEO-INFER-INTRA/docs/integration/manuscript_rendering.md) when invoking the pinned `docxology/template` lane.

The authored sections, bibliography, preamble, and configuration are under `docs/manuscript/`. The generator produces resolved sections and figures under `output/`. A development render can reuse an existing verification record or explicitly report that no matching verification was run. To make evidence claims about a publication revision, generate and retain a fresh verification record for that revision.
