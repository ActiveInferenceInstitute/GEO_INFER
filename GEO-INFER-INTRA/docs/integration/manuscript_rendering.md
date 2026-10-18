# Manuscript rendering with docxology/template

GEO-INFER measures its own source and produces the resolved manuscript.
[docxology/template](https://github.com/docxology/template) supplies an optional
PDF renderer with a title page, cover image, table of contents, bibliography
and multi-pass LaTeX compilation. It runs in an explicitly selected interpreter
with its own dependencies, preserving GEO-INFER's frozen scientific environment.

## Prepare current publication evidence

Follow [research artifact contracts](research_artifacts.md) to commit source,
refresh generator-owned config in a separate commit, and execute a clean
`--full-validation --publication` build. A render is not a substitute for
those verification commands. Recheck the artifact bundle with `--check`.

## Select a clean template checkout

Use a clean local checkout and retain its full Git commit. If the normal
template checkout contains active work, make a separate clone at the selected
commit; do not reset somebody else's changes. No clone, dependency installation,
network request or release operation is performed by GEO-INFER's renderer.

Set these paths to the actual checkout, installed template interpreter and
full commit before running the command:

```bash
GEO_TEMPLATE_ROOT=/absolute/path/to/clean/template
GEO_TEMPLATE_PYTHON=/absolute/path/to/template/environment/bin/python
GEO_TEMPLATE_COMMIT=full_40_character_git_commit
uv run --frozen python scripts/render_manuscript_pdf.py \
  --template-root "$GEO_TEMPLATE_ROOT" \
  --template-python "$GEO_TEMPLATE_PYTHON" \
  --template-revision "$GEO_TEMPLATE_COMMIT"
```

The command first hydrates GEO-INFER's resolved inputs using the GEO-INFER
interpreter, retains its explicit section order and all 45 module catalog
entries, then invokes the actual template `PDFRenderer.render_combined` in the
selected template interpreter. The default command without template options
uses the existing local Pandoc/XeLaTeX lane; that fallback does not consume
the template title-page cover slot or add a table of contents.

The template lane requires a clean GEO-INFER tree by default. For an explicit
development render, `--allow-dirty-template-render` retains the dirty state in
its receipt. A development render cannot establish clean-source publication
acceptance, even when the external renderer exits successfully.

## Cover and output checks

`manuscript/config.yaml` declares `paper.cover.image` and informative
`paper.cover.alt`. The cover uses the same generated `graphical_abstract.png`
as the Abstract section: four panels connect the measured module themes,
geospatial and inference components, verification loop and bounded claims.
The footer identifies the source commit, hash and version. Missing files,
escaping paths or empty alternative text are rejected before rendering.

Outputs are under `output/pdf/`. Every template attempt retains a separate
`template-attempts/<id>/receipt.json` and `render.log`. A completed worker also
records its measured `runtime.json`; early launch failures may lack that file.
The receipt binds template revision/tree/source hashes, prepared inputs,
worker bytes, process ownership/deadline evidence, original return code,
runtime, cover hash and final PDF hash. Failed attempts remain failed and
retain their diagnostics; they do not qualify a leftover PDF.

Review the actual final PDF: cover image and labels, every page, table
continuations, references, text boundaries, missing glyphs and LaTeX overflow.
Template version, interpreter, toolchain and pagination are part of that
render's evidence. A successful byte hash does not establish readability or
tagged-PDF accessibility certification.

## Python, CLI and MCP

Discover `geo_infer_test.manuscript_rendering:render_template_pdf` and
`template_source_identity` through `geo-infer catalog TEST` and `describe`.
The shared `AgentSession`/CLI/MCP surfaces expose the same calls and arguments.
Direct calls consume already prepared inputs; the repository render script
additionally runs hydration and section combination. See
[agent interfaces](agent_interfaces.md) for typed requests and failure handling.
