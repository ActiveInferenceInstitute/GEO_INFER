# GEO-INFER-INTRA/scripts/markdown_to_pdf

Markdown To Pdf workspace within `GEO-INFER-INTRA`.

## Contents

- `markdown_to_pdf.py`
- `README_markdown_to_pdf.md`
- `example-config.json`
- `md2pdf.sh`
- `package.json`

## Public Interface

- `markdown_to_pdf.py:MarkdownToPDFConverter` (class)
- `markdown_to_pdf.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-INTRA`
- Package: `geo_infer_intra`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-intra`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA`

## Dependencies

- `h3>=4.5.0,<5`
- `jsonschema>=4.0.0`
- `pillow>=10.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA
```


## Markdown to PDF Tool

- Converts Markdown documentation to PDF with the bundled `md2pdf.sh` wrapper and
  `markdown_to_pdf.py`; `README_markdown_to_pdf.md` documents configuration and
  usage.
- Node.js is a manual prerequisite; the script installs its remaining npm
  dependencies on first run.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
