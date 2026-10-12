#!/usr/bin/env python3
"""Render the GEO-INFER manuscript PDF with the repository's own render path.

This is the in-repo render stage that ROOT-01 deferred to the receipts
variant: it hydrates the evidence bundle and resolved manuscript with the
generator, combines the published sections into
``output/pdf/_combined_manuscript.md``, injects the LaTeX preamble blocks the
renderer is documented to concatenate, and runs pandoc and XeLaTeX.  The
artifacts it leaves behind are exactly the ones the render-dependent root
tests read:

- ``output/pdf/GEO-INFER_combined.pdf``  the rendered manuscript;
- ``output/pdf/_combined_manuscript.md`` the combined document (the paths
  test reads it);
- ``output/pdf/_combined_manuscript.tex`` / ``.log``  the final LaTeX pass
  (the box-warning test reads the log);
- ``output/figures/`` and ``output/manuscript/``  hydrated by the generator.

Fail-closed contract: every external tool must be present, every published
section must exist exactly once and in the documented order, LaTeX errors
(``! `` lines) and ``Missing character`` reports in the final pass log fail
the render, and a missing PDF fails the render.  Box warnings stay advisory:
the layout tests own them.  The text block is set by the geometry package
with ``margin=1.5cm`` on letterpaper (textwidth about 528.93673pt, textheight
about 709.61173pt), matching ``metadata.geometry`` in
``manuscript/config.yaml``; the layout floors in
``tests/test_manuscript_pdf_layout.py`` were calibrated against the
previous 430.00462pt x 556.47656pt block and must be re-checked against
the first render at the new margins.

The generator is invoked with ``--allow-dirty``, matching the documented
render-shim behavior for non-publication builds: a dirty tree is stamped
and counted, never silently attributed.  Verification is deliberately not
requested here — the render reuses or carries forward whatever record the
tree already holds, and never empties one.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "manuscript" / "generate_research_artifacts.py"
RESOLVED_DIR = REPO_ROOT / "output" / "manuscript"
PDF_DIR = REPO_ROOT / "output" / "pdf"
REQUIRED_TOOLS = ("pandoc", "pandoc-crossref", "xelatex")

# The published section order from manuscript/SYNTAX.md.  Adding a section
# to the manuscript must extend this list deliberately: like the generator's
# module table, the render refuses to drop a published file silently.
SECTION_ORDER: tuple[str, ...] = (
    "00_abstract.md",
    "01_introduction.md",
    "02_system_context.md",
    "03_methods.md",
    "04_artifacts_and_evidence.md",
    "05_reproducibility.md",
    "06_limitations_and_next_steps.md",
    "S01_source_surface.md",
    "S02_module_catalog.md",
    "98_symbols_glossary.md",
    "99_references.md",
)
RESOLVED_SUPPORT_FILES: tuple[str, ...] = (
    "config.yaml",
    "preamble.md",
    "references.bib",
)

# The supplemental module catalog: ``S02_module_catalog.md`` is the authored
# introduction and index; the per-module entries are concatenated after it
# from ``manuscript/sections/*.md`` in alphabetical stem order at combine
# time. Catalog files are tracked prose without tokens, labels, or figures,
# so they render from the source tree directly and the generator's resolved
# surface stays top-level. A missing or empty sections directory fails the
# render rather than publishing a catalog with silently dropped modules.
MODULE_CATALOG_ENTRY = "S02_module_catalog.md"
MODULE_SECTIONS_DIR = REPO_ROOT / "manuscript" / "sections"

# The renderer rewrites the ``output/figures/`` prefix to ``../figures/``
# inside Pandoc image targets only; ``tests/test_manuscript_paths.py`` holds
# this rule loud and local at both ends.
IMAGE_TARGET = re.compile(r"!\[[^\]]*\]\(([^)]*)\)")
FIGURE_PREFIX = "output/figures/"
FIGURE_REWRITE = "../figures/"

# Only ```latex fences are concatenated into the preamble; the illustrative
# whole document in the preamble is fenced ```tex on purpose to stay out.
LATEX_BLOCK = re.compile(r"^```latex\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)

# Page geometry: letterpaper with 1.5cm margins, matching
# metadata.geometry in manuscript/config.yaml. On letterpaper this is a
# text block of about 528.93673pt x 709.61173pt (7.319in x 9.817in); the
# generator's TEXT_BLOCK_* constants describe the same block and size and
# bound every figure against it.
GEOMETRY = "\\geometry{letterpaper, margin=1.5cm}\n"
WRAP_SETUP = "\\usepackage{fvextra}\n\\fvset{breaklines}\n"
FLOW_PENALTIES = (
    "\\widowpenalties 3 10000 10000 10000\n"
    "\\clubpenalties 3 10000 10000 10000\n"
    "\\displaywidowpenalties 3 10000 10000 10000\n"
)
JUSTIFY_SETUP = "\\tolerance=4000\n\\setlength{\\emergencystretch}{8em}\n"
MAX_XELATEX_PASSES = 4

_ERROR_PREFIX = "! "
_MISSING_CHARACTER = "Missing character"
_RERUN_MARK = "Rerun to get"


class RenderError(RuntimeError):
    """A fail-closed render failure with its evidence."""


def _require_tools() -> None:
    missing = [name for name in REQUIRED_TOOLS if shutil.which(name) is None]
    if missing:
        raise RenderError(f"required render tools are not installed: {missing}")


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(list(command), cwd=cwd, text=True)
    if completed.returncode != 0:
        raise RenderError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )
    return completed


def _hydrate() -> None:
    """Generate variables, figures, and resolved manuscript copies."""
    _run([sys.executable, str(GENERATOR), "--allow-dirty"], cwd=REPO_ROOT)


def _check_resolved_manuscript() -> None:
    expected = {*SECTION_ORDER, *RESOLVED_SUPPORT_FILES}
    found = {
        path.name
        for path in RESOLVED_DIR.iterdir()
        if path.suffix in (".md", ".bib", ".yaml")
    }
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        raise RenderError(
            "resolved manuscript disagrees with the published section list "
            f"(extend SECTION_ORDER deliberately): missing={missing} "
            f"extra={extra}"
        )


def _rewrite_image_targets(text: str) -> str:
    """Move the figure prefix inside Pandoc image targets only.

    The matched image syntax is rebuilt verbatim except for the target, so
    the leading ``![`` and the caption survive exactly as authored.
    """

    def rewrite(match: re.Match[str]) -> str:
        whole = match.group(0)
        target = match.group(1)
        if not target.startswith(FIGURE_PREFIX):
            return whole
        moved = FIGURE_REWRITE + target[len(FIGURE_PREFIX) :]
        caption = whole[2 : whole.index("](")]
        return f"![{caption}]({moved})"

    return IMAGE_TARGET.sub(rewrite, text)


def _combine_sections() -> Path:
    """Write the combined document in the published section order."""
    parts: list[str] = []
    for name in SECTION_ORDER:
        text = (RESOLVED_DIR / name).read_text(encoding="utf-8")
        if name == MODULE_CATALOG_ENTRY:
            text = text.rstrip("\n") + "\n\n" + _module_catalog_sections()
        parts.append(text.rstrip("\n"))
    combined = "\n\n".join(parts) + "\n"
    combined = _rewrite_image_targets(combined)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    destination = PDF_DIR / "_combined_manuscript.md"
    destination.write_text(combined, encoding="utf-8")
    return destination


def _module_catalog_sections() -> str:
    """Read the per-module catalog entries in alphabetical stem order.

    The entries are tracked prose, combined from the source tree rather
    than from ``output/manuscript/``: they carry no tokens, so resolving
    them through the generator would be a no-op pass, and the resolved
    surface stays the generator's top-level publication set. An empty or
    missing directory raises: the catalog is a published section, and a
    render that dropped its entries silently would misrepresent the
    module set.
    """
    if not MODULE_SECTIONS_DIR.is_dir():
        raise RenderError(
            f"module sections directory is missing: {MODULE_SECTIONS_DIR}"
        )
    files = sorted(MODULE_SECTIONS_DIR.glob("*.md"))
    if not files:
        raise RenderError(
            f"module sections directory holds no .md files: {MODULE_SECTIONS_DIR}"
        )
    return "\n\n".join(path.read_text(encoding="utf-8").strip() for path in files)


def _preamble_header() -> Path:
    """Concatenate the preamble's latex blocks plus the pinned geometry.

    ``\\PassOptionsToPackage{numbers}{natbib}`` precedes the blocks: the
    authored preamble loads natbib, while the bibliography itself is
    produced by pandoc's citeproc as a plain ``thebibliography``.  In its
    default author-year mode natbib rejects that bibliography outright; in
    numbers mode the patched environment accepts it, and no citation
    command ever reaches natbib because citeproc has already rendered the
    citations as text.
    """
    preamble = (RESOLVED_DIR / "preamble.md").read_text(encoding="utf-8")
    blocks = LATEX_BLOCK.findall(preamble)
    if not blocks:
        raise RenderError("the resolved preamble defines no ```latex blocks")
    header = "\n".join(block.strip("\n") for block in blocks)
    header = (
        "\\PassOptionsToPackage{numbers}{natbib}\n"
        + header.rstrip("\n")
        + "\n\n"
        + GEOMETRY
        + WRAP_SETUP
        + JUSTIFY_SETUP
    )
    destination = PDF_DIR / "_preamble.tex"
    destination.write_text(header, encoding="utf-8")
    return destination


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _config_scalar(config: str, key: str) -> str:
    match = re.search(
        rf'^\s+(?:-\s+)?{re.escape(key)}:\s*"(.*?)"\s*(?:#.*)?$', config, re.MULTILINE
    )
    if match is None:
        raise RenderError(f"config.yaml does not set {key}")
    return match.group(1)


def _config_list(config: str, key: str) -> list[str]:
    match = re.search(
        rf"^{re.escape(key)}:\s*\n((?:\s+-\s+.*\n)+)", config, re.MULTILINE
    )
    if match is None:
        raise RenderError(f"config.yaml does not set {key}")
    return re.findall(r'-\s+"(.*)"', match.group(1))


def _metadata_file() -> Path:
    """Write the Pandoc metadata block from the resolved config.yaml.

    The document class and base font size come from ``metadata.documentclass``
    and ``metadata.fontsize`` in ``manuscript/config.yaml`` — the same values
    the template-lane renderer consumes — so the config's declared layout is
    honored by the PDF lane instead of silently overridden here.
    """
    config = (RESOLVED_DIR / "config.yaml").read_text(encoding="utf-8")
    title = _config_scalar(config, "title")
    subtitle = _config_scalar(config, "subtitle")
    date = _config_scalar(config, "date")
    author = _config_scalar(config, "name")
    affiliation = _config_scalar(config, "affiliation")
    keywords = _config_list(config, "keywords")
    documentclass = _config_scalar(config, "documentclass")
    fontsize = _config_scalar(config, "fontsize")
    lines = [
        f"title: {_yaml_string(title)}",
        f"subtitle: {_yaml_string(subtitle)}",
        f"author: [{_yaml_string(f'{author} ({affiliation})')}]",
        f"date: {_yaml_string(date)}",
        "lang: en",
        "papersize: letter",
        f"documentclass: {_yaml_string(documentclass)}",
        f"fontsize: {_yaml_string(fontsize)}",
        "indent: true",
        f"keywords: [{', '.join(_yaml_string(word) for word in keywords)}]",
    ]
    destination = PDF_DIR / "_metadata.yaml"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _run_pandoc(header: Path, metadata: Path) -> Path:
    """Run pandoc to the standalone LaTeX source."""
    _run(
        [
            "pandoc",
            "_combined_manuscript.md",
            "--from",
            "markdown",
            "--to",
            "latex",
            "--standalone",
            "--number-sections",
            "--filter",
            "pandoc-crossref",
            "--citeproc",
            "--bibliography",
            "../manuscript/references.bib",
            "--metadata-file",
            metadata.name,
            "--include-in-header",
            header.name,
            "--output",
            "_combined_manuscript.tex",
        ],
        cwd=PDF_DIR,
    )
    destination = PDF_DIR / "_combined_manuscript.tex"
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RenderError("pandoc produced no LaTeX source")
    return destination


def _run_xelatex() -> Path:
    """Run XeLaTeX to a stable final pass and return the final PDF path."""
    log_path = PDF_DIR / "_combined_manuscript.log"
    pdf_path = PDF_DIR / "_combined_manuscript.pdf"
    passes = 0
    for _ in range(MAX_XELATEX_PASSES):
        passes += 1
        _run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-file-line-error",
                "_combined_manuscript.tex",
            ],
            cwd=PDF_DIR,
        )
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if _RERUN_MARK not in log:
            break
    else:
        raise RenderError(
            f"XeLaTeX still asks for a rerun after {MAX_XELATEX_PASSES} passes"
        )
    log = log_path.read_text(encoding="utf-8", errors="replace")
    errors = [line for line in log.splitlines() if line.startswith(_ERROR_PREFIX)]
    missing = [line for line in log.splitlines() if _MISSING_CHARACTER in line]
    if errors or missing:
        raise RenderError(
            "the final XeLaTeX pass is not clean:\n" + "\n".join(errors + missing[:20])
        )
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RenderError("XeLaTeX produced no PDF")
    print(f"xelatex: stable after {passes} pass(es)")
    return pdf_path


def _publish(pdf_path: Path) -> Path:
    destination = PDF_DIR / "GEO-INFER_combined.pdf"
    shutil.copy2(pdf_path, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    del argv  # the render takes no options: one deterministic build
    _require_tools()
    _hydrate()
    _check_resolved_manuscript()
    combined = _combine_sections()
    header = _preamble_header()
    metadata = _metadata_file()
    _run_pandoc(header, metadata)
    pdf = _run_xelatex()
    published = _publish(pdf)
    print(
        f"rendered {published.relative_to(REPO_ROOT)} from "
        f"{len(SECTION_ORDER)} sections ({combined.relative_to(REPO_ROOT)})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RenderError as error:
        print(f"render failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
