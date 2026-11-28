"""The preamble file is executable input, not documentation with examples.

``infrastructure/rendering/_pdf_latex_helpers.extract_preamble`` concatenates
*every* ```latex block in ``docs/manuscript/preamble.md`` and injects the result
immediately before ``\\begin{document}``.  The file also carries the rationale
for each setting, so it is natural to paste an illustrative LaTeX document into
it — and a fence that says ``latex`` puts ``\\documentclass`` and
``\\begin{document}`` into the preamble of the real build.  These tests hold
the fenced blocks to what a preamble may contain.
"""

from __future__ import annotations

import re
from pathlib import Path

# The renderer's own pattern, copied so this test fails when the file drifts
# from what the renderer will actually extract.
LATEX_BLOCK = re.compile(r"```\s*latex\s*\n(.*?)\n\s*```", re.DOTALL)
# Commands that belong to a document, not to a preamble.
FORBIDDEN = (
    r"\documentclass",
    r"\begin{document}",
    r"\end{document}",
)


class TestPreambleBlocks:
    def test_the_file_ships_latex_the_renderer_will_extract(
        self, repo_root: Path
    ) -> None:
        blocks = LATEX_BLOCK.findall(
            (repo_root / "docs" / "manuscript" / "preamble.md").read_text(encoding="utf-8")
        )
        assert blocks, "no latex-fenced block would be extracted from preamble.md"

    def test_no_extracted_block_carries_a_whole_document(self, repo_root: Path) -> None:
        blocks = LATEX_BLOCK.findall(
            (repo_root / "docs" / "manuscript" / "preamble.md").read_text(encoding="utf-8")
        )
        offenders = [
            (command, block.strip().split("\n")[0])
            for block in blocks
            for command in FORBIDDEN
            if command in block
        ]
        assert not offenders, (
            "a latex-fenced block in preamble.md is injected before "
            "\\begin{document}; fence illustrative documents as `tex`. "
            f"{offenders}"
        )
