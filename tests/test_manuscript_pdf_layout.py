"""Layout tests that read the shipped PDF instead of the settings behind it.

The float-placement defect — a page carrying one figure, its caption, and the
folio, against 1,800-2,000 characters on an ordinary page — was previously
covered by asserting that four ``\\renewcommand`` lines were present in
``manuscript/preamble.md``.  That assertion passed while the defect was still
visible in the PDF: the parameters were relaxed enough for two of the three
figures and the third still cleared ``\\floatpagefraction``.  These tests read
the artifact.

The PDF is a build product under the ignored ``output/`` tree, so these tests
fail when it has not been rendered: the suite's no-skip policy treats an
unavailable render as a failure, not a pass.  Nothing here is a substitute for
the generator-side bound in ``_assert_leaves_room_for_text``; that one fails
before a figure is written, this one fails after a render that shipped one
anyway.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

PDF = Path("output/pdf/GEO-INFER_combined.pdf")
# A float page carries its figure's caption and the folio and nothing else; the
# page this test was written for held one word besides the caption.  A page
# that carries a figure and at least a line of body text besides is a text page
# with a figure on it, which is what the float parameters exist to produce.
# Measured on this manuscript: the float page held 1, the sparsest legitimate
# figure page holds 32 (a figure plus a short closing subsection), and an
# ordinary text page holds 250-350.  The floor sits in that gap.
MINIMUM_NON_CAPTION_WORDS = 15
# A float page is the defect when the page carries a figure, a short caption,
# and nothing else.  A full-page figure whose caption itself carries the
# section's argument — the graphical abstract, set as a deliberate full-page
# figure — is not that defect: its caption is the page's content.  The floor
# sits above the longest caption the short-caption defect class shipped (the
# 60-word inventory caption of the original float-page defect).
SUBSTANTIVE_CAPTION_WORDS = 80


def _words(text: str) -> list[str]:
    """Normalised word list: ligatures folded, punctuation and folios dropped."""
    folded = unicodedata.normalize("NFKD", text).casefold()
    return [word for word in re.findall(r"[a-z0-9]+", folded) if not word.isdigit()]


def _remaining(page_words: list[str], caption_words: list[str]) -> list[str]:
    """The page's words with one occurrence of each caption word removed."""
    budget = Counter(caption_words)
    kept: list[str] = []
    for word in page_words:
        if budget[word]:
            budget[word] -= 1
            continue
        kept.append(word)
    return kept


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        pytest.fail(f"{name} is not installed (skips become failures)")
    return path


@pytest.fixture(scope="module")
def rendered_pdf(repo_root: Path) -> Path:
    pdf = repo_root / PDF
    if not pdf.is_file():
        pytest.fail(f"{PDF.as_posix()} has not been rendered (skips become failures)")
    return pdf


def _page_text(pdf: Path, page: int) -> str:
    completed = subprocess.run(
        [_tool("pdftotext"), "-f", str(page), "-l", str(page), str(pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def _figure_pages(pdf: Path) -> set[int]:
    completed = subprocess.run(
        [_tool("pdfimages"), "-list", str(pdf)],
        capture_output=True,
        text=True,
        check=True,
    )
    pages: set[int] = set()
    for line in completed.stdout.splitlines()[2:]:
        fields = line.split()
        if fields and fields[0].isdigit():
            pages.add(int(fields[0]))
    return pages


class TestFloatPlacement:
    def test_no_figure_takes_a_page_to_itself(
        self, repo_root: Path, rendered_pdf: Path
    ) -> None:
        registry = json.loads(
            (repo_root / "output" / "figures" / "figure_registry.json").read_text(
                encoding="utf-8"
            )
        )
        captions = [_words(entry["caption"]) for entry in registry["figures"]]
        pages = _figure_pages(rendered_pdf)
        assert pages, "the rendered PDF embeds no figures"
        for page in sorted(pages):
            words = _words(_page_text(rendered_pdf, page))
            # Which caption is on the page is decided by overlap rather than by
            # order, so a moved figure does not silently pass the test.  The
            # count that matters is what is left after the best-matching
            # caption's words are removed once each.
            best = min(captions, key=lambda caption: len(_remaining(words, caption)))
            remaining = _remaining(words, best)
            assert not (
                len(remaining) < MINIMUM_NON_CAPTION_WORDS
                and len(best) < SUBSTANTIVE_CAPTION_WORDS
            ), (
                f"page {page} carries a figure, its caption, and "
                f"{len(remaining)} other words ({remaining}): it is a float "
                "page with a caption too short to be the page's content"
            )


class TestTextBlock:
    def test_no_word_is_set_past_the_right_margin(self, rendered_pdf: Path) -> None:
        # The text block's right edge, for the landed margin=1.5cm geometry:
        # \oddsidemargin -29.59087pt + 1in + \textwidth 528.93673pt (letterpaper
        # at 1.5cm margins; the same numbers the render script's header records).
        # A word box beyond it is text in the margin — an unbreakable verbatim
        # line or an unbreakable monospace span in a narrow column.  One point
        # of tolerance covers glyph bounding boxes that overhang their advance
        # width; re-verify both constants against the first render log taken at
        # the new margins (output/pdf/_combined_manuscript.log).
        right_edge = -29.59087 + 72.0 + 528.93673 + 1.0
        completed = subprocess.run(
            [_tool("pdftotext"), "-bbox", str(rendered_pdf), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        page = 0
        offenders: list[str] = []
        for line in completed.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("<page"):
                page += 1
            elif 'xMax="' in stripped:
                value = float(stripped.split('xMax="')[1].split('"')[0])
                if value > right_edge:
                    offenders.append(f"page {page}: {stripped} at {value:.1f}pt")
        assert not offenders, "\n".join(offenders)


class TestBoxWarnings:
    """TeX's own report that content did not fit the measure it was given.

    Six ``Overfull \\hbox`` lines were once emitted by Table 3's Command
    column, where a long unbreakable command ran past its cell.  Nothing pins
    that: the shared template's LaTeX gate is fatal on ``! `` errors and
    ``Missing character`` and deliberately leaves box warnings advisory, and
    the right-margin test above cannot see a box that overflows a table cell
    without leaving the text block.  This reads the final pass's log, which is
    the only place the overflow is reported at all.
    """

    def test_the_final_pass_reports_no_overfull_hbox(self, repo_root: Path) -> None:
        log = repo_root / "output" / "pdf" / "_combined_manuscript.log"
        if not log.is_file():
            pytest.fail(
                "output/pdf/_combined_manuscript.log has not been rendered"
                " (skips become failures)"
            )
        text = log.read_text(encoding="utf-8", errors="replace")
        offenders = [
            line for line in text.splitlines() if line.startswith("Overfull \\hbox")
        ]
        assert not offenders, "\n".join(offenders)


# Physical page 22 of 27 once held the single word "section." plus the folio:
# the last line of a paragraph stranded by a section break, eight characters
# once the folio is discounted.  The sparsest legitimate page in this
# manuscript is a section tail before the template's inter-section \newpage,
# and those run into the hundreds of characters.  The floor sits well below
# them and well above a runt line, so it names the defect without pinning the
# layout.
MINIMUM_PAGE_CHARACTERS = 200


def _page_characters(pdf: Path, page: int) -> int:
    """Non-whitespace characters on a page, folio excluded."""
    text = _page_text(pdf, page)
    return len(re.sub(r"\s+", "", re.sub(r"^\s*\d+\s*$", "", text, flags=re.M)))


class TestNoStrandedLines:
    """A page may not be given over to the tail of a paragraph."""

    def test_no_page_is_nearly_empty(self, rendered_pdf: Path) -> None:
        completed = subprocess.run(
            [_tool("pdfinfo"), str(rendered_pdf)],
            capture_output=True,
            text=True,
            check=True,
        )
        pages = next(
            int(line.split(":", 1)[1])
            for line in completed.stdout.splitlines()
            if line.startswith("Pages:")
        )
        figure_pages = _figure_pages(rendered_pdf)
        sparse = [
            (page, _page_characters(rendered_pdf, page))
            # Page 1 is the title page, which is sparse by design; a page
            # carrying a figure is measured by TestFloatPlacement instead.
            for page in range(2, pages + 1)
            if page not in figure_pages
            and _page_characters(rendered_pdf, page) < MINIMUM_PAGE_CHARACTERS
        ]
        assert not sparse, (
            f"pages carry fewer than {MINIMUM_PAGE_CHARACTERS} characters: {sparse}"
        )


class TestMonospaceSpansBreakOnlyWhenTheyMustBreak:
    """A `\\seqsplit` break is invisible, so it may not happen in prose.

    ``research_inv`` / ``entory.json``, ``geo_infer_sp`` / ``ace.nested`` and
    ``GEO-INFER-R`` / ``ISK`` all shipped split across a line boundary with
    no hyphen and nothing else to mark the join.  All three fit a 430pt
    measure several times over; they were split because the threshold was a
    sixth of the line rather than the line.
    """

    # Every literal the manuscript sets in prose, drawn from the tokens and
    # authored spans that were observed to break.  A fragment pair is a
    # violation only when the two halves are adjacent lines of body text.
    def test_no_manuscript_literal_is_split_across_lines(
        self, repo_root: Path, rendered_pdf: Path
    ) -> None:
        literals = sorted(
            {
                match
                for source in (repo_root / "output" / "manuscript").glob("*.md")
                for match in re.findall(
                    r"`([A-Za-z0-9][A-Za-z0-9_./-]{6,})`",
                    source.read_text(encoding="utf-8"),
                )
            }
        )
        assert literals, "the resolved manuscript contains no code spans"
        completed = subprocess.run(
            [_tool("pdftotext"), "-layout", str(rendered_pdf), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.rstrip() for line in completed.stdout.splitlines()]
        offenders: list[str] = []
        for index in range(len(lines) - 1):
            tail = re.search(r"[A-Za-z0-9_./-]+$", lines[index])
            head = re.match(r"[A-Za-z0-9_./-]+", lines[index + 1].lstrip())
            if tail is None or head is None:
                continue
            joined = tail.group(0) + head.group(0)
            # A break after an explicit hyphen prints the hyphen, so the
            # reader can see where the string was divided.  That is the one
            # division in an identifier that costs nothing, and TeX makes it
            # available without `\seqsplit`.  Every other position divides
            # the string with no character marking the join, which is the
            # defect this test exists for.
            if tail.group(0).endswith("-"):
                continue
            for literal in literals:
                if joined == literal and tail.group(0) != literal:
                    offenders.append(
                        f"{literal!r} split as {tail.group(0)!r} / {head.group(0)!r}"
                    )
        assert not offenders, "\n".join(sorted(set(offenders)))


class TestPreambleDoesNotLeakIntoTheBody:
    """A preamble macro name in the typeset text means the preamble misparsed.

    ``\\language=\\l@nohyphenation`` was injected into a context where ``@`` is
    not a letter, so it parsed as the command ``\\l`` followed by the text
    ``@nohyphenation`` and that string was typeset in front of every monospace
    span in the build, 298 occurrences in the extracted text.  The render
    exited 0: there was no ``!`` error and no missing character, only wrong
    output, so the template's fail-closed LaTeX gate had nothing to fail on.
    """

    def test_no_preamble_token_reaches_the_page(
        self, repo_root: Path, rendered_pdf: Path
    ) -> None:
        preamble = (repo_root / "manuscript" / "preamble.md").read_text(
            encoding="utf-8"
        )
        names = sorted(
            {
                name
                for name in re.findall(r"\\([A-Za-z@]{4,})", preamble)
                # A control-sequence name that is also an ordinary English
                # word would flag the prose that uses it.
                if not name.isalpha() or name.lower() != name
            }
            | {"nohyphenation", "makeatletter", "makeatother"}
        )
        assert names, "the preamble defines or calls no control sequences"
        text = subprocess.run(
            [_tool("pdftotext"), str(rendered_pdf), "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        found = sorted({name for name in names if name in text})
        assert not found, (
            f"preamble control-sequence names appear in the typeset body: {found}"
        )
