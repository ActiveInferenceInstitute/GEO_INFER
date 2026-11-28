# LaTeX Preamble

This file contains project-local LaTeX additions consumed by the template
renderer, which injects it verbatim immediately before `\begin{document}` in
`output/pdf/_combined_manuscript.tex`. Keep it minimal until the manuscript
needs additional math, table, or figure support.

```latex
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{hyperref}
\usepackage[capitalise,noabbrev]{cleveref}
\usepackage{natbib}
% Shared line-breaking policy for both the local and template-backed renderers.
\tolerance=4000
\setlength{\emergencystretch}{8em}
% Four-digit subsection numbers need more than article's default 2.3em TOC box.
\makeatletter
\renewcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.8em}}
\makeatother
```

## Float placement

LaTeX's defaults reserve a whole page for any float taller than
`\topfraction` (0.7) of the text block, and let a float claim a page of its
own once it fills `\floatpagefraction` (0.5) of one. Every evidence figure
therefore landed on pages carrying a single caption and roughly 300
characters of text against 1,800-2,000 on an ordinary page. Raising
`\topfraction` lets a tall float sit at the top of a text page; raising
`\floatpagefraction` means LaTeX opens a float page only for a float that
nearly fills one.

These parameters alone were not enough for the module inventory: at one column
of 45 legible rows it came to 0.90 of the text block and still cleared the
raised threshold. The generator now lays that figure out in two panels and
refuses, in `_assert_leaves_room_for_text`, to write any figure whose float
would exceed `\floatpagefraction` — so the settings below and the drawn size
are held in agreement rather than one being tuned against the other.

```latex
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.85}
```

## Widow and orphan lines

TeX's default `\widowpenalty` and `\clubpenalty` are 150, which is cheap
enough that the page builder will strand a single line of a paragraph across
a page boundary to balance the page it came from. The template puts a
`\newpage` between every section file, so a stranded line does not merely sit
awkwardly at the top of the next page — it gets that whole page to itself.
Measured on this manuscript: physical page 22 of 27 carried the word
`section.` and the folio — eight characters once the folio is discounted,
against 1,800-3,200 on an ordinary page — because the last line of section
7.3's closing paragraph did not fit above it.

10000 is the value that forbids the split outright rather than pricing it.
Setting the scalar `\widowpenalty` alone was not enough here: the stranded
line belonged to a four-line paragraph, so forbidding a one-line remainder
simply moved two lines instead of one and left the page carrying 94
characters rather than 8, both counted without the folio. The eTeX plural forms take a penalty per remainder
length, which is what states the rule directly — at least three lines on each
side of a split, so a paragraph of five lines or fewer is moved whole rather
than divided at all, and a longer one is still divided where division does
not produce a runt. `\brokenpenalty` covers the same case for a remainder
whose first line is also hyphenated.

This is a global setting, not a fix aimed at one page: the page it was found
on is only where the default first became visible, and
`test_no_page_is_nearly_empty` reads the shipped PDF so the class stays
closed wherever it next appears.

```latex
\clubpenalties 3 10000 10000 150
\widowpenalties 3 10000 10000 150
\displaywidowpenalties 3 10000 10000 150
\brokenpenalty=10000
```

## Breakable monospace spans

Pandoc emits inline code as `\texttt{...}`, which cannot break. The renderer
rewrites long monospace spans to a breakable macro, but only when it can
decode Pandoc's serialised body: a span containing `--` reaches the `.tex` as
`-\/-`, whose backslash puts it outside that decoder's contract, and the span
stays unbreakable. Six of the seven default-tier commands carry a `--` flag,
and all six ran into the neighbouring Status column; the seventh, which has
no such flag, was rewritten by the renderer and broke cleanly.

`\seqsplit` adds break opportunities without inserting a character, so the
printed command still copies as one string — but for the same reason a break
it takes is invisible: the reader sees `research_inv` ending a line and
`entory.json` beginning the next with nothing to mark the join. That is
acceptable for a span that cannot fit its measure by any other means, and it
is a defect for one that can. The threshold decides which is which, and a
sixth of the line was far too low: at the full 430pt measure it is about
twelve monospace characters, so ordinary prose literals — `GEO-INFER-RISK`,
`research_inventory.json`, `geo_infer_space.nested` — were all re-typeset
character by character and all three shipped broken mid-token in the running
text.

The threshold is now the measure itself. A span wider than `\linewidth` fits
on no line at all and has to be given break opportunities somewhere; a span
narrower than it does not, and is set the way LaTeX would set it. `\linewidth`
is the *local* measure, so inside Table 3's Command column it is that
column's width and the long validator commands — several times the column —
are still split there, which is what the split was introduced for.

The fitting branch sets the span rather than shipping the measuring box, and
opens the one break in an identifier that costs the reader nothing: the
explicit hyphen, whose own character marks the join. A box is atomic, so
shipping it put a 70pt unbreakable `GEO-INFER-RISK` at the end of a nearly
full line on page 26 and produced a 6.9pt overfull box — an invisible break
traded for a word in the margin, the same defect wearing different clothes.

Setting the span is not enough on its own. TeX inserts the discretionary
after an explicit hyphen only when the current font declares a hyphen
character, and `\hyphenchar` is `-1` for the typewriter font while it is `45`
for the roman: `resource-lifecycle` breaks in prose and `GEO-INFER-RISK` does
not, measured with `\showthe\hyphenchar\font` in each. Declaring it for the
monospace font is what makes the break available, and it brings pattern
hyphenation with it — with the declaration alone and nothing else,
`documentation` set in a narrow measure came out as `documen-` / `tation`,
a hyphen inserted into a literal that does not contain one. `\language` set
to babel's `nohyphenation` inside the same group removes the patterns without
removing the explicit-hyphen break: measured at a 62pt measure,
`GEO-INFER-RISK` breaks after a hyphen it already has and `documentation`
does not break at all.

The renderer's own `\breaktt` is redefined through the same rule. It is
defined ahead of this preamble and splits unconditionally, which is what
broke `research_inventory.json` in prose — a span the renderer could decode,
so the `\texttt` threshold above never saw it. One rule now governs both.

Where a span genuinely has to be split, the split should still land where it
costs least. `\seqsplit` inserts its break opportunities at penalty zero,
which is no more expensive than the interword glue a command line already
carries and cheaper than the `\exhyphenpenalty` of 50 an explicit hyphen
carries, so a break inside a token was never the last option — Table 3's
Command column read `python -m compileall -q GEO-INFE` / `R-*/src`. Pricing
the character break at 200 puts it above both, so the breaks the material
already offers are taken first and the same column now reads
`python -m compileall -q GEO-` / `INFER-*/src`. Where the column leaves no
such break — `uv run python GEO-INFER-TEST/val` / `idate_repo_contracts.py`
— the character break is still there to take, which is the point of keeping
it rather than removing it.

Three tests hold the two failure modes apart, all of them reading the
artifact rather than the settings: `test_the_final_pass_reports_no_overfull_hbox`
and `test_no_word_is_set_past_the_right_margin` fail if a span was left
unbreakable when it needed a break, and
`test_no_manuscript_literal_is_split_across_lines` fails if one was broken
where no character marks the join.

The block is wrapped in `\makeatletter`. This file is injected verbatim into
the document preamble but after the point where `@` is a letter, so
`\language=\l@nohyphenation` parsed as the command `\l` followed by the text
`@nohyphenation`: the assignment silently took a different number and the
string `@nohyphenation` was typeset in front of every monospace span in the
build, 298 occurrences in the extracted text. The template's LaTeX gate did
not catch it — there was no `!` error and no missing character, only wrong
output — which is why
`test_no_preamble_token_reaches_the_page` reads the PDF for the names this
block defines.

```latex
\IfFileExists{seqsplit.sty}{\usepackage{seqsplit}}{\newcommand{\seqsplit}[1]{#1}}
\makeatletter
\newcount\GIsplitcost
\GIsplitcost=200
\def\seqinsert{%
  \ifmmode\allowbreak
  \else\penalty\GIsplitcost\hspace{0pt plus 0.02em}\fi}
\newsavebox{\GIttbox}
\newcount\GInohyph
\GInohyph=\@cclv
\AtBeginDocument{%
  \@ifundefined{l@nohyphenation}{}%
    {\GInohyph=\csname l@nohyphenation\endcsname}}
\protected\def\GIfitorsplit#1{%
  \begingroup\ttfamily
  \language=\GInohyph
  \hyphenchar\font=`\-\relax
  \sbox\GIttbox{#1}%
  \ifdim\wd\GIttbox>\linewidth\seqsplit{#1}\else#1\fi
  \endgroup}
\protected\def\texttt#1{\GIfitorsplit{#1}}
\AtBeginDocument{\protected\def\breaktt#1{\GIfitorsplit{#1}}}
\makeatother
```

## Multi-page tables

Splitting a `longtable` across a page prints one `ignored: Infinite glue
shrinkage found in box being split` line per split into the render log. It is
informational — TeX says it ignored the glue, the output is unaffected — and
it is intrinsic to `longtable`, not to anything this manuscript does. A
fourteen-line document reproduces it exactly. It is fenced as `tex` on
purpose: the renderer concatenates every latex-fenced block in this file into
the document preamble, so an illustrative whole document must never carry that
fence.

```tex
\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\begin{document}
\begin{longtable}[]{@{}ll@{}}
\caption{A long table.}\label{tbl:x}\tabularnewline
\toprule\noalign{}
Group & Status \\
\midrule\noalign{}
\endhead
\bottomrule\noalign{}
\endlastfoot
% 80 rows
\end{longtable}
\end{document}
```

The count therefore tracks how many of the manuscript's tables happen to cross
a page boundary. Do not shorten a table to lower it: the tables are evidence,
and the message costs the reader nothing.
