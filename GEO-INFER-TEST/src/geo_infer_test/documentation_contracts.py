"""Lightweight checks shared by documentation and manuscript entrypoints."""

from __future__ import annotations

import ast
import io
import re
import shlex
import tokenize
from collections.abc import Iterable, Mapping
from pathlib import Path

_TEST_FILE_CENSUS = re.compile(r"(\d+)\s+test files", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_SINGLE_BACKTICK_MARKER = re.compile(r"^\s*`(?:[A-Za-z][\w+-]*)?\s*$")
_UMBRELLA_FROM_IMPORT = re.compile(r"\bfrom\s+geo_infer(?:\.[A-Za-z_]\w*)*\s+import\b")


def _is_python_fence(info: str) -> bool:
    """Recognize plain and Pandoc attribute Python code-block languages."""
    info = info.strip()
    if not info:
        return False
    if info.split(maxsplit=1)[0].lower() in {"python", "py", "python3"}:
        return True
    if info.startswith("{") and info.endswith("}"):
        try:
            attributes = shlex.split(info[1:-1])
        except ValueError:
            return False
        return any(
            value.lower() in {".python", ".py", ".python3"} for value in attributes
        )
    return False


def skill_python_syntax_errors(
    content: str, label: str, *, require_example: bool = False
) -> list[str]:
    """AST-check only explicit Python skill fences, without executing code.

    Plain ``python``/``py``/``python3`` and Pandoc Python class attributes are
    recognized, with matching backtick or tilde fence delimiters. This contract
    applies to operational skill examples, not partial authored-doc fragments.
    Each block must parse independently; malformed/unclosed Python examples
    report their original skill line. Only the opening fence's Markdown
    indentation is removed from its body.
    """
    errors: list[str] = []
    fence = ""
    python = False
    indent = 0
    start = 0
    found = False
    block: list[str] = []
    for number, line in enumerate(content.splitlines(), 1):
        marker = _FENCE.fullmatch(line)
        if not fence:
            if marker:
                fence = marker[1]
                python = _is_python_fence(marker[2])
                found = found or python
                indent = len(line) - len(line.lstrip(" "))
                start = number + 1
            continue
        if (
            marker
            and marker[1][0] == fence[0]
            and len(marker[1]) >= len(fence)
            and not marker[2].strip()
        ):
            if python:
                try:
                    ast.parse("\n".join(block), filename=label)
                except SyntaxError as error:
                    errors.append(
                        f"{label}:{start + (error.lineno or 1) - 1}: "
                        f"invalid Python example: {error.msg}"
                    )
            fence = ""
            block = []
        elif python:
            block.append(line[indent:] if line.startswith(" " * indent) else line)
    if fence and python:
        errors.append(f"{label}:{start - 1}: unclosed Python code fence")
    if require_example and not found:
        errors.append(f"{label} No Python code examples found")
    return errors


def _python_umbrella_import_lines(text: str) -> set[int]:
    """Inspect import tokens without parsing intentionally partial signatures."""
    lines: set[int] = set()
    from_statement = False
    from_root: tuple[str, int] | None = None
    expect_from_root = False
    import_statement = False
    expect_import_root = False
    # Indentation is irrelevant to imports. Normalizing it keeps incomplete
    # signature/implementation fragments tokenizable without requiring an AST.
    source = "\n".join(line.lstrip() for line in text.splitlines())
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            kind, value = token.type, token.string
            if kind == tokenize.NEWLINE or (kind == tokenize.OP and value == ";"):
                from_statement = import_statement = False
                expect_from_root = expect_import_root = False
                from_root = None
            elif kind == tokenize.NAME:
                if value == "from":
                    from_statement = expect_from_root = True
                    from_root = None
                elif value == "import":
                    if from_statement:
                        if from_root and from_root[0] == "geo_infer":
                            lines.add(from_root[1])
                    else:
                        import_statement = expect_import_root = True
                elif expect_from_root:
                    from_root = (value, token.start[0])
                    expect_from_root = False
                elif import_statement and expect_import_root:
                    if value == "geo_infer":
                        lines.add(token.start[0])
                    expect_import_root = False
            elif kind == tokenize.OP:
                if expect_from_root and value == ".":
                    # A relative import does not name the nonexistent root.
                    expect_from_root = False
                elif import_statement and value == ",":
                    expect_import_root = True
    except (tokenize.TokenError, IndentationError):
        # A partial fragment may end inside a signature or expression; import
        # tokens encountered before that point still have their real identity.
        pass
    return lines


def documentation_snippet_errors(
    root: Path, pages: Iterable[Path] | None = None
) -> list[str]:
    """Reject malformed fences and nonexistent ``geo_infer`` root imports.

    By default every module ``docs/**/*.md`` page is checked, including authored
    guides and generated signposts; there are no historical-page exemptions.
    Python and unlabeled fenced fragments are tokenized, not AST-parsed or
    executed. Non-Python fenced payload and quoted/commented Python text remain
    literal data. Imports in unfenced prose are also checked because some older
    guides collapsed executable examples onto one physical line. This narrow
    check does not certify the execution or completeness of arbitrary snippets.
    """
    if pages is None:
        pages = sorted(
            page
            for doc_root in root.glob("GEO-INFER-*/docs")
            for page in doc_root.rglob("*.md")
        )
    errors: list[str] = []

    def check_fragment(
        fragment: list[str], relative: Path, start: int, *, unlabeled: bool = False
    ) -> None:
        positions = _python_umbrella_import_lines("\n".join(fragment))
        if unlabeled:
            # Some legacy pages put the language and commented example on a
            # single body line. That malformed layout is prose, not a valid
            # Python comment exemption for its documented import.
            positions.update(
                number
                for number, line in enumerate(fragment, 1)
                if line.lstrip().startswith("python ")
                and _UMBRELLA_FROM_IMPORT.search(line)
            )
        for number in sorted(positions):
            errors.append(
                f"{relative}:{start + number - 1}: "
                "nonexistent umbrella package geo_infer; use an owning geo_infer_* package"
            )

    for page in pages:
        relative = page.relative_to(root)
        fence = ""
        python_fragment = False
        unlabeled = False
        fragment: list[str] = []
        fragment_start = 0

        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            marker = _FENCE.fullmatch(line)
            if fence:
                if (
                    marker
                    and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence)
                    and not marker[2].strip()
                ):
                    if python_fragment:
                        check_fragment(
                            fragment, relative, fragment_start, unlabeled=unlabeled
                        )
                    fence = ""
                    fragment = []
                elif python_fragment:
                    fragment.append(line)
                continue
            if marker:
                fence = marker[1]
                info = marker[2].strip().split(maxsplit=1)
                unlabeled = not info
                python_fragment = unlabeled or _is_python_fence(marker[2])
                if python_fragment and len(info) > 1:
                    # Collapsed legacy examples may put import text in the
                    # info string; it must not hide a nonexistent namespace.
                    check_fragment([info[1]], relative, number)
                fragment_start = number + 1
                continue
            if _SINGLE_BACKTICK_MARKER.fullmatch(line):
                errors.append(
                    f"{relative}:{number}: malformed single-backtick code fence"
                )
            if "geo_infer" in line and "import" in line:
                # Outside a code fence, '#' is Markdown/prose rather than a
                # reliable Python comment boundary in collapsed old examples.
                if _UMBRELLA_FROM_IMPORT.search(line):
                    errors.append(
                        f"{relative}:{number}: nonexistent umbrella package geo_infer; "
                        "use an owning geo_infer_* package"
                    )
                else:
                    check_fragment([line.replace("#", " ")], relative, number)
        if fence and python_fragment:
            check_fragment(fragment, relative, fragment_start, unlabeled=unlabeled)
    return errors


def module_catalog_census_errors(
    root: Path, measured: Mapping[str, int] | None = None
) -> list[str]:
    """Reconcile literal catalog counts with each owning module's test tree.

    The manuscript's inventory counts ``test_*.py`` files under each source
    module's ``tests/`` directory. This is a file census, not executed test
    coverage; registered extra roots and root manuscript tests have separate
    execution receipts. Callers with an inventory may supply its measured
    counts. The early documentation gate measures this same population
    without importing the manuscript generator or application packages.
    """
    if measured is None:
        measured = {
            module.name: sum(
                path.is_file()
                and "__pycache__" not in path.parts
                and ".git" not in path.parts
                for path in (module / "tests").rglob("test_*.py")
            )
            for module in root.glob("GEO-INFER-*")
            if module.is_dir() and (module / "src").is_dir()
        }
    errors: list[str] = []
    for section in sorted((root / "manuscript" / "sections").glob("*.md")):
        stated = _TEST_FILE_CENSUS.findall(section.read_text(encoding="utf-8"))
        if not stated:
            continue
        relative = section.relative_to(root)
        name = f"GEO-INFER-{section.stem.upper()}"
        if name not in measured:
            errors.append(f"{relative} states a census for unmeasured module {name}")
        elif len({int(count) for count in stated}) > 1:
            errors.append(
                f"{relative} states conflicting test-file counts: {', '.join(stated)}"
            )
        elif int(stated[0]) != measured[name]:
            errors.append(
                f"{relative} states {stated[0]} test files for {name}, "
                f"but the measured count is {measured[name]}"
            )
    return errors
