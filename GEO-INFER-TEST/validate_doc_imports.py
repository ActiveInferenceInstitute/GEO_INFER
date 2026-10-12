#!/usr/bin/env python3
"""Validate that ``from geo_infer_*`` imports in the documentation hub and
every module's ``docs/`` tree resolve.

Historical hub pages shipped code snippets importing symbols no module defines
(phantom top-level facades such as ``SpatialAnalyzer``/``DataManager``) or
wrong module paths (``geo_infer_bayes.core.gaussian_process`` when the real GP
class lives in ``geo_infer_bayes/__init__.py``). The link validator is blind
to that drift; this checker closes the gap.

Resolution is static (no package imports at check time): each import path is
mapped to its package source tree and each imported name must be either a
submodule (``x/Y.py`` or ``x/Y/``) or a top-level definition, assignment, or
``__all__`` re-export in the target module file. Exemption banners fail the
gate: maintained guidance must describe current imports and behavior.

Exit codes: 0 = clean, 1 = at least one unresolvable import.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DOC_GLOB = "GEO-INFER-*/docs"


BANNER_MARKER = "Illustrative example notice"

IMPORT_RE = re.compile(
    r"from\s+(?P<module>geo_infer_[a-z_0-9]+(?:\.[A-Za-z_0-9]+)*)\s+import\s+"
    r"(?P<names>.+?)(?=from\s+geo_infer_|$)"
)
ALL_RE = re.compile(r"__all__\s*(?::\s*list\[[^\]]*\]\s*)?=\s*(\[[^\]]*\])", re.DOTALL)
# Conditional exports: packages extend ``__all__`` for optional modules
# (e.g. ``if "information_theory" in _available_core: __all__.extend([...])``).
# Those names are genuine package-root exports when available, so collect them.
ALL_EXTEND_RE = re.compile(r"__all__\.extend\(\s*(\[[^\]]*\])\s*\)", re.DOTALL)


def package_dirs(repo_root: Path) -> dict[str, Path]:
    """Map importable package name -> src package directory."""
    mapping: dict[str, Path] = {}
    for module_dir in sorted(repo_root.glob("GEO-INFER-*")):
        for package_dir in sorted((module_dir / "src").glob("geo_infer_*")):
            mapping.setdefault(package_dir.name, package_dir)
    return mapping


def module_file(package_dir: Path, dotted: str) -> Path | None:
    """Resolve ``dotted`` (relative to ``package_dir``) to a module file."""
    target = package_dir
    for part in dotted.split("."):
        candidate_dir = target / part
        candidate_file = target / f"{part}.py"
        if candidate_dir.is_dir():
            target = candidate_dir
        elif candidate_file.is_file():
            target = candidate_file
        else:
            return None
    if target.is_dir():
        target = target / "__init__.py"
        if not target.is_file():
            return None
    return target


def _all_names(module_text: str) -> set[str]:
    names: set[str] = set()
    for match in ALL_RE.finditer(module_text):
        names |= set(re.findall(r"[\w.]+", match.group(1)))
    for match in ALL_EXTEND_RE.finditer(module_text):
        names |= set(re.findall(r"[\w.]+", match.group(1)))
    for line in module_text.splitlines():
        stripped = line.strip()
        if "import" not in stripped or not stripped.startswith(("from ", "import ")):
            continue
        for piece in stripped.split("import", 1)[1].split(","):
            piece = piece.strip()
            if " as " in piece:
                alias = piece.split(" as ", 1)[1].strip()
                if alias.isidentifier():
                    names.add(alias)
            else:
                head = piece.split(".")[-1].strip()
                if head.isidentifier():
                    names.add(head)
    return names


def name_resolves(
    package_dir: Path, dotted: str, name: str, module_text_cache: dict[Path, str]
) -> bool:
    """Check whether ``name`` is importable from module ``package_dir/dotted``."""
    target_file = module_file(package_dir, dotted)
    if target_file is None:
        return False
    if (target_file.parent / name).is_dir() or (
        target_file.parent / f"{name}.py"
    ).is_file():
        return True
    text = module_text_cache.get(target_file)
    if text is None:
        text = target_file.read_text(encoding="utf-8", errors="replace")
        module_text_cache[target_file] = text
    if name in _all_names(text):
        return True
    if re.search(
        rf"^\s*(?:def|class)\s+{name}\b|^\s*{name}\s*(?::[^=\n]+)?=|\bimport\b[^#\n]*\bas\s+{name}\b",
        text,
        re.M,
    ):
        return True
    return False


def extract_imports(path: Path) -> list[tuple[int, str, list[str]]]:
    """Yield (line_number, module, names) for every import statement in a page.

    Matches mid-line imports (many hub pages collapsed to single physical
    lines), joins parenthesized multi-line import lists, and skips fenced
    code blocks in non-Python languages.
    """
    records: list[tuple[int, str, list[str]]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_fence = False
    fence_lang = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.count("```") % 2 == 1:
            in_fence = not in_fence
            if in_fence:
                fence_lang = line.strip()[3:].strip()
            index += 1
            continue
        index += 1
        if in_fence and fence_lang not in ("", "python", "py"):
            continue
        unit = line
        while unit.count("(") > unit.count(")"):
            if index >= len(lines):
                break
            unit += " " + lines[index]
            index += 1
        for match in IMPORT_RE.finditer(unit):
            module, names_text = match.group("module"), match.group("names")
            if names_text.strip() in ("*", ""):
                continue
            # cut at the next import statement or trailing comment
            names_text = re.split(r"\bimport\b|#", names_text, maxsplit=1)[0]
            names_text = names_text.replace("(", " ").replace(")", " ")
            names: list[str] = []
            for part in re.split(r"[,;]", names_text):
                head = part.strip().split(" as ")[0].strip().split(" ")[0]
                if head.isidentifier():
                    names.append(head)
            records.append((index, module, names))
    return records


def validate(repo_root: Path) -> tuple[list[str], list[str], int]:
    """Return (errors, diagnostics, pages_checked)."""
    errors: list[str] = []
    diagnostics: list[str] = []
    mapping = package_dirs(repo_root)
    text_cache: dict[Path, str] = {}
    pages_checked = 0
    pages = sorted(
        page for doc_root in repo_root.glob(DOC_GLOB) for page in doc_root.rglob("*.md")
    )
    for page in pages:
        rel = str(page.relative_to(repo_root))
        text = page.read_text(encoding="utf-8", errors="replace")
        if BANNER_MARKER in text:
            errors.append(f"{rel}: obsolete illustrative-example exemption")
        pages_checked += 1
        page_errors: list[str] = []
        for _, module, names in extract_imports(page):
            if not names:
                continue
            top, *rest = module.split(".")
            package_dir = mapping.get(top)
            if package_dir is None:
                page_errors.append(f"{rel}: phantom package {top} (no src tree)")
                continue
            for name in names:
                if name_resolves(package_dir, ".".join(rest), name, text_cache):
                    continue
                page_errors.append(f"{rel}: cannot resolve {module} import {name}")
        errors.extend(page_errors)
    return errors, diagnostics, pages_checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Emit diagnostics as stderr notes (same exit contract).",
    )
    args = parser.parse_args()

    errors, diagnostics, pages_checked = validate(REPO_ROOT)
    if args.strict:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
    if errors:
        print("Documentation import validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"Documentation import validation passed for {pages_checked} pages "
        f"({len(diagnostics)} diagnostic notes)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
