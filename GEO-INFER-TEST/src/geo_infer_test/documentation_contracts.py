"""Lightweight checks shared by documentation and manuscript entrypoints."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

_TEST_FILE_CENSUS = re.compile(r"(\d+)\s+test files", re.IGNORECASE)


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
