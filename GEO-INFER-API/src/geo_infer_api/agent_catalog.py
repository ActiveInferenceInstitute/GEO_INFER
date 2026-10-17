"""Import-free discovery of public Python definitions in installed GEO packages."""

from __future__ import annotations

import ast
import importlib.util
from importlib.metadata import PackageNotFoundError, distribution
from functools import lru_cache
from pathlib import Path
from typing import Any

MODULES = tuple(
    "ACT AG AGENT AI ANT API APP ART BAYES BIO CIV CLIMATE COG COMMS DATA ECON "
    "EDU EMERGENCY ENERGY EXAMPLES FOREST GIT HEALTH INSURANCE INTRA IOT LOG "
    "MARINE MATH METAGOV NORMS OPS ORG PEP PLACE REQ RISK SEC SIM SPACE SPM "
    "TEST TIME TRANSPORT WATER".split()
)


def package_name(module: str) -> str:
    """Normalize a workspace module name without accepting arbitrary imports."""
    name = module.upper().replace("GEO-INFER-", "").replace("GEO_INFER_", "")
    if name not in MODULES:
        raise ValueError(f"Unknown GEO-INFER module: {module}")
    return f"geo_infer_{name.lower()}"


def module_inventory() -> list[dict[str, Any]]:
    """Report installed packages; missing modules remain visible to agents."""
    rows = []
    for name in MODULES:
        package = package_name(name)
        spec = importlib.util.find_spec(package)
        try:
            metadata = distribution(package.replace("_", "-"))
        except PackageNotFoundError:
            metadata = None
        rows.append(
            {
                "module": name,
                "package": package,
                "installed": spec is not None,
                "version": metadata.version if metadata is not None else None,
                "cli": sorted(
                    entry.name
                    for entry in metadata.entry_points
                    if entry.group == "console_scripts"
                )
                if metadata is not None
                else [],
            }
        )
    return rows


def _definitions(
    body: list[ast.AST], module: str, prefix: str = ""
) -> list[dict[str, Any]]:
    rows = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            name = prefix + node.name
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            decorators = [ast.unparse(d) for d in node.decorator_list]
            if prefix and kind != "class":
                kind = "property" if "property" in decorators else "method"
            rows.append(
                {
                    "target": f"{module}:{name}",
                    "kind": kind,
                    "summary": (ast.get_docstring(node) or "").split("\n")[0],
                    "signature": None
                    if isinstance(node, ast.ClassDef)
                    else f"({ast.unparse(node.args)})",
                    "async": isinstance(node, ast.AsyncFunctionDef),
                }
            )
            if isinstance(node, ast.ClassDef):
                rows.extend(_definitions(node.body, module, name + "."))
        else:
            # Include conditional definitions, but never local function bodies.
            children = [
                child
                for child in ast.iter_child_nodes(node)
                if isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case))
            ]
            if children:
                rows.extend(_definitions(children, module, prefix))
    return rows


@lru_cache(maxsize=45)
def definitions(module: str) -> tuple[dict[str, Any], ...]:
    """Index source definitions without importing scientific backends or services.

    Re-exports, inherited members and generated methods are resolved by runtime
    describe/call instead. A static definition is discoverable, not a promise
    that its optional dependencies or external services are installed.
    """
    package = package_name(module)
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return ()
    rows: dict[str, dict[str, Any]] = {}
    for location in spec.submodule_search_locations:
        root = Path(location)
        for source in sorted(root.rglob("*.py")):
            relative = source.relative_to(root)
            parts = relative.with_suffix("").parts
            if any(p.startswith("_") and p != "__init__" for p in parts):
                continue
            dotted = ".".join((package, *parts))
            if dotted.endswith(".__init__"):
                dotted = dotted.removesuffix(".__init__")
            tree = ast.parse(source.read_text(encoding="utf-8-sig"))
            for row in _definitions(tree.body, dotted):
                rows[row["target"]] = row
    return tuple(rows[key] for key in sorted(rows))


def catalog(
    module: str, query: str = "", offset: int = 0, limit: int = 50
) -> dict[str, Any]:
    """Search a bounded, deterministic page of a module's source definitions."""
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer from 1 to 200")
    rows = definitions(module)
    needle = query.casefold()
    matches = (
        tuple(
            row
            for row in rows
            if needle in (row["target"] + " " + row["summary"]).casefold()
        )
        if needle
        else rows
    )
    next_offset = offset + limit if offset + limit < len(matches) else None
    return {
        "module": package_name(module),
        "total": len(matches),
        "offset": offset,
        "next_offset": next_offset,
        "items": [dict(row) for row in matches[offset : offset + limit]],
    }
