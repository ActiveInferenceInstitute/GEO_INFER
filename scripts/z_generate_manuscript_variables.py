#!/usr/bin/env python3
"""Hydrate GEO-INFER manuscript variables before every render.

The docxology template looks for exactly this path — ``scripts/
z_generate_manuscript_variables.py`` — in
:func:`infrastructure.rendering._manuscript_source.run_manuscript_variable_script`,
and when it is absent that function returns ``0`` with no log line, which the
pipeline reads as "hydrated OK".  GEO-INFER's generator lives at
``docs/manuscript/generate_research_artifacts.py``, a path the template never looks
for, so the manuscript's own invariant — "changing values must enter the
manuscript through the generator" — was enforced by human discipline alone and
``stage_03_render.py --project GEO-INFER`` would publish whatever happened to
be on disk and exit 0.

This shim closes that gap.  Every render now regenerates the evidence bundle
and then re-checks it, so a render either produces current values or fails.

Modes:
    default
        Regenerate from the working tree.  An unclean checkout is permitted
        but is stamped ``<sha>-dirty`` and its uncommitted-entry count is
        published as ``RESEARCH_TREE_DIRTY_FILE_COUNT`` — a render during
        ordinary development must work, but it must not claim to be a clean
        commit.  This mode runs no verification command, but it does not
        destroy the record of commands that did run: a stored record naming
        this source hash, commit, and tier is reused and republished, and a
        record describing a different tree is carried forward with its own
        provenance rather than replaced.  Without that, every
        default render — which is what ``stage_03_render.py --project
        GEO-INFER`` performs — would overwrite the evidence bundle with an
        empty one and publish ``not run``.
    ``GEO_INFER_MANUSCRIPT_VERIFY=1``
        Evidence build.  Publishes a real per-group record instead of an empty
        one.  A stored record is reused when it names the same source hash,
        commit, and tier; otherwise the commands run here, which can exceed the
        renderer's bounded hydration timeout — pre-run
        ``python docs/manuscript/generate_research_artifacts.py --verify`` on a
        clean tree first.  A failing group is published with its return code
        and does not abort the render; only a publication build refuses.  The
        run is widened to cover every group the stored record already holds,
        because replacing an eleven-group record with a seven-group one would
        delete measured outcomes this build never re-ran.
    ``GEO_INFER_MANUSCRIPT_FULL_VALIDATION=1``
        Request the second tier of command groups without asking for a
        publication build.  It selects which commands a verifying build
        *runs*; it does not select the denominator the manuscript publishes
        against.  That denominator is read from the record itself — from the
        tier its results were measured at — so a render that republishes an
        eleven-group record publishes eleven defined groups whether or not
        this variable is set.  It used to be the only thing standing between
        the shipped abstract and an impossible ``9 of 7``, which made a
        correct build depend on an operator remembering an environment
        variable.  It does not relax ``--publication``: a build that carries
        a failed group is still refused when publication is requested.
    ``GEO_INFER_MANUSCRIPT_PUBLICATION=1``
        Publication build.  Requires a clean checkout, a non-empty verification
        record with no failures, and runs the full validation suite.

Runs standalone as well: ``python scripts/z_generate_manuscript_variables.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "docs" / "manuscript" / "generate_research_artifacts.py"
PUBLICATION_ENV = "GEO_INFER_MANUSCRIPT_PUBLICATION"
VERIFY_ENV = "GEO_INFER_MANUSCRIPT_VERIFY"
FULL_VALIDATION_ENV = "GEO_INFER_MANUSCRIPT_FULL_VALIDATION"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _load_generator() -> ModuleType:
    """Import the generator by path, without requiring it to be a package."""
    spec = importlib.util.spec_from_file_location(
        "geo_infer_manuscript_generator", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the manuscript generator: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _publication_requested() -> bool:
    return _enabled(PUBLICATION_ENV)


def _verify_requested() -> bool:
    return _enabled(VERIFY_ENV) or _publication_requested()


def _full_validation_requested() -> bool:
    """Whether this build asks for the second tier of command groups.

    Deliberately does not imply :func:`_verify_requested`: asking for the
    wider tier is not asking to run anything, and coupling the two would make
    an ordinary render re-run eleven suites inside a 300-second hydration
    timeout.

    It is also not the denominator.  The manuscript counts recorded outcomes
    against the tier the *record* was measured at, which travels with the
    record, so a render that carries an eleven-group record forward publishes
    eleven defined groups with this variable unset.  A flag whose absence
    could publish more passes than defined groups was a denominator in the
    wrong place.
    """
    return _enabled(FULL_VALIDATION_ENV) or _publication_requested()


def main() -> int:
    if not GENERATOR_PATH.is_file():
        print(f"manuscript generator is missing: {GENERATOR_PATH}", file=sys.stderr)
        return 1
    generator = _load_generator()
    publication = _publication_requested()
    verify = _verify_requested()
    full_validation = _full_validation_requested()
    try:
        manifest = generator.generate(
            PROJECT_ROOT,
            verify=verify,
            full_validation=full_validation,
            allow_dirty=not publication,
            publication=publication,
            reuse_verification=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"manuscript variable hydration failed: {exc}", file=sys.stderr)
        return 1
    # A failed verification group is evidence, not a hydration error: the
    # generator has already written it into the record the manuscript
    # publishes. Report it and keep rendering, so the PDF states the failure
    # instead of the render hiding it behind an exit code.
    for name in manifest.get("verification_failures") or ():
        print(f"verification group recorded as failed: {name}", file=sys.stderr)
    problems = generator.check_published_artifacts(PROJECT_ROOT)
    for problem in problems:
        print(
            f"manuscript artifacts are stale after hydration: {problem}",
            file=sys.stderr,
        )
    if problems:
        return 1
    mode = "publication" if publication else "evidence" if verify else "working-tree"
    print(f"manuscript variables hydrated ({mode} build)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
