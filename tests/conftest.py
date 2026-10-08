"""Fixtures for the manuscript-generator regression suite.

These tests exercise ``manuscript/generate_research_artifacts.py`` directly.
The generator is a standalone script rather than an installed package, so it is
loaded by path and registered under a stable module name (frozen dataclasses
need their defining module to be importable while the class body executes).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "manuscript" / "generate_research_artifacts.py"
GENERATOR_MODULE = "geo_infer_manuscript_generator"


def _git_env(home: Path) -> dict[str, str]:
    """A git invocation environment whose ``PATH`` locates this machine's git.

    A hardcoded macOS PATH broke on any other host, and ``git`` is invoked
    with a scrubbed environment precisely so a developer's own config cannot
    leak into a test checkout.  The directory that actually contains the
    ``git`` binary being driven is the one that belongs on ``PATH``; the rest
    of the inherited path is kept so helper scripts git itself calls still
    resolve.
    """
    git_dir = ""
    resolved = shutil.which("git")
    if resolved:
        git_dir = f"{Path(resolved).parent}{os.pathsep}"
    return {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "PATH": f"{git_dir}{os.environ.get('PATH', '')}",
        "HOME": str(home),
    }


@pytest.fixture
def git_environment(tmp_path: Path) -> dict[str, str]:
    """A scrubbed environment for driving git inside throwaway checkouts."""
    return _git_env(tmp_path)


def _load_generator() -> ModuleType:
    if GENERATOR_MODULE in sys.modules:
        return sys.modules[GENERATOR_MODULE]
    spec = importlib.util.spec_from_file_location(GENERATOR_MODULE, GENERATOR_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise RuntimeError(f"cannot load generator from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[GENERATOR_MODULE] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def generator() -> ModuleType:
    """The manuscript generator module under test."""
    return _load_generator()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The real GEO-INFER checkout these tests measure."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def repo_inventory(generator: ModuleType, repo_root: Path):
    """One measured inventory of the real checkout, shared by the suite."""
    return generator.collect_inventory(repo_root)


@pytest.fixture
def figure_specs(generator: ModuleType):
    """A minimal, valid figure-spec tuple for variable-building tests."""
    return (
        generator.FigureSpec(
            label="fig:example",
            filename="example.png",
            caption="Example caption.",
            generated_by="tests",
            alt_text="Example alt text.",
        ),
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """An initialised, committed, clean git repository."""
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    env = _git_env(tmp_path)
    for args in (
        ["init", "-q", "-b", "main"],
        ["add", "-A"],
        ["commit", "-q", "-m", "seed"],
    ):
        subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
            check=True,
            env=env,
            capture_output=True,
        )
    return root


def _output_tree_digest(root: Path) -> str:
    """A fingerprint of the shipped artifact tree, path and content."""
    output = root / "output"
    if not output.is_dir():
        return "absent"
    digest = hashlib.sha256()
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def shipped_artifacts_are_read_only(repo_root: Path):
    """Fail the session if any test rewrites the shipped ``output/`` tree.

    ``output/`` is what the repository publishes: the resolved manuscript, the
    figures, and the evidence record whose commands take minutes to produce.
    A test that runs the real generator or the real render shim against the
    real checkout rewrites all three, and a later assertion then certifies
    whatever it wrote.  Reproduced twice: ``pytest tests/`` replaced the
    eleven-group evidence bundle with a seven-group one and the suite passed.

    Every test that needs a generator run has a checkout of its own — a
    synthetic tree, ``git_repo``, or ``shim_checkout``.  This fixture is the
    gate that keeps it that way, so the rule holds for tests not yet written.
    """
    before = _output_tree_digest(repo_root)
    yield
    after = _output_tree_digest(repo_root)
    if before != after:
        pytest.fail(
            "the test session rewrote the shipped output/ tree: digest "
            f"{before[:16]} -> {after[:16]}. Run the generator against a "
            "checkout of the test's own (see the shim_checkout fixture), "
            "never against repo_root."
        )


@pytest.fixture
def generatable_checkout(generator: ModuleType, tmp_path: Path) -> Path:
    """A committed checkout with the minimum shape ``generate`` accepts.

    The generator refuses a tree missing any themed module, so every declared
    module gets a one-file source package.  None of those counts are the
    subject here; the verification record is.  Shared by every test that
    drives the real ``generate`` against a synthetic tree.
    """
    root = tmp_path / "checkout"
    root.mkdir()
    manuscript = root / "manuscript"
    manuscript.mkdir()
    (manuscript / "config.yaml").write_text(
        "paper:\n"
        '  version: "0.0.0"\n'
        '  date: "1970-01-01T00:00:00+00:00"\n'
        "publication:\n"
        '  year: "1970"\n'
        "metadata:\n"
        '  license: "unset"\n'
        "bibliography:\n"
        '  references_path: "manuscript/references.bib"\n'
        "  fail_on_missing: true\n"
        "  fail_on_unused: true\n",
        encoding="utf-8",
    )
    (manuscript / "00_abstract.md").write_text(
        "# Abstract\n\nSource hash {{RESEARCH_SOURCE_HASH}}.\n", encoding="utf-8"
    )
    (manuscript / "references.bib").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "1.2.3"\nlicense = "MIT"\n',
        encoding="utf-8",
    )
    # The real checkout ignores ``/output/``, so writing the evidence bundle
    # does not itself make the tree dirty.  Mirroring that is what lets these
    # tests distinguish "the record moved" from "the tree moved".
    (root / ".gitignore").write_text("/output/\n", encoding="utf-8")
    for _theme, names in generator.MODULE_THEMES:
        for name in names:
            package = name.removeprefix("GEO-INFER-").lower()
            source = root / name / "src" / f"geo_infer_{package}"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    env = _git_env(tmp_path)
    for args in (
        ["init", "-q", "-b", "main"],
        ["add", "-A"],
        ["commit", "-q", "-m", "seed"],
    ):
        subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
            check=True,
            env=env,
            capture_output=True,
        )
    return root
