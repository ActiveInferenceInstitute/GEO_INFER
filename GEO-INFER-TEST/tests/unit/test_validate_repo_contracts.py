"""Unit tests for repository contract validator helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_PATH = REPO_ROOT / "GEO-INFER-TEST" / "validate_repo_contracts.py"
REWRITER_PATH = REPO_ROOT / "GEO-INFER-TEST" / "rewrite_readme_agents.py"


def load_contracts_module():
    spec = importlib.util.spec_from_file_location(
        "geo_infer_validate_repo_contracts", CONTRACTS_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_rewriter_module():
    spec = importlib.util.spec_from_file_location(
        "geo_infer_rewrite_readme_agents_for_test", REWRITER_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_signpost_public_interfaces_exclude_private_modules(tmp_path):
    (tmp_path / "_worker.py").write_text("def launch():\n    pass\n", encoding="utf-8")
    (tmp_path / "test_example.py").write_text(
        "def scenario():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "__init__.py").write_text(
        "def supported_api():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "public.py").write_text(
        "class PublicModel:\n    pass\ndef _private_helper():\n    pass\n",
        encoding="utf-8",
    )
    assert load_rewriter_module().public_symbols(tmp_path) == [
        "`__init__.py:supported_api` (function)",
        "`public.py:PublicModel` (class)",
    ]


def test_signpost_inventory_includes_new_files_and_excludes_deletions(
    tmp_path, monkeypatch
):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "core.fsmonitor", "false"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "tracked.py"
    tracked.write_text("tracked = True\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    tracked.unlink()

    added = tmp_path / "added.py"
    added.write_text("added = True\n")
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    (tmp_path / "ignored.py").write_text("ignored = True\n")

    rewriter = load_rewriter_module()
    monkeypatch.setattr(rewriter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", None)

    inventory = rewriter.tracked_files()
    assert added in inventory
    assert tracked not in inventory
    assert tmp_path / "ignored.py" not in inventory


def test_signpost_directory_index_preserves_deep_entries_and_visibility(
    tmp_path, monkeypatch
):
    """Deep registered descendants expose ancestors without admitting scratch dirs."""
    rewriter = load_rewriter_module()
    monkeypatch.setattr(rewriter, "REPO_ROOT", tmp_path)
    inventory = set()
    for name in (
        "alpha.py",
        "config.yaml",
        "README.md",
        "AGENTS.md",
        "deep/inner/model.py",
        ".venv/hidden.py",
        "__pycache__/hidden.py",
        ".git/hidden.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
        inventory.add(path)
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "unregistered.py").write_text("# scratch\n")
    (tmp_path / "empty").mkdir()
    deleted = tmp_path / "deleted" / "model.py"
    deleted.parent.mkdir()
    deleted.write_text("# removed directory\n")
    inventory.add(deleted)
    deleted.unlink()
    deleted.parent.rmdir()
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", inventory)

    assert rewriter.direct_contents(tmp_path) == (
        ["deep/"],
        ["alpha.py"],
        ["config.yaml"],
    )
    assert rewriter.direct_contents(tmp_path / "deep") == (["inner/"], [], [])
    assert rewriter.direct_contents(tmp_path / "deep" / "inner") == (
        [],
        ["model.py"],
        [],
    )


def test_signpost_standalone_queries_observe_same_size_mutation_and_reset(
    tmp_path, monkeypatch
):
    """A render cache cannot leak stale facts into mutable standalone helpers."""
    rewriter = load_rewriter_module()
    module_path = tmp_path / "GEO-INFER-SAMPLE"
    tests = module_path / "tests"
    tests.mkdir(parents=True)
    old = tests / "test_old.py"
    new = tests / "helper.py"
    old.write_text("def test_old(): assert True\n")
    new.write_text("def support(): return 1\n")
    inventory = {old}
    monkeypatch.setattr(rewriter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", inventory)
    module = rewriter.ModuleInfo(
        "GEO-INFER-SAMPLE", module_path, "geo_infer_sample", "Sample", "0.4.0", [], 0, 1
    )
    readme = tests / "README.md"
    assert rewriter.test_command(readme, module).endswith(
        f"-m pytest {tests.relative_to(tmp_path)}"
    )
    inventory.remove(old)
    inventory.add(new)
    assert rewriter.direct_contents(tests)[1] == ["helper.py"]
    assert rewriter.test_command(readme, module).endswith("--module SAMPLE")
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", {old})
    assert rewriter.direct_contents(tests)[1] == ["test_old.py"]
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", None)
    monkeypatch.setattr(rewriter, "git_ls_files", lambda: [new])
    assert rewriter.direct_contents(tests)[1] == ["helper.py"]
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = other_root / "new.py"
    other.write_text("def new_api(): return 1\n")
    (other_root / "scratch.py").write_text("def scratch_api(): return 2\n")
    prior_root_index = rewriter._build_directory_index(frozenset({old}))
    monkeypatch.setattr(rewriter, "_DIRECTORY_INDEX", prior_root_index)
    monkeypatch.setattr(rewriter, "REPO_ROOT", other_root)
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", {other})
    assert rewriter.direct_contents(other_root)[1] == ["new.py"]
    assert rewriter.public_symbols(other_root) == [
        "`new.py:new_api` (function)",
        "`scratch.py:scratch_api` (function)",
    ]


def test_signpost_render_owns_one_snapshot_and_restores_standalone_facts(
    tmp_path, monkeypatch
):
    """One render is consistent; later renders observe even same-size set edits."""
    rewriter = load_rewriter_module()
    alpha = tmp_path / "alpha.py"
    beta = tmp_path / "beta.py"
    alpha.write_text("def alpha_api(): return 1\n")
    beta.write_text("def beta_api(): return 2\n")
    readme, agents = tmp_path / "README.md", tmp_path / "AGENTS.md"
    inventory = {alpha}
    monkeypatch.setattr(rewriter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", inventory)
    monkeypatch.setattr(rewriter, "discover_modules", dict)
    monkeypatch.setattr(rewriter, "repository_doc_files", lambda: ([readme], [agents]))
    builds = []
    original_build = rewriter._build_directory_index

    def build(snapshot):
        builds.append(len(snapshot))
        return original_build(snapshot)

    def render_readme(*_args):
        return repr(
            (rewriter.direct_contents(tmp_path), rewriter.public_symbols(tmp_path))
        )

    def render_agents(*_args):
        inventory.discard(alpha)
        inventory.add(beta)
        return render_readme()

    monkeypatch.setattr(rewriter, "_build_directory_index", build)
    monkeypatch.setattr(rewriter, "render_readme", render_readme)
    monkeypatch.setattr(rewriter, "render_agents", render_agents)
    first = rewriter.expected_doc_files()
    assert first[0][1] == first[1][1]
    assert "alpha_api" in first[0][1] and "beta_api" not in first[0][1]
    assert builds == [1]
    assert rewriter._DIRECTORY_INDEX is None
    assert rewriter.direct_contents(tmp_path)[1] == ["beta.py"]
    second = rewriter.expected_doc_files()
    assert second[0][1] == second[1][1]
    assert "beta_api" in second[0][1] and "alpha_api" not in second[0][1]
    assert builds == [1, 1, 1]
    assert rewriter._DIRECTORY_INDEX is None


def test_signpost_render_restores_prior_index_when_discovery_fails(
    tmp_path, monkeypatch
):
    """A failed render must not leave its private snapshot active for callers."""
    rewriter = load_rewriter_module()
    monkeypatch.setattr(rewriter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", set())
    prior = rewriter._build_directory_index(frozenset())
    monkeypatch.setattr(rewriter, "_DIRECTORY_INDEX", prior)

    def fail():
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(rewriter, "discover_modules", fail)
    with pytest.raises(RuntimeError, match="discovery failed"):
        rewriter.expected_doc_files()
    assert rewriter._DIRECTORY_INDEX is prior


def test_signpost_index_visits_each_file_and_distinct_ancestor_once(monkeypatch):
    """Shared deep paths do not multiply ancestor walks by the number of files."""
    rewriter = load_rewriter_module()
    reads = []

    class Node:
        def __init__(self, name, parent=None):
            self.name = name
            self._parent = parent or self

        @property
        def parent(self):
            reads.append(self.name)
            return self._parent

    root = Node("root")
    module = Node("module", root)
    src = Node("src", module)
    left, right = Node("left", src), Node("right", src)
    files = [
        Node(f"file-{index}", left if index % 2 else right) for index in range(120)
    ]
    iterations = []

    class Inventory:
        def __iter__(self):
            iterations.append(1)
            return iter(files)

    index = rewriter._build_directory_index(Inventory())
    assert iterations == [1]
    assert len(reads) == len(files) + 5
    assert all(
        reads.count(name) == 1 for name in ("root", "module", "src", "left", "right")
    )
    assert index.children[root] == {module}
    assert index.children[src] == {left, right}
    assert set(index.files[left] + index.files[right]) == set(files)


def test_signpost_render_excludes_ignored_symbols_and_includes_new_sources(
    tmp_path, monkeypatch
):
    """Actual Git ignore rules bound generated exports while preserving new files."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "core.fsmonitor", "false"], cwd=tmp_path, check=True
    )
    (tmp_path / ".gitignore").write_text("ignored.py\nignored-tree/\n")
    (tmp_path / "ignored.py").write_text("def ignored_api(): return 1\n")
    (tmp_path / "new.py").write_text("def new_api(): return 2\n")
    ignored_tree = tmp_path / "ignored-tree"
    ignored_tree.mkdir()
    (ignored_tree / "scratch.py").write_text("def scratch_api(): return 3\n")
    for name in ("README.md", "AGENTS.md"):
        (tmp_path / name).write_text("# fixture\n")
    rewriter = load_rewriter_module()
    monkeypatch.setattr(rewriter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rewriter, "_TRACKED_FILES", None)
    monkeypatch.setattr(rewriter, "discover_modules", dict)
    monkeypatch.setattr(
        rewriter,
        "render_readme",
        lambda *_args: repr(rewriter.public_symbols(tmp_path)),
    )
    monkeypatch.setattr(
        rewriter,
        "render_agents",
        lambda *_args: repr(rewriter.direct_contents(tmp_path)),
    )
    rendered = dict(rewriter.expected_doc_files())
    assert "new_api" in rendered[tmp_path / "README.md"]
    assert "ignored_api" not in rendered[tmp_path / "README.md"]
    assert "ignored-tree/" not in rendered[tmp_path / "AGENTS.md"]
    assert rewriter._DIRECTORY_INDEX is None
    assert any("ignored_api" in symbol for symbol in rewriter.public_symbols(tmp_path))


def write_root_uv_files(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'requires-python = ">=3.11"',
                "",
                "[tool.uv.workspace]",
                'members = ["GEO-INFER-*"]',
                "",
            ]
        )
    )
    (root / ".python-version").write_text("3.12.11\n")
    (root / "uv.lock").write_text("")


def write_canonical_uv_docs(root: Path, contracts) -> None:
    for relative_path in contracts.CANONICAL_UV_DOC_FILES:
        doc_path = root / relative_path
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(f"Setup command: `{contracts.CANONICAL_UV_SYNC_COMMAND}`\n")


def test_uv_environment_contract_accepts_root_workspace(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    write_root_uv_files(tmp_path)
    report = contracts.ContractReport()

    contracts.validate_uv_environment(report)

    assert report.errors == []


def test_uv_environment_contract_rejects_missing_lock(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    write_root_uv_files(tmp_path)
    (tmp_path / "uv.lock").unlink()
    report = contracts.ContractReport()

    contracts.validate_uv_environment(report)

    assert any("uv.lock" in error for error in report.errors)


def test_uv_setup_documentation_requires_workspace_sync_command(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    write_canonical_uv_docs(tmp_path, contracts)
    (tmp_path / "README.md").write_text("Setup command: `uv sync --all-extras`\n")
    report = contracts.ContractReport()

    contracts.validate_uv_setup_documentation(report)

    assert any("README.md" in error for error in report.errors)


def test_uv_setup_documentation_accepts_canonical_sync_command(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    write_canonical_uv_docs(tmp_path, contracts)
    report = contracts.ContractReport()

    contracts.validate_uv_setup_documentation(report)

    assert report.errors == []


def test_test_inventory_contract_requires_minimum_files(tmp_path):
    contracts = load_contracts_module()
    module_dir = tmp_path / "GEO-INFER-SAMPLE"
    tests_dir = module_dir / "tests"
    tests_dir.mkdir(parents=True)
    for index in range(3):
        (tests_dir / f"test_case_{index}.py").write_text("def test_ok():\n    pass\n")
    report = contracts.ContractReport()

    contracts.validate_test_inventory([module_dir], report)

    assert any("expected at least" in error for error in report.errors)


def test_module_task_marker_contract_scans_source_and_tests(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    src_dir = tmp_path / "GEO-INFER-SAMPLE" / "src" / "geo_infer_sample"
    tests_dir = tmp_path / "GEO-INFER-SAMPLE" / "tests"
    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    marker = "TO" + "DO"
    (src_dir / "module.py").write_text(f"# {marker}: route work through tracker\n")
    (tests_dir / "test_module.py").write_text(
        f"def test_marker():\n    value = '{marker}'\n    assert value\n"
    )
    report = contracts.ContractReport()

    contracts.validate_module_task_markers(report)

    errors = "\n".join(report.errors)
    assert "module.py" in errors
    assert "test_module.py" in errors


def test_module_task_marker_contract_scans_module_root_scripts(tmp_path, monkeypatch):
    """GS19-85: module-root scripts are scanned; ledger references are not.

    A real marker in a module-root script must be caught by the scan, while
    the token alternation inside a validator's own pattern definition and a
    reference to the root ``TODO.md`` ledger filename must not trip it.
    """
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    module_dir = tmp_path / "GEO-INFER-SAMPLE"
    module_dir.mkdir()
    todo = "TO" + "DO"
    fixme = "FIX" + "ME"
    (module_dir / "needs_work.py").write_text(f"# {todo}: fix before release\n")
    (module_dir / "validate_sample.py").write_text(
        f'PATTERN = re.compile(r"\\b({todo}|{fixme})\\b")\n'
        f'HELP = "track planned work in root {todo}.md or issues"\n'
    )
    report = contracts.ContractReport()

    contracts.validate_module_task_markers(report)

    errors = "\n".join(report.errors)
    assert "GEO-INFER-SAMPLE/needs_work.py" in errors
    assert "validate_sample.py" not in errors


def test_logging_contract_rejects_library_basic_config(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    src_dir = tmp_path / "GEO-INFER-SAMPLE" / "src" / "geo_infer_sample"
    src_dir.mkdir(parents=True)
    (src_dir / "library.py").write_text(
        "import logging\nlogging.basicConfig(level=logging.INFO)\n"
    )
    (src_dir / "cli.py").write_text(
        "import logging\n"
        "if __name__ == '__main__':\n"
        "    logging.basicConfig(level=logging.INFO)\n"
    )
    report = contracts.ContractReport()

    contracts.validate_logging_configuration(report)

    errors = "\n".join(report.errors)
    assert "library.py" in errors
    assert "cli.py" not in errors


def test_python_source_syntax_contract_rejects_invalid_module_source(
    tmp_path, monkeypatch
):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    src_dir = tmp_path / "GEO-INFER-SAMPLE" / "src" / "geo_infer_sample"
    examples_dir = tmp_path / "GEO-INFER-SAMPLE" / "examples"
    src_dir.mkdir(parents=True)
    examples_dir.mkdir(parents=True)
    (src_dir / "valid.py").write_text("VALUE = 1\n")
    (examples_dir / "broken.py").write_text("def broken(:\n    pass\n")
    report = contracts.ContractReport()

    contracts.validate_python_source_syntax(report)

    errors = "\n".join(report.errors)
    assert str(Path("GEO-INFER-SAMPLE") / "examples" / "broken.py") in errors
    assert "invalid syntax" in errors


def test_concrete_pass_contract_rejects_non_abstract_source_pass(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    src_dir = tmp_path / "GEO-INFER-SAMPLE" / "src" / "geo_infer_sample"
    src_dir.mkdir(parents=True)
    (src_dir / "module.py").write_text("def real_behavior():\n    pass\n")
    report = contracts.ContractReport()

    contracts.validate_no_concrete_pass_bodies(report)

    errors = "\n".join(report.errors)
    assert "module.py:2" in errors
    assert "Concrete pass bodies" in errors


def test_concrete_pass_contract_allows_abstract_methods_and_except_handlers(
    tmp_path, monkeypatch
):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    src_dir = tmp_path / "GEO-INFER-SAMPLE" / "src" / "geo_infer_sample"
    src_dir.mkdir(parents=True)
    (src_dir / "module.py").write_text(
        "from abc import abstractmethod\n\n"
        "class Base:\n"
        "    @abstractmethod\n"
        "    def run(self):\n"
        "        pass\n\n"
        "def cleanup(value):\n"
        "    try:\n"
        "        return int(value)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    return None\n"
    )
    report = contracts.ContractReport()

    contracts.validate_no_concrete_pass_bodies(report)

    assert report.errors == []


def test_python_tool_targets_reject_ruff_target_below_python_311(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.ruff]\ntarget-version = "py310"\n')
    report = contracts.ContractReport()

    contracts.validate_python_tool_targets(report)

    assert any("py310" in error for error in report.errors)


def test_python_tool_targets_reject_retired_formatter_sections(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[tool.black]\nline-length = 88\n\n[tool.isort]\nprofile = 'black'\n"
    )
    report = contracts.ContractReport()

    contracts.validate_python_tool_targets(report)

    assert any("[tool.black]" in error for error in report.errors)
    assert any("[tool.isort]" in error for error in report.errors)


def test_pyproject_only_packaging_rejects_retired_mirrors(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    module_dir = tmp_path / "GEO-INFER-SAMPLE"
    module_dir.mkdir()
    (module_dir / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    (module_dir / "requirements.txt").write_text("numpy\n")
    report = contracts.ContractReport()

    contracts.validate_pyproject_only_packaging([module_dir], report)

    assert any("setup.py" in error for error in report.errors)
    assert any("requirements.txt" in error for error in report.errors)


def test_declared_requirements_reject_stdlib_dependencies(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    module_dir = tmp_path / "GEO-INFER-SAMPLE"
    module_dir.mkdir()
    (module_dir / "pyproject.toml").write_text(
        '[project]\nname = "geo-infer-sample"\ndependencies = ["json", "numpy"]\n'
    )
    report = contracts.ContractReport()

    contracts.validate_declared_requirements(report)

    assert any("json" in error for error in report.errors)
    assert not any("numpy" in error for error in report.errors)


def test_h3_dependency_metadata_rejects_old_h3_v4_floor(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    module_dir = tmp_path / "GEO-INFER-SAMPLE"
    module_dir.mkdir()
    (module_dir / "pyproject.toml").write_text(
        "[project]\ndependencies = ['h3>=4.0.0']\n"
    )
    report = contracts.ContractReport()

    contracts.validate_h3_dependency_metadata(report)

    assert any("h3>=4.5.0,<5" in error for error in report.errors)


def test_generated_doc_freshness_rejects_stale_rendered_docs(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("stale\n")

    class FakeRewriter:
        @staticmethod
        def expected_doc_files():
            return [(readme, "fresh\n")]

    report = contracts.ContractReport()

    contracts.validate_generated_doc_freshness(report, rewriter=FakeRewriter)

    assert any(
        "README.md/AGENTS.md files are stale" in error for error in report.errors
    )


def test_generated_artifact_contract_checks_root_test_output(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)

    class FakeCompleted:
        stdout = "?? test_output/\n"

    def fake_run(*args, **kwargs):
        return FakeCompleted()

    monkeypatch.setattr(contracts.subprocess, "run", fake_run)
    report = contracts.ContractReport()

    contracts.validate_generated_artifacts(report)

    assert any("test_output" in error for error in report.errors)


def test_pymdp_runtime_import_contract_rejects_legacy_runtime_paths(
    tmp_path, monkeypatch
):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    act_src = tmp_path / "GEO-INFER-ACT" / "src" / "geo_infer_act"
    tests_dir = tmp_path / "GEO-INFER-ACT" / "tests"
    act_src.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (act_src / "runtime.py").write_text(
        "from pymdp.control import construct_policies\n"
    )
    (tests_dir / "test_legacy_note.py").write_text("LEGACY = 'pymdp.control'\n")
    report = contracts.ContractReport()

    contracts.validate_pymdp_runtime_imports(report)

    errors = "\n".join(report.errors)
    assert "runtime.py" in errors
    assert "test_legacy_note.py" not in errors


def test_import_smoke_isolates_timeout_and_continues(tmp_path, monkeypatch):
    """A hanging package cannot prevent subsequent packages from being probed."""
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    modules = []
    for name, source in [
        ("slow", "import time\ntime.sleep(10)\n"),
        ("healthy", "VALUE = 42\n"),
    ]:
        module = tmp_path / ("GEO-INFER-" + name.upper())
        package = module / "src" / ("geo_infer_" + name)
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(source)
        (module / "pyproject.toml").write_text(
            '[project]\nname = "geo-infer-' + name + '"\n'
        )
        modules.append(module)
    report = contracts.ContractReport()
    contracts.validate_import_smoke(modules, report, timeout=2)
    assert len(report.warnings) == 1
    assert "timed out" in report.warnings[0]
    assert "geo_infer_healthy" not in sys.modules


def test_import_smoke_does_not_accept_cwd_shadow_of_broken_source(
    tmp_path, monkeypatch
):
    """A healthy same-named cwd module cannot hide a broken source package."""
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    module = tmp_path / "GEO-INFER-PROBE"
    package = module / "src" / "geo_infer_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('raise RuntimeError("actual source broken")\n')
    (module / "pyproject.toml").write_text('[project]\nname="geo-infer-probe"\n')
    (tmp_path / "geo_infer_probe.py").write_text("VALUE = 42\n")

    report = contracts.ContractReport()
    contracts.validate_import_smoke([module], report, timeout=2)

    assert len(report.warnings) == 1
    assert "actual source broken" in report.warnings[0]


def test_import_smoke_rejects_regular_package_from_another_source_root(
    tmp_path, monkeypatch
):
    """An empty namespace candidate cannot fall through to another checkout."""
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    target = tmp_path / "GEO-INFER-PROBE"
    (target / "src" / "geo_infer_probe").mkdir(parents=True)
    (target / "pyproject.toml").write_text('[project]\nname="geo-infer-probe"\n')
    other = tmp_path / "GEO-INFER-SUPPORT"
    for name in ("geo_infer_probe", "geo_infer_support"):
        package = other / "src" / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 42\n")
    (other / "pyproject.toml").write_text('[project]\nname="geo-infer-support"\n')

    report = contracts.ContractReport()
    contracts.validate_import_smoke([target, other], report, timeout=2)

    assert len(report.warnings) == 1
    assert "outside expected source" in report.warnings[0]
    assert str(other / "src" / "geo_infer_probe") in report.warnings[0]


@pytest.mark.parametrize("source", ["raise SystemExit(0)", "import os; os._exit(0)"])
def test_import_smoke_rejects_early_success_exit(tmp_path, monkeypatch, source):
    """A zero exit before source-origin verification is an incomplete probe."""
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    module = tmp_path / "GEO-INFER-PROBE"
    package = module / "src" / "geo_infer_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source)
    (module / "pyproject.toml").write_text('[project]\nname="geo-infer-probe"\n')
    report = contracts.ContractReport()
    contracts.validate_import_smoke([module], report, timeout=2)
    assert len(report.warnings) == 1
    assert "completion receipt" in report.warnings[0]


def test_import_smoke_timeout_stops_descendants(tmp_path, monkeypatch):
    """Source-import descendants are terminated before subsequent probes run."""
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    module = tmp_path / "GEO-INFER-PROBE"
    package = module / "src" / "geo_infer_probe"
    package.mkdir(parents=True)
    started = tmp_path / "child-started"
    finished = tmp_path / "child-finished"
    child = (
        f"import pathlib,time; pathlib.Path({str(started)!r}).touch(); "
        f"time.sleep(3.0); pathlib.Path({str(finished)!r}).touch()"
    )
    (package / "__init__.py").write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "time.sleep(10)\n"
    )
    (module / "pyproject.toml").write_text('[project]\nname="geo-infer-probe"\n')
    report = contracts.ContractReport()
    contracts.validate_import_smoke([module], report, timeout=0.5)
    assert len(report.warnings) == 1
    assert "timed out" in report.warnings[0]
    # Under a loaded parallel/coverage-traced runner the child's spawn is
    # not bounded by any small window: the interpreter can take >5 s to
    # exec under xdist while 45 modules' coverage data is written (observed
    # twice on CI on PR #48's validate job, 2026-09-28). Poll generously;
    # if the process-group kill won the race and reaped the child before
    # exec, `started` never appears and the termination property holds
    # trivially.
    deadline = time.monotonic() + 30.0
    started_seen = started.is_file()
    while not started_seen and time.monotonic() < deadline:
        time.sleep(0.25)
        started_seen = started.is_file()
    assert not finished.exists()
    if started_seen:
        # The child did launch; give a child that somehow survived the kill
        # ample time to reach its finished touch (3.0 s after ITS start) and
        # assert it never does. A late `started` implies the child was
        # reaped pre-exec, so this branch only runs for genuinely live
        # children.
        time.sleep(3.5)
        assert not finished.exists()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
def test_import_smoke_rejects_nonfinite_timeout(timeout):
    """The source probe rejects nonfinite limits before launching a process."""
    contracts = load_contracts_module()
    with pytest.raises(ValueError, match="finite and positive"):
        contracts.validate_import_smoke([], contracts.ContractReport(), timeout=timeout)


def test_import_smoke_strict_promotes_failures_to_errors(tmp_path, monkeypatch):
    """--strict-import-smoke makes a failed probe a contract error (GS-003)."""
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    module = tmp_path / "GEO-INFER-SLOW"
    package = module / "src" / "geo_infer_slow"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("import time\ntime.sleep(10)\n")
    (module / "pyproject.toml").write_text('[project]\nname = "geo-infer-slow"\n')
    report = contracts.ContractReport()

    contracts.validate_import_smoke([module], report, timeout=2, strict=True)

    assert len(report.errors) == 1
    assert not report.warnings
    assert "timed out" in report.errors[0]


def _write_rng_helper(root, module: str, body: str) -> None:
    package = root / f"GEO-INFER-{module}" / "src" / f"geo_infer_{module.lower()}"
    (package / "utils").mkdir(parents=True)
    (package / "utils" / "rng.py").write_text(body)


def test_rng_helper_parity_allows_docstring_and_default_seed_drift(
    tmp_path, monkeypatch
):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    shared = "def resolve_rng(seed=None):\n    return seed\n"
    _write_rng_helper(
        tmp_path, "AAA", '"""A."""\nDEFAULT_SEED: int | None = None\n' + shared
    )
    _write_rng_helper(
        tmp_path, "BBB", '"""B."""\nDEFAULT_SEED: int | None = 0\n' + shared
    )
    report = contracts.ContractReport()

    contracts.validate_rng_helper_parity(report)

    assert report.errors == []


def test_rng_helper_parity_rejects_behavioral_drift(tmp_path, monkeypatch):
    contracts = load_contracts_module()
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    _write_rng_helper(tmp_path, "AAA", "def resolve_rng(seed=None):\n    return seed\n")
    _write_rng_helper(tmp_path, "BBB", "def resolve_rng(seed=None):\n    return None\n")
    report = contracts.ContractReport()

    contracts.validate_rng_helper_parity(report)

    assert any("GEO-INFER-BBB" in error for error in report.errors)
