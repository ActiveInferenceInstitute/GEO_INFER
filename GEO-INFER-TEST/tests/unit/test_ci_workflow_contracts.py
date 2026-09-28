"""Pinned structural contracts for the GitHub Actions workflow definitions.

The workflow files are the behavioral surface of the CI-GATES items
(GS-001/002/005/006/007/008): these tests pin the YAML structure those fixes
depend on, the same way the acceptance probes grep the workflow sources.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _load(name: str) -> dict:
    document = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _trigger(document: dict) -> dict:
    # PyYAML resolves a bare `on:` key to the boolean True (YAML 1.1).
    return document.get("on", document.get(True, {}))


def _dump(job: dict) -> str:
    # A wide dump keeps wrapped line-folding from splitting markers.
    return yaml.safe_dump(job, width=10**6)


def test_ci_runs_on_release_tags_for_the_release_gate():
    """GS-002: CI must run on v* tags so the release ci-gate has a run to wait for."""
    push = _trigger(_load("ci.yml"))["push"]
    assert push["tags"] == ["v*"]


def test_release_requires_ci_gate_and_always_verifies_wheels():
    """GS-002: the release job is gated on CI success and tag builds verify."""
    release = _load("release.yml")
    assert "ci-gate" in release["jobs"]
    assert "ci-gate" in release["jobs"]["release"]["needs"]

    build_steps = [
        step
        for step in release["jobs"]["release"]["steps"]
        if step.get("name") == "Build multi-package wheels"
    ]
    assert len(build_steps) == 1
    script = build_steps[0]["run"]
    assert "'--verify'" in script
    # The inverted conditional makes tag builds (empty inputs) verify.
    assert "run_isolated_verify != 'false'" in script


def test_release_attaches_wheels_to_the_release_object_after_verification():
    """REL-02: wheels must reach the release object, not only the 30-day artifact.

    v0.3.0 shipped wheel-less behind a green pipeline because release.yml
    published its 45 wheels solely as the ``geo-infer-wheels`` artifact and
    the release object stayed PDF-only until a manual owner attach. This pins
    the codified attach: it runs after the namespace verification, the
    retention artifact survives beside it, and both skip paths are annotated.
    """
    job = _load("release.yml")["jobs"]["release"]
    # `gh release upload` writes assets; the workflow-level grant is read-only.
    assert job["permissions"]["contents"] == "write"

    names = [step.get("name", "") for step in job["steps"]]
    verify_index = names.index("Verify built wheels cover all distribution namespaces")
    attach_index = names.index("Attach wheels to the release object")
    assert attach_index > verify_index, "the attach must follow the wheel verification"
    # Retention artifact and release attach are complementary, not exclusive.
    assert "Upload wheel artifacts" in names

    attach = job["steps"][attach_index]
    assert attach["env"]["GH_TOKEN"] == "${{ github.token }}"
    script = attach["run"]
    assert 'gh release upload "$GITHUB_REF_NAME" dist/*.whl --clobber' in script
    # A dispatch may run at a branch, and a tag may precede its release
    # object; both cases skip loudly via workflow annotations, never silently.
    assert '[ "$GITHUB_REF_TYPE" != "tag" ]' in script
    assert "::notice::" in script
    assert "::warning::" in script
    # The attach is verified against the build: a green step that uploaded
    # nothing is exactly the REL-02 failure mode.
    assert '"$attached" -lt "$built"' in script


def test_validate_job_runs_gates_once_outside_test_matrix():
    """GS-005: interpreter-independent gates live in one non-matrix validate job."""
    jobs = _load("ci.yml")["jobs"]
    assert "validate" in jobs
    assert "strategy" not in jobs["validate"], "gate job must not be a matrix leg"
    gate_text = _dump(jobs["validate"])
    for marker in (
        "validate_repo_contracts.py",
        "--strict-source-language --strict-import-smoke",
        "validate_skills.py --check-xrefs --warnings-fatal",
        "validate_test_contracts.py",
        "validate_model_contracts.py",
        "run_model_audit.py",
        "gitleaks detect",
        "check_coverage_floor.py",
        # CI-04: the four ruff surfaces (changed-file check, changed-file
        # format, src runtime hygiene, module-test hygiene).
        "xargs -0 uv run --with 'ruff>=0.15.6,<0.16' ruff check",
        "ruff format --check",
        "--select F821,F823,E721,E722",
        "--select F401,F841,F811,F823",
        # CI-04: the validator commands the marker list did not pin.
        "validate_packaging.py --strict",
        "validate_logging_hygiene.py",
        "validate_documentation.py --strict",
        "rewrite_readme_agents.py --check",
    ):
        assert marker in gate_text, f"missing gate marker: {marker}"

    test_text = _dump(jobs["test"])
    for marker in (
        "validate_repo_contracts.py",
        "gitleaks",
        "check_coverage_floor.py",
    ):
        assert marker not in test_text, f"gate duplicated in test legs: {marker}"


def test_coverage_floor_falls_back_to_empty_tree():
    """GS-004: an unresolvable base must widen the diff, not degrade to no-op."""
    gate_text = _dump(_load("ci.yml")["jobs"]["validate"])
    assert "git hash-object -t tree /dev/null" in gate_text


def test_tag_push_diff_gates_resolve_full_tree_not_head_parent():
    """CI-02: a v* tag push carries a zero event.before and no PR base; both
    diff-scope gates must branch to the GS-004 empty-tree (full-tree)
    measurement instead of silently degrading to HEAD^, whose scope covers
    only the tagged commit."""
    steps = _load("ci.yml")["jobs"]["validate"]["steps"]
    gate_scripts = [
        str(step["run"])
        for step in steps
        if isinstance(step, dict) and 'base_sha="$BASE_SHA"' in str(step.get("run", ""))
    ]
    assert len(gate_scripts) == 2
    tag_branch = (
        'if [ "$GITHUB_REF_TYPE" = "tag" ] || [[ "$base_sha" =~ ^0{40}$ ]]; then'
    )
    head_fallback = 'git rev-parse --verify "$HEAD_SHA^"'
    empty_tree = 'base_sha="$(git hash-object -t tree /dev/null)"'
    for script in gate_scripts:
        assert tag_branch in script, "tag/zero-base branch missing"
        branch_pos = script.index(tag_branch)
        # Zero-SHA/tag detection precedes both the 40-hex regex that would
        # otherwise swallow the zero SHA and the HEAD^ fallback.
        assert branch_pos < script.index("^[0-9a-f]{40}$")
        assert branch_pos < script.index(head_fallback)
        branch_body = script[branch_pos : script.index("elif", branch_pos)]
        assert empty_tree in branch_body
        assert head_fallback not in branch_body


def test_each_category_run_is_immediately_followed_by_its_own_retention():
    """GS-001: a category's reports upload before any later category wipes them."""
    steps = _load("ci.yml")["jobs"]["test"]["steps"]
    version_token = "${{ matrix.python-version }}"
    for index, step in enumerate(steps):
        name = step.get("name", "")
        if not (name.startswith("Run ") and name.endswith(" tests")):
            continue
        category = name[len("Run ") : -len(" tests")]
        retain = steps[index + 1].get("name", "")
        assert retain == f"Retain {category} test reports", (
            f"{name} is not immediately followed by its own retention step"
        )
        artifact = steps[index + 1]["with"]["name"]
        assert artifact == "geo-infer-" + category + "-reports-py" + version_token


def test_test_matrix_covers_all_categories_per_interpreter():
    """GS-005: every category is its own matrix leg for both interpreters."""
    matrix = _load("ci.yml")["jobs"]["test"]["strategy"]["matrix"]
    assert matrix["python-version"] == ["3.11", "3.12"]
    assert matrix["category"] == [
        "unit",
        "integration",
        "performance",
        "system",
        "h3",
    ]


def test_build_smoke_builds_wheels_in_pr_ci():
    """GS-006: the real wheel build executes on pull requests, not only on tags."""
    build = _dump(_load("ci.yml")["jobs"]["build-smoke"])
    assert "build_package_wheels.py" in build
    assert "--outdir" in build


def test_workflow_definitions_are_linted_in_ci():
    """GS-008: a workflow-lint job runs actionlint on the workflow sources."""
    lint = _dump(_load("ci.yml")["jobs"]["workflow-lint"])
    assert "actionlint" in lint


def test_gnn_interchange_only_runs_on_contract_relevant_changes():
    """GS-007: no unfiltered push/PR legs duplicate the paired interchange run."""
    trigger = _trigger(_load("gnn-interchange.yml"))
    assert "push" not in trigger
    paths = trigger["pull_request"]["paths"]
    assert "GEO-INFER-ACT/**" in paths
    assert "GEO-INFER-TEST/validate_gnn_interchange.py" in paths
    assert ".github/gnn-pair.json" in paths
    assert ".github/workflows/gnn-interchange.yml" in paths


def test_every_workflow_definition_parses():
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        assert isinstance(document, dict), workflow.name
        assert "jobs" in document, workflow.name


def test_pr_triggers_share_one_branch_policy():
    """GS19-21: PR legs target main/develop across all three CI workflows."""
    ci_branches = _trigger(_load("ci.yml"))["pull_request"]["branches"]
    assert ci_branches == ["main", "develop"]
    for name in ("gnn-interchange.yml", "import-probes.yml"):
        branches = _trigger(_load(name))["pull_request"]["branches"]
        assert branches == ci_branches, name


def test_release_job_is_timeout_bounded_and_release_queue_is_serialized():
    """GS19-22: the release job has a ceiling; tag races queue, not race."""
    release = _load("release.yml")
    assert release["jobs"]["release"].get("timeout-minutes") == 60
    concurrency = release["concurrency"]
    assert concurrency["group"] == "geo-infer-release-${{ github.ref }}"
    assert concurrency.get("cancel-in-progress", False) is False


def test_import_probes_derive_pytest_from_the_workspace_lock():
    """GS19-23: no floating pytest pin; the version comes from uv.lock."""
    job = _dump(_load("import-probes.yml")["jobs"]["probes"])
    assert "steps.locked_tools.outputs.pytest_version" in job
    assert "pytest==8.4.2" not in job


def test_repo_wide_format_check_is_scheduled():
    """GS19-03: a periodic repo-wide ruff format --check surface exists."""
    document = _load("format-check.yml")
    triggers = _trigger(document)
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers
    assert "ruff format --check" in _dump(document["jobs"])


def test_generated_signposts_carry_ci_strict_flags():
    """GS19-02: rendered README/AGENTS commands mirror the CI invocations.

    The generator's embedded command blocks must not document flags weaker
    than the CI invocation (GS19-02): --strict-import-smoke and
    --warnings-fatal belong on the signpost commands exactly as in ci.yml.
    """
    spec = importlib.util.spec_from_file_location(
        "geo_infer_rewrite_readme_agents_for_contracts",
        REPO_ROOT / "GEO-INFER-TEST" / "rewrite_readme_agents.py",
    )
    assert spec is not None and spec.loader is not None
    rewriter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rewriter
    spec.loader.exec_module(rewriter)
    modules = {
        "GEO-INFER-TEST": rewriter.ModuleInfo(
            name="GEO-INFER-TEST",
            path=REPO_ROOT / "GEO-INFER-TEST",
            package="geo_infer_test",
            description="Unified testing framework",
            version="0.3.0",
            dependencies=[],
            source_files=1,
            test_files=1,
        ),
    }

    readme = rewriter.render_root_readme(modules, 1, 1)
    agents = rewriter.render_root_agents(modules)

    for rendered in (readme, agents):
        assert (
            "validate_repo_contracts.py --strict-source-language"
            " --strict-import-smoke" in rendered
        )
        assert "validate_skills.py --check-xrefs --warnings-fatal" in rendered
