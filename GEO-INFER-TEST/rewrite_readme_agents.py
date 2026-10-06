#!/usr/bin/env python3
"""Rewrite repository README.md and AGENTS.md files from repository facts.

The generated files are intentionally compact and operational. They avoid
roadmap language and only describe the current filesystem, package metadata,
test entry points, and public symbols discoverable from Python source files.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from _validator_common import discover_module_dirs, read_pyproject

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PREFIX = "GEO-INFER-"
MAX_SYMBOLS = 20
_TRACKED_FILES: set[Path] | None = None


@dataclass
class ModuleInfo:
    name: str
    path: Path
    package: str
    description: str
    version: str
    dependencies: list[str]
    source_files: int
    test_files: int


@dataclass
class _DirectoryIndex:
    root: Path
    files: dict[Path, list[Path]]
    children: dict[Path, set[Path]]


_DIRECTORY_INDEX: _DirectoryIndex | None = None


def _build_directory_index(inventory: set[Path] | frozenset[Path]) -> _DirectoryIndex:
    """Index each file and distinct ancestor once, without inventory rescans."""
    files: dict[Path, list[Path]] = {}
    children: dict[Path, set[Path]] = {}
    indexed: set[Path] = set()
    for path in inventory:
        directory = path.parent
        files.setdefault(directory, []).append(path)
        while directory not in indexed:
            indexed.add(directory)
            parent = directory.parent
            if parent == directory:
                break
            children.setdefault(parent, set()).add(directory)
            directory = parent
    return _DirectoryIndex(REPO_ROOT, files, children)


def _directory_index() -> _DirectoryIndex:
    """Reuse only a render's snapshot; standalone calls see current set edits."""
    if _DIRECTORY_INDEX is not None and _DIRECTORY_INDEX.root == REPO_ROOT:
        return _DIRECTORY_INDEX
    return _build_directory_index(tracked_files())


def git_ls_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [
        REPO_ROOT / line
        for line in result.stdout.splitlines()
        if line and (REPO_ROOT / line).is_file()
    ]


def tracked_files() -> set[Path]:
    """Return versioned and non-ignored worktree files, excluding deletions."""
    global _TRACKED_FILES
    if _TRACKED_FILES is None:
        _TRACKED_FILES = set(git_ls_files())
    return _TRACKED_FILES


def requirement_lines(module_dir: Path, limit: int = 12) -> list[str]:
    requirements = module_dir / "requirements.txt"
    if not requirements.exists():
        return []
    lines: list[str] = []
    for line in requirements.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
        if len(lines) >= limit:
            break
    return lines


def declared_dependencies(module_dir: Path, limit: int = 12) -> list[str]:
    """Return the module's declared runtime dependencies.

    ``[project.dependencies]`` in pyproject.toml is canonical. The
    requirements.txt mirror has drifted fleet-wide (stale floors, test-only
    entries, extras listed as runtime), so it is only a fallback for modules
    without a pyproject dependency list.
    """
    pyproject = read_pyproject(module_dir)
    deps = pyproject.get("project", {}).get("dependencies") or []
    if deps:
        return [str(dep) for dep in deps]
    return requirement_lines(module_dir, limit=limit)


def discover_modules() -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    for module_dir in discover_module_dirs(REPO_ROOT):
        pyproject = read_pyproject(module_dir)
        project = pyproject.get("project", {})
        package = str(project.get("name", module_dir.name.lower())).replace("-", "_")
        modules[module_dir.name] = ModuleInfo(
            name=module_dir.name,
            path=module_dir,
            package=package,
            description=str(project.get("description", module_dir.name)),
            version=str(project.get("version", "unversioned")),
            dependencies=declared_dependencies(module_dir),
            source_files=len(list((module_dir / "src").glob("**/*.py"))),
            test_files=len(
                {
                    *(module_dir / "tests").glob("**/test_*.py"),
                    *(module_dir / "tests").glob("**/*_test.py"),
                }
            ),
        )
    return modules


def module_for(path: Path, modules: dict[str, ModuleInfo]) -> ModuleInfo | None:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        return None
    if not relative.parts:
        return None
    return modules.get(relative.parts[0])


def repository_doc_files() -> tuple[list[Path], list[Path]]:
    """Return tracked and newly added repository signposts.

    New directory-level signposts are intentionally visible before a commit so
    the generator can render and validate them in the same working tree. Other
    untracked files remain excluded from the generated inventories.
    """
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "README.md",
            "**/README.md",
            "AGENTS.md",
            "**/AGENTS.md",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    files = {
        REPO_ROOT / line
        for line in result.stdout.splitlines()
        if Path(line).name in {"README.md", "AGENTS.md"}
        and (REPO_ROOT / line).is_file()
    }

    readmes = sorted(path for path in files if path.name == "README.md")
    agents = sorted(path for path in files if path.name == "AGENTS.md")
    return readmes, agents


def direct_contents(directory: Path) -> tuple[list[str], list[str], list[str]]:
    index = _directory_index()
    direct_files = index.files.get(directory, ())
    child_directories = index.children.get(directory, ())
    dirs = sorted(
        child.name + "/"
        for child in directory.iterdir()
        if child.is_dir()
        and child.name not in {".git", ".venv", "__pycache__"}
        and child in child_directories
    )
    py_files = sorted(path.name for path in direct_files if path.suffix == ".py")
    other_files = sorted(
        path.name
        for path in direct_files
        if path.name not in {"README.md", "AGENTS.md"} and path.suffix != ".py"
    )
    # Signposts are generated from the complete tracked directory contents.
    # Truncating this inventory made valid files disappear from AGENTS.md and
    # README.md while the freshness check still passed.
    return dirs, py_files, other_files


def public_symbols(directory: Path) -> list[str]:
    """Read public APIs, respecting the non-ignored snapshot during rendering."""
    symbols: list[str] = []
    candidates = (
        _directory_index().files.get(directory, ())
        if _DIRECTORY_INDEX is not None and _DIRECTORY_INDEX.root == REPO_ROOT
        else directory.glob("*.py")
    )
    for py_file in sorted(path for path in candidates if path.suffix == ".py"):
        if py_file.name.startswith("test_") or (
            py_file.name.startswith("_") and py_file.name != "__init__.py"
        ):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                symbols.append(f"`{py_file.name}:{node.name}` ({kind})")
                if len(symbols) >= MAX_SYMBOLS:
                    return symbols
    return symbols


def purpose_for(path: Path, module: ModuleInfo | None) -> str:
    rel = path.parent.relative_to(REPO_ROOT)
    if rel == Path("."):
        return "Repository root for the GEO-INFER geospatial inference framework."
    if module and path.parent == module.path:
        return module.description.rstrip(".") + "."
    parts = rel.parts
    folder = parts[-1].replace("_", " ").replace("-", " ")
    if module:
        return f"{folder.title()} workspace within `{module.name}`."
    return f"{folder.title()} workspace within GEO-INFER."


def test_command(path: Path, module: ModuleInfo | None) -> str:
    if path.parent == REPO_ROOT:
        return "uv run python GEO-INFER-TEST/run_unified_tests.py --category unit"
    if module and path.parent == module.path:
        return f"uv run python GEO-INFER-TEST/run_unified_tests.py --module {module.name.removeprefix(MODULE_PREFIX)}"
    if "tests" in path.parent.parts:
        test_files = [
            candidate
            for candidate in _directory_index().files.get(path.parent, ())
            if candidate.suffix == ".py"
            and (
                candidate.name.startswith("test_")
                or candidate.name.endswith("_test.py")
            )
        ]
        if not test_files and module:
            return (
                "uv run python GEO-INFER-TEST/run_unified_tests.py --module "
                f"{module.name.removeprefix(MODULE_PREFIX)}"
            )
        rel = path.parent.relative_to(REPO_ROOT)
        return f"uv run python -m pytest {rel}"
    if module:
        return f"uv run python GEO-INFER-TEST/run_unified_tests.py --module {module.name.removeprefix(MODULE_PREFIX)}"
    return "uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke"


def validation_commands(path: Path, module: ModuleInfo | None) -> str:
    """Return the validation command block for a documentation file."""
    commands: list[str] = []
    if module and module.name == "GEO-INFER-TEST":
        commands.append("uv sync --all-packages --all-extras --all-groups")
    commands.append(test_command(path, module))
    return "\n".join(commands)


def module_readme_notes(path: Path, module: ModuleInfo | None) -> str:
    """Return implemented module-specific README notes for module roots."""
    if not module or path.parent != module.path:
        return ""
    if module.name == "GEO-INFER-TEST":
        return """
## Strict Testing Contracts

- `src/geo_infer_test/testing.py` exports deterministic RNG, local filesystem,
  HTTP, SQLite, and service fixtures plus finite/probability/matrix/model and
  visualization-manifest assertions.
- `validate_test_contracts.py --strict` validates every module inventory,
  primary marker, forbidden pytest control, syntax tree, and behavior-test
  docstring.
- `validate_documentation.py --strict` validates the maintained documentation
  hub's relative links and rejects known stale current-state claims.
- `validate_model_contracts.py` checks representative ACT model contracts;
  `run_model_audit.py` emits finite statistics, a PNG visualization, SHA-256
  sidecars, and a deterministic manifest.
"""
    if module.name == "GEO-INFER-ACT":
        return """
## Implemented H3 Active Inference Contracts

- ACT uses `inferactively-pymdp==1.0.3` through
  `geo_infer_act.utils.pymdp_adapter` for categorical H3 active-inference
  runtime paths. H3 runtime cells are validated with real `h3>=4.5.0,<5`.
- Flat H3 APIs remain available through `GenerativeModel.enable_h3_spatial`,
  `GenerativeModel.update_h3_beliefs`, `ActiveInferenceModel.infer_over_h3_grid`,
  `SpatialActiveInferenceAgent.step`, and `simulate_h3_lattice`.
- Research trace APIs are available through
  `GenerativeModel.compute_h3_cell_diagnostics`,
  `ActiveInferenceModel.trace_over_h3_grid`,
  `ActiveInferenceModel.trace_over_nested_h3_grid`,
  `SpatialActiveInferenceAgent.trace_step`, and
  `SpatialActiveInferenceAgent.trace_nested_step`.
- Nested H3 APIs are opt-in through `enable_nested_h3_spatial`,
  `update_nested_h3_beliefs`, `infer_over_nested_h3_grid`,
  `SpatialActiveInferenceAgent.step_nested`, and
  `MultiAgentModel.simulate_nested_h3_lattice`.
- H3 diagnostics use `H3CellDiagnostics`, `H3EdgeDiagnostics`,
  `H3LevelDiagnostics`, and `SpatialInferenceTrace`; nested results use
  `NestedH3LevelSummary`, `NestedH3BeliefUpdateResult`, and
  `NestedH3GridInferenceResult` from `geo_infer_act`.
- Nested runner mode is enabled with `RunConfig.parameters["nested_h3"] = True`
  and emits `data/h3_hierarchy.csv`, `data/nested_h3_diagnostics.json`,
  `data/nested_h3_parent_child_diagnostics.csv`, and
  `visualizations/nested_h3_hierarchy_map.html`.
- Flat and nested H3 runner outputs include pymdp diagnostics in
  `data/pymdp_h3_diagnostics.json`, `data/pymdp_policy_posteriors.csv`, and
  `visualizations/pymdp_policy_free_energy.html`.
- Flat and nested H3 runner outputs also include
  `data/spatial_inference_trace.json`, `data/spatial_research_statistics.json`,
  `data/h3_cell_diagnostics.csv`, `data/h3_edge_diagnostics.csv`,
  `visualizations/h3_belief_flux_map.html`, `visualizations/h3_policy_surface.html`,
  `visualizations/h3_policy_transitions.html`,
  `visualizations/h3_spatial_autocorrelation.html`,
  `visualizations/h3_entropy_free_energy_phase.html`, and
  `visualizations/spatial_inference_research_report.html`.
- Research-profile runs are opt-in through
  `RunConfig.parameters["research_profile"] = True` or
  `geo-infer-act-run --research-profile`; they keep real H3 geometry and
  `inferactively-pymdp==1.0.3` while using deterministic offline spatial
  fields that avoid collapsed policy and entropy traces.
- Generate the deterministic four-run gallery with
  `uv run python GEO-INFER-ACT/examples/spatial_active_inference_gallery.py`.
  The supported runtime is `uv run`; system Python may contain older pymdp
  distributions and is not a valid H3 runtime contract.
- Optional Python model-source integrations (Bayeux, PyMC, and Pyro) require
  `config["allow_dynamic_code"] = True` and execute in per-call namespaces.
- Core inference utilities validate finite probability inputs, use local RNG
  instances for reproducible categorical sampling, and apply solve-based
  Joseph-form Gaussian updates. `PolicySelector` accepts
  `expected_posterior`/`posterior_beliefs` for KL information gain.
- `VariationalInference.structured_update` consumes explicit categorical factor
  tables with `variables` and `potential`/`values`/`table` fields, and
  `MultiAgentModel.step` runs a perception-action-resource cycle with optional
  movement and harvest fields in action dictionaries.

```python
import numpy as np
from geo_infer_act import ActiveInferenceModel, GenerativeModel

model = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
model.enable_nested_h3_spatial([7, 8, 9], cells=["89283082803ffff"])

agent = ActiveInferenceModel(model_type="categorical")
agent.set_generative_model(model)
result = agent.infer_over_nested_h3_grid(
    {model.h3_cells[0]: np.array([1.0, 0.0, 0.0, 0.0])},
    return_result=True,
)
trace = agent.trace_over_nested_h3_grid(
    {model.h3_cells[0]: np.array([1.0, 0.0, 0.0, 0.0])},
    grid_result=result,
)
```

## Visualization Contracts

- Belief, policy, free-energy, hierarchical, and H3-grid plots validate finite
  aligned inputs, preserve caller-supplied figure sizes, and avoid changing
  process-wide matplotlib or seaborn state.
- H3 static and animated outputs create their parent directories before writing;
  constant-valued grids still receive a valid color scale.

Nested validation command:

```bash
uv run pytest GEO-INFER-ACT/tests/unit/test_nested_h3_active_inference.py -q
uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py
uv run python GEO-INFER-TEST/validate_act_geospatial_contract.py
```
"""
    if module.name == "GEO-INFER-SPACE":
        return """
## Implemented Nested H3 Contracts

- `geo_infer_space.nested.NestedH3Grid` builds real `h3>=4.5.0,<5`
  hierarchies from seed cells or boundary vertices across ordered resolutions.
- Hierarchy outputs include deterministic `parent_child_map`,
  `child_parent_map`, `same_level_neighbors`, level summaries, validation
  diagnostics, and finite child-to-parent aggregation.
- Validation rejects invalid H3 cells, unordered resolutions, orphan children,
  wrong-resolution children, and parent/child mismatches.

```python
from geo_infer_space.nested import NestedH3Grid

grid = NestedH3Grid("sf_nested")
hierarchy = grid.build_h3_hierarchy_from_cells(
    ["89283082803ffff"],
    resolutions=[7, 8, 9],
)
assert hierarchy["validation"]["is_valid"]
assert hierarchy["validation"]["orphan_count"] == 0
```

## Visualization Contracts

- The visualization engine validates H3 resolution and finite geographic bounds
  at construction, and validates dashboard result/configuration mappings.
- Dashboard configuration honors validated `zoom_start` and `tiles` values.

Nested validation command:

```bash
uv run pytest GEO-INFER-SPACE/tests/unit/test_nested_h3_contract.py -q
uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py
```
"""
    visualization_notes = {
        "GEO-INFER-ACT": """
## Visualization Contracts

- Belief, policy, free-energy, hierarchical, and H3-grid plots validate finite
  aligned inputs, preserve caller-supplied figure sizes, and avoid changing
  process-wide matplotlib or seaborn state.
- H3 static and animated outputs create their parent directories before writing;
  constant-valued grids still receive a valid color scale.
""",
        "GEO-INFER-APP": """
## Visualization Contracts

- Agent map features validate finite longitude/latitude values and geographic
  bounds, and normalize metadata to JSON-safe values.
- Active-inference prediction and reinforcement-learning reward series are
  exposed in dashboard widgets when present in agent metadata.
""",
        "GEO-INFER-ART": """
## Visualization Contracts

- Map styling validates alpha and line-width values before applying them.
- Animation and multi-scale rendering validate nonempty supported styles,
  positive timing values, and use an immutable-safe default scale selection.
""",
        "GEO-INFER-BAYES": """
## Visualization Contracts

- Spatial prediction, uncertainty, posterior, and model-comparison plots
  validate finite aligned numeric inputs and confidence levels.
- Spatial prediction handles the single-panel case when uncertainty is omitted
  and returns a valid figure for both single- and multi-panel layouts.
""",
        "GEO-INFER-BIO": """
## Visualization Contracts

- Biological plotting helpers validate required columns, finite coordinates and
  values, sequence inputs, and geographic bounds before rendering.
- Plot helpers return their Matplotlib figure and create nested output parents;
  saved figures are closed after writing to avoid leaking global figure state.
""",
        "GEO-INFER-COG": """
## Visualization Contracts

- Human-centered visualization IDs are deterministic per visualizer instance,
  and proximity thresholds and color counts are validated at construction.
- Proximity grouping uses connected components and similarity grouping returns
  explicit geometry groups with confidence metadata.
""",
        "GEO-INFER-ECON": """
## Visualization Contracts

- Economic chart and map inputs validate nonempty finite numeric data, and
  figures save to nested output paths without mutating global plot style.
- Diagnostics handle absent optional metrics safely; dashboard HTML is written
  when an output path is provided.
""",
        "GEO-INFER-IOT": """
## Visualization Contracts

- Sensor and interpolation maps validate finite WGS84 coordinates and aligned
  value arrays, and all HTML/image writers create nested output parents.
- Saved Matplotlib figures are closed after writing to avoid leaking figure
  state across monitoring cycles.
""",
        "GEO-INFER-LOG": """
## Visualization Contracts

- Route plotting validates geographic coordinates and preserves the supplied
  path geometry; network highlighting rejects unknown nodes.
- Interactive map zoom and optional map basemaps are validated/guarded so
  plotting remains usable without contextily network access.
""",
        "GEO-INFER-PLACE": """
## Visualization Contracts

- The visualization engine validates H3 resolution and finite geographic bounds
  at construction, and validates dashboard result/configuration mappings.
- Dashboard configuration honors validated `zoom_start` and `tiles` values.
""",
        "GEO-INFER-SPACE": """
## Visualization Contracts

- The visualization engine validates H3 resolution and finite geographic bounds
  at construction, and validates dashboard result/configuration mappings.
- Dashboard configuration honors validated `zoom_start` and `tiles` values.
""",
        "GEO-INFER-SPM": """
## Random Field Theory Contracts

- `RandomFieldTheory` computes the complete Gaussian Euler characteristic from
  zero- through top-dimensional resel counts and exposes full-EC peak FWE
  thresholds.
- Cluster inference labels excursion components, measures extent in resels,
  and returns Poisson-clumping maximum-cluster FWE p-values; the default
  cluster-forming Gaussian tail is one-sided `p=0.001`.

## Visualization Contracts

- Statistical and interactive maps reject invalid contrast indices, empty or
  non-finite coordinates/statistics, and misaligned significance arrays.
- Diagnostic leverage and Cook's distance use a numerically stable hat-matrix
  calculation, and the package-level interactive-map export is unambiguous.
""",
        "GEO-INFER-TIME": """
## Visualization Contracts

- Temporal plotting helpers validate nonempty finite series, aligned timestamps,
  confidence bounds, and anomaly indices before rendering.
- Figure creation and saving remain scoped to each call, preserving reusable
  style configuration without leaking global matplotlib state.
""",
    }
    return visualization_notes.get(module.name, "")


def module_agent_notes(path: Path, module: ModuleInfo | None) -> str:
    """Return implemented module-specific AGENTS notes for module roots."""
    if not module or path.parent != module.path:
        return ""
    if module.name == "GEO-INFER-TEST":
        return """
## Strict Testing Contracts

- Reuse `geo_infer_test.testing` fixtures and assertions for local boundaries,
  model contracts, and visualization artifacts.
- Missing dependencies, unavailable backends, warnings, skips, xfails, and
  empty selections are failures; do not hide them with warning filters or
  conditional pytest controls.
- Keep `validate_test_contracts.py`, `validate_model_contracts.py`, and
  `run_model_audit.py` synchronized with the documented commands and output
  schemas.
- Keep `validate_documentation.py` synchronized with the maintained
  authoritative documentation paths when the hub moves.
"""
    if module.name == "GEO-INFER-ACT":
        return """
## Current H3 Contracts

- Production ACT H3 runtime paths must use
  `geo_infer_act.utils.pymdp_adapter` and fail if `inferactively-pymdp` is not
  exactly `1.0.3`.
- Keep flat H3 method signatures and dictionary return shapes backward-compatible.
- Keep trace diagnostics JSON-safe and backed by typed result classes:
  `H3CellDiagnostics`, `H3EdgeDiagnostics`, `H3LevelDiagnostics`, and
  `SpatialInferenceTrace`.
- Use nested methods only for opt-in nested H3 behavior:
  `enable_nested_h3_spatial`, `update_nested_h3_beliefs`,
  `infer_over_nested_h3_grid`, `trace_over_nested_h3_grid`, `step_nested`,
  `trace_nested_step`, and `simulate_nested_h3_lattice`.
- Nested inference must reject invalid cells, observations outside the enabled
  hierarchy, mixed unexpected resolutions, non-finite observations, and empty
  belief vectors.
- Runner nested mode must keep outputs under the configured output directory and
  list generated hierarchy, diagnostics, and visualization files in the manifest.
- Runner flat and nested H3 modes must emit pymdp posterior, negative-EFE,
  free-energy, entropy, and backend-version diagnostics through manifest-linked
  sidecars.
- Runner flat and nested H3 modes must emit spatial trace JSON/CSV outputs plus
  belief-flux, policy-surface, policy-transition, spatial-autocorrelation,
  entropy/free-energy phase, and research-report HTML visualizations through
  the manifest sidecar pipeline.
- Research-profile and gallery runs must remain deterministic, real-H3, and
  non-degenerate: policy probabilities, entropy, local coherence, and belief
  flux should show finite variation above the validator thresholds.
- Use `uv run` for ACT/H3 commands. A system Python installation with a legacy
  `inferactively-pymdp` distribution is outside the supported contract.
- `create_h3_spatial_model` enforces a default 100,000-cell budget; callers
  with intentionally larger domains must pass `config["max_cells"]` explicitly.
- `infer_over_h3_grid` is read-only with respect to the attached generative
  model; preserve this contract when adding grid diagnostics.
- Optional Python model-source integrations (Bayeux, PyMC, and Pyro) require
  `config["allow_dynamic_code"] = True` and execute in per-call namespaces;
  keep this opt-in boundary when adding integrations.
- Preserve the finite-input, solve-based Gaussian, factor-table variational,
  policy information-gain, and multi-agent perception/action contracts when
  extending ACT core behavior.

## Failure Triage

- If nested belief tests fail, inspect `geo_infer_act.core.generative_model`
  before changing agents or runners.
- If artifact validation fails, inspect `geo_infer_act.runners.scenarios` and
  `geo_infer_act.runners.io` together.
- Re-run `uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration`
  after changing any H3 or nested spatial inference path.

## Visualization Guidance

- Keep plotting inputs finite and shape-aligned; use local figure styling so a
  library call cannot mutate the caller's matplotlib or seaborn configuration.
- H3 visualization writers must create output directories and handle constant
  metric ranges without emitting invalid colorbar limits.
"""
    if module.name == "GEO-INFER-SPACE":
        return """
## Current Nested H3 Contracts

- `NestedH3Grid` owns H3 parent/child closure, validation, same-resolution
  neighbor maps, and child-to-parent aggregation.
- Build hierarchies with ordered real `h3>=4.5.0,<5` resolutions and
  deterministic cell ordering; do not pass synthetic cell IDs to nested H3
  paths.
- Validate orphan counts, parent/child membership, resolution matches, and
  finite aggregate values before handing hierarchies to ACT.

## Failure Triage

- If hierarchy validation fails, inspect
  `geo_infer_space.nested.core.nested_grid.NestedH3Grid` first.
- If ACT nested contracts fail after SPACE edits, run the SPACE nested unit test
  and then `uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py`.

## Visualization Guidance

- Validate H3 resolution, bounds, dashboard result mappings, and user-facing map
  options before building a dashboard.
"""
    if module.name == "GEO-INFER-DATA":
        return """
## Data Boundary Contracts

- Persistent cache filenames are derived from SHA-256 digests of logical keys;
  never reconstruct cache paths directly from caller-provided strings.
- Cache timestamps are normalized to UTC, and `ttl=0` means immediate expiry.
- Large DataFrame compression uses in-memory Parquet via a file-like reader;
  preserve this round-trip behavior when changing serializers.
- Temporal validators accept both timezone-naive and timezone-aware pandas
  datetime columns without mixing comparison timezones.
"""
    if module.name == "GEO-INFER-API":
        return """
## GeoJSON Contracts

- GeoJSON positions must be finite WGS84 longitude/latitude values.
- Polygon bbox filtering uses geometry extents, so containing and crossing
  polygons are not missed when no vertex lies inside the query bbox.
"""
    if module.name == "GEO-INFER-OPS":
        return """
## Cache Contracts

- Redis cache clearing must always execute an initial SCAN at cursor `0` and
  continue until Redis returns cursor `0` again.
"""
    if module.name == "GEO-INFER-EXAMPLES":
        return """
## Workflow Guard Contracts

- Conditional workflow expressions use the constrained data-only evaluator;
  function calls, imports, private attributes, and executable expressions are
  rejected.
"""
    visualization_notes = {
        "GEO-INFER-ACT": """
## Visualization Guidance

- Keep plotting inputs finite and shape-aligned; use local figure styling so a
  library call cannot mutate the caller's matplotlib or seaborn configuration.
- H3 visualization writers must create output directories and handle constant
  metric ranges without emitting invalid colorbar limits.
""",
        "GEO-INFER-APP": """
## Visualization Guidance

- Validate geographic coordinates before creating map features and keep emitted
  metadata JSON-safe for downstream GeoJSON/dashboard clients.
- Preserve dashboard widget schemas when adding agent state visualizations.
""",
        "GEO-INFER-ART": """
## Visualization Guidance

- Validate style alpha, line width, animation timing, and supported scale names
  at the public API boundary.
- Do not use mutable defaults for scale/style collections.
""",
        "GEO-INFER-BAYES": """
## Visualization Guidance

- Validate finite aligned spatial arrays and confidence levels before plotting.
- Normalize single-axis layouts before indexing axes so optional uncertainty
  panels work for one and many spatial predictions.
""",
        "GEO-INFER-BIO": """
## Visualization Guidance

- Keep spatial plotting inputs nonempty, finite, geographically bounded, and
  explicit about required columns; return figures for programmatic inspection.
- Create nested output directories before saving and close saved figures.
""",
        "GEO-INFER-COG": """
## Visualization Guidance

- Keep visualization IDs reproducible within a visualizer instance and validate
  cognitive thresholds/color counts at construction.
- Return meaningful grouping clusters and confidence metadata rather than
  placeholder groups.
""",
        "GEO-INFER-ECON": """
## Visualization Guidance

- Validate finite nonempty chart inputs and use call-local figure saving; do not
  mutate process-wide matplotlib/seaborn style from a visualizer constructor.
- Keep dashboard output paths operational and format optional metrics safely.
""",
        "GEO-INFER-IOT": """
## Visualization Guidance

- Validate sensor/interpolation coordinates and aligned values before passing
  data to Folium or Matplotlib; create parents before writing artifacts.
- Close saved figures and preserve explicit error dictionaries for invalid input.
""",
        "GEO-INFER-LOG": """
## Visualization Guidance

- Validate route points, zoom bounds, and network highlight nodes at public
  boundaries; draw route lines from the ordered input path, not its hull.
- Guard optional basemap integrations when contextily is unavailable.
""",
        "GEO-INFER-PLACE": """
## Visualization Guidance

- Validate H3 resolution, bounds, dashboard result mappings, and user-facing map
  options before building a dashboard.
""",
        "GEO-INFER-SPACE": """
## Visualization Guidance

- Validate H3 resolution, bounds, dashboard result mappings, and user-facing map
  options before building a dashboard.
""",
        "GEO-INFER-SPM": """
## Random Field Theory Guidance

- Preserve every Gaussian EC density and boundary resel term in peak inference;
  do not replace the full expected EC with only its top-dimensional term.
- Keep cluster extent in resel units, correct the maximum-cluster probability,
  and verify peak and cluster FWE with deterministic known-null simulations.

## Visualization Guidance

- Reject invalid contrast/statistic/coordinate inputs before map construction.
- Keep diagnostic leverage calculations numerically stable and maintain one
  canonical package-level interactive-map export.
""",
        "GEO-INFER-TIME": """
## Visualization Guidance

- Validate finite aligned series, timestamp lengths, confidence bounds, and
  anomaly indices at plotting boundaries; keep style changes call-local.
""",
    }
    return visualization_notes.get(module.name, "")


def render_root_readme(
    modules: dict[str, ModuleInfo], readme_count: int, agents_count: int
) -> str:
    source_files = sum(module.source_files for module in modules.values())
    test_files = sum(module.test_files for module in modules.values())
    module_rows = "\n".join(
        f"| `{module.name}` | `{module.package}` | {module.source_files} | {module.test_files} |"
        for module in modules.values()
    )
    theme_groups = [
        (
            "🌍 Spatial & Place-based",
            [
                "SPACE",
                "PLACE",
                "TIME",
                "MARINE",
                "WATER",
                "FOREST",
                "CLIMATE",
                "ENERGY",
                "TRANSPORT",
                "EMERGENCY",
            ],
        ),
        (
            "🧠 Bayesian & Active Inference",
            ["BAYES", "SIM", "SPM", "COG", "ACT", "MATH"],
        ),
        ("🤖 Agents & AI Orchestration", ["AGENT", "AG", "AI", "ANT", "OPS", "COMMS"]),
        (
            "🏛️ Governance, Risk & Domain",
            [
                "RISK",
                "INSURANCE",
                "METAGOV",
                "NORMS",
                "ECON",
                "PEP",
                "REQ",
                "SEC",
                "CIV",
                "HEALTH",
                "ORG",
            ],
        ),
        ("🗄️ Data, API & Applications", ["API", "APP", "DATA", "IOT", "ART", "EDU"]),
        (
            "🛠️ Infrastructure & Validation",
            ["INTRA", "TEST", "LOG", "GIT", "EXAMPLES", "BIO"],
        ),
    ]
    name_set = {m.name for m in modules.values()}
    themed = [f"GEO-INFER-{n}" for _, group in theme_groups for n in group]
    unthemed = sorted(name_set - set(themed))
    duplicated = sorted({n for n in themed if themed.count(n) > 1})
    if unthemed or duplicated:
        raise ValueError(
            "README theme groups must list every module exactly once; "
            f"missing={unthemed} duplicated={duplicated}"
        )
    theme_rows = []
    for icon, group in theme_groups:
        present = [f"GEO-INFER-{n}" for n in group if f"GEO-INFER-{n}" in name_set]
        theme_rows.append(f"| {icon} | `{'`, `'.join(present) if present else '—'}` |")
    theme_rows = "\n".join(theme_rows)
    return f"""# GEO-INFER Framework

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![uv workspaces](https://img.shields.io/badge/uv-workspace-4C65F6?logo=astral&logoColor=white)](pyproject.toml)
[![CI](https://img.shields.io/github/actions/workflow/status/ActiveInferenceInstitute/GEO_INFER/ci.yml?branch=main&label=CI)](.github/workflows/ci.yml)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)
[![Active Inference Institute](https://img.shields.io/badge/Active_Inference_Institute-6C3483?style=flat)](https://activeinference.org)

**GEO-INFER** is a {len(modules)}-module **geospatial inference monorepo** from the
Active Inference Institute — spatial analysis, Active Inference, Bayesian modeling,
domain modeling, agent workflows, and reproducible repository validation in one
`uv`/Python workspace.

> Build geospatial and place-based models, run Active-Inference and Bayesian
> inference over them, orchestrate agents and domain workflows, and keep the
> whole thing reproducible — served from a [user documentation hub](GEO-INFER-INTRA/docs/index.md)
> backed by an auto-generated, validation-gated [module catalog](GEO-INFER-INTRA/docs/modules/index.md).

## What's inside

- 🧭 **Spatial & place-based analysis** — geospatial data, H3 grids, place & time modeling, and Earth-system domains (water, marine, forest, climate, energy, transport, emergency).
- 🧠 **Active Inference & Bayesian modeling** — Active-Inference agents and Bayesian models (Bayes, simulation, SPM, cognition, math).
- 🤖 **Agent & AI orchestration** — agent workflows, AI/LLM integration, communications, and operations.
- 🏛️ **Governance, risk & domain modeling** — risk, meta-governance, norms, economics, policy, security, health, and civil domains.
- 🗄️ **Data, API & applications** — data pipelines, APIs, applications, IoT, art, and education.
- 🛠️ **Infrastructure & validation** — documentation hub (INTRA), the validation & test harness, logging, git, examples, and bio.

## Current Repository Facts

| Metric | Value |
| --- | ---: |
| Modules | {len(modules)} |
| Python source files | {source_files} |
| Python test files | {test_files} |
| Repository README.md files | {readme_count} |
| Repository AGENTS.md files | {agents_count} |

## Quick Start

```bash
uv sync --all-packages --all-extras --all-groups
python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke
uv run python GEO-INFER-TEST/validate_documentation.py --strict
uv run python manuscript/generate_research_artifacts.py
uv run python GEO-INFER-TEST/run_unified_tests.py --category unit
```

## Documentation Map

- User documentation hub: [`GEO-INFER-INTRA/docs/index.md`](GEO-INFER-INTRA/docs/index.md)
- Installation and first workflow: [`GEO-INFER-INTRA/docs/getting_started/index.md`](GEO-INFER-INTRA/docs/getting_started/index.md)
- Framework architecture: [`GEO-INFER-INTRA/docs/overview.md`](GEO-INFER-INTRA/docs/overview.md)
- Module catalog: [`GEO-INFER-INTRA/docs/modules/index.md`](GEO-INFER-INTRA/docs/modules/index.md)
- Developer workflow: [`GEO-INFER-INTRA/docs/developer_guide/index.md`](GEO-INFER-INTRA/docs/developer_guide/index.md)
- Test and release gates: [`GEO-INFER-TEST/docs/index.md`](GEO-INFER-TEST/docs/index.md)
- Manuscript generation and evidence: [`manuscript/README.md`](manuscript/README.md)
- Active Inference reference: [`GEO-INFER-INTRA/docs/active_inference_guide.md`](GEO-INFER-INTRA/docs/active_inference_guide.md)
- Spatial/H3 reference: [`GEO-INFER-INTRA/docs/geospatial/data_formats/h3/index.md`](GEO-INFER-INTRA/docs/geospatial/data_formats/h3/index.md)
- Contribution rules: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security reporting: [`SECURITY.md`](SECURITY.md)
- Release history: [`CHANGELOG.md`](CHANGELOG.md)

The repository root and module-level `README.md`/`AGENTS.md` files are generated
signposts. Conceptual tutorials, integration guidance, and policy live in the
INTRA documentation hub; executable behavior and public exports remain owned by
the module source and its tests.

## Choose an Installation Profile

The repository is a uv workspace. Use the full sync when working across module
boundaries, or sync a single package when developing one module:

```bash
uv sync --all-packages --all-extras --all-groups
uv sync --package geo-infer-act
uv sync --package geo-infer-space
uv sync --package geo-infer-ant
```

`--all-extras` installs optional scientific, Bayesian, web, IoT, performance,
quality, and documentation dependencies. CI intentionally omits native-only
extras that cannot build on its CPU runner; see `.github/workflows/ci.yml` for
the exact reproducible exception list.

## Module Themes

| Theme | Modules |
| --- | --- |
{theme_rows}

## Module Index

| Module | Package | Source files | Test files |
| --- | --- | ---: | ---: |
{module_rows}

## Modular Hygiene

- Root `pyproject.toml`, `uv.lock`, and `.python-version` are the canonical uv environment surfaces.
- Keep `uv.lock` as LF text through root `.gitattributes` so raw receipt hashes agree across platforms.
- Sync the full workspace with `uv sync --all-packages --all-extras --all-groups` before repo-wide validation.
- Each module owns importable behavior under `src/` and keeps at least four pytest files under `tests/`.
- Planned work belongs in root `TODO.md` or a tracked issue, not source or test task markers.
- Importable libraries use `logging.getLogger(__name__)`; process-wide logging configuration belongs in CLI entrypoints.

## Validation

- Repository contracts: `uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke`
- Documentation links and current-state claims: `uv run python GEO-INFER-TEST/validate_documentation.py --strict`
- Syntax gate: `python -m compileall GEO-INFER-*/src GEO-INFER-*/examples`
- Skill contracts: `uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal`
- Secrets: `gitleaks detect --source . --config .gitleaks.toml --redact --verbose` (local replication: GEO-INFER-TEST/docs/secret_scan_policy.md)
- Unit tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --category unit`
- Integration tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --category integration`
- System tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --category system`
- Performance tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --category performance`
- Coverage gate: `uv run python GEO-INFER-TEST/run_unified_tests.py --category coverage --timeout 900`
- H3 contracts: `uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration`
- Test contract: `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`
- Model contract: `uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42`
- Reproducible model audit: `uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible`
- Lint contract: `uv run --with 'ruff>=0.15.6,<0.16' ruff check .`
- Manuscript variables, figures, captions, and resolved copies: `uv run python manuscript/generate_research_artifacts.py`

## Repo-wide Change Workflow

1. Inspect the owning module and keep behavior in its `src/` package.
2. Add or update a focused test in the owning module's `tests/` directory.
3. Run the focused test, then compile and run the contract validators.
4. Refresh generated signposts with `uv run python GEO-INFER-TEST/rewrite_readme_agents.py`.
5. Confirm generated documentation is stable with `uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check`.

## Artifact and Output Hygiene

- Test reports belong under `.geo-infer-test-results/`.
- Model-audit artifacts are emitted under `.geo-infer-test-results/model-audit/`.
- The manuscript pipeline is the only approved writer to ignored repository-root
  `output/`; scenario and other visualization outputs must use an explicit
  output directory and must not write there.
- Generated signposts must describe tracked files only; local caches and build
  products are intentionally excluded.

README.md and AGENTS.md files below the repository root are generated signposts.
The generator derives their contents from tracked files, public symbols, module
metadata, validation commands, and test inventories; update the generator when
the documentation contract itself changes.

## Failure Triage

- `validate_repo_contracts.py`: source layout, language, dependency, logger, and documentation contract.
- `validate_logging_hygiene.py`: passive library-logging contract (no root-logger mutation; module-local `getLogger(__name__)` loggers only).
- `validate_test_contracts.py`: test inventories, markers, fixtures, skips, and warning policy.
- `run_unified_tests.py`: module behavior by unit, integration, performance, or H3 category.
- `validate_model_contracts.py` and `run_model_audit.py`: deterministic model outputs and reproducibility artifacts.
- `rewrite_readme_agents.py --check`: generated README/AGENTS drift; rerun the generator after intentional tracked-file changes.

## Zero-warning test policy

The shared pytest policy treats warnings as errors, requires strict markers/configuration, assigns exactly one primary marker to every test, and rejects skips, xfails, xpasses, collection errors, missing dependencies, missing fixtures, and empty selections. Every module has a test inventory at `GEO-INFER-*/tests/README.md`; the inventory records purpose, fixtures, dependencies, artifacts, and triage commands.

## Documentation Policy

README.md and AGENTS.md files describe current, discoverable repository state. Do not add aspirational APIs to these files unless the implementation, export path, and validation command exist in this checkout. Keep module-local public exports and test commands synchronized through the generator.
"""


def render_root_agents(modules: dict[str, ModuleInfo]) -> str:
    module_names = ", ".join(f"`{name}`" for name in modules)
    return f"""# GEO-INFER Agent Instructions

Use this file as the repository-level operating contract for automated agents working in GEO-INFER.

## Cold Start (agent orientation)

New to this repo? In order:

1. **What this is**: a multi-module geospatial inference monorepo (uv workspace, Python 3.11+); see [README.md](README.md) "What's inside".
2. **Module map**: [README.md Module Index](README.md#module-index) (source/test counts per module) and [GEO-INFER-INTRA/docs/modules/index.md](GEO-INFER-INTRA/docs/modules/index.md) (conceptual catalog).
3. **Where to change what**: module behavior in `GEO-INFER-*/src/` (owning package); cross-module docs in `GEO-INFER-INTRA/docs/`; tests/validation in `GEO-INFER-TEST/`.
4. **Health in one command**: `uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check` (generated-signpost drift) plus the validators under Standard Commands.
5. **Backlog with acceptance lines**: [TODO.md](TODO.md).
6. **GNN companion pin**: to bump the paired-interchange GNN revision, follow
   the [pair-pin bump procedure](GEO-INFER-TEST/docs/gnn_space_time_2026_09.md#gnn-pair-pin-bump-procedure).

## Repository Scope

- Root path: repository root (`.`; resolve it from the checkout in use)
- Modules: {module_names}
- Package manager: `uv`
- Python target: 3.11+

## Required Workflow

1. Inspect the relevant module before editing.
2. Keep functionality in the owning module under `src/`.
3. Keep scripts and examples as thin orchestration surfaces.
4. Update README.md and AGENTS.md when behavior, commands, exports, or dependencies change.
5. Run the narrowest relevant test first, then the repo contract validators.

## Standard Commands

```bash
uv sync --all-packages --all-extras --all-groups
python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke
uv run python GEO-INFER-TEST/validate_logging_hygiene.py
uv run python GEO-INFER-TEST/validate_documentation.py --strict
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal
uv run python manuscript/generate_research_artifacts.py
uv run python GEO-INFER-TEST/run_unified_tests.py --category unit
uv run python GEO-INFER-TEST/run_unified_tests.py --category integration
uv run python GEO-INFER-TEST/run_unified_tests.py --category performance
uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict
uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42
uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible
uv run --with 'ruff>=0.15.6,<0.16' ruff check .
gitleaks detect --source . --config .gitleaks.toml --redact --verbose
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
```

## Modular Hygiene Contract

- Use root `pyproject.toml`, `uv.lock`, and `.python-version` as the shared uv environment contract.
- Preserve `uv.lock` LF bytes through root `.gitattributes`; receipt custody uses raw file hashes on every platform.
- Preserve captured Cascadia publisher bytes and LF replay outputs through the exact paths in root `.gitattributes`; do not normalize hash-bound archives.
- Sync the shared workspace with `uv sync --all-packages --all-extras --all-groups`.
- Keep module behavior in the owning `GEO-INFER-*` package under `src/`; keep scripts and examples as orchestration surfaces.
- Treat `manuscript/generate_research_artifacts.py` as the only producer of manuscript variables, figure captions, figure registries, and resolved manuscript copies; never hand-edit ignored `output/`.
- Keep every module's local test inventory above the minimum release gate of four pytest files.
- Put planned work in root `TODO.md` or a tracked issue; do not leave task markers in module source or tests.
- Use module loggers in libraries and configure handlers only from CLI entrypoints.

## Documentation Contract

Agent-facing documentation must be operational: current paths, commands, package names, public exports, test surfaces, and failure triage. Do not advertise planned APIs in AGENTS.md; use issues, roadmaps, or implementation status files for future work.

## Documentation Workflow

- Put conceptual, cross-module, and user-facing guidance in
  `GEO-INFER-INTRA/docs/`.
- Keep module READMEs focused on the current filesystem, imports, dependencies,
  and verification commands; they are generated from tracked repository facts.
- Keep `SKILL.md` action-oriented and synchronized with real public APIs.
- Validate relative documentation links and generated signposts before handoff.
- Preserve existing work from other agents in a shared checkout; inspect the
  diff before staging generated documentation.

## Documentation and Release Gate

- Run `uv run python GEO-INFER-TEST/rewrite_readme_agents.py` after changing tracked module files, public exports, tests, or validation commands.
- Treat the generated README/AGENTS diff as a review surface: it should reflect the intended source, test, dependency, and signpost changes only.
- Before integrating to `main`, run the strict repository, test, model, skill, source-hygiene, unit, integration, performance, and H3 gates and record any environment-only setup warnings separately from repository failures.
"""


def render_readme(
    path: Path, module: ModuleInfo | None, modules: dict[str, ModuleInfo]
) -> str:
    if path.parent == REPO_ROOT:
        readmes, agents = repository_doc_files()
        return render_root_readme(modules, len(readmes), len(agents))

    rel = path.parent.relative_to(REPO_ROOT)
    dirs, py_files, other_files = direct_contents(path.parent)
    symbols = public_symbols(path.parent)
    title = rel.as_posix()
    content_lines = (
        "\n".join(f"- `{item}`" for item in dirs + py_files + other_files)
        or "- No direct tracked child entries."
    )
    symbol_lines = (
        "\n".join(f"- {symbol}" for symbol in symbols)
        or "- No public Python symbols are defined directly in this directory."
    )
    module_lines = ""
    if module:
        deps = (
            "\n".join(f"- `{dep}`" for dep in module.dependencies)
            or "- Dependencies are declared in `pyproject.toml` or inherited from the workspace."
        )
        install = f"uv sync --package {module.package.replace('_', '-')}"
        module_lines = f"""
## Module Metadata

- Module: `{module.name}`
- Package: `{module.package}`
- Version: `{module.version}`
- Install: `{install}`
- Tests: `{test_command(path, module)}`

## Dependencies

{deps}
"""

    test_inventory = ""
    if module and path.parent.name == "tests":
        test_inventory = f"""
## Strict Test Inventory

- Purpose: validate the `{module.name}` module's current behavior through unit,
  integration, system, and performance test surfaces.
- Primary marker: tests receive exactly one primary marker from their canonical
  directory; additive domain markers remain allowed.
- Required fixtures: local `tests/conftest.py` fixtures and shared
  `geo_infer_test.testing` fixtures for deterministic RNG, filesystem, HTTP,
  SQLite, service, model, and artifact boundaries.
- Dependencies: required test/runtime dependencies are installed by
  `uv sync --all-packages --all-extras --all-groups`; missing backends are failures.
- Expected artifacts: JUnit XML under `.geo-infer-test-results/`; model and
  visualization outputs require finite statistics, sidecars, hashes, and a
  manifest.
- Failure triage: `env -u VIRTUAL_ENV uv run pytest -c pyproject.toml -q
  {module.name}/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.
"""

    return f"""# {title}

{purpose_for(path, module)}

## Contents

{content_lines}

## Public Interface

{symbol_lines}
{module_lines}
{test_inventory}
## Validation

```bash
{validation_commands(path, module)}
```

{module_readme_notes(path, module)}
## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
"""


def render_agents(
    path: Path, module: ModuleInfo | None, modules: dict[str, ModuleInfo]
) -> str:
    if path.parent == REPO_ROOT:
        return render_root_agents(modules)

    rel = path.parent.relative_to(REPO_ROOT)
    dirs, py_files, other_files = direct_contents(path.parent)
    content_lines = (
        "\n".join(f"- `{item}`" for item in dirs + py_files + other_files)
        or "- No direct tracked child entries."
    )
    # Non-module directories own no Python package; state the honest role
    # instead of emitting a bogus package name (the generator fallback).
    if module:
        scope_lines = (
            f"- Owning module: `{module.name}`\n"
            f"- Python package: `{module.package}`\n"
            f"- Directory role: {purpose_for(path, module)}"
        )
        integration_line = (
            f"- Integrates through `{module.package}` and the owning "
            "module's public contracts."
        )
    else:
        scope_lines = (
            "- Owning module: none (repository-level directory outside the "
            "GEO-INFER-* module fleet)\n"
            f"- Directory role: {purpose_for(path, module)}"
        )
        integration_line = (
            "- Integrates through the repository's module-level public "
            "contracts; this directory defines no Python package."
        )
    return f"""# Agent Instructions: {rel.as_posix()}

## Scope

{scope_lines}

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
{integration_line}

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

{content_lines}

## Validation

```bash
{validation_commands(path, module)}
```

{module_agent_notes(path, module)}
## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
"""


def expected_doc_files() -> list[tuple[Path, str]]:
    """Return tracked documentation files and their generated contents."""
    global _DIRECTORY_INDEX
    previous_index = _DIRECTORY_INDEX
    _DIRECTORY_INDEX = _build_directory_index(frozenset(tracked_files()))
    try:
        modules = discover_modules()
        readmes, agents = repository_doc_files()
        expected: list[tuple[Path, str]] = []

        for readme in readmes:
            expected.append(
                (readme, render_readme(readme, module_for(readme, modules), modules))
            )
        for agents_file in agents:
            expected.append(
                (
                    agents_file,
                    render_agents(
                        agents_file, module_for(agents_file, modules), modules
                    ),
                )
            )

        return expected
    finally:
        _DIRECTORY_INDEX = previous_index


def check_docs_current() -> list[Path]:
    """Return generated documentation files whose tracked content is stale."""
    stale: list[Path] = []
    for path, expected_text in expected_doc_files():
        current = path.read_text(encoding="utf-8")
        if current != expected_text:
            stale.append(path)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated README.md or AGENTS.md files are stale.",
    )
    args = parser.parse_args()

    expected = expected_doc_files()
    if args.check:
        stale = [
            path
            for path, expected_text in expected
            if path.read_text(encoding="utf-8") != expected_text
        ]
        if stale:
            print(
                "Generated documentation is stale. First mismatches: "
                + ", ".join(str(path.relative_to(REPO_ROOT)) for path in stale[:20]),
                file=sys.stderr,
            )
            return 1
        print(f"All {len(expected)} generated README.md/AGENTS.md files are current.")
        return 0

    for path, content in expected:
        path.write_text(content, encoding="utf-8")

    readme_count = sum(1 for path, _ in expected if path.name == "README.md")
    agents_count = sum(1 for path, _ in expected if path.name == "AGENTS.md")
    print(f"Rewrote {readme_count} README.md files and {agents_count} AGENTS.md files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
