#!/usr/bin/env python3
"""Cascadia river homing: typed geospatial active inference, end to end.

An adult salmon returns from the Pacific to its natal stream, Rowdy Creek, a
tributary of the lower Smith River in Del Norte County, California. The
pipeline has five stages; each writes manifest-listed artifacts:

1. compile: trace the retained NHDPlus HR lower Smith River flow lines onto H3
   resolution 8, select a typed sea/main/slough/tributary/natal network, and
   compile exact dyadic A-E matrices (:mod:`geo_infer_act.models.river_homing`);
2. gnn: render the GNN v1 Markdown source. With ``--gnn-repo`` the source is
   exported by GNN's own strict ``gnn.export.geo_infer`` in GNN's environment;
   otherwise the committed artifact fixture is checked against the source;
3. infer: run the closed loop through ``run_gnn_inference`` (pymdp 1.0.3);
4. analyze: EFE terms, the pymdp identity gap, the light cone, homing events;
5. figures: context and network maps, traces, EFE panels, belief animation.

Run from the GEO-INFER repository root::

    uv run python GEO-INFER-ACT/examples/cascadia/cascadia_homing.py
    uv run python GEO-INFER-ACT/examples/cascadia/cascadia_homing.py \\
        --gnn-repo ../Generalized_Notation_Notation \\
        --gnn-python ../Generalized_Notation_Notation/.venv/bin/python
"""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any
from collections.abc import Mapping, Sequence

import h3
import numpy as np
import yaml

from geo_infer_act.core.efe_decomposition import support
from geo_infer_act.core.gnn_contract import GNNArtifact
from geo_infer_act.core.h3_flow_network import (
    FlowNetworkSelection,
    H3FlowNetwork,
    compile_flow_network,
)
from geo_infer_act.models.river_homing import (
    ACTIONS,
    HABITATS,
    ODORS,
    RiverHomingModel,
    RiverHomingParameters,
    compile_river_homing,
)
from geo_infer_act.runners.closed_loop import (
    ClosedLoopResult,
    run_closed_loop,
    trace_diagnostics,
)
from geo_infer_act.utils import flow_network_visualization as viz

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIR / "cascadia_homing.yaml"
DEFAULT_OUTPUT = EXAMPLE_DIR.parent / "outputs" / "cascadia_homing"
STEM = "smith_river_homing"
FIXTURE_NAMES = {
    "source": f"{STEM}.gnn.md",
    "state_ids": f"{STEM}.state_ids.json",
    "domain": f"{STEM}.domain.json",
    "artifact": f"{STEM}.geo-infer.json",
    "export": f"{STEM}.export.json",
}
LANDMARK_LABELS = {
    "sea": "Pacific Ocean",
    "outlet": "Smith River mouth",
    "decoy_confluence": "Tryon Creek joins",
    "natal_confluence": "Rowdy Creek joins",
    "home": "home: upper Rowdy Creek",
}
ANNOTATION = """\
Adult salmon homing on the lower Smith River (Del Norte County, California) as
one-factor categorical active inference over H3 resolution-8 cells. Hidden
state: the cell the fish occupies, in state_ids order (sea cell, main stem from
the mouth upstream, estuary sloughs, lower Tryon Creek, lower Rowdy Creek).
Observation: one of 12 outcomes, habitat cue (marine, estuary, river, creek)
times natal-odor cue (none, faint, strong), index = 3 * habitat + odor.
Actions: hold, downstream, upstream (continue the current stream), turn (enter
the stream joining at this cell). Cells and flow relations are traced from the
measured USGS NHDPlus HR excerpt retained by GEO-INFER-PLACE; habitat classes,
odor levels, movement probabilities and preferences are declared assumptions,
not fitted to telemetry. Every probability is an exact dyadic fraction."""


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of a file's bytes."""
    return sha256_bytes(Path(path).read_bytes())


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML config and resolve data paths against its directory."""
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "geo-infer-act/cascadia-homing/1":
        raise ValueError("Unsupported Cascadia homing config schema")
    base = config_path.parent
    config["data"] = {
        key: (base / value).resolve() for key, value in config["data"].items()
    }
    config["_path"] = config_path
    config["_sha256"] = sha256_file(config_path)
    return config


def load_flowlines(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the flow lines after checking the retained manifest digest."""
    flowlines_path = config["data"]["flowlines"]
    manifest = json.loads(Path(config["data"]["manifest"]).read_text(encoding="utf-8"))
    digest = sha256_file(flowlines_path)
    if manifest.get("status") != "complete" or manifest.get("sha256") != digest:
        raise ValueError("Flow-line bytes do not match their retained manifest")
    features = json.loads(Path(flowlines_path).read_text(encoding="utf-8"))["features"]
    if len(features) != manifest.get("feature_count"):
        raise ValueError("Flow-line feature count does not match the manifest")
    provenance = {
        "dataset": manifest["dataset"],
        "sha256": digest,
        "feature_count": len(features),
        "source_url": manifest["source_url"],
        "selection": manifest["selection"],
        "acquired_at": manifest["acquired_at"],
    }
    return features, provenance


def build_model(
    config: Mapping[str, Any],
) -> tuple[RiverHomingModel, list[dict[str, Any]], dict[str, Any]]:
    """Compile the typed network and exact homing model from the config."""
    features, provenance = load_flowlines(config)
    network = compile_flow_network(features, FlowNetworkSelection(**config["network"]))
    model = compile_river_homing(
        network, RiverHomingParameters.from_mapping(config["parameters"])
    )
    return model, features, provenance


def render_sources(
    model: RiverHomingModel, config: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, str]:
    """Return the GNN source, state-ID list and domain sidecar texts."""
    spec = model.gnn_spec(
        model_name=config["model_name"],
        annotation=ANNOTATION,
        footer="Cascadia river homing - GEO-INFER-ACT example; see GEO-INFER-ACT/docs/cascadia_river_homing.md.",
    )
    source = spec.render()
    domain = model.domain_record()
    domain["gnn_source_sha256"] = sha256_bytes(source.encode("utf-8"))
    domain["flowlines"] = dict(provenance)
    domain["network_selection"] = dict(config["network"])
    return {
        "source": source,
        "state_ids": json.dumps(list(model.network.cells), indent=1) + "\n",
        "domain": json.dumps(domain, indent=1, sort_keys=True) + "\n",
    }


def export_with_gnn(
    source: Path,
    state_ids: Path,
    artifact: Path,
    *,
    gnn_repo: Path,
    gnn_python: Path,
    step_seconds: int,
) -> dict[str, Any]:
    """Run GNN's strict exporter in GNN's environment and record its provenance."""
    root = gnn_repo.resolve(strict=True)
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    command = [
        str(gnn_python),
        "-m",
        "gnn.export.geo_infer",
        str(source),
        str(artifact),
        "--step-seconds",
        str(step_seconds),
        "--space-kind",
        "h3",
        "--state-ids",
        str(state_ids),
    ]
    subprocess.run(command, cwd=root, env=environment, check=True, timeout=300)

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, timeout=60
        ).strip()

    version = subprocess.check_output(
        [
            str(gnn_python),
            "-c",
            "import sys, h3; print(sys.version.split()[0], h3.__version__)",
        ],
        cwd=root,
        env=environment,
        text=True,
        timeout=60,
    ).split()
    return {
        "producer_module": "gnn.export.geo_infer",
        "command": [
            "python",
            "-m",
            "gnn.export.geo_infer",
            FIXTURE_NAMES["source"],
            FIXTURE_NAMES["artifact"],
            "--step-seconds",
            str(step_seconds),
            "--space-kind",
            "h3",
            "--state-ids",
            FIXTURE_NAMES["state_ids"],
        ],
        "gnn_revision": git("rev-parse", "HEAD"),
        "gnn_dirty": bool(
            git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock")
        ),
        "gnn_python": version[0],
        "gnn_h3": version[1],
    }


def fixture_problems(
    model: RiverHomingModel, sources: Mapping[str, str], fixtures: Path
) -> list[str]:
    """Compare committed fixtures with freshly compiled sources; empty when fresh."""
    problems = []
    for key in ("source", "state_ids", "domain"):
        path = fixtures / FIXTURE_NAMES[key]
        if not path.exists() or path.read_text(encoding="utf-8") != sources[key]:
            problems.append(f"{FIXTURE_NAMES[key]} differs from the compiled source")
    artifact_path = fixtures / FIXTURE_NAMES["artifact"]
    if not artifact_path.exists():
        return problems + [f"{FIXTURE_NAMES['artifact']} is missing"]
    data = GNNArtifact.load(artifact_path).to_dict()
    if data["provenance"]["source_sha256"] != sha256_bytes(
        sources["source"].encode("utf-8")
    ):
        problems.append("artifact provenance does not match the GNN source digest")
    spec = model.gnn_spec(model_name="check", annotation="check")
    if data["matrices"] != spec.float_matrices():
        problems.append("artifact matrices differ from the exact compiled matrices")
    if data["space"] != {"kind": "h3", "state_ids": list(model.network.cells)}:
        problems.append("artifact state order or space kind differs")
    if data["time"]["step_seconds"] != model.parameters.step_seconds:
        problems.append("artifact step_seconds differs from the declared step")
    return problems


def exact_first_update(model: RiverHomingModel, observation: int) -> dict[str, str]:
    """Exact Bayes posterior after the first observation, as fraction strings."""
    joint = [
        model.D[state] * model.A[observation][state] for state in range(len(model.D))
    ]
    evidence = sum(joint, Fraction(0))
    return {
        "evidence": f"{evidence.numerator}/{evidence.denominator}",
        **{
            model.network.cells[
                state
            ]: f"{(value / evidence).numerator}/{(value / evidence).denominator}"
            for state, value in enumerate(joint)
            if value
        },
    }


def analyze(
    model: RiverHomingModel,
    result: ClosedLoopResult,
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Homing events, light-cone and identity checks, and headline metrics."""
    network = model.network
    cells = network.cells
    roles = [network.roles[cell] for cell in cells]
    landmarks = model.landmarks
    home = cells.index(landmarks["home"])
    confluence = cells.index(landmarks["natal_confluence"])
    hops = network.hop_distance()
    steps = result.trace["steps"]
    priors = [np.asarray(step["prior"]) for step in steps]
    posteriors = [np.asarray(step["posterior"]) for step in steps]
    start = [cells[index] for index in support(model.D)]
    cone = [network.reachable(start, step) for step in range(len(steps))]
    cone_indices = [{cells.index(cell) for cell in reach} for reach in cone]
    prior_violations = [
        step
        for step, prior in enumerate(priors)
        if not support(prior) <= cone_indices[step]
    ]
    posterior_violations = [
        step
        for step, (pre, post) in enumerate(zip(priors, posteriors))
        if not support(post) <= support(pre)
    ]
    home_bound = next(
        step
        for step in range(len(cells) + 1)
        if landmarks["home"] in network.reachable(start, step)
    )
    true_path = result.true_states

    def first(predicate: Any) -> int | None:
        return next(
            (step for step, state in enumerate(true_path) if predicate(state)), None
        )

    overshoot = [
        cells.index(cell)
        for cell in cells
        if network.roles[cell] == "main"
        and hops[cell] > hops[landmarks["natal_confluence"]]
    ]
    final = posteriors[-1]
    natal = [index for index, role in enumerate(roles) if role == "natal"]
    return {
        "steps": len(steps),
        "step_hours": model.parameters.step_seconds / 3600,
        "true_start": cells[true_path[0]],
        "entered_river_step": first(lambda state: roles[state] != "sea"),
        "entered_natal_step": first(lambda state: roles[state] == "natal"),
        "reached_home_step": first(lambda state: state == home),
        "home_lower_bound_steps": home_bound,
        "tributary_steps": sum(roles[state] == "tributary" for state in true_path),
        "slough_steps": sum(roles[state] == "slough" for state in true_path),
        "overshoot_steps": sum(state in overshoot for state in true_path),
        "confluence_steps": sum(state == confluence for state in true_path),
        "action_counts": {
            label: sum(step["action"] == index for step in steps)
            for index, label in enumerate(ACTIONS)
        },
        "final_posterior_home": float(final[home]),
        "final_posterior_natal": float(final[natal].sum()),
        "final_map_cell": cells[int(np.argmax(final))],
        "final_map_role": roles[int(np.argmax(final))],
        "max_pymdp_identity_gap": max(
            record["pymdp_identity_gap"] for record in diagnostics
        ),
        "min_decision_margin": min(
            float(np.diff(np.sort(record["policy_posterior"])[-2:])[0])
            for record in diagnostics
        ),
        "prior_light_cone_violations": prior_violations,
        "posterior_support_violations": posterior_violations,
        "home_prior_mass_before_bound": max(
            float(prior[home]) for prior in priors[:home_bound]
        ),
        "mean_free_energy": float(
            np.mean([record["free_energy"] for record in diagnostics])
        ),
        "exact_first_update": exact_first_update(model, result.observations[0]),
        "first_posterior_float": {
            cells[index]: float(posteriors[0][index])
            for index in support(posteriors[0])
        },
    }


def save_context_map(path: Path, layers: Path, network: H3FlowNetwork) -> Path:
    """Regional Cascadia context: HU4 watersheds, volcanoes, the convergent boundary."""
    from matplotlib.patches import Polygon, Rectangle  # noqa: PLC0415

    def read(name: str) -> list[dict[str, Any]]:
        return json.loads((layers / name).read_text(encoding="utf-8"))["features"]

    figure = viz._figure(7.2, 8.0)
    ax = figure.add_subplot(1, 1, 1)
    ax.set_facecolor("#eaf1f5")
    palette = ["#e8e2d0", "#dfe7d0", "#ece0d6", "#dde3e8"]
    for position, feature in enumerate(read("cascadia_major_watersheds.geojson")):
        geometry = feature["geometry"]
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        for polygon in polygons:
            ax.add_patch(
                Polygon(
                    polygon[0],
                    closed=True,
                    facecolor=palette[position % 4],
                    edgecolor="#9a9a8a",
                    linewidth=0.5,
                    zorder=1,
                )
            )
        lng, lat = np.mean(np.asarray(polygons[0][0]), axis=0)
        ax.text(
            lng,
            lat,
            feature["properties"]["name"],
            fontsize=5.2,
            color="#6b6b5e",
            ha="center",
            zorder=2,
        )
    for feature in read("ca_del_norte_boundary.geojson"):
        ring = feature["geometry"]["coordinates"][0]
        ax.add_patch(
            Polygon(
                ring,
                closed=True,
                facecolor="none",
                edgecolor="#9b2d1f",
                linewidth=0.9,
                linestyle="--",
                zorder=3,
            )
        )
        ax.text(
            ring[0][0] - 0.15,
            ring[0][1],
            "Del Norte County\n(simplified outline)",
            fontsize=5.8,
            color="#9b2d1f",
            ha="right",
            va="bottom",
            zorder=4,
        )
    for feature in read("cascadia_subduction_zone.geojson"):
        xs, ys = zip(*feature["geometry"]["coordinates"])
        ax.plot(xs, ys, color="#1d2a33", linewidth=1.6, zorder=3)
        ax.text(
            xs[len(xs) // 2] - 0.3,
            ys[len(ys) // 2],
            "Cascadia\nsubduction zone",
            fontsize=6.5,
            color="#1d2a33",
            ha="right",
            zorder=4,
        )
    volcanoes = read("cascadia_volcanoes.geojson")
    ax.scatter(
        [feature["geometry"]["coordinates"][0] for feature in volcanoes],
        [feature["geometry"]["coordinates"][1] for feature in volcanoes],
        marker="^",
        s=22,
        color="#c0392b",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    west, east, south, north = viz._extent(network, pad=0.02)
    ax.add_patch(
        Rectangle(
            (west, south),
            east - west,
            north - south,
            fill=False,
            edgecolor="#e4704f",
            linewidth=1.6,
            zorder=5,
        )
    )
    ax.annotate(
        "lower Smith River\n(this model)",
        (east, north),
        xytext=(18, 12),
        textcoords="offset points",
        fontsize=7,
        color="#9b2d1f",
        arrowprops=dict(arrowstyle="-", color="#9b2d1f", linewidth=0.8),
    )
    ax.set_xlim(-128.5, -114.0)
    ax.set_ylim(39.5, 50.0)
    ax.set_aspect(1.0 / math.cos(math.radians(45)))
    ax.set_title(
        "Cascadia study window: USGS WBD HU4 subregions (clipped), USGS volcanoes (▲),\n"
        "Cascadia convergent boundary",
        fontsize=8.5,
        loc="left",
    )
    ax.tick_params(labelsize=6.5)
    figure.tight_layout()
    return viz._save(figure, path)


def write_figures(
    output: Path,
    config: Mapping[str, Any],
    model: RiverHomingModel,
    features: Sequence[Mapping[str, Any]],
    result: ClosedLoopResult,
    diagnostics: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> list[Path]:
    """Render every figure and the belief animation; return written paths."""
    network = model.network
    figures = output / "figures"
    steps = result.trace["steps"]
    hours = model.parameters.step_seconds / 3600
    posteriors = [step["posterior"] for step in steps]
    actions = [step["action"] for step in steps]
    landmarks = dict(model.landmarks)
    written = [
        save_context_map(
            figures / "cascadia_context.png", config["data"]["regional_layers"], network
        ),
        viz.save_network_figure(
            figures / "smith_river_network.png",
            network,
            features,
            habitat=model.habitat,
            odor=model.odor,
            habitat_names=HABITATS,
            odor_names=ODORS,
            landmarks=landmarks,
            landmark_labels=LANDMARK_LABELS,
        ),
        viz.save_belief_snapshots(
            figures / "belief_snapshots.png",
            network,
            features,
            posteriors,
            result.true_states,
            [
                step
                for step in config["simulation"]["snapshot_steps"]
                if step < len(steps)
            ],
            actions=actions,
            action_labels=ACTIONS,
            step_hours=hours,
        ),
        viz.save_trace_figure(
            figures / "free_energy_trace.png",
            diagnostics,
            action_labels=ACTIONS,
            step_hours=hours,
            events={
                name: step
                for name, step in (
                    ("enters river", summary["entered_river_step"]),
                    ("enters Rowdy Creek", summary["entered_natal_step"]),
                    ("home", summary["reached_home_step"]),
                )
                if step is not None
            },
        ),
        viz.save_efe_figure(
            figures / "efe_decomposition.png",
            diagnostics,
            action_labels=ACTIONS,
            steps=[
                step for step in config["simulation"]["efe_steps"] if step < len(steps)
            ],
            step_hours=hours,
        ),
        viz.save_light_cone_figure(
            figures / "light_cone.png",
            network,
            [step["prior"] for step in steps],
            [network.cells[index] for index in support(model.D)],
            home=landmarks["home"],
        ),
    ]
    captions = [
        f"t = {index} ({index * hours:g} h)  obs {model.observation_labels[step['observation']]}  "
        f"→ {ACTIONS[step['action']]}"
        for index, step in enumerate(steps)
    ]
    written += viz.save_belief_animation(
        figures / "belief_animation",
        network,
        features,
        posteriors,
        result.true_states,
        captions=captions,
    )
    return written


def write_json(path: Path, payload: Any) -> Path:
    """Write sorted, indented JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    gnn_repo: Path | None = None,
    gnn_python: Path | None = None,
    refresh_fixtures: bool = False,
    figures: bool = True,
    steps: int | None = None,
) -> dict[str, Any]:
    """Run every stage; return the run manifest."""
    config = load_config(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, features, provenance = build_model(config)
    sources = render_sources(model, config, provenance)
    fixtures = config["data"]["fixtures"]
    export_record: dict[str, Any] | None = None
    if gnn_repo is not None:
        if gnn_python is None:
            raise ValueError("--gnn-repo requires --gnn-python")
        with tempfile.TemporaryDirectory(prefix="cascadia-gnn-") as temp:
            temp_dir = Path(temp)
            for key in ("source", "state_ids"):
                (temp_dir / FIXTURE_NAMES[key]).write_text(
                    sources[key], encoding="utf-8"
                )
            exported = temp_dir / FIXTURE_NAMES["artifact"]
            export_record = export_with_gnn(
                temp_dir / FIXTURE_NAMES["source"],
                temp_dir / FIXTURE_NAMES["state_ids"],
                exported,
                gnn_repo=gnn_repo,
                gnn_python=gnn_python,
                step_seconds=model.parameters.step_seconds,
            )
            artifact_text = exported.read_text(encoding="utf-8")
        artifact = GNNArtifact(artifact_text)
        export_record["source_sha256"] = sha256_bytes(sources["source"].encode("utf-8"))
        export_record["artifact_canonical_sha256"] = artifact.digest
        if refresh_fixtures:
            fixtures.mkdir(parents=True, exist_ok=True)
            for key in ("source", "state_ids", "domain"):
                (fixtures / FIXTURE_NAMES[key]).write_text(
                    sources[key], encoding="utf-8"
                )
            (fixtures / FIXTURE_NAMES["artifact"]).write_text(
                artifact_text, encoding="utf-8"
            )
            write_json(fixtures / FIXTURE_NAMES["export"], export_record)
    elif refresh_fixtures:
        raise ValueError(
            "--refresh-fixtures requires --gnn-repo so GNN produces the artifact"
        )
    problems = fixture_problems(model, sources, fixtures)
    if problems and gnn_repo is None:
        raise RuntimeError("Stale Cascadia fixtures: " + "; ".join(problems))
    if gnn_repo is None:
        artifact = GNNArtifact.load(fixtures / FIXTURE_NAMES["artifact"])
    elif (
        artifact.to_dict()["matrices"]
        != model.gnn_spec(model_name="check", annotation="check").float_matrices()
    ):
        raise RuntimeError("GNN export changed the compiled matrices")
    simulation = config["simulation"]
    total = steps or simulation["steps"]
    result = run_closed_loop(
        artifact,
        initial_state=model.network.index()[
            model.landmarks[simulation["initial_state"]]
        ],
        steps=total,
        process_seed=simulation["process_seed"],
        agent_seed=simulation["agent_seed"],
        start=datetime.fromisoformat(simulation["start"]),
    )
    diagnostics = trace_diagnostics(artifact, result)
    summary = analyze(model, result, diagnostics)
    written = [
        write_json(output / "closed_loop.json", result.to_dict()),
        write_json(output / "diagnostics.json", diagnostics),
        write_json(output / "summary.json", summary),
        write_json(output / "domain.json", json.loads(sources["domain"])),
    ]
    (output / FIXTURE_NAMES["source"]).write_text(sources["source"], encoding="utf-8")
    written.append(output / FIXTURE_NAMES["source"])
    if figures:
        written += write_figures(
            output, config, model, features, result, diagnostics, summary
        )
    manifest = {
        "schema_version": "geo-infer-act/cascadia-homing-run/1",
        "config": str(config["_path"].name),
        "config_sha256": config["_sha256"],
        "flowlines": provenance,
        "network_sha256": model.network.digest,
        "gnn_source_sha256": sha256_bytes(sources["source"].encode("utf-8")),
        "artifact_canonical_sha256": artifact.digest,
        "gnn_export": export_record
        or json.loads((fixtures / FIXTURE_NAMES["export"]).read_text(encoding="utf-8")),
        "fixtures_fresh": not problems,
        "versions": {
            "python": platform.python_version(),
            "inferactively-pymdp": version("inferactively-pymdp"),
            "h3": h3.__version__,
            "numpy": np.__version__,
        },
        "summary": summary,
        "files": {
            str(path.relative_to(output)): sha256_file(path) for path in sorted(written)
        },
    }
    write_json(output / "run_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gnn-repo", type=Path, help="GNN checkout used to export the artifact"
    )
    parser.add_argument("--gnn-python", type=Path, help="Python of the GNN environment")
    parser.add_argument(
        "--refresh-fixtures",
        action="store_true",
        help="Rewrite committed fixtures from this compile and GNN export",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--steps", type=int, help="Override the configured step count")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and print the headline summary."""
    args = parse_args(argv)
    manifest = run(
        args.config,
        args.output,
        gnn_repo=args.gnn_repo,
        gnn_python=args.gnn_python,
        refresh_fixtures=args.refresh_fixtures,
        figures=not args.no_figures,
        steps=args.steps,
    )
    summary = manifest["summary"]
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "entered_natal_step",
                    "reached_home_step",
                    "home_lower_bound_steps",
                    "final_posterior_home",
                    "final_posterior_natal",
                    "max_pymdp_identity_gap",
                    "prior_light_cone_violations",
                )
            },
            indent=1,
        )
    )
    print(f"wrote {len(manifest['files'])} files under {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
