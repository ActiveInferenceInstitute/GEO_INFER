"""Deterministic figures for typed H3 flow networks and belief traces.

Every function builds its own :class:`matplotlib.figure.Figure` with the Agg
canvas, so no call mutates process-wide matplotlib state. Hexagons come from
real ``h3.cell_to_boundary`` geometry; flow lines are drawn beneath them in
longitude/latitude with a local equal-distance aspect. Output directories are
created before writing.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any
from collections.abc import Mapping, Sequence

import h3
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon

from geo_infer_act.core.h3_flow_network import H3FlowNetwork, line_parts

ROLE_COLORS = {
    "sea": "#2f6f9f",
    "main": "#3a9d9b",
    "slough": "#9c8ac4",
    "tributary": "#a58b62",
    "natal": "#e4704f",
}
HABITAT_COLORS = ("#2f6f9f", "#7fb8c9", "#3a9d9b", "#7cae5a")
ODOR_COLORS = ("#e9e5dc", "#f2b38b", "#d4492a")
ACTION_COLORS = ("#8c8c8c", "#2f6f9f", "#3a9d9b", "#e4704f")
BELIEF_MAP = LinearSegmentedColormap.from_list(
    "belief", ["#f7f4ee", "#f6c7a5", "#e4704f", "#9b2d1f", "#4a1410"]
)
INK = "#1d2a33"


def _figure(width: float, height: float, dpi: int = 150) -> Figure:
    figure = Figure(figsize=(width, height), dpi=dpi, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _save(figure: Figure, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        target, bbox_inches="tight", facecolor="white", metadata={"Software": None}
    )
    return target


def _extent(
    network: H3FlowNetwork, pad: float = 0.004
) -> tuple[float, float, float, float]:
    points = [point for cell in network.cells for point in h3.cell_to_boundary(cell)]
    lats = [lat for lat, _ in points]
    lngs = [lng for _, lng in points]
    return min(lngs) - pad, max(lngs) + pad, min(lats) - pad, max(lats) + pad


def draw_network(
    ax: Any,
    network: H3FlowNetwork,
    *,
    flowlines: Sequence[Mapping[str, Any]] = (),
    values: Sequence[float] | None = None,
    colors: Sequence[str] | None = None,
    vmax: float | None = None,
    show_relations: bool = True,
    landmarks: Mapping[str, str] | None = None,
    landmark_labels: Mapping[str, str] | None = None,
    marker: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw hexagons (by role, explicit colors or values) over flow lines.

    Returns the colormap normalization when ``values`` are drawn, else None.
    """
    west, east, south, north = _extent(network)
    ax.set_facecolor("#fbfaf7")
    for feature in flowlines:
        properties = feature.get("properties") or {}
        coast = properties.get("ftype") == 566
        for vertices in line_parts(feature["geometry"]):
            xs, ys = zip(*vertices)
            ax.plot(
                xs,
                ys,
                color="#8a8a8a" if coast else "#5b8fb0",
                linewidth=1.2 if coast else 0.8,
                alpha=0.9 if coast else 0.75,
                zorder=1,
            )
    norm = None
    if values is not None:
        upper = vmax if vmax is not None else max(float(np.max(values)), 1e-12)
        norm = Normalize(vmin=0.0, vmax=upper)
    for position, cell in enumerate(network.cells):
        boundary = [(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]
        if values is not None:
            face = BELIEF_MAP(norm(float(values[position])))
            alpha = 0.92
        elif colors is not None:
            face, alpha = colors[position], 0.85
        else:
            face, alpha = ROLE_COLORS[network.roles[cell]], 0.8
        ax.add_patch(
            Polygon(
                boundary,
                closed=True,
                facecolor=face,
                edgecolor="white",
                linewidth=0.9,
                alpha=alpha,
                zorder=2,
            )
        )
    if show_relations:
        for relation, source, target in network.relation_pairs():
            if relation != "successor":
                continue
            y0, x0 = h3.cell_to_latlng(source)
            y1, x1 = h3.cell_to_latlng(target)
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    color=INK,
                    linewidth=0.6,
                    alpha=0.55,
                    zorder=3,
                    shrinkA=4,
                    shrinkB=4,
                )
            )
    if marker is not None:
        lat, lng = h3.cell_to_latlng(marker)
        ax.plot(
            [lng],
            [lat],
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=1.6,
            zorder=5,
        )
    for name, cell in (landmarks or {}).items():
        label = (landmark_labels or {}).get(name)
        if not label:
            continue
        lat, lng = h3.cell_to_latlng(cell)
        ax.annotate(
            label,
            (lng, lat),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=6.5,
            color=INK,
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
        )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1.0 / math.cos(math.radians((south + north) / 2)))
    ax.tick_params(labelsize=6, colors="#555555")
    ax.set_xlabel("longitude", fontsize=7, color="#555555")
    ax.set_ylabel("latitude", fontsize=7, color="#555555")
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    return norm


def save_network_figure(
    path: str | Path,
    network: H3FlowNetwork,
    flowlines: Sequence[Mapping[str, Any]],
    *,
    habitat: Sequence[int],
    odor: Sequence[int],
    habitat_names: Sequence[str],
    odor_names: Sequence[str],
    landmarks: Mapping[str, str],
    landmark_labels: Mapping[str, str],
) -> Path:
    """Three panels: typed roles with flow arrows, habitat classes, odor field."""
    figure = _figure(13.5, 6.2)
    axes = figure.subplots(1, 3)
    draw_network(
        axes[0],
        network,
        flowlines=flowlines,
        landmarks=landmarks,
        landmark_labels=landmark_labels,
        title="Typed H3 flow network (res 8)",
    )
    draw_network(
        axes[1],
        network,
        flowlines=flowlines,
        show_relations=False,
        colors=[HABITAT_COLORS[value] for value in habitat],
        title="Habitat class",
    )
    draw_network(
        axes[2],
        network,
        flowlines=flowlines,
        show_relations=False,
        colors=[ODOR_COLORS[value] for value in odor],
        title="Natal-odor level",
    )
    for ax, palette, names in (
        (axes[0], [ROLE_COLORS[role] for role in ROLE_COLORS], list(ROLE_COLORS)),
        (axes[1], list(HABITAT_COLORS), list(habitat_names)),
        (axes[2], list(ODOR_COLORS), list(odor_names)),
    ):
        handles = [
            Polygon([(0, 0)], facecolor=color, edgecolor="white") for color in palette
        ]
        ax.legend(handles, names, loc="lower left", fontsize=6.5, frameon=False)
    figure.tight_layout()
    return _save(figure, path)


def save_belief_snapshots(
    path: str | Path,
    network: H3FlowNetwork,
    flowlines: Sequence[Mapping[str, Any]],
    beliefs: Sequence[Sequence[float]],
    true_states: Sequence[int],
    steps: Sequence[int],
    *,
    actions: Sequence[int],
    action_labels: Sequence[str],
    step_hours: float,
) -> Path:
    """Posterior beliefs on the map at selected steps, true cell circled."""
    figure = _figure(3.3 * len(steps), 4.3)
    axes = figure.subplots(1, len(steps))
    for ax, step in zip(np.atleast_1d(axes), steps):
        draw_network(
            ax,
            network,
            flowlines=flowlines,
            values=beliefs[step],
            vmax=1.0,
            show_relations=False,
            marker=network.cells[true_states[step]],
            title=f"t = {step} ({step * step_hours:g} h) · {action_labels[actions[step]]}",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
    figure.tight_layout()
    return _save(figure, path)


def save_trace_figure(
    path: str | Path,
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    action_labels: Sequence[str],
    step_hours: float,
    events: Mapping[str, int] | None = None,
) -> Path:
    """Free energy, belief entropy, true-cell mass and the chosen actions over time."""
    figure = _figure(9.5, 6.4)
    axes = figure.subplots(
        3, 1, sharex=True, gridspec_kw={"height_ratios": [2, 2, 0.6]}
    )
    hours = np.array([record["step"] * step_hours for record in diagnostics])
    axes[0].plot(
        hours,
        [record["free_energy"] for record in diagnostics],
        color="#9b2d1f",
        linewidth=1.6,
        label="free energy  −log p(o_t)",
    )
    axes[0].plot(
        hours,
        [record["posterior_entropy"] for record in diagnostics],
        color="#2f6f9f",
        linewidth=1.4,
        label="posterior entropy  H[q(s_t)]",
    )
    axes[0].set_ylabel("nats", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[1].plot(
        hours,
        [record["posterior_true_state"] for record in diagnostics],
        color="#3a9d9b",
        linewidth=1.6,
        label="q(true cell)",
    )
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("probability", fontsize=8)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    for record in diagnostics:
        axes[2].axvspan(
            record["step"] * step_hours - step_hours / 2,
            record["step"] * step_hours + step_hours / 2,
            color=ACTION_COLORS[record["action"]],
            linewidth=0,
        )
    axes[2].set_yticks([])
    axes[2].set_xlabel("hours since arrival", fontsize=8)
    handles = [Polygon([(0, 0)], facecolor=color) for color in ACTION_COLORS]
    axes[2].legend(
        handles,
        action_labels,
        ncol=len(action_labels),
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.9),
    )
    for rank, (name, step) in enumerate((events or {}).items()):
        for ax in axes[:2]:
            ax.axvline(
                step * step_hours, color="#555555", linewidth=0.8, linestyle="--"
            )
        axes[0].annotate(
            name,
            (step * step_hours, axes[0].get_ylim()[1]),
            xytext=(3, -10 - 11 * rank),
            textcoords="offset points",
            fontsize=7,
            color="#555555",
        )
    for ax in axes:
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.tight_layout()
    return _save(figure, path)


def save_efe_figure(
    path: str | Path,
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    action_labels: Sequence[str],
    steps: Sequence[int],
    step_hours: float,
) -> Path:
    """Risk + ambiguity per action at chosen steps, and q(π) over the whole run."""
    figure = _figure(11.5, 6.0)
    grid = figure.add_gridspec(2, len(steps), height_ratios=[1.25, 1])
    width = 0.7
    for column, step in enumerate(steps):
        ax = figure.add_subplot(grid[0, column])
        record = diagnostics[step]
        positions = np.arange(len(action_labels))
        risk = [policy["risk"] for policy in record["policies"]]
        ambiguity = [policy["ambiguity"] for policy in record["policies"]]
        ax.bar(positions, risk, width, color="#e4704f", label="risk")
        ax.bar(
            positions, ambiguity, width, bottom=risk, color="#2f6f9f", label="ambiguity"
        )
        chosen = record["action"]
        ax.bar(
            [chosen],
            [risk[chosen] + ambiguity[chosen]],
            width,
            fill=False,
            edgecolor=INK,
            linewidth=1.6,
        )
        ax.set_xticks(positions, action_labels, fontsize=6.5, rotation=30)
        ax.set_title(f"t = {step} ({step * step_hours:g} h)", fontsize=8, loc="left")
        ax.tick_params(labelsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if column == 0:
            ax.set_ylabel("expected free energy (nats)", fontsize=7)
            handles, labels = ax.get_legend_handles_labels()
            figure.legend(
                handles[:2],
                labels[:2],
                loc="upper right",
                ncol=2,
                fontsize=7,
                frameon=False,
            )
    ax = figure.add_subplot(grid[1, :])
    matrix = np.array([record["policy_posterior"] for record in diagnostics]).T
    image = ax.imshow(
        matrix, aspect="auto", cmap=BELIEF_MAP, vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_yticks(range(len(action_labels)), action_labels, fontsize=7)
    ax.set_xlabel("step (6 h each)", fontsize=7)
    ax.set_title(
        "policy posterior q(π) = σ(−G(π) + ln E) at every step; outlined bars above are the chosen action",
        fontsize=8,
        loc="left",
    )
    ax.tick_params(labelsize=6.5)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    colorbar.ax.tick_params(labelsize=6.5)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(figure, path)


def save_light_cone_figure(
    path: str | Path,
    network: H3FlowNetwork,
    priors: Sequence[Sequence[float]],
    start_cells: Sequence[str],
    *,
    home: str,
) -> Path:
    """Belief support size versus the reachable cone, and the home-cell bound."""
    figure = _figure(9.0, 3.6)
    axes = figure.subplots(1, 2)
    steps = np.arange(len(priors))
    cone = [len(network.reachable(start_cells, int(step))) for step in steps]
    support = [int(np.count_nonzero(np.asarray(prior) > 0)) for prior in priors]
    axes[0].step(
        steps,
        cone,
        where="post",
        color="#9c8ac4",
        linewidth=1.6,
        label="reachable cone |R_t|",
    )
    axes[0].step(
        steps,
        support,
        where="post",
        color="#9b2d1f",
        linewidth=1.6,
        label="prior support |supp q_t|",
    )
    axes[0].set_xlabel("step", fontsize=8)
    axes[0].set_ylabel("cells", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    first = next(
        (
            int(step)
            for step in steps
            if home in network.reachable(start_cells, int(step))
        ),
        None,
    )
    home_index = network.cells.index(home)
    axes[1].plot(
        steps,
        [float(prior[home_index]) for prior in priors],
        color="#e4704f",
        linewidth=1.6,
        label="prior mass on home cell",
    )
    if first is not None:
        axes[1].axvspan(
            -0.5,
            first - 0.5,
            color="#9c8ac4",
            alpha=0.18,
            label=f"forbidden: t < {first}",
        )
    axes[1].set_xlabel("step", fontsize=8)
    axes[1].legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.tight_layout()
    return _save(figure, path)


def _frame(
    network: H3FlowNetwork,
    flowlines: Sequence[Mapping[str, Any]],
    belief: Sequence[float],
    true_cell: str,
    caption: str,
    width: int,
    height: int,
) -> Any:
    from PIL import Image  # noqa: PLC0415

    figure = _figure(width / 100, height / 100, dpi=100)
    ax = figure.add_axes([0.06, 0.08, 0.9, 0.84])
    draw_network(
        ax,
        network,
        flowlines=flowlines,
        values=belief,
        vmax=1.0,
        show_relations=False,
        marker=true_cell,
        title=caption,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white", metadata={"Software": None})
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_belief_animation(
    path: str | Path,
    network: H3FlowNetwork,
    flowlines: Sequence[Mapping[str, Any]],
    beliefs: Sequence[Sequence[float]],
    true_states: Sequence[int],
    *,
    captions: Sequence[str],
    frame_ms: int = 400,
    width: int = 900,
    height: int = 1000,
    mp4: bool = True,
) -> list[Path]:
    """Write a GIF (and an H.264 MP4 when ffmpeg is on PATH) of belief maps."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        _frame(network, flowlines, belief, network.cells[state], caption, width, height)
        for belief, state, caption in zip(beliefs, true_states, captions)
    ]
    gif = target.with_suffix(".gif")
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )
    written = [gif]
    ffmpeg = shutil.which("ffmpeg")
    if mp4 and ffmpeg:
        video = target.with_suffix(".mp4")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{frames[0].width}x{frames[0].height}",
            "-r",
            f"{1000 / frame_ms:g}",
            "-i",
            "-",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            str(video),
        ]
        payload = b"".join(frame.tobytes() for frame in frames)
        subprocess.run(command, input=payload, check=True, timeout=600)
        written.append(video)
    return written
