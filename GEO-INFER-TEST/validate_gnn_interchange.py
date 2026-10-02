#!/usr/bin/env python3
"""Exercise a real GNN exporter in its own environment and consume in GEO-INFER."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import platform
import sys
import shutil
import uuid
from contextlib import nullcontext

from geo_infer_test.process import run_process


def revision_receipt(root: Path) -> dict:
    """Identify a checkout and expose local modifications in verification receipts."""
    return {
        # Fresh CI checkouts of the paired repo can make even `git status
        # --porcelain` exceed 10s on cold caches; 30s bounds the receipt
        # without masking a genuinely hung checkout.
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=30
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, timeout=30
            ).strip()
        ),
    }


def detect_export_module_prefix(root: Path, interpreter: Path) -> str:
    """Probe the GNN checkout layout once and return the exporter module prefix.

    GNN revisions after its 2026-09 package reorganization ship the editable
    `gnn` package (`src/gnn/`), so the exporter is `gnn.export.geo_infer`.
    Earlier revisions expose `export.geo_infer` directly under `src/`. The
    probe follows the declared filesystem layout. An import failure fails
    that layout rather than selecting an unrelated legacy exporter.
    """
    if (root / "src/gnn/export/geo_infer.py").is_file() or (
        root / "src/gnn/export/geo_infer/__init__.py"
    ).is_file():
        prefix = "gnn.export"
    elif (root / "src/export/geo_infer.py").is_file() or (
        root / "src/export/geo_infer/__init__.py"
    ).is_file():
        prefix = "export"
    else:
        raise ValueError("GNN checkout contains no supported exporter layout")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    probe = run_process(
        [
            str(interpreter),
            "-c",
            f"import {prefix}.geo_infer as exporter; from pathlib import Path; assert Path(exporter.__file__).resolve().is_relative_to(Path({str(root / 'src')!r}).resolve())",
        ],
        env=environment,
        cwd=root,
        timeout=60,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"GNN exporter import failed in declared layout {prefix}: {probe.stderr}"
        )
    print(f"GNN layout: {prefix}.geo_infer", file=sys.stderr)
    return prefix


def run_export(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    check: bool = True,
    artifact_dir: Path | None = None,
) -> None:
    """Capture exporter diagnostics separately from the JSON receipt stream."""
    completed = run_process(command, cwd=cwd, env=env, timeout=timeout)
    if artifact_dir is not None:
        attempt = artifact_dir / f"export-{uuid.uuid4().hex}"
        attempt.mkdir()
        (attempt / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (attempt / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        (attempt / "command.json").write_text(
            json.dumps(
                {"command": command, "returncode": completed.returncode}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    if completed.stdout:
        print(completed.stdout, file=sys.stderr, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if check:
        completed.check_returncode()


def validate_interchange(
    gnn_repo: Path, gnn_python: Path, *, artifact_dir: Path | None = None
) -> dict:
    """Export the tracked gridworld, validate provenance, and verify real replay."""
    root = gnn_repo.resolve(strict=True)
    if artifact_dir is not None:
        artifact_dir = artifact_dir.absolute()
    interpreter = gnn_python.absolute()
    if not interpreter.is_file():
        raise ValueError("GNN interpreter must be an existing file")
    export_module_prefix = detect_export_module_prefix(root, interpreter)
    import numpy as np
    from geo_infer_act.core.gnn_contract import GNNArtifact, run_gnn_inference

    # Fixture location follows the same layout: post-reorg checkouts keep
    # interchange fixtures under tests/export/, pre-reorg ones under
    # src/tests/export/. Derived from the probe above, never re-probed.
    if export_module_prefix == "gnn.export":
        tests_root = "tests"
        print("GNN layout: post-reorg (fixtures under tests/export/)", file=sys.stderr)
    else:
        tests_root = "src/tests"
        print(
            "GNN layout: pre-reorg (fixtures under src/tests/export/)", file=sys.stderr
        )
    source = root / "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md"
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=False)
    context = (
        nullcontext(str(artifact_dir))
        if artifact_dir is not None
        else tempfile.TemporaryDirectory(prefix="gnn-geo-contract-")
    )
    with context as temp:
        artifact_path = Path(temp) / "model.json"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        run_export(
            [
                str(interpreter),
                "-m",
                f"{export_module_prefix}.geo_infer",
                str(source),
                str(artifact_path),
                "--step-seconds",
                "60",
            ],
            cwd=root,
            env=environment,
            check=True,
            timeout=120,
            artifact_dir=Path(temp),
        )
        artifact = GNNArtifact.load(artifact_path)
        assert (
            artifact.to_dict()["provenance"]["source_sha256"]
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        records = [
            dict(timestamp=f"2026-01-01T00:0{i}:00Z", observation=i) for i in range(3)
        ]
        first = run_gnn_inference(artifact, records, random_seed=42)
        assert first == run_gnn_inference(artifact, records, random_seed=42)
        matrices = {k: np.asarray(v) for k, v in artifact.to_dict()["matrices"].items()}
        for index, step in enumerate(first["steps"]):
            prior = (
                matrices["D"]
                if index == 0
                else np.asarray(first["steps"][index - 1]["next_prior"])
            )
            likelihood = matrices["A"][records[index]["observation"]]
            posterior = prior * likelihood / np.sum(prior * likelihood)
            np.testing.assert_allclose(step["posterior"], posterior)
            np.testing.assert_allclose(
                step["next_prior"], matrices["B"][:, :, step["action"]] @ posterior
            )
        # Compile a genuine H3 stay/diffuse operator into GNN source, then
        # export in the GNN environment and consume again without reordering.
        import h3
        from geo_infer_space.core.state_space import H3StateSpace

        center = h3.latlng_to_cell(41.75, -124.2, 8)
        cells = sorted(h3.grid_disk(center, 1), reverse=True)
        space = H3StateSpace(cells)
        n = len(cells)
        expected_B = space.dense_transition_tensor()
        parameters = dict(
            A=np.eye(n).tolist(),
            B=expected_B.tolist(),
            C=[0.0] * n,
            D=[1 / n] * n,
            E=[0.25, 0.75],
        )
        source_text = (
            f"""# GNN model
## GNNVersionAndFlags
GNN v1.0
## ModelName
H3 stay and diffuse
## StateSpaceBlock
s[{n},1,type=int]
o[{n},1,type=int]
u[1,type=int]
π[2,type=float]
A[{n},{n},type=float]
B[{n},{n},2,type=float]
C[{n},type=float]
D[{n},type=float]
E[2,type=float]
## Connections
D>s
s>A
o>A
s>B
u>B
E>π
## InitialParameterization
"""
            + "\n".join(f"{key}={value!r}" for key, value in parameters.items())
            + f"""
## ModelParameters
num_states={n}
num_observations={n}
num_actions=2
num_timesteps=2
b_tensor_order=next_state_previous_state_action
## Time
Dynamic
DiscreteTime=t
"""
        )
        h3_source = Path(temp) / "h3.md"
        h3_source.write_text(source_text, encoding="utf-8")
        ids_path = Path(temp) / "state_ids.json"
        ids_path.write_text(json.dumps(cells), encoding="utf-8")
        spatial_path = Path(temp) / "h3.json"
        run_export(
            [
                str(interpreter),
                "-m",
                f"{export_module_prefix}.geo_infer",
                str(h3_source),
                str(spatial_path),
                "--step-seconds",
                "60",
                "--space-kind",
                "h3",
                "--state-ids",
                str(ids_path),
            ],
            cwd=root,
            env=environment,
            check=True,
            timeout=120,
            artifact_dir=Path(temp),
        )
        spatial = GNNArtifact.load(spatial_path)
        assert spatial.to_dict()["space"]["state_ids"] == cells
        np.testing.assert_array_equal(spatial.to_dict()["matrices"]["B"], expected_B)
        spatial_trace = run_gnn_inference(spatial, records[:2], random_seed=42)
        # A non-square Gaussian fixture catches transposed observation/control
        # axes and accidental Euler integration of a discrete transition.
        from geo_infer_act.core.gnn_gaussian_contract import (
            GaussianGNNArtifact,
            run_gaussian_gnn_inference,
        )

        gaussian_source = root / f"{tests_root}/export/gaussian_rectangular.md"
        units_path = Path(temp) / "units.json"
        units_path.write_text(
            json.dumps(
                {
                    "states": ["m", "m/s", "K"],
                    "observations": ["m", "m/s"],
                    "controls": ["N"],
                }
            ),
            encoding="utf-8",
        )
        gaussian_path = Path(temp) / "gaussian.json"
        run_export(
            [
                str(interpreter),
                "-m",
                f"{export_module_prefix}.geo_infer",
                str(gaussian_source),
                str(gaussian_path),
                "--step-seconds",
                "2",
                "--model-type",
                "linear_gaussian",
                "--units",
                str(units_path),
            ],
            cwd=root,
            env=environment,
            check=True,
            timeout=120,
            artifact_dir=Path(temp),
        )
        gaussian = GaussianGNNArtifact.load(gaussian_path)
        assert (
            gaussian.to_dict()["provenance"]["source_sha256"]
            == hashlib.sha256(gaussian_source.read_bytes()).hexdigest()
        )
        gaussian_records = [
            {
                "timestamp": "2026-09-04T00:00:00Z",
                "observation": [1, 2],
                "control": [0.25],
            },
            {
                "timestamp": "2026-09-04T00:00:02Z",
                "observation": [0, 1],
                "control": [0],
            },
        ]
        gaussian_trace = run_gaussian_gnn_inference(gaussian, gaussian_records)
        first_gaussian = gaussian_trace["steps"][0]
        np.testing.assert_allclose(first_gaussian["posterior_mean"], [2 / 3, 4 / 3, 0])
        np.testing.assert_allclose(
            first_gaussian["posterior_covariance"], np.diag([1 / 3, 4 / 3, 9])
        )
        np.testing.assert_allclose(
            first_gaussian["next_prior_mean"], [19 / 12, 11 / 6, 0]
        )
        np.testing.assert_allclose(
            first_gaussian["next_prior_covariance"],
            np.diag([43 / 30, 43 / 30, 47 / 20]),
        )
        assert gaussian_trace == run_gaussian_gnn_inference(gaussian, gaussian_records)
        from geo_infer_act.core.gnn_factored_contract import (
            FactoredGNNArtifact,
            infer_factored_step,
        )

        factored_source = root / f"{tests_root}/export/factored_example.json"
        assert (
            factored_source.read_bytes()
            == (
                Path(__file__).resolve().parents[1]
                / "GEO-INFER-ACT/tests/unit/factored_example.json"
            ).read_bytes()
        )
        factored_path = Path(temp) / "factored.json"
        run_export(
            [
                str(interpreter),
                "-m",
                f"{export_module_prefix}.geo_infer_factored",
                str(factored_source),
                str(factored_path),
                "--step-seconds",
                "60",
            ],
            cwd=root,
            env=environment,
            check=True,
            timeout=120,
            artifact_dir=Path(temp),
        )
        factored = FactoredGNNArtifact.load(factored_path)
        factored_data = factored.to_dict()
        assert (
            factored_data["provenance"]["source_sha256"]
            == hashlib.sha256(factored_source.read_bytes()).hexdigest()
        )
        factored_trace = infer_factored_step(factored, [0, 2])
        a0, a1 = (np.asarray(m["likelihood"]) for m in factored_data["modalities"])
        likelihood = np.array(
            [a0[0, s0] * a1[2, s1, s0] for s0 in range(2) for s1 in range(3)]
        )
        expected = np.asarray(factored_data["initial_joint"]) * likelihood
        expected /= expected.sum()
        np.testing.assert_allclose(factored_trace["posterior"], expected)
        assert len(factored_trace["policy_posterior"]) == len(factored_data["policies"])
        assert factored_trace == infer_factored_step(factored, [0, 2])
        retained = {}
        if artifact_dir is not None:
            for name, path in {
                "categorical-source.md": source,
                "categorical.json": artifact_path,
                "h3-source.md": h3_source,
                "h3-state-ids.json": ids_path,
                "h3.json": spatial_path,
                "gaussian-source.md": gaussian_source,
                "gaussian.json": gaussian_path,
                "factored-source.json": factored_source,
                "factored.json": factored_path,
            }.items():
                target = artifact_dir / name
                if path.resolve() != target.resolve():
                    shutil.copyfile(path, target)
                retained[name] = hashlib.sha256(target.read_bytes()).hexdigest()
            retained = {
                str(path.relative_to(artifact_dir)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in artifact_dir.rglob("*")
                if path.is_file()
            }
        return dict(
            schema_version="1.0",
            success=True,
            artifacts=retained,
            geo=revision_receipt(Path(__file__).resolve().parents[1]),
            gnn=revision_receipt(root),
            python=sys.version,
            platform=platform.platform(),
            gaussian_artifact_sha256=gaussian.digest,
            gaussian_trace=gaussian_trace,
            factored_artifact_sha256=factored.digest,
            factored_trace=factored_trace,
            h3_state_count=n,
            h3_artifact_sha256=spatial.digest,
            h3_trace=spatial_trace,
            contract=artifact.to_dict()["schema_version"],
            artifact_sha256=artifact.digest,
            source_sha256=artifact.to_dict()["provenance"]["source_sha256"],
            steps=len(first["steps"]),
            deterministic_replay=True,
            trace=first,
        )


def main() -> int:
    """Validate two explicitly selected environments and print a JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-repo", type=Path, required=True)
    parser.add_argument("--gnn-python", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a machine-readable receipt and retain exported source/artifact bytes beside it.",
    )
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new immutable receipt path")
    try:
        receipt = validate_interchange(
            args.gnn_repo,
            args.gnn_python,
            artifact_dir=args.output.parent / f"{args.output.stem}-artifacts"
            if args.output
            else None,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        AssertionError,
        subprocess.SubprocessError,
    ) as exc:
        receipt = {
            "schema_version": "1.0",
            "success": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(f"Interchange failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    from geo_infer_test.execution import runtime_receipt

    receipt["geo_execution"] = runtime_receipt()
    receipt["selection"] = {
        "gnn_repo": str(args.gnn_repo.absolute()),
        "gnn_python": str(args.gnn_python.absolute()),
    }
    gnn_lock = args.gnn_repo / "uv.lock"
    receipt["gnn_lock_sha256"] = (
        hashlib.sha256(gnn_lock.read_bytes()).hexdigest()
        if gnn_lock.is_file()
        else None
    )
    if args.output:
        retained_dir = args.output.parent / f"{args.output.stem}-artifacts"
        receipt["artifacts"] = {
            str(path.relative_to(retained_dir)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in retained_dir.rglob("*")
            if path.is_file()
        }
    serialized = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(serialized)
    else:
        sys.stdout.write(serialized)
    return 0 if receipt["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
