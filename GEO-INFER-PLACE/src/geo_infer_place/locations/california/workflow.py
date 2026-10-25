"""Create-only statewide orchestration with retained inputs and a portable byte receipt."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
from typing import Literal

from geo_infer_place.locations.california.acquisition import check_sources
from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    Digest,
    ObservatoryConfig,
    StrictModel,
    digest,
    load_config,
    strict_bytes,
    write_record,
)
from geo_infer_place.locations.california.geometry import Landscape, prepare_landscape
from geo_infer_place.locations.california.monitoring import (
    Experiment,
    check_experiment,
    qualify_models,
    run_experiment,
)
from geo_infer_place.locations.california.report import build_report
from geo_infer_place.locations.california.report_integrity import (
    ReportIdentity,
    check_report,
)


class WorkflowReceipt(StrictModel):
    """Complete retained-byte inventory; current runtime qualification is separate."""

    schema_version: Literal["geo-infer-place/california-workflow/1"]
    config: ObservatoryConfig
    files: dict[str, Digest]


def _inventory(folder: Path) -> dict[str, str]:
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise CaliforniaError("workflow contains a symlink")
        if path.is_file() and path != folder / "workflow.json":
            result[path.relative_to(folder).as_posix()] = digest(path)
        elif not path.is_dir() and path != folder / "workflow.json":
            raise CaliforniaError("workflow contains a nonregular artifact")
    return result


def check_workflow(folder: Path, *, current_execution: bool = False) -> WorkflowReceipt:
    """Check portable custody and stage links; optionally qualify the current runtime."""
    if folder.is_symlink():
        raise CaliforniaError("workflow folder is a symlink")
    receipt = WorkflowReceipt.model_validate_json(
        strict_bytes(folder / "workflow.json")
    )
    if (
        any(
            PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name
            for name in receipt.files
        )
        or _inventory(folder) != receipt.files
    ):
        raise CaliforniaError("workflow files differ from the retained inventory")
    source = check_sources(folder / "sources")
    landscape_path = folder / "landscape" / "landscape.json"
    landscape = Landscape.model_validate_json(strict_bytes(landscape_path))
    execution = check_experiment(
        folder / "experiment", current_execution=current_execution
    )
    experiment_path = folder / "experiment" / "experiment.json"
    experiment = Experiment.model_validate_json(
        strict_bytes(experiment_path, limit=256 * 1024 * 1024)
    )
    report = check_report(folder / "report")
    identity = ReportIdentity.model_validate_json(
        strict_bytes(folder / "report" / "report.json", limit=256 * 1024 * 1024)
    )
    models = qualify_models(receipt.config, folder / "models")[1]
    if (
        load_config(folder / "design.json") != receipt.config
        or source.config != receipt.config
        or landscape.config != receipt.config
        or experiment.config != receipt.config
        or identity.config != receipt.config
        or landscape.source_receipt_sha256
        != digest(folder / "sources" / "sources_receipt.json")
        or experiment.landscape_sha256 != digest(landscape_path)
        or identity.landscape_sha256 != experiment.landscape_sha256
        or identity.source_receipt_sha256 != landscape.source_receipt_sha256
        or identity.experiment_sha256 != execution.files["experiment.json"]
        or report.report_sha256 != receipt.files["report/report.json"]
        or experiment.models != models
    ):
        raise CaliforniaError("workflow stage bindings differ")
    return receipt


def run_workflow(design: Path, sources: Path, models: Path, output: Path) -> Path:
    """Retain inputs, prepare geography, run monitoring, export and check one edition."""
    config = load_config(design)
    source = check_sources(sources)
    if source.config != config:
        raise CaliforniaError(
            "workflow design differs from the retained source capture"
        )
    qualify_models(config, models)
    if any(
        output.resolve().is_relative_to(path.resolve()) for path in (sources, models)
    ):
        raise CaliforniaError("workflow output must be outside its input folders")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(sources, output / "sources")
    (output / "models").mkdir()
    for path in sorted(models.iterdir()):
        shutil.copyfile(path, output / "models" / path.name)
    write_record(output / "design.json", config)
    if check_sources(output / "sources") != source:
        raise CaliforniaError("source bytes changed while retaining workflow inputs")
    landscape = prepare_landscape(config, output / "sources", output / "landscape")
    experiment = run_experiment(landscape, output / "models", output / "experiment")
    build_report(landscape, experiment, output / "report")
    path = output / "workflow.json"
    write_record(
        path,
        WorkflowReceipt(
            schema_version="geo-infer-place/california-workflow/1",
            config=config,
            files=_inventory(output),
        ),
    )
    check_workflow(output, current_execution=True)
    return path
