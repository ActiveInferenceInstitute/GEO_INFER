from __future__ import annotations

import json
from pathlib import Path
import shutil

import h3
import numpy as np
from pyproj import Transformer
import pytest
from shapely.geometry import Polygon
from shapely.ops import transform

from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    Hazard,
    ObservatoryConfig,
    Strategy,
    write_record,
)
from geo_infer_place.locations.california.geometry import (
    Cell,
    County,
    CountyShare,
    Indicator,
    Landscape,
    geometry_record,
)
from geo_infer_place.locations.california.models import (
    model_source,
    write_model_sources,
)
from geo_infer_place.locations.california.monitoring import (
    Experiment,
    check_experiment,
    evaluate_policy,
    joint_belief,
    qualify_models,
    rotating_allocation,
    run_experiment,
    run_scenario,
    score_beliefs,
    simulated_report,
)

MODELS = Path(__file__).resolve().parents[2] / "locations/california/models"


def numerical_landscape(*, flipped: bool = False) -> Landscape:
    config = ObservatoryConfig(
        first_year=2016,
        target_year=2018,
        rounds=2,
        seeds=(0,),
        budget=6,
        hazard_budget_floor=1,
    )
    identifiers = sorted(h3.grid_disk(h3.latlng_to_cell(38, -121, 5), 1))[:3]
    projector = Transformer.from_crs(4326, 3310, always_xy=True)
    cells = []
    for identifier in identifiers:
        polygon = Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(identifier)])
        lat, lng = h3.cell_to_latlng(identifier)
        cells.append(
            Cell(
                id=identifier,
                parent=h3.cell_to_parent(identifier, 3),
                latitude=lat,
                longitude=lng,
                area_km2=transform(projector.transform, polygon).area / 1e6,
                geometry=geometry_record(polygon),
                county_shares=(CountyShare(fips="06001", fraction=1.0),),
                raw_county_coverage_ratio=1.0,
            )
        )
    counties = tuple(
        County(
            fips=f"06{index:03d}",
            name=f"Numerical fixture county {index}",
            geometry=cells[0].geometry,
            area_km2=1.0,
        )
        for index in range(1, 59)
    )
    indicators = []
    for index, hazard in enumerate(Hazard):
        positive = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1]])
        if flipped:
            positive[-1] = 1 - positive[-1]
        positive = np.roll(positive, index, axis=1)
        training = positive[:-1].astype(float)
        prior = (
            training.sum(axis=0) + config.prior_strength * training.mean() + 0.5
        ) / (2 + config.prior_strength + 1)
        indicators.append(
            Indicator(
                hazard=hazard,
                unit="declared numerical test indicator",
                years=(2016, 2017, 2018),
                dates=("2016", "2017", "2018"),
                values=tuple(tuple(map(float, row)) for row in positive),
                positive=tuple(tuple(map(int, row)) for row in positive),
                prior=tuple(map(float, prior)),
                source_records=(1, 1, 1),
            )
        )
    edges = tuple(
        (i, j)
        for i, identifier in enumerate(identifiers)
        for j, other in enumerate(identifiers)
        if i < j and h3.are_neighbor_cells(identifier, other)
    )
    return Landscape(
        schema_version="geo-infer-place/california-landscape/1",
        config=config,
        source_receipt_sha256="0" * 64,
        projection="EPSG:3310",
        counties=counties,
        cells=tuple(cells),
        edges=edges,
        indicators=tuple(indicators),
        repairs=(),
        administrative_area_km2=sum(cell.area_km2 for cell in cells),
        uncovered_fraction=0.0,
        county_overlap_km2=0.0,
        county_weighting="normalized-intersection-area",
    )


def test_declared_sources_match_independently_produced_gnn_artifacts(
    tmp_path: Path,
) -> None:
    """Match authored GNN sources with actual exported artifacts and exact policy costs."""
    landscape = numerical_landscape()
    artifacts, records = qualify_models(landscape.config, MODELS)
    assert tuple(record.hazard for record in records) == tuple(Hazard)
    paths = write_model_sources(landscape.config, tmp_path / "sources")
    assert all(
        path.read_text() == model_source(landscape.config, hazard)
        for path, hazard in zip(paths, Hazard, strict=True)
    )
    with pytest.raises(FileExistsError):
        write_model_sources(landscape.config, tmp_path / "sources")
    step = evaluate_policy(artifacts[0], 0.4)
    assert step.evidence is None and step.free_energy is None
    assert abs(step.expected_free_energy[0]) < 1e-12
    assert step.expected_free_energy[2] < step.expected_free_energy[1] < 0


@pytest.mark.parametrize(
    "mutation", ["source", "matrix", "clock", "provenance", "duplicate", "roster"]
)
def test_model_custody_rejects_drift(tmp_path: Path, mutation: str) -> None:
    """Reject independently exported model drift, false clocks and incomplete rosters."""
    folder = tmp_path / "models"
    shutil.copytree(MODELS, folder)
    path = folder / "fire.geo-infer.json"
    if mutation == "roster":
        folder.joinpath("extra.geo-infer.json").write_text("{}")
    elif mutation == "source":
        path = folder / "fire.gnn.json"
        path.write_text(path.read_text() + " ")
    elif mutation == "duplicate":
        path.write_text(
            path.read_text().replace("{", '{"schema_version":"ignored",', 1)
        )
    else:
        data = json.loads(path.read_bytes())
        if mutation == "clock":
            data["time"]["step_seconds"] = 1
        elif mutation == "provenance":
            data["provenance"]["source_sha256"] = "f" * 64
        else:
            data["modalities"][0]["preferences"][0] = 1
        path.write_text(json.dumps(data))
    with pytest.raises(CaliforniaError):
        qualify_models(numerical_landscape().config, folder)


def test_planner_cannot_access_target_truth_and_sensor_counter_replays() -> None:
    """Keep initial plans independent of target truth and reproduce counterseeded reports."""
    landscape, flipped = numerical_landscape(), numerical_landscape(flipped=True)
    artifacts, _ = qualify_models(landscape.config, MODELS)
    for strategy in Strategy:
        first = run_scenario(landscape, artifacts, strategy, 0)
        second = run_scenario(flipped, artifacts, strategy, 0)
        assert first.frames[1].modes == second.frames[1].modes
        assert (
            first.frames[1].expected_free_energy
            == second.frames[1].expected_free_energy
        )
        assert all(frame.spent <= landscape.config.budget for frame in first.frames)
        if strategy == Strategy.QUOTA_ACTIVE:
            assert all(
                sum(
                    (0, 1, 3)[mode]
                    for mode in first.frames[1].modes[index * 3 : (index + 1) * 3]
                )
                >= 1
                for index in range(3)
            )
    sampled = simulated_report(
        artifacts[0], 1, 2, 0, Hazard.FIRE, landscape.cells[0].id, 1
    )
    repeated = simulated_report(
        artifacts[0], 1, 2, 0, Hazard.FIRE, landscape.cells[0].id, 1
    )
    assert sampled[0] == repeated[0]
    np.testing.assert_array_equal(sampled[1], repeated[1])


def test_create_only_experiment_and_trace_negative_controls(tmp_path: Path) -> None:
    """Bind execution bytes and reject corrupted traces without overwriting a run."""
    landscape = numerical_landscape()
    path = tmp_path / "landscape.json"
    write_record(path, landscape)
    result = run_experiment(path, MODELS, tmp_path / "run")
    parsed = Experiment.model_validate_json(result.read_bytes())
    receipt = check_experiment(result.parent)
    assert receipt.execution.owners
    assert "numpy" in receipt.execution.packages
    assert len(parsed.scenarios) == 3
    assert parsed.observations_kind == "simulated-sensors-on-historical-indicators"
    before = result.read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(path, MODELS, tmp_path / "run")
    assert result.read_bytes() == before
    scenario_file = result.parent / "active_0.json"
    retained = scenario_file.read_bytes()
    scenario_file.write_bytes(retained + b" ")
    with pytest.raises(CaliforniaError, match="sealed bytes"):
        check_experiment(result.parent)
    scenario_file.write_bytes(retained)
    sealed = result.parent / "execution.json"
    receipt_data = json.loads(sealed.read_bytes())
    receipt_data["execution"]["python"] = "changed runtime"
    sealed.write_text(json.dumps(receipt_data), encoding="utf-8")
    with pytest.raises(CaliforniaError, match="historical"):
        check_experiment(result.parent)
    assert (
        check_experiment(result.parent, current_execution=False).files == receipt.files
    )
    for mutation in (
        "seed",
        "round",
        "posterior",
        "budget",
        "cost",
        "evidence",
        "quota",
    ):
        data = json.loads(before)
        frame = data["scenarios"][0]["frames"][1]
        if mutation == "seed":
            data["scenarios"][0]["seed"] = 7
        elif mutation == "round":
            frame["round"] = 9
        elif mutation == "posterior":
            frame["fields"][0]["posterior"][0] = 1
        elif mutation == "budget":
            frame["spent"] += 1
        elif mutation == "cost":
            frame["expected_free_energy"][0][0] += 1
        elif mutation == "evidence":
            frame["exact_observation_evidence"][0] = -1
        else:
            data["config"]["hazard_budget_floor"] = 2
        with pytest.raises(ValueError):
            Experiment.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "probability,mode", [(0, 0), (1, 0), (np.nan, 0), (0.5, True), (0.5, 3)]
)
def test_local_joint_support_boundaries(probability: float, mode: int) -> None:
    """Reject unsupported probabilities and malformed controlled sensor modes."""
    with pytest.raises(CaliforniaError):
        joint_belief(probability, mode)


def test_score_sensor_and_baseline_error_paths() -> None:
    """Check a hand-computed Brier score and reject invalid scoring or simulation inputs."""
    assert score_beliefs(np.array([0.2, 0.8]), np.array([0, 1])).brier == pytest.approx(
        0.04
    )
    with pytest.raises(CaliforniaError):
        score_beliefs(np.array([0.2]), np.array([0, 1]))
    with pytest.raises(CaliforniaError):
        score_beliefs(np.array([0.0]), np.array([0]))
    with pytest.raises(CaliforniaError):
        rotating_allocation(2, True, 0, 0, np.zeros((2, 3)))
    artifacts, _ = qualify_models(numerical_landscape().config, MODELS)
    with pytest.raises(CaliforniaError):
        simulated_report(artifacts[0], 1, 3, 0, Hazard.FIRE, "cell", 0)
    with pytest.raises(CaliforniaError):
        run_scenario(numerical_landscape(), artifacts, Strategy.ACTIVE, 7)


def test_report_reads_real_trace_and_conserves_spatial_aggregations(
    tmp_path: Path,
) -> None:
    """Render trace-derived figures and conserve parent and county area measurements."""
    from PIL import Image
    from geo_infer_place.locations.california.contracts import digest
    from geo_infer_place.locations.california.report import (
        build_report,
        graph_diagnostic,
        group_summaries,
        indicator_labels,
    )

    landscape = numerical_landscape()
    path = tmp_path / "landscape.json"
    write_record(path, landscape)
    experiment = run_experiment(path, MODELS, tmp_path / "experiment")
    result = build_report(path, experiment, tmp_path / "report")
    facts = json.loads(result.read_bytes())
    assert facts["cells"] == 3 and facts["config"]["target_year"] == 2018
    assert len(facts["scenario_scores"]) == 3
    assert len(facts["aggregations"]) == 27
    files = json.loads(result.with_name("files.json").read_bytes())
    assert files["report.json"] == digest(result)
    pngs = list(result.parent.glob("*.png"))
    assert len(pngs) == 15
    for image_path in pngs:
        assert files[image_path.name] == digest(image_path)
        with Image.open(image_path) as image:
            assert image.size == (1920, 1080)
    assert indicator_labels(landscape)[Hazard.SEISMIC].startswith("M3+")
    aggregation = group_summaries(landscape, np.array([0.2, 0.3, 0.4]))
    total = sum(cell.area_km2 for cell in landscape.cells)
    assert sum(
        group["allocated_area"] for group in aggregation["parents"]
    ) == pytest.approx(total)
    assert not graph_diagnostic(landscape, np.zeros(3))["defined"]
    assert graph_diagnostic(landscape, np.array([0.0, 1.0, 1.0]))["defined"]
    with pytest.raises(CaliforniaError):
        graph_diagnostic(landscape, np.zeros((3, 2)))
    with pytest.raises(FileExistsError):
        build_report(path, experiment, result.parent)
    changed = json.loads(experiment.read_bytes())
    changed["landscape_sha256"] = "f" * 64
    alternate = tmp_path / "unbound.json"
    alternate.write_text(json.dumps(changed))
    with pytest.raises(CaliforniaError):
        build_report(path, alternate, tmp_path / "unbound-report")
    assert not tmp_path.joinpath("unbound-report").exists()


def test_report_rejects_changed_experiment_before_creating_output(
    tmp_path: Path,
) -> None:
    """Reject a valid-shaped edited score before creating any report artifacts."""
    from geo_infer_place.locations.california.report import build_report

    landscape = tmp_path / "landscape.json"
    write_record(landscape, numerical_landscape())
    experiment = run_experiment(landscape, MODELS, tmp_path / "experiment")
    changed = json.loads(experiment.read_bytes())
    changed["scenarios"][0]["frames"][0]["fields"][0]["scores"]["brier"] = 0.99
    experiment.write_text(json.dumps(changed), encoding="utf-8")
    output = tmp_path / "report"
    with pytest.raises(CaliforniaError, match="sealed bytes"):
        build_report(landscape, experiment, output)
    assert not output.exists()


def test_evaluation_recomputes_scores_and_counts_unique_area() -> None:
    """Compare losses and cumulative coverage with independently derived numerical values."""
    from geo_infer_place.locations.california.evaluation import evaluate_experiment

    landscape = numerical_landscape()
    artifacts, records = qualify_models(landscape.config, MODELS)
    scenarios = tuple(
        run_scenario(landscape, artifacts, strategy, 0) for strategy in Strategy
    )
    experiment = Experiment(
        schema_version="geo-infer-place/california-experiment/1",
        config=landscape.config,
        landscape_sha256="0" * 64,
        models=records,
        backend="geo-infer-exact-joint",
        spatial_backend="binary-mean-field/1",
        observations_kind="simulated-sensors-on-historical-indicators",
        scenarios=scenarios,
    )
    rows, comparisons = evaluate_experiment(landscape, experiment)
    areas = np.asarray([cell.area_km2 for cell in landscape.cells])
    for row in rows:
        scenario = next(
            item for item in scenarios if item.strategy.value == row.strategy
        )
        hazard_index = list(Hazard).index(Hazard(row.hazard))
        selected = {
            index
            for frame in scenario.frames[1 : row.round + 1]
            for index, mode in enumerate(
                frame.modes[hazard_index * 3 : (hazard_index + 1) * 3]
            )
            if mode > 0
        }
        assert row.unique_selected_cells == len(selected)
        assert row.cumulative_selected_area_fraction == pytest.approx(
            sum(areas[index] for index in selected) / areas.sum()
        )
        q = np.asarray(scenario.frames[row.round].fields[hazard_index].posterior)
        truth = np.asarray(landscape.indicators[hazard_index].positive[-1])
        assert row.area_weighted_brier == pytest.approx(
            float(((q - truth) ** 2 @ areas) / areas.sum())
        )
    for comparison in comparisons:
        assert comparison.brier_reduction == pytest.approx(
            comparison.initial_brier - comparison.final_brier
        )
        if comparison.strategy == "rotating":
            assert comparison.rotating_brier_advantage == 0
    changed = json.loads(experiment.model_dump_json())
    changed["scenarios"][0]["frames"][0]["fields"][0]["scores"]["brier"] = 0.99
    with pytest.raises(CaliforniaError, match="recorded scores"):
        evaluate_experiment(
            landscape, Experiment.model_validate_json(json.dumps(changed))
        )


@pytest.mark.parametrize(
    "mutation", ["changed", "extra", "missing", "symlink", "resealed-roster"]
)
def test_report_integrity_rejects_file_mutations(tmp_path: Path, mutation: str) -> None:
    """Reject changed, missing, additional, linked and resealed-incomplete report files."""
    from geo_infer_place.locations.california.report_integrity import check_report
    from geo_infer_place.locations.california.report import build_report

    path = tmp_path / "landscape.json"
    write_record(path, numerical_landscape())
    experiment = run_experiment(path, MODELS, tmp_path / "experiment")
    report = build_report(path, experiment, tmp_path / "report")
    assert check_report(report.parent).files["report.json"]
    target = report.parent / "comparison.csv"
    match mutation:
        case "changed":
            target.write_text("altered", encoding="utf-8")
        case "extra":
            (report.parent / "extra.txt").write_text("extra", encoding="utf-8")
        case "missing":
            target.rename(tmp_path / "preserved.csv")
        case "symlink":
            target.rename(tmp_path / "preserved.csv")
            target.symlink_to(tmp_path / "preserved.csv")
        case "resealed-roster":
            target.rename(tmp_path / "preserved.csv")
            manifest_path = report.parent / "files.json"
            manifest = json.loads(manifest_path.read_bytes())
            del manifest["comparison.csv"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CaliforniaError):
        check_report(report.parent)


def test_landscape_semantics_reject_resealed_structural_mutations() -> None:
    """Reject reserialized carrier, temporal and empirical-prior semantic mutations."""
    original = numerical_landscape().model_dump_json()
    for mutation in (
        "county",
        "cell",
        "hierarchy",
        "share",
        "edges",
        "hazard",
        "year",
        "prior",
        "positive",
        "fraction",
    ):
        data = json.loads(original)
        if mutation == "county":
            data["counties"].pop()
        elif mutation == "cell":
            data["cells"][1]["id"] = data["cells"][0]["id"]
        elif mutation == "hierarchy":
            data["cells"][0]["parent"] = data["cells"][0]["id"]
        elif mutation == "share":
            data["cells"][0]["county_shares"][0]["fraction"] = 0.5
        elif mutation == "edges":
            data["edges"] = []
        elif mutation == "hazard":
            data["indicators"][0]["hazard"] = "drought"
        elif mutation == "year":
            data["indicators"][0]["years"][0] = 2015
        elif mutation == "prior":
            data["indicators"][0]["prior"][0] = 0.9
        elif mutation == "positive":
            data["indicators"][0]["positive"][0][0] = 1
        else:
            data["indicators"][0]["values"][0][0] = 1.1
            data["indicators"][0]["positive"][0][0] = 1
        with pytest.raises(ValueError):
            Landscape.model_validate_json(json.dumps(data))
