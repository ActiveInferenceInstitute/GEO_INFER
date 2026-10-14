"""Real GNN policy evaluation with explicitly simulated historical-indicator sensors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)
from scipy.special import xlogy

from geo_infer_act.core.gnn_factored_batch import FactoredPolicyEvaluator
from geo_infer_act.core.resource_allocation import allocate_resources
from geo_infer_act.core.gnn_factored_contract import (
    FactoredGNNArtifact,
    infer_factored_step,
)
from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    Digest,
    Hazard,
    ObservatoryConfig,
    Strategy,
    StrictModel,
    digest,
    execution_binding,
    ExecutionBinding,
    strict_bytes,
    write_record,
)
from geo_infer_place.locations.california.geometry import Landscape
from geo_infer_place.locations.california.inference import (
    Allocation,
    COSTS,
    allocate_budget,
    infer_field,
)
from geo_infer_place.locations.california.models import model_source

JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class ExactStep(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    posterior: list[float]
    evidence: float | None
    free_energy: float | None
    policy_posterior: list[float]
    expected_free_energy: list[float]
    selected_policy: int
    selected_action: list[int]
    next_prior: list[float]
    backend: Literal["geo-infer-exact-joint"]
    artifact_sha256: Digest


class QualifiedModel(StrictModel):
    hazard: Hazard
    source_sha256: Digest
    artifact_sha256: Digest
    artifact_file_sha256: Digest


class Scores(StrictModel):
    brier: float = Field(ge=0, le=1, allow_inf_nan=False)
    log_score: float = Field(ge=0, allow_inf_nan=False)
    mean_entropy: float = Field(ge=0, le=0.693147180559946, allow_inf_nan=False)


class FieldFrame(StrictModel):
    hazard: Hazard
    posterior: tuple[float, ...]
    scores: Scores
    iterations: int = Field(ge=1)
    residual: float = Field(ge=0, le=1e-10, allow_inf_nan=False)
    variational_free_energy: float = Field(allow_inf_nan=False)


class RoundFrame(StrictModel):
    round: int = Field(ge=0)
    fields: tuple[FieldFrame, ...]
    modes: tuple[int, ...]
    observations: tuple[int, ...]
    expected_free_energy: tuple[tuple[float, ...], ...]
    policy_posterior: tuple[tuple[float, ...], ...]
    information_gain: tuple[tuple[float, ...], ...]
    expected_preference_cost: tuple[tuple[float, ...], ...]
    exact_observation_evidence: tuple[float | None, ...]
    spent: int = Field(ge=0, le=512)
    objective_gain: float = Field(allow_inf_nan=False)


class Scenario(StrictModel):
    seed: int
    strategy: Strategy
    frames: tuple[RoundFrame, ...]


class Experiment(StrictModel):
    schema_version: Literal["geo-infer-place/california-experiment/1"]
    config: ObservatoryConfig
    landscape_sha256: Digest
    models: tuple[QualifiedModel, ...]
    backend: Literal["geo-infer-exact-joint"]
    spatial_backend: Literal["binary-mean-field/1"]
    observations_kind: Literal["simulated-sensors-on-historical-indicators"]
    scenarios: tuple[Scenario, ...]

    @model_validator(mode="after")
    def numerical_trace(self) -> Self:
        expected = tuple(
            (seed, strategy) for seed in self.config.seeds for strategy in Strategy
        )
        if (
            tuple((scenario.seed, scenario.strategy) for scenario in self.scenarios)
            != expected
        ):
            raise CaliforniaError("experiment scenario roster differs from its design")
        if tuple(model.hazard for model in self.models) != tuple(Hazard):
            raise CaliforniaError("experiment model roster differs")
        count = (
            len(self.scenarios[0].frames[0].fields[0].posterior)
            if self.scenarios[0].frames and self.scenarios[0].frames[0].fields
            else 0
        )
        if not 1 <= count <= self.config.max_cells:
            raise CaliforniaError("experiment has an invalid posterior carrier")
        for scenario in self.scenarios:
            if tuple(frame.round for frame in scenario.frames) != tuple(
                range(self.config.rounds + 1)
            ):
                raise CaliforniaError("experiment round roster differs")
            for frame in scenario.frames:
                if tuple(field.hazard for field in frame.fields) != tuple(Hazard):
                    raise CaliforniaError("experiment field roster differs")
                for field in frame.fields:
                    q = np.asarray(field.posterior)
                    if (
                        q.shape != (count,)
                        or not np.all(np.isfinite(q))
                        or np.any(q <= 0)
                        or np.any(q >= 1)
                    ):
                        raise CaliforniaError(
                            "experiment posteriors lack finite full support"
                        )
                arrays = (
                    frame.expected_free_energy,
                    frame.policy_posterior,
                    frame.information_gain,
                    frame.expected_preference_cost,
                )
                if frame.round == 0:
                    if (
                        any(arrays)
                        or frame.modes
                        or frame.observations
                        or frame.exact_observation_evidence
                        or frame.spent
                        or frame.objective_gain
                    ):
                        raise CaliforniaError(
                            "initial frame falsely includes monitoring"
                        )
                    continue
                costs, probabilities, gain, preferences = (
                    np.asarray(array) for array in arrays
                )
                if any(
                    array.shape != (3 * count, 3) or not np.all(np.isfinite(array))
                    for array in (costs, probabilities, gain, preferences)
                ):
                    raise CaliforniaError(
                        "experiment policy arrays differ from the carrier"
                    )
                if (
                    not np.allclose(costs, preferences - gain, atol=1e-12, rtol=0)
                    or np.any(probabilities < 0)
                    or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)
                ):
                    raise CaliforniaError(
                        "experiment policy costs or probabilities are inconsistent"
                    )
                if (
                    any(
                        len(values) != 3 * count
                        for values in (
                            frame.modes,
                            frame.observations,
                            frame.exact_observation_evidence,
                        )
                    )
                    or any(mode not in (0, 1, 2) for mode in frame.modes)
                    or any(outcome not in (0, 1, 2) for outcome in frame.observations)
                ):
                    raise CaliforniaError("experiment monitoring carrier is invalid")
                if any(
                    (mode == 0 and (outcome != 0 or evidence is not None))
                    or (
                        mode > 0
                        and (
                            evidence is None
                            or not np.isfinite(evidence)
                            or not 0 < evidence <= 1
                        )
                    )
                    for mode, outcome, evidence in zip(
                        frame.modes,
                        frame.observations,
                        frame.exact_observation_evidence,
                        strict=True,
                    )
                ):
                    raise CaliforniaError(
                        "experiment evidence differs from allocated sensor modes"
                    )
                spent = sum(COSTS[mode] for mode in frame.modes)
                objective = sum(
                    costs[index, 0] - costs[index, mode]
                    for index, mode in enumerate(frame.modes)
                )
                if (
                    spent != frame.spent
                    or spent > self.config.budget
                    or not np.isclose(
                        objective, frame.objective_gain, atol=1e-10, rtol=0
                    )
                ):
                    raise CaliforniaError(
                        "experiment resource accounting or objective differs"
                    )
                if scenario.strategy == Strategy.QUOTA_ACTIVE and any(
                    sum(
                        COSTS[mode]
                        for mode in frame.modes[index * count : (index + 1) * count]
                    )
                    < self.config.hazard_budget_floor
                    for index in range(3)
                ):
                    raise CaliforniaError(
                        "experiment hazard expenditure floor is violated"
                    )
        return self


def qualify_models(
    config: ObservatoryConfig, folder: Path
) -> tuple[tuple[FactoredGNNArtifact, ...], tuple[QualifiedModel, ...]]:
    expected_files = {
        f"{hazard.value}.{suffix}"
        for hazard in Hazard
        for suffix in ("gnn.json", "geo-infer.json")
    }
    actual_files = {path.name for path in folder.iterdir()}
    if (
        folder.is_symlink()
        or not expected_files <= actual_files
        or actual_files - expected_files - {"README.md", "AGENTS.md"}
    ):
        raise CaliforniaError("model source/artifact file roster differs")
    artifacts, records = [], []
    for hazard in Hazard:
        source = folder / f"{hazard.value}.gnn.json"
        path = folder / f"{hazard.value}.geo-infer.json"
        raw = strict_bytes(source, limit=4 * 1024 * 1024)
        expected_source = model_source(config, hazard).encode()
        if raw != expected_source:
            raise CaliforniaError(
                "declared sensor source differs from the configured GNN model"
            )
        expected = JSON_OBJECT.validate_json(raw)
        actual = JSON_OBJECT.validate_json(strict_bytes(path, limit=4 * 1024 * 1024))
        if any(actual.get(key) != value for key, value in expected.items()):
            raise CaliforniaError("GNN artifact differs from declared source fields")
        if actual.get("time") != {"step_seconds": 86400} or actual.get(
            "provenance"
        ) != {
            "producer": "GNN explicit factored JSON exporter",
            "source_kind": "explicit_factored_json",
            "source_sha256": digest(source),
        }:
            raise CaliforniaError("GNN clock or structured-source provenance differs")
        artifact = FactoredGNNArtifact.load(path)
        artifacts.append(artifact)
        records.append(
            QualifiedModel(
                hazard=hazard,
                source_sha256=digest(source),
                artifact_sha256=artifact.digest,
                artifact_file_sha256=digest(path),
            )
        )
    return tuple(artifacts), tuple(records)


def joint_belief(probability: float, mode: int = 0) -> list[float]:
    if (
        not np.isfinite(probability)
        or not 0 < probability < 1
        or type(mode) is not int
        or not 0 <= mode <= 2
    ):
        raise CaliforniaError("local indicator belief or sensor mode is invalid")
    values = [0.0] * 6
    values[mode], values[3 + mode] = 1 - probability, probability
    return values


def evaluate_policy(artifact: FactoredGNNArtifact, probability: float) -> ExactStep:
    step = ExactStep.model_validate(
        infer_factored_step(
            artifact,
            [0, 0],
            posterior=joint_belief(probability),
        )
    )
    if (
        step.evidence is not None
        or step.free_energy is not None
        or len(step.expected_free_energy) != 3
    ):
        raise CaliforniaError(
            "policy-only GNN inference unexpectedly conditioned an observation"
        )
    return step


def score_beliefs(probabilities: np.ndarray, truth: np.ndarray) -> Scores:
    if probabilities.shape != truth.shape or probabilities.ndim != 1 or not len(truth):
        raise CaliforniaError("score arrays have incompatible dimensions")
    if (
        not np.all(np.isin(truth, (0, 1)))
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0)
        or np.any(probabilities >= 1)
    ):
        raise CaliforniaError("scores require binary truth and full-support beliefs")
    return Scores(
        brier=float(np.mean((probabilities - truth) ** 2)),
        log_score=float(
            -np.mean(
                truth * np.log(probabilities) + (1 - truth) * np.log1p(-probabilities)
            )
        ),
        mean_entropy=float(
            np.mean(
                -xlogy(probabilities, probabilities)
                - xlogy(1 - probabilities, 1 - probabilities)
            )
        ),
    )


def rotating_allocation(
    count: int, budget: int, round_index: int, seed: int, costs: np.ndarray
) -> Allocation:
    if (
        type(count) is not int
        or count < 1
        or type(budget) is not int
        or not 0 <= budget <= 512
        or costs.shape != (count, 3)
    ):
        raise CaliforniaError("rotating carrier, budget or costs are invalid")
    order = np.random.default_rng(seed).permutation(count)
    modes = np.zeros(count, dtype=int)
    for offset in range(min(count, budget)):
        modes[order[(round_index * budget + offset) % count]] = 1
    gain = float(
        sum(costs[index, 0] - costs[index, mode] for index, mode in enumerate(modes))
    )
    return Allocation(tuple(map(int, modes)), int(modes.sum()), gain)


def simulated_report(
    artifact: FactoredGNNArtifact,
    truth: int,
    mode: int,
    seed: int,
    hazard: Hazard,
    cell: str,
    round_index: int,
) -> tuple[int, np.ndarray]:
    if (
        type(truth) is not int
        or truth not in (0, 1)
        or type(mode) is not int
        or mode not in (0, 1, 2)
    ):
        raise CaliforniaError("simulation truth or sensor mode is invalid")
    spec = artifact.to_dict()["modalities"][0]
    likelihood = np.asarray(spec["likelihood"], dtype=float)
    key = f"{seed}/{hazard.value}/{cell}/{round_index}/{mode}".encode()
    counter = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    outcome = int(
        np.random.default_rng(counter).choice(3, p=likelihood[:, truth, mode])
    )
    return outcome, likelihood[outcome, :, mode]


def run_scenario(
    landscape: Landscape,
    artifacts: tuple[FactoredGNNArtifact, ...],
    strategy: Strategy,
    seed: int,
) -> Scenario:
    config, count = landscape.config, len(landscape.cells)
    if len(artifacts) != 3 or seed not in config.seeds or strategy not in Strategy:
        raise CaliforniaError("scenario model, seed or strategy is outside its design")
    log_likelihood = [np.zeros((count, 2)) for _ in Hazard]
    evaluators = tuple(FactoredPolicyEvaluator(artifact) for artifact in artifacts)
    priors = [np.asarray(item.prior) for item in landscape.indicators]
    fields = [
        infer_field(prior, log_likelihood[index], landscape.edges, config.coupling)
        for index, prior in enumerate(priors)
    ]
    frames = []

    def field_records() -> tuple[FieldFrame, ...]:
        return tuple(
            FieldFrame(
                hazard=indicator.hazard,
                posterior=tuple(map(float, field.posterior)),
                scores=score_beliefs(
                    field.posterior, np.asarray(indicator.positive[-1])
                ),
                iterations=field.iterations,
                residual=field.residual,
                variational_free_energy=field.variational_free_energy,
            )
            for indicator, field in zip(landscape.indicators, fields, strict=True)
        )

    frames.append(
        RoundFrame(
            round=0,
            fields=field_records(),
            modes=(),
            observations=(),
            expected_free_energy=(),
            policy_posterior=(),
            information_gain=(),
            expected_preference_cost=(),
            exact_observation_evidence=(),
            spent=0,
            objective_gain=0.0,
        )
    )
    for round_index in range(config.rounds):
        batches = tuple(
            evaluator.evaluate(
                np.asarray(
                    [
                        joint_belief(float(probability))
                        for probability in field.posterior
                    ]
                )
            )
            for evaluator, field in zip(evaluators, fields, strict=True)
        )
        costs = np.concatenate([batch.expected_free_energy for batch in batches])
        if strategy == Strategy.ACTIVE:
            allocation = allocate_budget(costs, config.budget)
        elif strategy == Strategy.QUOTA_ACTIVE:
            result = allocate_resources(
                costs,
                COSTS,
                config.budget,
                groups=tuple(hazard.value for hazard in Hazard for _ in range(count)),
                minimum_spend={
                    hazard.value: config.hazard_budget_floor for hazard in Hazard
                },
            )
            allocation = Allocation(result.actions, result.spent, result.objective_gain)
        else:
            allocation = rotating_allocation(
                len(costs), config.budget, round_index, seed, costs
            )
        observations, evidence = [], []
        for site, mode in enumerate(allocation.modes):
            hazard_index, cell_index = divmod(site, count)
            indicator, cell = (
                landscape.indicators[hazard_index],
                landscape.cells[cell_index],
            )
            artifact = artifacts[hazard_index]
            observation, likelihood = simulated_report(
                artifact,
                indicator.positive[-1][cell_index],
                mode,
                seed,
                indicator.hazard,
                cell.id,
                round_index,
            )
            observations.append(observation)
            if mode:
                step = ExactStep.model_validate(
                    infer_factored_step(
                        artifact,
                        [observation, mode],
                        prior=joint_belief(
                            float(fields[hazard_index].posterior[cell_index]), mode
                        ),
                    )
                )
                evidence.append(step.evidence)
                if np.any(likelihood <= 0):
                    raise CaliforniaError(
                        "active sensor observation has unsupported likelihood"
                    )
                log_likelihood[hazard_index][cell_index] += np.log(likelihood)
            else:
                evidence.append(None)
        fields = [
            infer_field(prior, log_likelihood[index], landscape.edges, config.coupling)
            for index, prior in enumerate(priors)
        ]
        frames.append(
            RoundFrame(
                round=round_index + 1,
                fields=field_records(),
                modes=allocation.modes,
                observations=tuple(observations),
                expected_free_energy=tuple(tuple(map(float, row)) for row in costs),
                policy_posterior=tuple(
                    tuple(map(float, row))
                    for batch in batches
                    for row in batch.policy_posterior
                ),
                information_gain=tuple(
                    tuple(map(float, row))
                    for batch in batches
                    for row in batch.information_gain
                ),
                expected_preference_cost=tuple(
                    tuple(map(float, row))
                    for batch in batches
                    for row in batch.expected_preference_cost
                ),
                exact_observation_evidence=tuple(evidence),
                spent=allocation.spent,
                objective_gain=allocation.objective_gain,
            )
        )
    return Scenario(seed=seed, strategy=strategy, frames=tuple(frames))


class ExperimentReceipt(StrictModel):
    schema_version: Literal["geo-infer-place/california-execution/1"]
    execution: ExecutionBinding
    files: dict[str, Digest]


def check_experiment(
    folder: Path, *, current_execution: bool = True
) -> ExperimentReceipt:
    receipt = ExperimentReceipt.model_validate_json(
        strict_bytes(folder / "execution.json", limit=1024 * 1024)
    )
    expected = set(receipt.files) | {"execution.json"}
    if (
        any(Path(name).name != name for name in receipt.files)
        or {path.name for path in folder.iterdir()} != expected
    ):
        raise CaliforniaError("experiment output roster differs from its receipt")
    for name, checksum in receipt.files.items():
        path = folder / name
        if path.is_symlink() or not path.is_file() or digest(path) != checksum:
            raise CaliforniaError("experiment output differs from its sealed bytes")
    experiment = Experiment.model_validate_json(
        strict_bytes(folder / "experiment.json", limit=256 * 1024 * 1024)
    )
    expected_files = {"experiment.json"} | {
        f"{scenario.strategy.value}_{scenario.seed}.json"
        for scenario in experiment.scenarios
    }
    if set(receipt.files) != expected_files:
        raise CaliforniaError("experiment scenario file roster differs")
    for scenario in experiment.scenarios:
        retained = Scenario.model_validate_json(
            strict_bytes(
                folder / f"{scenario.strategy.value}_{scenario.seed}.json",
                limit=64 * 1024 * 1024,
            )
        )
        if retained != scenario:
            raise CaliforniaError("scenario file differs from aggregate trace")
    if current_execution and receipt.execution != execution_binding():
        raise CaliforniaError("experiment execution binding is historical")
    return receipt


def run_experiment(landscape_path: Path, models: Path, output: Path) -> Path:
    binding = execution_binding()
    landscape = Landscape.model_validate_json(strict_bytes(landscape_path))
    artifacts, model_records = qualify_models(landscape.config, models)
    if (
        len(landscape.cells)
        * len(Hazard)
        * len(Strategy)
        * len(landscape.config.seeds)
        * landscape.config.rounds
        > 600000
    ):
        raise CaliforniaError(
            "actual experiment exceeds 600000 local policy evaluations"
        )
    output.mkdir(parents=True, exist_ok=False)
    source_digest = digest(landscape_path)
    scenarios = []
    for seed in landscape.config.seeds:
        for strategy in Strategy:
            scenario = run_scenario(landscape, artifacts, strategy, seed)
            write_record(output / f"{strategy.value}_{seed}.json", scenario)
            scenarios.append(scenario)
    if (
        digest(landscape_path) != source_digest
        or qualify_models(landscape.config, models)[1] != model_records
        or execution_binding() != binding
    ):
        raise CaliforniaError("source or model bytes changed during experiment")
    result = Experiment(
        schema_version="geo-infer-place/california-experiment/1",
        config=landscape.config,
        landscape_sha256=source_digest,
        models=model_records,
        backend="geo-infer-exact-joint",
        spatial_backend="binary-mean-field/1",
        observations_kind="simulated-sensors-on-historical-indicators",
        scenarios=tuple(scenarios),
    )
    path = output / "experiment.json"
    write_record(path, result)
    write_record(
        output / "execution.json",
        ExperimentReceipt(
            schema_version="geo-infer-place/california-execution/1",
            execution=binding,
            files={item.name: digest(item) for item in sorted(output.iterdir())},
        ),
    )
    return path
