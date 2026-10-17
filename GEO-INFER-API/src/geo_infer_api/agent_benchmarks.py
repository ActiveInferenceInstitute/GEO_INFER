"""Repeatable native agent workloads; elapsed samples include request overhead."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

from .agent_catalog import MODULES, definitions
from .agent_diagnostics import runtime_diagnostics
from .agent_runtime import AgentSession


async def benchmark_agent(
    *, iterations: int = 100, repeats: int = 5, warmup: int = 1
) -> dict[str, Any]:
    """Measure schema, catalog, direct calls and 25-step batches sequentially.

    Compare unprofiled samples on the same host, source and environment. Warmup
    is outside measured samples; batch throughput counts completed steps.
    Scientific backends are not imported or qualified by this control workload.
    """
    for name, value, minimum, maximum in [
        ("iterations", iterations, 1, 100000),
        ("repeats", repeats, 1, 100),
        ("warmup", warmup, 0, 100),
    ]:
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    started = time.perf_counter()
    counts = {module: len(definitions(module)) for module in MODULES}
    catalog_seconds = time.perf_counter() - started
    call = dict(
        op="call", target="geo_infer_api.agent_catalog:package_name", args=["API"]
    )
    cases = {
        "schema": dict(op="schema"),
        "catalog": dict(op="catalog", module="API", limit=5),
        "call": call,
        "batch": dict(op="batch", steps=[call for _ in range(25)]),
    }
    measurements = {}
    async with AgentSession() as session:
        for name, request in cases.items():
            samples = []
            for repeat in range(warmup + repeats):
                started = time.perf_counter()
                for _ in range(iterations):
                    result = await session.aexecute(request)
                    if not result["ok"]:
                        raise RuntimeError(f"{name} workload failed: {result}")
                if repeat >= warmup:
                    samples.append(time.perf_counter() - started)
            operations = iterations * (25 if name == "batch" else 1)
            median = statistics.median(samples)
            measurements[name] = dict(
                samples_seconds=samples,
                median_seconds=median,
                min_seconds=min(samples),
                max_seconds=max(samples),
                operations_per_sample=operations,
                median_operations_per_second=operations / median if median else None,
            )
    return dict(
        schema="geo-infer-agent-benchmark/1",
        runtime=runtime_diagnostics(),
        iterations=iterations,
        repeats=repeats,
        warmup=warmup,
        catalog_scan_seconds=catalog_seconds,
        definitions=sum(counts.values()),
        module_definitions=counts,
        measurements=measurements,
    )


def main(argv: list[str] | None = None) -> int:
    """Emit JSON measurements; failed operations return a nonzero exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    options = vars(parser.parse_args(argv))
    try:
        result = asyncio.run(benchmark_agent(**options))
    except Exception as error:
        print(
            json.dumps(
                dict(
                    success=False,
                    error=dict(type=type(error).__name__, message=str(error)),
                )
            )
        )
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
