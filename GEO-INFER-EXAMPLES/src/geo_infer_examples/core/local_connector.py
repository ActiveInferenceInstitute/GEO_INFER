"""Explicit in-process operations for the existing workflow engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import inspect
import math
import time
from typing import Any

from .module_orchestrator import APIConnector


@dataclass(frozen=True)
class LocalResponse:
    """A completed local operation using the connector response interface."""

    payload: dict[str, Any]
    status_code: int = 200
    text: str = ""

    def json(self) -> dict[str, Any]:
        """Return native local results; no serialization or network is involved."""
        return self.payload


class LocalAPIConnector(APIConnector):
    """Run explicitly supplied module/endpoint callables in the current process.

    Each callable accepts one input dictionary and returns a dictionary or an
    awaitable dictionary. Inputs are copied independently for each operation.
    Async work supports cancellation. Synchronous work runs on the event loop:
    its elapsed deadline is checked after return and cannot preempt native code.
    Use an async subprocess adapter when hard termination is required.
    """

    def __init__(
        self, operations: Mapping[tuple[str, str], Callable[[dict[str, Any]], Any]]
    ) -> None:
        self._operations = dict(operations)
        for key, operation in self._operations.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(value, str) and value for value in key)
                or not key[1].startswith("/")
                or key[1] == "/health"
                or not callable(operation)
            ):
                raise ValueError(
                    "operations require (module, /endpoint) callable entries"
                )
        self.declared_modules = tuple(sorted({key[0] for key in self._operations}))

    def get(self, *, module: str, endpoint: str, **_: Any) -> LocalResponse:
        """Report only the locally registered module's actual operations."""
        if module not in self.declared_modules or endpoint != "/health":
            raise KeyError(f"Unregistered local endpoint: {module}{endpoint}")
        return LocalResponse(
            {
                "capabilities": sorted(
                    path for name, path in self._operations if name == module
                )
            }
        )

    async def get_async(self, **kwargs: Any) -> LocalResponse:
        """Read local health without network access."""
        return self.get(**kwargs)

    async def post_async(
        self,
        *,
        module: str,
        endpoint: str,
        data: dict[str, Any],
        timeout: float,
        **_: Any,
    ) -> LocalResponse:
        """Execute one registered operation and propagate its actual errors."""
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("operation timeout must be finite and positive")
        operation = self._operations.get((module, endpoint))
        if operation is None:
            raise KeyError(f"Unregistered local endpoint: {module}{endpoint}")
        started = time.monotonic()
        result = operation(deepcopy(data))
        if inspect.isawaitable(result):
            result = await result
        if time.monotonic() - started >= timeout:
            raise TimeoutError(
                f"Local operation exceeded its deadline: {module}{endpoint}"
            )
        if not isinstance(result, dict):
            raise TypeError("local operations must return a dictionary")
        return LocalResponse(result)
