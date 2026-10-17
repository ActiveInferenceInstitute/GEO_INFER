"""Shared trusted-local execution for scripts, JSON CLI and MCP.

This is a Python API bridge, not a sandbox. Calls have the authority and side
effects of the underlying API. No eval, pickle, implicit network service or
automatic retries are used. Handles retain native scientific objects in memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import math
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from geo_infer_api.agent_catalog import MODULES, catalog, module_inventory, package_name
from geo_infer_api.agent_schema import request_schema, validate_request
from geo_infer_api.agent_skills import read_guidance, skill_inventory
from geo_infer_api.agent_diagnostics import runtime_diagnostics

_PACKAGE_NAMES = frozenset(package_name(name) for name in MODULES)

_CODEC_FIELDS = {
    "ndarray": {"dtype"},
    "dataframe": {"columns", "index", "dtype"},
    "series": {"index", "name", "dtype"},
    "dataarray": {"dims", "coords", "name", "attrs"},
    "dataset": set(),
    "geodataframe": {"crs"},
    "geometry": set(),
    "datetime": set(),
    "date": set(),
    "path": set(),
    "tuple": set(),
    "set": set(),
    "float": set(),
}


def resolve(target: str) -> Any:
    """Resolve an explicit public GEO member; imports retain native failures."""
    module, separator, member = target.partition(":")
    parts = module.split(".")
    if (
        not separator
        or not member
        or any(
            not part.isidentifier() or part.startswith("_")
            for part in (*parts, *member.split("."))
        )
    ):
        raise ValueError("Use a public target 'geo_infer_module.submodule:Member'")
    if parts[0] not in _PACKAGE_NAMES:
        raise ValueError("Only GEO-INFER packages may be resolved")
    value = importlib.import_module(module)
    for name in member.split("."):
        value = getattr(value, name)
    return value


def _public_member(value: Any, name: str) -> Any:
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError("Object member must be one public identifier")
    return getattr(value, name)


def describe(value: Any) -> dict[str, Any]:
    """Return runtime signatures, parameter kinds and docs without calling APIs."""
    parameters = []
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for parameter in signature.parameters.values():
            parameters.append(
                {
                    "name": parameter.name,
                    "kind": parameter.kind.name,
                    "required": parameter.default is inspect.Parameter.empty
                    and parameter.kind
                    not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD),
                    "annotation": None
                    if parameter.annotation is inspect.Parameter.empty
                    else inspect.formatannotation(parameter.annotation),
                    "default": None
                    if parameter.default is inspect.Parameter.empty
                    else repr(parameter.default)[:500],
                }
            )
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "callable": callable(value),
        "signature": str(signature) if signature is not None else None,
        "parameters": parameters,
        "doc": inspect.getdoc(value) or "",
        "members": sorted(name for name in dir(value) if not name.startswith("_")),
    }


class AgentSession:
    """Explicit object lifecycle and sequential composition for agent operations.

    JSON arguments support $ref handles, $callable GEO targets, and tagged numpy,
    pandas, date/time, path, tuple, set and nonfinite-float values. Native objects
    that exceed JSON limits stay in memory and return a handle, never repr data.
    """

    def __init__(self, max_handles: int = 256, max_items: int = 10000) -> None:
        if any(
            type(bound) is not int or bound < 1 for bound in (max_handles, max_items)
        ):
            raise ValueError("Session bounds must be positive")
        self.max_handles = max_handles
        self.max_items = max_items
        self.objects: dict[str, Any] = {}
        self.contexts: dict[str, contextlib.AsyncExitStack] = {}
        self.closed = False

    async def _close(self, exc_info: tuple = (None, None, None)) -> dict[str, Any]:
        counts = {
            "closed": True,
            "released_handles": len(self.objects),
            "exited_contexts": len(self.contexts),
        }
        stack = contextlib.AsyncExitStack()
        try:
            for entry in self.contexts.values():
                stack.push_async_exit(entry)
            await stack.__aexit__(*exc_info)
        finally:
            self.contexts.clear()
            self.objects.clear()
            self.closed = True
        return counts

    async def aclose(self) -> dict[str, Any]:
        """Exit owned contexts in reverse order and release all handles.

        Arbitrary objects' close methods are never invoked implicitly. Use
        enter/exit for resources requiring deterministic native cleanup.
        """
        with contextlib.redirect_stdout(sys.stderr):
            return await self._close()

    def close(self) -> dict[str, Any]:
        """Synchronous cleanup; async applications use ``await aclose``."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aclose())
        raise RuntimeError("Use await session.aclose() inside an event loop")

    def __enter__(self) -> AgentSession:
        if self.closed:
            raise RuntimeError("Session is closed")
        return self

    def __exit__(self, *exc_info) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            with contextlib.redirect_stdout(sys.stderr):
                asyncio.run(self._close(exc_info))
            return False
        raise RuntimeError("Use async with AgentSession() inside an event loop")

    async def __aenter__(self) -> AgentSession:
        return self.__enter__()

    async def __aexit__(self, *exc_info) -> bool:
        with contextlib.redirect_stdout(sys.stderr):
            await self._close(exc_info)
        return False

    def _store(self, value: Any, name: str | None = None) -> dict[str, str]:
        name = name or uuid4().hex
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError("Handle name must contain 1 to 128 characters")
        if name in self.objects:
            raise ValueError(f"Handle already exists: {name}")
        if len(self.objects) >= self.max_handles:
            raise ValueError("Handle limit reached; release unused objects")
        self.objects[name] = value
        return {
            "$ref": name,
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }

    def decode(self, value: Any) -> Any:
        """Decode explicit native types without interpreting arbitrary Python."""
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$literal" in value:
            if set(value) != {"$literal"}:
                raise ValueError("$literal must be the only field")
            return value["$literal"]
        if "$ref" in value:
            if set(value) - {"$ref", "type", "path"}:
                raise ValueError("Unknown $ref fields; use $literal for literal data")
            if not isinstance(value.get("path", []), list):
                raise ValueError("$ref path must be an array of index values")
            result = self.objects[value["$ref"]]
            for part in value.get("path", []):
                result = result[self.decode(part)]
            return result
        if "$callable" in value:
            if set(value) != {"$callable"}:
                raise ValueError("$callable must be the only field")
            result = resolve(value["$callable"])
            if not callable(result):
                raise ValueError("$callable must resolve to a callable")
            return result
        if "$type" in value:
            tag = value["$type"]
            if not isinstance(tag, str) or tag not in _CODEC_FIELDS:
                raise ValueError(f"Unknown $type: {tag}")
            unexpected = set(value) - {"$type", "data"} - _CODEC_FIELDS[tag]
            if unexpected or "data" not in value:
                raise ValueError(
                    f"$type {tag!r} requires data and rejects unknown fields: {sorted(unexpected)}"
                )
            data = self.decode(value["data"])
            if tag == "ndarray":
                import numpy as np

                return np.asarray(data, dtype=self.decode(value.get("dtype")))
            if tag in {"dataframe", "series"}:
                import pandas as pd

                if tag == "dataframe":
                    return pd.DataFrame(
                        data,
                        columns=self.decode(value.get("columns")),
                        index=self.decode(value.get("index")),
                        dtype=self.decode(value.get("dtype")),
                    )
                return pd.Series(
                    data,
                    index=self.decode(value.get("index")),
                    name=value.get("name"),
                    dtype=self.decode(value.get("dtype")),
                )
            if tag == "dataarray":
                import xarray as xr

                return xr.DataArray(
                    data,
                    dims=self.decode(value.get("dims")),
                    coords=self.decode(value.get("coords")),
                    name=value.get("name"),
                    attrs=self.decode(value.get("attrs")),
                )
            if tag == "dataset":
                import xarray as xr

                return xr.Dataset.from_dict(data)
            if tag == "geodataframe":
                import geopandas as gpd

                if isinstance(data, dict) and data.get("features") == []:
                    return gpd.GeoDataFrame(
                        geometry=[], crs=self.decode(value.get("crs", "EPSG:4326"))
                    )
                return gpd.GeoDataFrame.from_features(
                    data, crs=self.decode(value.get("crs", "EPSG:4326"))
                )
            if tag == "geometry":
                from shapely.geometry import shape

                return shape(data)
            if tag == "datetime":
                return datetime.fromisoformat(data)
            if tag == "date":
                return date.fromisoformat(data)
            if tag == "path":
                return Path(data)
            if tag == "tuple":
                return tuple(data)
            if tag == "set":
                return set(data)
            if tag == "float" and data in {"nan", "inf", "-inf"}:
                return float(data)
            raise ValueError(f"Unknown or invalid $type: {tag}")
        return {key: self.decode(item) for key, item in value.items()}

    def _json(self, value: Any, budget: list[int], seen: set[int]) -> Any:
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("Result exceeds JSON item limit")
        if isinstance(value, Enum):
            return self._json(value.value, budget, seen)
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return (
                value
                if math.isfinite(value)
                else {"$type": "float", "data": str(value)}
            )
        if isinstance(value, (datetime, date, Path)):
            tag = (
                "datetime"
                if isinstance(value, datetime)
                else "date"
                if isinstance(value, date)
                else "path"
            )
            return {"$type": tag, "data": str(value)}
        if id(value) in seen:
            raise ValueError("Cyclic result requires a native handle")
        seen.add(id(value))
        try:
            if isinstance(value, dict):
                if not all(isinstance(key, str) for key in value):
                    raise ValueError(
                        "Non-string dictionary keys require a native handle"
                    )
                return {
                    key: self._json(item, budget, seen) for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [self._json(item, budget, seen) for item in value]
            # Import no optional backend just to encode an unrelated result.
            root = type(value).__module__.split(".")[0]
            if root == "numpy" and hasattr(value, "tolist"):
                if getattr(value, "size", 1) > budget[0]:
                    raise ValueError("Array exceeds JSON item limit")
                return self._json(value.tolist(), budget, seen)
            if root == "pydantic" or any(
                base.__module__.startswith("pydantic") for base in type(value).__mro__
            ):
                return self._json(value.model_dump(mode="python"), budget, seen)
            raise ValueError("Result requires a native handle")
        finally:
            seen.remove(id(value))

    def encode(self, value: Any) -> Any:
        """Encode finite JSON losslessly where possible; retain opaque objects."""
        try:
            return self._json(value, [self.max_items], set())
        except (TypeError, ValueError, RecursionError):
            return self._store(value)

    async def _dispatch(self, request: dict[str, Any]) -> Any:
        op = request["op"]
        if op == "diagnostics":
            return {
                **runtime_diagnostics(),
                "session": {
                    "closed": self.closed,
                    "handles": len(self.objects),
                    "active_contexts": len(self.contexts),
                    "max_handles": self.max_handles,
                    "max_items": self.max_items,
                },
            }
        if op == "close":
            return await self.aclose()
        if op == "schema":
            return request_schema()
        if op == "skills":
            return skill_inventory()
        if op in {"skill", "guide"}:
            return read_guidance("GUIDE" if op == "guide" else request["module"])
        if op == "modules":
            return module_inventory()
        if op == "catalog":
            return catalog(
                request["module"],
                request.get("query", ""),
                request.get("offset", 0),
                request.get("limit", 50),
            )
        if op == "release":
            name = request["handle"]
            if name in self.contexts:
                raise ValueError("Exit the active context before releasing its handle")
            del self.objects[name]
            return {"released": name}
        if op in {"enter", "exit"}:
            name = request["handle"]
            if op == "enter":
                if name in self.contexts:
                    raise ValueError("Context is already entered")
                manager = self.objects[name]
                entry = contextlib.AsyncExitStack()
                # Native protocols look up special methods on the type, and
                # bind the exit before entry. Retain that exact callback.
                if hasattr(type(manager), "__aenter__"):
                    value = await entry.enter_async_context(manager)
                else:
                    value = entry.enter_context(manager)
                self.contexts[name] = entry
                return (
                    self._store(value, request["save_as"])
                    if "save_as" in request
                    else self.encode(value)
                )
            if name not in self.contexts:
                raise ValueError("Context was not entered by this session")
            entry = self.contexts.pop(name)
            await entry.aclose()
            return {"exited": name}
        if op == "handles":
            return [
                {
                    "$ref": name,
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                }
                for name, value in self.objects.items()
            ]
        if op == "batch":
            steps = request["steps"]
            if not isinstance(steps, list) or not 1 <= len(steps) <= 100:
                raise ValueError("batch requires 1 to 100 steps")
            results = []
            for step in steps:
                if not isinstance(step, dict) or step.get("op") == "batch":
                    raise ValueError("Each step must be a non-batch request")
                result = await self._execute_validated(step)
                results.append(result)
                if not result["ok"]:
                    break
            return {
                "completed": len(results),
                "total": len(steps),
                "steps": results,
                "ok": len(results) == len(steps) and all(row["ok"] for row in results),
            }
        if op not in {"describe", "call", "get"}:
            raise ValueError(f"Unknown operation: {op}")
        if "handle" in request:
            value = self.objects[request["handle"]]
            if request.get("member"):
                value = _public_member(value, request["member"])
        else:
            value = resolve(request["target"])
        if op == "describe":
            return describe(value)
        if op == "call":
            args = self.decode(request.get("args", []))
            kwargs = self.decode(request.get("kwargs", {}))
            if not isinstance(args, list) or not isinstance(kwargs, dict):
                raise ValueError("args must be an array and kwargs an object")
            if not callable(value):
                raise ValueError("Selected value is not callable; use get")
            # Binding validates arguments before running code; extension types
            # without signatures retain their own native argument validation.
            try:
                signature = inspect.signature(value)
            except (ValueError, TypeError):
                signature = None
            if signature is not None:
                signature.bind(*args, **kwargs)
            value = value(*args, **kwargs)
            if inspect.isawaitable(value):
                value = await value
        if "save_as" in request:
            return self._store(value, request["save_as"])
        return self.encode(value)

    async def aexecute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute once, returning a uniform result/error envelope.

        Redirect Python prints to stderr; process entrypoints also isolate
        native/child stdout at the OS descriptor level.
        Stdout redirection is process-wide: execute sessions sequentially in
        one process, using separate worker processes for parallel execution.
        A failure may follow side effects; a batch is not a transaction.
        """
        try:
            request = validate_request(request)
        except (Exception, SystemExit) as error:
            return self._error(error)
        return await self._execute_validated(request)

    @staticmethod
    def _error(error: BaseException) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }

    async def _execute_validated(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            if self.closed and request["op"] not in {"close", "diagnostics", "handles"}:
                raise RuntimeError("Session is closed; create a new AgentSession")
            if "save_as" in request:
                name = request["save_as"]
                if not isinstance(name, str) or not name or len(name) > 128:
                    raise ValueError("save_as must contain 1 to 128 characters")
                if name in self.objects or len(self.objects) >= self.max_handles:
                    raise ValueError("save_as exists or handle limit reached")
            with contextlib.redirect_stdout(sys.stderr):
                result = await self._dispatch(request)
            ok = not (request.get("op") == "batch" and not result["ok"])
            return {"ok": ok, "result": result}
        except (Exception, SystemExit) as error:
            return self._error(error)

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Synchronous script API; async applications use ``await aexecute``."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aexecute(request))
        raise RuntimeError("Use await session.aexecute(request) inside an event loop")
