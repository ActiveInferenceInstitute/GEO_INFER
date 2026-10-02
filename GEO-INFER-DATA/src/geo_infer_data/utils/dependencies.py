"""Load optional backend clients only when the selected operation needs them."""

from importlib import import_module
from types import ModuleType


class MissingOptionalDependency(ImportError):
    """A selected DATA integration requires an uninstalled package extra."""


def require_dependency(module: str, extra: str) -> ModuleType:
    """Import an integration, preserving errors inside installed libraries."""
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module.split(".")[0]:
            raise
        raise MissingOptionalDependency(
            f"This operation requires {module}; install geo-infer-data[{extra}]"
        ) from exc
