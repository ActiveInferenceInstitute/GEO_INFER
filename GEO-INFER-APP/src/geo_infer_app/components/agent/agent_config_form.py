"""Data model for an agent configuration form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable


@dataclass
class AgentConfigForm:
    """Container describing an agent configuration form payload."""

    schema: dict[str, Any]
    initial_values: dict[str, Any]
    on_submit: Callable[[dict[str, Any]], Any] | None = None
    on_cancel: Callable[[], Any] | None = None
    is_loading: bool = False
    error: str | None = None

    def submit(self, values: dict[str, Any] | None = None) -> Any | None:
        """Submit a normalized configuration payload."""
        payload = dict(self.initial_values)
        if values:
            payload.update(values)
        if self.on_submit is None:
            return payload
        return self.on_submit(payload)

    def cancel(self) -> Any | None:
        """Execute the optional cancellation handler."""
        if self.on_cancel is None:
            return None
        return self.on_cancel()
