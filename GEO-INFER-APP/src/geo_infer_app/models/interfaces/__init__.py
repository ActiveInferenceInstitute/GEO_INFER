"""
Agent Interface Implementations

This package provides concrete implementations of the AgentInterface for
different agent types. Each implementation connects to the corresponding
agent implementation in GEO-INFER-AGENT.

Importing this package explicitly registers every bundled interface with
``AgentFactory`` (currently only the BDI interface ships with APP).
"""

from geo_infer_app.models.interfaces import bdi_interface

__all__ = ["bdi_interface"]
