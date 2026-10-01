#!/usr/bin/env python3
"""GEO-INFER-APP module orchestrator.

Runs one documented end-to-end APP operation on synthetic UI/session data:
validate a synthetic agent configuration against the module's schema, create
a real BDI agent interface through ``AgentFactory``, drive a fleet of three
synthetic field-survey agents through the full BDI cycle (belief update,
desire addition, deliberation, execution, movement), filter the roster by
status and location radius, convert live agent states into map and dashboard
payloads via ``AgentVisualization``, and count lifecycle events captured by a
registered event handler. No server bind and no network: everything runs
in-process on the module's public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_app.models.agent_configuration import AgentConfiguration, AgentType
    from geo_infer_app.models.agent_factory import AgentFactory
    from geo_infer_app.models.agent_visualization import AgentVisualization
    from geo_infer_app.models.interfaces.bdi_interface import BDIAgentInterface

    # --- Configuration schema validation on synthetic UI submissions ----
    default_config = AgentConfiguration.get_default_config(AgentType.BDI)
    valid_submission: dict[str, Any] = {
        **default_config,
        "name": "Del Norte Field Survey Fleet",
        "description": "Synthetic UI session for a three-agent BDI survey fleet",
    }
    validation_errors_valid = AgentConfiguration.validate_config(
        AgentType.BDI, valid_submission
    )

    invalid_submission: dict[str, Any] = {
        "description": 123,
        "unexpected_field": True,
    }
    validation_errors_invalid = AgentConfiguration.validate_config(
        AgentType.BDI, invalid_submission
    )

    # --- Factory: create the BDI interface and a synthetic agent fleet ---
    available_types = AgentFactory.get_available_agent_types()
    interface = AgentFactory.create_interface(AgentType.BDI)
    if not isinstance(interface, BDIAgentInterface):
        raise RuntimeError(
            f"AgentFactory returned {type(interface).__name__}, "
            "expected BDIAgentInterface"
        )

    events: list[dict[str, Any]] = []
    interface.register_event_handler(
        "agent_created", lambda payload: events.append({"type": "agent_created"})
    )
    interface.register_event_handler(
        "agent_updated", lambda payload: events.append({"type": "agent_updated"})
    )

    fleet_specs: list[dict[str, Any]] = [
        {
            "agent_id": "ui-agent-001",
            "name": "Scout-Alpha",
            "beliefs": {"sector": "redwood_grove", "battery": 0.92},
            "desires": ["survey_sector"],
            "initial_location": {"lat": 41.7558, "lng": -124.2026},
        },
        {
            "agent_id": "ui-agent-002",
            "name": "Scout-Bravo",
            "beliefs": {"sector": "riparian_corridor", "battery": 0.64},
            "desires": ["monitor_waterline"],
            "initial_location": {"lat": 41.7900, "lng": -124.1500},
        },
        {
            "agent_id": "ui-agent-003",
            "name": "Scout-Charlie",
            "beliefs": {"sector": "ridge_line", "battery": 0.11},
            "desires": ["return_to_base"],
            "initial_location": {"lat": 42.0060, "lng": -123.9000},
        },
    ]

    created_ids: list[str] = []
    for spec in fleet_specs:
        agent_id = interface.create_agent(AgentType.BDI, spec)
        created_ids.append(agent_id)

    # --- Drive the full BDI cycle over the synthetic session -------------
    command_results: dict[str, Any] = {}
    command_results["add_belief"] = interface.send_command(
        "ui-agent-001",
        "add_belief",
        {"belief": {"fog_detected": True}},
    )
    command_results["add_desire"] = interface.send_command(
        "ui-agent-001",
        "add_desire",
        {"desire": {"name": "photograph_canopy", "priority": 0.7}},
    )
    command_results["deliberate"] = interface.send_command(
        "ui-agent-001", "deliberate", {}
    )
    state_after_deliberate = interface.get_agent_state("ui-agent-001")
    intentions_after_deliberate = len(state_after_deliberate.metadata["intentions"])
    command_results["execute"] = interface.send_command("ui-agent-001", "execute", {})
    state_after_execute = interface.get_agent_state("ui-agent-001")
    command_results["move"] = interface.send_command(
        "ui-agent-002",
        "move",
        {"location": {"lat": 41.7800, "lng": -124.1600}},
    )

    # --- Roster queries: unfiltered, by status, by location radius -------
    all_agents = interface.list_agents()
    idle_agents = interface.list_agents(filter_params={"status": "idle"})
    agents_near_base = interface.list_agents(
        filter_params={
            "location": {
                "center": {"lat": 41.7558, "lng": -124.2026},
                "radius": 10.0,
            }
        }
    )

    # --- Visualization payloads from live agent state --------------------
    map_feature = AgentVisualization.state_to_map_feature(
        interface.get_agent_state("ui-agent-002")
    )
    dashboard = AgentVisualization.state_to_dashboard_data(
        interface.get_agent_state("ui-agent-001")
    )

    event_counts: dict[str, int] = {}
    for event in events:
        event_counts[event["type"]] = event_counts.get(event["type"], 0) + 1

    return {
        "operation": "bdi_fleet_session_on_synthetic_ui_data",
        "configuration": {
            "default_fields": sorted(str(k) for k in default_config.keys()),
            "valid_submission_errors": validation_errors_valid,
            "invalid_submission_errors": validation_errors_invalid,
        },
        "factory": {
            "available_agent_types": available_types,
            "interface_created": type(interface).__name__,
        },
        "fleet": {
            "created_agents": created_ids,
            "total_agents": len(all_agents),
            "idle_agents_after_execute": len(idle_agents),
            "agents_within_10km_of_base": len(agents_near_base),
        },
        "command_results": command_results,
        "bdi_state_after_deliberate": {
            "intentions": intentions_after_deliberate,
            "status_after_execute": state_after_execute.status,
        },
        "visualization": {
            "map_feature_keys": sorted(map_feature.keys()),
            "dashboard_title": dashboard.get("title"),
            "dashboard_status": dashboard.get("status"),
            "dashboard_widgets": sorted(dashboard.get("widgets", {}).keys()),
        },
        "event_counts": event_counts,
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("APP", _operation))
