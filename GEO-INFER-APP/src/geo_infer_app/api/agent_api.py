#!/usr/bin/env python3

"""

GEO-INFER-APP Agent API

In-process agent registry for the application. Agents are created,
managed, and inspected entirely within this process: state is persisted
to a JSON file (no HTTP transport, no external GEO-INFER-AGENT service),
with status callbacks, monitoring, and per-agent operational counters.
"""

import os
import asyncio
import logging
import json
import uuid
from typing import Any
from collections.abc import Callable
from datetime import datetime, UTC

# Set up logger
logger = logging.getLogger(__name__)

# Supported command types and their descriptions
_SUPPORTED_COMMANDS = {
    "query": "Query the agent's current state",
    "update": "Update agent parameters",
    "execute": "Execute a specific agent action",
    "pause": "Pause agent execution",
    "resume": "Resume agent execution",
    "reset": "Reset agent state to defaults",
}

# Canonical agent-type vocabulary mirrors AgentType values in
# geo_infer_app.models.agent_interface; "rl" is accepted as an alias.
_AGENT_TYPE_VALUES = {
    "bdi",
    "active_inference",
    "reinforcement_learning",
    "rule_based",
    "hybrid",
}
_AGENT_TYPE_ALIASES = {"rl": "reinforcement_learning"}


class AgentAPIClient:
    """
    In-process agent registry for GEO-INFER-APP.

    Provides methods for creating, managing, and communicating with
    intelligent agents from within the GEO-INFER-APP process. Agent
    records are kept in an in-memory dict persisted to a local JSON
    file; no network transport is involved.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the Agent API client.

        Args:
            config: Configuration options for the API client
        """
        self.config = config or {}
        self.agents: dict[str, dict[str, Any]] = {}
        self.agent_status_callbacks: dict[str, list[Callable[[str, str], None]]] = {}
        self._status_monitoring_task: asyncio.Task[None] | None = None
        # Per-agent operational counters: {agent_id: {"decision_count": int, "success_count": int}}
        self._agent_counters: dict[str, dict[str, int]] = {}

    async def initialize(self) -> None:
        """Initialize the API client: start status monitoring and load persisted agents."""
        logger.info("Initializing Agent API client")
        # Start status monitoring task
        self._status_monitoring_task = asyncio.create_task(self._monitor_agent_status())

        # Load any persisted agent configurations
        await self._load_saved_agents()

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down Agent API client")

        # Cancel status monitoring task
        if self._status_monitoring_task:
            self._status_monitoring_task.cancel()
            try:
                await self._status_monitoring_task
            except asyncio.CancelledError:
                pass

        # Save agent configurations
        await self._save_agents()

    async def create_agent(self, agent_type: str, config: dict[str, Any]) -> str:
        """
        Create a new agent.

        Args:
            agent_type: One of the AgentType values — "bdi",
                "active_inference", "reinforcement_learning",
                "rule_based", "hybrid". The alias "rl" is normalized to
                "reinforcement_learning".
            config: Agent configuration

        Returns:
            Unique ID of the created agent

        Raises:
            ValueError: If agent_type is not a recognized agent type
        """
        normalized_type = _AGENT_TYPE_ALIASES.get(agent_type, agent_type)
        if normalized_type not in _AGENT_TYPE_VALUES:
            raise ValueError(
                f"Unknown agent type '{agent_type}'. "
                f"Supported: {', '.join(sorted(_AGENT_TYPE_VALUES))}"
            )
        logger.info(f"Creating agent of type: {normalized_type}")

        agent_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        self.agents[agent_id] = {
            "id": agent_id,
            "type": normalized_type,
            "config": config,
            "status": "created",
            "created_at": now,
            "last_update": now,
            "started_at": None,
        }
        self._agent_counters[agent_id] = {
            "decision_count": 0,
            "success_count": 0,
            "command_count": 0,
        }

        return agent_id

    async def start_agent(self, agent_id: str) -> bool:
        """
        Start an agent.

        Args:
            agent_id: ID of agent to start

        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return False

        logger.info(f"Starting agent: {agent_id}")
        now = datetime.now(UTC).isoformat()

        self.agents[agent_id]["status"] = "running"
        self.agents[agent_id]["last_update"] = now
        self.agents[agent_id]["started_at"] = now

        await self._notify_status_change(agent_id, "running")
        return True

    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop an agent.

        Args:
            agent_id: ID of agent to stop

        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return False

        logger.info(f"Stopping agent: {agent_id}")
        now = datetime.now(UTC).isoformat()

        self.agents[agent_id]["status"] = "stopped"
        self.agents[agent_id]["last_update"] = now

        await self._notify_status_change(agent_id, "stopped")
        return True

    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent.

        Args:
            agent_id: ID of agent to delete

        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return False

        logger.info(f"Deleting agent: {agent_id}")

        await self._notify_status_change(agent_id, "deleted")

        del self.agents[agent_id]
        self._agent_counters.pop(agent_id, None)
        return True

    async def get_agent_status(self, agent_id: str) -> dict[str, Any] | None:
        """
        Get current status of an agent.

        Args:
            agent_id: ID of agent

        Returns:
            Status information or None if agent not found
        """
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return None

        return self.agents[agent_id].copy()

    async def list_agents(self) -> list[dict[str, Any]]:
        """
        List all agents.

        Returns:
            List of agent information dictionaries
        """
        return list(self.agents.values())

    async def send_command(
        self, agent_id: str, command: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Send a command to an agent and return a structured result.

        Supported command_type values: query, update, execute, pause, resume, reset.

        Args:
            agent_id: ID of agent
            command: Command dict with at minimum a "command_type" key

        Returns:
            Command result dict or None if the agent is not found / not running
        """
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return None

        if self.agents[agent_id]["status"] != "running":
            logger.error(f"Agent not running: {agent_id}")
            return None

        command_type = command.get("command_type", "unknown")
        command_id = command.get("command_id", f"cmd_{uuid.uuid4().hex[:8]}")
        parameters = command.get("parameters", {})

        logger.info(f"Sending command '{command_type}' to agent {agent_id}")

        counters = self._agent_counters.setdefault(
            agent_id, {"decision_count": 0, "success_count": 0, "command_count": 0}
        )
        counters["command_count"] += 1
        counters["decision_count"] += 1

        result = self._route_command(agent_id, command_type, parameters)
        if result.get("status") == "success":
            counters["success_count"] += 1

        self.agents[agent_id]["last_update"] = datetime.now(UTC).isoformat()

        return {
            "status": result["status"],
            "command_id": command_id,
            "command_type": command_type,
            "result": result.get("result"),
            "message": result.get("message", ""),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _route_command(
        self, agent_id: str, command_type: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Route a command to the appropriate handler and return the result.

        Args:
            agent_id: Target agent ID
            command_type: One of the supported command types
            parameters: Command-specific parameters

        Returns:
            Dict with "status", "result", and optional "message" keys
        """
        agent = self.agents[agent_id]

        if command_type == "query":
            return {
                "status": "success",
                "result": {
                    "id": agent_id,
                    "type": agent["type"],
                    "status": agent["status"],
                    "config_keys": list(agent["config"].keys()),
                },
            }

        if command_type == "update":
            updates = parameters.get("config", {})
            if not isinstance(updates, dict):
                return {
                    "status": "error",
                    "message": "parameters.config must be a dict",
                }
            agent["config"].update(updates)
            agent["last_update"] = datetime.now(UTC).isoformat()
            return {
                "status": "success",
                "result": {"updated_keys": list(updates.keys())},
            }

        if command_type == "execute":
            action = parameters.get("action")
            if not action:
                return {"status": "error", "message": "parameters.action is required"}
            return {"status": "success", "result": {"action": action, "executed": True}}

        if command_type == "pause":
            agent["status"] = "paused"
            agent["last_update"] = datetime.now(UTC).isoformat()
            return {"status": "success", "result": {"status": "paused"}}

        if command_type == "resume":
            if agent["status"] != "paused":
                return {
                    "status": "error",
                    "message": f"Cannot resume agent in '{agent['status']}' state",
                }
            agent["status"] = "running"
            agent["last_update"] = datetime.now(UTC).isoformat()
            return {"status": "success", "result": {"status": "running"}}

        if command_type == "reset":
            counters = self._agent_counters.get(agent_id, {})
            counters["decision_count"] = 0
            counters["success_count"] = 0
            counters["command_count"] = 0
            return {"status": "success", "result": {"reset": True}}

        return {
            "status": "error",
            "message": (
                f"Unknown command_type '{command_type}'. "
                f"Supported: {', '.join(_SUPPORTED_COMMANDS)}"
            ),
        }

    async def get_agent_metrics(self, agent_id: str) -> dict[str, Any] | None:
        """
        Get performance metrics for an agent.

        Args:
            agent_id: ID of agent

        Returns:
            Metrics dict or None if agent not found
        """
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return None

        logger.info(f"Getting metrics for agent: {agent_id}")

        agent = self.agents[agent_id]
        counters = self._agent_counters.get(
            agent_id, {"decision_count": 0, "success_count": 0, "command_count": 0}
        )

        decision_count = counters["decision_count"]
        success_count = counters["success_count"]
        success_rate = success_count / decision_count if decision_count > 0 else 0.0

        # Calculate uptime in seconds from started_at timestamp
        uptime_seconds = 0
        started_at = agent.get("started_at")
        if started_at and agent["status"] in ("running", "paused"):
            started_dt = datetime.fromisoformat(started_at)
            now_dt = datetime.now(UTC)
            uptime_seconds = int((now_dt - started_dt).total_seconds())

        return {
            "decision_count": decision_count,
            "success_count": success_count,
            "command_count": counters["command_count"],
            "success_rate": round(success_rate, 4),
            "uptime_seconds": uptime_seconds,
            "status": agent["status"],
        }

    def register_status_callback(
        self, agent_id: str, callback: Callable[[str, str], None]
    ) -> None:
        """
        Register a callback for agent status changes.

        Args:
            agent_id: ID of agent to monitor
            callback: Function called on status change — signature (agent_id, status)
        """
        if agent_id not in self.agent_status_callbacks:
            self.agent_status_callbacks[agent_id] = []

        self.agent_status_callbacks[agent_id].append(callback)

    def unregister_status_callback(
        self, agent_id: str, callback: Callable[[str, str], None]
    ) -> bool:
        """
        Unregister a status callback.

        Args:
            agent_id: ID of agent
            callback: Callback to remove

        Returns:
            True if callback was removed, False otherwise
        """
        if agent_id not in self.agent_status_callbacks:
            return False

        try:
            self.agent_status_callbacks[agent_id].remove(callback)
            return True
        except ValueError:
            return False

    async def _notify_status_change(self, agent_id: str, status: str) -> None:
        """
        Notify all registered callbacks of a status change.

        Args:
            agent_id: ID of agent
            status: New status
        """
        if agent_id in self.agent_status_callbacks:
            for callback in self.agent_status_callbacks[agent_id]:
                try:
                    callback(agent_id, status)
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")

    async def _monitor_agent_status(self) -> None:
        """Periodically check and log the status of all agents."""
        try:
            while True:
                await asyncio.sleep(5)

                running = sum(
                    1 for a in self.agents.values() if a.get("status") == "running"
                )
                if self.agents:
                    logger.debug(
                        f"Status monitor: {running}/{len(self.agents)} agents running"
                    )
        except asyncio.CancelledError:
            logger.info("Agent status monitoring task cancelled")
            raise

    async def _load_saved_agents(self) -> None:
        """Load saved agent configurations from disk."""
        config_path = self.config.get(
            "agents_config_path",
            os.path.join(
                os.path.expanduser("~"), ".geo_infer_app", "agent_configs.json"
            ),
        )

        if not os.path.exists(config_path):
            return

        try:
            with open(config_path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            # ValueError covers json.JSONDecodeError. Surface the corrupt
            # file instead of silently continuing with empty state.
            logger.error(
                f"Could not load saved agent configurations from {config_path} "
                f"({e}); starting with an empty agent registry"
            )
            return

        agents = data.get("agents", {})
        counters = data.get("counters", {})
        if not isinstance(agents, dict) or not isinstance(counters, dict):
            logger.error(
                f"Saved agent configurations in {config_path} have unexpected "
                "shape; expected objects for 'agents' and 'counters' — "
                "starting with an empty agent registry"
            )
            return
        self.agents = agents
        self._agent_counters = counters
        logger.info(f"Loaded {len(self.agents)} agent configurations")

    async def _save_agents(self) -> None:
        """Save agent configurations to disk."""
        config_path = self.config.get(
            "agents_config_path",
            os.path.join(
                os.path.expanduser("~"), ".geo_infer_app", "agent_configs.json"
            ),
        )

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(
                {"agents": self.agents, "counters": self._agent_counters},
                f,
                indent=2,
            )
        logger.info(f"Saved {len(self.agents)} agent configurations")


class AgentManager:
    """
    High-level manager for agents in the application.

    Provides a simplified interface for working with agents
    and manages agent lifecycle in the application context.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the agent manager.

        Args:
            config: Configuration options
        """
        self.config = config or {}
        api_config = self.config.get("api_config") or {}
        self.api_client = AgentAPIClient(api_config)
        self.active_agents: set = set()

    async def initialize(self) -> None:
        """Initialize the agent manager."""
        await self.api_client.initialize()

        if self.config.get("auto_start_agents", False):
            await self._start_saved_agents()

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        for agent_id in list(self.active_agents):
            await self.stop_agent(agent_id)

        await self.api_client.shutdown()

    async def create_agent(
        self, agent_type: str, name: str, config: dict[str, Any]
    ) -> str:
        """
        Create a new agent with the given configuration.

        Args:
            agent_type: One of the AgentType values — "bdi",
                "active_inference", "reinforcement_learning",
                "rule_based", "hybrid" ("rl" is accepted as an alias)
            name: Human-readable name for the agent
            config: Agent configuration

        Returns:
            ID of the created agent
        """
        config["name"] = name
        return await self.api_client.create_agent(agent_type, config)

    async def start_agent(self, agent_id: str) -> bool:
        """
        Start an agent.

        Args:
            agent_id: ID of agent to start

        Returns:
            True if successful, False otherwise
        """
        success = await self.api_client.start_agent(agent_id)
        if success:
            self.active_agents.add(agent_id)
        return success

    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop an agent.

        Args:
            agent_id: ID of agent to stop

        Returns:
            True if successful, False otherwise
        """
        success = await self.api_client.stop_agent(agent_id)
        if success and agent_id in self.active_agents:
            self.active_agents.remove(agent_id)
        return success

    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent.

        Args:
            agent_id: ID of agent to delete

        Returns:
            True if successful, False otherwise
        """
        if agent_id in self.active_agents:
            await self.stop_agent(agent_id)

        return await self.api_client.delete_agent(agent_id)

    async def send_command(
        self,
        agent_id: str,
        command_type: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Send a command to an agent.

        Args:
            agent_id: ID of agent
            command_type: Type of command to send
            parameters: Command parameters

        Returns:
            Command result or None if failed
        """
        command = {
            "command_type": command_type,
            "command_id": f"cmd_{uuid.uuid4().hex[:8]}",
            "parameters": parameters or {},
        }
        return await self.api_client.send_command(agent_id, command)

    async def get_agent_info(self, agent_id: str) -> dict[str, Any] | None:
        """
        Get information about an agent.

        Args:
            agent_id: ID of agent

        Returns:
            Agent information or None if not found
        """
        return await self.api_client.get_agent_status(agent_id)

    async def list_agents(
        self,
        filter_type: str | None = None,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List agents, optionally filtered.

        Args:
            filter_type: Filter by agent type
            active_only: Only include active agents

        Returns:
            List of agent information dictionaries
        """
        agents = await self.api_client.list_agents()

        if active_only:
            agents = [a for a in agents if a["id"] in self.active_agents]

        if filter_type:
            agents = [a for a in agents if a["type"] == filter_type]

        return agents

    async def get_agent_metrics(self, agent_id: str) -> dict[str, Any] | None:
        """
        Get performance metrics for an agent.

        Args:
            agent_id: ID of agent

        Returns:
            Metrics or None if agent not found
        """
        return await self.api_client.get_agent_metrics(agent_id)

    def register_status_callback(
        self, agent_id: str, callback: Callable[[str, str], None]
    ) -> None:
        """
        Register a callback for agent status changes.

        Args:
            agent_id: ID of agent to monitor
            callback: Function to call when status changes
        """
        self.api_client.register_status_callback(agent_id, callback)

    def unregister_status_callback(
        self, agent_id: str, callback: Callable[[str, str], None]
    ) -> bool:
        """
        Unregister a status callback.

        Args:
            agent_id: ID of agent
            callback: Callback to remove

        Returns:
            True if callback was removed, False otherwise
        """
        return self.api_client.unregister_status_callback(agent_id, callback)

    async def _start_saved_agents(self) -> None:
        """Start all previously saved agents marked as active."""
        agents = await self.api_client.list_agents()
        for agent in agents:
            if agent.get("status") == "running":
                await self.start_agent(agent["id"])
