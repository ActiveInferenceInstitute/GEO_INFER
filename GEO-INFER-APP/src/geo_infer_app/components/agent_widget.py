#!/usr/bin/env python3

"""
GEO-INFER-APP Agent Widget

This module provides a widget for displaying and interacting with
intelligent agents in the application interface.
"""

import asyncio
import html
import logging
from typing import Any
from collections.abc import Callable
from datetime import datetime

from geo_infer_app.api.agent_api import AgentManager

# Set up logger
logger = logging.getLogger(__name__)


class AgentWidget:
    """
    Widget for displaying and interacting with agents.

    This is a base class that can be extended for specific UI frameworks
    (e.g., web, desktop, mobile).
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        agent_id: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the agent widget.

        Args:
            agent_manager: Agent manager instance
            agent_id: ID of agent to display (if None, allows selection)
            config: Widget configuration
        """
        self.agent_manager = agent_manager
        self.agent_id = agent_id
        self.config = config or {}

        # Widget state
        self.status = "initializing"
        self.agent_info: dict[str, Any] = {}
        self.agent_metrics: dict[str, Any] = {}
        self.command_history: list[dict[str, Any]] = []
        self.max_history = self.config.get("max_history", 100)

        # Update interval (milliseconds)
        self.update_interval = self.config.get("update_interval", 1000)
        self._update_task: asyncio.Task[None] | None = None

        # Callbacks
        self.status_callbacks: list[Callable[[str], None]] = []

    async def initialize(self) -> None:
        """Initialize the widget."""
        logger.info(f"Initializing agent widget for agent: {self.agent_id}")

        # Register for status updates
        if self.agent_id:
            self.agent_manager.register_status_callback(
                self.agent_id, self._handle_status_change
            )

            # Get initial agent info
            await self._update_agent_info()

        # Start update task
        self._update_task = asyncio.create_task(self._update_loop())

        self.status = "initialized"

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down agent widget")

        # Cancel update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        # Unregister status callback
        if self.agent_id:
            self.agent_manager.unregister_status_callback(
                self.agent_id, self._handle_status_change
            )

        self.status = "shutdown"

    async def set_agent(self, agent_id: str) -> bool:
        """
        Set the agent to display.

        Args:
            agent_id: ID of agent

        Returns:
            True if successful, False otherwise
        """
        # Unregister previous callback
        if self.agent_id:
            self.agent_manager.unregister_status_callback(
                self.agent_id, self._handle_status_change
            )

        # Set new agent
        self.agent_id = agent_id
        self.command_history = []

        # Register for status updates
        if self.agent_id:
            self.agent_manager.register_status_callback(
                self.agent_id, self._handle_status_change
            )

            # Get initial agent info
            await self._update_agent_info()
            return True

        return False

    async def start_agent(self) -> bool:
        """
        Start the current agent.

        Returns:
            True if successful, False otherwise
        """
        if not self.agent_id:
            logger.warning("No agent selected")
            return False

        return await self.agent_manager.start_agent(self.agent_id)

    async def stop_agent(self) -> bool:
        """
        Stop the current agent.

        Returns:
            True if successful, False otherwise
        """
        if not self.agent_id:
            logger.warning("No agent selected")
            return False

        return await self.agent_manager.stop_agent(self.agent_id)

    async def send_command(
        self, command_type: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """
        Send a command to the current agent.

        Args:
            command_type: Type of command to send
            parameters: Command parameters

        Returns:
            Command result or None if failed
        """
        if not self.agent_id:
            logger.warning("No agent selected")
            return None

        result = await self.agent_manager.send_command(
            self.agent_id, command_type, parameters
        )

        if result:
            # Add to command history
            entry = {
                "timestamp": datetime.now().isoformat(),
                "command_type": command_type,
                "parameters": parameters,
                "result": result,
            }

            self.command_history.append(entry)

            # Trim history if needed
            while len(self.command_history) > self.max_history:
                self.command_history.pop(0)

        return result

    async def get_agent_list(self) -> list[dict[str, Any]]:
        """
        Get list of available agents.

        Returns:
            List of agent information dictionaries
        """
        return await self.agent_manager.list_agents()

    def register_status_callback(self, callback: Callable[[str], None]) -> None:
        """
        Register a callback for widget status changes.

        Args:
            callback: Function to call when status changes
        """
        self.status_callbacks.append(callback)

    def unregister_status_callback(self, callback: Callable[[str], None]) -> bool:
        """
        Unregister a status callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was removed, False otherwise
        """
        try:
            self.status_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_status_change(self) -> None:
        """Notify all registered callbacks of a status change."""
        for callback in self.status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def _handle_status_change(self, agent_id: str, status: str) -> None:
        """
        Handle agent status change.

        Args:
            agent_id: ID of agent
            status: New status
        """
        # Only process if it's our agent
        if agent_id != self.agent_id:
            return

        # Update our status
        self.status = f"agent_{status}"

        # Notify callbacks
        self._notify_status_change()

    async def _update_loop(self) -> None:
        """Periodically update agent information."""
        try:
            while True:
                if self.agent_id:
                    await self._update_agent_info()

                # Wait for next update
                await asyncio.sleep(self.update_interval / 1000)
        except asyncio.CancelledError:
            logger.info("Agent widget update task cancelled")
            raise

    async def _update_agent_info(self) -> None:
        """Update agent information and metrics."""
        if not self.agent_id:
            return

        try:
            # Get agent info
            info = await self.agent_manager.get_agent_info(self.agent_id)
            if info:
                self.agent_info = info

            # Get agent metrics
            metrics = await self.agent_manager.get_agent_metrics(self.agent_id)
            if metrics:
                self.agent_metrics = metrics
        except Exception as e:
            logger.error(f"Error updating agent information: {e}")


class WebAgentWidget(AgentWidget):
    """
    Web-specific implementation of the agent widget.

    This class extends the base AgentWidget with web-specific
    functionality for use in the web interface.
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        agent_id: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the web agent widget.

        Args:
            agent_manager: Agent manager instance
            agent_id: ID of agent to display
            config: Widget configuration
        """
        super().__init__(agent_manager, agent_id, config)

        # Web-specific properties
        self.element_id = self.config.get("element_id", "agent-widget")
        self.css_class = self.config.get("css_class", "agent-widget")
        self.template = self.config.get("template", "default")

    def render(self) -> str:
        """
        Render the widget as HTML.

        Returns:
            HTML representation of the widget
        """
        # This is a simplified example; in a real implementation,
        # this would use a template engine or front-end framework

        agent_name = html.escape(
            self.agent_info.get("config", {}).get("name", "Unknown Agent")
        )
        agent_type = html.escape(self.agent_info.get("type", "unknown"))
        agent_status = html.escape(self.agent_info.get("status", "unknown"))
        element_id = html.escape(self.element_id)
        css_class = html.escape(self.css_class)
        agent_id_escaped = html.escape(self.agent_id or "")

        markup = f"""
        <div id="{element_id}" class="{css_class}">
            <div class="agent-header">
                <h3>{agent_name}</h3>
                <span class="agent-type">{agent_type}</span>
                <span class="agent-status status-{agent_status}">{agent_status}</span>
            </div>
            <div class="agent-controls">
                <button class="start-btn" onclick="startAgent('{agent_id_escaped}')">Start</button>
                <button class="stop-btn" onclick="stopAgent('{agent_id_escaped}')">Stop</button>
            </div>
            <div class="agent-metrics">
                <div class="metric">
                    <span class="metric-name">Decisions:</span>
                    <span class="metric-value">{self.agent_metrics.get("decision_count", 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Success Rate:</span>
                    <span class="metric-value">{self.agent_metrics.get("success_rate", 0.0):.2f}</span>
                </div>
            </div>
            <div class="agent-command">
                <select id="command-type">
                    <option value="query">Query</option>
                    <option value="update">Update</option>
                    <option value="execute">Execute</option>
                </select>
                <input type="text" id="command-params" placeholder="Parameters (JSON)">
                <button onclick="sendCommand('{agent_id_escaped}')">Send</button>
            </div>
            <div class="agent-history">
                <h4>Command History</h4>
                <ul>
        """

        # Add command history — escape all user-sourced strings
        for cmd in reversed(self.command_history[-5:]):
            ts = html.escape(cmd["timestamp"].split("T")[1].split(".")[0])
            cmd_type = html.escape(cmd["command_type"])
            result_status = html.escape(cmd["result"].get("status", ""))
            markup += f"""
                    <li>
                        <span class="timestamp">{ts}</span>
                        <span class="command">{cmd_type}</span>
                        <span class="result">{result_status}</span>
                    </li>
            """

        markup += """
                </ul>
            </div>
        </div>
        """

        return markup

    def get_javascript(self) -> str:
        """
        Get JavaScript code for widget functionality.

        Returns:
            JavaScript code
        """
        return """
        async function startAgent(agentId) {
            const response = await fetch(`/api/agents/${agentId}/start`, {
                method: 'POST'
            });
            if (response.ok) {
                // Refresh widget
                refreshAgentWidget();
            }
        }
        
        async function stopAgent(agentId) {
            const response = await fetch(`/api/agents/${agentId}/stop`, {
                method: 'POST'
            });
            if (response.ok) {
                // Refresh widget
                refreshAgentWidget();
            }
        }
        
        async function sendCommand(agentId) {
            const commandType = document.getElementById('command-type').value;
            const paramsText = document.getElementById('command-params').value;
            
            let parameters = {};
            if (paramsText) {
                try {
                    parameters = JSON.parse(paramsText);
                } catch (e) {
                    alert('Invalid JSON parameters');
                    return;
                }
            }
            
            const response = await fetch(`/api/agents/${agentId}/command`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    command_type: commandType,
                    parameters: parameters
                })
            });
            
            if (response.ok) {
                // Refresh widget
                refreshAgentWidget();
            }
        }
        
        function refreshAgentWidget() {
            // In a real implementation, this would use AJAX to refresh the widget
            location.reload();
        }
        """
