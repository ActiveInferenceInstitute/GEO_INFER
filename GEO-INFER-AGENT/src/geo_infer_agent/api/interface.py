#!/usr/bin/env python3

"""
Unified interface for external systems to interact with agents.

This module provides a high-level API that abstracts the underlying
agent implementation details and provides a clean interface for
external systems to interact with the GEO-INFER-AGENT system.
"""

import logging
import asyncio
from typing import Any

from geo_infer_agent.api.agent_endpoints import agent_registry
from geo_infer_agent.api.messaging import messaging_service, Message
from geo_infer_agent.api.telemetry import telemetry_service

logger = logging.getLogger("geo_infer_agent.api.interface")
_pending_start_tasks: set = set()


def _on_start_task_done(task: asyncio.Task) -> None:
    """Log any exception from a fire-and-forget start task, then discard it."""
    _pending_start_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error(f"Background agent start task failed: {task.exception()}")


class AgentInterface:
    """
    High-level interface for working with agents.

    This class provides a simplified API for:
    - Creating and managing agents
    - Controlling agent execution
    - Retrieving agent state and results
    - Agent communication
    """

    def __init__(self) -> None:
        """Initialize the agent interface."""
        logger.info("Agent interface initialized")

    async def initialize_services(self, reporting_interval: int = 60) -> None:
        """
        Initialize all required services.

        Args:
            reporting_interval: Telemetry reporting interval in seconds
        """
        # Start messaging service
        await messaging_service.start()

        # Start telemetry service
        await telemetry_service.start(reporting_interval)

        logger.info("Agent interface services initialized")

    async def shutdown_services(self) -> None:
        """Shutdown all services."""
        # Stop messaging service
        await messaging_service.stop()

        # Stop telemetry service
        await telemetry_service.stop()

        logger.info("Agent interface services shutdown")

    async def create_agent(
        self,
        agent_type: str,
        config: dict[str, Any],
        agent_id: str | None = None,
        region: str | None = None,
    ) -> str:
        """
        Create a new agent.

        Args:
            agent_type: Type of agent to create
            config: Agent configuration
            agent_id: Custom ID for the agent (auto-generated if None)
            region: Geospatial region for agent operation (GeoJSON)

        Returns:
            ID of the created agent
        """
        agent_id = await agent_registry.create_agent(
            agent_type=agent_type, agent_id=agent_id, config=config, region=region
        )

        logger.info(f"Created agent {agent_id} of type {agent_type}")
        return agent_id

    async def start_agent(self, agent_id: str) -> bool:
        """
        Start an agent.

        Args:
            agent_id: ID of the agent to start

        Returns:
            True if the agent was started successfully
        """
        try:
            if agent_registry.is_agent_running(agent_id):
                logger.warning(f"Agent {agent_id} is already running")
                return False

            # Start the agent
            task = asyncio.create_task(agent_registry.start_agent(agent_id))
            _pending_start_tasks.add(task)
            task.add_done_callback(_on_start_task_done)
            logger.info(f"Started agent {agent_id}")
            return True
        except KeyError:
            logger.error(f"Agent {agent_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to start agent {agent_id}: {str(e)}")
            return False

    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop an agent.

        Args:
            agent_id: ID of the agent to stop

        Returns:
            True if the agent was stopped successfully
        """
        try:
            await agent_registry.stop_agent(agent_id)
            logger.info(f"Stopped agent {agent_id}")
            return True
        except KeyError:
            logger.error(f"Agent {agent_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to stop agent {agent_id}: {str(e)}")
            return False

    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent.

        Args:
            agent_id: ID of the agent to delete

        Returns:
            True if the agent was deleted successfully
        """
        try:
            # Stop the agent if it's running
            if agent_registry.is_agent_running(agent_id):
                await agent_registry.stop_agent(agent_id)

            # Remove the agent
            agent_registry.remove_agent(agent_id)

            # Unregister from messaging service
            messaging_service.unregister_agent(agent_id)

            logger.info(f"Deleted agent {agent_id}")
            return True
        except KeyError:
            logger.error(f"Agent {agent_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to delete agent {agent_id}: {str(e)}")
            return False

    def list_agents(self) -> list[dict[str, Any]]:
        """
        List all registered agents.

        Returns:
            List of agent information dictionaries
        """
        return agent_registry.list_agents()

    def get_agent_info(self, agent_id: str) -> dict[str, Any]:
        """
        Get information about an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Dictionary with agent information

        Raises:
            KeyError: If the agent does not exist
        """
        return agent_registry.get_agent_info(agent_id)

    async def get_agent_state(self, agent_id: str) -> dict[str, Any]:
        """
        Get the current state of an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Dictionary with agent state

        Raises:
            KeyError: If the agent does not exist
        """
        return await agent_registry.get_agent_state(agent_id)

    async def perform_action(
        self, agent_id: str, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Perform an action on an agent.

        Args:
            agent_id: ID of the agent
            action: Action to perform
            parameters: Action parameters

        Returns:
            Result of the action

        Raises:
            KeyError: If the agent does not exist
            ValueError: If the action is invalid
        """
        return await agent_registry.agent_action(
            agent_id=agent_id, action=action, parameters=parameters
        )

    async def send_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        content: dict[str, Any],
        priority: int = 1,
    ) -> bool:
        """
        Send a message from one agent to another.

        Args:
            from_agent_id: ID of the sending agent
            to_agent_id: ID of the receiving agent
            content: Message content
            priority: Message priority (1-10)

        Returns:
            True if the message was sent successfully
        """
        # Create message
        message = Message(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            content=content,
            priority=priority,
        )

        # Send the message
        success = await messaging_service.send_message(message)

        if success:
            logger.debug(f"Message sent from {from_agent_id} to {to_agent_id}")
        else:
            logger.warning(
                f"Failed to send message from {from_agent_id} to {to_agent_id}"
            )

        return success

    async def broadcast_message(
        self,
        from_agent_id: str,
        content: dict[str, Any],
        channel: str,
        priority: int = 1,
    ) -> int:
        """
        Broadcast a message to a channel.

        Args:
            from_agent_id: ID of the sending agent
            content: Message content
            channel: Channel to broadcast to
            priority: Message priority (1-10)

        Returns:
            Number of agents the message was sent to
        """
        sent_count = await messaging_service.broadcast_message(
            from_agent_id=from_agent_id,
            content=content,
            channel=channel,
            priority=priority,
        )

        logger.debug(
            f"Broadcast message from {from_agent_id} to channel {channel} sent to {sent_count} agents"
        )
        return sent_count

    def subscribe_to_channel(self, agent_id: str, channel: str) -> None:
        """
        Subscribe an agent to a channel.

        Args:
            agent_id: ID of the agent
            channel: Channel to subscribe to
        """
        messaging_service.subscribe(agent_id, channel)
        logger.debug(f"Agent {agent_id} subscribed to channel {channel}")

    def get_agent_metrics(self, agent_id: str) -> dict[str, dict[str, Any]]:
        """
        Get metrics for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Dictionary of metrics for the agent
        """
        return telemetry_service.get_metrics(agent_id)

    def get_agent_health(self, agent_id: str) -> dict[str, Any]:
        """
        Get health status for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Health status for the agent
        """
        health = telemetry_service.get_health_status(agent_id)
        return health.get(agent_id, {"status": "unknown"})


# Global instance
agent_interface = AgentInterface()
