"""
BDI Agent Example

This example demonstrates how to create and interact with a BDI agent
using the GEO-INFER-APP agent interface.
"""

import json
import logging
from geo_infer_app.models.agent_interface import AgentType
from geo_infer_app.models.agent_factory import AgentFactory
from geo_infer_app.models.agent_visualization import AgentVisualization

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def on_agent_updated(event_data):
    """Event handler for agent updates."""
    agent_id = event_data["agent_id"]
    state = event_data["state"]
    logger.info(f"Agent {agent_id} was updated!")
    logger.info(f"Status: {state.status}")
    logger.info(f"Location: {state.location}")
    logger.info(f"Beliefs: {state.beliefs}")
    logger.info(f"Goals: {state.goals}")
    logger.info(f"Intentions: {state.metadata.get('intentions', [])}")


def main():
    """Main function demonstrating agent creation and interaction."""
    logger.info("Starting BDI agent example")

    # Create an agent factory
    factory = AgentFactory()

    # Create a BDI agent interface
    agent_interface = factory.create_interface(AgentType.BDI)

    # Register an event handler for agent updates
    agent_interface.register_event_handler("agent_updated", on_agent_updated)
    agent_interface.register_event_handler("agent_created", on_agent_updated)

    # Create a BDI agent with initial configuration
    config = {
        "name": "Exploration Agent",
        "beliefs": {
            "location": {"lat": 40.7128, "lng": -74.0060},
            "weather": "sunny",
            "time_of_day": "morning",
        },
        "desires": ["explore_area", "collect_data", "return_home"],
        "initial_location": {"lat": 40.7128, "lng": -74.0060},
    }

    logger.info("Creating agent with configuration:")
    logger.info(json.dumps(config, indent=2))

    agent_id = agent_interface.create_agent(AgentType.BDI, config)
    logger.info(f"Created agent with ID: {agent_id}")

    # Run the agent's deliberation process
    logger.info("Running agent deliberation...")
    response = agent_interface.send_command(agent_id, "deliberate", {})
    logger.info(f"Deliberation response: {response}")

    # Get the agent's current state
    state = agent_interface.get_agent_state(agent_id)
    logger.info("Current agent state:")
    logger.info(f"Status: {state.status}")
    logger.info(f"Location: {state.location}")
    logger.info(f"Beliefs: {state.beliefs}")
    logger.info(f"Goals: {state.goals}")
    logger.info(f"Intentions: {state.metadata.get('intentions', [])}")

    # Add a new belief
    logger.info("Adding a new belief...")
    agent_interface.send_command(
        agent_id, "add_belief", {"belief": {"temperature": 25, "humidity": 60}}
    )

    # Execute one step
    logger.info("Executing agent intentions...")
    agent_interface.send_command(agent_id, "execute", {})

    # Move the agent
    logger.info("Moving agent to a new location...")
    new_location = {"lat": 40.7135, "lng": -74.0080}
    agent_interface.send_command(agent_id, "move", {"location": new_location})

    # Get the agent's updated state
    state = agent_interface.get_agent_state(agent_id)

    # Generate map visualization
    try:
        map_feature = AgentVisualization.state_to_map_feature(state)
        logger.info("Map feature for agent:")
        logger.info(json.dumps(map_feature, indent=2))
    except ValueError as e:
        logger.error(f"Error generating map feature: {e}")

    # Generate dashboard visualization
    dashboard_data = AgentVisualization.state_to_dashboard_data(state)
    logger.info("Dashboard data for agent:")
    logger.info(json.dumps(dashboard_data, indent=2))

    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
