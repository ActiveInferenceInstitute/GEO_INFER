#!/usr/bin/env python3

"""
GEO-INFER-APP Agent Integration Example

This script demonstrates how to integrate GEO-INFER-AGENT with
the GEO-INFER-APP for building intelligent geospatial applications.
"""

import os
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import agent API
from geo_infer_app.api.agent_api import AgentManager
from geo_infer_app.components.agent_widget import WebAgentWidget

# Set paths for agent configuration and state
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../config/agents")
os.makedirs(CONFIG_DIR, exist_ok=True)


async def geo_agent_example():
    """
    Example demonstrating a geospatial agent that monitors locations
    and provides spatial recommendations.
    """
    print("\n--- Geospatial Agent Example ---")

    # Create configuration for agent manager
    manager_config = {
        "api_config": {
            "agents_config_path": os.path.join(CONFIG_DIR, "agent_configs.json")
        },
        "auto_start_agents": True,
    }

    # Create agent manager
    manager = AgentManager(config=manager_config)
    await manager.initialize()

    # BDI Agent for spatial monitoring
    bdi_config = {
        "name": "Spatial Monitor",
        "description": "Monitors locations and provides recommendations",
        "initial_beliefs": [
            {
                "name": "current_location",
                "value": {"lat": 40.7128, "lng": -74.0060},
                "confidence": 1.0,
            },
            {
                "name": "poi_categories",
                "value": ["restaurant", "park", "museum"],
                "confidence": 1.0,
            },
        ],
        "initial_desires": [
            {
                "name": "monitor_location",
                "description": "Monitor the current location",
                "priority": 0.9,
            },
            {
                "name": "recommend_pois",
                "description": "Recommend points of interest",
                "priority": 0.7,
            },
        ],
        "plans": [
            {
                "name": "location_monitoring_plan",
                "desire_name": "monitor_location",
                "actions": [
                    {
                        "action_type": "log",
                        "action_id": "log_location",
                        "parameters": {
                            "level": "info",
                            "message": "Monitoring location: {{current_location}}",
                        },
                    },
                    {
                        "action_type": "wait",
                        "action_id": "wait_for_update",
                        "parameters": {"duration": 5.0},
                    },
                ],
            },
            {
                "name": "poi_recommendation_plan",
                "desire_name": "recommend_pois",
                "actions": [
                    {
                        "action_type": "log",
                        "action_id": "log_recommendations",
                        "parameters": {
                            "level": "info",
                            "message": "Generating recommendations for {{current_location}}",
                        },
                    },
                    {
                        "action_type": "wait",
                        "action_id": "wait_for_data",
                        "parameters": {"duration": 2.0},
                    },
                ],
            },
        ],
    }

    # Create BDI agent
    bdi_agent_id = await manager.create_agent(
        agent_type="bdi", name="Spatial Monitor", config=bdi_config
    )

    print(f"Created BDI agent: {bdi_agent_id}")

    # RL Agent for optimizing search patterns
    rl_config = {
        "name": "Spatial Optimizer",
        "description": "Optimizes search patterns in spatial regions",
        "state_size": 10,
        "action_size": 4,
        "learning_rate": 0.1,
        "epsilon": 0.3,
    }

    # Create RL agent
    rl_agent_id = await manager.create_agent(
        agent_type="rl", name="Spatial Optimizer", config=rl_config
    )

    print(f"Created RL agent: {rl_agent_id}")

    # Rule-based Agent for alerts
    rule_based_config = {
        "name": "Spatial Alert System",
        "description": "Provides alerts for spatial events",
        "rules": [
            {
                "id": "proximity_alert",
                "condition": {"proximity": {"$lt": 100}},
                "action": {
                    "action_type": "update_fact",
                    "action_id": "set_proximity_alert",
                    "parameters": {"key": "proximity_alert", "value": True},
                },
                "priority": 10,
                "description": "Alert when proximity is less than 100 meters",
            }
        ],
        "initial_facts": {"proximity": 500, "proximity_alert": False},
    }

    # Create rule-based agent
    rule_agent_id = await manager.create_agent(
        agent_type="rule_based", name="Spatial Alert System", config=rule_based_config
    )

    print(f"Created rule-based agent: {rule_agent_id}")

    # Hybrid agent combining all agent types
    hybrid_config = {
        "name": "Geo Intelligence System",
        "description": "Combines multiple agent types for comprehensive geospatial intelligence",
        "decision_policy": "priority",
        "sub_agents": [
            {
                "type": "bdi",
                "priority": 10,
                "description": "Strategic planning agent",
                "config": {
                    "initial_beliefs": [
                        {
                            "name": "operational_area",
                            "value": {"lat": 40.7128, "lng": -74.0060, "radius": 10000},
                        }
                    ],
                    "initial_desires": [
                        {
                            "name": "strategic_monitoring",
                            "description": "Monitor area strategically",
                            "priority": 0.9,
                        }
                    ],
                },
            },
            {
                "type": "rl",
                "priority": 5,
                "description": "Tactical optimization agent",
                "config": {"state_size": 8, "action_size": 4},
            },
            {
                "type": "rule_based",
                "priority": 8,
                "description": "Reactive alert agent",
                "activation_conditions": {"alert_level": {"$gt": 0}},
                "config": {
                    "rules": [
                        {
                            "id": "emergency_response",
                            "condition": {"alert_level": {"$gt": 1}},
                            "action": {
                                "action_type": "log",
                                "action_id": "log_emergency",
                                "parameters": {
                                    "level": "warning",
                                    "message": "Emergency response required!",
                                },
                            },
                            "priority": 100,
                        }
                    ]
                },
            },
        ],
        "initial_context": {"alert_level": 0, "operational_status": "normal"},
    }

    # Create hybrid agent
    hybrid_agent_id = await manager.create_agent(
        agent_type="hybrid", name="Geo Intelligence System", config=hybrid_config
    )

    print(f"Created hybrid agent: {hybrid_agent_id}")

    # Start agents
    await manager.start_agent(bdi_agent_id)
    await manager.start_agent(rl_agent_id)
    await manager.start_agent(rule_agent_id)
    await manager.start_agent(hybrid_agent_id)

    print("\nRunning agents for demonstration...")

    # Run for a short time to demonstrate
    await asyncio.sleep(2)

    # Update the BDI agent's config (supported command: "update")
    await manager.send_command(
        bdi_agent_id,
        command_type="update",
        parameters={
            "config": {
                "current_location": {"lat": 40.7308, "lng": -73.9973},
            }
        },
    )

    # Update the rule-based agent's config (supported command: "update")
    await manager.send_command(
        rule_agent_id,
        command_type="update",
        parameters={"config": {"proximity_threshold": 50}},
    )

    # Query each agent's current state (supported command: "query")
    for agent_id in (bdi_agent_id, rl_agent_id, rule_agent_id, hybrid_agent_id):
        result = await manager.send_command(agent_id, command_type="query")
        info = result["result"]
        print(
            f"\nAgent {info['type']} state: status={info['status']}, config keys: {info['config_keys']}"
        )

    # Wait a bit longer
    await asyncio.sleep(2)

    # List all agents
    agents = await manager.list_agents()

    print("\nAll registered agents:")
    for agent in agents:
        status = await manager.get_agent_info(agent["id"])
        print(f" - {status['config']['name']} ({agent['id']}): {status['status']}")

    # Stop all agents
    for agent in agents:
        await manager.stop_agent(agent["id"])

    # Create a web widget for one of the agents
    widget = WebAgentWidget(
        agent_manager=manager,
        agent_id=bdi_agent_id,
        config={
            "element_id": "spatial-monitor-widget",
            "css_class": "geo-agent-widget",
        },
    )

    await widget.initialize()

    # Render the widget (this would be displayed in a web interface)
    html = widget.render()
    print("\nWeb widget HTML preview:")
    print(html[:500] + "...")  # Show just a preview

    # Shutdown
    await widget.shutdown()
    await manager.shutdown()


async def map_exploration_example():
    """
    Example demonstrating an agent for map exploration and feature detection.
    """
    print("\n--- Map Exploration Agent Example ---")

    # Create agent manager
    manager = AgentManager()
    await manager.initialize()

    # Create a hybrid agent for map exploration
    hybrid_config = {
        "name": "Map Explorer",
        "description": "Explores maps and detects features",
        "decision_policy": "priority",
        "sub_agents": [
            {
                "type": "bdi",
                "priority": 10,
                "description": "Strategic exploration agent",
                "config": {
                    "initial_beliefs": [
                        {"name": "explored_percentage", "value": 0.0},
                        {
                            "name": "map_bounds",
                            "value": {
                                "min_lat": 40.70,
                                "max_lat": 40.80,
                                "min_lng": -74.05,
                                "max_lng": -73.95,
                            },
                        },
                    ],
                    "initial_desires": [
                        {
                            "name": "explore_map",
                            "description": "Explore the map completely",
                            "priority": 0.9,
                        },
                        {
                            "name": "detect_features",
                            "description": "Detect interesting features",
                            "priority": 0.8,
                        },
                    ],
                },
            },
            {
                "type": "rl",
                "priority": 8,
                "description": "Exploration optimization agent",
                "config": {
                    "state_size": 16,
                    "action_size": 8,
                    "learning_rate": 0.05,
                    "epsilon": 0.2,
                },
            },
        ],
        "initial_context": {
            "exploration_mode": "systematic",
            "feature_detection_threshold": 0.7,
        },
    }

    # Create hybrid agent
    agent_id = await manager.create_agent(
        agent_type="hybrid", name="Map Explorer", config=hybrid_config
    )

    print(f"Created map exploration agent: {agent_id}")

    # Start agent
    await manager.start_agent(agent_id)

    # Simulate map exploration
    for i in range(5):
        # Update exploration progress
        explored = (i + 1) * 20.0  # 20%, 40%, etc.

        print(f"\nExploration cycle {i + 1}")
        print(f"Map explored: {explored}%")

        # Send a config update to the agent (supported command: "update")
        await manager.send_command(
            agent_id,
            command_type="update",
            parameters={"config": {"explored_percentage": explored}},
        )

        # Simulate a feature detection if we're at 60%
        if explored == 60.0:
            await manager.send_command(
                agent_id,
                command_type="update",
                parameters={
                    "config": {
                        "detected_feature": {
                            "type": "water_body",
                            "location": {"lat": 40.75, "lng": -74.0},
                            "confidence": 0.85,
                        }
                    }
                },
            )

            print("Detected a water body feature!")

    # Query the final state (supported command: "query")
    result = await manager.send_command(agent_id, command_type="query")

    if result and result.get("status") == "success":
        info = result["result"]
        print(
            f"\nAgent performance: status={info['status']}, config keys: {info['config_keys']}"
        )

    # Stop agent
    await manager.stop_agent(agent_id)

    # Shutdown manager
    await manager.shutdown()


async def main():
    """Run integration examples."""
    print("GEO-INFER-APP Agent Integration Examples")

    await geo_agent_example()
    await map_exploration_example()


if __name__ == "__main__":
    asyncio.run(main())
