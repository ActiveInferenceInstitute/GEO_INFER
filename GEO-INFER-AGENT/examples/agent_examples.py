#!/usr/bin/env python3

"""
GEO-INFER-AGENT Examples

This script demonstrates how to use the different agent architectures
provided by the GEO-INFER-AGENT module.
"""

import asyncio
import logging
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import agent models
from geo_infer_agent.models import (
    BDIAgent,
    ActiveInferenceAgent,
    RLAgent,
    RuleBasedAgent,
    HybridAgent,
)


# Recorded sensor feed for the examples
class RecordedSensor:
    """Deterministic sensor feed backed by caller-provided readings."""

    def __init__(self, name, readings=None):
        self.name = name
        self.data = readings or {}

    async def read(self):
        """Read the configured sensor record."""
        if not self.data:
            raise RuntimeError(f"No readings configured for sensor {self.name}")
        return {
            "timestamp": datetime.now().isoformat(),
            "readings": dict(self.data),
        }


async def run_bdi_agent_example():
    """Example using a BDI agent for environmental monitoring."""
    print("\n--- BDI Agent Example ---")

    # Create configuration
    config = {
        "name": "Environmental Monitor",
        "initial_beliefs": [
            {
                "name": "safe_temperature_range",
                "value": {"min": 18.0, "max": 26.0},
                "confidence": 1.0,
            },
            {
                "name": "safe_humidity_range",
                "value": {"min": 30.0, "max": 60.0},
                "confidence": 1.0,
            },
        ],
        "initial_desires": [
            {
                "name": "monitor_environment",
                "description": "Continuously monitor environmental conditions",
                "priority": 0.9,
            },
            {
                "name": "report_anomalies",
                "description": "Report any anomalies detected",
                "priority": 0.8,
            },
        ],
        "plans": [
            {
                "name": "environment_monitoring_plan",
                "desire_name": "monitor_environment",
                "actions": [
                    {
                        "action_type": "wait",
                        "action_id": "wait_for_sensor",
                        "parameters": {"duration": 2.0},
                    },
                    {
                        "action_type": "log",
                        "action_id": "log_status",
                        "parameters": {
                            "level": "info",
                            "message": "Continuing environmental monitoring...",
                        },
                    },
                ],
            }
        ],
    }

    # Create a BDI agent
    agent = BDIAgent(agent_id="env_monitor_bdi", config=config)

    # Register a recorded sensor feed
    sensor = RecordedSensor(
        "env_sensor", {"temperature": 22.5, "humidity": 45.0, "pressure": 1013.0}
    )
    agent.register_sensor("env_sensor", sensor.read)

    # Initialize agent
    await agent.initialize()

    # Run agent cycle a few times
    for i in range(3):
        print(f"\nAgent cycle {i + 1}:")

        # Perceive environment
        perceptions = await agent.perceive()
        print(f"  Perceptions: {json.dumps(perceptions, indent=2)}")

        # Make decision
        decision = await agent.decide()
        print(f"  Decision: {json.dumps(decision, indent=2)}")

        # Execute action
        if decision:
            result = await agent.act(decision)
            print(f"  Action result: {json.dumps(result, indent=2)}")

    # Shutdown agent
    await agent.shutdown()


async def run_active_inference_agent_example():
    """Example using an Active Inference agent for environmental prediction."""
    print("\n--- Active Inference Agent Example ---")

    # Create configuration
    config = {
        "name": "Environmental Predictor",
        "state_dimensions": 4,
        "observation_dimensions": 3,
        "control_dimensions": 2,
        "planning_horizon": 2,
        "default_preferences": [0.0, 0.0, 0.0],  # Neutral preferences initially
    }

    # Create an Active Inference agent
    agent = ActiveInferenceAgent(agent_id="env_predictor_ai", config=config)

    # Register a recorded sensor feed
    sensor = RecordedSensor(
        "env_sensor", {"temperature": 22.5, "humidity": 45.0, "pressure": 1013.0}
    )
    agent.register_sensor("env_sensor", sensor.read)

    # Initialize agent
    await agent.initialize()

    # Run agent cycle a few times to learn
    for i in range(3):
        print(f"\nAgent cycle {i + 1}:")

        # Perceive environment
        perceptions = await agent.perceive()
        print("  Perceptions: sensor data received")

        # Update preferences based on cycle
        if i == 1:
            # Set preference for comfortable temperature
            pref_action = {
                "action_type": "update_preferences",
                "action_id": "set_comfort_preference",
                "parameters": {"preferences": [1.0, 0.5, 0.0]},
            }
            result = await agent.act(pref_action)
            print(f"  Updated preferences: {json.dumps(result, indent=2)}")

        # Make decision
        decision = await agent.decide()
        print(f"  Decision: {json.dumps(decision, indent=2)}")

        # Execute action
        if decision:
            result = await agent.act(decision)
            print(f"  Action result: {json.dumps(result, indent=2)}")

    # Query model state
    query_action = {
        "action_type": "query_model",
        "action_id": "check_model_state",
        "parameters": {"query_type": "state"},
    }
    result = await agent.act(query_action)
    print(f"\nModel state: state vector with {len(result['state_belief'])} dimensions")

    # Shutdown agent
    await agent.shutdown()


async def run_rl_agent_example():
    """Example using a Reinforcement Learning agent for optimal sensing."""
    print("\n--- RL Agent Example ---")

    # Create configuration
    config = {
        "name": "Optimal Sensor",
        "state_size": 8,
        "action_size": 4,
        "learning_rate": 0.1,
        "epsilon": 0.3,
        "epsilon_decay": 0.98,
        "discount_factor": 0.95,
        "train_frequency": 1,
    }

    # Create an RL agent
    agent = RLAgent(agent_id="optimal_sensor_rl", config=config)

    # Register a recorded sensor feed
    sensor = RecordedSensor(
        "env_sensor", {"temperature": 22.5, "humidity": 45.0, "pressure": 1013.0}
    )
    agent.register_sensor("env_sensor", sensor.read)

    # Initialize agent
    await agent.initialize()

    # Run agent cycle a few times to learn
    for i in range(4):
        print(f"\nAgent cycle {i + 1}:")

        # Perceive environment
        perceptions = await agent.perceive()
        print("  Perceptions: sensor data received")

        # Make decision
        decision = await agent.decide()
        print(f"  Decision: {json.dumps(decision, indent=2)}")

        # Execute action with reward feedback
        if decision:
            # Simulate reward based on action
            action_idx = decision.get("parameters", {}).get("index", 0)
            reward = 0.1 * (action_idx + 1)  # Simple reward function

            result = await agent.act(decision)
            result["reward"] = reward
            result["episode_done"] = i == 3  # Last cycle ends episode

            print(f"  Action result: {json.dumps(result, indent=2)}")

    # Query agent state
    query_action = {
        "action_type": "query_state",
        "action_id": "check_learning_progress",
        "parameters": {"query_type": "performance"},
    }
    result = await agent.act(query_action)
    print(f"\nAgent performance: {json.dumps(result, indent=2)}")

    # Shutdown agent
    await agent.shutdown()


async def run_rule_based_agent_example():
    """Example using a Rule-Based agent for environmental alerts."""
    print("\n--- Rule-Based Agent Example ---")

    # Create configuration
    config = {
        "name": "Environmental Alert System",
        "rules": [
            {
                "id": "high_temperature_alert",
                "condition": {"sensor_temperature": {"$gt": 24.0}},
                "action": {
                    "action_type": "update_fact",
                    "action_id": "set_alert",
                    "parameters": {"key": "temperature_alert", "value": True},
                },
                "priority": 10,
                "description": "Alert when temperature exceeds threshold",
            },
            {
                "id": "normal_temperature",
                "condition": {"sensor_temperature": {"$lte": 24.0}},
                "action": {
                    "action_type": "update_fact",
                    "action_id": "clear_alert",
                    "parameters": {"key": "temperature_alert", "value": False},
                },
                "priority": 5,
                "description": "Clear temperature alert when normal",
            },
        ],
        "initial_facts": {"temperature_alert": False, "system_status": "normal"},
    }

    # Create a Rule-Based agent
    agent = RuleBasedAgent(agent_id="env_alert_rb", config=config)

    # Register a recorded sensor feed
    sensor = RecordedSensor(
        "env_sensor", {"temperature": 22.5, "humidity": 45.0, "pressure": 1013.0}
    )
    agent.register_sensor("env_sensor", sensor.read)

    # Initialize agent
    await agent.initialize()

    # Run agent cycle a few times
    for i in range(3):
        print(f"\nAgent cycle {i + 1}:")

        # Perceive environment
        perceptions = await agent.perceive()
        print("  Perceptions: sensor data received")

        # Make decision
        decision = await agent.decide()
        print(f"  Decision: {json.dumps(decision, indent=2)}")

        # Execute action
        if decision:
            result = await agent.act(decision)
            print(f"  Action result: {json.dumps(result, indent=2)}")

        # Query facts after action
        query_action = {
            "action_type": "query_facts",
            "action_id": "check_facts",
            "parameters": {},
        }
        result = await agent.act(query_action)
        print(
            f"  Current facts: temperature_alert = {result['facts'].get('temperature_alert', False)}"
        )

    # Shutdown agent
    await agent.shutdown()


async def run_hybrid_agent_example():
    """Example using a Hybrid agent that combines multiple architectures."""
    print("\n--- Hybrid Agent Example ---")

    # Create configuration
    config = {
        "name": "Environmental Management System",
        "decision_policy": "priority",
        "initial_context": {"system_status": "normal", "alert_level": 0},
        "sub_agents": [
            {
                "type": "rule_based",
                "id": "alert_agent",
                "priority": 10,
                "description": "Rule-based agent for alerts",
                "activation_conditions": {"system_status": "normal"},
                "config": {
                    "rules": [
                        {
                            "id": "high_temperature_alert",
                            "condition": {"sensor_temperature": {"$gt": 24.0}},
                            "action": {
                                "action_type": "update_context",
                                "action_id": "set_alert",
                                "parameters": {"key": "alert_level", "value": 1},
                            },
                            "priority": 10,
                        }
                    ]
                },
            },
            {
                "type": "bdi",
                "id": "monitor_agent",
                "priority": 5,
                "description": "BDI agent for monitoring",
                "config": {
                    "initial_desires": [
                        {
                            "name": "monitor_environment",
                            "description": "Continuously monitor environmental conditions",
                            "priority": 0.9,
                        }
                    ],
                    "plans": [
                        {
                            "name": "environment_monitoring_plan",
                            "desire_name": "monitor_environment",
                            "actions": [
                                {
                                    "action_type": "wait",
                                    "action_id": "wait_for_sensor",
                                    "parameters": {"duration": 1.0},
                                }
                            ],
                        }
                    ],
                },
            },
        ],
    }

    # Create a Hybrid agent
    agent = HybridAgent(agent_id="env_management_hybrid", config=config)

    # Register a recorded sensor feed
    sensor = RecordedSensor(
        "env_sensor", {"temperature": 22.5, "humidity": 45.0, "pressure": 1013.0}
    )
    agent.register_sensor("env_sensor", sensor.read)

    # Initialize agent
    await agent.initialize()

    # Run agent cycle a few times
    for i in range(3):
        print(f"\nAgent cycle {i + 1}:")

        # Perceive environment
        perceptions = await agent.perceive()
        print("  Perceptions: sensor data received")

        # Query active agents
        query_action = {
            "action_type": "query_agents",
            "action_id": "check_active",
            "parameters": {"query_type": "active"},
        }
        result = await agent.act(query_action)
        print(f"  Active agents: {[a['id'] for a in result['active_agents']]}")

        # Make decision
        decision = await agent.decide()
        print(f"  Decision: {json.dumps(decision, indent=2)}")

        # Execute action
        if decision:
            result = await agent.act(decision)
            print(f"  Action result: {json.dumps(result, indent=2)}")

    # Shutdown agent
    await agent.shutdown()


async def main():
    """Run all agent examples."""
    print("GEO-INFER-AGENT Examples")

    # Make sure examples directory exists
    os.makedirs("output", exist_ok=True)

    # Run each agent example
    await run_bdi_agent_example()
    await run_active_inference_agent_example()
    await run_rl_agent_example()
    await run_rule_based_agent_example()
    await run_hybrid_agent_example()


if __name__ == "__main__":
    asyncio.run(main())
