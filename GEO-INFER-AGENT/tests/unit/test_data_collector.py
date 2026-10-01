#!/usr/bin/env python3

"""
Tests for the DataCollectorAgent: data collection, source monitoring, processing.

Note: DataCollectorAgent depends on BDIAgent, pandas, and requests.
These tests focus on the initialization and config handling that can be tested
without external dependencies making network calls.
"""

import asyncio
import unittest

from geo_infer_agent.models import BDIAgent


class TestDataCollectorConfig(unittest.TestCase):
    """Tests for data collector configuration and belief initialization.

    Since DataCollectorAgent has external deps (requests, pandas),
    we test the foundational BDI patterns it relies on directly.
    """

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_bdi_agent_with_data_collector_config(self) -> None:
        """A BDI agent can be configured with data-collector-like plans and desires."""
        config = {
            "initial_desires": [
                {
                    "name": "collect_data",
                    "description": "Collect data from sources",
                    "priority": 0.8,
                },
                {
                    "name": "process_data",
                    "description": "Process collected data",
                    "priority": 0.7,
                },
            ],
            "plans": [
                {
                    "name": "collection_plan",
                    "desire_name": "collect_data",
                    "actions": [
                        {"type": "log", "message": "Collecting data", "level": "info"},
                    ],
                },
                {
                    "name": "processing_plan",
                    "desire_name": "process_data",
                    "actions": [
                        {"type": "log", "message": "Processing data", "level": "info"},
                    ],
                    "context_conditions": {"has_unprocessed_data": True},
                },
            ],
            "initial_beliefs": {
                "has_unprocessed_data": False,
                "total_collected_datasets": 0,
            },
        }
        agent = BDIAgent(agent_id="collector-sim", config=config)
        self._run(agent.initialize())

        # Check desires loaded
        desires = agent.state.get_desires_by_priority()
        names = [d.name for d in desires]
        self.assertIn("collect_data", names)
        self.assertIn("process_data", names)

        # Check beliefs loaded
        belief = agent.state.get_belief("has_unprocessed_data")
        self.assertIsNotNone(belief)
        self.assertFalse(belief.value)

    def test_data_source_beliefs_initialization(self) -> None:
        """Data source beliefs can be initialized through the BDI belief system."""
        agent = BDIAgent(agent_id="src-init", config={})
        self._run(agent.initialize())

        # Simulate what DataCollectorAgent._initialize_data_source_beliefs does
        sources = [
            {
                "id": "weather_api",
                "name": "Weather",
                "type": "api",
                "url": "http://example.com",
            },
            {
                "id": "sensor_net",
                "name": "Sensors",
                "type": "sensor",
                "sensor_id": "s1",
            },
        ]
        for source in sources:
            sid = source["id"]
            agent.state.update_belief(f"data_source.{sid}.name", source["name"])
            agent.state.update_belief(f"data_source.{sid}.type", source["type"])
            agent.state.update_belief(f"data_source.{sid}.available", False)
            agent.state.update_belief(f"data_source.{sid}.last_check", None)

        # Verify beliefs were set
        weather_name = agent.state.get_belief("data_source.weather_api.name")
        self.assertIsNotNone(weather_name)
        self.assertEqual(weather_name.value, "Weather")

        sensor_avail = agent.state.get_belief("data_source.sensor_net.available")
        self.assertIsNotNone(sensor_avail)
        self.assertFalse(sensor_avail.value)

    def test_collection_plan_execution_through_bdi(self) -> None:
        """The collection plan can be executed through the BDI decide/act cycle."""
        config = {
            "initial_desires": [
                {"name": "collect", "description": "Collect", "priority": 0.9},
            ],
            "plans": [
                {
                    "name": "collect_plan",
                    "desire_name": "collect",
                    "actions": [
                        {
                            "type": "log",
                            "message": "Starting collection",
                            "level": "info",
                        },
                        {
                            "type": "update_belief",
                            "belief_name": "last_collection",
                            "belief_value": "done",
                        },
                    ],
                }
            ],
        }
        agent = BDIAgent(agent_id="exec-test", config=config)
        self._run(agent.initialize())

        # First decide should return the first action from the plan
        action = self._run(agent.decide())
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "log")

        # Execute it
        result = self._run(agent.act(action))
        self.assertTrue(result.get("success", False))

        # Second decide should return the update_belief action
        action2 = self._run(agent.decide())
        self.assertIsNotNone(action2)
        self.assertEqual(action2["type"], "update_belief")

    def test_conditional_plan_not_triggered_without_data(self) -> None:
        """A processing plan with has_unprocessed_data condition is skipped if False."""
        config = {
            "initial_desires": [
                {"name": "process", "description": "Process", "priority": 0.7},
            ],
            "initial_beliefs": {"has_unprocessed_data": False},
            "plans": [
                {
                    "name": "process_plan",
                    "desire_name": "process",
                    "actions": [
                        {"type": "log", "message": "processing", "level": "info"}
                    ],
                    "context_conditions": {"has_unprocessed_data": True},
                },
            ],
        }
        agent = BDIAgent(agent_id="cond-test", config=config)
        self._run(agent.initialize())

        # Decide should return None since condition is not met
        action = self._run(agent.decide())
        self.assertIsNone(action)

    def test_conditional_plan_triggers_when_data_available(self) -> None:
        """Processing plan triggers when has_unprocessed_data belief is True."""
        config = {
            "initial_desires": [
                {"name": "process", "description": "Process", "priority": 0.7},
            ],
            "initial_beliefs": {"has_unprocessed_data": True},
            "plans": [
                {
                    "name": "process_plan",
                    "desire_name": "process",
                    "actions": [
                        {"type": "log", "message": "processing", "level": "info"}
                    ],
                    "context_conditions": {"has_unprocessed_data": True},
                },
            ],
        }
        agent = BDIAgent(agent_id="cond-true", config=config)
        self._run(agent.initialize())

        action = self._run(agent.decide())
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "log")


if __name__ == "__main__":
    unittest.main()
