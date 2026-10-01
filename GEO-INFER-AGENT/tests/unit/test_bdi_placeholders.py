#!/usr/bin/env python3

"""
Tests for BDIAgent ``$CONFIG:<key>`` placeholder resolution in plan templates.
"""

import asyncio
import unittest

from geo_infer_agent import BDIAgent


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestConfigPlaceholders(unittest.TestCase):
    """Plan templates referencing config values resolve at instantiation."""

    def _make_agent(self) -> BDIAgent:
        config = {
            "collection_interval": 300,
            "initial_beliefs": {},
            "initial_desires": [
                {"name": "collect", "description": "Collect data", "priority": 0.9}
            ],
            "plans": [
                {
                    "name": "collection_plan",
                    "desire_name": "collect",
                    "actions": [
                        {"type": "wait", "duration": "$CONFIG:collection_interval"},
                    ],
                }
            ],
        }
        return BDIAgent(agent_id="ph-agent", config=config)

    def test_decide_resolves_config_placeholder_to_value(self) -> None:
        agent = self._make_agent()
        _run(agent.initialize())
        action = _run(agent.decide())

        # The placeholder must be replaced by the numeric config value —
        # a literal "$CONFIG:collection_interval" string would break
        # asyncio.sleep in the wait handler.
        self.assertEqual(action, {"type": "wait", "duration": 300})

        result = _run(agent.act({"type": "wait", "duration": 0.01}))
        self.assertTrue(result["success"])

    def test_unknown_config_key_raises(self) -> None:
        agent = self._make_agent()
        _run(agent.initialize())
        agent.state.update_belief("sensor.missing", True)
        agent.plan_library["collection_plan"]["actions"] = [
            {"type": "wait", "duration": "$CONFIG:no_such_key"}
        ]

        with self.assertRaises(ValueError):
            _run(agent.decide())


if __name__ == "__main__":
    unittest.main()
