#!/usr/bin/env python3

"""
Tests for BDIAgent ``$CONFIG:<key>`` placeholder resolution in plan templates.
"""

import unittest

import pytest

from geo_infer_agent import BDIAgent


class TestConfigPlaceholders:
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

    @pytest.mark.asyncio(loop_scope="function")
    async def test_decide_resolves_config_placeholder_to_value(self) -> None:
        agent = self._make_agent()
        await agent.initialize()
        action = await agent.decide()

        # The placeholder must be replaced by the numeric config value —
        # a literal "$CONFIG:collection_interval" string would break
        # asyncio.sleep in the wait handler.
        unittest.TestCase().assertEqual(action, {"type": "wait", "duration": 300})

        result = await agent.act({"type": "wait", "duration": 0.01})
        unittest.TestCase().assertTrue(result["success"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_unknown_config_key_raises(self) -> None:
        agent = self._make_agent()
        await agent.initialize()
        agent.state.update_belief("sensor.missing", True)
        agent.plan_library["collection_plan"]["actions"] = [
            {"type": "wait", "duration": "$CONFIG:no_such_key"}
        ]

        with unittest.TestCase().assertRaises(ValueError):
            await agent.decide()
