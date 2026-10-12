#!/usr/bin/env python3

"""
Unit tests for BDI agent.
"""

import pytest
from datetime import datetime, timedelta

# BDIState and BDIAgent are in bdi.py file, import from models package
from geo_infer_agent.models import BDIState, BDIAgent
from geo_infer_agent.models.bdi import Belief, Desire, Plan


class TestBelief:
    """Tests for the Belief class."""

    def test_belief_initialization(self):
        """Test that a belief can be initialized with correct values."""
        belief = Belief(
            name="test_belief",
            value="test_value",
            confidence=0.8,
            metadata={"source": "test"},
        )

        assert belief.name == "test_belief"
        assert belief.value == "test_value"
        assert belief.confidence == 0.8
        assert belief.metadata == {"source": "test"}
        assert isinstance(belief.timestamp, datetime)
        assert len(belief.history) == 0

    def test_belief_update(self):
        """Test that a belief can be updated."""
        belief = Belief(name="test_belief", value="test_value")

        # Update belief
        belief.update(value="new_value", confidence=0.9, metadata={"updated": True})

        assert belief.value == "new_value"
        assert belief.confidence == 0.9
        assert belief.metadata == {"updated": True}
        assert len(belief.history) == 1

        # Check history
        history_entry = belief.history[0]
        assert history_entry["value"] == "test_value"

    def test_belief_to_dict(self):
        """Test conversion to dictionary."""
        belief = Belief(name="test_belief", value="test_value")

        belief_dict = belief.to_dict()

        assert belief_dict["name"] == "test_belief"
        assert belief_dict["value"] == "test_value"
        assert isinstance(belief_dict["timestamp"], str)

    def test_belief_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test_belief",
            "value": "test_value",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"source": "test"},
        }

        belief = Belief.from_dict(data)

        assert belief.name == "test_belief"
        assert belief.value == "test_value"
        assert belief.confidence == 0.8
        assert belief.metadata == {"source": "test"}


class TestDesire:
    """Tests for the Desire class."""

    def test_desire_initialization(self):
        """Test that a desire can be initialized with correct values."""
        desire = Desire(
            name="test_desire",
            description="A test desire",
            priority=0.7,
            deadline=datetime.now() + timedelta(hours=1),
            conditions={"test_condition": True},
        )

        assert desire.name == "test_desire"
        assert desire.description == "A test desire"
        assert desire.priority == 0.7
        assert isinstance(desire.deadline, datetime)
        assert desire.conditions == {"test_condition": True}
        assert desire.achieved is False
        assert desire.achieved_at is None

    def test_desire_set_achieved(self):
        """Test setting a desire as achieved."""
        desire = Desire(name="test_desire", description="A test desire")

        # Set as achieved
        desire.set_achieved(True)

        assert desire.achieved is True
        assert isinstance(desire.achieved_at, datetime)

        # Set as not achieved
        desire.set_achieved(False)

        assert desire.achieved is False
        assert desire.achieved_at is None

    def test_desire_is_expired(self):
        """Test checking if a desire has expired."""
        # Create a desire with a deadline in the past
        past_deadline = datetime.now() - timedelta(hours=1)
        expired_desire = Desire(
            name="expired_desire",
            description="An expired desire",
            deadline=past_deadline,
        )

        assert expired_desire.is_expired() is True

        # Create a desire with a deadline in the future
        future_deadline = datetime.now() + timedelta(hours=1)
        active_desire = Desire(
            name="active_desire",
            description="An active desire",
            deadline=future_deadline,
        )

        assert active_desire.is_expired() is False

        # Create a desire with no deadline
        indefinite_desire = Desire(
            name="indefinite_desire", description="A desire with no deadline"
        )

        assert indefinite_desire.is_expired() is False

    def test_desire_to_dict(self):
        """Test conversion to dictionary."""
        desire = Desire(name="test_desire", description="A test desire")

        desire_dict = desire.to_dict()

        assert desire_dict["name"] == "test_desire"
        assert desire_dict["description"] == "A test desire"
        assert desire_dict["achieved"] is False

    def test_desire_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test_desire",
            "description": "A test desire",
            "priority": 0.7,
            "deadline": None,
            "conditions": {},
            "created_at": datetime.now().isoformat(),
            "achieved": False,
            "achieved_at": None,
        }

        desire = Desire.from_dict(data)

        assert desire.name == "test_desire"
        assert desire.description == "A test desire"
        assert desire.priority == 0.7
        assert desire.achieved is False


class TestPlan:
    """Tests for the Plan class."""

    def test_plan_initialization(self):
        """Test that a plan can be initialized with correct values."""
        actions = [
            {"action_type": "test", "action_id": "action1"},
            {"action_type": "test", "action_id": "action2"},
        ]

        plan = Plan(
            name="test_plan",
            desire_name="test_desire",
            actions=actions,
            context_conditions={"condition1": True},
        )

        assert plan.name == "test_plan"
        assert plan.desire_name == "test_desire"
        assert plan.actions == actions
        assert plan.context_conditions == {"condition1": True}
        assert plan.current_action_index == 0
        assert plan.complete is False
        assert plan.successful is None

    def test_plan_next_action(self):
        """Test getting the next action from a plan."""
        actions = [
            {"action_type": "test", "action_id": "action1"},
            {"action_type": "test", "action_id": "action2"},
        ]

        plan = Plan(name="test_plan", desire_name="test_desire", actions=actions)

        # Get first action
        action = plan.next_action()
        assert action == actions[0]

        # Advance to next action
        plan.advance()
        action = plan.next_action()
        assert action == actions[1]

        # Advance past end of actions
        plan.advance()
        action = plan.next_action()
        assert action is None

    def test_plan_mark_complete(self):
        """Test marking a plan as complete."""
        plan = Plan(name="test_plan", desire_name="test_desire", actions=[])

        # Mark as successful
        plan.mark_complete(True)

        assert plan.complete is True
        assert plan.successful is True

        # Create new plan and mark as failed
        plan = Plan(name="test_plan", desire_name="test_desire", actions=[])
        plan.mark_complete(False)

        assert plan.complete is True
        assert plan.successful is False

    def test_plan_to_dict(self):
        """Test conversion to dictionary."""
        actions = [{"action_type": "test", "action_id": "action1"}]
        plan = Plan(name="test_plan", desire_name="test_desire", actions=actions)

        plan_dict = plan.to_dict()

        assert plan_dict["name"] == "test_plan"
        assert plan_dict["desire_name"] == "test_desire"
        assert plan_dict["actions"] == actions
        assert plan_dict["complete"] is False

    def test_plan_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test_plan",
            "desire_name": "test_desire",
            "actions": [{"action_type": "test", "action_id": "action1"}],
            "context_conditions": {},
            "created_at": datetime.now().isoformat(),
            "current_action_index": 0,
            "complete": False,
            "successful": None,
            "action_results": [],
        }

        plan = Plan.from_dict(data)

        assert plan.name == "test_plan"
        assert plan.desire_name == "test_desire"
        assert len(plan.actions) == 1
        assert plan.current_action_index == 0
        assert plan.complete is False


class TestBDIState:
    """Tests for the BDIState class."""

    def test_state_initialization(self):
        """Test that a BDI state can be initialized."""
        state = BDIState()

        assert hasattr(state, "beliefs")
        assert hasattr(state, "desires")
        assert hasattr(state, "intentions")
        assert hasattr(state, "current_intention")

    def test_add_belief(self):
        """Test adding a belief to the state."""
        state = BDIState()

        # Add a belief
        belief = Belief(name="test_belief", value="test_value")
        state.add_belief(belief)

        assert "test_belief" in state.beliefs
        assert state.beliefs["test_belief"].value == "test_value"

    def test_update_belief(self):
        """Test updating a belief in the state."""
        state = BDIState()

        # Add a belief
        belief = Belief(name="test_belief", value="test_value")
        state.add_belief(belief)

        # Update the belief
        state.update_belief("test_belief", "new_value", 0.9)

        assert state.beliefs["test_belief"].value == "new_value"
        assert state.beliefs["test_belief"].confidence == 0.9

    def test_add_desire(self):
        """Test adding a desire to the state."""
        state = BDIState()

        # Add a desire
        desire = Desire(name="test_desire", description="A test desire")
        state.add_desire(desire)

        assert "test_desire" in state.desires
        assert state.desires["test_desire"].description == "A test desire"

    def test_get_desires_by_priority(self):
        """Test getting desires sorted by priority."""
        state = BDIState()

        # Add desires with different priorities
        desire1 = Desire(name="desire1", description="Desire 1", priority=0.5)
        desire2 = Desire(name="desire2", description="Desire 2", priority=0.8)
        desire3 = Desire(name="desire3", description="Desire 3", priority=0.2)

        state.add_desire(desire1)
        state.add_desire(desire2)
        state.add_desire(desire3)

        # Get desires by priority
        desires = state.get_desires_by_priority()

        assert len(desires) == 3
        assert desires[0].name == "desire2"  # Highest priority
        assert desires[1].name == "desire1"
        assert desires[2].name == "desire3"  # Lowest priority

    def test_add_intention(self):
        """Test adding an intention to the state."""
        state = BDIState()

        # Add an intention
        plan = Plan(name="test_plan", desire_name="test_desire", actions=[])
        state.add_intention(plan)

        assert len(state.intentions) == 1
        assert state.intentions[0].name == "test_plan"

    def test_set_current_intention(self):
        """Test setting the current intention."""
        state = BDIState()

        # Add an intention
        plan = Plan(name="test_plan", desire_name="test_desire", actions=[])

        # Set as current intention
        state.set_current_intention(plan)

        assert state.current_intention is not None
        assert state.current_intention.name == "test_plan"

        # Clear current intention
        state.set_current_intention(None)

        assert state.current_intention is None

    def test_to_dict_and_from_dict(self):
        """Test conversion to and from dictionary."""
        state = BDIState()

        # Add beliefs, desires, intentions
        state.add_belief(Belief(name="test_belief", value="test_value"))
        state.add_desire(Desire(name="test_desire", description="A test desire"))
        state.add_intention(
            Plan(name="test_plan", desire_name="test_desire", actions=[])
        )

        # Convert to dictionary
        state_dict = state.to_dict()

        assert "beliefs" in state_dict
        assert "desires" in state_dict
        assert "intentions" in state_dict

        # Create new state from dictionary
        new_state = BDIState.from_dict(state_dict)

        assert "test_belief" in new_state.beliefs
        assert "test_desire" in new_state.desires
        assert len(new_state.intentions) == 1


@pytest.mark.asyncio
class TestBDIAgent:
    """Tests for the BDIAgent class."""

    async def test_agent_initialization(self):
        """Test that a BDI agent can be initialized."""
        agent = BDIAgent(agent_id="test_agent")

        assert agent.id == "test_agent"
        assert isinstance(agent.state, BDIState)

        # Initialize agent
        await agent.initialize()

    async def test_agent_perceive_decide_act_cycle(self):
        """Test the perceive-decide-act cycle."""
        agent = BDIAgent(agent_id="test_agent")
        await agent.initialize()

        # Add test belief
        agent.state.add_belief(Belief(name="test_belief", value="test_value"))

        # Add test desire
        desire = Desire(name="test_desire", description="A test desire", priority=0.9)
        agent.state.add_desire(desire)

        # Add test plan
        actions = [
            {
                "action_type": "log",
                "action_id": "test_log",
                "parameters": {"level": "info", "message": "Test message"},
            }
        ]
        plan = Plan(name="test_plan", desire_name="test_desire", actions=actions)
        agent.register_action_handler("log", self._mock_log_handler)

        # Add plan to agent
        agent.state.add_intention(plan)
        agent.state.set_current_intention(plan)

        # Perceive - in a real scenario, this would get data from sensors
        await agent.perceive()

        # Decide
        action = await agent.decide()

        assert action is not None
        assert action["action_type"] == "log"
        assert action["action_id"] == "test_log"

        # Act
        result = await agent.act(action)

        assert result["status"] == "success"

        # Shutdown agent
        await agent.shutdown()

    @staticmethod
    async def _mock_log_handler(agent, action):
        """Mock handler for log actions."""
        return {
            "status": "success",
            "action_id": action["action_id"],
            "message": f"Logged: {action['parameters']['message']}",
        }
