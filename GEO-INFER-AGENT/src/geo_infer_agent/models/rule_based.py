#!/usr/bin/env python3

"""
Rule-based Agent.

This module implements a rule-based agent architecture that uses
predefined rules to determine behavior. The agent operates on
a set of conditions and actions, executing actions when
their associated conditions are met.
"""

import os
import logging
import asyncio
import re
from typing import Any, cast
from collections.abc import Callable
from datetime import datetime
import json

from geo_infer_agent.core.agent_base import BaseAgent, AgentState

logger = logging.getLogger("geo_infer_agent.models.rule_based")


class Rule:
    """
    Represents a single rule with condition and action.

    A rule consists of:
    - A condition that evaluates the current state
    - An action to execute when the condition is met
    - Optional metadata (priority, description, etc.)
    """

    def __init__(
        self,
        rule_id: str,
        condition: dict[str, Any] | Callable | str,
        action: dict[str, Any],
        priority: int = 0,
        description: str = "",
        enabled: bool = True,
    ):
        """
        Initialize a rule.

        Args:
            rule_id: Unique identifier for the rule
            condition: Condition specification (dict, function, or pattern string)
            action: Action to execute when condition is met
            priority: Priority level (higher numbers have higher priority)
            description: Human-readable description of the rule
            enabled: Whether the rule is currently enabled
        """
        self.id = rule_id
        self.condition = condition
        self.action = action
        self.priority = priority
        self.description = description
        self.enabled = enabled

        # Compile regex patterns if condition is a string
        self._compiled_pattern = None
        if isinstance(condition, str):
            try:
                self._compiled_pattern = re.compile(condition)
            except re.error:
                logger.warning(f"Failed to compile regex pattern for rule {rule_id}")

        # Stats
        self.match_count = 0
        self.last_matched: datetime | None = None

    def matches(self, state: dict[str, Any]) -> bool:
        """
        Check if rule condition matches the current state.

        Args:
            state: Current state dictionary

        Returns:
            True if the condition matches, False otherwise
        """
        if not self.enabled:
            return False

        matched = False

        # Different condition types
        if isinstance(self.condition, dict):
            # Dictionary condition: all key-value pairs must match
            matched = self._dict_condition_matches(state)
        elif callable(self.condition):
            # Function condition: call with state
            try:
                matched = bool(self.condition(state))
            except Exception as e:
                logger.error(
                    f"Error evaluating function condition for rule {self.id}: {e}"
                )
                matched = False
        elif isinstance(self.condition, str) and self._compiled_pattern:
            # Regex pattern: check against state_string field
            state_string = state.get("state_string", "")
            if isinstance(state_string, str):
                matched = bool(self._compiled_pattern.search(state_string))

        # Update stats if matched
        if matched:
            self.match_count += 1
            self.last_matched = datetime.now()

        return matched

    def _dict_condition_matches(self, state: dict[str, Any]) -> bool:
        """
        Check if dictionary condition matches state.

        Args:
            state: Current state dictionary

        Returns:
            True if all condition key-value pairs match state
        """
        cond: dict[str, Any] = cast(dict[str, Any], self.condition)
        for key, expected_value in cond.items():
            # Check if key exists in state
            if key not in state:
                return False

            actual_value = state[key]

            # Check value match
            if isinstance(expected_value, dict) and isinstance(actual_value, dict):
                # Recursive check for nested dictionaries
                if not self._nested_dict_matches(expected_value, actual_value):
                    return False
            elif expected_value != actual_value:
                return False

        return True

    def _nested_dict_matches(self, expected: dict, actual: dict) -> bool:
        """
        Check if nested dictionary matches.

        Args:
            expected: Expected dictionary values
            actual: Actual dictionary values

        Returns:
            True if all expected key-value pairs match actual dict
        """
        for key, expected_value in expected.items():
            if key not in actual:
                return False

            actual_value = actual[key]

            if isinstance(expected_value, dict) and isinstance(actual_value, dict):
                # Recursive check
                if not self._nested_dict_matches(expected_value, actual_value):
                    return False
            elif expected_value != actual_value:
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert rule to dictionary representation."""
        # Handle condition serialization
        if callable(self.condition):
            # For function conditions, we can't serialize the function
            # Instead, store a baseline
            condition_repr = {"_type": "function", "description": self.description}
        elif isinstance(self.condition, str):
            condition_repr = {"_type": "pattern", "pattern": self.condition}
        else:
            condition_repr = self.condition

        return {
            "id": self.id,
            "condition": condition_repr,
            "action": self.action,
            "priority": self.priority,
            "description": self.description,
            "enabled": self.enabled,
            "stats": {
                "match_count": self.match_count,
                "last_matched": (
                    self.last_matched.isoformat() if self.last_matched else None
                ),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rule":
        """Create rule from dictionary representation."""
        # Handle condition deserialization
        condition = data["condition"]
        if isinstance(condition, dict) and "_type" in condition:
            if condition["_type"] == "pattern":
                condition = condition["pattern"]
            elif condition["_type"] == "function":
                # A function cannot be deserialized; the rule never matches.
                logger.warning(
                    f"Cannot deserialize function condition for rule {data['id']}"
                )

                def condition(state: Any) -> bool:
                    """Never-matching stand-in for a non-serializable condition."""
                    return False

        rule = cls(
            rule_id=data["id"],
            condition=condition,
            action=data["action"],
            priority=data["priority"],
            description=data["description"],
            enabled=data["enabled"],
        )

        # Restore stats if available
        if "stats" in data:
            rule.match_count = data["stats"]["match_count"]
            if data["stats"]["last_matched"]:
                rule.last_matched = datetime.fromisoformat(
                    data["stats"]["last_matched"]
                )

        return rule


class RuleSet:
    """
    A collection of rules with management and selection functionality.
    """

    def __init__(self) -> None:
        """Initialize an empty rule set."""
        self.rules: dict[str, Rule] = {}  # rule_id -> Rule

    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the rule set.

        Args:
            rule: Rule to add
        """
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rule from the rule set.

        Args:
            rule_id: ID of rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def get_rule(self, rule_id: str) -> Rule | None:
        """
        Get a rule by ID.

        Args:
            rule_id: ID of rule to get

        Returns:
            Rule if found, None otherwise
        """
        return self.rules.get(rule_id)

    def enable_rule(self, rule_id: str) -> bool:
        """
        Enable a rule.

        Args:
            rule_id: ID of rule to enable

        Returns:
            True if rule was enabled, False if not found
        """
        rule = self.get_rule(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """
        Disable a rule.

        Args:
            rule_id: ID of rule to disable

        Returns:
            True if rule was disabled, False if not found
        """
        rule = self.get_rule(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False

    def find_matching_rules(self, state: dict[str, Any]) -> list[Rule]:
        """
        Find all rules that match the current state.

        Args:
            state: Current state dictionary

        Returns:
            List of matching rules sorted by priority (highest first)
        """
        matching_rules = []

        for rule in self.rules.values():
            if rule.matches(state):
                matching_rules.append(rule)

        # Sort by priority (highest first)
        return sorted(matching_rules, key=lambda r: r.priority, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert rule set to dictionary representation."""
        return {
            "rules": {rule_id: rule.to_dict() for rule_id, rule in self.rules.items()}
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleSet":
        """Create rule set from dictionary representation."""
        rule_set = cls()

        for rule_data in data["rules"].values():
            rule = Rule.from_dict(rule_data)
            rule_set.add_rule(rule)

        return rule_set


class RuleBasedState(AgentState):
    """
    State for a rule-based agent.

    Tracks rules, facts, and execution history.
    """

    def __init__(self) -> None:
        """Initialize rule-based agent state."""
        super().__init__()

        # Rule set
        self.rule_set = RuleSet()

        # Facts (current state knowledge)
        self.facts: dict[str, Any] = {}

        # Execution history
        self.execution_history: list[dict[str, Any]] = []
        self.max_history_size = 100

    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the agent's rule set.

        Args:
            rule: Rule to add
        """
        self.rule_set.add_rule(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rule from the agent's rule set.

        Args:
            rule_id: ID of rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        return self.rule_set.remove_rule(rule_id)

    def update_fact(self, key: str, value: Any) -> None:
        """
        Update a fact in the agent's knowledge base.

        Args:
            key: Fact key
            value: Fact value
        """
        self.facts[key] = value

    def get_fact(self, key: str, default: Any = None) -> Any:
        """
        Get a fact from the agent's knowledge base.

        Args:
            key: Fact key
            default: Default value if fact not found

        Returns:
            Fact value or default
        """
        return self.facts.get(key, default)

    def remove_fact(self, key: str) -> bool:
        """
        Remove a fact from the agent's knowledge base.

        Args:
            key: Fact key

        Returns:
            True if fact was removed, False if not found
        """
        if key in self.facts:
            del self.facts[key]
            return True
        return False

    def record_execution(
        self, rule_id: str, action: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """
        Record rule execution in history.

        Args:
            rule_id: ID of rule that was executed
            action: Action that was executed
            result: Result of the action
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "rule_id": rule_id,
            "action": action,
            "result": result,
        }

        self.execution_history.append(entry)

        # Trim history if needed
        while len(self.execution_history) > self.max_history_size:
            self.execution_history.pop(0)

    def find_matching_rules(self) -> list[Rule]:
        """
        Find all rules that match the current facts.

        Returns:
            List of matching rules sorted by priority
        """
        return self.rule_set.find_matching_rules(self.facts)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "rule_set": self.rule_set.to_dict(),
            "facts": self.facts,
            "execution_history": self.execution_history,
            "max_history_size": self.max_history_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleBasedState":
        """Create state from dictionary representation."""
        state = cls()

        state.rule_set = RuleSet.from_dict(data["rule_set"])
        state.facts = data["facts"]
        state.execution_history = data["execution_history"]
        state.max_history_size = data["max_history_size"]

        return state


class RuleBasedAgent(BaseAgent):
    """
    Implementation of a rule-based agent.

    This agent:
    1. Maintains a set of facts about the world
    2. Evaluates rules against these facts
    3. Executes actions based on matching rules
    """

    state: RuleBasedState

    def __init__(self, agent_id: str | None = None, config: dict | None = None):
        """
        Initialize rule-based agent.

        Args:
            agent_id: Unique identifier for the agent
            config: Configuration parameters
        """
        super().__init__(agent_id=agent_id, config=config or {})

        # Configuration
        self.config = config or {}

        # Initialize state
        self.state = RuleBasedState()

        # Configure state
        self._configure_state()

        # Register action handlers
        self._register_default_action_handlers()

        # Register perception handlers
        self._register_default_perception_handlers()

    def _configure_state(self) -> None:
        """Configure state from config."""
        # Set maximum history size
        if "max_history_size" in self.config:
            self.state.max_history_size = self.config["max_history_size"]

    async def initialize(self) -> None:
        """Initialize the agent."""
        logger.info(f"Initializing rule-based agent: {self.agent_id}")

        # Load rules from config
        self._load_rules_from_config()

        # Load initial facts
        self._load_initial_facts()

        # Load saved state if available
        state_path = self.config.get("state_path")
        if state_path and os.path.exists(state_path):
            self._load_state(state_path)

        logger.info("Rule-based agent %s initialization complete", self.agent_id)

    def _load_rules_from_config(self) -> None:
        """Load rules from configuration."""
        rules_config = self.config.get("rules", [])

        for rule_config in rules_config:
            try:
                rule = Rule(
                    rule_id=rule_config["id"],
                    condition=rule_config["condition"],
                    action=rule_config["action"],
                    priority=rule_config.get("priority", 0),
                    description=rule_config.get("description", ""),
                    enabled=rule_config.get("enabled", True),
                )
                self.state.add_rule(rule)
                logger.debug(f"Loaded rule: {rule.id}")
            except KeyError as e:
                logger.error(f"Missing required field in rule config: {e}")
            except Exception as e:
                logger.error(f"Error loading rule: {e}")

    def _load_initial_facts(self) -> None:
        """Load initial facts from configuration."""
        initial_facts = self.config.get("initial_facts", {})

        for key, value in initial_facts.items():
            self.state.update_fact(key, value)
            logger.debug(f"Set initial fact: {key} = {value}")

    async def perceive(self) -> dict[str, Any]:
        """
        Perceive the environment.

        Returns:
            Dictionary of perceptions
        """
        perceptions = self._configured_perceptions()

        # Update facts based on perceptions
        if perceptions:
            self._update_facts_from_perceptions(perceptions)

        return perceptions

    def update_beliefs(self, perception: dict[str, Any]) -> None:
        """Update facts from a perception dict (mirrors perceive())."""
        if perception:
            self._update_facts_from_perceptions(perception)

    def _update_facts_from_perceptions(self, perceptions: dict[str, Any]) -> None:
        """
        Update facts based on perceptions.

        Args:
            perceptions: Dictionary of perceptions
        """
        # Update facts from perceptions
        for key, value in perceptions.items():
            # Skip internal keys
            if key.startswith("_"):
                continue

            # Update fact
            self.state.update_fact(key, value)

        # Add timestamp
        self.state.update_fact("_last_perception_time", datetime.now().isoformat())

    async def decide(self) -> dict[str, Any] | None:
        """
        Decide on the next action.

        Returns:
            Action dictionary or None if no rule matches
        """
        # Find matching rules
        matching_rules = self.state.find_matching_rules()

        if not matching_rules:
            logger.debug("No matching rules found")

            # Get default action if configured
            default_action = self.config.get("default_action")
            if default_action:
                logger.debug("Using default action")
                return cast(dict[str, Any], default_action.copy())

            return None

        # Take highest priority rule
        selected_rule = matching_rules[0]
        logger.debug(f"Selected rule: {selected_rule.id}")

        # Store selected rule ID in action for later reference
        action = selected_rule.action.copy()
        action["_rule_id"] = selected_rule.id

        return action

    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Execute an action.

        Args:
            action: Action to execute

        Returns:
            Result of the action
        """
        # Extract rule ID
        rule_id = action.pop("_rule_id", "unknown")

        result = await self._dispatch_action(action)

        # Record execution
        self.state.record_execution(rule_id, action, result)

        # Update facts from result
        if "facts" in result:
            for key, value in result["facts"].items():
                self.state.update_fact(key, value)

        return result

    async def shutdown(self) -> None:
        """Clean up resources when shutting down the agent."""
        # Save state if configured
        if "state_save_path" in self.config:
            self._save_state(self.config["state_save_path"])

        logger.info("Rule-based agent %s shutdown complete", self.agent_id)

    def _register_default_action_handlers(self) -> None:
        """Register default action handlers."""
        self.register_action_handler("wait", self._handle_wait_action)
        self.register_action_handler("update_fact", self._handle_update_fact)
        self.register_action_handler("remove_fact", self._handle_remove_fact)
        self.register_action_handler("add_rule", self._handle_add_rule)
        self.register_action_handler("remove_rule", self._handle_remove_rule)
        self.register_action_handler("enable_rule", self._handle_enable_rule)
        self.register_action_handler("disable_rule", self._handle_disable_rule)
        self.register_action_handler("query_facts", self._handle_query_facts)

    def _register_default_perception_handlers(self) -> None:
        """Register default perception handlers."""
        self.register_perception_handler("sensor_data", self._handle_sensor_perceptions)

    async def _handle_wait_action(
        self, agent: "RuleBasedAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle a wait action.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        # Extract duration
        duration = action.get("parameters", {}).get("duration", 1.0)

        # Wait for the specified duration
        await asyncio.sleep(duration)

        return {
            "status": "success",
            "action_id": action.get("action_id", ""),
            "message": f"Waited for {duration} seconds",
        }

    async def _handle_update_fact(
        self, agent: "RuleBasedAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle updating a fact.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})

        if "key" not in params or "value" not in params:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": "Missing required parameters: key and value",
            }

        key = params["key"]
        value = params["value"]

        # Update fact
        agent.state.update_fact(key, value)

        return {
            "status": "success",
            "action_id": action.get("action_id", ""),
            "message": f"Updated fact: {key} = {value}",
            "facts": {key: value},
        }

    async def _handle_remove_fact(
        self, agent: "RuleBasedAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle removing a fact.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})

        if "key" not in params:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": "Missing required parameter: key",
            }

        key = params["key"]

        # Remove fact
        removed = agent.state.remove_fact(key)

        if removed:
            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "message": f"Removed fact: {key}",
            }
        else:
            return {
                "status": "warning",
                "action_id": action.get("action_id", ""),
                "message": f"Fact not found: {key}",
            }

    async def _handle_add_rule(
        self, agent: "RuleBasedAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle adding a rule.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})

        required_params = ["id", "condition", "action"]
        for param in required_params:
            if param not in params:
                return {
                    "status": "error",
                    "action_id": action.get("action_id", ""),
                    "message": f"Missing required parameter: {param}",
                }

        # Create rule
        try:
            rule = Rule(
                rule_id=params["id"],
                condition=params["condition"],
                action=params["action"],
                priority=params.get("priority", 0),
                description=params.get("description", ""),
                enabled=params.get("enabled", True),
            )

            # Add rule
            agent.state.add_rule(rule)

            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "message": f"Added rule: {rule.id}",
            }
        except Exception as e:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": f"Error adding rule: {e}",
            }

    async def _handle_remove_rule(
        self, agent: "RuleBasedAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle removing a rule.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})

        if "id" not in params:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": "Missing required parameter: id",
            }

        rule_id = params["id"]

        # Remove rule
        removed = agent.state.remove_rule(rule_id)

        if removed:
            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "message": f"Removed rule: {rule_id}",
            }
        else:
            return {
                "status": "warning",
                "action_id": action.get("action_id", ""),
                "message": f"Rule not found: {rule_id}",
            }

    async def _handle_enable_rule(
        self, agent: "RuleBasedAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle enabling a rule.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})

        if "id" not in params:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": "Missing required parameter: id",
            }

        rule_id = params["id"]

        # Enable rule
        enabled = agent.state.rule_set.enable_rule(rule_id)

        if enabled:
            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "message": f"Enabled rule: {rule_id}",
            }
        else:
            return {
                "status": "warning",
                "action_id": action.get("action_id", ""),
                "message": f"Rule not found: {rule_id}",
            }

    async def _handle_disable_rule(
        self, agent: "RuleBasedAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle disabling a rule.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})

        if "id" not in params:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": "Missing required parameter: id",
            }

        rule_id = params["id"]

        # Disable rule
        disabled = agent.state.rule_set.disable_rule(rule_id)

        if disabled:
            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "message": f"Disabled rule: {rule_id}",
            }
        else:
            return {
                "status": "warning",
                "action_id": action.get("action_id", ""),
                "message": f"Rule not found: {rule_id}",
            }

    async def _handle_query_facts(
        self, agent: "RuleBasedAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle querying facts.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary with requested facts
        """
        params = action.get("parameters", {})

        if "keys" in params:
            # Query specific facts
            keys = params["keys"]
            result_facts = {}

            for key in keys:
                if key in agent.state.facts:
                    result_facts[key] = agent.state.facts[key]

            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "facts": result_facts,
                "message": f"Queried {len(result_facts)} facts",
            }
        else:
            # Return all facts
            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "facts": agent.state.facts,
                "message": f"Queried all facts ({len(agent.state.facts)})",
            }

    def _handle_sensor_perceptions(
        self, agent: "RuleBasedAgent", perception: dict[str, Any]
    ) -> None:
        """
        Process sensor perceptions.

        Args:
            agent: Agent receiving the perception
            perception: Perception data
        """
        # Extract sensor readings
        sensor_readings = perception.get("readings", {})

        # Update facts with sensor readings
        for key, value in sensor_readings.items():
            agent.state.update_fact(f"sensor_{key}", value)

    def _save_state(self, path: str) -> None:
        """
        Save agent state to file.

        Args:
            path: Path to save state
        """
        state_data = self.state.to_dict()

        try:
            with open(path, "w") as f:
                json.dump(state_data, f, indent=2)
            logger.info(f"State saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self, path: str) -> None:
        """
        Load agent state from file.

        Args:
            path: Path to load state from
        """
        try:
            with open(path) as f:
                state_data = json.load(f)

            self.state = RuleBasedState.from_dict(state_data)
            logger.info(f"State loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
