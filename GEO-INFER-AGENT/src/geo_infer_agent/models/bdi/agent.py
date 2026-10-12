"""
BDI Agent implementation.

This module contains the BDIState and BDIAgent classes that implement the
Belief-Desire-Intention cognitive architecture for geospatial agents.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, cast
from collections.abc import Callable

from geo_infer_agent.core.agent_base import BaseAgent, AgentState
from geo_infer_agent.models.bdi.belief import Belief
from geo_infer_agent.models.bdi.desire import Desire
from geo_infer_agent.models.bdi.plan import Plan

logger = logging.getLogger("geo_infer_agent.models.bdi")

# ---------------------------------------------------------------------------
# BDIState
# ---------------------------------------------------------------------------


class BDIState(AgentState):
    """
    Extended agent state for BDI agents.

    Tracks beliefs, desires, and intentions (plans the agent is committed to).
    """

    def __init__(self, capacity: int = 1000) -> None:
        super().__init__(capacity)
        self.beliefs_dict: dict[str, Belief] = {}
        self.desires_dict: dict[str, Desire] = {}
        self.intentions: list[Plan] = []  # type: ignore[assignment]
        self.current_intention: Plan | None = None
        # Expose beliefs/desires via the parent-class attribute names.
        self.beliefs = self.beliefs_dict
        self.desires = self.desires_dict  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Belief management
    # ------------------------------------------------------------------

    def add_belief(self, belief: Belief) -> None:
        """Add a new belief or merge into an existing one."""
        if belief.name in self.beliefs_dict:
            old = self.beliefs_dict[belief.name]
            old_value = old.value
            old_confidence = old.confidence
            old.update(
                value=belief.value,
                confidence=belief.confidence,
                metadata=belief.metadata,
            )
            self.add_to_memory(
                {
                    "type": "belief_updated",
                    "name": belief.name,
                    "old_value": old_value,
                    "new_value": belief.value,
                    "old_confidence": old_confidence,
                    "new_confidence": belief.confidence,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            self.beliefs_dict[belief.name] = belief
            self.add_to_memory(
                {
                    "type": "belief_added",
                    "name": belief.name,
                    "value": belief.value,
                    "confidence": belief.confidence,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        self.last_update = datetime.now()

    def update_belief(
        self,
        name: str,
        value: Any,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update an existing belief or create it if absent."""
        if name in self.beliefs_dict:
            old = self.beliefs_dict[name]
            old_value = old.value
            old_confidence = old.confidence
            old.update(value=value, confidence=confidence, metadata=metadata)
            self.add_to_memory(
                {
                    "type": "belief_updated",
                    "name": name,
                    "old_value": old_value,
                    "new_value": value,
                    "old_confidence": old_confidence,
                    "new_confidence": confidence
                    if confidence is not None
                    else old_confidence,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            self.beliefs_dict[name] = Belief(
                name=name,
                value=value,
                confidence=confidence if confidence is not None else 1.0,
                metadata=metadata or {},
            )
            self.add_to_memory(
                {
                    "type": "belief_added",
                    "name": name,
                    "value": value,
                    "confidence": confidence if confidence is not None else 1.0,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        self.last_update = datetime.now()

    def get_belief(self, name: str) -> Belief | None:
        """Return the named belief, or None if absent."""
        return self.beliefs_dict.get(name)

    # ------------------------------------------------------------------
    # Desire management
    # ------------------------------------------------------------------

    def add_desire(self, desire: Desire) -> None:  # type: ignore[override]
        """Add a desire to the desire set."""
        self.desires_dict[desire.name] = desire
        self.add_to_memory(
            {
                "type": "desire_added",
                "name": desire.name,
                "description": desire.description,
                "priority": desire.priority,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_desire(self, name: str) -> Desire | None:
        """Return the named desire, or None if absent."""
        return self.desires_dict.get(name)

    def get_desires_by_priority(self) -> list[Desire]:
        """Return all desires ordered by priority (highest first)."""
        return sorted(
            self.desires_dict.values(), key=lambda d: d.priority, reverse=True
        )

    # ------------------------------------------------------------------
    # Intention (plan) management
    # ------------------------------------------------------------------

    def add_intention(self, plan: Plan) -> None:
        """Commit to a new intention."""
        self.intentions.append(plan)
        self.add_to_memory(
            {
                "type": "intention_added",
                "plan_name": plan.name,
                "desire_name": plan.desire_name,
                "actions_count": len(plan.actions),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def set_current_intention(self, plan: Plan | None) -> None:
        """Set the currently active intention."""
        self.current_intention = plan
        if plan:
            self.add_to_memory(
                {
                    "type": "intention_selected",
                    "plan_name": plan.name,
                    "desire_name": plan.desire_name,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            self.add_to_memory(
                {"type": "intention_cleared", "timestamp": datetime.now().isoformat()}
            )

    def get_current_intention(self) -> Plan | None:
        """Return the currently active intention."""
        return self.current_intention

    def get_intentions_for_desire(self, desire_name: str) -> list[Plan]:
        """Return all non-complete intentions targeting the given desire."""
        return [
            p
            for p in self.intentions
            if p.desire_name == desire_name and not p.complete
        ]

    def remove_completed_intentions(self) -> int:
        """Remove all completed intentions and return the count removed."""
        before = len(self.intentions)
        self.intentions = [i for i in self.intentions if not i.complete]
        return before - len(self.intentions)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "beliefs": {n: b.to_dict() for n, b in self.beliefs_dict.items()},
                "desires": {n: d.to_dict() for n, d in self.desires_dict.items()},
                "intentions": [i.to_dict() for i in self.intentions],
                "current_intention": self.current_intention.to_dict()
                if self.current_intention
                else None,
            }
        )
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BDIState":
        state = cls()
        if "beliefs" in data:
            for name, bdata in data["beliefs"].items():
                if isinstance(bdata, dict):
                    state.beliefs_dict[name] = Belief.from_dict(bdata)
                else:
                    state.beliefs_dict[name] = Belief(name=name, value=bdata)
        if "desires" in data:
            for name, ddata in data["desires"].items():
                if isinstance(ddata, dict):
                    state.desires_dict[name] = Desire.from_dict(ddata)
        if "intentions" in data:
            state.intentions = [Plan.from_dict(p) for p in data["intentions"]]
        if data.get("current_intention"):
            state.current_intention = Plan.from_dict(data["current_intention"])
        return state


# ---------------------------------------------------------------------------
# BDIAgent
# ---------------------------------------------------------------------------


class BDIAgent(BaseAgent):
    """
    Belief-Desire-Intention (BDI) agent.

    Implements the BDI cognitive architecture, cycling through:
    1. Perceive the environment.
    2. Update beliefs from perceptions.
    3. Deliberate over desires to select an intention.
    4. Execute the next action of the current intention.
    """

    state: BDIState

    def __init__(
        self, agent_id: str | None = None, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(agent_id, config)

        # Convenience alias used by some callers.
        self.id = self.agent_id

        self.state = BDIState(capacity=self.config.get("memory_capacity", 1000))
        self.plan_library: dict[str, dict[str, Any]] = {}
        self.action_handlers: dict[str, Callable[..., Any]] = {}
        self.perception_handler_list: list[Callable[..., Any]] = []

        self.deliberation_interval: float = self.config.get("deliberation_interval", 5)
        self.commitment_strategy: str = self.config.get(
            "commitment_strategy", "single_minded"
        )

        logger.info("BDI agent %s initialized", self.agent_id)

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_action_handler(self, action_type: str, handler: Callable) -> None:
        """Register a callable to handle actions of the given type."""
        self.action_handlers[action_type] = handler
        logger.debug("Registered action handler for type: %s", action_type)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Set up default handlers, load config data, and seed beliefs/desires."""
        logger.info("Initializing BDI agent %s", self.agent_id)
        self._register_default_action_handlers()
        self._register_default_perception_handlers()
        self._load_plans_from_config()
        self._initialize_beliefs()
        self._initialize_desires()
        logger.info("BDI agent %s initialization complete", self.agent_id)

    async def perceive(self) -> dict[str, Any]:
        """
        Return a perception dict from the environment.

        Subclasses should override this to integrate real sensors or data
        sources.  Sensor readings can also be provided via the
        ``sensor_readings`` key in the agent config for deterministic or
        test deployments.
        """
        perceptions: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
        }
        if "region" in self.config:
            perceptions["region"] = self.config["region"]

        static_readings = self.config.get("sensor_readings")
        if static_readings and isinstance(static_readings, dict):
            perceptions["sensors"] = dict(static_readings)
        else:
            perceptions["sensors"] = {}

        logger.debug("BDI agent %s perceptions: %s", self.agent_id, perceptions)
        return perceptions

    def update_beliefs(self, perception: dict[str, Any]) -> None:
        """Update beliefs from a perception dict by running all perception handlers."""
        for handler in self.perception_handler_list:
            handler(self, perception)
        self.state.update_belief("last_perception_time", datetime.now())
        logger.debug("BDI agent %s beliefs updated from perception", self.agent_id)

    async def decide(self) -> dict[str, Any] | None:
        """
        Select the next action to perform.

        Continues the current intention when valid, or deliberates to find a
        new one.
        """
        current = self.state.get_current_intention()
        if current and not current.complete and self._is_intention_valid(current):
            action = current.next_action()
            if action:
                logger.debug(
                    "BDI agent %s continuing intention %s", self.agent_id, current.name
                )
                return action

        # Clear stale/invalid intention and deliberate.
        self.state.set_current_intention(None)
        for desire in self.state.get_desires_by_priority():
            if desire.achieved or desire.is_expired():
                continue
            plan = self._find_plan_for_desire(desire.name)
            if plan:
                self.state.set_current_intention(plan)
                action = plan.next_action()
                if action:
                    logger.debug(
                        "BDI agent %s selected intention %s", self.agent_id, plan.name
                    )
                    return action

        logger.debug("BDI agent %s found no valid intentions", self.agent_id)
        return None

    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute the given action using registered handlers."""
        if not action or ("type" not in action and "action_type" not in action):
            logger.warning(
                "BDI agent %s received invalid action: %s", self.agent_id, action
            )
            return {"success": False, "error": "Invalid action"}

        action_type = action.get("type") or action.get("action_type")
        if action_type not in self.action_handlers:
            logger.warning(
                "BDI agent %s has no handler for action %s", self.agent_id, action_type
            )
            return {
                "success": False,
                "error": f"No handler for action type {action_type}",
            }

        try:
            result = await self.action_handlers[action_type](self, action)
            current = self.state.get_current_intention()
            if current:
                current.record_action_result(
                    current.current_action_index, result, result.get("success", False)
                )
                if result.get("success", False):
                    current.advance()
                    if current.complete and self._is_desire_satisfied(
                        current.desire_name
                    ):
                        desire = self.state.get_desire(current.desire_name)
                        if desire:
                            desire.set_achieved(True)
                            logger.info(
                                "BDI agent %s achieved desire %s",
                                self.agent_id,
                                desire.name,
                            )
            return cast(dict[str, Any], result)
        except Exception as exc:
            logger.error(
                "BDI agent %s error executing action %s: %s",
                self.agent_id,
                action_type,
                exc,
            )
            return {"success": False, "error": str(exc)}

    async def shutdown(self) -> None:
        """Shut down the agent."""
        logger.info("BDI agent %s shutting down", self.agent_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register_default_action_handlers(self) -> None:
        self.action_handlers["wait"] = self._handle_wait_action
        self.action_handlers["update_belief"] = self._handle_update_belief_action
        self.action_handlers["query_belief"] = self._handle_query_belief_action
        self.action_handlers["log"] = self._handle_log_action

    def _register_default_perception_handlers(self) -> None:
        self.perception_handler_list.append(self._handle_sensor_perceptions)

    def _handle_sensor_perceptions(
        self, agent: "BDIAgent", perception: dict[str, Any]
    ) -> None:
        if "sensors" in perception:
            for sensor_name, value in perception["sensors"].items():
                agent.state.update_belief(f"sensor.{sensor_name}", value)

    async def _handle_wait_action(
        self, agent: "BDIAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        duration = action.get("duration", 1)
        await asyncio.sleep(duration)
        return {"success": True, "duration": duration}

    async def _handle_update_belief_action(
        self, agent: "BDIAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        belief_name = action.get("belief_name")
        if not belief_name:
            return {"success": False, "error": "Missing belief name"}
        agent.state.update_belief(
            belief_name,
            action.get("belief_value"),
            action.get("confidence"),
            action.get("metadata"),
        )
        return {
            "success": True,
            "belief_name": belief_name,
            "belief_value": action.get("belief_value"),
        }

    async def _handle_query_belief_action(
        self, agent: "BDIAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        belief_name = action.get("belief_name")
        if not belief_name:
            return {"success": False, "error": "Missing belief name"}
        belief = agent.state.get_belief(belief_name)
        if not belief:
            return {"success": False, "error": f"Belief {belief_name} not found"}
        return {
            "success": True,
            "belief_name": belief_name,
            "belief_value": belief.value,
            "confidence": belief.confidence,
            "timestamp": belief.timestamp.isoformat(),
        }

    async def _handle_log_action(
        self, agent: "BDIAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        message = action.get("message", "")
        level = action.get("level", "info")
        log_fn = {
            "debug": logger.debug,
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
        }.get(level, logger.info)
        log_fn("BDI agent %s: %s", agent.agent_id, message)
        return {"success": True, "message": message, "level": level}

    def _load_plans_from_config(self) -> None:
        for template in self.config.get("plans", []):
            if not all(k in template for k in ("name", "desire_name", "actions")):
                logger.warning(
                    "BDI agent %s skipping invalid plan template: %s",
                    self.agent_id,
                    template,
                )
                continue
            self.plan_library[template["name"]] = template
        logger.debug(
            "BDI agent %s loaded %d plan templates",
            self.agent_id,
            len(self.plan_library),
        )

    def _initialize_beliefs(self) -> None:
        for belief_name, belief_data in self.config.get("initial_beliefs", {}).items():
            if isinstance(belief_data, dict):
                belief = Belief(
                    name=belief_name,
                    value=belief_data.get("value"),
                    confidence=belief_data.get("confidence", 1.0),
                    metadata=belief_data.get("metadata", {}),
                )
                self.state.add_belief(belief)
            else:
                self.state.update_belief(belief_name, belief_data)
        logger.debug("BDI agent %s initialized beliefs from config", self.agent_id)

    def _initialize_desires(self) -> None:
        for desire_data in self.config.get("initial_desires", []):
            if "name" not in desire_data or "description" not in desire_data:
                logger.warning(
                    "BDI agent %s skipping invalid desire: %s",
                    self.agent_id,
                    desire_data,
                )
                continue
            deadline = None
            if "deadline" in desire_data:
                try:
                    deadline = datetime.fromisoformat(desire_data["deadline"])
                except (ValueError, TypeError):
                    logger.warning(
                        "BDI agent %s invalid deadline format: %s",
                        self.agent_id,
                        desire_data["deadline"],
                    )
            desire = Desire(
                name=desire_data["name"],
                description=desire_data["description"],
                priority=desire_data.get("priority", 0.5),
                deadline=deadline,
                conditions=desire_data.get("conditions", {}),
            )
            self.state.add_desire(desire)
        logger.debug("BDI agent %s initialized desires from config", self.agent_id)

    def _find_plan_for_desire(self, desire_name: str) -> Plan | None:
        """Find or create a plan for the given desire."""
        # Reuse an existing non-complete intention.
        for plan in self.state.get_intentions_for_desire(desire_name):
            if not plan.complete:
                return plan

        # Instantiate from the plan library.
        for plan_name, template in self.plan_library.items():
            if template["desire_name"] != desire_name:
                continue
            if self._check_context_conditions(template.get("context_conditions", {})):
                plan = Plan(
                    name=plan_name,
                    desire_name=desire_name,
                    actions=self._resolve_placeholders(template["actions"]),
                    context_conditions=template.get("context_conditions", {}),
                )
                self.state.add_intention(plan)
                return plan
        return None

    def _resolve_placeholders(self, value: Any) -> Any:
        """
        Resolve ``$CONFIG:<key>`` placeholders in plan-template values.

        Plan templates loaded from config may reference agent configuration
        values with the ``$CONFIG:<key>`` syntax (e.g. a wait duration of
        ``$CONFIG:collection_interval``).  Strings that consist exactly of
        one ``$CONFIG:`` token are replaced by the configured value itself (so
        numeric config values stay numeric); nested dicts and lists are
        resolved recursively.

        Args:
            value: A plan-template value (action dict, list, or scalar)

        Returns:
            The value with placeholders resolved

        Raises:
            ValueError: If a ``$CONFIG:`` reference names a missing config key
        """
        if isinstance(value, str):
            match = re.fullmatch(r"\$CONFIG:([A-Za-z_][A-Za-z0-9_]*)", value)
            if match:
                key = match.group(1)
                if key not in self.config:
                    raise ValueError(
                        f"Plan template references unknown config key {key!r}"
                    )
                return self.config[key]
            return value
        if isinstance(value, dict):
            return {k: self._resolve_placeholders(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_placeholders(item) for item in value]
        return value

    def _check_context_conditions(self, conditions: dict[str, Any]) -> bool:
        """Return True if every condition matches the corresponding belief value."""
        for belief_name, expected in conditions.items():
            belief = self.state.get_belief(belief_name)
            if not belief or belief.value != expected:
                return False
        return True

    def _is_intention_valid(self, intention: Plan) -> bool:
        """Return True if the intention's desire still exists, is unachieved, and unexpired."""
        desire = self.state.get_desire(intention.desire_name)
        if not desire or desire.achieved or desire.is_expired():
            return False
        return self._check_context_conditions(intention.context_conditions)

    def _is_desire_satisfied(self, desire_name: str) -> bool:
        """Return True if all conditions of the desire match current beliefs."""
        desire = self.state.get_desire(desire_name)
        if not desire:
            return False
        for belief_name, expected in desire.conditions.items():
            belief = self.state.get_belief(belief_name)
            if not belief or belief.value != expected:
                return False
        return True
