"""
Agent Configuration Module

Provides utilities for configuring agents in the GEO-INFER-APP UI.
This module includes schema definitions, validation, and UI components
for configuring different types of agents.
"""

from typing import Any
import logging
import math
from enum import Enum
from dataclasses import dataclass, field
from .agent_interface import AgentType

# Configure logging
logger = logging.getLogger(__name__)


class ConfigFieldType(Enum):
    """Enumeration of supported configuration field types."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    SELECT = "select"
    MULTISELECT = "multiselect"
    GEOLOCATION = "geolocation"
    FILE = "file"
    CODE = "code"
    COLOR = "color"


@dataclass
class ConfigField:
    """Definition of a configuration field for agent configuration."""

    name: str
    field_type: ConfigFieldType
    label: str
    description: str | None = None
    default_value: Any = None
    required: bool = False
    options: list[dict[str, Any]] | None = None
    validation: dict[str, Any] | None = None
    dependencies: list[str] | None = None
    group: str | None = None
    order: int = 0
    advanced: bool = False


@dataclass
class AgentConfigSchema:
    """Schema for agent configuration."""

    agent_type: AgentType
    title: str
    description: str
    version: str
    fields: list[ConfigField] = field(default_factory=list)
    groups: list[dict[str, Any]] | None = field(default_factory=list)


class AgentConfiguration:
    """
    Class for managing agent configurations.

    This class provides:
    1. Schema definitions for different agent types
    2. Validation of agent configurations
    3. Default configurations for different agent types
    4. Conversion between UI-friendly and agent-compatible formats
    """

    # Dictionary mapping agent types to configuration schemas
    _schemas: dict[AgentType, AgentConfigSchema] = {}

    @classmethod
    def register_schema(cls, schema: AgentConfigSchema) -> None:
        """
        Register a configuration schema for an agent type.

        Args:
            schema: Configuration schema to register
        """
        cls._schemas[schema.agent_type] = schema
        logger.info(
            f"Registered configuration schema for agent type {schema.agent_type.value}"
        )

    @classmethod
    def get_schema(cls, agent_type: AgentType) -> AgentConfigSchema:
        """
        Get the configuration schema for the specified agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Configuration schema for the agent type

        Raises:
            ValueError: If no schema is registered for the agent type
        """
        if agent_type not in cls._schemas:
            # Generate a default schema
            default_schema = cls._generate_default_schema(agent_type)
            cls._schemas[agent_type] = default_schema
            logger.warning(f"Using default schema for agent type {agent_type.value}")
            return default_schema

        return cls._schemas[agent_type]

    @classmethod
    def validate_config(
        cls, agent_type: AgentType, config: dict[str, Any]
    ) -> list[str]:
        """
        Validate a configuration against the schema for the specified agent type.

        Args:
            agent_type: Type of agent
            config: Configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        schema = cls.get_schema(agent_type)
        errors = []

        # Check required fields
        for field in schema.fields:
            if field.required and (
                field.name not in config or config[field.name] is None
            ):
                errors.append(f"Missing required field: {field.name}")

        # Check field types and validation rules
        for field_name, field_value in config.items():
            matching_fields = [f for f in schema.fields if f.name == field_name]
            if not matching_fields:
                errors.append(f"Unknown field: {field_name}")
                continue

            field = matching_fields[0]

            # Type validation
            if field.field_type == ConfigFieldType.STRING and not isinstance(
                field_value, str
            ):
                errors.append(f"Field {field_name} must be a string")
            elif field.field_type == ConfigFieldType.NUMBER and not isinstance(
                field_value, (int, float)
            ):
                errors.append(f"Field {field_name} must be a number")
            elif field.field_type == ConfigFieldType.BOOLEAN and not isinstance(
                field_value, bool
            ):
                errors.append(f"Field {field_name} must be a boolean")
            elif field.field_type == ConfigFieldType.OBJECT and not isinstance(
                field_value, dict
            ):
                errors.append(f"Field {field_name} must be an object")
            elif field.field_type == ConfigFieldType.ARRAY and not isinstance(
                field_value, list
            ):
                errors.append(f"Field {field_name} must be an array")
            elif (
                field.field_type == ConfigFieldType.GEOLOCATION
                and not AgentConfiguration._validate_geolocation(field_value)
            ):
                errors.append(
                    f"Field {field_name} must be a geolocation object with numeric "
                    f"'lat' in [-90, 90] and 'lng' in [-180, 180]"
                )

            # Select/multiselect validation
            if (
                field.field_type
                in (ConfigFieldType.SELECT, ConfigFieldType.MULTISELECT)
                and field.options
            ):
                valid_values = {opt["value"] for opt in field.options}
                if field.field_type == ConfigFieldType.SELECT:
                    if field_value not in valid_values:
                        errors.append(f"Invalid value for {field_name}: {field_value}")
                else:
                    if not isinstance(field_value, list):
                        errors.append(f"Field {field_name} must be a list")
                    else:
                        for value in field_value:
                            if value not in valid_values:
                                errors.append(f"Invalid value in {field_name}: {value}")

            # Custom validation rules
            if field.validation:
                # min/max only apply to numeric values; comparing a non-number
                # against min/max would raise TypeError instead of validating.
                is_number = isinstance(field_value, (int, float)) and not isinstance(
                    field_value, bool
                )
                if (
                    "min" in field.validation
                    and is_number
                    and field_value < field.validation["min"]
                ):
                    errors.append(
                        f"Field {field_name} must be at least {field.validation['min']}"
                    )
                if (
                    "max" in field.validation
                    and is_number
                    and field_value > field.validation["max"]
                ):
                    errors.append(
                        f"Field {field_name} must be at most {field.validation['max']}"
                    )
                if "pattern" in field.validation and isinstance(field_value, str):
                    import re

                    if not re.match(field.validation["pattern"], field_value):
                        errors.append(
                            f"Field {field_name} does not match required pattern"
                        )

        return errors

    @staticmethod
    def _validate_geolocation(value: Any) -> bool:
        """
        Validate a geolocation value: {"lat": <number>, "lng": <number>}
        with lat in [-90, 90] and lng in [-180, 180].

        Args:
            value: Candidate geolocation value

        Returns:
            True if value is a well-formed, in-bounds geolocation
        """
        if not isinstance(value, dict):
            return False
        lat = value.get("lat")
        lng = value.get("lng")
        if not isinstance(lat, (int, float)) or isinstance(lat, bool):
            return False
        if not isinstance(lng, (int, float)) or isinstance(lng, bool):
            return False
        if not (math.isfinite(lat) and math.isfinite(lng)):
            return False
        return -90 <= lat <= 90 and -180 <= lng <= 180

    @classmethod
    def get_default_config(cls, agent_type: AgentType) -> dict[str, Any]:
        """
        Get a default configuration for the specified agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Default configuration dictionary
        """
        schema = cls.get_schema(agent_type)
        default_config = {}

        for field in schema.fields:
            if field.default_value is not None:
                default_config[field.name] = field.default_value

        return default_config

    @staticmethod
    def _generate_default_schema(agent_type: AgentType) -> AgentConfigSchema:
        """
        Generate a default schema for the specified agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Default configuration schema
        """
        fields = [
            ConfigField(
                name="name",
                field_type=ConfigFieldType.STRING,
                label="Agent Name",
                description="A unique name for this agent",
                required=True,
                order=1,
            ),
            ConfigField(
                name="description",
                field_type=ConfigFieldType.STRING,
                label="Description",
                description="A brief description of this agent's purpose",
                required=False,
                order=2,
            ),
        ]

        # Add type-specific fields
        if agent_type == AgentType.BDI:
            fields.extend(
                [
                    ConfigField(
                        name="beliefs",
                        field_type=ConfigFieldType.OBJECT,
                        label="Initial Beliefs",
                        description="Initial beliefs for the BDI agent",
                        required=False,
                        group="BDI Components",
                        order=10,
                    ),
                    ConfigField(
                        name="desires",
                        field_type=ConfigFieldType.ARRAY,
                        label="Desires",
                        description="List of agent desires",
                        required=False,
                        group="BDI Components",
                        order=11,
                    ),
                    ConfigField(
                        name="intention_selection",
                        field_type=ConfigFieldType.SELECT,
                        label="Intention Selection Strategy",
                        description="Strategy for selecting intentions from desires",
                        default_value="priority",
                        options=[
                            {"label": "Priority-based", "value": "priority"},
                            {"label": "Utility-based", "value": "utility"},
                            {"label": "Context-sensitive", "value": "context"},
                        ],
                        group="BDI Components",
                        order=12,
                    ),
                ]
            )
        elif agent_type == AgentType.ACTIVE_INFERENCE:
            fields.extend(
                [
                    ConfigField(
                        name="precision",
                        field_type=ConfigFieldType.NUMBER,
                        label="Precision",
                        description="Precision parameter for active inference",
                        default_value=1.0,
                        validation={"min": 0.0, "max": 10.0},
                        group="Active Inference Parameters",
                        order=10,
                    ),
                    ConfigField(
                        name="learning_rate",
                        field_type=ConfigFieldType.NUMBER,
                        label="Learning Rate",
                        description="Learning rate for model updates",
                        default_value=0.1,
                        validation={"min": 0.0, "max": 1.0},
                        group="Active Inference Parameters",
                        order=11,
                    ),
                ]
            )
        elif agent_type == AgentType.RL:
            fields.extend(
                [
                    ConfigField(
                        name="learning_rate",
                        field_type=ConfigFieldType.NUMBER,
                        label="Learning Rate",
                        description="Learning rate for RL algorithms",
                        default_value=0.1,
                        validation={"min": 0.0, "max": 1.0},
                        group="RL Parameters",
                        order=10,
                    ),
                    ConfigField(
                        name="discount_factor",
                        field_type=ConfigFieldType.NUMBER,
                        label="Discount Factor",
                        description="Discount factor for future rewards",
                        default_value=0.9,
                        validation={"min": 0.0, "max": 1.0},
                        group="RL Parameters",
                        order=11,
                    ),
                    ConfigField(
                        name="exploration_rate",
                        field_type=ConfigFieldType.NUMBER,
                        label="Exploration Rate",
                        description="Initial exploration rate",
                        default_value=0.3,
                        validation={"min": 0.0, "max": 1.0},
                        group="RL Parameters",
                        order=12,
                    ),
                    ConfigField(
                        name="algorithm",
                        field_type=ConfigFieldType.SELECT,
                        label="RL Algorithm",
                        description="Reinforcement learning algorithm to use",
                        default_value="q_learning",
                        options=[
                            {"label": "Q-Learning", "value": "q_learning"},
                            {"label": "SARSA", "value": "sarsa"},
                            {"label": "DQN", "value": "dqn"},
                            {"label": "PPO", "value": "ppo"},
                        ],
                        group="RL Parameters",
                        order=13,
                    ),
                ]
            )
        elif agent_type == AgentType.RULE_BASED:
            fields.extend(
                [
                    ConfigField(
                        name="rules",
                        field_type=ConfigFieldType.ARRAY,
                        label="Rules",
                        description="List of rule definitions",
                        required=True,
                        group="Rule Definitions",
                        order=10,
                    ),
                    ConfigField(
                        name="conflict_resolution",
                        field_type=ConfigFieldType.SELECT,
                        label="Conflict Resolution Strategy",
                        description="Strategy for resolving conflicts between rules",
                        default_value="priority",
                        options=[
                            {"label": "Priority", "value": "priority"},
                            {"label": "Specificity", "value": "specificity"},
                            {"label": "Recency", "value": "recency"},
                        ],
                        group="Rule Definitions",
                        order=11,
                    ),
                ]
            )

        # Common fields for all agent types
        fields.extend(
            [
                ConfigField(
                    name="initial_location",
                    field_type=ConfigFieldType.GEOLOCATION,
                    label="Initial Location",
                    description="Starting location for this agent",
                    required=False,
                    group="Spatial Parameters",
                    order=20,
                ),
                ConfigField(
                    name="movement_speed",
                    field_type=ConfigFieldType.NUMBER,
                    label="Movement Speed",
                    description="Movement speed in units per time step",
                    default_value=1.0,
                    validation={"min": 0.0},
                    group="Spatial Parameters",
                    order=21,
                ),
                ConfigField(
                    name="sensor_radius",
                    field_type=ConfigFieldType.NUMBER,
                    label="Sensor Radius",
                    description="Radius within which the agent can sense",
                    default_value=10.0,
                    validation={"min": 0.0},
                    group="Spatial Parameters",
                    order=22,
                ),
                ConfigField(
                    name="communication_enabled",
                    field_type=ConfigFieldType.BOOLEAN,
                    label="Enable Communication",
                    description="Whether the agent can communicate with other agents",
                    default_value=True,
                    group="Communication",
                    order=30,
                ),
                ConfigField(
                    name="communication_range",
                    field_type=ConfigFieldType.NUMBER,
                    label="Communication Range",
                    description="Maximum distance for agent communication",
                    default_value=20.0,
                    validation={"min": 0.0},
                    dependencies=["communication_enabled"],
                    group="Communication",
                    order=31,
                ),
            ]
        )

        # Define groups
        groups = [
            {"name": "Basic", "label": "Basic Information", "order": 1},
            {"name": "Spatial Parameters", "label": "Spatial Parameters", "order": 2},
            {"name": "Communication", "label": "Communication", "order": 3},
        ]

        # Add type-specific groups
        if agent_type == AgentType.BDI:
            groups.append(
                {"name": "BDI Components", "label": "BDI Components", "order": 4}
            )
        elif agent_type == AgentType.ACTIVE_INFERENCE:
            groups.append(
                {
                    "name": "Active Inference Parameters",
                    "label": "Active Inference Parameters",
                    "order": 4,
                }
            )
        elif agent_type == AgentType.RL:
            groups.append(
                {"name": "RL Parameters", "label": "RL Parameters", "order": 4}
            )
        elif agent_type == AgentType.RULE_BASED:
            groups.append(
                {"name": "Rule Definitions", "label": "Rule Definitions", "order": 4}
            )

        return AgentConfigSchema(
            agent_type=agent_type,
            title=f"{agent_type.name} Agent Configuration",
            description=f"Configuration schema for {agent_type.name} agents",
            version="1.0.0",
            fields=fields,
            groups=groups,
        )
