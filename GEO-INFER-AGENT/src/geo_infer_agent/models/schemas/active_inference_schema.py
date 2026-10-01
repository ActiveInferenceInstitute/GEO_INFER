"""
JSON schemas for Active Inference components.

This module provides JSON schemas for validating:
- Generative model configurations
- Active inference state representations
- Active inference agent configurations

These schemas can be used for configuration validation and API documentation.
"""

from typing import Any

# Schema for generative model configuration
GENERATIVE_MODEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["state_dimensions", "observation_dimensions", "control_dimensions"],
    "properties": {
        "state_dimensions": {
            "type": "integer",
            "description": "Number of dimensions in the state space",
            "minimum": 1,
        },
        "observation_dimensions": {
            "type": "integer",
            "description": "Number of dimensions in the observation space",
            "minimum": 1,
        },
        "control_dimensions": {
            "type": "integer",
            "description": "Number of possible control actions",
            "minimum": 1,
        },
        "learning_rate": {
            "type": "number",
            "description": "Rate at which the model updates based on new evidence",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.01,
        },
        "initial_A": {
            "type": "array",
            "description": "Initial likelihood mapping (observation given state)",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "initial_B": {
            "type": "array",
            "description": "Initial transition probabilities (next state given current state and action)",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
            },
        },
        "initial_C": {
            "type": "array",
            "description": "Initial prior preferences over observations",
            "items": {"type": "number"},
        },
        "initial_D": {
            "type": "array",
            "description": "Initial prior beliefs about states",
            "items": {"type": "number"},
        },
    },
}

# Schema for active inference state
ACTIVE_INFERENCE_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["state_dimensions", "observation_dimensions", "control_dimensions"],
    "properties": {
        "state_dimensions": {
            "type": "integer",
            "description": "Number of dimensions in the state space",
            "minimum": 1,
        },
        "observation_dimensions": {
            "type": "integer",
            "description": "Number of dimensions in the observation space",
            "minimum": 1,
        },
        "control_dimensions": {
            "type": "integer",
            "description": "Number of possible control actions",
            "minimum": 1,
        },
        "generative_model": {
            "type": "object",
            "description": "The agent's generative model configuration",
            "$ref": "#/definitions/generative_model",
        },
        "observation_history": {
            "type": "array",
            "description": "History of observations",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "format": "date-time"},
                    "observation": {"type": "array", "items": {"type": "number"}},
                    "state_belief": {"type": "array", "items": {"type": "number"}},
                },
            },
        },
        "action_history": {
            "type": "array",
            "description": "History of actions",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "format": "date-time"},
                    "action": {"type": "integer"},
                    "reward": {"type": "number"},
                },
            },
        },
    },
}

# Schema for active inference agent configuration
ACTIVE_INFERENCE_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "Unique identifier for the agent",
        },
        "state_dimensions": {
            "type": "integer",
            "description": "Number of dimensions in the state space",
            "minimum": 1,
            "default": 10,
        },
        "observation_dimensions": {
            "type": "integer",
            "description": "Number of dimensions in the observation space",
            "minimum": 1,
            "default": 10,
        },
        "control_dimensions": {
            "type": "integer",
            "description": "Number of possible control actions",
            "minimum": 1,
            "default": 5,
        },
        "learning_rate": {
            "type": "number",
            "description": "Learning rate for model updates",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.01,
        },
        "planning_horizon": {
            "type": "integer",
            "description": "Number of steps to look ahead when planning",
            "minimum": 1,
            "default": 1,
        },
        "perception_handlers": {
            "type": "object",
            "description": "Custom perception handler functions",
            "additionalProperties": {"type": "string"},
        },
        "action_handlers": {
            "type": "object",
            "description": "Custom action handler functions",
            "additionalProperties": {"type": "string"},
        },
        "model_path": {
            "type": "string",
            "description": "Path to load/save the agent's model",
        },
        "observation_encoder": {
            "type": "string",
            "description": "Name of the encoder to use for processing raw observations",
        },
        "action_decoder": {
            "type": "string",
            "description": "Name of the decoder to use for converting action indices to actions",
        },
    },
}

SCHEMAS = {
    "generative_model": GENERATIVE_MODEL_SCHEMA,
    "active_inference_state": ACTIVE_INFERENCE_STATE_SCHEMA,
    "active_inference_agent": ACTIVE_INFERENCE_AGENT_SCHEMA,
}
