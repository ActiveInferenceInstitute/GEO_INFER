"""
GEO-INFER-AGENT - Autonomous agent framework for geospatial applications

This module provides functionality for:
- Creating intelligent autonomous agents
- Orchestrating multi-agent systems
- Implementing active inference for decision-making
- Supporting various agent architectures
- Handling geospatial perception and action
"""

__version__ = "0.4.0"

# All imported subpackages ship with this module, so their imports are
# unconditional: a failure here is a packaging bug and must surface, not be
# swallowed. Genuinely optional third-party dependencies (numpy, torch, ...)
# are guarded inside the submodules themselves with explicit HAS_<DEP> flags.
from geo_infer_agent.core.agent_base import BaseAgent, AgentState
from geo_infer_agent.core.agent_registry import AgentRegistry

from geo_infer_agent.models import (
    BDIAgent,
    BDIState,
    ActiveInferenceAgent,
    ActiveInferenceState,
    GenerativeModel,
    RLAgent,
    RLState,
    RuleBasedAgent,
    RuleBasedState,
    HybridAgent,
    HybridState,
    Belief,
    Desire,
    Plan,
)

from geo_infer_agent.api.messaging import MessagingService, Message
from geo_infer_agent.api.telemetry import TelemetryService

from geo_infer_agent.core.llm_proxy import (
    LLMProxyPolicy,
    LLMProxyPolicyError,
    TokenBucket,
    check_allowed_model,
    check_output_tokens,
    check_request_size,
    enforce_llm_proxy_policy,
)

__all__ = [
    "__version__",
    "BaseAgent",
    "AgentState",
    "AgentRegistry",
    "BDIAgent",
    "BDIState",
    "ActiveInferenceAgent",
    "ActiveInferenceState",
    "GenerativeModel",
    "RLAgent",
    "RLState",
    "RuleBasedAgent",
    "RuleBasedState",
    "HybridAgent",
    "HybridState",
    "Belief",
    "Desire",
    "Plan",
    "MessagingService",
    "Message",
    "TelemetryService",
    "LLMProxyPolicy",
    "LLMProxyPolicyError",
    "TokenBucket",
    "check_allowed_model",
    "check_output_tokens",
    "check_request_size",
    "enforce_llm_proxy_policy",
]
