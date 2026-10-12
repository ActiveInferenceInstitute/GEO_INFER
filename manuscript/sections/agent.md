## GEO-INFER-AGENT — Autonomous Agents with Active Inference, BDI, and RL

**Purpose.** GEO-INFER-AGENT provides intelligent autonomous agents for geospatial decision-making, perception, and action. Its README describes agent architectures — Active Inference, BDI (belief–desire–intention), and reinforcement learning, alongside rule-based and hybrid models — making this module the behavioral layer that sits on top of the framework's inference primitives. A `tools/` directory supplements the standard `src/`/`tests/` layout.

**Public API.** The `geo_infer_agent` package's `__all__` groups agent families with matching state objects: `BDIAgent`/`BDIState`, `ActiveInferenceAgent`/`ActiveInferenceState` (with an associated `GenerativeModel`), `RLAgent`/`RLState`, `RuleBasedAgent`/`RuleBasedState`, and `HybridAgent`/`HybridState`. BDI cognition is represented by `Belief`, `Desire`, and `Plan`. All agents share the `BaseAgent`/`AgentState` contract and register through `AgentRegistry`. Inter-agent coordination is provided by `MessagingService`/`Message` and observability by `TelemetryService`, both from the `api` subpackage; `LLMProxyPolicy` and its validation helpers enforce the declared language-model request limits.

The tests exercise agent families, asynchronous lifecycle behavior, messaging, and telemetry. The shared repository runner selects this module with `--module AGENT`.

**Theme role.** Agents: this is the framework's primary agent runtime. It consumes active inference machinery from the ACT band and exposes agent endpoints upward through the APP interface band.
