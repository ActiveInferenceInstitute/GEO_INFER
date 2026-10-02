## GEO-INFER-AGENT — Autonomous Agents with Active Inference, BDI, and RL

**Purpose.** GEO-INFER-AGENT provides intelligent autonomous agents for geospatial decision-making, perception, and action. Its README enumerates three agent architectures — Active Inference, BDI (belief–desire–intention), and reinforcement learning — making this module the behavioral layer that sits on top of the framework's inference primitives. A `tools/` directory supplements the standard `src/`/`tests/` layout.

**Public API.** The `geo_infer_agent` package's `__all__` groups four agent families with matching state objects: `BDIAgent`/`BDIState`, `ActiveInferenceAgent`/`ActiveInferenceState` (with an associated `GenerativeModel`), `RLAgent`/`RLState`, `RuleBasedAgent`/`RuleBasedState`, and `HybridAgent`/`HybridState`. BDI cognition is represented by `Belief`, `Desire`, and `Plan`. All agents share the `BaseAgent`/`AgentState` contract and register through `AgentRegistry`. Inter-agent coordination is provided by `MessagingService`/`Message` and observability by `TelemetryService`, both from the `api` subpackage; a `llm_proxy` component rounds out the core.

**Verification status.** The `tests/` directory is present with 28 test files, covering the agent families and the messaging/telemetry services; testing routes through the unified runner (`--module AGENT`).

**Theme role.** Agents: this is the framework's primary agent runtime. It consumes active inference machinery from the ACT band and exposes agent endpoints upward through the APP interface band.
