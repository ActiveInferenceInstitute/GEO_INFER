# Active Inference for Agents

> **Two implementations, one canonical export.** This module ships two
> Active Inference implementations under the same class name
> `ActiveInferenceAgent`:
>
> 1. `geo_infer_agent.models.active_inference.ActiveInferenceAgent` — numpy
>    matrix model. **This is the canonical implementation** and the one
>    exported from the package root (`from geo_infer_agent import
>    ActiveInferenceAgent`). It is config-dict-driven and dependency-light.
> 2. `geo_infer_agent.core.active_inference.ActiveInferenceAgent` — torch
>    neural generative model (`GenerativeModel` with encode/plan/update).
>    It is NOT exported from the package root; import it explicitly from
>    `geo_infer_agent.core.active_inference` when you need the deep-learning
>    variant.
>
> The two share only the name and are not interchangeable.

## Introduction

This document describes how Active Inference is implemented in the GEO-INFER agent framework, enabling autonomous agents that perceive, act, and learn in geospatial environments.

## Agent Architecture

### Core Components

An Active Inference agent consists of:

1. **Generative Model**: Internal model of how the world works
2. **Belief State**: Current beliefs about hidden states
3. **Policy Evaluator**: Selects actions via expected free energy
4. **Learning Module**: Updates model parameters

```python
class ActiveInferenceAgent:
    def __init__(self, generative_model):
        self.model = generative_model
        self.beliefs = initialize_beliefs()
        self.policy_evaluator = PolicyEvaluator(self.model)
        self.learner = ModelLearner(self.model)
```

## The Perception-Action Loop

### 1. Observation

Agent receives sensory input from environment:

```python
observation = environment.observe(agent.location)
```

### 2. Belief Update (Perception)

Update beliefs to minimize variational free energy:

```python
agent.beliefs = agent.model.infer_states(
    observation=observation,
    prior=agent.beliefs
)
```

### 3. Policy Evaluation

Evaluate policies by computing expected free energy:

```python
G = []
for policy in agent.policies:
    efe = agent.policy_evaluator.expected_free_energy(
        beliefs=agent.beliefs,
        policy=policy
    )
    G.append(efe)
```

### 4. Action Selection

Select action from best policy:

```python
policy_probs = softmax(-precision * G)
selected_policy = sample(policies, policy_probs)
action = selected_policy[0]
```

### 5. Learning

Update model from experience:

```python
agent.learner.update(
    observation=observation,
    beliefs=agent.beliefs,
    action=action
)
```

## Geospatial Agent Types

The package ships concrete agent architectures under `geo_infer_agent.models`.
Create them directly or through the registry:

### Rule-Based Survey Agent

```python
from geo_infer_agent import RuleBasedAgent

agent = RuleBasedAgent(
    agent_id="survey-01",
    config={"name": "survey_agent", "description": "Cover the study area"},
)
```

### Monitoring with Reinforcement Learning

```python
from geo_infer_agent import RLAgent

agent = RLAgent(
    agent_id="monitor-01",
    config={"name": "air_quality_monitor"},
)
```

### Coordinating through the Registry

```python
from geo_infer_agent import AgentRegistry

registry = AgentRegistry()
survey_id = await registry.create_agent(
    agent_type="rule_based",
    config={"name": "survey_agent"},
    region=study_area_geojson,
)
monitor_id = await registry.create_agent(
    agent_type="reinforcement_learning",
    config={"name": "air_quality_monitor"},
)
await registry.start_agent(survey_id)
await registry.start_agent(monitor_id)
```

## Multi-Agent Systems

### Decentralized Coordination

Multiple agents share beliefs and coordinate:

```python
from geo_infer_agent import AgentRegistry, MessagingService, Message

registry = AgentRegistry()
messaging = MessagingService()

coordinator_id = await registry.create_agent(
    agent_type="active_inference",
    config={"name": "coverage_coordinator"},
)

message = Message(
    from_agent_id=coordinator_id,
    to_agent_id="survey-01",
    content={"instruction": "expand_coverage"},
    message_type="request",
)
delivered = await messaging.send_message(message)
```

## Integration with GEO-INFER

| Module | Agent Integration |
|--------|-------------------|
| **GEO-INFER-ACT** | Core inference |
| **GEO-INFER-SPACE** | Spatial states |
| **GEO-INFER-IOT** | Sensor data |
| **GEO-INFER-COMMS** | Agent messaging |
