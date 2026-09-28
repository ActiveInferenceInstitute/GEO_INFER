# Active Inference Diagrams

## Overview

This document provides visual diagrams explaining Active Inference concepts and their implementation in GEO-INFER agents.

## The Free Energy Principle

```mermaid
graph TB
    subgraph "Free Energy Minimization"
        FE[Free Energy F]
        ACC[Accuracy]
        COMP[Complexity]
        
        FE --> |=| COMP
        COMP --> |minus| ACC
    end
    
    subgraph "Two Routes"
        PERC[Perception<br>Update beliefs]
        ACT[Action<br>Change world]
    end
    
    FE --> PERC
    FE --> ACT
```

## Agent Architecture

```mermaid
graph LR
    subgraph Agent
        GM[Generative<br>Model]
        B[Beliefs<br>q(s)]
        PE[Policy<br>Evaluator]
        L[Learner]
    end
    
    subgraph Environment
        S[States]
        O[Observations]
    end
    
    O -->|perceive| B
    B -->|plan| PE
    PE -->|act| S
    S --> O
    O -.->|learn| L
    L -.-> GM
```

## Generative Model Structure

```mermaid
graph TD
    subgraph "Generative Model p(o,s)"
        D[Prior p(s₀)]
        B[Transitions p(s'|s,a)]
        A[Likelihood p(o|s)]
        C[Preferences p(o)]
    end
    
    D --> |initial state| B
    B --> |state evolution| A
    A --> |observations| C
```

## Expected Free Energy Components

```mermaid
graph LR
    subgraph "Expected Free Energy G"
        EPI[Epistemic Value<br>Information Gain]
        PRAG[Pragmatic Value<br>Goal Achievement]
    end
    
    EPI --> |exploration| G[G = Epistemic + Pragmatic]
    PRAG --> |exploitation| G
```

## Perception-Action Loop

```mermaid
sequenceDiagram
    participant E as Environment
    participant A as Agent
    participant M as Model
    
    E->>A: Observation o
    A->>M: Update beliefs q(s)
    M->>A: Posterior beliefs
    A->>M: Evaluate policies G(π)
    M->>A: Policy probabilities
    A->>E: Execute action a
    E->>E: State transition
```

## Multi-Agent Coordination

```mermaid
graph TB
    subgraph "Swarm of Agents"
        A1[Agent 1]
        A2[Agent 2]
        A3[Agent 3]
    end
    
    subgraph "Shared Beliefs"
        SB[Common<br>World Model]
    end
    
    subgraph "Environment"
        ENV[Geospatial<br>Environment]
    end
    
    A1 <--> SB
    A2 <--> SB
    A3 <--> SB
    
    A1 --> ENV
    A2 --> ENV
    A3 --> ENV
```

## Hierarchical State Space (H3)

```mermaid
graph TD
    subgraph "H3 Hierarchy"
        R4[Resolution 4<br>~1,000 km]
        R6[Resolution 6<br>~36 km]
        R8[Resolution 8<br>~460 m]
        R10[Resolution 10<br>~65 m]
    end
    
    R4 --> R6
    R6 --> R8
    R8 --> R10
```

## Belief Update Dynamics

```mermaid
graph LR
    subgraph "Belief Update"
        PRIOR[Prior Beliefs<br>q(s)]
        OBS[Observation o]
        LIK[Likelihood<br>p(o|s)]
        POST[Posterior<br>q(s|o)]
    end
    
    PRIOR --> |×| LIK
    OBS --> LIK
    LIK --> |normalize| POST
```


