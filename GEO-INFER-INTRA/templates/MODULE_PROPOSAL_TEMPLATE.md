---
title: "GEO-INFER-{MODULE}: {Module Title}"
description: "{Brief description of module purpose and capabilities}"
purpose: "{Explanation of what this module provides}"
module_theme: "{README Module Theme, e.g. Spatial & Place-based, Governance, Risk & Domain}"
status: "Planning"
dependencies: ["{DEP1}", "{DEP2}", "{DEP3}"]
optional_dependencies: ["{OPT_DEP1}", "{OPT_DEP2}"]
estimated_development_time: "{X-Y months}"
tags: ["{tag1}", "{tag2}", "{tag3}"]
---

# GEO-INFER-{MODULE}: {Module Title}

> **Purpose**: {One-sentence description of module purpose}
>
> {2-3 sentence explanation of what this module provides and why it is needed}

## Overview

{Overview of the module, its role in the GEO-INFER ecosystem, and key concepts}

### Core Concept

{Explain the core concept or methodology this module implements}

## Core Objectives

- **Objective 1**: {Description}
- **Objective 2**: {Description}
- **Objective 3**: {Description}

## Core Features

### 1. {Feature Name}

**Purpose**: {What this feature does}

**Capabilities**:

- {Capability 1}
- {Capability 2}

**Example**:

```python
from geo_infer_{module} import {MainClass}

instance = {MainClass}(param1=value1, param2=value2)
result = instance.{method}(args)
```

### 2. {Feature Name}

{Repeat the structure for each major feature}

## Integration with GEO-INFER Modules

### GEO-INFER-{DEPENDENCY} Integration

{Describe how this module integrates with each dependency}

```python
from geo_infer_{dependency} import {DependencyClass}
from geo_infer_{module} import {Class}

dependency_instance = {DependencyClass}()
module_instance = {Class}(dependency=dependency_instance)
```

## Use Cases

### 1. {Use Case Name}

**Scenario**: {Description of the use case scenario}

**Modules used**: {List of modules involved}

## API Reference

### Core Classes

#### {ClassName}

{Description of the class}

```python
from typing import Any

import numpy as np

from geo_infer_{module}.utils.rng import SeedLike


class {ClassName}:
    def __init__(
        self,
        param1: {Type},
        param2: {Type},
        config: dict[str, Any] | None = None,
        seed: SeedLike = None,
    ) -> None:
        """Initialize {ClassName}.

        Args:
            param1: {Description}
            param2: {Description}
            config: {Description}
            seed: Seed or ``numpy.random.Generator`` resolved via ``resolve_rng``.
        """

    def {method_name}(self, arg1: {Type}, arg2: {Type}) -> {ReturnType}:
        """{Method description}.

        Args:
            arg1: {Description}
            arg2: {Description}

        Returns:
            {Return description}
        """
```

## Key Concepts

### {Concept Name}

{Explanation of key concepts, methodologies, or theoretical foundations}

## Configuration

```yaml
# config/example.yaml
module_config:
  setting1: value1
  setting2: value2
```

## Testing and Validation

{Testing approach and validation methods}

## Getting Started

### Installation

The module is a member of the root uv workspace:

```bash
uv sync --package geo-infer-{module}
```

### Quick Start Example

```python
from geo_infer_{module} import {MainClass}

instance = {MainClass}()
result = instance.{method}(data)
```

## Related Documentation

- [GEO-INFER Framework README](../../README.md)
- [Module Development Guidelines](./MODULE_DEVELOPMENT_GUIDELINES.md)
- [Module Integration Guide](../docs/guides/MODULE_INTEGRATION_GUIDE.md)

## Further Reading

{References to academic papers, documentation, or related resources}
