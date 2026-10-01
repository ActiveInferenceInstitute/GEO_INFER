# GEO-INFER-[MODULE]

[One sentence describing what this module does and its primary use case within the GEO-INFER framework.]

## Overview

[2-3 sentences expanding on the module's purpose. Explain what problem it solves
and what domain it targets. Describe how this module connects to the GEO-INFER
ecosystem and Active Inference principles where applicable.]

[Optional: 1-2 sentences on the mathematical or theoretical foundations this
module implements.]

## Key Capabilities

- **[Capability 1 Name]**: [One sentence describing this capability.]
- **[Capability 2 Name]**: [One sentence describing this capability.]
- **[Capability 3 Name]**: [One sentence describing this capability.]
- **[Capability 4 Name]**: [One sentence describing this capability.]
- **[Capability 5 Name]**: [One sentence describing this capability.]

## Quick Start

### Installation

```bash
# From the GEO-INFER root directory
uv pip install -e ./GEO-INFER-[MODULE]

# With development dependencies
uv pip install -e "./GEO-INFER-[MODULE][dev]"
```

### Basic Usage

```python
from geo_infer_[module].core.[submodule] import [MainClass]

# Initialize
instance = [MainClass](
    [param1]=[value1],
    [param2]=[value2],
)

# Run core operation
result = instance.[primary_method])
print(result)
```

### Example: [Descriptive Name of Example]

```python
from geo_infer_[module].core.[submodule] import [MainClass]
import numpy as np

# [Step 1: Prepare input data]
[input_variable] = [realistic_data_creation]

# [Step 2: Create and configure the component]
[component] = [MainClass]([parameters])

# [Step 3: Execute the operation]
[result] = [component].[method]([input_variable])

# [Step 4: Inspect the output]
print(f"[Description]: {[result]}")
```

## Core Components

| Component | Module Path | Description |
|-----------|------------|-------------|
| [Component 1] | `geo_infer_[module].core.[name]` | [What it does] |
| [Component 2] | `geo_infer_[module].core.[name]` | [What it does] |
| [Component 3] | `geo_infer_[module].models.[name]` | [What it does] |
| [Component 4] | `geo_infer_[module].api.[name]` | [What it does] |
| [Component 5] | `geo_infer_[module].utils.[name]` | [What it does] |

### Package Structure

```text
GEO-INFER-[MODULE]/
    src/geo_infer_[module]/
        __init__.py
        core/
            __init__.py
            [core_module_1].py
            [core_module_2].py
        models/
            __init__.py
            [model_1].py
        api/
            __init__.py
            [api_module].py
        utils/
            __init__.py
            [utility].py
    tests/
        unit/
            test_[core_module_1].py
            test_[core_module_2].py
        integration/
            test_[integration_scenario].py
    pyproject.toml
    README.md
    SKILL.md
    AGENTS.md
```

## Integration

### Upstream Dependencies

This module depends on:

- **GEO-INFER-[DEP1]**: [What this module uses from DEP1]
- **GEO-INFER-[DEP2]**: [What this module uses from DEP2]

### Downstream Consumers

These modules use this module's output:

- **GEO-INFER-[CONSUMER1]**: [How CONSUMER1 uses this module]
- **GEO-INFER-[CONSUMER2]**: [How CONSUMER2 uses this module]

### Cross-Module Example

```python
from geo_infer_[module].core.[submodule] import [MainClass]
from geo_infer_[upstream].core.[submodule] import [UpstreamClass]

# Load data using upstream module
data = [UpstreamClass].[load_method]("[data_source]")

# Process using this module
processor = [MainClass]([parameters])
result = processor.[method](data)

# Result can be consumed by downstream modules
print(f"Output shape: {result.shape}")
```

## API Reference

### [MainClass]

```python
class [MainClass]:
    """[One-line description of the class purpose.]

    [2-3 sentences explaining what this class does and when to use it.]

    Attributes:
        [attr1]: [Description of attribute 1.]
        [attr2]: [Description of attribute 2.]
    """

    def __init__(self, [param1]: [Type1], [param2]: [Type2] = [default]):
        """Initialize [MainClass].

        Args:
            [param1]: [Description of parameter 1.]
            [param2]: [Description of parameter 2.]
        """

    def [primary_method]) -> [OutputType]:
        """[Imperative description of what this method does.]

        Args:
            [input]: [Description of the input parameter.]

        Returns:
            [Description of the return value.]

        Raises:
            ValueError: [When this error is raised.]
        """
```

### [SecondaryClass]

```python
class [SecondaryClass]:
    """[One-line description.]"""

    def [method](self, [param]: [Type]) -> [ReturnType]:
        """[Description.]

        Args:
            [param]: [Description.]

        Returns:
            [Description.]
        """
```

### Utility Functions

```python
def [utility_function]) -> [ReturnType]:
    """[Imperative description of the function.]

    Args:
        [param1]: [Description.]
        [param2]: [Description.]

    Returns:
        [Description.]
    """
```

## Configuration

### pyproject.toml

```toml
[project]
name = "geo-infer-[module]"
version = "0.1.0"
description = "[Module description]"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24.0",
    "[other_dependency]>=x.y.z",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "mypy>=1.0",
]
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEO_INFER_[MODULE]_[SETTING]` | [default] | [Description] |
| `GEO_INFER_LOG_LEVEL` | `INFO` | Logging verbosity |

## Testing

### Run All Tests

```bash
uv run python -m pytest GEO-INFER-[MODULE]/tests/ -v
```

### Run Unit Tests Only

```bash
uv run python -m pytest GEO-INFER-[MODULE]/tests/unit/ -v
```

### Run Integration Tests

```bash
uv run python -m pytest GEO-INFER-[MODULE]/tests/integration/ -v
```

### Run with Coverage

```bash
uv run python -m pytest GEO-INFER-[MODULE]/tests/ \
    --cov=GEO-INFER-[MODULE]/src \
    --cov-report=html \
    --cov-report=term-missing
```

### Run via Unified Test Runner

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module [MODULE]
```

## Documentation Hub

- [GEO-INFER Framework Overview](overview.md)
- [Active Inference Guide](active_inference_guide.md)
- [Geospatial Standards](geospatial_standards.md)
- [Data Dictionary](data_dictionary.md)
- [Installation Guide](installation.md)
- [Examples Gallery](examples_gallery.md)
- [Terminology Glossary](terminology.md)

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International —
see [LICENSE](../../LICENSE) for details.
