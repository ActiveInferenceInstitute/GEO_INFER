# GEO-INFER Documentation Guide

> **Illustrative example notice.** This page contains historical or
> conceptual integration sketches. Names such as `SpatialAnalyzer` and
> domain-specific facade classes are not public GEO-INFER exports in the
> current checkout; verify imports against each module's `src/` package
> and use the module README/tests for executable examples.


This guide defines the documentation standards for all 45 GEO-INFER modules.
Every document, docstring, README, and code example in the framework must follow
these conventions to maintain consistency and technical precision.

## Documentation Philosophy

### Technical Precision

GEO-INFER documentation uses precise, technical language. Avoid unnecessary
adjectives, superlatives, and marketing phrasing. Every sentence should either
convey factual information, explain a concept, or demonstrate usage.

**Do:**
- "This function computes the Haversine distance between two points."
- "The module provides Bayesian inference using MCMC and variational methods."
- "H3 resolution 9 produces cells of approximately 0.1 km^2."

**Do not:**
- "This powerful function efficiently computes distances in a sophisticated way."
- "Our cutting-edge module provides state-of-the-art Bayesian inference."
- "The highly optimized H3 system delivers exceptional spatial resolution."

### Active Inference Alignment

Documentation should reference Active Inference concepts where they apply to the
module's function. This does not mean forcing Active Inference terminology into
every sentence, but rather explaining how a module's capabilities connect to the
broader framework of perception, belief updating, and action selection.

### Concise Professional Language

- Use imperative mood for function descriptions: "Compute the free energy" not
  "This function computes the free energy."
- Use present tense: "Returns a GeoDataFrame" not "Will return a GeoDataFrame."
- Define acronyms on first use within each document.
- Keep paragraphs to 3-5 sentences. Break longer explanations into subsections.

## README Structure

Module-root `README.md` and `AGENTS.md` files are generated signposts
(`uv run python GEO-INFER-TEST/rewrite_readme_agents.py`) and are not written by
hand. The structure below applies to hand-written module overview pages, such
as `GEO-INFER-INTRA/docs/modules/geo-infer-<module>.md` and a module's `docs/`
pages, in this order:

### Required Sections

1. **Title and Badge Line**
   - Module name as H1 heading
   - Python version badge, license badge, test status badge

2. **Overview**
   - 2-3 sentences describing the module's purpose
   - How it relates to the GEO-INFER ecosystem
   - What problems it solves

3. **Key Capabilities**
   - Bulleted list of 4-8 primary features
   - Each item: bold feature name followed by 1-sentence description

4. **Quick Start**
   - Installation command: `uv sync --package geo-infer-module`
   - Minimal working code example (under 15 lines)
   - Expected output or result description

5. **Core Components**
   - Table with columns: Component, Module Path, Description
   - List every subpackage in `src/geo_infer_module/`

6. **Integration**
   - Which upstream modules this module depends on
   - Which downstream modules consume this module's output
   - Code example showing a cross-module workflow

7. **API Reference**
   - Key classes with constructor signatures
   - Key functions with parameter types and return types
   - Link to full API docs if generated

8. **Configuration**
   - `pyproject.toml` settings relevant to this module
   - Environment variables recognized by the module
   - Default values and overrides

9. **Testing**
   - Command to run the module's tests: `uv run python -m pytest GEO-INFER-MODULE/tests/ -v`
   - Test categories available (unit, integration, etc.)
   - Coverage command

10. **Documentation Hub**
    - Link to GEO-INFER-INTRA documentation hub
    - Links to relevant guides (active inference, geospatial standards, etc.)

11. **License**
    - "CC BY-NC-SA 4.0" with link to LICENSE file

## SKILL.md Structure

Each module contains a `SKILL.md` file that Claude Code auto-discovers. This file
teaches the AI assistant how to work with the module.

### Required Format

```markdown
---
name: GEO-INFER-MODULE Skill
description: Brief description of what this skill enables
prerequisites:
  - Python 3.11+
  - uv package manager
  - List specific dependencies
difficulty: beginner | intermediate | advanced
estimated_time: 15 minutes | 30 minutes | 1 hour
---

# GEO-INFER-MODULE

## Instructions

[Detailed instructions for Claude Code on how to work with this module.
Include import patterns, common operations, testing commands, and
integration patterns with other modules.]

## Examples

[3-5 concrete examples showing common tasks. Each example should include
the input, the code to run, and the expected output.]

## Common Issues

[List 3-5 common errors and their solutions.]
```

### SKILL.md Guidelines

- Keep instructions action-oriented: "To create a spatial index, use..."
- Include error recovery patterns: "If import fails, check that..."
- Reference the module's actual class and function names from source code
- Update SKILL.md whenever the module's public API changes

## AGENTS.md Structure

Each module has an `AGENTS.md` describing its capabilities for multi-agent
orchestration.

### Required Format

```markdown
# GEO-INFER-MODULE Agent Capabilities

## Capabilities

- **Capability 1**: What this module can do as an agent component
- **Capability 2**: Another capability
- ...

## Integration Patterns

### Input Formats
[What data this module accepts and from which other modules]

### Output Formats
[What data this module produces and for which other modules]

## Agent Communication

[How this module participates in multi-agent workflows,
including message formats and coordination patterns]
```

## Code Example Guidelines

All code examples in documentation must follow these rules:

### Must Be Functional

Every code example must run without modification when the module is installed.
No placeholder comments like `# Implementation here` or `# TODO: add logic`.

```python
# CORRECT: functional example
import numpy as np
from geo_infer_act.core.free_energy import FreeEnergyCalculator

calculator = FreeEnergyCalculator()
beliefs = np.array([0.25, 0.25, 0.25, 0.25])
observations = np.array([0.7, 0.1, 0.1, 0.1])
fe = calculator.compute_categorical_free_energy(beliefs, observations)
print(f"Free energy: {fe:.4f}")
```

```python
# INCORRECT: non-functional stub
from geo_infer_act import SomeClass
result = SomeClass().do_something()  # process the data
# ... more processing ...
```

### Use Real Imports

Import from actual module paths that exist in the codebase. Do not invent
module paths or class names.

### Use Realistic Data

Examples should use data that makes domain sense. For geospatial examples, use
real coordinates (Portland: 45.5231, -122.6765). For Active Inference, use
probability vectors that sum to 1.0.

### Include Expected Output

Where practical, show what the code produces:

```python
cell = h3.latlng_to_cell(45.5231, -122.6765, 9)
print(cell)
# Output: 8928308280fffff
```

### Code Block Language Tags

Always specify the language in fenced code blocks:
- Python: ` ```python `
- Bash: ` ```bash `
- JSON: ` ```json `
- YAML: ` ```yaml `
- Plain text or output: ` ```text `

## Link Conventions

### Internal Links

Use relative paths from the document's location:

```markdown
<!-- From GEO-INFER-INTRA/docs/overview.md -->
[Installation Guide](installation.md)
[ACT Module](../../GEO-INFER-ACT/README.md)
```

### External Links

Use full URLs with descriptive link text:

```markdown
[H3 Documentation](https://h3geo.org/docs/)
[GeoPandas User Guide](https://geopandas.org/en/stable/docs/user_guide.html)
```

### Cross-Module References

When referencing another GEO-INFER module from documentation, use the format:

```markdown
See [GEO-INFER-BAYES](../../GEO-INFER-BAYES/README.md) for Bayesian inference.
```

## Docstring Standards

All Python code uses Google-style docstrings with full type annotations:

```python
def compute_spatial_autocorrelation(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    method: str = "moran",
    weights: str | None = "queen",
) -> dict[str, float]:
    """Compute spatial autocorrelation statistics for a GeoDataFrame.

    Calculates the specified spatial autocorrelation statistic using the
    given spatial weights matrix. Supports Moran's I and Geary's C.

    Args:
        gdf: GeoDataFrame with geometry and value columns. Must have a
            valid CRS set.
        value_column: Name of the column containing numeric values to
            analyze.
        method: Autocorrelation method. One of "moran" (Moran's I) or
            "geary" (Geary's C). Default: "moran".
        weights: Spatial weights type. One of "queen", "rook", or "knn".
            Default: "queen".

    Returns:
        Dictionary with keys:
        - "statistic": The computed autocorrelation value.
        - "p_value": Statistical significance.
        - "z_score": Standard normal deviate.
        - "expected": Expected value under null hypothesis.

    Raises:
        ValueError: If value_column is not in gdf or contains non-numeric data.
        ValueError: If gdf has no CRS set.
    """
```

## Review Checklist

Before submitting documentation, verify all items:

1. [ ] All code examples run without errors when the module is installed
2. [ ] All imports reference real module paths and class names
3. [ ] No marketing language, superlatives, or unnecessary adjectives
4. [ ] All acronyms defined on first use
5. [ ] Type annotations present on all function signatures in examples
6. [ ] Links tested and pointing to correct targets
7. [ ] H3 v4 API used (latlng_to_cell, not geo_to_h3)
8. [ ] CRS explicitly set on all GeoDataFrame examples
9. [ ] Probability vectors sum to 1.0 in Active Inference examples
10. [ ] SKILL.md YAML front matter complete with all required fields

## Common Documentation Anti-Patterns

### 1. Aspirational Documentation

Writing documentation for features that do not exist yet. Every documented
function, class, or workflow must have a working implementation in the codebase.

### 2. Copy-Paste Syndrome

Duplicating the same explanation across multiple documents. Instead, write it
once in the canonical location and link to it from other documents.

### 3. Screenshot-Only Explanations

Relying on screenshots to explain code output. Always include text-based output
alongside any visual. Screenshots break when code changes; text can be validated.

### 4. Undocumented Parameters

Listing parameters in a function signature but omitting them from the docstring.
Every parameter must have a description, type, and default value (if applicable).

### 5. Stale Examples

Code examples that referenced an older API version. When the API changes, all
examples referencing the changed functions must be updated in the same commit.

### 6. Vague Error Descriptions

Writing "Raises an error if input is invalid" without specifying which exception
type or what constitutes invalid input.

### 7. Missing CRS Context

Geospatial examples that create geometries or coordinates without specifying the
coordinate reference system.

### 8. Mixing Coordinate Orders

Using `(lat, lng)` in one example and `(lng, lat)` in another without flagging
the difference. Always specify which convention is in use.

## Related Documents

- [Module README Template](module_readme_template.md) -- copy-paste template
- [Terminology](terminology.md) -- standard terms and definitions
- [Geospatial Standards](geospatial_standards.md) -- CRS and format conventions
- [Data Dictionary](data_dictionary.md) -- data structure reference
