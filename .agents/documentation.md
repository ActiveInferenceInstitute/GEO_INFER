# Documentation Standards

## Docstring Format (Google-style)

```python
def compute_risk(
    region: dict[str, Any],
    hazard_type: str,
    return_period: int = 100,
) -> dict[str, float]:
    """Compute risk metrics for a geographic region.

    Calculates expected annual loss (EAL), probable maximum loss (PML),
    and tail value at risk (TVaR) using the region's exposure and
    vulnerability data.

    Args:
        region: GeoJSON-like dict with 'features' containing exposure data.
        hazard_type: One of 'earthquake', 'flood', 'hurricane', 'wildfire'.
        return_period: Return period in years for PML calculation.

    Returns:
        Dictionary with keys 'eal', 'pml', 'tvar'.

    Raises:
        ValueError: If hazard_type is not recognised.
        DataValidationError: If region features are malformed.

    Example:
        >>> result = compute_risk(region_data, 'earthquake', 250)
        >>> result['eal']
        1250000.0
    """
```

Every public function/method must include: `Args`, `Returns`, `Raises`, and `Example`.

## Generated README.md and AGENTS.md

Module and directory `README.md`/`AGENTS.md` files are generated signposts
(contents, public interface, module metadata, dependencies, validation
commands). Regenerate them with
`uv run python GEO-INFER-TEST/rewrite_readme_agents.py` and never edit them by
hand; CI runs `rewrite_readme_agents.py --check`. Conceptual guidance,
tutorials and integration notes belong in `GEO-INFER-INTRA/docs/` or a
module's `docs/` directory.

## SKILL.md

Each module's hand-written `SKILL.md` starts with YAML front matter (required
`name` and `description`; optional `prerequisites`) and must contain `## Instructions`,
`## Examples` and `## Guidelines` with an `### Integrations` subsection.
Validate with:

```bash
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal
```

## CHANGELOG.md

The root `CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/):

```markdown
# Changelog

## [0.2.0] - 2026-02-25
### Added
- Spatial statistics (Moran's I, Geary C)
### Fixed
- Placeholder implementations replaced with real logic
### Changed
- Updated H3 API to v4
```

## Cross-Reference Standards

- Link to related modules: `See [GEO-INFER-SPACE](../GEO-INFER-SPACE/README.md)`
- Link to specific files: `See [risk_engine.py](../GEO-INFER-RISK/src/geo_infer_risk/core/risk_engine.py)`
- Reference other agent rules: `See principles.md for logging standards`

## Language Guidelines

- Use precise, technical language over marketing terms
- Prefer "provides" over "provides comprehensive and sophisticated"
- Choose "implements" over "implements advanced and cutting-edge"
- Eliminate redundant adjectives that don't add technical value
- Focus on capabilities and functionality rather than superlatives

## API Documentation

For modules with REST APIs, maintain OpenAPI specs:

```yaml
# docs/api_schema.yaml
openapi: 3.0.0
info:
  title: GEO-INFER-MODULE API
  version: 0.2.0
paths:
  /api/v1/analyse:
    post:
      summary: Run analysis
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AnalysisRequest'
```

## Documentation Resources

- **Standards**: `GEO-INFER-INTRA/docs/DOCUMENTATION_STANDARDS.md`
- **Templates**: `GEO-INFER-INTRA/docs/templates/`
- **Module Index**: `GEO-INFER-INTRA/docs/modules/index.md`
