# Critical Requirements

## NEVER Do These Things

| Rule | Reason |
|------|--------|
| Create mock, stub, or placeholder implementations | Every function must have real logic |
| Hardcode credentials, API keys, or secrets | Use `os.environ.get()` or config files |
| Use bare `except:` or `except Exception:` without logging | Always catch specific exceptions |
| Use `print()` in library code | Use `logging.getLogger(__name__)` |
| Use `yaml.load()` without `Loader` | Use `yaml.safe_load()` |
| Break established module interfaces | Extend, don't replace |
| Duplicate functionality that exists in another module | Import and reuse |
| Ignore error conditions or fail silently | Log and raise or handle gracefully |
| Use excessive adjectives in documentation | Technical precision over marketing |
| Use H3 v3 API methods | Use H3 v4 exclusively |
| Use `pip install` directly | Use `uv sync` / `uv add --package` against the root workspace |
| Draw from the global `numpy.random` stream in library code | Take `seed`/`rng` and call the module's `utils/rng.py` `resolve_rng` |

## ALWAYS Do These Things

| Rule | How |
|------|-----|
| Implement complete, working functionality | No placeholders, stubs, or TODOs in production |
| Use structured logging | `logging.getLogger(__name__)` with appropriate levels |
| Lint and format with Ruff | `uv run --with 'ruff>=0.15.6,<0.16' ruff check .` and `... ruff format .` |
| Write tests for every change | Unit + integration; touched modules keep their floor in `GEO-INFER-TEST/coverage_baseline.json` |
| Type-hint all function signatures | Parameters, returns, class attributes; `list[int]`, `X \| None` |
| Validate input data | Pydantic models or explicit checks at boundaries |
| Handle optional dependencies gracefully | `try/except ImportError` with warning |
| Use `uv` for all package operations | `uv sync --all-packages --all-extras --all-groups`, `uv run python` |
| Update docs with code changes | README.md, docstrings, AGENTS.md |
| Use precise, technical language | "Show don't tell" |

## Package Management

### Correct Usage

```bash
# Synchronize the workspace (all packages, all extras) or one package
uv sync --all-packages --all-extras --all-groups
uv sync --package geo-infer-module

# Add a dependency to a module's pyproject.toml and refresh uv.lock
uv add --package geo-infer-module package-name

# Run scripts and tests
uv run python script.py
uv run python -m pytest GEO-INFER-MODULE/tests/

# In error messages for optional dependencies
raise ImportError("Install with: uv sync --package geo-infer-module --extra spatial")
```

### Incorrect Usage (NEVER)

```bash
# ❌ pip install package-name
# ❌ python -m pip install package-name
# ❌ pip install -r requirements.txt  (dependencies live in pyproject.toml + uv.lock)
# ❌ conda install package-name
# ❌ python script.py  (use uv run python)
```

## Security Requirements

- Store secrets in environment variables, never in source code
- Use `os.environ.get("KEY")` with sensible defaults or explicit errors
- Validate and sanitise all external inputs (API payloads, file uploads)
- Monitor dependencies for CVEs (`uv run --with pip-audit pip-audit`); CI scans history for secrets with gitleaks
- Follow the principle of least privilege for file/network access

## Licence Compliance

- All modules are licensed under **CC BY-NC-SA 4.0**
- Ensure all dependencies are compatible with this licence
- Include licence headers in new source files if required by deps
- Verify compliance in `pyproject.toml` metadata
