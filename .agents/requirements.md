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
| Use `pip install` directly | Always use `uv pip install` |

## ALWAYS Do These Things

| Rule | How |
|------|-----|
| Implement complete, working functionality | No placeholders, stubs, or TODOs in production |
| Use structured logging | `logging.getLogger(__name__)` with appropriate levels |
| Follow PEP 8 + Black + isort | `black . && isort . && ruff check .` |
| Write comprehensive tests | ≥80% coverage, unit + integration |
| Type-hint all function signatures | Parameters, returns, class attributes |
| Validate input data | Pydantic models or explicit checks at boundaries |
| Handle optional dependencies gracefully | `try/except ImportError` with warning |
| Use `uv` for all package operations | `uv pip install`, `uv run python` |
| Update docs with code changes | README.md, docstrings, AGENTS.md |
| Use precise, technical language | "Show don't tell" |

## Package Management

### Correct Usage

```bash
# Install packages
uv pip install package-name
uv pip install -e ./GEO-INFER-MODULE
uv sync --all-packages --all-extras

# Run scripts
uv run python script.py
uv run pytest tests/

# In error messages
raise ImportError("Install with: uv pip install package-name")
```

### Incorrect Usage (NEVER)

```bash
# ❌ pip install package-name
# ❌ python -m pip install package-name
# ❌ conda install package-name
# ❌ python script.py  (use uv run python)
```

## Security Requirements

- Store secrets in environment variables, never in source code
- Use `os.environ.get("KEY")` with sensible defaults or explicit errors
- Validate and sanitise all external inputs (API payloads, file uploads)
- Monitor dependencies for CVEs (`uv pip audit` or safety checks)
- Follow the principle of least privilege for file/network access

## Licence Compliance

- All modules are licensed under **CC BY-NC-SA 4.0**
- Ensure all dependencies are compatible with this licence
- Include licence headers in new source files if required by deps
- Verify compliance in `pyproject.toml` metadata
