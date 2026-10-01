# Realms API Testing Guide

## Overview

The `realms_api_probe.py` script tests the documented Realms API endpoints. It
validates responses against `realm_schema.json` and writes a report.

## Installation

The script needs `requests` and `jsonschema`, both installed by the workspace
sync; there is no separate requirements file.

```bash
# From the repository root
uv sync --all-packages --all-extras --all-groups
```

Run the script from this directory so the default `realm_schema.json` path
resolves.

## Usage

### Basic Usage

```bash
cd GEO-INFER-INTRA/docs/realms

# Run all tests with default parameters
uv run python realms_api_probe.py

# Run with a custom schema file
uv run python realms_api_probe.py --schema /path/to/realm_schema.json

# Quick test mode (fewer test cases)
uv run python realms_api_probe.py --quick
```

### Options

```bash
# Custom search terms
uv run python realms_api_probe.py --search-terms "Forest" "Ocean" "Park"

# Custom realm IDs to test
uv run python realms_api_probe.py --realm-ids 2188 6472 8155

# Custom timeout (seconds, default 30)
uv run python realms_api_probe.py --timeout 60

# Combined options
uv run python realms_api_probe.py --quick --search-terms "Avana" --realm-ids 2188
```

## What Gets Tested

1. **Search Realms by Name**
   - Endpoint: `GET https://api.guardiansofearth.io/realms`
   - Tests: multiple search terms with various parameters
   - Validates: response structure, required fields, data types
2. **Get All Realms**
   - Endpoint: `GET https://portal.biosmart.life/api/v1/contest/109/regions.json`
   - Tests: pagination, sorting by ID and bioscore
   - Validates: schema compliance, data consistency
3. **Get Realm by ID**
   - Endpoint: `GET https://portal.biosmart.life/api/v1/region/{id}`
   - Tests: multiple realm IDs (extracted from previous tests)
   - Validates: schema compliance, ID matching
4. **Error Cases**
   - Tests: invalid IDs, malformed parameters, edge cases
   - Validates: error handling and status codes

## Output

Each run writes to `outputs/test_run_YYYYMMDD_HHMMSS/`:

1. `test_execution.log`: progress and per-request log
2. `test_results.json`: per-test results and summary

The process exits with status 0 when every test passes and 1 otherwise.

## Schema Validation

Responses are validated against `realm_schema.json`:

- data types for all fields
- required fields present
- structure of arrays and objects
- every schema violation is reported

## Error Handling

- **Network issues**: timeouts, connection errors
- **HTTP errors**: 4xx and 5xx status codes
- **Data issues**: invalid JSON, schema violations
- **Rate limiting**: fixed delays between requests

## Troubleshooting

1. **Schema file not found**: run from `GEO-INFER-INTRA/docs/realms/` or pass
   `--schema`.
2. **Network timeouts**: increase `--timeout`.
3. **API rate limiting**: reduce scope with `--quick`.
4. **Authentication errors**: no authentication is documented for these
   endpoints; if one becomes required, add the key or token to the request
   headers in `RealmsAPITester`.

## Known Limits

- Base URLs are hardcoded from the published API documentation.
- Only documented endpoints are tested.
- Rate limits are not documented; the script uses fixed delays.
