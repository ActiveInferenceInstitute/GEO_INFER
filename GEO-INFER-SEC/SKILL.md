---
name: geo-infer-sec
description: Security and threat detection for geospatial systems. Use when implementing spatial access control, authentication and TOTP-based MFA, geospatial data anonymization, security auditing, or encrypting spatial data.
prerequisites:
  required: []
  recommended: []
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-SEC

## Instructions

### Core Capabilities

- **Access control**: Role-based and spatial-boundary-based authorization
  (RBAC + SBAC) via `GeospatialAccessManager` (JWT tokens, point-in-permission
  checks, GeoDataFrame filtering).
- **Authentication**: JWT token lifecycle and RFC 6238 TOTP multi-factor
  authentication via `AuthenticationManager` (`enable_mfa` with a base32
  secret; authentication is denied when MFA is enabled and no valid code is
  supplied).
- **Threat detection**: `DigitalSecurityManager` (digital), `CognitiveSecurityManager`
  (behavioral), `PhysicalSecurityManager`, and `IntegratedSecurityManager`
  (cross-domain correlation). Threat indicators are SIMULATION-ONLY, loaded
  from a configurable YAML file — see `DigitalSecurityManager` docstring.
- **Audit logging**: Security event tracking with spatial context via
  `AuditLogger` (`log_authentication`, `log_authorization`, `log_data_access`,
  `generate_compliance_report`).
- **Data protection**: Spatial data anonymization and k-anonymity via
  `GeospatialAnonymizer` (`location_perturbation`, `spatial_k_anonymity`,
  `geographic_masking`); AES/Fernet encryption of text, JSON, coordinates, and
  GeoDataFrames via `GeospatialEncryption`.
- **Integrity**: Data integrity verification, tampering detection.

### Key Imports

```python
from geo_infer_sec.core.access_control import (
    GeospatialAccessManager,
    Role,
    SpatialPermission,
)
from geo_infer_sec.core.anonymization import GeospatialAnonymizer
from geo_infer_sec.core.authentication import AuthenticationManager
from geo_infer_sec.core.encryption import GeospatialEncryption
from geo_infer_sec.core.audit import AuditLogger
from geo_infer_sec.core.integrated_security import IntegratedSecurityManager
```

## Examples

### Spatially-scoped access control

```python
mgr = GeospatialAccessManager(secret_key="demo-secret")
mgr.add_role(
    Role(
        name="analyst",
        permissions=[SpatialPermission(name="view", attributes=["value"])],
    )
)
mgr.assign_role_to_user("alice", "analyst")

payload = mgr.validate_token(mgr.generate_token("alice"))
print(payload["user_id"])                       # alice
print(mgr.can_access_location("alice", 40.75, -73.98))  # True / False
```

### Anonymizing point data (EPSG:4326)

```python
anon = GeospatialAnonymizer(seed=42)
perturbed = anon.location_perturbation(gdf, epsilon=100.0)  # meters, seeded
grouped = anon.spatial_k_anonymity(gdf, k=5, h3_resolution=8)
```
```python
import base64

from geo_infer_sec.core.authentication import AuthenticationManager, generate_totp

auth = AuthenticationManager(secret_key="signing-secret")
auth.register_user("alice", "correct-horse-battery")
secret = base64.b32encode(b"0123456789abcdef").decode()
auth.enable_mfa("alice", secret)

auth.authenticate("alice", "correct-horse-battery")                 # denied: no code
code = generate_totp(secret)                                # RFC 6238 code
auth.authenticate("alice", "correct-horse-battery", mfa_code=code)  # TokenInfo
```

### CLI

```bash
geo-infer-sec --help   # anonymize / encrypt / decrypt / check-compliance /
                       # audit / risk-assessment / generate-token
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog SEC` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- `location_perturbation` assumes EPSG:4326 input and converts meters to
  degrees with a flat 111,000 m/deg factor; it raises on other CRS. Accuracy
  degrades away from the equator.
- Digital/cognitive/physical threat intelligence and background monitoring
  loops are simulation-only defaults; wire real feeds via the
  `threat_indicators_file` config before treating outputs as actionable.

### Integrations

- Integrates with API for endpoint security (`geo_infer_sec.api.security_api`,
  Flask blueprint; requires `init_security_api(app, secret_key)` before use).
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SEC`
