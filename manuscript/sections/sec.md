## GEO-INFER-SEC — Security and Privacy

GEO-INFER-SEC provides a comprehensive security and privacy framework for geospatial information systems with encryption, access control, and compliance (per its `README.md`). The package `geo_infer_sec` organizes `api/`, `core/`, `models/`, `config/`, and `utils/` plus a `cli.py`.

The public interface, verified from `__init__.py`, exports authentication, authorization, encryption and audit components. Authentication: `AuthenticationManager`, `UserCredentials`, and `TokenInfo` for identity and token lifecycle. Authorization: `GeospatialAccessManager` (with a general `AccessManager`), `Role`, and `SpatialPermission` — the latter making permissions spatially aware, a requirement specific to geospatial systems where access depends on region and layer. Encryption: `GeospatialEncryption` for protecting data at rest and in transit. Audit and threat posture: `AuditLogger`, `AuditEvent`, `AuditEventType`, `AuditEventSeverity`, `SecurityEvent`, and `ThreatLevel` for immutable activity records and classification, with `SecurityUtils` collecting common helpers.

The tests exercise authentication flows, permission boundaries, encryption round-trips, and audit trails. Security acceptance also requires fresh independent review; the test inventory is not a security certification.

Under the root README's Module Themes, SEC belongs to Governance, Risk & Domain together with METAGOV, NORMS, ORG, PEP, REQ, and RISK. Its role there is the protection layer: it enforces who may see and change geospatial state, and supplies the audit evidence on which the repository's accountability claims rest.
