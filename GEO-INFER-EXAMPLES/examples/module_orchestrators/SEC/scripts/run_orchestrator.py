#!/usr/bin/env python3
"""GEO-INFER-SEC module orchestrator.

Runs one documented end-to-end SEC operation on synthetic payloads: input
normalization of hostile strings, a PBKDF2 password hash/verify round trip,
AES-256 encrypt/decrypt round trip, HMAC token issue/validate (including a
tamper rejection), spatial anonymization, and an audit trail with a
compliance report. All work goes through the real ``geo_infer_sec`` public
API.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

import numpy as np

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    import pandas as pd

    from geo_infer_sec import AuditEventSeverity, AuditEventType, AuditLogger
    from geo_infer_sec import SecurityUtils
    from geo_infer_sec.utils.security_utils import SecurityConfig

    config = SecurityConfig(token_secret="synthetic-sec-demo-secret")
    utils = SecurityUtils(config)

    # 1. Input normalization on synthetic hostile payloads.
    payloads: dict[str, str] = {
        "sql_like": "'; DROP TABLE sensors;--",
        "xss_like": "<script>alert('x')</script>",
        "pipe_like": "cat report.txt | rm -rf /tmp",
    }
    normalized: dict[str, str] = {
        name: utils.strip_dangerous_chars(payload) for name, payload in payloads.items()
    }
    chars_removed = sum(
        len(p) - len(n) for p, n in zip(payloads.values(), normalized.values())
    )

    # 2. Password hashing round trip (PBKDF2).
    stored_hash, salt = utils.hash_password("synthetic-passphrase-42")
    correct = utils.verify_password("synthetic-passphrase-42", stored_hash, salt)
    wrong = utils.verify_password("synthetic-passphrase-43", stored_hash, salt)

    # 3. AES-256 encrypt/decrypt round trip.
    key = utils.generate_secure_key(32)
    secret_text = "synthetic site coordinates: 44.0N, -124.0W"
    encrypted, iv = utils.encrypt_data(secret_text, key)
    decrypted = utils.decrypt_data(encrypted, key, iv).decode("utf-8")
    roundtrip_ok = decrypted == secret_text

    # 4. HMAC token issue/validate, including tamper rejection.
    token = utils.generate_secure_token("field-agent-7", expiration_hours=1)
    token_user = utils.validate_token(token)
    tampered = token[:-2] + ("xx" if token[-2:] != "xx" else "yy")
    tampered_user = utils.validate_token(tampered)

    # 5. Spatial anonymization: precision reduction on synthetic coordinates.
    rng = np.random.default_rng(42)
    n_rows = 24
    spatial = pd.DataFrame(
        {
            "lat": 44.0 + rng.uniform(0.0, 0.5, n_rows),
            "lon": -124.0 + rng.uniform(0.0, 0.5, n_rows),
        }
    )
    anonymized = utils.anonymize_spatial_data(spatial, precision=2)
    max_shift = float(
        (anonymized["lat"] - spatial["lat"]).abs().max()
        + (anonymized["lon"] - spatial["lon"]).abs().max()
    )

    # 6. Audit trail and compliance report over synthetic events.
    with tempfile.TemporaryDirectory() as tmp_dir:
        audit = AuditLogger(
            log_file=Path(tmp_dir) / "synthetic_audit.log",
            enable_console=False,
            enable_file=True,
        )
        events: list[Any] = [
            audit.log_event(
                AuditEventType.DATA_ACCESS,
                user_id="field-agent-7",
                resource="site_polygon_01",
                action="read",
                result="success",
                severity=AuditEventSeverity.LOW,
            ),
            audit.log_event(
                AuditEventType.DATA_ACCESS,
                user_id="field-agent-7",
                resource="site_polygon_02",
                action="read",
                result="success",
                severity=AuditEventSeverity.LOW,
            ),
            audit.log_event(
                AuditEventType.AUTHORIZATION,
                user_id="field-agent-9",
                resource="site_polygon_01",
                action="write",
                result="denied",
                severity=AuditEventSeverity.HIGH,
            ),
            audit.log_event(
                AuditEventType.AUTHENTICATION,
                user_id="field-agent-9",
                action="login",
                result="failure",
                severity=AuditEventSeverity.MEDIUM,
            ),
        ]
        now = datetime.now(UTC).replace(tzinfo=None)
        compliance = audit.generate_compliance_report(
            now - timedelta(hours=1), now + timedelta(minutes=1)
        )
    high_severity_events = audit.get_events(severity=AuditEventSeverity.HIGH)

    return {
        "operation": "input_validation_and_audit",
        "payloads_normalized": {name: text for name, text in normalized.items()},
        "characters_removed_total": chars_removed,
        "password_verify_correct": correct,
        "password_verify_wrong_rejected": not wrong,
        "encrypt_decrypt_roundtrip_ok": roundtrip_ok,
        "ciphertext_bytes": len(encrypted),
        "token_validates_for_expected_user": token_user == "field-agent-7",
        "tampered_token_rejected": tampered_user is None,
        "anonymized_rows": int(len(anonymized)),
        "anonymization_max_abs_shift_deg": round(max_shift, 4),
        "audit_events_logged": len(events),
        "compliance_high_severity_events": len(high_severity_events),
        "compliance_critical_events": compliance["critical_events_count"],
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("SEC", _operation))
