"""Unit tests for WebSocket authentication fallback semantics (M3-04).

Covers the hard-reject contract: when ``COMMS_JWT_SECRET`` is configured but
PyJWT is unavailable, authentication must fail closed instead of falling back
to hash-derived identity (which would let any token string authenticate).
"""

import asyncio
import hashlib
import importlib.util
import json
import sys
from typing import Any

import jwt

from geo_infer_comms.api.websocket_api import WebSocketConnection


class FakeWebSocket:
    """Minimal async websocket double recording sent frames."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)


class FakeManager:
    """Minimal manager double; authentication does not touch the manager."""

    def __init__(self) -> None:
        self.subscriptions: dict[str, Any] = {}


def _connection() -> WebSocketConnection:
    return WebSocketConnection("conn-1", FakeWebSocket(), FakeManager())


def _last_message(conn: WebSocketConnection) -> dict[str, Any]:
    assert conn.websocket.sent, "expected an outbound frame"
    return json.loads(conn.websocket.sent[-1])


def test_pyjwt_is_importable_in_declared_environments():
    """pyjwt is a declared COMMS dependency, so it resolves in a synced env."""
    assert importlib.util.find_spec("jwt") is not None


def test_rejects_when_secret_configured_but_pyjwt_absent(monkeypatch):
    """Secret configured + PyJWT missing must reject, not authenticate."""
    monkeypatch.setenv(
        "COMMS_JWT_SECRET", "unit-test-secret-0123456789abcdef-0123456789abcdef"
    )
    monkeypatch.setitem(sys.modules, "jwt", None)

    conn = _connection()
    asyncio.run(conn._handle_authentication({"token": "forged-token"}))

    assert conn.authenticated is False
    assert conn.user_id is None
    message = _last_message(conn)
    assert message["type"] == "error"
    assert "Invalid authentication token" in message["message"]


def test_hash_identity_fallback_when_no_secret_configured(monkeypatch):
    """Without a configured secret the documented hash identity still applies."""
    monkeypatch.delenv("COMMS_JWT_SECRET", raising=False)
    monkeypatch.setitem(sys.modules, "jwt", None)

    conn = _connection()
    asyncio.run(conn._handle_authentication({"token": "some-token"}))

    expected = "user_" + hashlib.sha256(b"some-token").hexdigest()[:8]
    assert conn.authenticated is True
    assert conn.user_id == expected


def test_accepts_valid_jwt_when_secret_configured(monkeypatch):
    """A correctly signed HS256 JWT authenticates with its ``sub`` claim."""
    secret = "unit-test-secret-0123456789abcdef-0123456789abcdef"
    monkeypatch.setenv("COMMS_JWT_SECRET", secret)
    token = jwt.encode({"sub": "alice"}, secret, algorithm="HS256")

    conn = _connection()
    asyncio.run(conn._handle_authentication({"token": token}))

    assert conn.authenticated is True
    assert conn.user_id == "alice"


def test_rejects_tampered_jwt_when_secret_configured(monkeypatch):
    """A token signed with the wrong secret is rejected."""
    monkeypatch.setenv(
        "COMMS_JWT_SECRET", "unit-test-secret-0123456789abcdef-0123456789abcdef"
    )
    token = jwt.encode(
        {"sub": "mallory"},
        "other-secret-0123456789abcdef-0123456789abcdef",
        algorithm="HS256",
    )

    conn = _connection()
    asyncio.run(conn._handle_authentication({"token": token}))

    assert conn.authenticated is False
    assert conn.user_id is None
    message = _last_message(conn)
    assert message["type"] == "error"
