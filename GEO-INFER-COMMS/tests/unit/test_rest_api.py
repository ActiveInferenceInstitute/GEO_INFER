"""Regression tests for REST API error and configuration contracts."""

import asyncio
import inspect
import time
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from geo_infer_comms import GeospatialCommunicationSystem, MessageRequest
from geo_infer_comms.api.rest_api import CommunicationAPI


def _api() -> CommunicationAPI:
    system = GeospatialCommunicationSystem(config={"enable_persistence": False})
    return CommunicationAPI(system, enable_auth=False, enable_cors=False)


def test_cors_origins_default_is_not_mutable() -> None:
    parameter = inspect.signature(CommunicationAPI).parameters["cors_origins"]
    assert parameter.default is None


def test_invalid_message_content_preserves_http_400() -> None:
    api = _api()
    route = next(route for route in api.app.routes if route.path == "/messages")
    request = MessageRequest(content="   ", recipients=["recipient"])

    with pytest.raises(HTTPException) as error:
        asyncio.run(route.endpoint(request, BackgroundTasks(), None))

    assert error.value.status_code == 400
    assert error.value.detail == "Invalid message content"


class TestCredentialValidation:
    def _credentials(self, token: str):
        return SimpleNamespace(credentials=token)

    def test_valid_jwt_returns_subject(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import jwt as pyjwt

        monkeypatch.setenv(
            "COMMS_JWT_SECRET", "unit-test-secret-0123456789abcdef-0123456789abcdef"
        )
        api = _api()
        token = pyjwt.encode(
            {"sub": "alice", "exp": int(time.time()) + 60},
            "unit-test-secret-0123456789abcdef-0123456789abcdef",
            algorithm="HS256",
        )

        user_id = api._validate_credentials(self._credentials(token))

        assert user_id == "alice"

    def test_invalid_jwt_is_rejected_with_401(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jwt as pyjwt

        monkeypatch.setenv(
            "COMMS_JWT_SECRET", "correct-secret-0123456789abcdef-0123456789abcdef"
        )
        api = _api()
        # Signed with the wrong secret → decode fails → must be rejected
        bad_token = pyjwt.encode(
            {"sub": "attacker", "exp": int(time.time()) + 60},
            "wrong-secret-0123456789abcdef-0123456789abcdef",
            algorithm="HS256",
        )

        with pytest.raises(HTTPException) as error:
            api._validate_credentials(self._credentials(bad_token))

        assert error.value.status_code == 401

    def test_expired_jwt_is_rejected_with_401(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jwt as pyjwt

        monkeypatch.setenv(
            "COMMS_JWT_SECRET", "correct-secret-0123456789abcdef-0123456789abcdef"
        )
        api = _api()
        expired = pyjwt.encode(
            {"sub": "alice", "exp": int(time.time()) - 60},
            "correct-secret-0123456789abcdef-0123456789abcdef",
            algorithm="HS256",
        )

        with pytest.raises(HTTPException) as error:
            api._validate_credentials(self._credentials(expired))

        assert error.value.status_code == 401

    def test_hash_fallback_applies_without_jwt_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hashlib

        monkeypatch.delenv("COMMS_JWT_SECRET", raising=False)
        api = _api()

        user_id = api._validate_credentials(self._credentials("raw-opaque-token"))

        expected = "user_" + hashlib.sha256(b"raw-opaque-token").hexdigest()[:8]
        assert user_id == expected


class TestAuthenticatedHttpContract:
    """End-to-end REST authentication contract (#132 repo-wide consistency)."""

    SECRET = "http-contract-secret-0123456789abcdef-0123456789abcdef"

    def _client(self, monkeypatch: pytest.MonkeyPatch):
        from fastapi.testclient import TestClient

        monkeypatch.setenv("COMMS_JWT_SECRET", self.SECRET)
        system = GeospatialCommunicationSystem(config={"enable_persistence": False})
        api = CommunicationAPI(system, enable_auth=True, enable_cors=False)
        return api, TestClient(api.app, raise_server_exceptions=False)

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": "Basic dXNlcjpwYXNz"},
            {"Authorization": "Bearer short"},
            {"Authorization": "Bearer not-a-valid-jwt-token"},
        ],
        ids=["missing", "non-bearer", "too-short", "invalid-jwt"],
    )
    def test_rejected_credentials_are_a_401_bearer_challenge(
        self, monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
    ) -> None:
        _, client = self._client(monkeypatch)

        for method, path, kwargs in (
            ("GET", "/messages", {}),
            ("POST", "/messages", {"json": {"content": "hi", "recipients": ["r"]}}),
        ):
            response = client.request(method, path, headers=headers, **kwargs)

            assert response.status_code == 401, (method, path, response.text)
            assert response.headers.get("WWW-Authenticate") == "Bearer"

    def test_valid_bearer_token_reaches_the_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jwt as pyjwt

        _, client = self._client(monkeypatch)
        token = pyjwt.encode(
            {"sub": "alice", "exp": int(time.time()) + 60},
            self.SECRET,
            algorithm="HS256",
        )

        response = client.get("/messages", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200, response.text
        assert isinstance(response.json(), list)

    def test_missing_credentials_contract_is_owned_by_the_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api, _ = self._client(monkeypatch)

        # FastAPI < 0.122 answers a missing Bearer header with 403 when
        # auto_error is enabled; the package accepts fastapi>=0.100.0.
        assert api.security is not None
        assert api.security.auto_error is False
