"""Tests for the Flask security API surface (security_api)."""

import pytest
from geo_infer_sec.api.security_api import init_security_api
from geo_infer_sec.core.access_control import Role


@pytest.fixture
def client():
    from flask import Flask

    app = Flask(__name__)
    init_security_api(
        app,
        secret_key="test-secret",
        enable_anonymization=False,
        enable_compliance=False,
    )
    app.config["TESTING"] = True
    return app.test_client()


def test_token_requires_roles(client):
    response = client.post("/api/security/token", json={"user_id": "stranger"})
    assert response.status_code == 403


def test_token_issue_and_use(client):
    from geo_infer_sec.api.security_api import _SECURITY_STATE_KEY

    # The test client does not expose the app directly; re-initialize a known app.
    from flask import Flask

    app = Flask(__name__)
    init_security_api(app, secret_key="test-secret", enable_anonymization=False)
    app.extensions[_SECURITY_STATE_KEY]["access_manager"].add_role(Role(name="viewer"))
    app.extensions[_SECURITY_STATE_KEY]["access_manager"].assign_role_to_user(
        "u1", "viewer"
    )
    client = app.test_client()

    response = client.post("/api/security/token", json={"user_id": "u1"})
    assert response.status_code == 200
    token = response.get_json()["token"]

    anonymized = client.post(
        "/api/security/anonymize",
        headers={"Authorization": f"Bearer {token}"},
        json={"features": []},
    )
    assert anonymized.status_code == 503
    assert "not enabled" in anonymized.get_json()["error"]


def test_uninitialized_app_returns_503():
    from flask import Flask

    from geo_infer_sec.api import security_api as sa

    app = Flask(__name__)
    app.register_blueprint(sa.security_api, url_prefix="/api/security")
    app.config["TESTING"] = True
    response = app.test_client().post("/api/security/token", json={"user_id": "x"})
    assert response.status_code == 503
    assert "not initialized" in response.get_json()["error"]


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer not-a-valid-token"},
    ],
    ids=["missing", "non-bearer", "invalid-token"],
)
def test_rejected_credentials_are_a_401_bearer_challenge(client, headers):
    response = client.get("/api/security/roles", headers=headers)

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"
