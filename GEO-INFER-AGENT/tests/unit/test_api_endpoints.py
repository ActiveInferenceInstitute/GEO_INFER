#!/usr/bin/env python3

"""
Smoke tests for the FastAPI application in geo_infer_agent.api.agent_endpoints.

Exercises the real HTTP surface with starlette's TestClient: agent creation,
listing, lookup, state retrieval, message delivery, stop and delete.
"""

import pytest
from fastapi.testclient import TestClient

from geo_infer_agent.api.agent_endpoints import agent_registry, app


@pytest.fixture()
def client():
    """TestClient with a cleaned app registry so tests stay isolated."""
    assert not agent_registry.agents
    assert not agent_registry.agent_tasks
    with TestClient(app) as test_client:
        try:
            yield test_client
        finally:
            for agent_id in tuple(agent_registry.agents):
                response = test_client.delete(f"/agents/{agent_id}")
                assert response.status_code == 200, response.text
            assert not agent_registry.agent_tasks
            assert not agent_registry.running_agents


def _create_agent(client: TestClient, agent_id: str) -> None:
    response = client.post(
        "/agents",
        json={
            "agent_id": agent_id,
            "agent_type": "default",
            "config": {"max_runtime": 0},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    assert body["data"]["agent_id"] == agent_id


class TestAgentEndpoints:
    """HTTP-level smoke tests against the FastAPI app."""

    def test_create_list_and_get_agent(self, client: TestClient) -> None:
        _create_agent(client, "api-agent-1")

        listed = client.get("/agents")
        assert listed.status_code == 200
        assert any(a["agent_id"] == "api-agent-1" for a in listed.json())

        fetched = client.get("/agents/api-agent-1")
        assert fetched.status_code == 200
        assert fetched.json()["success"] is True

    def test_get_unknown_agent_returns_404(self, client: TestClient) -> None:
        response = client.get("/agents/does-not-exist")
        assert response.status_code == 404

    def test_create_duplicate_agent_returns_400(self, client: TestClient) -> None:
        _create_agent(client, "api-agent-dup")
        response = client.post(
            "/agents",
            json={"agent_id": "api-agent-dup", "agent_type": "default", "config": {}},
        )
        assert response.status_code == 400

    def test_get_agent_state(self, client: TestClient) -> None:
        _create_agent(client, "api-agent-state")
        response = client.get("/agents/api-agent-state/state")
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert "state" in body["data"]
        assert "beliefs" in body["data"]["state"]

    def test_message_delivered_between_registered_agents(
        self, client: TestClient
    ) -> None:
        _create_agent(client, "api-sender")
        _create_agent(client, "api-receiver")

        response = client.post(
            "/agents/api-sender/message",
            json={"to_agent_id": "api-receiver", "content": {"cmd": "ping"}},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

        receiver = agent_registry.get_agent("api-receiver")
        assert not receiver.message_queue.empty()
        delivered = receiver.message_queue.get_nowait()
        assert delivered["content"] == {"cmd": "ping"}
        assert delivered["from"] == "api-sender"

    def test_stop_then_delete_agent(self, client: TestClient) -> None:
        _create_agent(client, "api-agent-cycle")

        stopped = client.post("/agents/api-agent-cycle/stop")
        assert stopped.status_code == 200

        deleted = client.delete("/agents/api-agent-cycle")
        assert deleted.status_code == 200
        assert deleted.json()["success"] is True

        assert client.get("/agents/api-agent-cycle").status_code == 404

    def test_cors_preflight_rejects_unknown_origin(self, client: TestClient) -> None:
        """Attacker origins never get wildcard+credentialed CORS approval."""
        response = client.options(
            "/agents",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        acao = response.headers.get("access-control-allow-origin")
        acac = response.headers.get("access-control-allow-credentials")
        assert not (acao == "*" and acac == "true")
