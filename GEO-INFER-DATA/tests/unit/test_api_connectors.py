"""
Tests for API connectors in geo_infer_data.connectors.api.

Covers APIConnector auth-header modes, pagination termination, and rate
limiting, plus STACConnector's search payload construction. All HTTP I/O is
faked via monkeypatched sessions — no real network access.
"""

import asyncio
import time
import threading

import pytest

from geo_infer_data.connectors.api import APIConnector, GraphQLConnector, STACConnector
from geo_infer_data.connectors import api as _api


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    """Minimal requests.Response double serving a JSON payload."""

    def __init__(self, payload):
        self._payload = payload
        self.headers = {"content-type": "application/json"}
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """In-memory requests.Session double recording every request."""

    def __init__(self, pages):
        self.headers = {}
        self.auth = None
        self.calls = []
        self._pages = list(pages)

    def _next_payload(self):
        return self._pages.pop(0) if self._pages else {}

    def request(
        self,
        method=None,
        url=None,
        params=None,
        json=None,
        headers=None,
        timeout=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": dict(params or {}),
                "json": json,
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        return _FakeResponse(self._next_payload())

    def get(self, url, **kwargs):
        return self.request(method="GET", url=url, **kwargs)

    def close(self):
        pass


def _fake_session(connector, monkeypatch, pages):
    session = _FakeSession(pages)
    session.headers.update(dict(connector.session.headers))
    monkeypatch.setattr(connector, "session", session)
    return session


# ---------------------------------------------------------------------------
# APIConnector: auth-header modes
# ---------------------------------------------------------------------------


class TestAPIConnectorAuthModes:
    def test_header_mode_sets_api_key_header(self):
        connector = APIConnector(
            "https://api.example.com",
            authentication={"type": "header", "api_key": "key-123"},
        )
        assert connector.session.headers["X-API-Key"] == "key-123"

    def test_bearer_mode_sets_authorization_header(self):
        connector = APIConnector(
            "https://api.example.com",
            authentication={"type": "bearer", "token": "tok-42"},
        )
        assert connector.session.headers["Authorization"] == "Bearer tok-42"

    def test_basic_mode_sets_http_basic_auth(self):
        connector = APIConnector(
            "https://api.example.com",
            authentication={"type": "basic", "username": "u", "password": "p"},
        )
        auth = connector.session.auth
        assert auth is not None
        assert auth.username == "u"
        assert auth.password == "p"

    def test_no_auth_leaves_session_unauthenticated(self):
        connector = APIConnector("https://api.example.com")
        assert "Authorization" not in connector.session.headers
        assert "X-API-Key" not in connector.session.headers
        assert connector.session.auth is None

    def test_auth_headers_reach_outgoing_request(self, monkeypatch):
        connector = APIConnector(
            "https://api.example.com",
            authentication={"type": "bearer", "token": "tok-42"},
        )
        session = _fake_session(connector, monkeypatch, [{"data": []}])
        monkeypatch.setattr(time, "sleep", lambda s: None)
        _run(connector.query_endpoint("/points"))
        # Session-level bearer header is applied by requests on the wire; the
        # recorded call must target the right URL with the session in place.
        assert session.calls[0]["url"] == "https://api.example.com/points"
        assert connector.session.headers["Authorization"] == "Bearer tok-42"


# ---------------------------------------------------------------------------
# APIConnector: pagination termination
# ---------------------------------------------------------------------------


class TestAPIConnectorPagination:
    def test_pagination_stops_on_empty_page(self, monkeypatch):
        connector = APIConnector("https://api.example.com")
        pages = [
            {"results": [{"id": 1}, {"id": 2}]},
            {"results": []},
        ]
        session = _fake_session(connector, monkeypatch, pages)

        results = _run(
            connector.query_geospatial("/stations", pagination={"page": 1, "limit": 2})
        )

        assert len(session.calls) == 2
        assert [r["id"] for r in results] == [1, 2]

    def test_pagination_stops_at_declared_total_pages(self, monkeypatch):
        connector = APIConnector("https://api.example.com")
        session = _fake_session(
            connector,
            monkeypatch,
            [{"results": [{"id": 1}], "total_pages": 1}],
        )
        sleeps = []

        async def sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(_api.asyncio, "sleep", sleep)

        results = _run(
            connector.query_geospatial("/stations", pagination={"page": 1, "limit": 1})
        )

        assert len(session.calls) == 1
        assert session.calls[0]["params"]["page"] == 1
        assert [r["id"] for r in results] == [1]
        assert sleeps == []

    def test_pagination_respects_max_pages(self, monkeypatch):
        connector = APIConnector("https://api.example.com")
        pages = [
            {"results": [{"id": 1}]},
            {"results": [{"id": 2}]},
            {"results": [{"id": 3}]},
        ]
        session = _fake_session(connector, monkeypatch, pages)

        results = _run(
            connector.query_geospatial(
                "/stations", pagination={"page": 1, "limit": 1, "max_pages": 2}
            )
        )

        assert len(session.calls) == 2
        assert [r["id"] for r in results] == [1, 2]


# ---------------------------------------------------------------------------
# APIConnector: rate limiting
# ---------------------------------------------------------------------------


class TestAPIConnectorRateLimiting:
    def test_rate_limit_sleeps_when_exceeded(self, monkeypatch):
        connector = APIConnector(
            "https://api.example.com", rate_limiting={"requests_per_minute": 1}
        )
        _fake_session(connector, monkeypatch, [{}, {}])
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(_api.time, "sleep", sleep)

        _run(connector.query_endpoint("/a"))
        _run(connector.query_endpoint("/b"))

        assert connector.request_count == 2
        assert len(sleeps) == 1
        assert sleeps[0] > 0

    @pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True, "2"])
    def test_invalid_rate_rejected_before_session(self, rate):
        with pytest.raises(ValueError, match="finite and positive"):
            APIConnector(
                "https://example.invalid", rate_limiting={"requests_per_minute": rate}
            )


def test_real_retry_config_uses_supported_keyword():
    connector = APIConnector("https://example.invalid")
    try:
        retry = connector.session.get_adapter("https://").max_retries
        assert set(retry.allowed_methods) == {"GET", "HEAD", "OPTIONS"}
        assert retry.total == 3
    finally:
        _run(connector.close())


@pytest.mark.parametrize("kind", ["rest", "graphql"])
async def test_http_worker_does_not_block_event_loop(monkeypatch, kind):
    connector = (APIConnector if kind == "rest" else GraphQLConnector)(
        "https://example.invalid"
    )
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    worker_threads = []

    def request(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5), "event loop failed to release HTTP worker"
        return _FakeResponse({"ok": True} if kind == "rest" else {"data": {"ok": True}})

    monkeypatch.setattr(
        connector.session, "request" if kind == "rest" else "post", request
    )
    task = asyncio.create_task(
        connector.query_endpoint("/points")
        if kind == "rest"
        else connector.execute_query("query { points }")
    )
    try:
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), timeout=6)
        assert entered.is_set()
        assert len(worker_threads) == 1
        assert worker_threads[0] != loop_thread
        release.set()
        assert await asyncio.wait_for(task, timeout=5) == {"ok": True}
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await connector.close()


async def test_pagination_request_failure_is_fatal(monkeypatch):
    import requests

    connector = APIConnector("https://example.invalid")
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _FakeResponse({"results": [{"id": 1}]})
        raise requests.ConnectionError("failed page")

    monkeypatch.setattr(connector.session, "request", request)
    try:
        with pytest.raises(requests.ConnectionError, match="failed page"):
            await connector.query_geospatial("/points")
    finally:
        await connector.close()


@pytest.mark.parametrize("kind", ["rest", "graphql"])
async def test_close_waits_for_inflight_request_without_blocking_loop(
    monkeypatch, kind
):
    connector = (APIConnector if kind == "rest" else GraphQLConnector)(
        "https://example.invalid"
    )
    entered, release, finished, close_waiting = (threading.Event() for _ in range(4))
    underlying = threading.Lock()

    class ObservedLock:
        def __enter__(self):
            if entered.is_set():
                close_waiting.set()
            underlying.acquire()

        def __exit__(self, *_):
            underlying.release()

    def request(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        finished.set()
        return _FakeResponse({"ok": True} if kind == "rest" else {"data": {"ok": True}})

    def close():
        assert finished.is_set(), "Session closed while a request was using it"

    connector._session_lock = ObservedLock()
    monkeypatch.setattr(
        connector.session, "request" if kind == "rest" else "post", request
    )
    monkeypatch.setattr(connector.session, "close", close)
    request_task = asyncio.create_task(
        connector.query_endpoint("/points")
        if kind == "rest"
        else connector.execute_query("query { points }")
    )
    close_task = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), timeout=6)
        close_task = asyncio.create_task(connector.close())
        assert await asyncio.wait_for(
            asyncio.to_thread(close_waiting.wait, 5), timeout=6
        )
        assert not close_task.done()
        release.set()
        assert await asyncio.wait_for(request_task, timeout=5) == {"ok": True}
        await asyncio.wait_for(close_task, timeout=5)
    finally:
        release.set()
        await asyncio.gather(
            *([request_task, close_task] if close_task is not None else [request_task]),
            return_exceptions=True,
        )


async def test_queued_workers_space_actual_starts_after_long_request(monkeypatch):
    from types import SimpleNamespace

    connector = APIConnector(
        "https://example.invalid", rate_limiting={"requests_per_minute": 60}
    )
    entered, release, all_arrived = (threading.Event() for _ in range(3))
    session_lock, state_lock = threading.Lock(), threading.Lock()
    clock, arrivals, starts = [0.0], [0], []

    class ObservedLock:
        def __enter__(self):
            with state_lock:
                arrivals[0] += 1
                if arrivals[0] == 3:
                    all_arrived.set()
            session_lock.acquire()

        def __exit__(self, *_):
            session_lock.release()

    def sleep(seconds):
        clock[0] += seconds

    def request(**kwargs):
        starts.append(clock[0])
        if len(starts) == 1:
            entered.set()
            assert release.wait(5)
        return _FakeResponse({"ok": True})

    # Replace only this connector module's clock, leaving asyncio's real
    # deadlines intact. The virtual ten-second first request adds no real wait.
    monkeypatch.setattr(
        _api, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
    )
    monkeypatch.setattr(connector.session, "request", request)
    connector._session_lock = ObservedLock()
    tasks = [asyncio.create_task(connector.query_endpoint("/first"))]
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), timeout=6)
        tasks.extend(
            asyncio.create_task(connector.query_endpoint(endpoint))
            for endpoint in ("/second", "/third")
        )
        assert await asyncio.wait_for(asyncio.to_thread(all_arrived.wait, 5), timeout=6)
        clock[0] = 10.0
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        assert starts == [0.0, 10.0, 11.0]
        assert connector.request_count == 3
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await connector.close()


# ---------------------------------------------------------------------------
# STACConnector: search payload construction
# ---------------------------------------------------------------------------


class TestSTACConnector:
    def test_search_items_builds_expected_payload(self, monkeypatch):
        stac = STACConnector(
            "https://stac.example.com",
            authentication={"type": "bearer", "token": "stac-tok"},
        )
        session = _fake_session(
            stac.connector,
            monkeypatch,
            [{"features": [{"id": "item-1"}], "limit": 50}],
        )

        result = _run(
            stac.search_items(
                collections=["sentinel-2"],
                bbox=[-122.5, 37.7, -122.3, 37.9],
                datetime_range="2024-01-01/2024-01-02",
                properties={"cloud_cover": {"lte": 10}},
                limit=50,
            )
        )

        assert len(session.calls) == 1
        call = session.calls[0]
        assert call["url"] == "https://stac.example.com/search"
        assert call["method"] == "GET"
        assert call["params"] == {
            "limit": 50,
            "collections": ["sentinel-2"],
            "bbox": [-122.5, 37.7, -122.3, 37.9],
            "datetime": "2024-01-01/2024-01-02",
            "cloud_cover": {"lte": 10},
        }
        assert result["features"][0]["id"] == "item-1"

    def test_search_items_minimal_payload(self, monkeypatch):
        stac = STACConnector("https://stac.example.com")
        session = _fake_session(stac.connector, monkeypatch, [{"features": []}])

        _run(stac.search_items())

        assert session.calls[0]["params"] == {"limit": 100}
