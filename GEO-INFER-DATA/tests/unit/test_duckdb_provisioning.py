"""Signed extension provisioning contracts at the real HTTP/engine boundary."""

import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil
import pytest

import geo_infer_data.utils.duckdb_spatial as backend
from geo_infer_data.utils import _duckdb_extension_worker as worker


@pytest.fixture
def isolated_engine(tmp_path, monkeypatch):
    """Keep every real DuckDB installation in the fixture's own directory."""
    import duckdb

    directory = tmp_path / "extensions"
    real_connect = duckdb.connect

    def connect(*args, config=None, **kwargs):
        config = dict(config or {})
        config["extension_directory"] = str(directory)
        return real_connect(*args, config=config, **kwargs)

    monkeypatch.setattr(duckdb, "connect", connect)
    return directory


@pytest.fixture
def http_boundary(tmp_path, monkeypatch):
    """Inject only HTTP opening in an actual isolated worker subprocess."""
    routes = {}
    requested = []
    stopped = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requested.append(self.path)
            mode, payload = routes[self.path]
            try:
                if mode == "headers":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Drip: ")
                    self.wfile.flush()
                    while not stopped.wait(0.04):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                    return
                if mode == "error":
                    self.send_error(503, "independent network failure")
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if mode == "body":
                    for offset in range(3):
                        self.wfile.write(payload[offset : offset + 1])
                        self.wfile.flush()
                        if stopped.wait(0.12):
                            return
                    stopped.wait(5)
                else:
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05))
    thread.start()
    harness = tmp_path / "http_worker.py"
    pid_file = tmp_path / "worker-pids.jsonl"
    gzip_marker = tmp_path / "gzip-started"
    harness.write_text(
        f"""
import importlib.util,json,os,sys,urllib.request
from pathlib import Path
spec=importlib.util.spec_from_file_location('worker', {str(Path(worker.__file__).resolve())!r})
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
with Path({str(pid_file)!r}).open('a') as f:f.write(json.dumps(os.getpid())+'\\n')
url=sys.argv[1]
def local_http_boundary(official_url,timeout):
    response=urllib.request.urlopen('http://127.0.0.1:{server.server_port}/'+official_url.rsplit('/',1)[-1],timeout=timeout)
    response.geturl=lambda:official_url
    return response
worker.open_extension=local_http_boundary
original_gzip=worker.gzip.open
def observed_gzip(*args,**kwargs):
    Path({str(gzip_marker)!r}).write_text('started')
    return original_gzip(*args,**kwargs)
worker.gzip.open=observed_gzip
worker.main()
"""
    )
    real_run = subprocess.run

    def run(command, **kwargs):
        assert command[:2] == [sys.executable, "-I"]
        assert Path(command[2]) == Path(worker.__file__)
        return real_run([*command[:2], str(harness), *command[3:]], **kwargs)

    monkeypatch.setattr(backend.subprocess, "run", run)
    try:
        yield routes, requested, pid_file, gzip_marker
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
        if pid_file.exists():
            for pid in map(json.loads, pid_file.read_text().splitlines()):
                assert not psutil.pid_exists(pid), (
                    "owned download worker was not reaped"
                )


def route(routes, name, payload, mode="normal"):
    routes[f"/{name}.duckdb_extension.gz"] = (mode, payload)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_invalid_deadline_rejected_before_connect(value, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid deadline opened an engine")

    monkeypatch.setattr(backend._DUCKDB, "connect", forbidden)
    with pytest.raises(ValueError, match="finite and positive"):
        backend.provision_spatial_extension(download_timeout=value)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_budget_rejected_before_network(value, monkeypatch):
    with pytest.raises(ValueError, match="positive integer"):
        backend.provision_spatial_extension(max_extension_bytes=value)


@pytest.mark.parametrize(
    "url",
    [
        "http://extensions.duckdb.org/a",
        "https://evil.example/a",
        "https://extensions.duckdb.org.evil.example/a",
        "https://user@extensions.duckdb.org/a",
        "https://extensions.duckdb.org:8443/a",
    ],
)
def test_origin_and_redirect_rejected_before_open(url, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("foreign origin opened")

    monkeypatch.setattr(worker.urllib.request, "build_opener", forbidden)
    with pytest.raises(ValueError, match="official HTTPS"):
        worker.open_extension(url, 1)
    req = worker.urllib.request.Request("https://extensions.duckdb.org/original")
    with pytest.raises(ValueError, match="official HTTPS"):
        worker.OfficialExtensionRedirects().redirect_request(
            req, None, 302, "", {}, url
        )


def test_open_identifies_product_and_keeps_default_tls(monkeypatch):
    captured = {}

    class Opener:
        def open(self, request, timeout):
            captured.update(request=request, timeout=timeout)
            return object()

    def opener(*handlers):
        assert len(handlers) == 1
        assert isinstance(handlers[0], worker.OfficialExtensionRedirects)
        return Opener()

    monkeypatch.setattr(worker.urllib.request, "build_opener", opener)
    worker.open_extension("https://extensions.duckdb.org/a", 2.5)
    assert captured["request"].get_header("User-agent") == "GEO-INFER-DATA/0.4.0"
    assert captured["timeout"] == 2.5


@pytest.mark.parametrize("mode", ["headers", "body"])
def test_actual_stalled_transport_has_total_deadline(http_boundary, tmp_path, mode):
    routes, requested, _, _ = http_boundary
    route(routes, "httpfs", gzip.compress(b"x" * 100), mode=mode)
    started = time.monotonic()
    with pytest.raises(backend.DuckDBSpatialError, match="deadline exceeded"):
        backend._download_extension(
            "https://extensions.duckdb.org/v1.5.5/linux_amd64/httpfs.duckdb_extension.gz",
            tmp_path / "extension",
            deadline=started + 0.8,
            max_bytes=10_000,
        )
    assert 0.7 < time.monotonic() - started < 2
    assert requested == ["/httpfs.duckdb_extension.gz"]


def test_actual_gzip_header_parser_has_total_deadline(http_boundary, tmp_path):
    routes, _, _, gzip_marker = http_boundary
    archive = gzip.compress(b"valid")
    # Valid gzip FNAME header parsing reads each byte separately. This long
    # filename exercises real decompression work, not a substituted sleep.
    payload = (
        archive[:3] + b"\x08" + archive[4:10] + b"x" * 32_000_000 + b"\0" + archive[10:]
    )
    route(routes, "httpfs", payload)
    started = time.monotonic()
    with pytest.raises(backend.DuckDBSpatialError, match="deadline exceeded"):
        backend._download_extension(
            "https://extensions.duckdb.org/v1.5.5/linux_amd64/httpfs.duckdb_extension.gz",
            tmp_path / "extension",
            deadline=started + 0.8,
            max_bytes=40_000_000,
        )
    assert time.monotonic() - started < 2
    assert gzip_marker.exists(), "fixture never entered the real gzip parser"


@pytest.mark.parametrize(
    "first,second,budget,diagnostic",
    [
        (b"not-gzip", gzip.compress(b"x"), 10_000, "Not a gzipped file"),
        (gzip.compress(b""), gzip.compress(b"x"), 10_000, "no extension bytes"),
        (b"x" * 101, gzip.compress(b"x"), 100, "archive exceeds"),
        (
            gzip.compress(b"x" * 101),
            gzip.compress(b"x"),
            100,
            "decompressed byte budget",
        ),
        (gzip.compress(b"httpfs"), b"invalid-spatial", 10_000, "Not a gzipped file"),
    ],
)
def test_incomplete_archives_never_install(
    http_boundary, isolated_engine, first, second, budget, diagnostic
):
    routes, requested, _, _ = http_boundary
    route(routes, "httpfs", first)
    route(routes, "spatial", second)
    with pytest.raises(backend.DuckDBSpatialError, match=diagnostic):
        backend.provision_spatial_extension(max_extension_bytes=budget)
    assert not list(isolated_engine.rglob("*.duckdb_extension"))
    assert requested[0] == "/httpfs.duckdb_extension.gz"


def test_network_failure_is_retained_without_install(http_boundary, isolated_engine):
    routes, requested, _, _ = http_boundary
    route(routes, "httpfs", b"", mode="error")
    with pytest.raises(backend.DuckDBSpatialError, match="HTTP Error 503") as raised:
        backend.provision_spatial_extension()
    assert raised.value.runtime_version.startswith("v")
    assert raised.value.platform
    assert '"http_status": 503' in str(raised.value)
    assert requested == ["/httpfs.duckdb_extension.gz"]
    assert not list(isolated_engine.rglob("*.duckdb_extension"))


def test_real_engine_rejects_unsigned_native_payload_after_both_downloads(
    http_boundary, isolated_engine
):
    import duckdb

    routes, requested, _, _ = http_boundary
    route(routes, "httpfs", gzip.compress(b"hostile arbitrary native bytes"))
    route(routes, "spatial", gzip.compress(b"another unsigned payload"))
    with pytest.raises(duckdb.Error):
        backend.provision_spatial_extension()
    assert requested == ["/httpfs.duckdb_extension.gz", "/spatial.duckdb_extension.gz"]
    assert not list(isolated_engine.rglob("*.duckdb_extension"))


def test_real_worker_returns_independently_hashed_archive(http_boundary, tmp_path):
    routes, _, _, _ = http_boundary
    content = b"independently known extension bytes" * 100
    route(routes, "httpfs", gzip.compress(content))
    url = "https://extensions.duckdb.org/v1.5.5/linux_amd64/httpfs.duckdb_extension.gz"
    target = tmp_path / "archive"
    result = backend._download_extension(
        url, target, deadline=time.monotonic() + 5, max_bytes=10_000
    )
    assert target.read_bytes() == content
    assert result == {
        "download_url": url,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_missing_backend_is_explicit(monkeypatch):
    monkeypatch.setattr(backend, "_DUCKDB", None)
    with pytest.raises(backend.DuckDBSpatialError, match="integrations"):
        backend.provision_spatial_extension()


def load_cli():
    path = Path(__file__).parents[2] / "provision_duckdb_spatial.py"
    spec = importlib.util.spec_from_file_location("provision_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_failure_keeps_json_and_stderr(capsys, monkeypatch):
    cli = load_cli()
    error = backend.DuckDBSpatialError("independent network diagnostic")
    error.runtime_version = "v1.5.5"
    error.platform = "linux_amd64"

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(cli, "provision_spatial_extension", fail)
    assert cli.main([]) == 1
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "failed"
    assert result["error_type"] == "DuckDBSpatialError"
    assert result["diagnostic"] == "independent network diagnostic"
    assert result["runtime_version"] == "v1.5.5"
    assert result["platform"] == "linux_amd64"
    assert result["duration_seconds"] >= 0
    assert "independent network diagnostic" in captured.err


def test_cli_invalid_deadline_is_failed_json(capsys):
    assert load_cli().main(["--download-timeout", "nan"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "failed"
    assert "finite and positive" in captured.err
