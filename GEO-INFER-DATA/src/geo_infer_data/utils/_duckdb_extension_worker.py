"""Private, stdlib-only signed-extension archive worker.

The calling process enforces a total deadline and reaps this single child.
This worker creates no child processes. Keeping opening, header parsing and
gzip parsing in the same child bounds operations with per-read timeout APIs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import urlsplit


def validate_extension_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "extensions.duckdb.org"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Extension downloads require the official HTTPS origin")


class OfficialExtensionRedirects(urllib.request.HTTPRedirectHandler):
    """Validate each redirect before opening its connection."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_extension_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_extension(url: str, timeout: float):
    validate_extension_url(url)
    request = urllib.request.Request(
        url, headers={"User-Agent": "GEO-INFER-DATA/0.4.0"}
    )
    return urllib.request.build_opener(OfficialExtensionRedirects()).open(
        request, timeout=timeout
    )


def download_extension(
    url: str, target: Path, *, timeout: float, max_bytes: int
) -> dict:
    """Download/decompress within byte budgets; caller owns the hard deadline."""
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("DuckDB extension download deadline exceeded")
        return seconds

    compressed = target.with_suffix(target.suffix + ".gz")
    with open_extension(url, remaining()) as response:
        validate_extension_url(response.geturl())
        size = 0
        with compressed.open("wb") as output:
            while True:
                remaining()
                chunk = response.read1(min(64 * 1024, max_bytes - size + 1))
                remaining()
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("DuckDB extension archive exceeds byte budget")
                output.write(chunk)
    size = 0
    digest = hashlib.sha256()
    with gzip.open(compressed, "rb") as archive, target.open("wb") as output:
        while True:
            remaining()
            chunk = archive.read(min(64 * 1024, max_bytes - size + 1))
            remaining()
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("DuckDB extension exceeds decompressed byte budget")
            digest.update(chunk)
            output.write(chunk)
    if not size:
        raise ValueError("DuckDB extension archive contains no extension bytes")
    return {"download_url": url, "sha256": digest.hexdigest(), "bytes": size}


def main() -> None:
    url, target, timeout, max_bytes = sys.argv[1:]
    try:
        result = download_extension(
            url, Path(target), timeout=float(timeout), max_bytes=int(max_bytes)
        )
    except urllib.error.HTTPError as exc:
        # Public origin/status diagnostics remain useful without retaining
        # cookies, authorization headers or unrelated process environment.
        print(
            json.dumps(
                {
                    "download_url": url,
                    "http_status": exc.code,
                    "headers": {
                        name: exc.headers[name]
                        for name in ("Date", "Content-Type", "Server", "CF-Ray")
                        if name in exc.headers
                    },
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
