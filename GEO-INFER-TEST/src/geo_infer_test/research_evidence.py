"""Bounded custody checks for five manuscript JSON outputs, not producer acceptance.

No source bytes are copied. Callers must upload the exact allowlist only after
``outcome == 'success'`` and retain the receipt separately on either outcome.
Reported identities are producer assertions; freshness and producer success are
not established. Output must remain quiescent between checking and uploading.
The filesystem boundary requires POSIX no-follow descriptor operations; platforms
without those primitives fail with sanitized diagnostics.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

SOURCE_PATHS = (
    "data/research_manifest.json",
    "data/research_inventory.json",
    "data/research_verification.json",
    "data/manuscript_variables.json",
    "figures/figure_registry.json",
)
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_RESULTS = 256


class UnsupportedFilesystemError(OSError):
    """The platform cannot provide the required no-follow filesystem boundary."""


def _directory(path: Path) -> int:
    """Open every path component without following symbolic links."""
    if not all(
        hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    ) or not {os.open, os.stat, os.unlink, os.rename}.issubset(os.supports_dir_fd):
        raise UnsupportedFilesystemError("Required filesystem primitives unavailable")
    absolute = Path(os.path.abspath(path))
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _source_paths_unchanged(output_root: Path, root_fd: int, opened: list) -> bool:
    """Rewalk current names without following aliases before authorizing upload.

    Held descriptors alone can still refer to renamed files or directories.
    This final check binds current upload names to the bytes checked earlier;
    it does not replace the caller's quiescence requirement during upload.
    """
    try:
        with ExitStack() as stack:
            current_root = _directory(output_root)
            stack.callback(os.close, current_root)
            if not _same_identity(os.fstat(current_root), os.fstat(root_fd)):
                return False
            for record, fd, before, directory_fd in opened:
                parent, filename = record["path"].split("/")
                current_directory = os.open(
                    parent,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current_root,
                )
                stack.callback(os.close, current_directory)
                if not _same_identity(
                    os.fstat(current_directory), os.fstat(directory_fd)
                ):
                    return False
                current = os.stat(
                    filename, dir_fd=current_directory, follow_symlinks=False
                )
                held = os.fstat(fd)
                if not _same_identity(current, held) or any(
                    getattr(info, field) != getattr(before, field)
                    for info in (current, held)
                    for field in ("st_size", "st_mtime_ns", "st_ctime_ns")
                ):
                    return False
    except OSError:
        return False
    return True


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _number(value: str) -> float:
    import math

    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _identity(value: object) -> str | None:
    # Never retain arbitrary strings (commands, output tails or private paths).
    if isinstance(value, str) and re.fullmatch(
        r"(?:[0-9a-f]{7,64}(?:-dirty)?|unstamped|unknown)", value
    ):
        return value
    return None


def _provenance(path: str, payload: dict) -> dict:
    commit_key = (
        "commit" if path.endswith("research_inventory.json") else "source_commit"
    )
    hash_key = "source_hash"
    if path.endswith("manuscript_variables.json"):
        commit_key, hash_key = "RESEARCH_COMMIT", "RESEARCH_SOURCE_HASH"
    result = {
        "reported_source_commit": _identity(payload.get(commit_key)),
        "reported_source_hash": _identity(payload.get(hash_key)),
    }
    expected_schema = (
        "geo-infer-manuscript-figures/v1"
        if path.startswith("figures/")
        else "geo-infer-manuscript-evidence/v1"
    )
    result["reported_schema_version"] = (
        expected_schema if payload.get("schema_version") == expected_schema else None
    )
    if path.endswith("research_manifest.json"):
        result.update(
            {
                "reported_verification_source_commit": _identity(
                    payload.get("verification_source_commit")
                ),
                "reported_verification_source_hash": _identity(
                    payload.get("verification_source_hash")
                ),
                "reported_verification_measured_elsewhere": (
                    payload.get("verification_measured_elsewhere")
                    if type(payload.get("verification_measured_elsewhere")) is bool
                    else None
                ),
            }
        )
    return result


def _verification(payload: dict) -> dict:
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) > MAX_RESULTS:
        raise ValueError("invalid or unbounded verification results")
    retained = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid verification result")
        name, status, code = row.get("name"), row.get("status"), row.get("return_code")
        # Enumerate statuses rather than reflecting untrusted strings into receipt.
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name)
            or status
            not in ("passed", "failed", "timeout", "timed_out", "error", "not_run")
            or not (code is None or (type(code) is int and -(2**31) <= code < 2**31))
        ):
            raise ValueError("invalid verification status")
        retained.append({"name": name, "status": status, "return_code": code})
    tier = payload.get("full_validation_requested")
    if type(tier) is not bool:
        raise ValueError("invalid verification tier")
    return {
        "full_validation_requested": tier,
        "results": retained,
        "has_failure_evidence": any(
            row["status"] != "passed" or row["return_code"] != 0 for row in retained
        ),
        "producer_success_established": False,
    }


def validate_research_evidence(output_root: Path) -> dict:
    """Return a bounded receipt; all errors use codes and allowlisted paths only.

    Preflight checks all sizes before any read. Descriptor-relative no-follow
    opens reject symlinks, directories, devices and path-component replacement.
    JSON must be UTF-8, a top-level object, finite and free of duplicate keys.
    Semantic producer checks and cross-file freshness are deliberately excluded.
    """
    records = [{"path": path, "errors": []} for path in SOURCE_PATHS]
    receipt = {
        "schema_version": "geo-infer-research-custody/v1",
        "outcome": "failure",
        "limits": {
            "file_bytes": MAX_FILE_BYTES,
            "total_bytes": MAX_TOTAL_BYTES,
            "receipt_bytes": MAX_RECEIPT_BYTES,
        },
        "files": records,
        "errors": [],
        "upload_source_paths": [],
        "claims": {
            "producer_success_established": False,
            "freshness_established": False,
        },
    }
    with ExitStack() as stack:
        try:
            root_fd = _directory(Path(output_root))
            stack.callback(os.close, root_fd)
        except UnsupportedFilesystemError:
            receipt["errors"].append("unsupported_filesystem_primitives")
            return receipt
        except OSError:
            receipt["errors"].append("unsafe_or_unavailable_output_root")
            return receipt
        opened = []
        total = 0
        for record in records:
            parent, filename = record["path"].split("/")
            try:
                directory_fd = os.open(
                    parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
                )
                stack.callback(os.close, directory_fd)
                fd = os.open(
                    filename,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=directory_fd,
                )
                stack.callback(os.close, fd)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    record["errors"].append("not_regular_file")
                    continue
                record["size_bytes"] = info.st_size
                total += info.st_size
                if info.st_size > MAX_FILE_BYTES:
                    record["errors"].append("file_size_limit")
                opened.append((record, fd, info, directory_fd))
            except FileNotFoundError:
                record["errors"].append("missing")
            except OSError:
                record["errors"].append("unsafe_or_unreadable_file")
        receipt["total_size_bytes"] = total
        if total > MAX_TOTAL_BYTES:
            receipt["errors"].append("total_size_limit")
            return receipt
        for record, fd, before, _directory_fd in opened:
            if record["errors"]:
                continue
            try:
                chunks = []
                remaining = before.st_size + 1
                while remaining:
                    chunk = os.read(fd, min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
                after = os.fstat(fd)
                if (
                    len(raw) != before.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                    or before.st_ctime_ns != after.st_ctime_ns
                    or before.st_size != after.st_size
                ):
                    record["errors"].append("changed_during_read")
                    continue
                record["sha256"] = hashlib.sha256(raw).hexdigest()
                payload = json.loads(
                    raw.decode("utf-8"),
                    parse_constant=_reject_constant,
                    parse_float=_number,
                    object_pairs_hook=_pairs,
                )
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
                record["provenance"] = _provenance(record["path"], payload)
                if record["path"] == "data/research_verification.json":
                    record["producer_verification"] = _verification(payload)
            except (ValueError, UnicodeError, RecursionError):
                record["errors"].append("invalid_json_or_evidence_shape")
            except OSError:
                record["errors"].append("read_error")
        if not _source_paths_unchanged(Path(output_root), root_fd, opened):
            receipt["errors"].append("source_paths_changed")
        if not receipt["errors"] and not any(row["errors"] for row in records):
            receipt["outcome"] = "success"
            receipt["upload_source_paths"] = list(SOURCE_PATHS)
    return receipt


def write_research_receipt(
    receipt: dict, receipt_path: Path, *, output_root: Path
) -> None:
    """Atomically write a bounded receipt without overwriting an allowlisted source.

    The caller supplies an existing trusted destination directory; symlinks in
    its path are rejected. Unwritable destinations remain an operational error.
    """
    destination = Path(os.path.abspath(receipt_path))
    root = Path(os.path.abspath(output_root))
    try:
        canonical_destination = destination.resolve(strict=False)
        canonical_sources = {
            (root / path).resolve(strict=False) for path in SOURCE_PATHS
        }
    except (OSError, RuntimeError) as error:
        raise ValueError("unsafe source or receipt path") from error
    if canonical_destination in canonical_sources:
        raise ValueError("receipt destination overlaps source allowlist")
    encoded = (
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ValueError("receipt size limit")
    fd = _directory(destination.parent)
    temporary = ".research-custody-" + uuid.uuid4().hex
    try:
        try:
            info = os.stat(destination.name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            info = None
        if info is not None and not stat.S_ISREG(info.st_mode):
            raise ValueError("unsafe receipt destination")
        file_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=fd,
        )
        with os.fdopen(file_fd, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary, destination.name, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError:
            pass
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    """CLI with a mandatory receipt path and sanitized operational diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--receipt-path", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate_research_evidence(args.output_root)
    try:
        write_research_receipt(receipt, args.receipt_path, output_root=args.output_root)
    except (OSError, ValueError):
        print("Research custody: failure (receipt destination unavailable or unsafe)")
        return 1
    print(
        f"Research custody: {receipt['outcome']} (producer success and freshness not established)"
    )
    return 0 if receipt["outcome"] == "success" else 1
