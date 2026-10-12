"""Archive-traversal (Tar Slip) regression tests for GEO-INFER-DATA.

Covers the safe-extract guards added to ``geo_infer_data.connectors.file``:
extraction must reject archive members that resolve outside the target
directory instead of writing arbitrary files.
"""

import tarfile
import zipfile

import pytest

from geo_infer_data.connectors.file import (
    FileConnector,
    _safe_extract_tar,
    _safe_extract_zip,
)


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _evil_zip(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../evil.txt", "pwned")


def _evil_tar(path):
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("../evil.txt")
        payload = b"pwned"
        info.size = len(payload)
        archive.addfile(info, __import__("io").BytesIO(payload))


def test_zip_traversal_member_rejected(tmp_path):
    """Zip members escaping the target directory raise ValueError."""
    archive = tmp_path / "evil.zip"
    _evil_zip(archive)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        _run(FileConnector(base_path=str(tmp_path)).extract_archive(archive, out))
    assert not (tmp_path / "evil.txt").exists()


def test_zip_traversal_member_rejected_direct(tmp_path):
    """_safe_extract_zip rejects traversal on its own."""
    archive = tmp_path / "evil.zip"
    _evil_zip(archive)
    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ValueError, match="Unsafe archive member path"):
            _safe_extract_zip(zf, out)


def test_tar_traversal_member_rejected(tmp_path):
    """Tar members escaping the target directory raise ValueError."""
    archive = tmp_path / "evil.tar"
    _evil_tar(archive)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        _run(FileConnector(base_path=str(tmp_path)).extract_archive(archive, out))
    assert not (tmp_path / "evil.txt").exists()


def test_tar_traversal_member_rejected_direct(tmp_path):
    """_safe_extract_tar rejects traversal on its own."""
    archive = tmp_path / "evil.tar"
    _evil_tar(archive)
    out = tmp_path / "out"
    out.mkdir()
    with tarfile.open(archive) as tf:
        with pytest.raises(ValueError, match="Unsafe archive member path"):
            _safe_extract_tar(tf, out)


def test_zip_benign_round_trip_still_works(tmp_path):
    """Normal zip extraction is unaffected by the guards."""
    src = tmp_path / "payload.txt"
    src.write_text("hello")
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(src, "payload.txt")
    out = tmp_path / "out"
    out.mkdir()
    extracted = _run(
        FileConnector(base_path=str(tmp_path)).extract_archive(archive, out)
    )
    assert len(extracted) == 1
    assert (out / "payload.txt").read_text() == "hello"
