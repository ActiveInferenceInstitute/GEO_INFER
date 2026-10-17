"""Keep native libraries and inherited child stdout off machine transports."""

from __future__ import annotations

import ctypes
import io
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO


def _flush_native() -> None:
    """Flush the process C runtime before changing its stdout descriptor."""
    runtime = ctypes.CDLL("ucrtbase" if os.name == "nt" else None)
    flush = runtime.fflush
    flush.argtypes = [ctypes.c_void_p]
    flush.restype = ctypes.c_int
    flush(None)


def _windows_stdout(descriptor: int) -> None:
    """Keep Windows child-process standard handles aligned with CRT stdout."""
    if os.name != "nt" or descriptor != 1:
        return
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    setter = kernel.SetStdHandle
    setter.argtypes = [wintypes.DWORD, wintypes.HANDLE]
    setter.restype = wintypes.BOOL
    if not setter(wintypes.DWORD(-11), msvcrt.get_osfhandle(descriptor)):
        raise ctypes.WinError(ctypes.get_last_error())


@contextmanager
def protocol_stdout() -> Iterator[TextIO]:
    """Yield a dedicated protocol stream while fd 1 points at stderr.

    Only process entrypoints use this process-wide redirection. The script
    session API leaves native descriptors alone. The dedicated duplicate is
    non-inheritable, so child processes retain stderr as their native stdout.
    Restoration runs on error and normal completion; stream encodings are UTF-8.
    """
    original = sys.stdout
    original.flush()
    sys.stderr.flush()
    _flush_native()
    descriptor = original.fileno()
    saved = os.dup(descriptor)
    stream = io.TextIOWrapper(
        os.fdopen(os.dup(descriptor), "wb", buffering=0),
        encoding="utf-8",
        write_through=True,
    )
    try:
        os.dup2(sys.stderr.fileno(), descriptor)
        _windows_stdout(descriptor)
        yield stream
    finally:
        # CPython stream writes and buffered C printf output must reach stderr
        # before restoring the JSON/MCP destination.
        original.flush()
        sys.stderr.flush()
        _flush_native()
        os.dup2(saved, descriptor)
        _windows_stdout(descriptor)
        os.close(saved)
        stream.close()
