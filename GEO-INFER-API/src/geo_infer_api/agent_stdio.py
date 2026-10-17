"""Reserve stdin/stdout for machine protocols; isolate native child streams."""

from __future__ import annotations

import ctypes
import io
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, ExitStack
from typing import TextIO


def _flush_native() -> None:
    """Flush the process C runtime before changing its stdout descriptor."""
    runtime = ctypes.CDLL("ucrtbase" if os.name == "nt" else None)
    flush = runtime.fflush
    flush.argtypes = [ctypes.c_void_p]
    flush.restype = ctypes.c_int
    flush(None)


def _windows_standard_handle(descriptor: int) -> None:
    """Align Windows child-process standard handles with CRT stdin/stdout."""
    if os.name != "nt" or descriptor not in (0, 1):
        return
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    setter = kernel.SetStdHandle
    setter.argtypes = [wintypes.DWORD, wintypes.HANDLE]
    setter.restype = wintypes.BOOL
    if not setter(
        wintypes.DWORD(-10 if descriptor == 0 else -11),
        msvcrt.get_osfhandle(descriptor),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


@contextmanager
def protocol_stdio() -> Iterator[tuple[TextIO, TextIO]]:
    """Yield dedicated protocol streams; native stdin gets EOF, stdout stderr.

    Only process entrypoints use this process-wide redirection. The script
    session API leaves native descriptors alone. The dedicated duplicate is
    non-inheritable. Child processes cannot consume MCP input or write to its
    output. Native stdin receives EOF, preventing interactive/forwarding hangs.
    Restoration runs on error and normal completion; stream encodings are UTF-8.
    """
    original = sys.stdout
    original.flush()
    sys.stderr.flush()
    _flush_native()
    input_descriptor, output_descriptor = sys.stdin.fileno(), original.fileno()
    with ExitStack() as resources:
        saved_input = os.dup(input_descriptor)
        resources.callback(os.close, saved_input)
        saved_output = os.dup(output_descriptor)
        resources.callback(os.close, saved_output)
        reader = resources.enter_context(
            io.TextIOWrapper(
                os.fdopen(os.dup(input_descriptor), "rb", buffering=0), encoding="utf-8"
            )
        )
        writer = resources.enter_context(
            io.TextIOWrapper(
                os.fdopen(os.dup(output_descriptor), "wb", buffering=0),
                encoding="utf-8",
                write_through=True,
            )
        )
        null = resources.enter_context(open(os.devnull, "rb", buffering=0))
        try:
            os.dup2(null.fileno(), input_descriptor)
            _windows_standard_handle(input_descriptor)
            os.dup2(sys.stderr.fileno(), output_descriptor)
            _windows_standard_handle(output_descriptor)
            yield reader, writer
        finally:
            try:
                original.flush()
                sys.stderr.flush()
                _flush_native()
            finally:
                os.dup2(saved_input, input_descriptor)
                _windows_standard_handle(input_descriptor)
                os.dup2(saved_output, output_descriptor)
                _windows_standard_handle(output_descriptor)


@contextmanager
def protocol_stdout() -> Iterator[TextIO]:
    """CLI wrapper after reading its JSON request, reserving native stdio."""
    with protocol_stdio() as (_reader, writer):
        yield writer


@contextmanager
def _redirect_stdin(stream: TextIO) -> Iterator[None]:
    previous = sys.stdin
    sys.stdin = stream
    try:
        yield
    finally:
        sys.stdin = previous
