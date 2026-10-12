#!/usr/bin/env python3
"""Explicit PyTensor C-compiler profile for macOS 27 and Apple Clang.

Use PYTENSOR_FLAGS="cxx=/absolute/path/to/pytensor_clang++_macos27.py".
This removes only PyTensor's obsolete -ld64 flag; all other arguments reach the
real compiler unchanged. It retains native compilation and never selects a
Python or numerical fallback. See pymc-devs/pytensor issue 2268.
"""

from __future__ import annotations

import os
import platform
import sys


COMPILER = "/usr/bin/clang++"


def compiler_arguments(arguments: list[str], system: str, version: str) -> list[str]:
    """Validate the named profile and remove only the obsolete linker flag."""
    if system != "Darwin" or int(version.split(".")[0]) < 27:
        raise ValueError("This compiler profile requires macOS 27 or newer")
    return [COMPILER, *(argument for argument in arguments if argument != "-ld64")]


def main() -> None:
    """Replace this process with the native Apple compiler using argument arrays."""
    arguments = compiler_arguments(
        sys.argv[1:], platform.system(), platform.mac_ver()[0]
    )
    os.execv(COMPILER, arguments)


if __name__ == "__main__":
    main()
