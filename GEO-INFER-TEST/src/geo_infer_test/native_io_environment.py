"""Explicit Linux ecCodes/GIS library lookup before a new process starts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib.metadata
import json
import os
from pathlib import Path
import sys

PROFILE = "linux-ecmwf-local-v1"
VARIABLES = frozenset(
    {
        "FINDLIBS_DISABLE_PACKAGE",
        "FINDLIBS_DISABLE_PYTHON",
        "FINDLIBS_DISABLE_HOME",
        "ECCODESLIB_HOME",
        "LD_LIBRARY_PATH",
    }
)


class NativeIOProfileError(ValueError):
    """The selected native wheel layout cannot provide the Linux profile."""


def _library_directory(root: Path, library: str) -> Path:
    resolved = root.resolve()
    if any(character in str(resolved) for character in (":", "\n", "\r", "\0")):
        raise NativeIOProfileError(
            "Native library paths cannot contain colon or line separators"
        )
    matches = [
        resolved / name
        for name in ("lib", "lib64")
        if (resolved / name / library).is_file()
    ]
    if len(matches) != 1:
        raise NativeIOProfileError(
            f"Expected exactly one lib/lib64 directory containing {library} under {resolved}"
        )
    return matches[0]


def build_environment(
    eccodes_root: Path, eckit_root: Path, original: Mapping[str, str]
) -> dict[str, str]:
    """Resolve wheel paths only; this neither loads libraries nor verifies I/O."""
    codes = _library_directory(eccodes_root, "libeccodes.so")
    eckit = _library_directory(eckit_root, "libeckit_geo.so")
    existing = original.get("LD_LIBRARY_PATH", "")
    if any(character in existing for character in ("\n", "\r", "\0")):
        raise NativeIOProfileError("LD_LIBRARY_PATH contains a line separator")
    search = [str(eckit), str(codes)]
    if existing:
        search.append(existing)
    return {
        "FINDLIBS_DISABLE_PACKAGE": "yes",
        "FINDLIBS_DISABLE_PYTHON": "yes",
        "FINDLIBS_DISABLE_HOME": "no",
        "ECCODESLIB_HOME": str(eccodes_root.resolve()),
        "LD_LIBRARY_PATH": ":".join(search),
    }


def installed_environment(original: Mapping[str, str]) -> dict[str, str]:
    """Build this Linux-only profile from this interpreter's installed wheels."""
    if sys.platform != "linux":
        raise NativeIOProfileError("The native I/O startup profile is Linux-only")
    roots = []
    for name in ("eccodeslib", "eckitlib"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise NativeIOProfileError(
                f"Missing native dependency {name}; install the declared GRIB extras"
            ) from error
        roots.append(Path(distribution.locate_file(name)))
    return build_environment(roots[0], roots[1], original)


def write_github_environment(path: Path, bindings: Mapping[str, str]) -> None:
    """Append one validated profile to the runner's process-environment file."""
    if set(bindings) != VARIABLES:
        raise NativeIOProfileError(
            "Incomplete or unknown native I/O environment variables"
        )
    if any(any(c in value for c in ("\n", "\r", "\0")) for value in bindings.values()):
        raise NativeIOProfileError("Native I/O environment values must occupy one line")
    payload = "".join(f"{name}={bindings[name]}\n" for name in sorted(bindings))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--describe",
        action="store_true",
        help="Describe lookup configuration; no native execution claim",
    )
    modes.add_argument(
        "--github-env", type=Path, help="Append bindings for subsequent CI processes"
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command after --; receives the profile before startup",
    )
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command and (args.describe or args.github_env is not None):
        parser.error("Select a command or an environment-output mode")
    if not command and not args.describe and args.github_env is None:
        parser.error("A command after --, --describe or --github-env is required")
    try:
        bindings = installed_environment(os.environ)
        if args.describe:
            print(
                json.dumps(
                    {
                        "profile": PROFILE,
                        "environment": bindings,
                        "execution_verified": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.github_env is not None:
            write_github_environment(args.github_env, bindings)
            print(f"Configured {PROFILE}; actual I/O requires its execution checks")
            return 0
    except (NativeIOProfileError, OSError) as error:
        parser.error(str(error))
    child_environment = dict(os.environ)
    child_environment.update(bindings)
    try:
        os.execvpe(command[0], command, child_environment)
    except OSError as error:
        status = 127 if isinstance(error, FileNotFoundError) else 126
        print(
            f"Native I/O command could not start: {type(error).__name__}",
            file=sys.stderr,
        )
        return status


if __name__ == "__main__":
    raise SystemExit(main())
