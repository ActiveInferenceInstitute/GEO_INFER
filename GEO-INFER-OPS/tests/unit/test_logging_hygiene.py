"""Import-time logging hygiene for GEO-INFER-OPS.

Importing the library must not mutate the root logger: no handlers added,
no level changed. Global configuration belongs exclusively to
``shared_logging.configure_logging``, invoked by CLI entrypoints.
"""

import json
import subprocess
import sys


HYGIENE_SCRIPT = """
import json
import logging

before_handlers = len(logging.root.handlers)
before_level = logging.root.level

import geo_infer_ops  # noqa: F401 -- import side effect under test
import geo_infer_ops.core  # noqa: F401 -- import side effect under test
import geo_infer_ops.utils.shared_logging  # noqa: F401 -- import side effect under test

after_handlers = len(logging.root.handlers)
after_level = logging.root.level

print(json.dumps({
    "before_handlers": before_handlers,
    "after_handlers": after_handlers,
    "before_level": before_level,
    "after_level": after_level,
}))
"""


def test_import_does_not_add_root_handlers() -> None:
    """Importing geo_infer_ops in a fresh interpreter leaves root untouched."""
    result = subprocess.run(
        [sys.executable, "-c", HYGIENE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["after_handlers"] == report["before_handlers"], (
        "importing geo_infer_ops added root logger handlers: "
        f"{report['before_handlers']} -> {report['after_handlers']}"
    )
    assert report["after_level"] == report["before_level"], (
        "importing geo_infer_ops changed the root logger level: "
        f"{report['before_level']} -> {report['after_level']}"
    )
