#!/usr/bin/env python3

"""
Package import hygiene tests for geo_infer_agent.

Importing the package must be side-effect free with respect to logging:
no handlers may be attached and the root logger level must stay at the
interpreter default. The imports run in a subprocess so interpreter-global
state is captured cleanly regardless of pytest's own logging configuration.
"""

import json
import subprocess
import sys
import unittest


def _probe_import(target: str) -> dict:
    """Import *target* in a subprocess and report root-logging state."""
    if target:
        import_line = f"import {target}\n"
    else:
        import_line = ""
    probe = (
        "import json, logging\n"
        f"{import_line}"
        "root = logging.getLogger()\n"
        "print(json.dumps({\n"
        "    'root_handlers': len(root.handlers),\n"
        "    'root_level': root.level,\n"
        "}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestPackageImportHygiene(unittest.TestCase):
    """Importing geo_infer_agent must not configure logging."""

    def test_import_attaches_no_handlers_and_sets_no_root_level(self) -> None:
        """No handlers attach and root level matches the interpreter default."""
        baseline = _probe_import("")
        after_import = _probe_import("geo_infer_agent")

        self.assertEqual(baseline["root_handlers"], 0)
        self.assertEqual(
            after_import["root_handlers"],
            baseline["root_handlers"],
        )
        self.assertEqual(
            after_import["root_level"],
            baseline["root_level"],
        )


if __name__ == "__main__":
    unittest.main()
