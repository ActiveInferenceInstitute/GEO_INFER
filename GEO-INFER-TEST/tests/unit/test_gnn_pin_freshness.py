"""Real temporary Git histories for the GNN pin freshness boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from geo_infer_test.gnn_pin_freshness import (
    read_pin_report,
    validate_gnn_pin_freshness,
)

REPOSITORY = "ActiveInferenceInstitute/Generalized_Notation_Notation"


def git(root, *arguments):
    """Operate only on this test's disposable repository."""
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT, timeout=15
    ).strip()


class GnnPinFreshnessTests(unittest.TestCase):
    """Exercise lineage, release tags and currency escalation."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="geo-gnn-pin-freshness-")
        self.addCleanup(self.temp.cleanup)
        self.geo = Path(self.temp.name) / "geo"
        self.gnn = Path(self.temp.name) / "gnn"
        for root in (self.geo, self.gnn):
            root.mkdir()
            git(root, "init", "--quiet")
            git(root, "config", "user.name", "Pin freshness test")
            git(root, "config", "user.email", "fixture@example.invalid")
            git(root, "config", "commit.gpgsign", "false")
            git(root, "config", "tag.gpgsign", "false")
        self.write(self.gnn, "source.txt", "released source\n")
        self.released = self.commit(self.gnn, "released source")
        git(self.gnn, "tag", "-a", "v1.0.0", "-m", "released source")
        self.write(self.gnn, "source.txt", "second release\n")
        self.second = self.commit(self.gnn, "second release")
        git(self.gnn, "tag", "-a", "v1.1.0", "-m", "second release")
        self.write(self.gnn, "source.txt", "checkout head\n")
        self.head = self.commit(self.gnn, "checkout head")
        self.manifest = {
            "schema": "geo-infer-companion-releases/v1",
            "geo": {
                "repository": "ActiveInferenceInstitute/GEO_INFER",
                "target_version": "0.4.0",
                "baseline_revision": self.released,
            },
            "companions": {"gnn": {"revision": self.released}},
            "interchange_schemas": [
                "gnn-geo-infer/1",
                "gnn-geo-infer/2",
                "gnn-geo-infer/factored/1",
            ],
            "evidence_boundaries": {},
        }
        self.write(
            self.geo, ".github/companion-releases.json", json.dumps(self.manifest)
        )

    def write(self, root, relative, text):
        """Write one fixture-owned path."""
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def pin(self, revision, tag=None):
        """Return the ``pin`` record a ``gnn.importers check --json`` report holds.

        The review never reads ``.github/gnn-pair.json`` itself, so the fixture
        supplies the reader's normalized record instead of a second pin shape.
        """
        return {
            "commit": revision,
            "repository": REPOSITORY,
            "party": None,
            "path": ".github/gnn-pair.json",
            "schema": "gnn-importer-pin/v1",
            "recorded": "2026-11-12",
            "spec_files": [],
            "tag": tag,
        }

    def commit(self, root, message):
        """Commit all fixture changes and return their actual Git identity."""
        git(root, "add", ".")
        git(root, "commit", "--quiet", "-m", message)
        return git(root, "rev-parse", "HEAD")

    def test_released_pin_reports_lineage_and_release_tag(self):
        result = validate_gnn_pin_freshness(
            self.geo, self.gnn, pin=self.pin(self.second), require_current=True
        )
        self.assertEqual(result["status"], "passed_pin_freshness_review")
        self.assertEqual(result["pin"]["revision"], self.second)
        self.assertEqual(result["pin"]["release_tags_at_pin"], ["v1.1.0"])
        self.assertEqual(result["published_companion"]["revision"], self.released)
        self.assertEqual(result["checkout"]["commits_ahead_of_pin"], 1)
        self.assertTrue(result["newest_release"]["equals_pin"])
        self.assertTrue(result["claims"]["release_currency"])

    def test_pin_record_comes_from_the_shared_reader_report(self):
        """A stale pin file cannot change the answer: only the report is read."""
        self.write(
            self.geo,
            ".github/gnn-pair.json",
            json.dumps({"repository": "example/other", "revision": self.released}),
        )
        result = validate_gnn_pin_freshness(
            self.geo, self.gnn, pin=self.pin(self.second, tag="v1.1.0")
        )
        self.assertEqual(result["pin"]["revision"], self.second)
        self.assertEqual(result["pin"]["tag"], "v1.1.0")
        self.assertEqual(result["pin"]["repository"], REPOSITORY)

    def test_pin_ahead_of_the_newest_release_is_current(self):
        result = validate_gnn_pin_freshness(
            self.geo, self.gnn, pin=self.pin(self.head), require_current=True
        )
        self.assertFalse(result["newest_release"]["equals_pin"])
        self.assertTrue(result["newest_release"]["is_ancestor_of_pin"])
        self.assertTrue(result["newest_release"]["pin_is_current"])
        self.assertEqual(result["pin"]["release_tags_at_pin"], [])

    def test_pin_behind_the_newest_release_escalates_only_when_required(self):
        pin = self.pin(self.released)
        result = validate_gnn_pin_freshness(self.geo, self.gnn, pin=pin)
        self.assertFalse(result["newest_release"]["pin_is_current"])
        self.assertTrue(result["newest_release"]["pin_is_ancestor"])
        self.assertEqual(result["newest_release"]["commits_behind"], 1)
        self.assertFalse(result["claims"]["release_currency"])
        with self.assertRaisesRegex(ValueError, "behind the newest GNN release tag"):
            validate_gnn_pin_freshness(
                self.geo, self.gnn, pin=pin, require_current=True
            )

    def test_absent_or_divergent_pins_refuse(self):
        with self.assertRaisesRegex(ValueError, "absent from the checkout"):
            validate_gnn_pin_freshness(self.geo, self.gnn, pin=self.pin("0" * 40))
        git(self.gnn, "checkout", "--quiet", "--orphan", "divergent")
        self.write(self.gnn, "source.txt", "divergent source\n")
        divergent = self.commit(self.gnn, "divergent source")
        with self.assertRaisesRegex(
            ValueError, "does not retain the published companion revision"
        ):
            validate_gnn_pin_freshness(self.geo, self.gnn, pin=self.pin(divergent))
        git(self.gnn, "checkout", "--quiet", self.head)

    def test_unresolvable_commit_or_malformed_manifest_refuses(self):
        with self.assertRaisesRegex(ValueError, "absent from the checkout"):
            validate_gnn_pin_freshness(self.geo, self.gnn, pin=self.pin("not-an-oid"))
        with self.assertRaisesRegex(ValueError, "pin report records no commit"):
            validate_gnn_pin_freshness(self.geo, self.gnn, pin={"commit": ""})
        self.manifest["schema"] = "geo-infer-companion-releases/v2"
        self.write(
            self.geo, ".github/companion-releases.json", json.dumps(self.manifest)
        )
        with self.assertRaisesRegex(ValueError, "unknown companion manifest schema"):
            validate_gnn_pin_freshness(self.geo, self.gnn, pin=self.pin(self.second))
        self.manifest["schema"] = "geo-infer-companion-releases/v1"
        self.manifest["companions"]["gnn"]["revision"] = "0" * 40
        self.write(
            self.geo, ".github/companion-releases.json", json.dumps(self.manifest)
        )
        with self.assertRaisesRegex(
            ValueError, "published companion revision is absent"
        ):
            validate_gnn_pin_freshness(self.geo, self.gnn, pin=self.pin(self.second))

    def test_pin_report_reader_refuses_a_report_without_a_pin(self):
        report = Path(self.temp.name) / "report.json"
        report.write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pin report is not JSON"):
            read_pin_report(report)
        report.write_text(json.dumps({"ok": True, "pin": None}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pin report records no pin"):
            read_pin_report(report)
        report.write_text(json.dumps({"pin": {"path": "x"}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pin report records no commit"):
            read_pin_report(report)
        accepted = read_pin_report(
            self.write_report(self.pin(self.second, tag="v1.1.0"))
        )
        self.assertEqual(accepted["commit"], self.second)
        self.assertEqual(accepted["tag"], "v1.1.0")

    def write_report(self, pin):
        """Write a reader report and return its path."""
        report = Path(self.temp.name) / "gnn-importer-report.json"
        report.write_text(json.dumps({"ok": True, "pin": pin}), encoding="utf-8")
        return report

    def test_cli_emits_actual_result_or_bounded_refusal(self):
        module = Path(__file__).resolve().parents[2]
        environment = dict(os.environ, PYTHONPATH=str(module / "src"))
        report = self.write_report(self.pin(self.second))
        command = [
            sys.executable,
            "-B",
            str(module / "validate_gnn_pin_freshness.py"),
            "--geo-root",
            str(self.geo),
            "--gnn-root",
            str(self.gnn),
            "--pin-report",
            str(report),
            "--require-current",
        ]
        accepted = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        payload = json.loads(accepted.stdout)
        self.assertEqual(payload["pin"]["revision"], self.second)
        self.write_report(self.pin(self.released))
        refused = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(refused.returncode, 1)
        self.assertEqual(
            json.loads(refused.stdout)["status"], "refused_pin_freshness_review"
        )
        self.assertNotIn(self.temp.name, refused.stdout + refused.stderr)


if __name__ == "__main__":
    unittest.main()
