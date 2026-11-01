"""Real temporary Git histories for the GNN pin freshness boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from geo_infer_test.gnn_pin_freshness import GNN_REPOSITORY, validate_gnn_pin_freshness


def git(root, *arguments):
    """Operate only on this test's disposable repository."""
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT, timeout=15
    ).strip()


class GnnPinFreshnessTests(unittest.TestCase):
    """Exercise pin existence, lineage, release tags and currency escalation."""

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
        self.pin(self.second)

    def write(self, root, relative, text):
        """Write one fixture-owned path."""
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def pin(self, revision):
        """Write a real two-field pin, never a synthetic validation result."""
        self.write(
            self.geo,
            ".github/gnn-pair.json",
            json.dumps({"repository": GNN_REPOSITORY, "revision": revision}),
        )

    def commit(self, root, message):
        """Commit all fixture changes and return their actual Git identity."""
        git(root, "add", ".")
        git(root, "commit", "--quiet", "-m", message)
        return git(root, "rev-parse", "HEAD")

    def test_released_pin_reports_lineage_and_release_tag(self):
        result = validate_gnn_pin_freshness(self.geo, self.gnn, require_current=True)
        self.assertEqual(result["status"], "passed_pin_freshness_review")
        self.assertEqual(result["pin"]["revision"], self.second)
        self.assertEqual(result["pin"]["release_tags_at_pin"], ["v1.1.0"])
        self.assertEqual(result["published_companion"]["revision"], self.released)
        self.assertEqual(result["checkout"]["commits_ahead_of_pin"], 1)
        self.assertTrue(result["newest_release"]["equals_pin"])
        self.assertTrue(result["claims"]["release_currency"])

    def test_pin_ahead_of_the_newest_release_is_current(self):
        self.pin(self.head)
        result = validate_gnn_pin_freshness(self.geo, self.gnn, require_current=True)
        self.assertFalse(result["newest_release"]["equals_pin"])
        self.assertTrue(result["newest_release"]["is_ancestor_of_pin"])
        self.assertTrue(result["newest_release"]["pin_is_current"])
        self.assertEqual(result["pin"]["release_tags_at_pin"], [])

    def test_pin_behind_the_newest_release_escalates_only_when_required(self):
        self.pin(self.released)
        result = validate_gnn_pin_freshness(self.geo, self.gnn)
        self.assertFalse(result["newest_release"]["pin_is_current"])
        self.assertTrue(result["newest_release"]["pin_is_ancestor"])
        self.assertEqual(result["newest_release"]["commits_behind"], 1)
        self.assertFalse(result["claims"]["release_currency"])
        with self.assertRaisesRegex(ValueError, "behind the newest GNN release tag"):
            validate_gnn_pin_freshness(self.geo, self.gnn, require_current=True)

    def test_absent_or_divergent_pins_refuse(self):
        self.pin("0" * 40)
        with self.assertRaisesRegex(ValueError, "absent from the checkout"):
            validate_gnn_pin_freshness(self.geo, self.gnn)
        self.pin(self.second)
        git(self.gnn, "checkout", "--quiet", "--orphan", "divergent")
        self.write(self.gnn, "source.txt", "divergent source\n")
        divergent = self.commit(self.gnn, "divergent source")
        self.pin(divergent)
        with self.assertRaisesRegex(
            ValueError, "does not retain the published companion revision"
        ):
            validate_gnn_pin_freshness(self.geo, self.gnn)
        git(self.gnn, "checkout", "--quiet", self.head)

    def test_malformed_pin_or_manifest_refuses(self):
        self.write(
            self.geo,
            ".github/gnn-pair.json",
            json.dumps({"repository": GNN_REPOSITORY, "revision": "not-an-oid"}),
        )
        with self.assertRaisesRegex(ValueError, "invalid pin revision"):
            validate_gnn_pin_freshness(self.geo, self.gnn)
        self.write(
            self.geo,
            ".github/gnn-pair.json",
            json.dumps({"repository": "example/other", "revision": self.second}),
        )
        with self.assertRaisesRegex(ValueError, "unexpected pin repository"):
            validate_gnn_pin_freshness(self.geo, self.gnn)
        self.write(
            self.geo,
            ".github/gnn-pair.json",
            json.dumps(
                {
                    "repository": GNN_REPOSITORY,
                    "revision": self.second,
                    "extra": True,
                }
            ),
        )
        with self.assertRaisesRegex(ValueError, "invalid pin keys"):
            validate_gnn_pin_freshness(self.geo, self.gnn)
        self.pin(self.second)
        self.manifest["schema"] = "geo-infer-companion-releases/v2"
        self.write(
            self.geo, ".github/companion-releases.json", json.dumps(self.manifest)
        )
        with self.assertRaisesRegex(ValueError, "unknown companion manifest schema"):
            validate_gnn_pin_freshness(self.geo, self.gnn)
        self.manifest["schema"] = "geo-infer-companion-releases/v1"
        self.manifest["companions"]["gnn"]["revision"] = "0" * 40
        self.write(
            self.geo, ".github/companion-releases.json", json.dumps(self.manifest)
        )
        with self.assertRaisesRegex(
            ValueError, "published companion revision is absent"
        ):
            validate_gnn_pin_freshness(self.geo, self.gnn)

    def test_cli_emits_actual_result_or_bounded_refusal(self):
        module = Path(__file__).resolve().parents[2]
        environment = dict(os.environ, PYTHONPATH=str(module / "src"))
        command = [
            sys.executable,
            "-B",
            str(module / "validate_gnn_pin_freshness.py"),
            "--geo-root",
            str(self.geo),
            "--gnn-root",
            str(self.gnn),
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
        self.pin(self.released)
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
