"""Real temporary Git histories for offline companion identity boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from geo_infer_test.companion_releases import (
    EVIDENCE_BOUNDARIES,
    INTERCHANGE_SCHEMAS,
    MANIFEST_PATH,
    REPOSITORIES,
    validate_companion_releases,
)


def git(root, *arguments):
    """Operate only on this test's disposable repository."""
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT, timeout=15
    ).strip()


class CompanionReleaseTests(unittest.TestCase):
    """Exercise source, tags, version metadata, pin graph and failure limits."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="geo-companion-identity-")
        self.addCleanup(self.temp.cleanup)
        self.roots = {}
        for name in REPOSITORIES:
            root = Path(self.temp.name) / name
            root.mkdir()
            git(root, "init", "--quiet")
            git(root, "config", "user.name", "Companion identity test")
            git(root, "config", "user.email", "fixture@example.invalid")
            git(root, "config", "commit.gpgsign", "false")
            git(root, "config", "tag.gpgsign", "false")
            self.roots[name] = root
        self.write("geo", "pyproject.toml", '[project]\nversion = "0.4.0"\n')
        self.write("geo", "CITATION.cff", 'version: "0.4.0"\n')
        self.baseline = self.commit("geo", "accepted GEO baseline")
        for name, version in (("fep", "2.1.0"), ("gnn", "5.0.1")):
            self.write(name, "pyproject.toml", f'[project]\nversion = "{version}"\n')
            self.write(name, "CITATION.cff", f'version: "{version}"\n')
            if name == "gnn":
                self.pin(
                    "gnn", ".github/fep-lean-pair.json", "fep", self.revision("fep")
                )
                self.pin("gnn", ".github/gnn-pair.json", "geo", self.baseline)
            self.commit(name, "released source")
            git(self.roots[name], "tag", "-a", "v" + version, "-m", "accepted source")
        self.manifest = {
            "schema": "geo-infer-companion-releases/v1",
            "geo": {
                "repository": REPOSITORIES["geo"],
                "target_version": "0.4.0",
                "baseline_revision": self.baseline,
            },
            "companions": {
                name: self.release(name, version)
                for name, version in (("gnn", "5.0.1"), ("fep", "2.1.0"))
            },
            "interchange_schemas": list(INTERCHANGE_SCHEMAS),
            "evidence_boundaries": dict(EVIDENCE_BOUNDARIES),
        }
        self.pin("geo", ".github/gnn-pair.json", "gnn", self.revision("gnn"))
        self.save_manifest()

    def write(self, name, relative, text):
        """Write one fixture-owned path."""
        path = self.roots[name] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def pin(self, owner, relative, companion, revision):
        """Write a real two-field pin, never a synthetic validation result."""
        self.write(
            owner,
            relative,
            json.dumps({"repository": REPOSITORIES[companion], "revision": revision}),
        )

    def commit(self, name, message):
        """Commit all fixture changes and return their actual Git identity."""
        git(self.roots[name], "add", ".")
        git(self.roots[name], "commit", "--quiet", "-m", message)
        return self.revision(name)

    def revision(self, name):
        """Read the fixture's actual commit."""
        return git(self.roots[name], "rev-parse", "HEAD")

    def release(self, name, version):
        """Measure actual fixture tree and annotated tag objects."""
        root = self.roots[name]
        return {
            "repository": REPOSITORIES[name],
            "version": version,
            "tag": "v" + version,
            "revision": self.revision(name),
            "tree": git(root, "rev-parse", "HEAD^{tree}"),
            "tag_object": git(root, "rev-parse", "refs/tags/v" + version),
            "source_doi": "10.5281/zenodo.23631909"
            if name == "gnn"
            else "10.5281/zenodo.23631895",
        }

    def save_manifest(self):
        """Commit a changed fixture manifest and current source pin."""
        self.write("geo", MANIFEST_PATH, json.dumps(self.manifest))
        self.commit("geo", "record declared companion identities")

    def review(self, **kwargs):
        """Invoke the production reader on three real repositories."""
        return validate_companion_releases(
            self.roots["geo"], self.roots["gnn"], self.roots["fep"], **kwargs
        )

    def update_gnn_release(self):
        """Bind intentionally changed fixture content so deeper checks execute."""
        self.commit("gnn", "changed fixture source")
        git(
            self.roots["gnn"],
            "tag",
            "-f",
            "-a",
            "v5.0.1",
            "-m",
            "changed fixture source",
        )
        self.manifest["companions"]["gnn"] = self.release("gnn", "5.0.1")
        self.pin("geo", ".github/gnn-pair.json", "gnn", self.revision("gnn"))
        self.save_manifest()

    def test_valid_releases_preserve_asymmetric_pin_and_narrow_claims(self):
        result = self.review(expected_geo_revision=self.revision("geo"))
        self.assertEqual(result["status"], "passed_local_identity_and_pin_review")
        self.assertEqual(result["gnn_published_geo_baseline"], self.baseline)
        self.assertNotEqual(self.baseline, self.revision("geo"))
        self.assertTrue(result["source_and_input_bookends_equal"])
        self.assertFalse(result["claims"]["native_Lean_proof"])
        self.assertFalse(result["claims"]["public_bytes_or_DOI_metadata"])
        self.assertEqual(len(result["input_sha256"]), 10)

    def test_wrong_event_source_and_nonroot_or_alias_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "event revision"):
            self.review(expected_geo_revision=self.baseline)
        with self.assertRaisesRegex(ValueError, "checkout root"):
            validate_companion_releases(
                self.roots["geo"] / ".github", self.roots["gnn"], self.roots["fep"]
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_companion_releases(
                self.roots["geo"], self.roots["geo"], self.roots["fep"]
            )

    def test_dirty_companion_is_not_a_released_source(self):
        self.write("fep", "pyproject.toml", '[project]\nversion = "2.2.0"\n')
        with self.assertRaisesRegex(ValueError, "not clean"):
            self.review()

    def test_lightweight_and_reannotated_tags_are_rejected(self):
        root = self.roots["gnn"]
        git(root, "tag", "-d", "v5.0.1")
        git(root, "tag", "v5.0.1")
        with self.assertRaisesRegex(ValueError, "annotated"):
            self.review()
        git(root, "tag", "-f", "-a", "v5.0.1", "-m", "changed annotation only")
        with self.assertRaisesRegex(ValueError, "tag_object differs"):
            self.review()

    def test_wrong_tree_or_source_oid_cannot_reuse_a_tag(self):
        for field in ("revision", "tree", "tag_object"):
            original = self.manifest["companions"]["fep"][field]
            self.manifest["companions"]["fep"][field] = "0" * 40
            self.save_manifest()
            with self.assertRaisesRegex(ValueError, field + " differs"):
                self.review()
            self.manifest["companions"]["fep"][field] = original

    def test_actual_fep_pin_and_published_geo_baseline_are_required(self):
        self.pin("gnn", ".github/fep-lean-pair.json", "fep", "0" * 40)
        self.update_gnn_release()
        with self.assertRaisesRegex(ValueError, "gnn companion pin differs"):
            self.review()
        self.pin("gnn", ".github/fep-lean-pair.json", "fep", self.revision("fep"))
        self.pin("gnn", ".github/gnn-pair.json", "geo", self.revision("geo"))
        self.update_gnn_release()
        with self.assertRaisesRegex(ValueError, "gnn companion pin differs"):
            self.review()

    def test_version_metadata_cannot_be_relabelled_by_the_manifest(self):
        self.write("gnn", "pyproject.toml", '[project]\nversion = "5.0.2"\n')
        self.update_gnn_release()
        with self.assertRaisesRegex(ValueError, "project version differs"):
            self.review()

    def test_citation_version_is_checked_without_importing_companion_code(self):
        self.write("gnn", "CITATION.cff", 'version: "5.0.2"\n')
        self.update_gnn_release()
        with self.assertRaisesRegex(ValueError, "citation version differs"):
            self.review()

    def test_thin_cli_emits_actual_identity_or_a_bounded_refusal(self):
        module = Path(__file__).resolve().parents[2]
        environment = dict(os.environ, PYTHONPATH=str(module / "src"))
        command = [
            sys.executable,
            "-B",
            str(module / "validate_companion_releases.py"),
            "--geo-root",
            str(self.roots["geo"]),
            "--gnn-root",
            str(self.roots["gnn"]),
            "--fep-root",
            str(self.roots["fep"]),
            "--expected-geo-revision",
            self.revision("geo"),
        ]
        accepted = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(
            json.loads(accepted.stdout)["source"]["geo"]["revision"],
            self.revision("geo"),
        )
        self.write("fep", "uncommitted-input", "not a clean source")
        refused = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(refused.returncode, 1)
        self.assertEqual(
            json.loads(refused.stdout)["status"], "refused_local_identity_or_pin_review"
        )
        self.assertNotIn(self.temp.name, refused.stdout + refused.stderr)

    def test_disconnected_geo_history_cannot_claim_baseline_ancestry(self):
        git(self.roots["geo"], "checkout", "--orphan", "disconnected")
        self.commit("geo", "unrelated source with identical files")
        with self.assertRaisesRegex(ValueError, "baseline ancestry"):
            self.review()

    def test_extra_keys_schema_or_promoted_claims_fail_closed(self):
        self.manifest["companions"]["gnn"]["extra"] = True
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "keys"):
            self.review()
        del self.manifest["companions"]["gnn"]["extra"]
        self.manifest["interchange_schemas"].append("gnn-geo-infer/3")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "schema roster"):
            self.review()
        self.manifest["interchange_schemas"] = list(INTERCHANGE_SCHEMAS)
        self.manifest["evidence_boundaries"]["fep_proves_geo_implementation"] = True
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "unsupported evidence claim"):
            self.review()

    def test_duplicate_json_and_nonregular_or_oversized_manifests_fail(self):
        self.write("geo", MANIFEST_PATH, '{"schema":"first","schema":"second"}')
        self.commit("geo", "duplicate JSON keys")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.review()
        path = self.roots["geo"] / MANIFEST_PATH
        path.unlink()
        path.symlink_to("../pyproject.toml")
        self.commit("geo", "nonregular manifest")
        with self.assertRaisesRegex(ValueError, "nonregular"):
            self.review()
        path.unlink()
        self.write("geo", MANIFEST_PATH, " " * (128 * 1024 + 1))
        self.commit("geo", "oversized manifest")
        with self.assertRaisesRegex(ValueError, "oversized"):
            self.review()


if __name__ == "__main__":
    unittest.main()
