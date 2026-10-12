"""Tests that the published source fingerprint fingerprints what ships.

``manuscript/config.yaml`` is both a generator-owned file and a hashed input.
Writing it after the digest was taken made ``RESEARCH_SOURCE_HASH``
irreproducible by the procedure its own glossary describes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def manuscript_tree(git_repo: Path) -> Path:
    """A checkout with the minimum shape the fingerprint helpers need."""
    manuscript = git_repo / "manuscript"
    manuscript.mkdir()
    (manuscript / "config.yaml").write_text(
        "paper:\n"
        '  version: "0.0.0"\n'
        '  date: "1970-01-01T00:00:00+00:00"\n'
        "publication:\n"
        '  year: "1970"\n'
        "metadata:\n"
        '  license: "unset"\n',
        encoding="utf-8",
    )
    (git_repo / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "1.2.3"\nlicense = "MIT"\n',
        encoding="utf-8",
    )
    return git_repo


class TestSourceHashCoversConfig:
    def test_config_is_inside_the_hashed_input_set(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        before = generator._source_hash(manuscript_tree)
        config = manuscript_tree / "manuscript" / "config.yaml"
        config.write_text(config.read_text(encoding="utf-8") + "# touched\n", "utf-8")
        assert generator._source_hash(manuscript_tree) != before

    def test_hash_taken_after_the_refresh_is_reproducible(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        values = generator.config_metadata_values(manuscript_tree)
        assert generator.refresh_config_metadata(manuscript_tree, values)
        digest = generator._source_hash(manuscript_tree)
        # Re-running the whole ordering must land on the same fingerprint.
        assert generator.refresh_config_metadata(manuscript_tree, values) == ()
        assert generator._source_hash(manuscript_tree) == digest

    def test_hash_taken_before_the_refresh_is_not_reproducible(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        # The defect, pinned: hashing first and writing second publishes a
        # fingerprint of a tree that no longer exists when the run ends.
        digest = generator._source_hash(manuscript_tree)
        generator.refresh_config_metadata(
            manuscript_tree, generator.config_metadata_values(manuscript_tree)
        )
        assert generator._source_hash(manuscript_tree) != digest

    def test_refresh_writes_measured_values(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        generator.refresh_config_metadata(
            manuscript_tree, generator.config_metadata_values(manuscript_tree)
        )
        text = (manuscript_tree / "manuscript" / "config.yaml").read_text("utf-8")
        assert '  version: "1.2.3"  # generator-owned (PROJECT_VERSION)' in text
        assert '  license: "MIT"  # generator-owned (PROJECT_LICENSE)' in text

    def test_dry_run_reports_without_writing(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        config = manuscript_tree / "manuscript" / "config.yaml"
        original = config.read_text(encoding="utf-8")
        stale = generator.refresh_config_metadata(
            manuscript_tree,
            generator.config_metadata_values(manuscript_tree),
            dry_run=True,
        )
        assert set(stale) == {
            "paper.version",
            "paper.date",
            "publication.year",
            "metadata.license",
        }
        assert config.read_text(encoding="utf-8") == original


def _publish(generator: ModuleType, root: Path, **overrides: str) -> None:
    """Write a published-variables file that matches ``root`` unless overridden."""
    generator.refresh_config_metadata(root, generator.config_metadata_values(root))
    published = dict(generator.config_metadata_values(root))
    published["RESEARCH_SOURCE_HASH"] = generator._source_hash(root)
    # A published bundle carries the verification summary too, and --check
    # compares it to the record on disk.  There is no record in these trees,
    # so the only coherent summary is "every defined group unrun".
    defined = generator.defined_command_groups(full_validation=False)
    published["VERIFICATION_DEFINED_COUNT"] = str(len(defined))
    published["VERIFICATION_PASS_COUNT"] = "0"
    published["VERIFICATION_FAIL_COUNT"] = "0"
    published["VERIFICATION_UNRUN_COUNT"] = str(len(defined))
    published["VERIFICATION_STATUS"] = "not run"
    published["VERIFICATION_RECORD_TIER"] = "default"
    published.update(overrides)
    data = root / "output" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "manuscript_variables.json").write_text(
        json.dumps(published), encoding="utf-8"
    )


class TestConfigDateIsSettleable:
    """Recording the refreshed config must not move the date it carries."""

    def _commit(self, root: Path, message: str, env: dict[str, str]) -> None:
        for args in (["add", "-A"], ["commit", "-q", "-m", message]):
            subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
                check=True,
                env=env,
                capture_output=True,
            )

    def test_committing_the_refreshed_config_reaches_a_fixed_point(
        self,
        generator: ModuleType,
        manuscript_tree: Path,
        git_environment: dict[str, str],
    ) -> None:
        self._commit(manuscript_tree, "add manuscript", git_environment)
        generator.refresh_config_metadata(
            manuscript_tree, generator.config_metadata_values(manuscript_tree)
        )
        self._commit(manuscript_tree, "record generated config", git_environment)
        # A second refresh after that commit must be a no-op, so a clean-tree
        # publication build is reachable.
        assert (
            generator.refresh_config_metadata(
                manuscript_tree, generator.config_metadata_values(manuscript_tree)
            )
            == ()
        )
        assert generator._dirty_file_count(manuscript_tree) == 0

    def test_head_date_would_not_settle(
        self,
        generator: ModuleType,
        manuscript_tree: Path,
        git_environment: dict[str, str],
    ) -> None:
        # Why the source date excludes config.yaml: HEAD's date always moves
        # when the refreshed config is recorded.
        self._commit(manuscript_tree, "add manuscript", git_environment)
        head_before = generator._run_git(manuscript_tree, "show", "-s", "--format=%cI")
        source_before = generator._manuscript_source_date(manuscript_tree)
        assert head_before == source_before
        (manuscript_tree / "manuscript" / "config.yaml").write_text(
            (manuscript_tree / "manuscript" / "config.yaml").read_text("utf-8")
            + "# recorded\n",
            encoding="utf-8",
        )
        self._commit(manuscript_tree, "record generated config", git_environment)
        assert generator._manuscript_source_date(manuscript_tree) == source_before


class TestCheckMode:
    def test_missing_published_variables_fail_closed(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        problems = generator.check_published_artifacts(manuscript_tree)
        assert any("no published variables" in problem for problem in problems)

    def test_stale_source_hash_is_detected(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        _publish(generator, manuscript_tree, RESEARCH_SOURCE_HASH="0000000000000000")
        problems = generator.check_published_artifacts(manuscript_tree)
        assert any("RESEARCH_SOURCE_HASH is stale" in p for p in problems)

    def test_a_title_page_that_disagrees_with_the_build_is_detected(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        # config.yaml supplies the title page. If it says one licence and the
        # evidence bundle beside it says another, the published artifact is
        # internally inconsistent.
        _publish(generator, manuscript_tree, PROJECT_LICENSE="Some-Other-Licence")
        problems = generator.check_published_artifacts(manuscript_tree)
        assert any("metadata.license" in problem for problem in problems)

    def test_published_variables_without_config_metadata_fail_closed(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        generator.refresh_config_metadata(
            manuscript_tree, generator.config_metadata_values(manuscript_tree)
        )
        data = manuscript_tree / "output" / "data"
        data.mkdir(parents=True)
        (data / "manuscript_variables.json").write_text(
            json.dumps(
                {"RESEARCH_SOURCE_HASH": generator._source_hash(manuscript_tree)}
            ),
            encoding="utf-8",
        )
        problems = generator.check_published_artifacts(manuscript_tree)
        assert any("omit config metadata" in problem for problem in problems)

    def test_a_matching_tree_reports_no_problems(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        _publish(generator, manuscript_tree)
        assert generator.check_published_artifacts(manuscript_tree) == ()

    def test_the_reference_is_the_published_build_not_head(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        # The tracked config is always recorded by a commit newer than the
        # commit date it holds, so comparing against HEAD could never pass.
        _publish(generator, manuscript_tree)
        (manuscript_tree / "later.txt").write_text("later\n", encoding="utf-8")
        assert generator.check_published_artifacts(manuscript_tree) == ()


class TestPublishedCountsAgainstRecord:
    """The published counts are compared to the record they summarise.

    The summary and the record are two files, and nothing compared them —
    that is how a bundle shipped "9 passed" beside an eleven-result record.
    ``--check`` now recomputes the counts from the record on disk, at the
    record's own tier, and refuses a bundle whose numbers disagree.
    """

    @staticmethod
    def _store_record(generator: ModuleType, root: Path) -> dict:
        """Write a one-passed-result record stamped with this tree."""
        inventory = generator.collect_inventory(root)
        name = generator.VERIFICATION_COMMANDS[0][0]
        record = generator.VerificationRecord(
            results=(
                generator.VerificationResult(
                    name=name,
                    command="run it",
                    status="passed",
                    return_code=0,
                    duration_seconds=1.0,
                    output_tail="",
                ),
            ),
            source_commit=inventory.commit,
            source_hash=inventory.source_hash,
            full_validation_requested=False,
        )
        payload = generator._verification_payload(record)
        path = generator._verification_record_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def test_counts_disagreeing_with_the_record_are_reported(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        self._store_record(generator, manuscript_tree)
        # _publish writes the coherent "no record" summary — every group
        # unrun — which disagrees with the one-passed-group record above.
        _publish(generator, manuscript_tree)
        problems = generator.check_published_artifacts(manuscript_tree)
        assert any(
            "disagrees with the stored verification record" in problem
            for problem in problems
        )

    def test_counts_matching_the_record_report_no_problem(
        self, generator: ModuleType, manuscript_tree: Path
    ) -> None:
        payload = self._store_record(generator, manuscript_tree)
        results = [
            generator.VerificationResult(
                name=entry["name"],
                command=entry["command"],
                status=entry["status"],
                return_code=entry["return_code"],
                duration_seconds=entry["duration_seconds"],
                output_tail=entry["output_tail"],
            )
            for entry in payload["results"]
        ]
        summary, passed, failed, unrun = generator._verification_summary(
            results, full_validation=False
        )
        _publish(
            generator,
            manuscript_tree,
            VERIFICATION_DEFINED_COUNT=str(
                len(generator.defined_command_groups(full_validation=False))
            ),
            VERIFICATION_PASS_COUNT=str(passed),
            VERIFICATION_FAIL_COUNT=str(failed),
            VERIFICATION_UNRUN_COUNT=str(unrun),
            VERIFICATION_STATUS=summary,
            VERIFICATION_RECORD_TIER="default",
            VERIFICATION_RECORD_COMMIT=payload["source_commit"],
            VERIFICATION_RECORD_SOURCE_HASH=payload["source_hash"],
        )
        assert generator.check_published_artifacts(manuscript_tree) == ()


class TestGitReadsStartNoDaemon:
    """Generator git reads must not leave Git's fsmonitor daemon behind.

    A user-level ``core.fsmonitor=true`` makes ordinary ``git status`` start a
    detached daemon that outlives the generator, and the test harness then
    reports it as a leaked owned background process.
    """

    def test_status_and_revision_reads_leave_no_watching_daemon(
        self,
        generator: ModuleType,
        git_repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        global_config = tmp_path / "user.gitconfig"
        global_config.write_text("[core]\n\tfsmonitor = true\n", encoding="utf-8")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
        status = ["git", "-c", "core.fsmonitor=false", "-C", str(git_repo)]
        try:
            assert generator._dirty_file_count(git_repo) == 0
            assert generator._run_git(git_repo, "rev-parse", "HEAD") != "unavailable"
            watching = subprocess.run(
                [*status, "fsmonitor--daemon", "status"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert watching.returncode != 0, watching.stdout
        finally:
            subprocess.run(
                [*status, "fsmonitor--daemon", "stop"],
                capture_output=True,
                check=False,
            )
