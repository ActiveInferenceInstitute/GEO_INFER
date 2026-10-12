"""Inventory-measurement tests for the manuscript generator.

Each published count is checked against an independent derivation, so a
classifier that structurally cannot see what it claims to count fails here
rather than in the PDF.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest


def test_generator_rejects_catalog_drift(generator, tmp_path):
    """Generation checks its measured inventory before accepting narrative counts."""
    sections = tmp_path / "manuscript" / "sections"
    sections.mkdir(parents=True)
    (sections / "space.md").write_text("The module has 7 test files.")
    inventory = SimpleNamespace(
        modules=[SimpleNamespace(name="GEO-INFER-SPACE", test_files=2)]
    )
    with pytest.raises(ValueError, match="measured count is 2"):
        generator.audit_module_census(tmp_path, inventory)
    (sections / "space.md").write_text("See the generated inventory table.")
    generator.audit_module_census(tmp_path, inventory)


def _find_h3_test_files(repo_root: Path) -> int:
    """Count H3-named test files with find(1), independently of the generator."""
    completed = subprocess.run(
        [
            "find",
            *(str(path) for path in sorted(repo_root.glob("GEO-INFER-*/tests"))),
            "-name",
            "test_*h3*.py",
            "-not",
            "-path",
            "*/__pycache__/*",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(bool(line.strip()) for line in completed.stdout.splitlines())


class TestH3TestFiles:
    """H3 is a subject tag, not a directory category."""

    def test_h3_files_exist_and_are_counted(
        self, generator: ModuleType, repo_root: Path, repo_inventory
    ) -> None:
        expected = _find_h3_test_files(repo_root)
        assert expected > 0, "fixture assumption: this checkout has H3 test files"
        assert repo_inventory.h3_test_files == expected

    def test_published_token_is_nonzero_for_this_checkout(
        self, generator: ModuleType, repo_inventory, figure_specs
    ) -> None:
        variables = generator.build_variables(
            repo_inventory,
            figure_specs,
            generator.VerificationRecord.unmeasured(),
        )
        assert int(variables["H3_TEST_FILE_COUNT"]) > 0

    def test_h3_is_not_a_directory_category(
        self, generator: ModuleType, repo_inventory
    ) -> None:
        # The old classifier put "h3" on the mutually-exclusive axis, where it
        # was unreachable for any test living under unit/ or integration/.
        assert "h3" not in repo_inventory.test_files_by_category
        assert set(generator.TEST_CATEGORIES) == {
            "unit",
            "integration",
            "performance",
            "other",
        }

    def test_h3_tests_are_also_counted_in_their_directory_category(
        self, generator: ModuleType, repo_root: Path, repo_inventory
    ) -> None:
        # Orthogonal means H3 files still appear in the distribution total.
        assert repo_inventory.h3_test_files <= repo_inventory.test_files


class TestCategoryDistribution:
    """The published parts must sum to the published total."""

    def test_named_categories_sum_to_the_total(
        self, generator: ModuleType, repo_inventory, figure_specs
    ) -> None:
        variables = generator.build_variables(
            repo_inventory,
            figure_specs,
            generator.VerificationRecord.unmeasured(),
        )
        parts = sum(
            int(variables[f"{name.upper()}_TEST_FILE_COUNT"])
            for name in generator.TEST_CATEGORIES
        )
        assert parts == repo_inventory.test_files
        assert variables["TEST_FILE_COUNT"] == f"{repo_inventory.test_files:,}"

    def test_other_bucket_is_published(
        self, generator: ModuleType, repo_inventory, figure_specs
    ) -> None:
        variables = generator.build_variables(
            repo_inventory,
            figure_specs,
            generator.VerificationRecord.unmeasured(),
        )
        assert variables["OTHER_TEST_FILE_COUNT"] == str(
            repo_inventory.test_files_by_category.get("other", 0)
        )

    def test_a_non_summing_distribution_fails_the_build(
        self, generator: ModuleType, repo_inventory, figure_specs
    ) -> None:
        import dataclasses

        import pytest

        broken = dataclasses.replace(
            repo_inventory, test_files=repo_inventory.test_files + 1
        )
        with pytest.raises(ValueError, match="distribution sums to"):
            generator.build_variables(
                broken,
                figure_specs,
                generator.VerificationRecord.unmeasured(),
            )

    def test_an_unnamed_category_fails_the_build(
        self, generator: ModuleType, repo_inventory, figure_specs
    ) -> None:
        import dataclasses

        import pytest

        distribution = dict(repo_inventory.test_files_by_category)
        distribution["h3"] = 0
        broken = dataclasses.replace(
            repo_inventory, test_files_by_category=distribution
        )
        with pytest.raises(ValueError, match="unnamed"):
            generator.build_variables(
                broken,
                figure_specs,
                generator.VerificationRecord.unmeasured(),
            )
