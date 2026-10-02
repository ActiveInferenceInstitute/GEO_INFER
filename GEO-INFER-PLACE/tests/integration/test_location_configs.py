"""
Integration tests for GEO-INFER-PLACE locations subsystem.

Verifies:
- Location directory structure and required files
- README formatting (proper multi-line markdown)
- Requirements files contain only real pip-installable packages
- AGENTS.md files are location-specific (not generic stubs)
- Location registry completeness
"""

import logging
import tomllib
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCATIONS_DIR = PROJECT_ROOT / "locations"

# Expected locations
EXPECTED_LOCATIONS = [
    "australia",
    "cascadia",
    "del_norte_county",
    "del_norte_county_synthetic",
    "houston",
    "siberia",
]

# Known fabricated package prefixes that should NOT appear as real deps
FABRICATED_PACKAGE_PATTERNS = [
    "bom-weather",
    "ala-tools",
    "australia-maps",
    "permafrost-tools",
    "ground-temp",
    "active-layer",
    "arctic-climate",
    "sea-ice-analysis",
    "roshydromet-api",
    "calfire-api",
    "fire-weather-index",
    "lidar-processor",
    "forestsat",
    "tree-health",
    "coastal-erosion",
    "drought-indices",
    "species-distribution",
    "habitat-connectivity",
    "social-analysis",
    "demographic-tools",
    "survey-tools",
]


class TestLocationDirectoryStructure:
    """Verify that all expected location directories exist with required files."""

    def test_locations_dir_exists(self):
        assert LOCATIONS_DIR.is_dir(), (
            f"locations/ directory not found at {LOCATIONS_DIR}"
        )
        logger.info("locations/ directory found at %s", LOCATIONS_DIR)

    @pytest.mark.parametrize("location", EXPECTED_LOCATIONS)
    def test_location_dir_exists(self, location):
        loc_dir = LOCATIONS_DIR / location
        assert loc_dir.is_dir(), f"Location directory '{location}' not found"
        logger.info("Location directory '%s' exists", location)

    @pytest.mark.parametrize("location", EXPECTED_LOCATIONS)
    def test_readme_exists(self, location):
        readme = LOCATIONS_DIR / location / "README.md"
        assert readme.is_file(), f"README.md missing for {location}"
        content = readme.read_text(encoding="utf-8")
        assert len(content) > 100, (
            f"README.md for {location} is too short ({len(content)} bytes)"
        )
        logger.info("README.md for '%s' exists (%d bytes)", location, len(content))

    @pytest.mark.parametrize("location", EXPECTED_LOCATIONS)
    def test_agents_md_exists(self, location):
        agents = LOCATIONS_DIR / location / "AGENTS.md"
        assert agents.is_file(), f"AGENTS.md missing for {location}"
        content = agents.read_text(encoding="utf-8")
        assert len(content) > 50, f"AGENTS.md for {location} is too short"
        logger.info("AGENTS.md for '%s' exists (%d bytes)", location, len(content))


class TestREADMEFormatting:
    """Verify READMEs are properly formatted markdown, not single-line blobs."""

    @pytest.mark.parametrize("location", EXPECTED_LOCATIONS)
    def test_readme_has_multiple_lines(self, location):
        readme = LOCATIONS_DIR / location / "README.md"
        content = readme.read_text(encoding="utf-8")
        line_count = content.count("\n")
        assert line_count >= 20, (
            f"README.md for {location} has only {line_count} lines "
            "(likely still malformed single-line)"
        )
        logger.info("README.md for '%s' has %d lines", location, line_count)

    @pytest.mark.parametrize("location", EXPECTED_LOCATIONS)
    def test_readme_has_heading_structure(self, location):
        readme = LOCATIONS_DIR / location / "README.md"
        content = readme.read_text(encoding="utf-8")
        lines = content.splitlines()
        h1_count = sum(1 for line in lines if line.startswith("# "))
        h2_count = sum(1 for line in lines if line.startswith("## "))
        assert h1_count >= 1, f"README.md for {location} missing H1 heading"
        assert h2_count >= 2, f"README.md for {location} has fewer than 2 H2 sections"
        logger.info(
            "README.md for '%s' has %d H1 and %d H2 headings",
            location,
            h1_count,
            h2_count,
        )

    @pytest.mark.parametrize("location", EXPECTED_LOCATIONS)
    def test_readme_no_broken_headings(self, location):
        """Check for malformed headings like '### Geographi\\nc\\n Context'."""
        readme = LOCATIONS_DIR / location / "README.md"
        content = readme.read_text(encoding="utf-8")
        lines = content.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            # A line that is just a single lowercase letter is likely a broken heading
            if len(stripped) == 1 and stripped.isalpha() and stripped.islower():
                # Check if previous line ends with partial word
                if i > 0:
                    prev = lines[i - 1].rstrip()
                    if prev and prev[-1].isalpha():
                        pytest.fail(
                            f"Broken heading detected in {location}/README.md "
                            f"at line {i + 1}: '{prev}' / '{stripped}'"
                        )


class TestRequirementsFiles:
    """Verify requirements.txt files contain real pip-installable packages."""

    LOCATIONS_WITH_REQUIREMENTS = [
        "australia",
        "del_norte_county",
        "del_norte_county_synthetic",
        "siberia",
    ]

    def test_cascadia_dependencies_belong_to_place_extra(self):
        """Cascadia uses the owning distribution instead of a second toolchain."""
        cascadia = LOCATIONS_DIR / "cascadia"
        for name in (
            "requirements.txt",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
        ):
            assert not (cascadia / name).exists()
        project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
        dependencies = project["project"]["optional-dependencies"]["cascadia"]
        assert any(
            dep.startswith("geo-infer-space")
            for dep in project["project"]["dependencies"]
        )
        assert any(dep.startswith("datashader") for dep in dependencies)
        assert all(not dep.startswith("src") for dep in dependencies)

    @pytest.mark.parametrize("location", LOCATIONS_WITH_REQUIREMENTS)
    def test_requirements_exists(self, location):
        req_path = LOCATIONS_DIR / location / "requirements.txt"
        assert req_path.is_file(), f"requirements.txt missing for {location}"

    @pytest.mark.parametrize("location", LOCATIONS_WITH_REQUIREMENTS)
    def test_no_fabricated_packages(self, location):
        """Check that fabricated packages are not listed as real dependencies."""
        req_path = LOCATIONS_DIR / location / "requirements.txt"
        content = req_path.read_text(encoding="utf-8")
        active_deps = []
        for line in content.splitlines():
            stripped = line.strip()
            # Skip comments, empty lines, and commented-out aspirational packages
            if not stripped or stripped.startswith("#"):
                continue
            pkg_name = stripped.split(">=")[0].split("==")[0].split("[")[0].strip()
            active_deps.append(pkg_name)

        for dep in active_deps:
            assert dep not in FABRICATED_PACKAGE_PATTERNS, (
                f"Fabricated package '{dep}' found as active dependency "
                f"in {location}/requirements.txt — should be commented out"
            )
        logger.info(
            "requirements.txt for '%s' has %d active dependencies (no fabricated)",
            location,
            len(active_deps),
        )


class TestAGENTSMDQuality:
    """Verify AGENTS.md files are location-specific, not generic stubs."""

    GENERIC_STUB_PHRASES = [
        "No public classes or functions found",
        "This directory contains src components for the module",
    ]

    @pytest.mark.parametrize("location", EXPECTED_LOCATIONS)
    def test_agents_not_generic_stub(self, location):
        agents = LOCATIONS_DIR / location / "AGENTS.md"
        content = agents.read_text(encoding="utf-8")
        for phrase in self.GENERIC_STUB_PHRASES:
            assert phrase not in content, (
                f"AGENTS.md for {location} still contains generic stub phrase: "
                f"'{phrase}'"
            )
        logger.info("AGENTS.md for '%s' is location-specific", location)

    @pytest.mark.parametrize("location", EXPECTED_LOCATIONS)
    def test_agents_has_scope_section(self, location):
        agents = LOCATIONS_DIR / location / "AGENTS.md"
        content = agents.read_text(encoding="utf-8")
        assert "## Scope" in content, (
            f"AGENTS.md for {location} missing '## Scope' section"
        )

    @pytest.mark.parametrize("location", EXPECTED_LOCATIONS)
    def test_agents_has_capabilities(self, location):
        agents = LOCATIONS_DIR / location / "AGENTS.md"
        content = agents.read_text(encoding="utf-8")
        has_caps = "## Capabilities" in content or "## Key Modules" in content
        assert has_caps, (
            f"AGENTS.md for {location} missing '## Capabilities' or equivalent section"
        )


class TestLocationRegistry:
    """Verify the locations root README acts as a proper registry."""

    def test_root_readme_exists(self):
        readme = LOCATIONS_DIR / "README.md"
        assert readme.is_file()
        content = readme.read_text(encoding="utf-8")
        assert len(content) > 200

    def test_root_readme_lists_all_locations(self):
        readme = LOCATIONS_DIR / "README.md"
        content = readme.read_text(encoding="utf-8")
        for location in EXPECTED_LOCATIONS:
            assert location in content, (
                f"Location '{location}' not mentioned in locations/README.md"
            )
        logger.info("Root README references all %d locations", len(EXPECTED_LOCATIONS))

    def test_root_agents_exists(self):
        agents = LOCATIONS_DIR / "AGENTS.md"
        assert agents.is_file()
        content = agents.read_text(encoding="utf-8")
        assert "## Scope" in content
