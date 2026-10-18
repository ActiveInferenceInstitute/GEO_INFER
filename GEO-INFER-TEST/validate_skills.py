#!/usr/bin/env python3
"""
Validate all SKILL.md files across the GEO-INFER ecosystem.

Checks:
  - Existence: every GEO-INFER-*/ directory must contain a SKILL.md
  - YAML frontmatter: must have `name` and `description` fields
  - Required sections: ## Instructions, ## Examples, ## Guidelines
  - Integration section: ### Integrations
  - Line count bounds: 30 ≤ lines ≤ 500
  - Naming convention: name must be `geo-infer-{module}` (lowercase)
  - Root SKILL.md: must exist at repository root

Usage:
    python GEO-INFER-TEST/validate_skills.py
    python GEO-INFER-TEST/validate_skills.py --verbose
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from geo_infer_test.documentation_contracts import skill_python_syntax_errors

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger("validate_skills")

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_LINES = 30
MAX_LINES = 500
REQUIRED_SECTIONS = ["## Instructions", "## Examples", "## Guidelines"]
REQUIRED_SUBSECTIONS = ["### Integrations"]
FRONTMATTER_FIELDS = ["name", "description"]
NAME_PATTERN = re.compile(r"^geo-infer(-[a-z0-9]+)?$")
SKILL_CLAIM_DEBT_PATTERN = re.compile(
    r"\b(mock|stub|fake|placeholder|planned|roadmap|coming soon|TODO|TBD)\b",
    re.IGNORECASE,
)
SKILL_CLAIM_ALLOWLIST = (
    "html form inputs use standard `placeholder` attributes",
    "sql uses parameterized queries",
    "students fill in the `pass` block",
    "`exercises.py` contains a `pass` inside a template string",
    "no mock/stub/placeholder code",
    "no planned, fake, mock, stub, or placeholder behavior",
    "planned work belongs in root",
    "`todo.md` or issues",
    "todo.md or issues",
    "do not add inert placeholders",
    "real, not stub",
    "zero mock implementations",
    "not placeholder",
    "not placeholders",
    "no placeholders",
    "former placeholder",
    "verified clean",
    "not fake",
    "not `ax.plot([0],[0])`",
    "not inert placeholders",
)


def parse_frontmatter(content: str) -> dict[str, str]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    fm_block = content[3:end].strip()
    result: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip().strip("'\"")
    return result


def find_module_dirs() -> list[Path]:
    """Find all GEO-INFER-* module directories."""
    dirs = sorted(
        d for d in REPO_ROOT.iterdir() if d.is_dir() and d.name.startswith("GEO-INFER-")
    )
    return dirs


def validate_skill_file(
    skill_path: Path,
    module_name: str,
    verbose: bool = False,
) -> list[str]:
    """Validate a single SKILL.md file. Returns list of error messages."""
    errors: list[str] = []
    label = f"[{module_name}]"

    # Existence
    if not skill_path.exists():
        errors.append(f"{label} SKILL.md not found at {skill_path}")
        return errors

    content = skill_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    line_count = len(lines)

    # Line count bounds
    if line_count < MIN_LINES:
        errors.append(f"{label} Too few lines: {line_count} (min {MIN_LINES})")
    if line_count > MAX_LINES:
        errors.append(f"{label} Too many lines: {line_count} (max {MAX_LINES})")

    # YAML frontmatter
    fm = parse_frontmatter(content)
    if not fm:
        errors.append(f"{label} No YAML frontmatter found")
    else:
        for field in FRONTMATTER_FIELDS:
            if field not in fm:
                errors.append(f"{label} Missing frontmatter field: {field}")
        # Name convention
        name_val = fm.get("name", "")
        if name_val and not NAME_PATTERN.match(name_val):
            errors.append(
                f"{label} Name '{name_val}' doesn't match pattern "
                f"'geo-infer-{{module}}'"
            )
        # Expected name
        expected_name = "geo-infer-" + module_name.lower().replace("geo-infer-", "")
        if name_val and name_val != expected_name:
            errors.append(
                f"{label} Name mismatch: got '{name_val}', expected '{expected_name}'"
            )

    # Required sections
    for section in REQUIRED_SECTIONS:
        if section not in content:
            errors.append(f"{label} Missing section: {section}")

    # Required subsections
    for subsection in REQUIRED_SUBSECTIONS:
        if subsection not in content:
            errors.append(f"{label} Missing subsection: {subsection}")

    # Operational skill examples are complete Python, unlike authored API
    # signature fragments; validate each explicit block independently.
    errors.extend(skill_python_syntax_errors(content, label, require_example=True))

    errors.extend(validate_skill_claim_language(content, label))

    if verbose and not errors:
        logger.info(f"  ✅ {label} OK ({line_count} lines)")

    return errors


def validate_root_skill(verbose: bool = False) -> list[str]:
    """Validate the root-level SKILL.md."""
    root_skill = REPO_ROOT / "SKILL.md"
    errors: list[str] = []
    label = "[ROOT]"

    if not root_skill.exists():
        errors.append(f"{label} Root SKILL.md not found")
        return errors

    content = root_skill.read_text(encoding="utf-8")
    lines = content.splitlines()

    fm = parse_frontmatter(content)
    if not fm:
        errors.append(f"{label} No YAML frontmatter found")
    elif fm.get("name") != "geo-infer":
        errors.append(
            f"{label} Root name should be 'geo-infer', got '{fm.get('name', '')}'"
        )

    for section in REQUIRED_SECTIONS:
        if section not in content:
            errors.append(f"{label} Missing section: {section}")

    errors.extend(skill_python_syntax_errors(content, label))
    errors.extend(validate_skill_claim_language(content, label))

    if verbose and not errors:
        logger.info(f"  ✅ {label} OK ({len(lines)} lines)")

    return errors


def validate_skill_claim_language(content: str, label: str) -> list[str]:
    """Reject stale or aspirational skill claims unless explicitly scoped."""
    errors: list[str] = []
    in_code_block = False
    for lineno, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if not SKILL_CLAIM_DEBT_PATTERN.search(line):
            continue
        normalized = line.lower()
        if any(allowed in normalized for allowed in SKILL_CLAIM_ALLOWLIST):
            continue
        errors.append(
            f"{label} Unscoped stale/planned skill claim at line {lineno}: {stripped}"
        )
    return errors


def validate_cross_references(verbose: bool = False) -> list[str]:
    """Check that README.md and AGENTS.md reference SKILL.md."""
    warnings: list[str] = []

    for module_dir in find_module_dirs():
        module = module_dir.name
        readme = module_dir / "README.md"
        agents = module_dir / "AGENTS.md"

        if readme.exists():
            content = readme.read_text(encoding="utf-8")
            if "SKILL.md" not in content and "SKILL" not in content:
                warnings.append(f"[{module}] README.md does not reference SKILL.md")

        if agents.exists():
            content = agents.read_text(encoding="utf-8")
            if "SKILL.md" not in content and "SKILL" not in content:
                warnings.append(f"[{module}] AGENTS.md does not reference SKILL.md")

    return warnings


def main() -> int:
    """Run all SKILL.md validations."""
    parser = argparse.ArgumentParser(
        description="Validate SKILL.md files across GEO-INFER"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show passing checks too",
    )
    parser.add_argument(
        "--check-xrefs",
        action="store_true",
        help="Also check README/AGENTS cross-references "
        "(advisory unless --warnings-fatal is set)",
    )
    parser.add_argument(
        "--warnings-fatal",
        action="store_true",
        help="Fail when cross-reference warnings are present, so broken "
        "README/AGENTS<->SKILL links can fail CI.",
    )
    args = parser.parse_args()

    logger.info("🔍 Validating SKILL.md files across GEO-INFER...")
    logger.info(f"   Repository root: {REPO_ROOT}")
    logger.info("")

    all_errors: list[str] = []
    all_warnings: list[str] = []
    module_dirs = find_module_dirs()

    # Validate root
    root_errors = validate_root_skill(verbose=args.verbose)
    all_errors.extend(root_errors)

    # Validate each module
    for module_dir in module_dirs:
        skill_path = module_dir / "SKILL.md"
        module_name = module_dir.name.lower()
        errors = validate_skill_file(skill_path, module_name, verbose=args.verbose)
        all_errors.extend(errors)

    # Cross-reference check (optional)
    if args.check_xrefs:
        xref_warnings = validate_cross_references(verbose=args.verbose)
        all_warnings.extend(xref_warnings)

    # Summary
    logger.info("")
    total_files = 1 + len(module_dirs)  # root + modules
    passing = total_files - len(set(e.split("]")[0] + "]" for e in all_errors))

    if all_errors:
        logger.error(f"❌ {len(all_errors)} errors in {total_files} files:")
        for error in all_errors:
            logger.error(f"  • {error}")
    else:
        logger.info(f"✅ All {total_files} SKILL.md files pass validation")

    if all_warnings:
        logger.warning(f"⚠️  {len(all_warnings)} cross-reference warnings:")
        for warn in all_warnings:
            logger.warning(f"  • {warn}")

    logger.info("")
    logger.info(
        f"📊 Summary: {passing}/{total_files} passing, "
        f"{len(all_errors)} errors, {len(all_warnings)} warnings"
    )

    return 1 if (all_errors or (args.warnings_fatal and all_warnings)) else 0


if __name__ == "__main__":
    sys.exit(main())
