"""Execute maintained documentation examples with immutable attempt receipts."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import uuid

from . import execution

PYTHON_FENCE = re.compile(
    r"^```(?:python|py)\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL
)


def load_manifest(repository: Path, manifest: Path) -> list[Path]:
    """Require explicit, unique, repository-contained pages; never infer skips."""
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Documentation example manifest must be a nonempty list")
    pages = []
    for entry in entries:
        if not isinstance(entry, str) or Path(entry).is_absolute():
            raise ValueError("Manifest entries must be relative path strings")
        page = (repository / entry).resolve()
        if (
            not page.is_relative_to(repository.resolve())
            or page.suffix != ".md"
            or not page.is_file()
        ):
            raise ValueError(f"Invalid documentation page: {entry}")
        pages.append(page)
    if len(set(pages)) != len(pages):
        raise ValueError("Documentation example manifest contains duplicates")
    return pages


def example_code(page: Path) -> str:
    """Require executable Python and an assertion-bearing example on each page."""
    text = page.read_text()
    if "Illustrative example notice" in text:
        raise ValueError(f"Obsolete example exemption: {page}")
    blocks = PYTHON_FENCE.findall(text)
    if not blocks:
        raise ValueError(f"No Python examples in maintained page: {page}")
    code = "\n\n".join(blocks)
    tree = ast.parse(code, filename=str(page))
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        raise ValueError(f"Example needs an explicit acceptance assertion: {page}")
    return code


def verify_page(page: Path, *, timeout: float) -> execution.CommandResult:
    """Execute a page in a fresh temporary directory with no automatic retries."""
    code = example_code(page)
    token = uuid.uuid4().hex
    relative = str(page.relative_to(execution.PROJECT_ROOT))
    completion = {"status": "ok", "completion_token": token, "page": relative}
    script = (
        "import faulthandler,json\n"
        f"faulthandler.dump_traceback_later({min(timeout / 2, 60)!r})\n"
        f"exec(compile({code!r}, {relative!r}, 'exec'))\n"
        "faulthandler.cancel_dump_traceback_later()\n"
        f"print(json.dumps({completion!r}))\n"
    )
    environment = {
        name: "1"
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
        )
    }
    with tempfile.TemporaryDirectory(prefix="geo-doc-example-") as temporary:
        return execution.run_command(
            [sys.executable, "-I", "-c", script],
            "documentation example: " + relative,
            timeout,
            cwd=Path(temporary),
            env_overrides=environment,
            completion_token=token,
        )


def main(argv: list[str] | None = None) -> int:
    """Run the maintained manifest, rejecting empty or unverifiable examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=execution.PROJECT_ROOT / "GEO-INFER-TEST" / "doc_examples.json",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="Monotonic deadline per page, in seconds",
    )
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if args.results_dir:
        execution.RESULTS_DIR = args.results_dir.resolve()
    try:
        pages = load_manifest(execution.PROJECT_ROOT, args.manifest)
        # Validate the entire manifest before executing any code.
        for page in pages:
            example_code(page)
    except (OSError, ValueError, SyntaxError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    report = execution.SuiteReport()
    for page in pages:
        report.add(verify_page(page, timeout=args.timeout))
    execution.write_summary(report, show_failures=True)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
