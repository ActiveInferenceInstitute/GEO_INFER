"""Template render-override hook: delegate to the project's PDF renderer.

The template pipeline discovers ``scripts/_render_pdf_override.py`` and
transfers control to it instead of running its generic render path.  All
logic lives in ``render_manuscript_pdf.py``; this wrapper only forwards the
exit code so the pipeline's success/failure semantics are preserved.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    renderer = Path(__file__).with_name("render_manuscript_pdf.py")
    result = subprocess.run(
        [sys.executable, str(renderer)],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
