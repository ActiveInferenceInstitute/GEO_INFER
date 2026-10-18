"""Thin subprocess entrypoint for the explicitly selected template interpreter."""

from pathlib import Path
import shutil
import sys
import json
import platform
import importlib.metadata
import subprocess


def main() -> None:
    template, repo = (Path(value).resolve(strict=True) for value in sys.argv[1:3])
    runtime_path = Path(sys.argv[3])
    sys.path.insert(0, str(template))
    from infrastructure.rendering.config import RenderingConfig
    from infrastructure.rendering.pdf_renderer import PDFRenderer

    destination = repo / "output/pdf"
    # Keep the producer/section-combiner input unchanged when template writes
    # its own processed combined Markdown at the standard PDF destination.
    source = destination / "_template_input.md"
    shutil.copyfile(destination / "_combined_manuscript.md", source)
    shutil.copyfile(
        repo / "output/manuscript/references.bib", destination / "references.bib"
    )
    config = RenderingConfig(
        manuscript_dir=str(repo / "output/manuscript"),
        figures_dir=str(repo / "output/figures"),
        output_dir=str(repo / "output"),
        pdf_dir=str(destination),
    )
    result = PDFRenderer(config).render_combined(
        [source], repo / "output/manuscript", "GEO-INFER"
    )
    tools = {}
    for name in ("pandoc", "pandoc-crossref", "xelatex", "bibtex"):
        version = subprocess.run(
            [name, "--version"], capture_output=True, text=True, check=True, timeout=20
        )
        tools[name] = {
            "executable": shutil.which(name),
            "version": version.stdout.splitlines()[0],
        }
    runtime_path.write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "executable": sys.executable,
                "renderer_file": str(
                    Path(sys.modules[PDFRenderer.__module__].__file__).resolve()
                ),
                "tools": tools,
                "distributions": sorted(
                    (str(d.metadata["Name"]), d.version)
                    for d in importlib.metadata.distributions()
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
