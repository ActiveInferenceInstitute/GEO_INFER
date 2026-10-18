"""Figure legibility and provenance tests.

A figure drawn larger than the LaTeX text block is scaled down at typeset
time and its type shrinks with it.  The 45-module inventory was drawn on a
13-inch canvas at 8.5pt and printed at 3.1pt, roughly half the ~6pt floor for
legible print.  These tests hold the drawn size to the printed size.
"""

from __future__ import annotations

import hashlib
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest
import yaml

LEGIBLE_POINT_FLOOR = 6.0


@pytest.fixture(scope="module")
def rendered_figures(generator: ModuleType, repo_inventory, tmp_path_factory):
    """The real figure set, generated once into a scratch directory."""
    output_dir = tmp_path_factory.mktemp("figures")
    specs = generator.generate_figures(repo_inventory, output_dir)
    return output_dir, specs


def _print_scale(generator: ModuleType, path: Path) -> float:
    """The scale LaTeX applies to fit ``path`` inside the text block."""
    width_in, height_in = generator._png_size_inches(path, generator.FIGURE_DPI)
    return min(
        generator.TEXT_BLOCK_WIDTH_IN / width_in,
        generator.MAX_FIGURE_HEIGHT_IN / height_in,
    )


class TestFigureLegibility:
    def test_abstract_panel_borders_are_inside_the_export_viewport(
        self, generator: ModuleType, repo_inventory
    ) -> None:
        from matplotlib.patches import FancyBboxPatch

        _matplotlib, plt = generator._import_matplotlib()
        fig = generator._draw_graphical_abstract(plt, repo_inventory, None)
        try:
            fig.canvas.draw()
            axis = fig.axes[0]
            panels = [
                patch for patch in axis.patches if isinstance(patch, FancyBboxPatch)
            ]
            assert len(panels) == 4
            for panel in panels:
                bounds = panel.get_window_extent(fig.canvas.get_renderer())
                half_stroke = panel.get_linewidth() * fig.dpi / 72 / 2
                assert bounds.x0 - half_stroke >= axis.bbox.x0
                assert bounds.x1 + half_stroke <= axis.bbox.x1
                assert bounds.y0 - half_stroke >= axis.bbox.y0
                assert bounds.y1 + half_stroke <= axis.bbox.y1
        finally:
            plt.close(fig)

    def test_every_figure_prints_at_least_eighty_percent_of_native_width(
        self, generator: ModuleType, rendered_figures
    ) -> None:
        output_dir, specs = rendered_figures
        for spec in specs:
            scale = _print_scale(generator, output_dir / spec.filename)
            assert scale >= 0.8, f"{spec.filename} prints at {scale:.3f} of native"

    def test_authored_type_stays_above_the_legibility_floor(
        self, generator: ModuleType, rendered_figures
    ) -> None:
        output_dir, specs = rendered_figures
        authored_points = 8.0
        for spec in specs:
            scale = _print_scale(generator, output_dir / spec.filename)
            effective = authored_points * scale
            assert effective >= LEGIBLE_POINT_FLOOR, (
                f"{spec.filename} sets {authored_points}pt type that prints at "
                f"{effective:.2f}pt"
            )

    def test_no_figure_exceeds_the_printable_box(
        self, generator: ModuleType, rendered_figures
    ) -> None:
        output_dir, specs = rendered_figures
        for spec in specs:
            width_in, height_in = generator._png_size_inches(
                output_dir / spec.filename, generator.FIGURE_DPI
            )
            assert width_in <= generator.TEXT_BLOCK_WIDTH_IN * 1.02
            assert height_in <= generator.MAX_FIGURE_HEIGHT_IN * 1.02

    def test_an_oversized_figure_fails_the_build(
        self, generator: ModuleType, tmp_path: Path
    ) -> None:
        _matplotlib, plt = generator._import_matplotlib()
        fig = plt.figure(figsize=(13, 11))
        try:
            with pytest.raises(ValueError, match="printable box"):
                generator._save_figure(fig, tmp_path / "huge.png", "caption", "hash")
        finally:
            plt.close(fig)

    def test_the_preamble_keeps_half_height_floats_off_pages_of_their_own(
        self, repo_root: Path
    ) -> None:
        # LaTeX opens a float page for anything taller than \topfraction and
        # lets a float claim one once it fills \floatpagefraction. At the
        # defaults (0.7 / 0.5) the two half-height evidence figures each took a
        # page carrying ~300 characters of caption and nothing else.
        preamble = (repo_root / "manuscript" / "preamble.md").read_text("utf-8")
        for command, value in (
            ("topfraction", "0.9"),
            ("bottomfraction", "0.9"),
            ("textfraction", "0.08"),
            ("floatpagefraction", "0.85"),
        ):
            assert f"\\renewcommand{{\\{command}}}{{{value}}}" in preamble, (
                f"preamble.md no longer sets \\{command}"
            )

    def test_no_figure_would_be_granted_a_float_page(
        self, generator: ModuleType, rendered_figures
    ) -> None:
        # \floatpagefraction is measured against the whole float, image plus
        # caption.  Asserting the preamble sets it proves nothing about the
        # figures drawn against it: the two smaller figures cleared 0.85 and
        # the module inventory, at 0.90 of the text block, did not.
        output_dir, specs = rendered_figures
        limit = generator.TEXT_BLOCK_HEIGHT_IN * generator.FLOAT_PAGE_FRACTION
        for spec in specs:
            _width, height = generator._png_size_inches(
                output_dir / spec.filename, generator.FIGURE_DPI
            )
            float_height = height + generator.FIGURE_CAPTION_HEIGHT_IN
            assert float_height <= limit, (
                f"{spec.filename} would be a {float_height:.2f}in float against "
                f"a {limit:.2f}in threshold"
            )

    def test_the_guard_refuses_a_figure_that_would_claim_a_page(
        self, generator: ModuleType
    ) -> None:
        limit = generator.TEXT_BLOCK_HEIGHT_IN * generator.FLOAT_PAGE_FRACTION
        generator._assert_leaves_room_for_text(
            limit - generator.FIGURE_CAPTION_HEIGHT_IN, "fits.png"
        )
        with pytest.raises(ValueError, match="page of its own"):
            generator._assert_leaves_room_for_text(
                limit - generator.FIGURE_CAPTION_HEIGHT_IN + 0.01, "tall.png"
            )

    def test_the_float_page_bound_matches_the_preamble(
        self, generator: ModuleType, repo_root: Path
    ) -> None:
        # The generator refuses a figure the preamble would send to a float
        # page, so the two have to name the same fraction.
        preamble = (repo_root / "manuscript" / "preamble.md").read_text("utf-8")
        assert (
            f"\\renewcommand{{\\floatpagefraction}}{{{generator.FLOAT_PAGE_FRACTION}}}"
            in preamble
        )

    def test_the_height_bound_matches_the_render_config(
        self, generator: ModuleType, repo_root: Path
    ) -> None:
        config = yaml.safe_load(
            (repo_root / "manuscript" / "config.yaml").read_text(encoding="utf-8")
        )
        rendering = config["rendering"]
        assert float(rendering["figure_height_fraction"]) == pytest.approx(
            generator.FIGURE_HEIGHT_FRACTION
        )
        assert float(rendering["front_matter_figure_height_fraction"]) == pytest.approx(
            generator.FIGURE_HEIGHT_FRACTION
        )


class TestFigureProvenance:
    def test_vector_and_data_exports_are_accessible_and_reconcile_to_inventory(
        self, repo_inventory, rendered_figures
    ) -> None:
        output_dir, specs = rendered_figures
        for spec in specs:
            assert {item.media_type for item in spec.artifacts} == {
                "image/svg+xml",
                "application/json",
                "text/csv",
            }
            for item in spec.artifacts:
                assert (
                    item.sha256
                    == hashlib.sha256(
                        (output_dir / item.filename).read_bytes()
                    ).hexdigest()
                )
            vector = ET.parse(output_dir / Path(spec.filename).with_suffix(".svg"))
            text = " ".join(vector.getroot().itertext())
            assert spec.alt_text in text
            assert vector.findall(".//{http://www.w3.org/2000/svg}text")
            data = json.loads(
                (output_dir / Path(spec.filename).with_suffix(".json")).read_text(
                    "utf-8"
                )
            )
            assert data["source_hash"] == repo_inventory.source_hash
            assert data["figure_label"] == spec.label
            with (output_dir / Path(spec.filename).with_suffix(".csv")).open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == sum(len(table) for table in data["tables"].values())
            assert {row["source_hash"] for row in rows} == {repo_inventory.source_hash}
            if spec.label == "fig:module_inventory":
                measured = {
                    module.name: (module.source_files, module.test_files)
                    for module in repo_inventory.modules
                }
                assert {
                    row["module"]: (row["source_files"], row["test_files"])
                    for row in data["tables"]["modules"]
                } == measured
                assert (
                    sum(int(row["source_files"]) for row in rows)
                    == repo_inventory.source_files
                )

    @pytest.mark.parametrize("extension", [".png", ".svg", ".json", ".csv"])
    def test_registry_rejects_corrupted_delivered_bytes(
        self,
        generator: ModuleType,
        repo_inventory,
        rendered_figures,
        tmp_path: Path,
        extension: str,
    ) -> None:
        output_dir, specs = rendered_figures
        spec = specs[0]
        for name in [spec.filename, *(item.filename for item in spec.artifacts)]:
            shutil.copyfile(output_dir / name, tmp_path / name)
        target = tmp_path / Path(spec.filename).with_suffix(extension)
        target.write_bytes(target.read_bytes() + b"corruption")
        with pytest.raises(ValueError, match="digest disagrees"):
            generator.write_figure_registry(
                tmp_path / "registry.json", [spec], repo_inventory
            )

    def test_registry_rejects_external_or_linked_artifacts(
        self,
        generator: ModuleType,
        repo_inventory,
        rendered_figures,
        tmp_path: Path,
    ) -> None:
        import dataclasses

        output_dir, specs = rendered_figures
        spec = specs[0]
        with pytest.raises(ValueError, match="unsafe artifact filename"):
            generator.write_figure_registry(
                tmp_path / "registry.json",
                [dataclasses.replace(spec, filename="../outside.png")],
                repo_inventory,
            )
        (tmp_path / spec.filename).symlink_to(output_dir / spec.filename)
        with pytest.raises(ValueError, match="regular file"):
            generator.write_figure_registry(
                tmp_path / "registry.json", [spec], repo_inventory
            )

    def test_failed_export_closes_only_the_owned_figure(
        self,
        generator: ModuleType,
        repo_inventory,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        _matplotlib, plt = generator._import_matplotlib()
        caller = plt.figure()
        before = set(plt.get_fignums())

        def fail(*args, **kwargs):
            raise OSError("export failed")

        monkeypatch.setattr(generator, "_save_figure", fail)
        try:
            with pytest.raises(OSError, match="export failed"):
                generator.generate_figures(repo_inventory, tmp_path)
            assert set(plt.get_fignums()) == before
        finally:
            plt.close(caller)

    def test_each_spec_carries_the_digest_of_its_own_bytes(
        self, rendered_figures
    ) -> None:
        output_dir, specs = rendered_figures
        for spec in specs:
            written = (output_dir / spec.filename).read_bytes()
            assert spec.sha256 == hashlib.sha256(written).hexdigest()

    def test_registry_publishes_a_digest_per_figure(
        self, generator: ModuleType, repo_inventory, rendered_figures
    ) -> None:
        import json

        output_dir, specs = rendered_figures
        registry_path = output_dir / "figure_registry.json"
        generator.write_figure_registry(registry_path, specs, repo_inventory)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        assert registry["source_hash"] == repo_inventory.source_hash
        digests = {entry["filename"]: entry["sha256"] for entry in registry["figures"]}
        assert digests == {spec.filename: spec.sha256 for spec in specs}
        assert len(set(digests.values())) == len(digests)

    def test_registry_refuses_a_figure_without_a_digest(
        self, generator: ModuleType, repo_inventory, rendered_figures
    ) -> None:
        import dataclasses

        output_dir, specs = rendered_figures
        undigested = tuple(dataclasses.replace(spec, sha256="") for spec in specs[:1])
        with pytest.raises(ValueError, match="no content digest"):
            generator.write_figure_registry(
                output_dir / "broken_registry.json", undigested, repo_inventory
            )
