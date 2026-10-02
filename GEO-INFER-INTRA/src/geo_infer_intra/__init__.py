"""GEO-INFER-INTRA - Reproducible documentation previews and configuration utilities for the GEO-INFER ecosystem."""

from geo_infer_intra.core.documentation.visual_preview import (
    MODULE_PROFILES,
    SpatialPreviewArtifacts,
    generate_all_module_previews,
    generate_module_preview_suite,
    render_leaflet_html,
    render_png_card,
    render_svg_card,
)

__version__ = "0.4.0"
__author__ = "GEO-INFER Team"
__email__ = "info@geo-infer.org"
__license__ = "CC-BY-NC-SA-4.0"

__all__ = [
    "MODULE_PROFILES",
    "SpatialPreviewArtifacts",
    "generate_all_module_previews",
    "generate_module_preview_suite",
    "render_leaflet_html",
    "render_png_card",
    "render_svg_card",
]
