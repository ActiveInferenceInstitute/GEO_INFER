# GEO-INFER-ART API Specification

Reference for the real public API. All classes below are importable from
`geo_infer_art` (package level) or their subpackages and are covered by unit
tests. This document replaces an earlier draft that described classes
(`CartographicDesigner`, `GeoVisualizer`, `GenerativeArtist`) which never
existed in the codebase.

## Core Classes

### GeoArt — `geo_infer_art.core.visualization.geo_art`

Artistic visualization of geospatial data (vector or raster).

```python
GeoArt(data: GeoDataFrame | np.ndarray | None = None,
       metadata: dict | None = None,
       crs: str | None = "EPSG:4326")

GeoArt.load_geojson(file_path: str) -> GeoArt
GeoArt.load_raster(file_path: str) -> GeoArt
apply_style(style="default", color_palette=None, line_width=1.0, alpha=0.8,
            background_color="white", map_style=None, legend=False,
            title=None) -> GeoArt
create_animation(output_path: str, style_sequence: list[str], ...) -> str
add_interactive_elements(interactive_type: str = "zoom") -> GeoArt
create_interactive_web_map(output_file="interactive_map.html", ...)   # folium
create_plotly_visualization(plot_type: str = "scatter", **kwargs)     # plotly
create_3d_visualization(...)                                          # mayavi/plotly
save(output_path: str, dpi: int = 300) -> str
show() -> None
```

### MapStyle — `geo_infer_art.core.visualization.map_styling`

Advanced map styling and theming (works standalone or with `GeoArt.apply_style`).

```python
MapStyle.create_themed_style(theme: str,
                             color_palette: str | ColorPalette | None = None,
                             ...) -> MapStyle
apply_to_axes(ax, data_bounds=None) -> None
get_colormap() -> LinearSegmentedColormap
get_color_list() -> list[str]
get_background_color() -> str
```

### ColorPalette — `geo_infer_art.core.aesthetics.color_palette`

Predefined and derived color palettes. Predefined names: `viridis`, `pastel`,
`earth`, `bright`, `grayscale`, `blue`, `autumn`, `sunset`, `ocean`, `forest`.

```python
ColorPalette.get_palette(name: str) -> ColorPalette
ColorPalette.from_color_theory(base_color: str, scheme="complementary") -> ColorPalette
ColorPalette.from_image(image_path: str, n_colors: int = 6) -> ColorPalette
invert() / blend_with(other, ratio=0.5) / create_gradient(n_colors=256)
adjust_brightness(factor) / adjust_saturation(factor)
analyze_harmony() -> dict
to_css() -> str
```

### StyleTransfer — `geo_infer_art.core.aesthetics.style_transfer`

Neural style transfer (VGG-19) on geospatial data. Requires `tensorflow`
(`neural` extra); raises a clear `ImportError` when absent.

```python
StyleTransfer.apply(geo_data, style="watercolor", content_image=None,
                    style_image=None, style_weight=1e-2, content_weight=1e4,
                    iterations=100, color_palette=None) -> PIL.Image.Image
StyleTransfer.blend_styles(geo_data, styles: list[str], ...) -> PIL.Image.Image
StyleTransfer.create_style_variation(geo_data, base_style: str, ...)
get_style_info(style_name: str) -> dict
save(image: PIL.Image.Image, output_path: str) -> str
```

### GenerativeMap — `geo_infer_art.core.generation.generative_map`

Generative art from geospatial data (elevation-driven abstract maps).

```python
GenerativeMap.from_elevation(region, resolution: int = 512, ...) -> GenerativeMap
create_animation(output_path: str, parameter_sweep: str, ...) -> str
apply_texture(texture_type: str = "noise", **kwargs) -> GenerativeMap
blend_with(other_map: GenerativeMap, alpha: float = 0.5) -> GenerativeMap
add_effects(effects: list[str], **kwargs) -> GenerativeMap
export_multi_format(base_path: str, formats: list[str] | None = None) -> list[str]
save(output_path: str) -> str
```

### ProceduralArt — `geo_infer_art.core.generation.procedural_art`

Rule-based procedural art from 23 algorithms: `l_system`,
`cellular_automata`, `reaction_diffusion`, `noise_field`, `voronoi`,
`fractal_tree`, `mandelbrot`, `julia_set`, `perlin_noise`, `simplex_noise`,
`wave_function_collapse`, `marching_squares`, `space_colonization`, `boids`,
`particle_system`, `diffusion_limited_aggregation`, `turtle_graphics`,
`sierpinski`, `dragon_curve`, `hilbert_curve`, `koch_snowflake`,
`barnsley_fern`, `ifs_fractal` (see `ProceduralArt.ALGORITHMS`).

```python
ProceduralArt(algorithm: str = "noise_field",
              params: dict | None = None,
              resolution: tuple[int, int] = (800, 800))
ProceduralArt.from_geo_coordinates(lat: float, lon: float,
                                   algorithm="noise_field",
                                   additional_params=None) -> ProceduralArt
ProceduralArt.from_geo_features(feature_type: str, feature_count: int,
                                algorithm="l_system",
                                additional_params=None) -> ProceduralArt
generate() -> ProceduralArt   # dispatches to _generate_<algorithm>
save(output_path: str) -> str
show() -> None
```

### PlaceArt — `geo_infer_art.core.place.place_art`

Art generated from place characteristics.

```python
PlaceArt.from_coordinates(lat: float, lon: float, style: str = "abstract", ...)
PlaceArt.from_place_name(place_name: str, style: str = "abstract", ...)
add_metadata_overlay(position="bottom", opacity=0.7) -> PlaceArt
create_series(styles: list[str], output_dir="output") -> list[str]
save(output_path: str) -> str
```

### CulturalMap — `geo_infer_art.core.place.cultural_map`

Maps integrating cultural and historical context.

```python
CulturalMap.from_region(region_name: str, cultural_theme: str = "historical", ...)
CulturalMap.from_coordinates(lat: float, lon: float, ...)
add_narrative(narrative: str, position="bottom") -> CulturalMap
apply_cultural_style(style: str = "artistic") -> CulturalMap
create_timeline_view(time_periods: list[str]) -> list[CulturalMap]
save(output_path: str) -> str
```

### CustomAlgorithmFramework — `geo_infer_art.core.generation.custom_algorithms`

Register and execute user-defined procedural algorithms.

Persistence uses JSON schema version 1 containing trusted registry keys and
metadata. Construct with `CustomAlgorithmFramework(algorithm_registry={"key":
callable})` before loading custom keys. Builtin keys are `spiral`,
`cellular_growth` and `fractal_landscape`. Files cannot define or import Python
code; legacy `source`/`function_name` files must be rebuilt from trusted
application registrations. Loading validates the entire document before
changing registrations, and saving rejects callables absent from the registry.

```python
register_algorithm(name: str, algorithm_function: Callable,
                   description: str, parameters: dict, ...) -> None
unregister_algorithm(name: str) -> None
list_algorithms() -> list[str]
get_algorithm_info(name: str) -> dict
execute_algorithm(name: str, data: Any, width: int, height: int,
                  **params: Any) -> Any
save_algorithms_to_file(filepath: str) -> None
load_algorithms_from_file(filepath: str) -> None
```

### PerformanceOptimizer — `geo_infer_art.core.generation.performance_optimizer`

Caching, parallel execution, benchmarking, and adaptive resolution search.

```python
cached_execution(func, args=(), kwargs=None) -> Any
parallel_execution(func, parameter_sets: list[dict], max_workers=None) -> list
benchmark_function(func, args=(), iterations: int = 5) -> dict
optimize_resolution(target_time: float = 1.0, min_resolution: int = 100, ...) -> int
memory_efficient_processing(data: np.ndarray, chunk_size: int = 1000) -> np.ndarray
create_performance_report() -> dict
```

## Optional Backends

| Extra | Packages | Used by |
|-------|----------|---------|
| `neural` | tensorflow | `StyleTransfer` |
| `integrations` | plotly, folium, psutil | `GeoArt` web/plotly visualizations, memory detection |
| `viz3d` | mayavi (py<3.12) | `GeoArt.create_3d_visualization` |

All optional backends are imported in guarded try/except blocks and raise
informative errors when missing; core functionality works without them.

---

**Last Updated**: 2026-09-04
