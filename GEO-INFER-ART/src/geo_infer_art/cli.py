#!/usr/bin/env python
"""
Command-line interface for GEO-INFER-ART.
"""

from typing import Any
import argparse
import os
import sys

import numpy as np
from PIL import Image

from geo_infer_art import (
    GeoArt,
    StyleTransfer,
    GenerativeMap,
    ProceduralArt,
    PlaceArt,
    CulturalMap,
    CustomAlgorithmFramework,
    PerformanceOptimizer,
)


def ensure_directory(directory: str) -> None:
    """Create output directory if it doesn't exist."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def _generative_map_from_geo_art(geo_art: GeoArt) -> GenerativeMap:
    """Build a GenerativeMap for parameter sweeps from loaded GeoArt data."""
    if isinstance(geo_art.data, np.ndarray):
        raster = np.asarray(geo_art.data, dtype=float)
        if raster.ndim == 3:
            # Rasterio arrays are band-first (bands, height, width)
            raster = raster.mean(axis=0)
        return GenerativeMap(data=raster)
    bounds = tuple(geo_art.data.total_bounds)
    return GenerativeMap.from_elevation(region=bounds)


def process_geo_art(args: argparse.Namespace) -> int:
    """Process geospatial data with GeoArt."""
    print(f"Loading geospatial data from {args.input}...")

    # Load the data
    if args.input.lower().endswith((".geojson", ".json")):
        geo_art = GeoArt.load_geojson(args.input)
    elif args.input.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
        geo_art = GeoArt.load_raster(args.input)
    else:
        print(f"Error: Unsupported input file format: {args.input}")
        print("Supported formats: .geojson, .json, .tif, .tiff, .jpg, .png")
        return 1

    # Apply style
    print(f"Applying style: {args.style}")
    geo_art.apply_style(
        style=args.style,
        color_palette=args.color_palette,
        line_width=args.line_width,
        alpha=args.alpha,
        background_color=args.background_color,
        map_style=args.map_style,
        legend=args.legend,
        title=args.title,
    )

    # Save the result
    output_dir = os.path.dirname(args.output)
    ensure_directory(output_dir)

    print(f"Saving result to {args.output}...")
    geo_art.save(args.output, dpi=args.dpi)

    print("Done!")
    return 0


def process_style_transfer(args: argparse.Namespace) -> int:
    """Process style transfer on geospatial data."""
    print(f"Loading geospatial data from {args.input}...")

    # Load the data
    if args.input.lower().endswith((".geojson", ".json")):
        geo_data = GeoArt.load_geojson(args.input).data
    elif args.input.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
        geo_data = GeoArt.load_raster(args.input).data
    else:
        print(f"Error: Unsupported input file format: {args.input}")
        print("Supported formats: .geojson, .json, .tif, .tiff, .jpg, .png")
        return 1

    # Check if TensorFlow is available
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("Error: TensorFlow is required for style transfer")
        print("Install it with 'uv pip install tensorflow'")
        return 1

    # Apply style transfer
    print(f"Applying style transfer: {args.style_transfer}")
    try:
        styled_image = StyleTransfer.apply(
            geo_data=geo_data,
            style=args.style_transfer,
            iterations=args.iterations,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            color_palette=args.color_palette,
        )
    except Exception as e:
        print(f"Error during style transfer: {str(e)}")
        return 1

    # Save the result
    output_dir = os.path.dirname(args.output)
    ensure_directory(output_dir)

    print(f"Saving result to {args.output}...")
    styled_image.save(args.output)

    print("Done!")
    return 0


def process_place_art(args: argparse.Namespace) -> int:
    """Generate place-based art."""
    if args.place:
        print(f"Generating art for place: {args.place}")
        place_art = PlaceArt.from_place_name(place_name=args.place, style=args.style)
    elif args.coordinates:
        try:
            lat, lon = map(float, args.coordinates.split(","))
            print(f"Generating art for coordinates: {lat}, {lon}")
            place_art = PlaceArt.from_coordinates(lat=lat, lon=lon, style=args.style)
        except ValueError:
            print(
                "Error: Invalid coordinates format. Use 'lat,lon' (e.g., '40.7128,-74.0060')"
            )
            return 1
    else:
        print("Error: Either --place or --coordinates must be specified")
        return 1

    # Add metadata overlay if requested
    if args.add_metadata:
        place_art.add_metadata_overlay(position="bottom")

    # Save the result
    output_dir = os.path.dirname(args.output)
    ensure_directory(output_dir)

    print(f"Saving result to {args.output}...")
    place_art.save(args.output)

    print("Done!")
    return 0


def process_generative_map(args: argparse.Namespace) -> int:
    """Generate art from elevation data."""
    if args.region:
        print(f"Generating elevation art for region: {args.region}")
        gen_map = GenerativeMap.from_elevation(
            region=args.region,
            resolution=args.resolution,
            abstraction_level=args.abstraction_level,
            style=args.style,
        )
    elif args.bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, args.bbox.split(","))
            bbox = (min_lon, min_lat, max_lon, max_lat)
            print(f"Generating elevation art for bbox: {bbox}")
            gen_map = GenerativeMap.from_elevation(
                region=bbox,
                resolution=args.resolution,
                abstraction_level=args.abstraction_level,
                style=args.style,
            )
        except ValueError:
            print("Error: Invalid bbox format. Use 'min_lon,min_lat,max_lon,max_lat'")
            return 1
    else:
        print("Error: Either --region or --bbox must be specified")
        return 1

    # Save the result
    output_dir = os.path.dirname(args.output)
    ensure_directory(output_dir)

    print(f"Saving result to {args.output}...")
    gen_map.save(args.output)

    print("Done!")
    return 0


def process_procedural_art(args: argparse.Namespace) -> int:
    """Generate procedural art."""
    if args.coordinates:
        try:
            lat, lon = map(float, args.coordinates.split(","))
            print(f"Generating procedural art for coordinates: {lat}, {lon}")
            art = ProceduralArt.from_geo_coordinates(
                lat=lat,
                lon=lon,
                algorithm=args.algorithm,
                additional_params={"color_palette": args.color_palette},
            )
        except ValueError:
            print(
                "Error: Invalid coordinates format. Use 'lat,lon' (e.g., '40.7128,-74.0060')"
            )
            return 1
    elif args.feature_type:
        print(f"Generating procedural art for feature type: {args.feature_type}")
        art = ProceduralArt.from_geo_features(
            feature_type=args.feature_type,
            feature_count=args.feature_count,
            algorithm=args.algorithm,
            additional_params={"color_palette": args.color_palette},
        )
    else:
        print(f"Generating procedural art with algorithm: {args.algorithm}")
        art = ProceduralArt(
            algorithm=args.algorithm,
            params={"color_palette": args.color_palette},
            resolution=(args.resolution, args.resolution),
        )
        art.generate()

    # Save the result
    output_dir = os.path.dirname(args.output)
    ensure_directory(output_dir)

    print(f"Saving result to {args.output}...")
    art.save(args.output)

    print("Done!")
    return 0


def process_cultural_map(args: argparse.Namespace) -> int:
    """Generate cultural maps."""
    if args.region:
        print(f"Generating cultural map for region: {args.region}")
        cultural_map = CulturalMap.from_region(
            region_name=args.region,
            cultural_theme=args.cultural_theme,
            style=args.style,
        )
    elif args.coordinates:
        try:
            lat, lon = map(float, args.coordinates.split(","))
            print(f"Generating cultural map for coordinates: {lat}, {lon}")
            cultural_map = CulturalMap.from_coordinates(
                lat=lat,
                lon=lon,
                radius_km=args.radius_km,
                cultural_theme=args.cultural_theme,
                style=args.style,
            )
        except ValueError:
            print(
                "Error: Invalid coordinates format. Use 'lat,lon' (e.g., '40.7128,-74.0060')"
            )
            return 1
    else:
        print("Error: Either --region or --coordinates must be specified")
        return 1

    # Add narrative if provided
    if args.narrative:
        cultural_map.add_narrative(args.narrative)

    # Apply cultural style
    if args.apply_cultural_style:
        cultural_map.apply_cultural_style()

    # Save the result
    output_dir = os.path.dirname(args.output)
    ensure_directory(output_dir)

    print(f"Saving result to {args.output}...")
    cultural_map.save(args.output)

    print("Done!")
    return 0


def process_map_style(args: argparse.Namespace) -> int:
    """Process map style demonstration."""
    print(f"Creating MapStyle: {args.style}")

    # Create a sample GeoDataFrame for demonstration
    import geopandas as gpd
    from shapely.geometry import Polygon

    # Create sample data
    geometries = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
    ]

    data = gpd.GeoDataFrame(
        {"name": ["Region A", "Region B", "Region C"]},
        geometry=geometries,
        crs="EPSG:4326",
    )

    # Create GeoArt with MapStyle
    geo_art = GeoArt(data=data)

    # Create or use MapStyle
    if args.theme:
        from geo_infer_art.core.visualization.map_styling import MapStyle

        map_style = MapStyle.create_themed_style(
            args.theme, color_palette=args.color_palette
        )
        geo_art.apply_style(style="default", map_style=map_style)
    else:
        geo_art.apply_style(style=args.style)

    # Save the result
    output_dir = os.path.dirname(args.output)
    ensure_directory(output_dir)

    print(f"Saving result to {args.output}...")
    geo_art.save(args.output, dpi=300)

    print("Done!")
    return 0


def process_animation(args: argparse.Namespace) -> int:
    """Process animation creation."""
    print(f"Creating {args.animation_type} animation...")

    # Load the data
    if args.input.lower().endswith((".geojson", ".json")):
        geo_art = GeoArt.load_geojson(args.input)
    elif args.input.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
        geo_art = GeoArt.load_raster(args.input)
    else:
        print(f"Error: Unsupported input file format: {args.input}")
        return 1

    # Create animation based on type
    if args.animation_type == "style_cycle":
        if not args.styles:
            print("Error: --styles required for style_cycle animation")
            return 1

        print(f"Creating style cycle animation with styles: {args.styles}")
        output_path = geo_art.create_animation(
            output_path=args.output,
            style_sequence=args.styles,
            duration=args.duration,
            fps=args.fps,
        )

    elif args.animation_type == "parameter_sweep":
        if not args.parameter or not args.values:
            print(
                "Error: --parameter and --values required for parameter_sweep animation"
            )
            return 1

        print(f"Creating parameter sweep animation for {args.parameter}")

        try:
            sweep_values: list[float | str] = (
                [float(v) for v in args.values]
                if args.parameter == "abstraction_level"
                else [str(v) for v in args.values]
            )
        except ValueError:
            print("Error: --values must be numeric for abstraction_level sweeps")
            return 1

        gen_map = _generative_map_from_geo_art(geo_art)
        try:
            output_path = gen_map.create_animation(
                output_path=args.output,
                parameter_sweep=args.parameter,
                values=sweep_values,
                duration=args.duration,
                fps=args.fps,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    print(f"Animation saved to {output_path}")
    return 0


def process_custom_algorithm(args: argparse.Namespace) -> int:
    """Process custom algorithm creation and execution."""
    print(f"Processing custom algorithm: {args.algorithm}")

    # Create custom algorithm framework
    framework = CustomAlgorithmFramework()

    if args.register:
        # Register a new algorithm
        if not args.algorithm_file:
            print("Error: --algorithm-file required when registering")
            return 1

        try:
            framework.load_algorithms_from_file(args.algorithm_file)
            print(f"Successfully loaded algorithms from {args.algorithm_file}")
        except Exception as e:
            print(f"Error loading algorithms: {str(e)}")
            return 1

    else:
        # Execute an algorithm
        if not args.input or not args.output:
            print("Error: --input and --output required for algorithm execution")
            return 1

        # Load input data
        if args.input.lower().endswith((".geojson", ".json")):
            geo_art = GeoArt.load_geojson(args.input)
        elif args.input.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
            geo_art = GeoArt.load_raster(args.input)
        else:
            print(f"Error: Unsupported input file format: {args.input}")
            return 1

        # Parse parameters
        params = {}
        if args.parameters:
            for param in args.parameters:
                if "=" in param:
                    key, value = param.split("=", 1)
                    try:
                        # Try to convert to number
                        if "." in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        params[key] = value

        # Execute the algorithm
        try:
            if args.algorithm in framework.list_algorithms():
                result = framework.execute_algorithm(
                    args.algorithm, geo_art.data, width=800, height=800, **params
                )

                # Save the result
                if hasattr(result, "save"):
                    result.savefig(args.output, dpi=300, bbox_inches="tight")
                elif isinstance(result, np.ndarray):
                    img = Image.fromarray(result)
                    img.save(args.output)
                else:
                    print(
                        f"Warning: Algorithm returned unsupported result type: {type(result)}"
                    )

                print(f"Algorithm result saved to {args.output}")

            else:
                print(f"Error: Algorithm '{args.algorithm}' not found")
                print(f"Available algorithms: {framework.list_algorithms()}")
                return 1

        except Exception as e:
            print(f"Error executing algorithm: {str(e)}")
            return 1

    return 0


def process_performance(args: argparse.Namespace) -> int:
    """Process performance analysis and optimization."""
    print("Processing performance analysis...")

    optimizer = PerformanceOptimizer()

    if args.cache_stats:
        print("Cache Statistics:")
        print(f"  Entries: {len(optimizer.cache)}")
        print(f"  Max entries: {optimizer.max_cache_size}")
        print(
            f"  Total size: {sum(optimizer.cache_sizes.values()) / (1024 * 1024):.2f} MB"
        )

    if args.benchmark:
        print("Running performance benchmark...")
        # Benchmark some basic operations
        benchmark = optimizer.benchmark_function(
            lambda: np.random.rand(100, 100), iterations=5
        )
        print(f"Benchmark results: {benchmark}")

    if args.optimize:
        print(f"Optimizing for target time: {args.target_time}s")
        optimal_res = optimizer.optimize_resolution(
            target_time=args.target_time, max_resolution=2000
        )
        print(f"Optimal resolution: {optimal_res}x{optimal_res}")

    # Show performance report
    report = optimizer.create_performance_report()
    print("\nPerformance Report:")
    for category, data in report.items():
        print(f"  {category}: {data}")

    return 0


def process_3d_viz(args: argparse.Namespace) -> int:
    """Process 3D visualization creation."""
    print(f"Creating 3D {args.viz_type} visualization...")

    # Load input data
    if args.input.lower().endswith((".geojson", ".json")):
        geo_art = GeoArt.load_geojson(args.input)
    elif args.input.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
        geo_art = GeoArt.load_raster(args.input)
    else:
        print(f"Error: Unsupported input file format: {args.input}")
        return 1

    # Create 3D visualization
    try:
        viz_3d = geo_art.create_3d_visualization(
            elevation_data=(
                GeoArt.load_raster(args.elevation).data if args.elevation else None
            ),
            z_column=args.z_column,
        )

        viz_3d.create_3d_surface(
            output_file=args.output,
            title=f"3D {args.viz_type.capitalize()} Visualization",
        )

        print(f"3D visualization saved to {args.output}")

    except Exception as e:
        print(f"Error creating 3D visualization: {str(e)}")
        return 1

    return 0


def process_realtime(args: argparse.Namespace) -> int:
    """Process real-time visualization creation."""
    print("Creating real-time visualization...")

    # Load input data
    if args.input.lower().endswith((".geojson", ".json")):
        geo_art = GeoArt.load_geojson(args.input)
    elif args.input.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
        geo_art = GeoArt.load_raster(args.input)
    else:
        print(f"Error: Unsupported input file format: {args.input}")
        return 1

    # Create data callback function
    def data_callback() -> Any:
        # In a real implementation, this would fetch live data
        # For demo, just return the same data
        return geo_art.data

    # Create real-time visualization
    try:
        realtime_viz = geo_art.create_realtime_visualization(
            data_callback=data_callback,
            update_interval=args.update_interval,
            style=args.style,
            max_updates=args.max_updates,
            output_file=args.output,
        )

        print("Real-time visualization created")
        print("Call realtime_viz.start() to begin updates")

        # Optionally start the visualization
        if args.output:
            print("Starting real-time updates...")
            realtime_viz.start(use_threading=False)  # Synchronous for CLI
            print(f"Snapshots saved to {args.output}")

    except Exception as e:
        print(f"Error creating real-time visualization: {str(e)}")
        return 1

    return 0


def process_web_map(args: argparse.Namespace) -> int:
    """Process interactive web map creation."""
    print("Creating interactive web map...")

    # Load input data
    if args.input.lower().endswith((".geojson", ".json")):
        geo_art = GeoArt.load_geojson(args.input)
    else:
        print("Error: Web maps currently support GeoJSON files only")
        return 1

    # Create interactive web map
    try:
        output_file = geo_art.create_interactive_web_map(
            output_file=args.output, tiles=args.tiles, zoom_start=args.zoom_start
        )

        print(f"Interactive web map saved to {output_file}")
        print(f"Open {output_file} in a web browser to view the map")

    except Exception as e:
        print(f"Error creating web map: {str(e)}")
        return 1

    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="GEO-INFER-ART: Artistic visualization of geospatial data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(title="commands", dest="command")

    # GeoArt command
    geo_art_parser = subparsers.add_parser(
        "geo-art", help="Create artistic visualization of geospatial data"
    )
    geo_art_parser.add_argument("--input", required=True, help="Input geospatial file")
    geo_art_parser.add_argument("--output", required=True, help="Output image file")
    geo_art_parser.add_argument(
        "--style", default="default", help="Visualization style"
    )
    geo_art_parser.add_argument("--color-palette", help="Color palette name")
    geo_art_parser.add_argument(
        "--line-width", type=float, default=1.0, help="Line width for vector data"
    )
    geo_art_parser.add_argument(
        "--alpha", type=float, default=0.8, help="Transparency level (0.0-1.0)"
    )
    geo_art_parser.add_argument(
        "--background-color", default="white", help="Background color"
    )
    geo_art_parser.add_argument(
        "--dpi", type=int, default=300, help="Output resolution (DPI)"
    )
    geo_art_parser.add_argument(
        "--map-style", help="MapStyle name for advanced styling"
    )
    geo_art_parser.add_argument(
        "--legend", action="store_true", help="Add legend to visualization"
    )
    geo_art_parser.add_argument("--title", help="Custom title for the visualization")

    # Style Transfer command
    style_transfer_parser = subparsers.add_parser(
        "style-transfer", help="Apply artistic style transfer to geospatial data"
    )
    style_transfer_parser.add_argument(
        "--input", required=True, help="Input geospatial file"
    )
    style_transfer_parser.add_argument(
        "--output", required=True, help="Output image file"
    )
    style_transfer_parser.add_argument(
        "--style-transfer", required=True, help="Style name or file path"
    )
    style_transfer_parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations"
    )
    style_transfer_parser.add_argument(
        "--style-weight", type=float, default=1e-2, help="Style weight"
    )
    style_transfer_parser.add_argument(
        "--content-weight", type=float, default=1e4, help="Content weight"
    )
    style_transfer_parser.add_argument(
        "--color-palette", help="Optional color palette to apply"
    )

    # Place Art command
    place_art_parser = subparsers.add_parser(
        "place-art", help="Create art based on geographic locations"
    )
    place_art_group = place_art_parser.add_mutually_exclusive_group(required=True)
    place_art_group.add_argument("--place", help="Name of the place")
    place_art_group.add_argument("--coordinates", help="Coordinates as 'lat,lon'")
    place_art_parser.add_argument("--output", required=True, help="Output image file")
    place_art_parser.add_argument("--style", default="abstract", help="Art style")
    place_art_parser.add_argument(
        "--add-metadata", action="store_true", help="Add location metadata overlay"
    )

    # Generative Map command
    gen_map_parser = subparsers.add_parser(
        "generative-map", help="Generate art from elevation data"
    )
    gen_map_group = gen_map_parser.add_mutually_exclusive_group(required=True)
    gen_map_group.add_argument(
        "--region", help="Named region (e.g., 'grand_canyon', 'everest')"
    )
    gen_map_group.add_argument(
        "--bbox", help="Bounding box as 'min_lon,min_lat,max_lon,max_lat'"
    )
    gen_map_parser.add_argument("--output", required=True, help="Output image file")
    gen_map_parser.add_argument("--style", default="contour", help="Generative style")
    gen_map_parser.add_argument(
        "--resolution", type=int, default=512, help="Output resolution"
    )
    gen_map_parser.add_argument(
        "--abstraction-level",
        type=float,
        default=0.5,
        help="Level of abstraction (0.0-1.0)",
    )

    # Procedural Art command
    proc_art_parser = subparsers.add_parser(
        "procedural-art", help="Generate procedural art"
    )
    proc_art_group = proc_art_parser.add_mutually_exclusive_group()
    proc_art_group.add_argument("--coordinates", help="Coordinates as 'lat,lon'")
    proc_art_group.add_argument("--feature-type", help="Geographic feature type")
    proc_art_parser.add_argument(
        "--feature-count", type=int, default=5, help="Number of features"
    )
    proc_art_parser.add_argument("--output", required=True, help="Output image file")
    proc_art_parser.add_argument(
        "--algorithm", default="noise_field", help="Procedural algorithm"
    )
    proc_art_parser.add_argument(
        "--color-palette", default="viridis", help="Color palette name"
    )
    proc_art_parser.add_argument(
        "--resolution", type=int, default=800, help="Output resolution"
    )

    # Cultural Map command
    cultural_map_parser = subparsers.add_parser(
        "cultural-map", help="Create maps with cultural context"
    )
    cultural_map_group = cultural_map_parser.add_mutually_exclusive_group(required=True)
    cultural_map_group.add_argument(
        "--region", help="Named region (e.g., 'mediterranean', 'east_asia')"
    )
    cultural_map_group.add_argument("--coordinates", help="Coordinates as 'lat,lon'")
    cultural_map_parser.add_argument(
        "--radius-km", type=float, default=100.0, help="Radius in kilometers"
    )
    cultural_map_parser.add_argument(
        "--output", required=True, help="Output image file"
    )
    cultural_map_parser.add_argument(
        "--cultural-theme", default="historical", help="Cultural theme"
    )
    cultural_map_parser.add_argument("--style", default="artistic", help="Visual style")
    cultural_map_parser.add_argument(
        "--narrative", help="Cultural narrative text to add"
    )
    cultural_map_parser.add_argument(
        "--apply-cultural-style", action="store_true", help="Apply cultural styling"
    )

    # Map Style command
    map_style_parser = subparsers.add_parser(
        "map-style", help="Create and demonstrate map styles"
    )
    map_style_parser.add_argument("--style", required=True, help="MapStyle name")
    map_style_parser.add_argument("--output", required=True, help="Output image file")
    map_style_parser.add_argument("--theme", help="Visual theme")
    map_style_parser.add_argument("--color-palette", help="Color palette name")

    # Animation command
    animation_parser = subparsers.add_parser(
        "animate", help="Create animated visualizations"
    )
    animation_parser.add_argument(
        "--input", required=True, help="Input geospatial file"
    )
    animation_parser.add_argument(
        "--output", required=True, help="Output animation file"
    )
    animation_parser.add_argument(
        "--animation-type",
        required=True,
        choices=["style_cycle", "parameter_sweep"],
        help="Type of animation",
    )
    animation_parser.add_argument(
        "--styles", nargs="+", help="Styles for style_cycle animation"
    )
    animation_parser.add_argument(
        "--parameter", help="Parameter for parameter_sweep animation"
    )
    animation_parser.add_argument(
        "--values",
        nargs="+",
        help="Values for parameter sweep (numbers for abstraction_level, "
        "style names for style)",
    )
    animation_parser.add_argument(
        "--duration", type=float, default=5.0, help="Animation duration"
    )
    animation_parser.add_argument(
        "--fps", type=int, default=24, help="Frames per second"
    )

    # Custom Algorithm command
    custom_algo_parser = subparsers.add_parser(
        "custom-algorithm", help="Create and execute custom algorithms"
    )
    custom_algo_parser.add_argument("--algorithm", required=True, help="Algorithm name")
    custom_algo_parser.add_argument(
        "--input", required=True, help="Input geospatial file"
    )
    custom_algo_parser.add_argument("--output", required=True, help="Output image file")
    custom_algo_parser.add_argument(
        "--register", action="store_true", help="Register a new algorithm"
    )
    custom_algo_parser.add_argument(
        "--algorithm-file", help="File containing algorithm definition"
    )
    custom_algo_parser.add_argument(
        "--parameters", nargs="*", help="Algorithm parameters as key=value"
    )

    # Performance command
    performance_parser = subparsers.add_parser(
        "performance", help="Performance analysis and optimization"
    )
    performance_parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    performance_parser.add_argument(
        "--cache-stats", action="store_true", help="Show cache statistics"
    )
    performance_parser.add_argument(
        "--optimize", action="store_true", help="Optimize performance settings"
    )
    performance_parser.add_argument(
        "--target-time",
        type=float,
        default=1.0,
        help="Target execution time for optimization",
    )

    # 3D Visualization command
    viz_3d_parser = subparsers.add_parser("3d-viz", help="Create 3D visualizations")
    viz_3d_parser.add_argument("--input", required=True, help="Input geospatial file")
    viz_3d_parser.add_argument(
        "--output", required=True, help="Output 3D visualization file"
    )
    viz_3d_parser.add_argument(
        "--viz-type",
        required=True,
        choices=["surface", "scatter", "mesh"],
        help="Type of 3D visualization",
    )
    viz_3d_parser.add_argument("--elevation", help="Elevation data file for terrain")
    viz_3d_parser.add_argument("--z-column", help="Column name for Z-axis values")

    # Real-time Visualization command
    realtime_parser = subparsers.add_parser(
        "realtime", help="Create real-time visualizations"
    )
    realtime_parser.add_argument("--input", required=True, help="Input geospatial file")
    realtime_parser.add_argument("--output", help="Output file for snapshots")
    realtime_parser.add_argument(
        "--update-interval", type=float, default=1.0, help="Update interval in seconds"
    )
    realtime_parser.add_argument(
        "--max-updates", type=int, help="Maximum number of updates"
    )
    realtime_parser.add_argument(
        "--style", default="default", help="Visualization style"
    )

    # Interactive Web Map command
    webmap_parser = subparsers.add_parser(
        "web-map", help="Create interactive web-based maps"
    )
    webmap_parser.add_argument("--input", required=True, help="Input geospatial file")
    webmap_parser.add_argument("--output", required=True, help="Output HTML file")
    webmap_parser.add_argument(
        "--tiles", default="OpenStreetMap", help="Map tiles to use"
    )
    webmap_parser.add_argument(
        "--zoom-start", type=int, default=10, help="Initial zoom level"
    )

    # Simplified command for basic usage
    simple_parser = subparsers.add_parser(
        "simple", help="Simplified command for common operations"
    )
    simple_parser.add_argument("--input", help="Input geospatial file")
    simple_parser.add_argument("--output", required=True, help="Output image file")
    simple_parser.add_argument("--style", default="default", help="Visualization style")
    simple_parser.add_argument("--place", help="Name of the place for place-based art")
    simple_parser.add_argument(
        "--style-transfer", help="Apply style transfer with this style name"
    )

    args = parser.parse_args()

    if args.command == "geo-art":
        return process_geo_art(args)
    elif args.command == "style-transfer":
        return process_style_transfer(args)
    elif args.command == "place-art":
        return process_place_art(args)
    elif args.command == "generative-map":
        return process_generative_map(args)
    elif args.command == "procedural-art":
        return process_procedural_art(args)
    elif args.command == "cultural-map":
        return process_cultural_map(args)
    elif args.command == "map-style":
        return process_map_style(args)
    elif args.command == "animate":
        return process_animation(args)
    elif args.command == "custom-algorithm":
        return process_custom_algorithm(args)
    elif args.command == "performance":
        return process_performance(args)
    elif args.command == "3d-viz":
        return process_3d_viz(args)
    elif args.command == "realtime":
        return process_realtime(args)
    elif args.command == "web-map":
        return process_web_map(args)
    elif args.command == "simple":
        # Process simplified commands
        if args.input and args.style_transfer:
            args.command = "style-transfer"
            args.iterations = 50  # Use fewer iterations for simple mode
            return process_style_transfer(args)
        elif args.input:
            args.command = "geo-art"
            return process_geo_art(args)
        elif args.place:
            args.command = "place-art"
            args.add_metadata = True
            return process_place_art(args)
        else:
            print("Error: --input or --place must be specified with the simple command")
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
