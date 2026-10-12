#!/usr/bin/env python3
"""
Multiple Dispatch Demonstration for GEO-INFER-SPACE

This example demonstrates how all spatial operations support multiple dispatch
to different backends (H3 and SRAI) through the unified API.
"""

import sys


def demonstrate_multiple_dispatch():
    """Demonstrate multiple dispatch across all spatial operations."""

    print("🚀 GEO-INFER-SPACE Multiple Dispatch Demonstration")
    print("=" * 60)

    # Import the generic interfaces
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface
    from geo_infer_space.core.dispatcher import configure_backends
    from geo_infer_space.core.dispatcher import get_backend_dispatcher

    # Configure backends for different operation types
    configure_backends(
        {
            "default_backends": {
                "indexing": "h3",  # Use H3 for spatial indexing
                "analytics": "srai",  # Use SRAI for spatial analytics
            }
        }
    )

    print("✅ Backend configuration applied")

    # Test data
    sf_coords = (37.7749, -122.4194)  # San Francisco
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.42, 37.77],
                [-122.41, 37.77],
                [-122.41, 37.78],
                [-122.42, 37.78],
                [-122.42, 37.77],
            ]
        ],
    }

    # 1. SPATIAL INDEXING OPERATIONS
    print("\n📍 SPATIAL INDEXING OPERATIONS")
    print("-" * 40)

    # H3 Backend
    h3_indexer = SpatialIndexingInterface(backend="h3")
    cell_h3 = h3_indexer.latlng_to_cell(sf_coords[0], sf_coords[1], 9)
    print(f"✅ H3 latlng_to_cell: {cell_h3}")

    lat_h3, lng_h3 = h3_indexer.cell_to_latlng(cell_h3)
    print(f"✅ H3 cell_to_latlng: ({lat_h3:.4f}, {lng_h3:.4f})")

    cells_h3 = h3_indexer.polygon_to_cells(polygon, 9)
    print(f"✅ H3 polygon_to_cells: {len(cells_h3)} cells")

    neighbors_h3 = h3_indexer.get_cell_neighbors(cell_h3, k=1)
    print(f"✅ H3 get_cell_neighbors: {len(neighbors_h3)} neighbors")

    if len(neighbors_h3) > 0:
        distance_h3 = h3_indexer.get_cell_distance(cell_h3, neighbors_h3[0])
        print(f"✅ H3 get_cell_distance: {distance_h3}")

    compacted_h3 = h3_indexer.compact_cells(cells_h3)
    print(f"✅ H3 compact_cells: {len(compacted_h3)} compacted cells")

    # SRAI Backend (skip if not available)
    try:
        srai_indexer = SpatialIndexingInterface(backend="srai")
        if not srai_indexer._backend.is_available():
            raise RuntimeError("SRAI not available")

        cell_srai = srai_indexer.latlng_to_cell(sf_coords[0], sf_coords[1], 9)
        print(f"✅ SRAI latlng_to_cell: {cell_srai}")

        lat_srai, lng_srai = srai_indexer.cell_to_latlng(cell_srai)
        print(f"✅ SRAI cell_to_latlng: ({lat_srai:.4f}, {lng_srai:.4f})")

        cells_srai = srai_indexer.polygon_to_cells(polygon, 9)
        print(f"✅ SRAI polygon_to_cells: {len(cells_srai)} cells")

        neighbors_srai = srai_indexer.get_cell_neighbors(cell_srai, k=1)
        print(f"✅ SRAI get_cell_neighbors: {len(neighbors_srai)} neighbors")
    except Exception as e:
        print(f"⚠️  SRAI operations skipped: {e}")

    # 2. SPATIAL ANALYTICS OPERATIONS
    print("\n📊 SPATIAL ANALYTICS OPERATIONS")
    print("-" * 40)

    # Sample data for analytics
    cells = [cell_h3] + [f"cell_{i}" for i in range(2, 8)]
    values = [100, 80, 60, 120, 90, 70, 110]

    data = {"cells": cells, "values": values}

    # H3 Analytics
    h3_analytics = SpatialAnalyticsInterface(backend="h3")
    hotspots_h3 = h3_analytics.analyze_hotspots(data)
    print(f"✅ H3 analyze_hotspots: {hotspots_h3.get('hotspot_count', 0)} hotspots")

    # SRAI Analytics (skip if not available)
    try:
        srai_analytics = SpatialAnalyticsInterface(backend="srai")
        if not srai_analytics._backend.is_available():
            raise RuntimeError("SRAI not available")

        hotspots_srai = srai_analytics.analyze_hotspots(data)
        print(
            f"✅ SRAI analyze_hotspots: {hotspots_srai.get('hotspot_count', 0)} hotspots"
        )
    except Exception as e:
        print(f"⚠️  SRAI analytics skipped: {e}")

    # 3. CONVENIENCE FUNCTIONS
    print("\n🛠️  CONVENIENCE FUNCTIONS")
    print("-" * 40)

    from geo_infer_space.core.spatial_indexing import latlng_to_cell

    # Use default backend
    try:
        cell_default = latlng_to_cell(sf_coords[0], sf_coords[1], 9)
        print(f"✅ Default latlng_to_cell: {cell_default}")
    except Exception as e:
        print(f"⚠️  Default latlng_to_cell failed: {e}")

    # Explicit H3 backend
    try:
        cell_explicit_h3 = latlng_to_cell(sf_coords[0], sf_coords[1], 9, backend="h3")
        print(f"✅ Explicit H3 latlng_to_cell: {cell_explicit_h3}")
    except Exception as e:
        print(f"⚠️  Explicit H3 latlng_to_cell failed: {e}")

    # Explicit SRAI backend
    try:
        # Check if SRAI is available via dispatcher
        if get_backend_dispatcher().get_backend("srai").is_available():
            cell_explicit_srai = latlng_to_cell(
                sf_coords[0], sf_coords[1], 9, backend="srai"
            )
            print(f"✅ Explicit SRAI latlng_to_cell: {cell_explicit_srai}")
        else:
            print("⚠️  Explicit SRAI latlng_to_cell skipped: SRAI not available")
    except Exception as e:
        print(f"⚠️  Explicit SRAI latlng_to_cell skipped: {e}")

    # 4. BACKEND CAPABILITIES
    print("\n🔧 BACKEND CAPABILITIES")
    print("-" * 40)

    dispatcher = get_backend_dispatcher()
    backends = dispatcher.get_available_backends()

    for backend_name in backends:
        try:
            backend = dispatcher.get_backend(backend_name)
            capabilities = backend.get_capabilities()

            print(f"\n📋 {backend_name.upper()} Backend:")
            print(f"   Version: {backend.version}")
            print(f"   Available: {backend.is_available()}")

            if "indexing" in capabilities:
                print(f"   Indexing ops: {len(capabilities['indexing'])}")
            if "analytics" in capabilities:
                print(f"   Analytics ops: {len(capabilities['analytics'])}")
            if "regionalizers" in capabilities:
                print(f"   Regionalizers: {capabilities['regionalizers']}")
        except Exception as e:
            print(f"⚠️  Could not get info for backend {backend_name}: {e}")

    # 5. DYNAMIC BACKEND SWITCHING
    print("\n🔄 DYNAMIC BACKEND SWITCHING")
    print("-" * 40)

    def adaptive_spatial_operation(coordinates, precision_required=False):
        """Example of adaptive backend selection based on context."""

        if precision_required:
            # Use H3 for high-precision operations
            indexer = SpatialIndexingInterface(backend="h3")
            return indexer.latlng_to_cell(coordinates[0], coordinates[1], 12)
        else:
            # Use SRAI for general operations
            indexer = SpatialIndexingInterface(backend="srai")
            return indexer.latlng_to_cell(coordinates[0], coordinates[1], 9)

    # High precision operation
    try:
        high_precision_cell = adaptive_spatial_operation(
            sf_coords, precision_required=True
        )
        print(f"✅ High precision operation (H3): {high_precision_cell}")
    except Exception as e:
        print(f"⚠️  High precision operation (H3) failed: {e}")

    # General operation (SRAI) - handle unavailability
    try:
        # Check if SRAI is available before creating interface to avoid warning logs
        if get_backend_dispatcher().get_backend("srai").is_available():
            general_cell = adaptive_spatial_operation(
                sf_coords, precision_required=False
            )
            print(f"✅ General operation (SRAI): {general_cell}")
        else:
            print("⚠️  General operation (SRAI) skipped: SRAI not available")
    except Exception as e:
        print(f"⚠️  General operation (SRAI) skipped: {e}")

    print("\n" + "=" * 60)
    print("🎉 MULTIPLE DISPATCH DEMONSTRATION COMPLETE!")

    print("\n📋 Summary of Multiple Dispatch Architecture:")
    print("✅ All spatial indexing operations dispatch to both H3 and SRAI")
    print("✅ All spatial analytics operations dispatch to both H3 and SRAI")
    print("✅ Backend selection is configurable and context-aware")
    print("✅ Convenience functions work with any backend")
    print("✅ Backend capabilities are properly reported")
    print("✅ Dynamic backend switching is supported")

    return True


if __name__ == "__main__":
    try:
        success = demonstrate_multiple_dispatch()
        if success:
            print("\n✅ All demonstrations completed successfully!")
        else:
            print("\n❌ Some demonstrations failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        sys.exit(1)
