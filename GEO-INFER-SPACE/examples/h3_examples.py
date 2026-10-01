#!/usr/bin/env python3
"""
H3 Backend Usage Examples

Comprehensive examples demonstrating all H3 operations available in GEO-INFER-SPACE.
Run with: uv run python examples/h3_examples.py
"""

import random

from geo_infer_space.backends.h3.h3_backend import H3Backend

# Initialize H3 backend
h3 = H3Backend()
print(f"H3 Backend v{h3.version}")
print("=" * 60)


# =============================================================================
# EXAMPLE 1: Basic Cell Indexing
# =============================================================================
print("\n📍 EXAMPLE 1: Basic Cell Indexing")
print("-" * 40)

# San Francisco coordinates
lat, lng = 37.7749, -122.4194
resolution = 8  # ~460m edge length

# Convert coordinates to H3 cell
cell = h3.latlng_to_cell(lat, lng, resolution)
print(f"Location: ({lat}, {lng})")
print(f"H3 Cell (res {resolution}): {cell}")

# Get cell center
center = h3.cell_to_latlng(cell)
print(f"Cell center: ({center[0]:.6f}, {center[1]:.6f})")

# Get cell properties
print(f"Cell resolution: {h3.get_cell_resolution(cell)}")
print(f"Cell area: {h3.get_cell_area(cell, 'km^2'):.4f} km²")
print(f"Is pentagon: {h3.is_pentagon(cell)}")


# =============================================================================
# EXAMPLE 2: Validation
# =============================================================================
print("\n✅ EXAMPLE 2: Validation")
print("-" * 40)

# Validate cell
print(f"Is valid cell: {h3.is_valid_cell(cell)}")
print(f"Is 'invalid' valid: {h3.is_valid_cell('invalid')}")

# Validate resolution
res_check = h3.validate_resolution(8)
print(f"Resolution 8 valid: {res_check['valid']}")

res_check = h3.validate_resolution(20)
print(f"Resolution 20 valid: {res_check['valid']} - {res_check['error']}")

# Validate coordinates
coord_check = h3.validate_coordinates(lat, lng)
print(f"Coordinates valid: {coord_check['valid']}")

coord_check = h3.validate_coordinates(91, 0)
print(f"Lat 91 valid: {coord_check['valid']} - {coord_check['errors']}")


# =============================================================================
# EXAMPLE 3: Neighbors and Relationships
# =============================================================================
print("\n🔗 EXAMPLE 3: Neighbors and Relationships")
print("-" * 40)

# Get neighbors
neighbors = h3.get_cell_neighbors(cell, k=1)
print(f"Ring 1 neighbors: {len(neighbors)} cells")

# Check if cells are neighbors
print(f"Are neighbors: {h3.are_neighbors(cell, neighbors[0])}")

# Get cell distance
dist = h3.get_cell_distance(cell, neighbors[0])
print(f"Grid distance to neighbor: {dist}")

# Get ring at distance 2
ring2 = h3.get_cell_ring(cell, k=2)
print(f"Ring 2 count: {len(ring2)} cells")


# =============================================================================
# EXAMPLE 4: Hierarchical Operations
# =============================================================================
print("\n📊 EXAMPLE 4: Hierarchical Operations")
print("-" * 40)

# Get parent at coarser resolution
parent = h3.get_cell_parent(cell, 5)
print(f"Parent (res 5): {parent}")

# Get children at finer resolution
children = h3.get_cell_children(cell, 10)
print(f"Children (res 10): {len(children)} cells")

# Get base cell
base = h3.get_base_cell(cell)
print(f"Base cell number: {base} (of 122 total)")

# Compact and uncompact
all_cells = [cell] + list(neighbors)
compacted = h3.compact_cells(all_cells)
print(f"Compacted {len(all_cells)} → {len(compacted)} cells")

uncompacted = h3.uncompact_cells(compacted, 8)
print(f"Uncompacted back to {len(uncompacted)} cells")


# =============================================================================
# EXAMPLE 5: Directed Edges
# =============================================================================
print("\n➡️ EXAMPLE 5: Directed Edges")
print("-" * 40)

# Get directed edge between neighbors
edge = h3.get_directed_edge(cell, neighbors[0])
print(f"Directed edge: {edge}")

# Get cells from edge
origin, dest = h3.edge_to_cells(edge)
print(f"Origin: {origin}")
print(f"Destination: {dest}")

# Get all cell edges
all_edges = h3.get_cell_edges(cell)
print(f"Cell has {len(all_edges)} edges")

# Get edge boundary
boundary = h3.get_edge_boundary(edge)
print(f"Edge boundary: {len(boundary)} vertices")


# =============================================================================
# EXAMPLE 6: Local IJ Coordinates
# =============================================================================
print("\n📐 EXAMPLE 6: Local IJ Coordinates")
print("-" * 40)

# Convert cell to local IJ
i, j = h3.cell_to_local_ij(cell, neighbors[0])
print(f"Neighbor IJ coords (relative to origin): ({i}, {j})")

# Convert back to cell
recovered = h3.local_ij_to_cell(cell, i, j)
print(f"Recovered cell matches: {recovered == neighbors[0]}")


# =============================================================================
# EXAMPLE 7: Geometric Calculations
# =============================================================================
print("\n🌍 EXAMPLE 7: Geometric Calculations")
print("-" * 40)

# Great circle distance
sf_lat, sf_lng = 37.7749, -122.4194
nyc_lat, nyc_lng = 40.7128, -74.0060

dist = h3.great_circle_distance(sf_lat, sf_lng, nyc_lat, nyc_lng, "km")
print(f"SF to NYC distance: {dist:.1f} km")

# Point to cell center distance
point_dist = h3.point_distance_to_cell_center(lat, lng, cell)
print(f"Point to cell center: {point_dist:.1f} m")

# Line to cells
cells_along_line = h3.line_to_cells(
    sf_lat,
    sf_lng,
    37.8749,
    -122.3194,  # Nearby point
    8,
)
print(f"Cells along line: {len(cells_along_line)}")

# Geodesic area
area = h3.cell_to_geodesic_area(cell, "km^2")
print(f"Geodesic area: {area:.4f} km²")

# Average edge length
edge_len = h3.average_edge_length(8, "m")
print(f"Avg edge length (res 8): {edge_len:.1f} m")


# =============================================================================
# EXAMPLE 8: Pentagons
# =============================================================================
print("\n⬠ EXAMPLE 8: Pentagons")
print("-" * 40)

# Get all pentagons at resolution
pentagons = h3.get_pentagons(8)
print(f"Pentagon count (res 8): {len(pentagons)}")

# Check first pentagon
print(f"Is pentagon: {h3.is_pentagon(pentagons[0])}")
print(f"Pentagon edges: {len(h3.get_cell_edges(pentagons[0]))}")


# =============================================================================
# EXAMPLE 9: Resolution Statistics
# =============================================================================
print("\n📈 EXAMPLE 9: Resolution Statistics")
print("-" * 40)

for res in [0, 5, 8, 12, 15]:
    stats = h3.get_resolution_stats(res)
    print(
        f"Res {res:2d}: {stats['total_cells']:>15,} cells, "
        f"area: {stats['average_area_km2']:.6f} km², "
        f"edge: {stats['average_edge_length_km'] * 1000:.1f} m"
    )


# =============================================================================
# EXAMPLE 10: Comprehensive Validation
# =============================================================================
print("\n🔍 EXAMPLE 10: Comprehensive Validation")
print("-" * 40)

# Validate a cell set
test_cells = [cell] + list(neighbors)[:3] + ["invalid_cell"]
validation = h3.validate_cell_set(test_cells)

print(f"Total cells: {validation['total_cells']}")
print(f"Valid: {validation['valid_count']}")
print(f"Invalid: {validation['invalid_count']}")
print(f"Uniform resolution: {validation['is_uniform_resolution']}")
print(f"All valid: {validation['all_valid']}")


# =============================================================================
# EXAMPLE 11: Polygon to Cells
# =============================================================================
print("\n🔷 EXAMPLE 11: Polygon to Cells")
print("-" * 40)

# Define a polygon (simple square)
polygon = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.42, 37.77],
            [-122.40, 37.77],
            [-122.40, 37.78],
            [-122.42, 37.78],
            [-122.42, 37.77],
        ]
    ],
}

cells_in_polygon = h3.polygon_to_cells(polygon, 9)
print(f"Cells in polygon (res 9): {len(cells_in_polygon)}")


# =============================================================================
# EXAMPLE 12: Spatial Analytics
# =============================================================================
print("\n📊 EXAMPLE 12: Spatial Analytics")
print("-" * 40)

random.seed(42)

# Create sample data
sample_cells = [cell] + list(neighbors)
sample_values = [random.uniform(10, 100) for _ in sample_cells]

# Analyze hotspots
hotspot_result = h3.analyze_hotspots({"cells": sample_cells, "values": sample_values})
print(
    f"Hotspot analysis: {len(hotspot_result.get('cells', sample_cells))} cells analyzed"
)

# Find clusters
cluster_result = h3.find_clusters(
    sample_cells, sample_values, min_cluster_size=2, distance_threshold=1
)
print(f"Clusters found: {cluster_result.get('num_clusters', 0)}")

# Calculate density
density_result = h3.calculate_density(sample_cells, sample_values, kernel_radius=1)
print(f"Density calculated for {len(density_result.get('densities', {}))} cells")


print("\n" + "=" * 60)
print("✅ All examples completed successfully!")
print("=" * 60)
