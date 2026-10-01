# Guide for Autonomous Agent Coders

> **Illustrative example notice.** This page contains historical or
> conceptual integration sketches. Names such as `SpatialAnalyzer` and
> domain-specific facade classes are not public GEO-INFER exports in the
> current checkout; verify imports against each module's `src/` package
> and use the module README/tests for executable examples.


## Introduction
This guide is designed specifically for autonomous agent coders working on the GEO-INFER framework. It provides essential information on navigating the codebase, understanding the project paradigm, and making effective contributions while maintaining consistency with project standards.

## Understanding the GEO-INFER Paradigm
GEO-INFER is built on several core principles that shape its architecture and development approach:

### 1. Active Inference Framework
GEO-INFER implements the active inference framework, a computational approach to perception, learning, and decision-making. This influences how modules interact, process information, and make decisions under uncertainty. Key components include:

- **Generative models**: Predicting sensory inputs based on internal models - **Variational inference**: Approximating posterior probabilities - **Free energy minimization**: Optimizing the balance between accuracy and complexity - **Policy selection**: Choosing actions that minimize expected free energy

### 2. Domain-Driven Modularity
Each module represents a specific domain expertise:

- **GEO-INFER-SPACE**: Spatial representation and analysis - **GEO-INFER-TIME**: Temporal representation and analysis - **GEO-INFER-DATA**: Data management and processing - **GEO-INFER-OPS**: Operational orchestration - **GEO-INFER-ACT**: Active inference implementation - **GEO-INFER-BAYES**: Bayesian statistical methods - (and others...)

### 3. Hierarchical Organization
The framework implements a hierarchical approach to geospatial modeling, with:

- Low-level spatial and temporal primitives - Mid-level analytical capabilities - High-level integrated processing components

## First Steps for Agent Coders

### 1. Orient Yourself
Before making any changes, understand the project structure:

```
bash # Navigate to project root cd /path/to/INFER-GEO # List modules ls -d GEO-INFER-*/ # Explore a specific module cd GEO-INFER-SPACE ls
```
 ### 2. Identify Module Boundaries Determine which module(s) are relevant to your task:
```
 GEO-INFER-SPACE/ - Spatial data processing and analysis GEO-INFER-TIME/ - Temporal data processing and analysis GEO-INFER-DATA/ - Data management and storage GEO-INFER-OPS/ - Operational orchestration and deployment GEO-INFER-API/ - API definitions and gateway GEO-INFER-APP/ - User-facing applications GEO-INFER-INTRA/ - Documentation and knowledge management
```
 ### 3. Understand Code Context Examine the structure and conventions of the module you're working with: - Review the module's README.md - Check the src/geo_infer_module structure - Identify key patterns used in the codebase - Examine test cases for usage examples ## Making Effective Contributions ### 1. Code Organization Always organize code according to existing patterns: - **Core functionality**: `src/geo_infer_module/core/` - **API endpoints**: `src/geo_infer_module/api/` - **Data models**: `src/geo_infer_module/models/` - **Utility functions**: `src/geo_infer_module/utils/` ### 2. Testing Approach Follow the established testing patterns: - **Unit tests**: Test individual functions and methods - **Integration tests**: Test interactions between components - **Functional tests**: Test end-to-end workflows - **Property-based tests**: Test with generated inputs Example:
```python
 # Unit test example - always in tests/ directory def test_coordinate_transformation(): # Given input_coord = Coordinate(lat=45.0, lon=-122.0) # When result = transform_coordinate(input_coord, "EPSG:4326", "EPSG:3857") # Then assert abs(result.x - -13580977.876779) < 0.1 assert abs(result.y - 5621521.486191) < 0.1
```
 ### 3. Documentation Standards Document your code following these patterns:
```python
 def calculate_distance(point_a, point_b, method="haversine"): """ Calculate the distance between two geographic points. Args: point_a: The first geographic point (lat, lon) point_b: The second geographic point (lat, lon) method: The calculation method, one of ["haversine", "vincenty"] Returns: float: The distance in meters Raises: ValueError: If the calculation method is not supported """ # Implementation...
```
 ### 4. Agent Communication When multiple agents work on related code, communication is crucial: - Use code comments to mark handoff points - Document design decisions in relevant docs - Update module README.md with significant changes - Add migration guides for breaking changes ## Common Workflows ### 1. Adding a Feature 1. Determine the appropriate module 2. Review similar features for patterns 3. Update documentation to describe the feature 4. Add tests specifying the behavior 5. Implement the feature 6. Update example code ### 2. Fixing a Bug 1. Add a test that reproduces the bug 2. Fix the implementation 3. Verify the test passes 4. Update documentation if necessary 5. Consider implications for other modules ### 3. Refactoring Code 1. Ensure test coverage 2. Make incremental changes 3. Maintain API compatibility 4. Update documentation to reflect changes 5. Add deprecation notices for changed APIs ## Working with GEO-INFER-OPS GEO-INFER-OPS is the orchestration layer for the project. When interacting with it: 1. Use established logging patterns:
```python
 from geo_infer_ops.core.logging import get_logger logger = get_logger(__name__) logger.info("Processing started", extra={"data_id": data_id})
```
 2. Integrate with monitoring:
```
python from geo_infer_ops.core.monitoring import metrics metrics.timing("process_time", duration_ms) metrics.increment("items_processed")
```
 3. Implement standardized error handling:
```
python try: # Operation except Exception as e: logger.error("Failed to process data", exc_info=True) raise OperationError("Processing failed") from e
```
 ## Spatial Data Handling When working with geospatial data: 1. Always specify coordinate reference systems (CRS) 2. Use standardized formats (GeoJSON, GeoPackage, etc.) 3. Implement appropriate spatial indexing 4. Consider performance for large datasets 5. Properly handle geodetic vs. projected coordinates Example:
```
python from geo_infer_space.models import GeoDataFrame from geo_infer_space.core import spatial_join # Always specify CRS data = GeoDataFrame.from_file("data.geojson", crs="EPSG:4326") # Use appropriate indexing for performance data.create_spatial_index("h3", resolution=8) # Apply spatial operations result = spatial_join(data, reference_data, predicate="intersects")
```
 ## Temporal Data Handling When working with temporal data: 1. Use ISO8601 format for timestamps 2. Always specify time zones 3. Use appropriate temporal indexing 4. Consider time series specific algorithms 5. Handle irregular sampling appropriately Example:
```
python from geo_infer_time.models import TimeSeriesDataFrame from geo_infer_time.core import resample # Load with explicit time handling data = TimeSeriesDataFrame.from_file( "data.csv", time_column="timestamp", time_format="%Y-%m-%dT%H:%M:%S%z" ) # Apply temporal operations resampled = resample(data, frequency="1D", method="mean")
```
 ## Integration Patterns When integrating across modules: 1. Use high-level APIs rather than internal components 2. Follow event-driven architecture for loose coupling 3. Implement appropriate error handling and retries 4. Document cross-module dependencies explicitly 5. Consider versioning compatibility Example:
```
python # Integrating space and time modules from geo_infer_space import SpatialDataFrame from geo_infer_time import temporal_aggregation # Load spatial data spatial_data = SpatialDataFrame.from_file("regions.geojson") # Integrate with temporal processing result = temporal_aggregation( spatial_data, time_series_data, aggregate_by="region_id", temporal_function="mean_monthly" )
```
 ## Active Inference Implementation When implementing active inference principles: 1. Define appropriate generative models 2. Implement variational inference methods 3. Structure processes to minimize free energy 4. Document model assumptions clearly 5. Provide performance metrics for models Example:
```
python from geo_infer_act.models import GenerativeModel from geo_infer_act.inference import variational_inference # Define generative model model = GenerativeModel( likelihood="gaussian", prior="multivariate_normal", parameters={"dimensions": 3} ) # Perform inference posterior = variational_inference( model=model, data=observations, method="mean_field", iterations=1000 )
```
 ## Troubleshooting Guide Common issues and solutions: 1. **Module Import Errors** - Check the module is installed in the environment - Verify import path follows conventions - Check for circular dependencies 2. **API Compatibility Issues** - Check version constraints in the module pyproject.toml and the root uv.lock - Review API documentation for breaking changes - Check for deprecation warnings 3. **Performance Problems** - Profile code to identify bottlenecks - Check spatial/temporal indexing - Review data structures for appropriate use - Consider parallel processing for large datasets 4. **Integration Failures** - Verify data format compatibility - Check event handlers are registered - Review error handling across boundaries - Check for environment differences ## Additional Resources - GEO-INFER-INTRA Documentation: `/GEO-INFER-INTRA/docs/` - Module Architecture Guides: `/GEO-INFER-INTRA/docs/architecture/` - API References: `/GEO-INFER-INTRA/docs/api/` - Example Notebooks: `/GEO-INFER-INTRA/examples/` ## Conclusion As an autonomous agent coder, your contributions are essential to the GEO-INFER ecosystem. By following these guidelines, you'll be able to navigate the codebase effectively, maintain consistency with project standards, and implement features that integrate seamlessly with the existing architecture.