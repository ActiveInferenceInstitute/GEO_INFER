# GEO-INFER-BIO Documentation

## Welcome to GEO-INFER-BIO

This module provides biological sequence analysis and spatially-referenced biological data
processing (climate, soil, microbiome) for the GEO-INFER framework.

## Documentation Index

### Getting Started

- [README](../README.md) - Module overview and quick start
- [AGENTS](../AGENTS.md) - Agent capabilities

### Guides

- [Sequence Analysis](#sequence-analysis)
- [Data Validation](#data-validation)
- [Visualization](#visualization)
- [Climate Data](#climate-data)
- [Soil Data](#soil-data)
- [Microbiome Data](#microbiome-data)
- [HTTP APIs](#http-apis)

## Sequence Analysis

### SequenceAnalyzer

```python
from geo_infer_bio import SequenceAnalyzer

analyzer = SequenceAnalyzer()

records = analyzer.load_sequence("sequences.fasta")   # SeqRecord or list[SeqRecord]
gc = analyzer.calculate_gc_content(records[0].seq)    # percent, 0-100
motifs = analyzer.find_motifs(records[0].seq, motif_length=6)  # repeated motifs -> positions
regions = analyzer.predict_coding_regions(records[0].seq)      # ORF frames

# Pairwise alignment of the first two records (Biopython PairwiseAligner)
alignment = analyzer.align_sequences(records[:2])
```

## Data Validation

### DataValidator

```python
from geo_infer_bio import DataValidator

validator = DataValidator()

validator.validate_sequence("ATCG", "DNA")        # True
validator.validate_sequence("AUCG", "RNA")        # True (A/U/C/G)
validator.validate_spatial_coordinates(45.0, -122.0)
validator.validate_gc_content(50.0, 100)          # bounds + non-zero length
```

## Visualization

### BioVisualizer

```python
from geo_infer_bio import BioVisualizer

visualizer = BioVisualizer()

# DataFrames must carry latitude/longitude columns plus the plotted metric
visualizer.plot_spatial_distribution(spatial_df, output_path="spatial.png")
visualizer.plot_gc_distribution(analysis_df, output_path="gc.png")
```

## Climate Data

### ClimateDataProcessor / ClimateDataset

```python
from geo_infer_bio import ClimateDataProcessor

processor = ClimateDataProcessor()

# Samples local WorldClim GeoTIFF rasters (requires the `integrations` extra: rasterio)
# dataset = processor.load_worldclim_data(
#     variables=["bio1", "bio12"],
#     coordinates=[(37.7, -122.4)],
#     data_path="worldclim/",
# )

dataset.get_variables()                  # available variable names
dataset.get_variable_data("bio1")        # DataFrame: latitude, longitude, value
dataset.get_all_variables_dataframe()    # merged on coordinates
```

## Soil Data

### SoilDataIntegrator / SoilDataset

```python
from geo_infer_bio import SoilDataIntegrator

integrator = SoilDataIntegrator()

# Live ISRIC SoilGrids REST query
# dataset = integrator.load_soilgrids_data(
#     coordinates=[(37.7, -122.4)],
#     properties=["phh2o", "soc"],
#     depths=["0-5cm", "5-15cm"],
# )

dataset.get_properties()                 # available soil property codes
dataset.get_depths()                     # available depth intervals
dataset.get_soil_profile(37.7, -122.4)   # profile by depth near a location
dataset.calculate_soil_health_indicators()
```

## Microbiome Data

### MicrobiomeDataLoader / MicrobiomeDataset

```python
from geo_infer_bio import MicrobiomeDataLoader

loader = MicrobiomeDataLoader()

# Local EMP metadata TSV/CSV with latitude/longitude columns
# dataset = loader.load_emp_data(metadata_path="emp_qiime_mapping_qc_filtered.tsv")

dataset.get_coordinates()                        # [(lat, lon), ...]
dataset.get_coordinates_gdf()                    # GeoDataFrame (EPSG:4326)
dataset.get_diversity_metrics()                  # diversity columns, if present
filtered = dataset.filter_by_coordinates((-180, -90, 180, 90))
```

## HTTP APIs

### REST (FastAPI)

```python
from geo_infer_bio.api import rest_api

app = rest_api.app  # run with: uvicorn geo_infer_bio.api.rest_api:app
```

Endpoints: `GET /`, `POST /analyze/sequence`, `POST /analyze/file` (FASTA upload),
`POST /visualize/spatial` (returns base64-encoded PNG plots), `GET /health`.

### GraphQL (Strawberry)

```python
from geo_infer_bio.api import graphql_api

app = graphql_api.app  # GraphQL playground at /graphql
```

Queries: `analyzeSequence`, `analyzeFile`, `visualizeSpatial`, `healthCheck`.
