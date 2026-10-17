---
name: geo-infer-bio
description: Biological sequence analysis and spatially-referenced bio data processing. Use when analyzing DNA sequences (GC content, motifs, coding regions, alignments), validating biological/spatial data, visualizing sequence metrics, or loading climate (WorldClim), soil (SoilGrids), and microbiome (EMP) datasets with coordinates.
prerequisites:
  required: []
  recommended: []
difficulty: intermediate
estimated_time: 30min
examples_dir: ../GEO-INFER-BIO/examples/
---

# GEO-INFER-BIO

## Instructions

### Core Capabilities

- **Sequence analysis**: GC content, motif detection, coding-region prediction, pairwise alignment (Biopython)
- **Data validation**: DNA/RNA/protein sequences, spatial coordinates, GC bounds, alignment checks
- **Visualization**: spatial distribution, GC distribution, motif density, coding potential plots
- **Climate data**: sampling WorldClim bioclimatic rasters (GeoTIFF, requires `rasterio`) at coordinates
- **Soil data**: querying ISRIC SoilGrids REST endpoint and loading custom soil CSV/TSV files
- **Microbiome data**: loading EMP metadata with spatial filtering and quality control
- **APIs**: FastAPI REST endpoints and a Strawberry GraphQL API for sequence analysis

### Key Imports

```python
from geo_infer_bio import (
    SequenceAnalyzer,
    DataValidator,
    BioVisualizer,
    ClimateDataProcessor,
    ClimateDataset,
    MicrobiomeDataLoader,
    MicrobiomeDataset,
    SoilDataIntegrator,
    SoilDataset,
)

# Submodules
from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer
from geo_infer_bio.utils.validation import DataValidator
from geo_infer_bio.utils.visualization import BioVisualizer
from geo_infer_bio.climate import ClimateDataProcessor
from geo_infer_bio.microbiome import MicrobiomeDataLoader
from geo_infer_bio.soil import SoilDataIntegrator
from geo_infer_bio.api import rest_api  # FastAPI app (`rest_api.app`)
from geo_infer_bio.api import graphql_api  # Strawberry schema (`graphql_api.app`)
```

## Examples

```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from geo_infer_bio import SequenceAnalyzer, DataValidator

analyzer = SequenceAnalyzer()
validator = DataValidator()

seq = Seq("ATCGATCGAAACCCGGGTTT")
validator.validate_sequence(seq, "DNA")          # True (RNA uses A/U/C/G)

analyzer.calculate_gc_content(seq)               # float, percent
motifs = analyzer.find_motifs(seq, motif_length=6)  # {motif: [positions]}
regions = analyzer.predict_coding_regions(seq)   # [{frame, start, end}, ...]

records = [SeqRecord(Seq("ATCG"), id="a"), SeqRecord(Seq("ATGG"), id="b")]
alignment = analyzer.align_sequences(records)    # pairwise alignment of first two
```

```python
from geo_infer_bio import SoilDataIntegrator, MicrobiomeDataLoader

soil = SoilDataIntegrator()
# Network call to ISRIC SoilGrids:
# dataset = soil.load_soilgrids_data(
#     coordinates=[(37.7, -122.4)], properties=["phh2o", "soc"], depths=["0-5cm"]
# )

micro = MicrobiomeDataLoader()
# dataset = micro.load_emp_data(metadata_path="emp_qiime_mapping_qc_filtered.tsv")
# dataset.filter_by_coordinates((-180, -90, 180, 90))
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog BIO` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Requires Python 3.11+ and Biopython; optional `rasterio` (extra: `integrations`) is needed only for WorldClim/custom climate rasters.
- Network-loading methods (`load_soilgrids_data`, `load_worldclim_data` with local rasters, `load_emp_data` from local files) need real data on disk or a live REST endpoint; there are no synthetic fallbacks.
- Dataset classes (`ClimateDataset`, `SoilDataset`, `MicrobiomeDataset`) expose `get_*` accessors, `export_for_h3_integration()` (plain dict of coordinates/metrics for downstream spatial modules), and filtering.

### Integrations

- SPACE: dataset classes provide `export_for_h3_integration()` producing coordinate/data dicts suitable for H3 ingestion in GEO-INFER-SPACE (no direct import at runtime).
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module BIO`
