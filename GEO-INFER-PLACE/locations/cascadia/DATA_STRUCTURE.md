# Cascadia Framework Data Structure

## Overview
The Cascadia Agricultural Analysis Framework uses a clean separation of concerns:

- **Source code** in `src/`
- **Tests** in `tests/`
- **Generated outputs** in `generated/`

## Directory Structure

```
 cascadia/ ├── src/ # All source code │ ├── data_modules/ # Data acquisition modules │ │ ├── zoning/ # Zoning data module │ │ ├── current_use/ # Current use module │ │ ├── ownership/ # Ownership module │ │ ├── improvements/ # Improvements module │ │ ├── water_rights/ # Water rights module │ │ ├── ground_water/ # Ground water module │ │ ├── power_source/ # Power source module │ │ └── mortgage_debt/ # Mortgage debt module │ └── core/ # Core processing utilities │ ├── enhanced_data_manager.py │ ├── enhanced_h3_fusion.py │ ├── analysis_engine.py │ ├── data_processor.py │ ├── reporting_engine.py │ └── visualization/ # Visualization engines │ ├── tests/ # All tests │ ├── unit/ # Unit tests │ ├── integration/ # Integration tests │ └── fixtures/ # Test fixtures │ ├── generated/ # All generated outputs │ ├── output/ # Analysis outputs │ ├── cache/ # H3 cache files │ └── logs/ # Log files │ ├── config/ # Configuration files │ ├── analysis_config.yaml # Main analysis configuration │ └── counties/ # County-specific configs │ ├── data/ # Shared static data │ └── cascadia_main.py # Entry point
```
 ## Module Data Structure Each data module in `src/data_modules/` follows:
```
 module_name/ ├── __init__.py ├── geo_infer_MODULE.py # Main module implementation ├── data_sources.py # Data source definitions └── data/ ├── empirical/ # Real acquired data ├── synthetic/ # Generated test data ├── cache/ # H3-processed cache ├── processed/ # Final outputs └── raw/ # Unprocessed data
```
 ## Data Flow 1. **Acquisition**: Real/synthetic data → `src/data_modules/*/data/` 2. **Processing**: Raw → H3 indexed → `*/data/cache/` 3. **Analysis**: Modules → Fusion → `generated/output/` ## Configuration Main config: `config/analysis_config.yaml` - H3 resolution - Target counties - Active modules - Output formats ## Running
```
bash # Validate configuration python cascadia_main.py --validate-config # Run analysis python cascadia_main.py # Cleanup generated files python cleanup_data.py
```