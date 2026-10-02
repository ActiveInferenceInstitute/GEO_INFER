# GEO-INFER-ECON/src/geo_infer_econ/bioregional

Bioregional workspace within `GEO-INFER-ECON`.

## Contents

- `__init__.py`
- `bioregional_governance.py`
- `bioregional_markets.py`
- `circular_economy.py`
- `ecological_economics.py`
- `ecosystem_services.py`
- `natural_capital.py`
- `spatial_ecology.py`
- `sustainability_metrics.py`

## Public Interface

- `bioregional_governance.py:BioregionalGovernanceModels` (class)
- `bioregional_governance.py:CommunityResourceManagement` (class)
- `bioregional_governance.py:AdaptiveManagementSystems` (class)
- `bioregional_governance.py:StakeholderEngagement` (class)
- `bioregional_governance.py:CooperativeEconomics` (class)
- `bioregional_markets.py:BioregionalAsset` (class)
- `bioregional_markets.py:MarketParticipant` (class)
- `bioregional_markets.py:EcosystemServiceCredit` (class)
- `bioregional_markets.py:BioregionalMarketDesign` (class)
- `bioregional_markets.py:EcosystemServicesMarkets` (class)
- `bioregional_markets.py:BiodiversityMarkets` (class)
- `bioregional_markets.py:LocalFoodSystems` (class)
- `bioregional_markets.py:CarbonMarkets` (class)
- `bioregional_markets.py:WaterMarkets` (class)
- `bioregional_markets.py:example_bioregional_market` (function)
- `circular_economy.py:CircularEconomyModels` (class)
- `circular_economy.py:MaterialFlowAnalysis` (class)
- `circular_economy.py:IndustrialEcologyModels` (class)
- `circular_economy.py:WasteToResourceSystems` (class)
- `circular_economy.py:RegenerativeDesign` (class)

## Module Metadata

- Module: `GEO-INFER-ECON`
- Package: `geo_infer_econ`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-econ`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ECON`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `geopandas>=0.13.0`
- `shapely>=2.0.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.12.0`
- `h3>=4.5.0,<5`
- `requests>=2.28.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ECON
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
