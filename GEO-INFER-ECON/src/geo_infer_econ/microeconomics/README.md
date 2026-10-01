# GEO-INFER-ECON/src/geo_infer_econ/microeconomics

Microeconomics workspace within `GEO-INFER-ECON`.

## Contents

- `__init__.py`
- `behavioral_economics.py`
- `consumer_theory.py`
- `game_theory.py`
- `market_structure.py`
- `producer_theory.py`

## Public Interface

- `behavioral_economics.py:BehavioralParameters` (class)
- `behavioral_economics.py:ProspectTheory` (class)
- `behavioral_economics.py:BoundedRationality` (class)
- `behavioral_economics.py:SocialPreferences` (class)
- `behavioral_economics.py:TimePreferences` (class)
- `behavioral_economics.py:MentalAccounting` (class)
- `behavioral_economics.py:NudgeAnalysis` (class)
- `behavioral_economics.py:BehavioralEconomicsEngine` (class)
- `consumer_theory.py:ConsumerProfile` (class)
- `consumer_theory.py:UtilityFunctions` (class)
- `consumer_theory.py:DemandFunctions` (class)
- `consumer_theory.py:ConsumerChoiceModels` (class)
- `consumer_theory.py:WelfareAnalysis` (class)
- `consumer_theory.py:ConsumerSurplus` (class)
- `consumer_theory.py:example_consumer_analysis` (function)
- `game_theory.py:Game` (class)
- `game_theory.py:ExtensiveFormGame` (class)
- `game_theory.py:NashEquilibrium` (class)
- `game_theory.py:AuctionTheory` (class)
- `game_theory.py:EvolutionaryGames` (class)

## Module Metadata

- Module: `GEO-INFER-ECON`
- Package: `geo_infer_econ`
- Version: `0.3.0`
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
