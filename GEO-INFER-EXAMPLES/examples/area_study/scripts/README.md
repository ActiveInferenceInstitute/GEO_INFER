# GEO-INFER-EXAMPLES/examples/area_study/scripts

Scripts workspace within `GEO-INFER-EXAMPLES`.

## Contents

- `dashboard_app.py`
- `launch_dashboard.py`
- `run_example.py`
- `show_results.py`
- `simple_launch.py`
- `quick_launch.sh`

## Public Interface

- `dashboard_app.py:setup_logging` (function)
- `dashboard_app.py:StreamlitAreaStudyDashboard` (class)
- `dashboard_app.py:main` (function)
- `launch_dashboard.py:setup_logging` (function)
- `launch_dashboard.py:AreaStudyDashboard` (class)
- `launch_dashboard.py:open_browser` (function)
- `launch_dashboard.py:check_server_connection` (function)
- `launch_dashboard.py:run_streamlit_app` (function)
- `launch_dashboard.py:check_dependencies` (function)
- `launch_dashboard.py:main` (function)
- `run_example.py:setup_logging` (function)
- `run_example.py:ComprehensiveAreaStudy` (class)
- `run_example.py:main` (function)
- `show_results.py:setup_logging` (function)
- `show_results.py:AreaStudyConsoleViewer` (class)
- `show_results.py:main` (function)
- `simple_launch.py:check_dependencies` (function)
- `simple_launch.py:launch_dashboard` (function)
- `simple_launch.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-EXAMPLES`
- Package: `geo_infer_examples`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-examples`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES`

## Dependencies

- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
