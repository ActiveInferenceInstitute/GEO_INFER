#!/usr/bin/env python3
"""Provision and verify the DuckDB Spatial extension for the installed runtime."""

import json

from geo_infer_data.utils.duckdb_spatial import provision_spatial_extension

if __name__ == "__main__":
    print(json.dumps(provision_spatial_extension(), sort_keys=True))
