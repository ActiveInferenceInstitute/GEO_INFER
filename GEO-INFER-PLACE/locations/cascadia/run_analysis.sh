#!/bin/bash
set -euo pipefail
location_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$location_dir/../../.." && pwd)"
uv sync --project "$repo_dir" --package geo-infer-place --extra cascadia --all-groups
exec uv run --project "$repo_dir" --no-sync python "$location_dir/cascadia_main.py" "$@"
