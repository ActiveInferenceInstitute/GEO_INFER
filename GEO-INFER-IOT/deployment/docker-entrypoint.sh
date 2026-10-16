#!/bin/bash
set -euo pipefail

# Keep the API process as PID 1 so container signals reach its own lifecycle.
# Configuration is read by the application factory; missing/invalid files fail.
mkdir -p /app/logs /app/data
exec "$@"
