# Runtime default: stable CPython 3.15, installed with exact uv 0.13.1.
# The published 3.14 image supplies system libraries until a 3.15-slim tag exists.
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN test "$(uv --version | cut -d ' ' -f 2)" = "0.13.1" \
    && uv python install 3.15 \
    && uv venv --seed --python 3.15 /opt/runtime
ENV PATH=/opt/runtime/bin:$PATH \
    UV_PYTHON=/opt/runtime/bin/python \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY . .
# Select this standalone module's declared base dependencies. Workspace-only
# integrations remain in the producer metadata and are not installed here.
RUN uv pip install --no-sources --python /opt/runtime/bin/python .
RUN mkdir -p /app/data /app/output \
    && test "$(uv --version | cut -d ' ' -f 2)" = "0.13.1" \
    && python -c "import sys; assert sys.version_info[:2] == (3, 15)"
EXPOSE 8000
CMD ["uv", "run", "--no-project", "--no-sync", "uvicorn", "geo_infer_bio.api.rest_api:app", "--host", "0.0.0.0", "--port", "8000"]
