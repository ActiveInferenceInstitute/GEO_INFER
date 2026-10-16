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

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
# Install the declared base service profile; workspace test groups are separate.
RUN uv pip install --no-sources --python /opt/runtime/bin/python .

FROM python:3.14-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV PATH=/opt/runtime/bin:$PATH \
    UV_PYTHON=/opt/runtime/bin/python \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GEO_INFER_OPS_CONFIG=/app/config/local.yaml \
    PYTHONPATH=/app/src
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/runtime /opt/runtime
COPY --from=builder /app/src /app/src
COPY config /app/config
RUN test "$(uv --version | cut -d ' ' -f 2)" = "0.13.1" \
    && python -c "import sys; assert sys.version_info[:2] == (3, 15)" \
    && useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/config /app/logs \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app \
    && chmod -R 777 /app/logs
USER appuser
VOLUME ["/app/config", "/app/logs"]
EXPOSE 8000 9090
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "geo_infer_ops.app"]
