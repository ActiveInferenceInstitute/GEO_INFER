# The frozen IoT profile needs CPython 3.13: locked pydantic-core 2.33.2
# embeds PyO3 0.24.1, whose native build refuses later Python minors.
# Requalify 3.15 after a separately reviewed producer-lock dependency update.
# GDAL/PROJ/Fortran prerequisites are independent of the Python exception.
FROM python:3.14-slim AS builder
# Published system seed; the executable runtime follows the qualified profile.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN test "$(uv --version | cut -d ' ' -f 2)" = "0.13.1" \
    && uv python install 3.13 \
    && uv venv --seed --python 3.13 /opt/runtime
ENV PATH=/opt/runtime/bin:$PATH \
    UV_PYTHON=/opt/runtime/bin/python \
    UV_CONCURRENT_BUILDS=1 \
    CARGO_BUILD_JOBS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libgdal-dev proj-bin gfortran \
    libopenblas-dev ninja-build libpq-dev postgresql-client redis-tools \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml uv.lock uv.toml ./
# Workspace discovery needs each member's metadata, but no unrelated source.
COPY GEO-INFER-ACT/pyproject.toml ./GEO-INFER-ACT/
COPY GEO-INFER-AG/pyproject.toml ./GEO-INFER-AG/
COPY GEO-INFER-AGENT/pyproject.toml ./GEO-INFER-AGENT/
COPY GEO-INFER-AI/pyproject.toml ./GEO-INFER-AI/
COPY GEO-INFER-ANT/pyproject.toml ./GEO-INFER-ANT/
COPY GEO-INFER-API/pyproject.toml ./GEO-INFER-API/
COPY GEO-INFER-APP/pyproject.toml ./GEO-INFER-APP/
COPY GEO-INFER-ART/pyproject.toml ./GEO-INFER-ART/
COPY GEO-INFER-BAYES/pyproject.toml ./GEO-INFER-BAYES/
COPY GEO-INFER-BIO/pyproject.toml ./GEO-INFER-BIO/
COPY GEO-INFER-CIV/pyproject.toml ./GEO-INFER-CIV/
COPY GEO-INFER-CLIMATE/pyproject.toml ./GEO-INFER-CLIMATE/
COPY GEO-INFER-COG/pyproject.toml ./GEO-INFER-COG/
COPY GEO-INFER-COMMS/pyproject.toml ./GEO-INFER-COMMS/
COPY GEO-INFER-DATA/pyproject.toml ./GEO-INFER-DATA/
COPY GEO-INFER-ECON/pyproject.toml ./GEO-INFER-ECON/
COPY GEO-INFER-EDU/pyproject.toml ./GEO-INFER-EDU/
COPY GEO-INFER-EMERGENCY/pyproject.toml ./GEO-INFER-EMERGENCY/
COPY GEO-INFER-ENERGY/pyproject.toml ./GEO-INFER-ENERGY/
COPY GEO-INFER-EXAMPLES/pyproject.toml ./GEO-INFER-EXAMPLES/
COPY GEO-INFER-FOREST/pyproject.toml ./GEO-INFER-FOREST/
COPY GEO-INFER-GIT/pyproject.toml ./GEO-INFER-GIT/
COPY GEO-INFER-HEALTH/pyproject.toml ./GEO-INFER-HEALTH/
COPY GEO-INFER-INSURANCE/pyproject.toml ./GEO-INFER-INSURANCE/
COPY GEO-INFER-INTRA/pyproject.toml ./GEO-INFER-INTRA/
COPY GEO-INFER-IOT/pyproject.toml ./GEO-INFER-IOT/
COPY GEO-INFER-LOG/pyproject.toml ./GEO-INFER-LOG/
COPY GEO-INFER-MARINE/pyproject.toml ./GEO-INFER-MARINE/
COPY GEO-INFER-MATH/pyproject.toml ./GEO-INFER-MATH/
COPY GEO-INFER-METAGOV/pyproject.toml ./GEO-INFER-METAGOV/
COPY GEO-INFER-NORMS/pyproject.toml ./GEO-INFER-NORMS/
COPY GEO-INFER-OPS/pyproject.toml ./GEO-INFER-OPS/
COPY GEO-INFER-ORG/pyproject.toml ./GEO-INFER-ORG/
COPY GEO-INFER-PEP/pyproject.toml ./GEO-INFER-PEP/
COPY GEO-INFER-PLACE/pyproject.toml ./GEO-INFER-PLACE/
COPY GEO-INFER-REQ/pyproject.toml ./GEO-INFER-REQ/
COPY GEO-INFER-RISK/pyproject.toml ./GEO-INFER-RISK/
COPY GEO-INFER-SEC/pyproject.toml ./GEO-INFER-SEC/
COPY GEO-INFER-SIM/pyproject.toml ./GEO-INFER-SIM/
COPY GEO-INFER-SPACE/pyproject.toml ./GEO-INFER-SPACE/
COPY GEO-INFER-SPM/pyproject.toml ./GEO-INFER-SPM/
COPY GEO-INFER-TEST/pyproject.toml ./GEO-INFER-TEST/
COPY GEO-INFER-TIME/pyproject.toml ./GEO-INFER-TIME/
COPY GEO-INFER-TRANSPORT/pyproject.toml ./GEO-INFER-TRANSPORT/
COPY GEO-INFER-WATER/pyproject.toml ./GEO-INFER-WATER/
RUN uv export --quiet --frozen --package geo-infer-iot --no-default-groups \
    --no-emit-workspace --format requirements-txt --output-file /tmp/iot-requirements.txt \
    && uv pip install --python /opt/runtime/bin/python --require-hashes -r /tmp/iot-requirements.txt
COPY GEO-INFER-BAYES/README.md ./GEO-INFER-BAYES/
COPY GEO-INFER-BAYES/src/ ./GEO-INFER-BAYES/src/
COPY GEO-INFER-SPACE/README.md ./GEO-INFER-SPACE/
COPY GEO-INFER-SPACE/src/ ./GEO-INFER-SPACE/src/
COPY GEO-INFER-TIME/README.md ./GEO-INFER-TIME/
COPY GEO-INFER-TIME/src/ ./GEO-INFER-TIME/src/
COPY GEO-INFER-IOT/README.md ./GEO-INFER-IOT/
COPY GEO-INFER-IOT/src/ ./GEO-INFER-IOT/src/
RUN uv pip install --python /opt/runtime/bin/python --no-sources --no-deps \
    ./GEO-INFER-BAYES ./GEO-INFER-SPACE ./GEO-INFER-TIME ./GEO-INFER-IOT \
    && uv pip install --python /opt/runtime/bin/python gunicorn \
    && uv pip check --python /opt/runtime/bin/python

FROM python:3.14-slim AS production
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal36 proj-bin libopenblas0-pthread libpq5 redis-tools curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/runtime /opt/runtime
ENV PATH=/opt/runtime/bin:$PATH \
    UV_PYTHON=/opt/runtime/bin/python \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    CONFIG_PATH=/app/config/iot_config.yaml
WORKDIR /app
COPY GEO-INFER-IOT/src/ /app/src/
COPY GEO-INFER-IOT/config/ /app/config/
COPY GEO-INFER-IOT/deployment/docker-entrypoint.sh /usr/local/bin/
RUN test "$(uv --version | cut -d ' ' -f 2)" = "0.13.1" \
    && python -c "import sys; assert sys.version_info[:2] == (3, 13)" \
    && groupadd -r geo_infer && useradd -r --create-home -g geo_infer geo_infer \
    && mkdir -p /app/logs /app/data /app/config \
    && chown -R geo_infer:geo_infer /app \
    && chmod +x /usr/local/bin/docker-entrypoint.sh
ENV PYTHONPATH=/app/src
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
USER geo_infer
EXPOSE 8000 8001
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "geo_infer_iot.api.sensor_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
