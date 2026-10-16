# Named frozen CPU service profile; independent of the formatter/tool runtime.
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_PYTHON=/opt/runtime/bin/python
RUN uv venv --python /usr/local/bin/python /opt/runtime
WORKDIR /build
COPY . .
RUN uv export --quiet --frozen --package geo-infer-iot --no-default-groups \
    --no-emit-workspace --format requirements-txt --output-file /tmp/requirements.txt \
    && uv pip install --python /opt/runtime/bin/python --require-hashes -r /tmp/requirements.txt \
    && uv pip install --python /opt/runtime/bin/python --no-sources --no-deps ./GEO-INFER-BAYES ./GEO-INFER-SPACE ./GEO-INFER-TIME ./GEO-INFER-IOT \
    && uv pip check --python /opt/runtime/bin/python

FROM python:3.12-slim-bookworm AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV PATH=/opt/runtime/bin:$PATH \
    UV_PYTHON=/opt/runtime/bin/python \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/runtime /opt/runtime
WORKDIR /app
COPY GEO-INFER-IOT/config/ /app/config/
COPY GEO-INFER-IOT/deployment/docker-entrypoint.sh /usr/local/bin/
ENV CONFIG_PATH=/app/config/service.yaml
RUN test "$(uv --version | cut -d ' ' -f 2)" = "0.13.1" \
    && python -c "import sys; assert sys.version_info[:2] == (3, 12)" \
    && useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/config /app/logs /app/data /app/output \
    && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"
ENTRYPOINT ["/bin/bash", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "geo_infer_iot.api.sensor_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
