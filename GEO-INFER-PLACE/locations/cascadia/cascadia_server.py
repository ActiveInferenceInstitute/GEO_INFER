#!/usr/bin/env python3
"""Lightweight HTTP server for Cascadia analysis outputs.

Serves HTML maps, GeoJSON layers, and pipeline status via a REST API.
Falls back to stdlib http.server if FastAPI/uvicorn are unavailable.

Usage:
    uv run python cascadia_server.py --port 8765 --output-dir output/
    uv run python cascadia_server.py --port 8765 --open-browser
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve paths relative to this file
# ---------------------------------------------------------------------------
CASCADIA_DIR = Path(__file__).resolve().parent
CONFIG_DIR = CASCADIA_DIR / "config"

try:
    import fastapi
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app(output_dir: Path) -> fastapi.FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="Cascadia Bioregion Analysis API",
        description="REST API for Cascadia pipeline outputs, maps, and ecological data.",
        version="1.0.0",
    )

    viz_dir = output_dir / "visualizations" / "interactive"

    # -- Redirect root to bioregion map -----------------------------------------
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/map/bioregion")

    # -- Map endpoints -----------------------------------------------------------
    @app.get("/map/bioregion", response_class=HTMLResponse)
    def serve_bioregion_map():
        path = viz_dir / "cascadia_bioregion_map.html"
        if not path.exists():
            raise HTTPException(404, f"Bioregion map not found. Run the pipeline first: {path}")
        return FileResponse(str(path), media_type="text/html")

    @app.get("/map/agricultural", response_class=HTMLResponse)
    def serve_agricultural_map():
        path = viz_dir / "cascadia_interactive_map.html"
        if not path.exists():
            raise HTTPException(404, f"Agricultural map not found. Run the pipeline first: {path}")
        return FileResponse(str(path), media_type="text/html")

    # -- Status endpoint ---------------------------------------------------------
    @app.get("/api/status")
    def api_status():
        bioregion_map = viz_dir / "cascadia_bioregion_map.html"
        agricultural_map = viz_dir / "cascadia_interactive_map.html"
        status_data: dict = {
            "pipeline_ran": agricultural_map.exists() or bioregion_map.exists(),
            "maps": {
                "bioregion": bioregion_map.exists(),
                "agricultural": agricultural_map.exists(),
            },
            "config_files": {},
            "output_dir": str(output_dir),
        }
        config_files = [
            "cascadia_volcanoes.geojson",
            "cascadia_subduction_zone.geojson",
            "cascadia_major_watersheds.geojson",
            "cascadia_bioregion_boundary.geojson",
            "cascadia_salmon_esus.yaml",
            "cascadia_ecoregions.yaml",
            "cascadia_indigenous_territories.yaml",
            "cascadia_climate_zones.yaml",
        ]
        for cf in config_files:
            status_data["config_files"][cf] = (CONFIG_DIR / cf).exists()
        return JSONResponse(status_data)

    # -- GeoJSON layer endpoints -------------------------------------------------
    def _serve_geojson(filename: str) -> JSONResponse:
        path = CONFIG_DIR / filename
        if not path.exists():
            raise HTTPException(404, f"{filename} not found at {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(data)

    @app.get("/api/layers/volcanoes")
    def layer_volcanoes():
        return _serve_geojson("cascadia_volcanoes.geojson")

    @app.get("/api/layers/csz")
    def layer_csz():
        return _serve_geojson("cascadia_subduction_zone.geojson")

    @app.get("/api/layers/watersheds")
    def layer_watersheds():
        return _serve_geojson("cascadia_major_watersheds.geojson")

    @app.get("/api/layers/bioregion")
    def layer_bioregion():
        return _serve_geojson("cascadia_bioregion_boundary.geojson")

    # -- YAML data endpoints (parsed as JSON) ------------------------------------
    def _serve_yaml(filename: str) -> JSONResponse:
        try:
            import yaml
        except ImportError:
            raise HTTPException(500, "pyyaml not installed")
        path = CONFIG_DIR / filename
        if not path.exists():
            raise HTTPException(404, f"{filename} not found at {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return JSONResponse(data or {})

    @app.get("/api/salmon")
    def api_salmon():
        return _serve_yaml("cascadia_salmon_esus.yaml")

    @app.get("/api/ecoregions")
    def api_ecoregions():
        return _serve_yaml("cascadia_ecoregions.yaml")

    @app.get("/api/indigenous")
    def api_indigenous():
        return _serve_yaml("cascadia_indigenous_territories.yaml")

    @app.get("/api/climate")
    def api_climate():
        return _serve_yaml("cascadia_climate_zones.yaml")

    # -- H3 hexagon data endpoint -----------------------------------------------
    @app.get("/api/h3/{resolution}")
    def api_h3(resolution: int):
        if not 1 <= resolution <= 15:
            raise HTTPException(400, "H3 resolution must be between 1 and 15")
        # Look for cached H3 data from pipeline outputs
        for pattern in [
            f"*res{resolution}*.geojson",
            f"*h3_{resolution}*.geojson",
            f"*unified*{resolution}*.geojson",
        ]:
            matches = list(output_dir.glob(f"**/{pattern}"))
            if matches:
                with open(matches[0], encoding="utf-8") as f:
                    return JSONResponse(json.load(f))
        return JSONResponse(
            {
                "type": "FeatureCollection",
                "features": [],
                "note": f"No H3 data found for resolution {resolution}. Run the pipeline first.",
            }
        )

    return app


# ---------------------------------------------------------------------------
# Stdlib fallback server
# ---------------------------------------------------------------------------


def run_stdlib_server(output_dir: Path, port: int) -> None:
    """Minimal stdlib HTTP server when FastAPI is not available."""
    import http.server
    import socketserver
    import functools

    viz_dir = output_dir / "visualizations" / "interactive"
    serve_dir = str(viz_dir) if viz_dir.exists() else str(output_dir)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=serve_dir)
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving {serve_dir} at http://localhost:{port}")
        print("Note: Full API requires FastAPI. Install with: pip install fastapi uvicorn")
        httpd.serve_forever()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cascadia analysis HTTP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python cascadia_server.py --port 8765
  uv run python cascadia_server.py --port 8765 --output-dir output/ --open-browser
        """,
    )
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Pipeline output directory"
    )
    parser.add_argument("--open-browser", action="store_true", help="Open browser after starting")
    parser.add_argument(
        "--reload", action="store_true", help="Enable uvicorn auto-reload (dev only)"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"http://{args.host}:{args.port}"
    print(f"🌲 Cascadia Server starting at {url}")
    print(f"   Output dir: {output_dir}")
    print(f"   Config dir: {CONFIG_DIR}")

    if args.open_browser:
        import threading
        import webbrowser
        import time

        def _open():
            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    if _FASTAPI_AVAILABLE:
        app = create_app(output_dir)
        print(f"   API docs:  {url}/docs")
        print(f"   Bioregion: {url}/map/bioregion")
        print(f"   Status:    {url}/api/status")
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    else:
        print("FastAPI/uvicorn not available — using stdlib fallback (no API endpoints)")
        run_stdlib_server(output_dir, args.port)


if __name__ == "__main__":
    main()
