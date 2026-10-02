"""Cascadia command entrypoint; behavior lives in GEO-INFER-PLACE."""

from geo_infer_place.locations.cascadia.server import create_app, main

__all__ = ["create_app", "main"]

if __name__ == "__main__":
    main()
