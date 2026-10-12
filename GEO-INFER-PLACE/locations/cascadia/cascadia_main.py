"""Cascadia command entrypoint; behavior lives in GEO-INFER-PLACE."""

from geo_infer_place.locations.cascadia.cli import main, parse_counties

__all__ = ["main", "parse_counties"]

if __name__ == "__main__":
    main()
