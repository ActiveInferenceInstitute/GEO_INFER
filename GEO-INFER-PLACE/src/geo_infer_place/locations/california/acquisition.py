"""Bounded capture of primary county, CAL FIRE, USGS and USDM source bytes."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Final, Self
from urllib.parse import urlencode, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    TypeAdapter,
    model_validator,
)

from geo_infer_place.core.regional_layers import _download
from geo_infer_place.locations.california.contracts import (
    RAW_BYTES,
    CaliforniaError,
    CapturedFile,
    ObservatoryConfig,
    SourceReceipt,
    StrictModel,
    digest,
    last_tuesday,
    strict_bytes,
    write_record,
)

COUNTIES: Final = "https://services.gis.ca.gov/arcgis/rest/services/Boundaries/CA_Counties/FeatureServer/0"
FIRES: Final = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/California_Historic_Fire_Perimeters/FeatureServer/0"
EARTHQUAKES: Final = "https://earthquake.usgs.gov/fdsnws/event/1/query"
DROUGHT: Final = "https://droughtmonitor.unl.edu/data/shapefiles_m"
CAPTURE_BYTES: Final = 128 * 1024 * 1024
LOGGER = logging.getLogger(__name__)
RETRYABLE_FAILURES: Final = {
    "Source network request failed",
    *(f"Source returned HTTP status {status}" for status in (408, 429, 500, 502, 503, 504)),
}


class SourceEndpoints(StrictModel):
    counties: str = COUNTIES
    fires: str = FIRES
    earthquakes: str = EARTHQUAKES
    drought: str = DROUGHT

    @model_validator(mode="after")
    def valid_locations(self) -> Self:
        for location in (self.counties, self.fires, self.earthquakes, self.drought):
            parsed = urlsplit(location)
            if (
                len(location) > 4096
                or not parsed.hostname
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
                or (
                    parsed.scheme != "https"
                    and not (
                        parsed.scheme == "http"
                        and parsed.hostname in {"127.0.0.1", "::1"}
                    )
                )
            ):
                raise CaliforniaError(
                    "source endpoints require HTTPS or explicit local HTTP"
                )
        return self


class ServiceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    name: str
    objectIdField: str
    maxRecordCount: int = Field(gt=0)
    editingInfo: dict[str, JsonValue] | None = None


class ObjectIds(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    objectIds: tuple[int, ...] = Field(min_length=1, max_length=15000)


class Feature(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    properties: dict[str, JsonValue]


class Page(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    type: str
    features: tuple[Feature, ...]
    exceededTransferLimit: bool = False


def acquire_sources(
    config: ObservatoryConfig,
    output: Path,
    *,
    timeout: float = 600,
    endpoints: SourceEndpoints = SourceEndpoints(),
) -> Path:
    """Retain requests and exact response bytes; failures never seal a source receipt."""
    if type(timeout) not in (float, int) or not 1 <= timeout <= 1800:
        raise CaliforniaError("capture timeout must be 1..1800 seconds")
    output.mkdir(parents=True, exist_ok=False)
    write_record(output / "capture_design.json", config)
    deadline = time.monotonic() + timeout
    captured: list[CapturedFile] = []

    def fetch(name: str, url: str) -> Path:
        remaining = CAPTURE_BYTES - sum(item.bytes for item in captured)
        if remaining <= 0:
            raise CaliforniaError("capture exceeds 128 MiB")
        attempt = 1
        while True:
            try:
                raw = _download(url, deadline, min(RAW_BYTES, remaining))
                break
            except TimeoutError:
                if attempt == 3 or time.monotonic() >= deadline:
                    raise
                LOGGER.warning("Retrying %s after timeout (attempt %d/3)", name, attempt)
            except RuntimeError as error:
                if str(error) not in RETRYABLE_FAILURES or attempt == 3:
                    raise
                LOGGER.warning("Retrying %s after %s (attempt %d/3)", name, error, attempt)
            attempt += 1
        path = output / name
        with path.open("xb") as stream:
            stream.write(raw)
        captured.append(
            CapturedFile(
                name=name,
                url=url,
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),
            )
        )
        return path

    def arcgis(name: str, service: str, where: str, fields: str) -> tuple[int, ...]:
        metadata = ServiceMetadata.model_validate_json(
            strict_bytes(fetch(f"{name}_metadata.json", service + "?f=json"))
        )
        ids_url = (
            service
            + "/query?"
            + urlencode({"f": "json", "where": where, "returnIdsOnly": "true"})
        )
        ids = ObjectIds.model_validate_json(
            strict_bytes(fetch(f"{name}_ids.json", ids_url))
        ).objectIds
        if len(set(ids)) != len(ids):
            raise CaliforniaError("ArcGIS ID selection contains duplicates")
        selected = tuple(sorted(ids))
        seen: list[int] = []
        for offset in range(0, len(selected), 250):
            requested = selected[offset : offset + 250]
            query = dict(
                f="geojson",
                objectIds=",".join(map(str, requested)),
                outFields=fields,
                outSR=4326,
                returnGeometry="true",
                geometryPrecision=config.geometry_precision,
                maxAllowableOffset=config.geometry_offset_degrees,
            )
            page = Page.model_validate_json(
                strict_bytes(
                    fetch(
                        f"{name}_{offset:05d}.geojson",
                        service + "/query?" + urlencode(query),
                    )
                )
            )
            page_ids = tuple(
                TypeAdapter(StrictInt).validate_python(
                    feature.properties[metadata.objectIdField]
                )
                for feature in page.features
            )
            if (
                page.type != "FeatureCollection"
                or page.exceededTransferLimit
                or set(page_ids) != set(requested)
                or len(page_ids) != len(requested)
            ):
                raise CaliforniaError(
                    "ArcGIS page selection is incomplete or duplicated"
                )
            seen.extend(page_ids)
        after = ServiceMetadata.model_validate_json(
            strict_bytes(fetch(f"{name}_metadata_after.json", service + "?f=json"))
        )
        after_ids = ObjectIds.model_validate_json(
            strict_bytes(fetch(f"{name}_ids_after.json", ids_url))
        ).objectIds
        if (
            metadata != after
            or selected != tuple(sorted(after_ids))
            or set(seen) != set(selected)
        ):
            raise CaliforniaError("ArcGIS service changed during source capture")
        return selected

    counties = arcgis("counties", endpoints.counties, "1=1", "OBJECTID,County,FIPS")
    fires = arcgis(
        "fires",
        endpoints.fires,
        f"YEAR_ >= {config.first_year} AND YEAR_ <= {config.target_year} AND STATE = 'CA'",
        "OBJECTID,YEAR_,FIRE_NAME,STATE,GIS_ACRES",
    )
    for year in range(config.first_year, config.target_year + 1):
        query = dict(
            format="geojson",
            starttime=f"{year}-01-01T00:00:00",
            endtime=f"{year}-12-31T23:59:59.999",
            minlatitude=32,
            maxlatitude=42.1,
            minlongitude=-125,
            maxlongitude=-114,
            minmagnitude=config.earthquake_magnitude,
            eventtype="earthquake",
            orderby="time-asc",
            limit=20000,
        )
        path = fetch(
            f"earthquakes_{year}.geojson",
            endpoints.earthquakes + "?" + urlencode(query),
        )
        page = Page.model_validate_json(strict_bytes(path))
        if page.type != "FeatureCollection" or len(page.features) >= 20000:
            raise CaliforniaError("USGS event selection is truncated")
        day = last_tuesday(year)
        fetch(f"drought_{day}.zip", f"{endpoints.drought}/usdm_{day}_M.zip")
    receipt = SourceReceipt(
        schema_version="geo-infer-place/california-sources/1",
        config=config,
        files=tuple(captured),
        county_ids=counties,
        fire_ids=fires,
        complete=True,
    )
    return_path = output / "sources_receipt.json"
    write_record(return_path, receipt)
    return return_path


def check_sources(folder: Path) -> SourceReceipt:
    """Check supplied-receipt integrity without network or authenticity claims."""
    if folder.is_symlink():
        raise CaliforniaError("source folder is a symlink")
    receipt = SourceReceipt.model_validate_json(
        strict_bytes(folder / "sources_receipt.json", limit=1024 * 1024)
    )
    expected = {item.name for item in receipt.files} | {
        "sources_receipt.json",
        "capture_design.json",
    }
    if {path.name for path in folder.iterdir()} != expected:
        raise CaliforniaError("source file roster differs")
    if (
        ObservatoryConfig.model_validate_json(
            strict_bytes(folder / "capture_design.json")
        )
        != receipt.config
    ):
        raise CaliforniaError("source capture design differs")
    for item in receipt.files:
        path = folder / item.name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item.bytes
            or digest(path) != item.sha256
        ):
            raise CaliforniaError("captured source bytes differ")
    return receipt
