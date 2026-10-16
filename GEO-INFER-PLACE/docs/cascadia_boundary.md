# Complete sourced Cascadia watershed boundary

The committed snapshot is Aquila Flower's **Cascadia Bioregion Boundary**,
[publisher item c6497941559d433da98e286fb3f63551](https://www.arcgis.com/home/item.html?id=c6497941559d433da98e286fb3f63551).
The publisher identifies it as a watershed-based delineation informed by the
Cascadia Institute and CEC watershed/land data, and labels it
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The retained publisher metadata and `cascadia_boundary.license.md` carry its
attribution and reuse terms. This is that named source's delineation, not a
political, regulatory or universally agreed regional boundary.

The capture contains every returned source feature: native FID `1`, a valid
MultiPolygon with 178 polygon parts, including islands. Its WGS84 envelope is
approximately `[-144.59723, 40.11000, -109.76541, 61.68473]`. The service's
before/after data-edit epoch is `1624387343182` (2021-06-22 UTC); the receipt
records the separate actual retrieval time. The item overview's stored envelope
is not used as the geometry oracle: the full layer's WGS84 extent query, count,
and native-ID inventory independently accompany the unfiltered geometry query.

No clipping, simplification, polygon repair or coordinate rounding is performed.
The original metadata and geometry responses retain exact publisher bytes.
The normalized GeoJSON adds provenance and deterministic serialization while
preserving the complete original feature objects. Source area attributes are
retained without relabeling them as geodesic areas.

## Scope and rendering

`geo_infer_place.core.bioregion_visualization.create_bioregion_map` now renders
all four configured layers with the committed config. It verifies this boundary's
captured source/license/geometry hashes, fits the full extent, and displays
publisher and license credit above the basemap attribution. The default
OpenStreetMap basemap is intended for ordinary interactive viewing under its
[tile usage policy](https://operations.osmfoundation.org/policies/tiles/); no tile
archive or prefetch is created. Pass `tiles=None` to omit external basemap requests,
or select a named Folium provider explicitly. Leaflet assets still require their
configured CDNs. `.layers.json` includes the boundary's source,
license, epoch, native IDs, extent and original-response digest.

The other three USGS display layers keep their narrower documented study windows.
Their old `cascadia_layers.provenance.json` remains unchanged as dated evidence;
its earlier missing-boundary entry does not qualify or invalidate this separate
capture. Missing layers still raise by default, and explicitly incomplete maps
retain their visible notice and partial status.

The source's coastline/land masking should not be used for precise site membership.
For example, the retained polygon excludes the sampled Vancouver downtown point
`(-123.12, 49.28)` although a nearby source edge is about 657 m away. This is a
measured source-resolution limitation, not an inferred political exclusion or a
reason to silently repair the geometry. Inland reference points in Seattle and
Boise are included, while Sacramento is south of the source extent.

## Offline verification

```bash
uv run --no-sync python -m geo_infer_place.core.cascadia_boundary \
  GEO-INFER-PLACE/locations/cascadia/config --check
```

Installed packages carry the same capture:

```python
from pathlib import Path
import geo_infer_place
from geo_infer_place.core.cascadia_boundary import verify_cascadia_boundary

config = Path(geo_infer_place.__file__).resolve().parent / "locations/cascadia/config"
receipt = verify_cascadia_boundary(config)
assert receipt["feature_count"] == 1 and receipt["native_ids"] == [1]
```

`verify_cascadia_boundary` rebuilds the derived bytes from captured responses and
checks the entire file inventory, metadata identity/license, native IDs/count,
source epoch and full WGS84 extent. It is read-only and performs no network or
subprocess work. SHA-256 establishes snapshot integrity, not a publisher digital
signature or currency against later server edits.

## Explicit fresh acquisition

```bash
uv run --no-sync python -m geo_infer_place.core.cascadia_boundary \
  output/cascadia_boundary_capture_new --capture
```

Only the fixed publisher endpoints are queried. All response workers share one
five-minute parent-enforced network deadline; requests and total persisted data
are bounded to 20 MiB. Local geometry and file work remain cooperative, with a
prewrite deadline check. The capture checks all responses before writing and
refuses existing target files. Writes are not a multi-file transaction; a failed
attempt without a complete valid receipt remains unqualified. A fresh directory
is required for a new attempt, and earlier snapshots are never overwritten.

The capture retains item, service, before/after layer, count, native-ID, WGS84
extent and full GeoJSON responses. Source count/IDs/extent must agree, the source
epoch must remain unchanged across acquisition, and changed publisher/license
metadata requires renewed review. Metadata/source snapshots, normalized output,
notice and receipt have explicit portable Git byte policies in both the location
and packaged configuration copies.

## Behavioral acceptance

`tests/integration/test_cascadia_boundary.py` checks complete source replay,
packaged/legacy parity, actual strict Folium rendering and full viewport,
attribution, receipt tampering, invalid licenses/identities/epochs/extents/CRS,
truncation, duplicate/nonfinite JSON and real CRLF checkout. The existing regional
negative tests use an explicitly incomplete legacy fixture. The installed
`geo_infer_place[cascadia]` wheel profile independently verifies and renders the
captured boundary outside the checkout. No live network is required by those
regressions; the explicit acquisition run has its separate retained receipt.
