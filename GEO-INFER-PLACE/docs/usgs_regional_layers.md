# Sourced Cascadia regional display layers

The September 2026 snapshot supplies three authentic source layers for the
existing bioregion renderer. The separate October 2026
[licensed boundary capture](cascadia_boundary.md) now supplies the complete
publisher-defined watershed boundary. Its receipt is independent of the retained
September three-layer receipt. Missing files still fail strict rendering;
`allow_missing_layers=True` explicitly records partial maps. Loading all four
configured layers does not extend the USGS datasets beyond their captured windows.

| Layer | Snapshot | Interpretation |
| --- | --- | --- |
| Volcanoes | 24 points, native USGS/GVP `vnum` IDs | USGS volcano inventory inside the land study window; not a fixed list of 12 volcanoes or a live alert feed |
| Major watersheds | 13 HU4 subregion polygons, native HUC4 IDs | USGS WBD region 17 plus HUC4 1801, intersecting the land study window; hydrologic subregions, not a new delineation of the bioregion |
| Subduction zone | 1 named convergent plate-boundary feature, native OBJECTID | USGS compilation's `North America:Juan de Fuca` line intersecting the offshore window; no rupture probability, hazard surface or inferred slab polygon |

The land window is WGS84 `[-124.8, 40, -114.5, 49]`. The offshore window is
`[-130, 40, -120, 51]`, widened explicitly to include the offshore plate boundary.
Returned source geometries are clipped to the relevant window. The WBD query
requests source-side display generalization at `maxAllowableOffset=0.002`
degrees and six decimal places; these polygons are for regional display, not
parcel boundaries or regulatory determinations. Source areas remain source
areas and are not relabeled as areas of clipped polygons. The complete raw
responses preserve original coordinates, attributes and native IDs for review.
Clipping must preserve the layer's geometry type. Polygon contacts that reduce
to a line or point and line contacts that reduce to a point are excluded. Mixed
geometry collections fail validation instead of being labeled as watershed
polygons or tectonic lines. Nonfinite JSON numbers are rejected at ingestion.

## Reproduction and limits

```bash
uv run python -m geo_infer_place.core.regional_layers \
  GEO-INFER-PLACE/locations/cascadia/config --offline
```

Omit `--offline` to fetch the three explicitly configured USGS endpoints. A batch
accepts at most 1,000 source features per response and at most 20 MiB of persisted
raw responses, output layers and the acquisition receipt. Network requests run
in private worker processes under one five-minute **parent-enforced wall-clock
deadline** for the batch. Worker startup and imports consume that budget, and
later requests do not reset it. At expiry the parent terminates the worker,
including a slow-drip response that never triggers a socket inactivity timeout.
Cleanup and reaping have a separate one-second bound; failure to reap is an
explicit error. This bounds network acquisition, not every local operation:
geometry processing, serialization and filesystem writes remain local, with
deadline checks between phases rather than forced termination during a phase.

Workers use the active interpreter with `-I`, ignoring caller `PYTHONPATH` and
user-site packages, and do not launch subprocesses. POSIX cleanup terminates the
isolated process group; Windows cleanup terminates and reaps the direct worker.
Connection/read inactivity timeouts remain at most 10/30 seconds, shortened to
the remaining budget when a request starts. Redirects are rejected. URLs require
HTTPS except literal loopback IP addresses used by local tests; credentials and
fragments are rejected. The worker caps decoded response bytes, and the parent
checks the returned size again. Failure messages do not include response bodies.
Offline replay launches no workers and preserves the existing hash contract.

`tests/integration/test_regional_download_worker.py` exercises real loopback
HTTP and child processes: slow-drip and delayed-header deadlines, process reaping
and pipe closure, exact response bytes, compressed-response limits, HTTP failures,
invalid request budgets, import isolation, and preservation of existing artifacts
when a later request times out after a successful download. No external download
is needed for these tests.
The committed snapshot is approximately 1.3 MiB including source metadata.
Output files are prepared only after all three responses validate; each final
file is replaced atomically. This is not a multi-file filesystem transaction.

`cascadia_layers.provenance.json` records source URLs, raw-response SHA-256,
derived-file SHA-256, source/output counts, generation time and unresolved work.
The generation timestamp describes the local build, not a claimed source survey
or retrieval date. Each GeoJSON repeats its source digest, citation, study bounds
and explicit `whole_cascadia_bioregion=false`. Source snapshots are named
`cascadia_{volcanoes,watersheds,tectonics}.source.json`. No current alert synopsis,
image, eruption date or lahar drainage is invented from a volcano name.

The renderer's threat labels remove the literal suffix ` Threat` from USGS
`nvewsThreat`; the original category is retained in `nvews_threat_source`. Missing
eruption and lahar information is labeled as not provided by this source.

## Sources and reuse evidence

- The [USGS volcano API documentation](https://volcanoes.usgs.gov/vsc/api/volcanoApi/)
  identifies its GeoJSON inventory endpoint and makes its application data freely
  available. The [USGS copyrights policy](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)
  supplies the agency's public-domain context. Images linked from the API are not
  downloaded or redistributed by this acquisition.
- The [USGS WBD service metadata](https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer)
  identifies HU4 as hydrologic subregions, states that the data are open and
  non-proprietary with no use constraints, and reports a July 2026 refresh.
- The [USGS Cascadia database](https://www.usgs.gov/special-topics/subduction-zone-science/science/cascadia-subduction-zone-database)
  links its authoritative web map and [ScienceBase release](https://www.sciencebase.gov/catalog/item/623cf2a6d34e915b67d47586),
  DOI `10.5066/P9O69X6E`. Its original metadata XML is retained as
  `cascadia_tectonics.metadata.xml`; both access and use constraints are `none`.
  `cascadia_regional_source_metadata.json` records the metadata download URL and
  checksum. The complete 3.44 GB map package was not downloaded.

The earlier review did not verify a downloadable, licensed full-boundary vector;
its `cascadia_layers.provenance.json` and unresolved entry are retained as dated
three-layer evidence. The separate [boundary capture](cascadia_boundary.md) now
verifies the publisher's licensed feature service, full native-ID inventory and
extent, original responses and source epoch. Replaying this USGS-only acquirer
does not qualify that separate boundary; use its named verifier as well.

## Verification

`tests/integration/test_regional_layer_acquisition.py` replays the captured raw
responses byte-for-byte, checks source/output hashes, WGS84 geometry, identities,
study extents and original metadata restrictions. It drives the actual Folium
renderer with an explicitly incomplete legacy fixture and confirms exactly three
loaded layers plus the missing-boundary notice. Complete boundary tests separately
verify all four layers, full extent, attribution and source custody. Negative cases reject duplicate IDs/JSON keys, projected coordinates,
wrong geometry, truncated responses and feature-count overflow. HTML generation
is verified; this is not a claim of a live interactive browser inspection.
