# Storage and spatial-index contracts

`PostgreSQLBackend.store` returns an opaque `pg_` identifier. Keep the returned
identifier; do not derive it from a timestamp. Tabular writes use a transaction
and reject an existing table rather than replace it. Generic JSON values live
in `generic_data_store` and support the same public `retrieve(id, {})` and
`delete(id)` operations. Generic values reject nonempty table queries.
Authenticated serialized envelopes are verified before unpickling.

Install the owning `postgres` extra for PostgreSQL/PostGIS operations. It includes
GeoAlchemy2, which GeoPandas requires for `GeoDataFrame.to_postgis`. GeoPandas
creates the database spatial index. Retrieval preserves the declared geometry
column and CRS, including custom names and projected coordinates; spatial query
bounds are WGS84 and are transformed to the stored geometry SRID. `SpatialIndexer` builds an in-memory index
from a GeoDataFrame and does not create database indexes.

Spatial indexes use row positions internally, preserving duplicate and string
index labels in returned frames. They retain a frame snapshot at construction;
rebuild the index after changing observations. Empty results preserve columns,
geometry and CRS. R-tree queries select intersecting geometry bounding boxes.
H3 queries select resolution-9 centroid cells covered by the query polygon;
this cell-based selection is not exact geometry intersection.

## Verification

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module DATA
```

`tests/unit/test_postgresql_roundtrip.py` uses real temporary SQLite SQL to
check generic and tabular routing, deletion, corruption rejection and write
collision protection. Its patched geospatial writer checks the transaction
interface. These tests do not establish live PostgreSQL/PostGIS acceptance.
`tests/unit/test_indexing.py` runs native H3 and R-tree queries against duplicate,
string and integer labels, mutation isolation and empty-result schema/CRS.

The CI `optional-data-services` job explicitly runs
`tests/service_contract_probe.py` against disposable loopback PostGIS and Redis
containers and a pinned official MinIO source build. It checks projected/custom
geometry queries, generic values, signed-payload tamper rejection and
S3-compatible byte operations. It uses test credentials only, never AWS account
access, and retains its operation receipt. Readiness polling is bounded; failed
operations fail the job without retries. This is S3 protocol acceptance against
MinIO, not live AWS account acceptance.
