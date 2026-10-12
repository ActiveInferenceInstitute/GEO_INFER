# Environmental Monitoring Integration

An environmental pipeline should retain the measurement's sensor identity, units,
location support, event time, and calibration provenance before spatial aggregation.
A numerical alert is a decision rule; its threshold and operational recipient must
be chosen explicitly. The packages do not infer these policies from a measurement.

## Replay a bounded transport input

`ReplayIngestAdapter` implements the real asynchronous source interface. Timestamp
normalization and finite numeric validation happen before the analytical boundary.
The observed zero remains a measurement, and replay disconnects after consumption.

```python
import asyncio
from geo_infer_time import ReplayIngestAdapter

async def collect():
    adapter = ReplayIngestAdapter([
        {"timestamp": "2023-12-31T16:00:00-08:00", "value": 0.0},
        {"timestamp": "2024-01-01T01:00:00Z", "value": 4.0},
    ])
    values = [adapter.normalize_record(record) async for record in adapter.stream_data(max_messages=2)]
    assert not adapter.is_connected
    return values

records = asyncio.run(collect())
assert records[0]["timestamp"] == "2024-01-01T00:00:00+00:00"
assert records[0]["value"] == 0.0
assert len(records) == 2
```

## From records to inference

Map locations with WGS84 H3 indexing, then align cell/timestamp/value records with
`align_h3_observations` on the domain and interval used by the application. Different
sensors inside one cell require an explicit aggregation or additional identity;
duplicate cell-time pairs are rejected. No observation is created for an absent pair.

Store raw measurements and processing lineage before publishing derived summaries.
TIME `StreamProcessor` has explicit window/buffer/history bounds and late-data
handling. A bounded replay test establishes normalization and source cleanup, not
a live MQTT/Kafka connection, durable consumer acknowledgment, or alert delivery.
Those transports have separate acceptance requirements.

Use [SPACE composition](../../../GEO-INFER-SPACE/docs/CROSS_MODULE_COMPOSITION.md)
and [TIME streaming migration](../../../GEO-INFER-TIME/docs/streaming_migration.md).
The [real DATA/SPACE/TIME/ACT contract](../../../GEO-INFER-TEST/tests/integration/test_space_time_composition_contract.py)
verifies a local HTTP source and a storage round trip before numerical inference.
