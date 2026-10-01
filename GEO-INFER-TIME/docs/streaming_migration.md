# Streaming API migration

The TIME adapters connect to actual WebSocket and Kafka services. Their optional
runtime dependencies are declared by `geo-infer-time[streaming]`. In the workspace,
include TIME's `streaming` extra when syncing the shared environment.

## Offline replay

`simulated_records`, `allow_simulated`, and the default synthetic network generators are removed.
Use the explicit replay source for fixtures, recorded events, and offline demos:

```python
import asyncio
from datetime import timedelta
from geo_infer_time import ReplayIngestAdapter, StreamProcessor

processor = StreamProcessor(timedelta(minutes=1))
records = [{"timestamp": "2024-01-01T00:00:00Z", "value": 21.5}]
count = asyncio.run(processor.ingest_adapter_stream(ReplayIngestAdapter(records)))
assert count == 1
```

Imports through `geo_infer_time`, `geo_infer_time.core`, and the previous
`geo_infer_time.core.stream_processing` path remain available. Transport
implementation now lives in `geo_infer_time.core.stream_ingest`.
`StreamIngestAdapter` is an abstract base: custom sources must implement both
`connect()` and `stream_data()`. Use an empty replay adapter when only record
parsing is needed.

## Event time and window capacity

Every record requires `timestamp`, `time`, or `datetime`. ISO timestamps and
numeric Unix seconds are accepted; magnitudes above `1e11` are milliseconds.
Kafka can use its broker timestamp when all payload time aliases are absent; broker
timestamps are explicitly milliseconds, including epoch zero. An unavailable
broker timestamp still fails validation. Payload event time always takes precedence.
`normalize_record()` returns a validated dictionary with canonical `timestamp`
(UTC ISO string), numeric `value`, and preserved metadata. Network adapters apply
this validation before yielding a record.

Epoch zero is preserved. Datetime and ISO-string timestamps must carry an
explicit UTC offset (`Z` or `+HH:MM`); naive values raise `ValueError` because
they are ambiguous between local wall-clock time and UTC. All output timestamps
are UTC-aware. Missing timestamps, booleans, nonfinite measurements and `NaT`
fail validation.

Window input stays in event-time order, with stable order for equal timestamps.
Watermarks never move backward. Records strictly before the watermark enter the
late-data buffer; equality is on time. Sessions split only for gaps strictly
larger than `session_gap`. Current retention remains one `window_size` behind the
largest event time. A lateness allowance larger than retention does not extend
retention; records outside retention are discarded from active windows.

`max_buffer_points` defaults to 10,000 for each active and late-data buffer.
Capacity exhaustion raises `BufferError` before accepting the point. Drain late
data with `flush_late_data()`, advance event time, or configure capacity for the
workload. `max_history_windows` defaults to 1,000; the oldest processed summaries
are evicted while cumulative statistics remain intact.

## WebSocket transport

`WebSocketIngestAdapter({"url": "ws://127.0.0.1:8765"})` opens a real socket.
`StreamProcessor.ingest_websocket_stream(url=..., max_messages=...)` owns cleanup.
Both convenience methods accept a prebuilt `adapter=`; its connection settings
replace the URL/topic/server arguments and its iterator is closed on exit.
WebSocket messages must contain one JSON object each. Application-specific
subscription handshakes are not inferred from topic names.

The adapter defaults to a 16-frame receive queue and 1 MiB maximum message size.
`max_queue` and `max_message_size` configure those bounds. The receive loop pulls
one record at a time, applying transport backpressure while processing it.

## Kafka delivery

`KafkaIngestAdapter` accepts `bootstrap_servers`, `topic`, `group_id`, and
`consumer_options` for options such as `auto_offset_reset`, `security_protocol`,
`ssl_context`, or SASL configuration. Automatic commits, deserializers, and buffer
controls cannot override the adapter's delivery contract. Default offset reset
follows aiokafka (`latest`); use `consumer_options={"auto_offset_reset":"earliest"}`
when a new consumer group should read retained events.

`ingest_adapter_stream()` calls `acknowledge(record)` only after parsing, insertion,
and optional `auto_process_windows` aggregation succeed. Each commit advances
only that record's partition to `offset + 1`. A direct `stream_data()` caller
must acknowledge the exact yielded record before requesting another. Source
coordinates appear in reserved `_kafka` metadata; payload-supplied `_kafka` is
replaced with broker coordinates.

Commit failure, rebalance, cancellation, or a crash can redeliver processed
records. Use idempotent durable downstream effects. The processor's buffers and
window summaries are in memory: acknowledgement is not a durable output guarantee.
A failed aggregation can leave its input in memory, so retrying into the same
processor can produce duplicates. No exactly-once guarantee is provided.

Kafka fetch byte limits use `max_message_size`; oversized returned records are
rejected without acknowledgement. Kafka can return an oversized first batch to
make progress, so broker/client fetch limits are not a hard process-memory cap.

## Lifecycle and verification

Network defaults are `connect_timeout=10`, `receive_timeout=30`,
`close_timeout=5`, `max_retries=3`, and `reconnect_interval=1` (seconds).
The retry budget applies to the entire streaming invocation. Network failures and
idle timeouts close the current connection before retrying; exhausted retries
raise `ConnectionError`. Normal WebSocket closure ends the stream. Invalid JSON
and record validation failures propagate immediately. Cancellation and processor
failure close the generator and connection. `max_messages=0` consumes nothing.

For direct iteration with an early break, use `contextlib.aclosing(adapter.stream_data())`
or explicitly disconnect in `finally`; Python does not immediately close a retained
async generator on `break`.

Run the TIME pytest suite for deterministic replay, actual local WebSocket tests,
and Kafka lifecycle/commit fault injection. A separate real-broker probe creates
and deletes one unique topic and checks unacknowledged redelivery and committed
resumption:

```bash
uv run python GEO-INFER-TIME/tests/integration/kafka_service_check.py --bootstrap-servers 127.0.0.1:9092
```

This probe requires a disposable Kafka broker; it is intentionally outside
hermetic test collection. A passing unit suite does not establish live Kafka
verification. Client contracts: [aiokafka consumer documentation](https://aiokafka.readthedocs.io/en/stable/consumer.html)
and [websockets asyncio client documentation](https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html).

For a local broker, the [Apache Kafka Docker quickstart](https://kafka.apache.org/quickstart/)
can be restricted to loopback. Start the container below, wait until its logs
report `Kafka Server started`, run the probe above, then stop this container:

```bash
docker run --rm -d --name geo-infer-time-kafka -p 127.0.0.1:9092:9092 apache/kafka:4.3.1
docker logs geo-infer-time-kafka
```

After the startup message appears:

```bash
uv run python GEO-INFER-TIME/tests/integration/kafka_service_check.py --bootstrap-servers 127.0.0.1:9092
docker stop geo-infer-time-kafka
```

## Reconciliation with the upstream streaming implementation

The upstream real transports, record normalization, configured adapter injection,
Kafka topic metadata and broker timestamp fallback are retained in the extracted
transport module. Lazy imports now raise actionable dependency errors directly;
the private `_import_websockets`/`_import_aiokafka` helpers and mutable availability
flags are superseded by those connection checks.

The upstream `allow_simulated` branch and synthetic default records are superseded
by `ReplayIngestAdapter`. The upstream consumer's default automatic commit is
replaced by explicit acknowledgement after successful processing. Existing replay,
local WebSocket and Kafka fault-injection tests cover those stronger contracts;
merge regressions also verify canonical records, dependency absence, configured
adapter reuse, and broker timestamp precedence and units.
