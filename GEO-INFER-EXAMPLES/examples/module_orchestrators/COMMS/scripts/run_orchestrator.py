#!/usr/bin/env python3
"""GEO-INFER-COMMS module orchestrator.

Runs one documented end-to-end COMMS operation on synthetic data: start a
real ``MessageBroker``, subscribe synthetic recipients, send priority- and
location-tagged messages through the broker's spatial index and priority
queue, measure delivery metrics, and format delivered messages for SMS and
email channels. All work goes through the real ``geo_infer_comms`` public
API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_comms import (
        GeospatialMetadata,
        GeospatialPoint,
        MessageBroker,
        MessageFormatter,
        MessagePriority,
        MessageRequest,
        MessageResponse,
        MessageType,
    )

    broker = MessageBroker(max_queue_size=1000, enable_persistence=False)

    # Synthetic subscribers: field crews that count deliveries.
    delivered: dict[str, list[str]] = {"crew-north": [], "crew-south": []}
    broker.subscribe(
        "crew-north", lambda msg: delivered["crew-north"].append(msg.message_id)
    )
    broker.subscribe(
        "crew-south", lambda msg: delivered["crew-south"].append(msg.message_id)
    )

    broker.start()
    try:
        senders = ["dispatch-01", "dispatch-02"]
        recipients = ["crew-north", "crew-south"]
        priorities = [
            MessagePriority.NORMAL,
            MessagePriority.URGENT,
            MessagePriority.LOW,
            MessagePriority.HIGH,
            MessagePriority.NORMAL,
            MessagePriority.HIGH,
        ]
        sent: list[MessageResponse] = []
        for i, priority in enumerate(priorities):
            geo = GeospatialMetadata(
                location=GeospatialPoint(
                    longitude=-124.20 + 0.01 * i,
                    latitude=41.74 + 0.01 * i,
                ),
                accuracy=5.0,
                source="GPS",
            )
            request = MessageRequest(
                content=f"Synthetic field update {i}: reach segment {i} before 15:00.",
                recipients=recipients,
                message_type=MessageType.TEXT,
                priority=priority,
                geospatial_data=geo,
            )
            sent.append(
                broker.send_message(request, sender_id=senders[i % len(senders)])
            )

        # Deterministic drain: wait until the broker thread processes the queue.
        broker.message_queue.join()
        metrics = broker.get_metrics()
    finally:
        broker.stop()

    stored = broker.get_messages()
    urgent = [m for m in sent if m.priority == MessagePriority.URGENT.value]
    sms_view = MessageFormatter.format_for_sms(urgent[0])

    # MessageFormatter.format_for_email reads ``priority.value``, but pydantic
    # use_enum_values stores plain strings on delivered messages; rebuild the
    # real delivered message without value-coercion so the module's own
    # formatter can consume it.
    first = sent[0]
    email_input = MessageResponse.model_construct(
        content=first.content,
        sender_id=first.sender_id,
        recipients=first.recipients,
        message_type=MessageType(first.message_type),
        priority=MessagePriority(first.priority),
        timestamp=first.timestamp,
        geospatial_data=first.geospatial_data,
    )
    email_view = MessageFormatter.format_for_email(email_input)

    return {
        "operation": "message_broker_delivery_and_formatting",
        "messages_sent": len(sent),
        "statuses_after_delivery": sorted({str(m.status) for m in sent}),
        "delivery_metrics": {
            "messages_delivered": broker.metrics.messages_delivered,
            "delivery_failures": broker.metrics.delivery_failures,
            "messages_stored": metrics.get("messages_stored"),
        },
        "subscriber_deliveries": {
            "crew-north": len(delivered["crew-north"]),
            "crew-south": len(delivered["crew-south"]),
        },
        "broker_query": {
            "messages_retrievable": len(stored),
            "spatial_index_entries": sum(
                len(v) for v in broker.spatial_index._index.values()
            ),
        },
        "urgent_sms_preview": sms_view,
        "email_subject_preview": email_view["subject"],
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("COMMS", _operation))
