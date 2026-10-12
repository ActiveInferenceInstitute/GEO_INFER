#!/usr/bin/env python3
"""
Basic usage example for GEO-INFER-COMMS.

This example demonstrates the core functionality of the geospatial
communication system including messaging, notifications, channels,
and event handling.
"""

import asyncio
import logging
from datetime import datetime, UTC

# Import the main communication system
from geo_infer_comms import (
    GeospatialCommunicationSystem,
    get_communication_system,
    GeospatialPoint,
    GeospatialBounds,
    GeospatialMetadata,
    MessageType,
    MessagePriority,
    NotificationType,
    ChannelType,
)
from geo_infer_comms.core.notifications import AlertRule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def message_callback(message):
    """Callback function for received messages."""
    logger.info(f"Received message: {message.content}")
    logger.info(f"From: {message.sender_id}")
    if message.geospatial_data:
        location = message.geospatial_data.location
        logger.info(f"Location: {location.latitude}, {location.longitude}")


def event_callback(event):
    """Callback function for received events."""
    logger.info(f"Received event: {event.event_type}")
    logger.info(f"Payload: {event.payload}")
    if event.geospatial_context:
        logger.info(f"Geospatial context: {event.geospatial_context}")


def notification_callback(notification):
    """Callback function for received notifications."""
    logger.info(f"Received notification: {notification.title}")
    logger.info(f"Content: {notification.content}")
    logger.info(f"Priority: {notification.priority.value}")


async def main():
    """Main example function."""
    logger.info("Starting GEO-INFER-COMMS basic usage example")

    # Initialize the communication system
    config = {
        "enable_persistence": False,  # Disable for demo
        "message_queue_size": 1000,
        "max_notifications": 1000,
        "max_channels": 100,
        "max_events": 1000,
    }

    # Method 1: Using the main system class
    system = GeospatialCommunicationSystem(config)
    system.start()

    try:
        # Example 1: Send a simple message
        logger.info("\n=== Example 1: Sending Messages ===")
        message_response = system.send_message(
            content="Hello from GEO-INFER-COMMS!",
            recipients=["user_1", "user_2"],
            message_type=MessageType.TEXT,
            priority=MessagePriority.NORMAL,
        )
        logger.info(f"Message sent: {message_response.message_id}")

        # Example 2: Send a location-based message
        logger.info("\n=== Example 2: Location-Based Messaging ===")
        location = GeospatialPoint(
            longitude=-122.4194, latitude=37.7749
        )  # San Francisco
        geospatial_data = GeospatialMetadata(
            location=location, accuracy=10.0, source="GPS"
        )

        location_message = system.send_message(
            content="Location update from San Francisco",
            recipients=["field_team"],
            message_type=MessageType.LOCATION,
            geospatial_data=geospatial_data,
        )
        logger.info(f"Location message sent: {location_message.message_id}")

        # Example 3: Create a notification
        logger.info("\n=== Example 3: Creating Notifications ===")
        notification = system.create_notification(
            title="System Maintenance Notice",
            content="Scheduled maintenance will occur tonight from 2-4 AM UTC",
            recipients=["admin@geo-infer.org", "ops@geo-infer.org"],
            notification_type=NotificationType.INFO,
            priority=MessagePriority.NORMAL,
            delivery_method=["email", "in_app"],
        )
        logger.info(f"Notification created: {notification.notification_id}")

        # Example 4: Create a communication channel
        logger.info("\n=== Example 4: Creating Channels ===")
        channel = system.create_channel(
            name="Emergency Response Team",
            channel_type=ChannelType.PRIVATE,
            description="Channel for emergency coordination",
        )
        logger.info(f"Channel created: {channel.channel_id}")
        logger.info(f"Channel name: {channel.name}")

        # Example 5: Publish an event
        logger.info("\n=== Example 5: Publishing Events ===")
        event = system.publish_event(
            event_type="system_alert",
            payload={
                "alert_level": "warning",
                "message": "High CPU usage detected",
                "affected_systems": ["server_01", "server_02"],
            },
            source="monitoring_system",
            priority=MessagePriority.HIGH,
        )
        logger.info(f"Event published: {event.event_id}")

        # Example 6: Subscribe to events
        logger.info("\n=== Example 6: Event Subscription ===")
        subscription_id = system.subscribe_to_events(
            subscriber_id="demo_subscriber",
            event_types=["system_alert", "data_update"],
            callback=event_callback,
        )
        logger.info(f"Event subscription created: {subscription_id}")

        # Example 7: Subscribe to messages
        logger.info("\n=== Example 7: Message Subscription ===")
        msg_subscription_id = system.message_broker.subscribe(
            subscriber_id="demo_subscriber", callback=message_callback
        )
        logger.info(f"Message subscription created: {msg_subscription_id}")

        # Example 8: Create an alert rule
        logger.info("\n=== Example 8: Creating Alert Rules ===")
        alert_rule = system.alert_system.create_alert_rule(
            AlertRule(
                name="High Temperature Alert",
                description="Alert when temperature exceeds threshold",
                conditions={"temperature": {"min": 35.0}},
                alert_title="High Temperature Warning",
                alert_content="Temperature has exceeded safe threshold",
                recipients=["admin@geo-infer.org"],
                priority=MessagePriority.HIGH,
                delivery_methods=["email", "sms"],
            )
        )
        logger.info(f"Alert rule created: {alert_rule}")

        # Example 9: Trigger an alert
        logger.info("\n=== Example 9: Triggering Alerts ===")
        alert_response = system.alert_system.trigger_alert(
            rule_id=alert_rule,
            trigger_data={"temperature": 38.5, "sensor_id": "temp_001"},
            geospatial_context=GeospatialMetadata(
                location=GeospatialPoint(longitude=-122.4194, latitude=37.7749),
                accuracy=10.0,
                source="GPS",
            ),
        )
        if alert_response:
            logger.info(f"Alert triggered: {alert_response.alert_id}")

        # Example 10: Get system health
        logger.info("\n=== Example 10: System Health Check ===")
        health = system.get_system_health()
        logger.info(f"System status: {health['status']}")
        logger.info(f"Uptime: {health['uptime_seconds']:.1f} seconds")

        # Example 11: Get comprehensive metrics
        metrics = system.get_comprehensive_metrics()
        logger.info(
            f"Messages sent: {metrics['message_metrics']['metrics']['messages_sent']}"
        )
        logger.info(
            f"Notifications created: "
            f"{metrics['notification_metrics']['metrics']['notifications_created']}"
        )
        logger.info(
            f"Channels created: "
            f"{metrics['channel_metrics']['metrics']['channels_created']}"
        )
        logger.info(
            f"Events published: {metrics['event_metrics']['metrics']['events_published']}"
        )

        # Brief pause to allow processing
        await asyncio.sleep(2.0)

    finally:
        # Clean up
        system.stop()
        logger.info("Example completed and system shut down")


def alternative_usage_example():
    """Alternative usage example using the global system."""
    logger.info("\n=== Alternative Usage Example ===")

    # Method 2: Using the global system function
    system = get_communication_system()

    # Send a quick message
    try:
        message = system.send_message(
            content="Quick test message", recipients=["test_user"]
        )
        logger.info(f"Quick message sent: {message.message_id}")
    except Exception as e:
        logger.error(f"Error sending quick message: {e}")


def geospatial_operations_example():
    """Example of geospatial operations."""
    logger.info("\n=== Geospatial Operations Example ===")

    # Create geospatial points
    sf_point = GeospatialPoint(longitude=-122.4194, latitude=37.7749)
    ny_point = GeospatialPoint(longitude=-74.0060, latitude=40.7128)

    # Calculate distance
    distance = sf_point.distance_to(ny_point)
    logger.info(f"Distance between SF and NY: {distance:.2f} meters")

    # Create geospatial bounds
    bounds = GeospatialBounds(
        min_longitude=-122.5, min_latitude=37.7, max_longitude=-122.3, max_latitude=37.8
    )

    # Check if point is within bounds
    is_within = sf_point.is_within_bounds(bounds)
    logger.info(f"SF point within bounds: {is_within}")

    # Create geospatial metadata
    metadata = GeospatialMetadata(
        location=sf_point,
        accuracy=5.0,
        source="GPS",
        timestamp=datetime.now(UTC),
    )

    logger.info(
        f"Geospatial metadata created: {metadata.location.latitude}, {metadata.location.longitude}"
    )


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())

    # Run additional examples
    alternative_usage_example()
    geospatial_operations_example()

    logger.info("All examples completed successfully!")
