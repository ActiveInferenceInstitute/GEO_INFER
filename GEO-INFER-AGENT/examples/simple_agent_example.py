#!/usr/bin/env python3

"""
Simple agent example for GEO-INFER-AGENT.

This script demonstrates the basic usage of the GEO-INFER-AGENT module
by creating and running a data collector agent.
"""

import json
import sys
import asyncio
import logging
import argparse
from datetime import datetime

from geo_infer_agent.agents.data_collector import DataCollectorAgent
from geo_infer_agent.api.interface import agent_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


async def run_simple_agent(args):
    """
    Run a simple agent example.

    Args:
        args: Command-line arguments
    """
    logging.info("Starting simple agent example")

    # Load agent configuration from file if provided
    config = {}
    if args.config:
        try:
            with open(args.config) as f:
                config = json.load(f)
            logging.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            sys.exit(1)

    # Use agent type from command line
    agent_type = args.agent_type

    # Create a region if not in config
    if "region" not in config and args.region:
        config["region"] = args.region

    # Create data sources if not in config
    if "data_sources" not in config and args.add_sample_sources:
        config["data_sources"] = [
            {
                "id": "weather_api",
                "name": "Weather API",
                "type": "api",
                "url": "https://example.com/weather/api",
            },
            {
                "id": "traffic_sensors",
                "name": "Traffic Sensors",
                "type": "sensor",
                "sensor_id": "traffic_network_1",
            },
        ]

    # Initialize agent interface
    await agent_interface.initialize_services()

    try:
        if agent_type == "data_collector":
            # Create and run a data collector agent directly
            agent = DataCollectorAgent(config=config)
            await agent.initialize()

            # Run the agent task
            agent_task = asyncio.create_task(agent.run())

            logging.info(f"Data collector agent {agent.agent_id} is running...")
            logging.info("Press Ctrl+C to stop")

            # Run until interrupted
            try:
                while True:
                    await asyncio.sleep(1)

                    # Print agent state every 10 seconds
                    current_time = datetime.now().timestamp()
                    if int(current_time) % 10 == 0:
                        logging.info(
                            f"Agent state: {len(agent.datasets)} datasets collected"
                        )
                        await asyncio.sleep(
                            1
                        )  # Wait to avoid multiple prints in the same second

            except KeyboardInterrupt:
                logging.info("Stopping agent...")
                agent.stop()
                await agent_task

        else:
            # Create an agent through the agent interface
            logging.info(f"Creating agent of type {agent_type}...")

            agent_id = await agent_interface.create_agent(
                agent_type=agent_type, config=config
            )

            logging.info(f"Agent created with ID: {agent_id}")

            # Start the agent
            await agent_interface.start_agent(agent_id)
            logging.info(f"Agent {agent_id} started")

            # Monitor agent state
            try:
                while True:
                    # Get agent state
                    agent_state = await agent_interface.get_agent_state(agent_id)

                    logging.info(
                        f"Agent {agent_id} state: {agent_state.get('state', {}).get('beliefs', {})}"
                    )

                    await asyncio.sleep(5)

            except KeyboardInterrupt:
                logging.info("Stopping agent...")
                await agent_interface.stop_agent(agent_id)
                logging.info(f"Agent {agent_id} stopped")

    finally:
        # Shutdown services
        await agent_interface.shutdown_services()
        logging.info("Agent interface services shutdown")


def main():
    """Parse command-line arguments and run the example."""
    parser = argparse.ArgumentParser(
        description="Simple agent example for GEO-INFER-AGENT"
    )

    parser.add_argument(
        "--agent-type",
        "-t",
        default="data_collector",
        choices=[
            "data_collector",
            "bdi",
            "active_inference",
            "reinforcement_learning",
            "rule_based",
        ],
        help="Type of agent to create",
    )

    parser.add_argument(
        "--config", "-c", help="Path to agent configuration file (JSON)"
    )

    parser.add_argument(
        "--region", "-r", help="Geospatial region for agent operation (GeoJSON)"
    )

    parser.add_argument(
        "--add-sample-sources",
        "-s",
        action="store_true",
        help="Add sample data sources if none in config",
    )

    parser.add_argument(
        "--runtime",
        "-d",
        type=int,
        default=0,
        help="Runtime in seconds (0 = unlimited)",
    )

    args = parser.parse_args()

    try:
        if args.runtime > 0:
            # Run for a limited time
            asyncio.run(asyncio.wait_for(run_simple_agent(args), args.runtime))
        else:
            # Run until interrupted
            asyncio.run(run_simple_agent(args))
    except TimeoutError:
        logging.info(f"Runtime of {args.runtime} seconds reached, stopping")
    except KeyboardInterrupt:
        logging.info("Example interrupted")


if __name__ == "__main__":
    main()
