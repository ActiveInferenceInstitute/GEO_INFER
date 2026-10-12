#!/usr/bin/env python3

"""
Data Collector Agent for GEO-INFER-AGENT.

This module implements an agent specialized in collecting geospatial data
from various sources, including remote APIs, sensors, and other data feeds.
"""

import os
import json
import logging
import asyncio
from typing import Any, cast
from datetime import datetime
import requests
import pandas as pd

# BDIState and BDIAgent are exported by the models package.
from geo_infer_agent.models import BDIAgent

logger = logging.getLogger("geo_infer_agent.agents.data_collector")


class DataCollectorAgent(BDIAgent):
    """
    Data Collector Agent implementation.

    This agent specializes in:
    - Gathering data from configured data sources
    - Processing and validating the collected data
    - Storing data for later analysis
    - Monitoring data sources for updates
    """

    def __init__(self, agent_id: str | None = None, config: dict | None = None):
        """Initialize the data collector agent."""
        # Ensure config has default values
        config = config or {}
        default_config = {
            "collection_interval": 300,  # 5 minutes
            "max_retries": 3,
            "timeout": 30,
            "data_sources": [],
            "storage_path": "data",
            "initial_desires": [
                {
                    "name": "collect_data",
                    "description": "Collect data from configured sources",
                    "priority": 0.8,
                },
                {
                    "name": "monitor_sources",
                    "description": "Monitor data sources for availability",
                    "priority": 0.6,
                },
                {
                    "name": "process_data",
                    "description": "Process and validate collected data",
                    "priority": 0.7,
                },
            ],
            "plans": [
                {
                    "name": "data_collection_plan",
                    "desire_name": "collect_data",
                    "actions": [
                        {
                            "type": "log",
                            "message": "Starting data collection",
                            "level": "info",
                        },
                        {"type": "collect_data_from_sources"},
                        {
                            "type": "log",
                            "message": "Data collection complete",
                            "level": "info",
                        },
                        {"type": "wait", "duration": "$CONFIG:collection_interval"},
                    ],
                },
                {
                    "name": "monitor_sources_plan",
                    "desire_name": "monitor_sources",
                    "actions": [
                        {
                            "type": "log",
                            "message": "Starting source monitoring",
                            "level": "info",
                        },
                        {"type": "check_sources_availability"},
                        {
                            "type": "log",
                            "message": "Source monitoring complete",
                            "level": "info",
                        },
                        {"type": "wait", "duration": 60},
                    ],
                },
                {
                    "name": "process_data_plan",
                    "desire_name": "process_data",
                    "context_conditions": {"has_unprocessed_data": True},
                    "actions": [
                        {
                            "type": "log",
                            "message": "Starting data processing",
                            "level": "info",
                        },
                        {"type": "process_collected_data"},
                        {
                            "type": "update_belief",
                            "belief_name": "has_unprocessed_data",
                            "belief_value": False,
                        },
                        {
                            "type": "log",
                            "message": "Data processing complete",
                            "level": "info",
                        },
                    ],
                },
            ],
        }

        # Merge default config with provided config
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        super().__init__(agent_id, config)

        # Initialize data storage
        self.storage_path = self.config["storage_path"]
        os.makedirs(self.storage_path, exist_ok=True)

        # Track collected datasets
        self.datasets: list[dict[str, Any]] = []
        self.unprocessed_data: list[dict[str, Any]] = []

        logger.info(f"Data collector agent {self.agent_id} initialized")

    async def initialize(self) -> None:
        """Initialize the agent."""
        # Register data collection action handlers
        self._register_data_action_handlers()

        # Initialize data source beliefs
        self._initialize_data_source_beliefs()

        logger.info(f"Data collector agent {self.agent_id} initialization complete")

    def _register_data_action_handlers(self) -> None:
        """Register data collection specific action handlers."""
        self.action_handlers["collect_data_from_sources"] = (
            self._handle_collect_data_action
        )
        self.action_handlers["check_sources_availability"] = (
            self._handle_check_sources_action
        )
        self.action_handlers["process_collected_data"] = (
            self._handle_process_data_action
        )

    def _initialize_data_source_beliefs(self) -> None:
        """Initialize beliefs about data sources."""
        data_sources = self.config.get("data_sources", [])

        # Create beliefs about each data source
        for i, source in enumerate(data_sources):
            source_id = source.get("id", f"source_{i}")
            source_name = source.get("name", f"Data Source {i}")
            source_type = source.get("type", "unknown")
            source_url = source.get("url", "")

            # Add beliefs about this source
            self.state.update_belief(f"data_source.{source_id}.name", source_name)
            self.state.update_belief(f"data_source.{source_id}.type", source_type)
            self.state.update_belief(f"data_source.{source_id}.url", source_url)
            self.state.update_belief(f"data_source.{source_id}.available", False)
            self.state.update_belief(f"data_source.{source_id}.last_check", None)
            self.state.update_belief(f"data_source.{source_id}.last_collection", None)

        # Initialize other data-related beliefs
        self.state.update_belief("last_collection_time", None)
        self.state.update_belief("last_monitoring_time", None)
        self.state.update_belief("last_processing_time", None)
        self.state.update_belief("has_unprocessed_data", False)
        self.state.update_belief("total_collected_datasets", 0)

        logger.info(
            f"Data collector agent initialized beliefs for {len(data_sources)} data sources"
        )

    async def _handle_collect_data_action(
        self, agent: "DataCollectorAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle data collection action.

        Args:
            agent: The agent instance
            action: Action parameters

        Returns:
            Action result
        """
        logger.info(
            f"Data collector agent {agent.agent_id} collecting data from sources"
        )

        data_sources = agent.config.get("data_sources", [])
        results = []
        collected_count = 0
        error_count = 0

        for source in data_sources:
            source_id = source.get("id", f"source_{data_sources.index(source)}")

            try:
                # Check if source is available
                is_available = agent.state.get_belief(
                    f"data_source.{source_id}.available"
                )
                if is_available and is_available.value is False:
                    logger.warning(f"Skipping unavailable data source {source_id}")
                    continue

                # Collect data from this source
                data = await self._collect_from_source(source)

                if data:
                    # Store the collected data
                    timestamp = datetime.now().isoformat()
                    filename = f"{agent.storage_path}/{source_id}_{timestamp}.json"

                    with open(filename, "w") as f:
                        json.dump(data, f, indent=2)

                    # Record the dataset
                    dataset_info = {
                        "source_id": source_id,
                        "timestamp": timestamp,
                        "filename": filename,
                        "processed": False,
                        "record_count": len(cast(Any, data)),
                    }
                    agent.datasets.append(dataset_info)
                    agent.unprocessed_data.append(dataset_info)

                    # Update beliefs
                    agent.state.update_belief(
                        f"data_source.{source_id}.last_collection", timestamp
                    )
                    collected_count += 1

                    # Set flag for unprocessed data
                    agent.state.update_belief("has_unprocessed_data", True)

                    results.append(
                        {
                            "source_id": source_id,
                            "success": True,
                            "record_count": dataset_info["record_count"],
                            "filename": filename,
                        }
                    )

                    logger.info(
                        f"Collected {dataset_info['record_count']} records from source {source_id}"
                    )
                else:
                    error_count += 1
                    results.append(
                        {
                            "source_id": source_id,
                            "success": False,
                            "error": "No data returned",
                        }
                    )
                    logger.warning(f"No data collected from source {source_id}")

            except Exception as e:
                error_count += 1
                results.append(
                    {"source_id": source_id, "success": False, "error": str(e)}
                )
                logger.error(f"Error collecting data from source {source_id}: {str(e)}")

        # Update total datasets count
        total_datasets = agent.state.get_belief("total_collected_datasets")
        if total_datasets:
            agent.state.update_belief(
                "total_collected_datasets", total_datasets.value + collected_count
            )
        else:
            agent.state.update_belief("total_collected_datasets", collected_count)

        agent.state.update_belief("last_collection_time", datetime.now().isoformat())

        return {
            "success": error_count == 0,
            "collected_count": collected_count,
            "error_count": error_count,
            "results": results,
        }

    async def _handle_check_sources_action(
        self, agent: "DataCollectorAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle checking data sources action.

        Args:
            agent: The agent instance
            action: Action parameters

        Returns:
            Action result
        """
        logger.info(f"Data collector agent {agent.agent_id} checking data sources")

        data_sources = agent.config.get("data_sources", [])
        results = []
        available_count = 0
        unavailable_count = 0

        for source in data_sources:
            source_id = source.get("id", f"source_{data_sources.index(source)}")
            source_url = source.get("url", "")

            if not source_url:
                continue

            try:
                # Check if the source is available
                is_available = await self._check_source_availability(source)

                # Update belief about source availability
                agent.state.update_belief(
                    f"data_source.{source_id}.available", is_available
                )
                agent.state.update_belief(
                    f"data_source.{source_id}.last_check", datetime.now().isoformat()
                )

                if is_available:
                    available_count += 1
                else:
                    unavailable_count += 1

                results.append({"source_id": source_id, "available": is_available})

                logger.debug(f"Data source {source_id} availability: {is_available}")

            except Exception as e:
                unavailable_count += 1
                agent.state.update_belief(f"data_source.{source_id}.available", False)
                agent.state.update_belief(
                    f"data_source.{source_id}.last_check", datetime.now().isoformat()
                )

                results.append(
                    {"source_id": source_id, "available": False, "error": str(e)}
                )

                logger.error(f"Error checking data source {source_id}: {str(e)}")

        agent.state.update_belief("last_monitoring_time", datetime.now().isoformat())

        return {
            "success": True,
            "available_count": available_count,
            "unavailable_count": unavailable_count,
            "results": results,
        }

    async def _handle_process_data_action(
        self, agent: "DataCollectorAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle processing collected data action.

        Args:
            agent: The agent instance
            action: Action parameters

        Returns:
            Action result
        """
        logger.info(f"Data collector agent {agent.agent_id} processing collected data")

        if not agent.unprocessed_data:
            return {
                "success": True,
                "processed_count": 0,
                "message": "No unprocessed data to process",
            }

        processed_count = 0
        error_count = 0
        results = []

        for dataset_info in list(agent.unprocessed_data):
            try:
                # Load the data
                with open(dataset_info["filename"]) as f:
                    data = json.load(f)

                # Process the data
                processed_data = await self._process_dataset(data, dataset_info)

                if processed_data:
                    # Save processed data
                    processed_filename = dataset_info["filename"].replace(
                        ".json", "_processed.json"
                    )
                    with open(processed_filename, "w") as f:
                        json.dump(processed_data, f, indent=2)

                    # Update dataset info
                    dataset_info["processed"] = True
                    dataset_info["processed_timestamp"] = datetime.now().isoformat()
                    dataset_info["processed_filename"] = processed_filename

                    # Remove from unprocessed list
                    agent.unprocessed_data.remove(dataset_info)

                    processed_count += 1
                    results.append(
                        {
                            "source_id": dataset_info["source_id"],
                            "success": True,
                            "original_file": dataset_info["filename"],
                            "processed_file": processed_filename,
                        }
                    )

                    logger.info(
                        f"Processed data from source {dataset_info['source_id']}"
                    )
                else:
                    error_count += 1
                    results.append(
                        {
                            "source_id": dataset_info["source_id"],
                            "success": False,
                            "error": "Processing returned no data",
                        }
                    )

                    logger.warning(
                        f"Processing returned no data for source {dataset_info['source_id']}"
                    )

            except Exception as e:
                error_count += 1
                results.append(
                    {
                        "source_id": dataset_info["source_id"],
                        "success": False,
                        "error": str(e),
                    }
                )

                logger.error(
                    f"Error processing data from source {dataset_info['source_id']}: {str(e)}"
                )

        # Record the processing pass on the agent-level belief
        agent.state.update_belief("last_processing_time", datetime.now().isoformat())

        agent.state.update_belief(
            "has_unprocessed_data", len(agent.unprocessed_data) > 0
        )

        return {
            "success": error_count == 0,
            "processed_count": processed_count,
            "error_count": error_count,
            "results": results,
        }

    async def _collect_from_source(
        self, source: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Collect data from a specific source.

        Args:
            source: Source configuration

        Returns:
            Collected data or None if collection failed
        """
        source_type = source.get("type", "")

        if source_type == "api":
            return await self._collect_from_api(source)
        elif source_type == "file":
            return await self._collect_from_file(source)
        elif source_type == "sensor":
            return await self._collect_from_sensor(source)
        else:
            raise ValueError(f"Unsupported data source type: {source_type!r}")

    async def _collect_from_api(self, source: dict[str, Any]) -> dict[str, Any] | None:
        """
        Collect data from an API source.

        Args:
            source: Source configuration

        Returns:
            Collected data or None if collection failed
        """
        url = source.get("url", "")
        if not url:
            return None

        response = await asyncio.to_thread(
            requests.get,
            url,
            params=source.get("params"),
            timeout=self.config["timeout"],
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return {"source_id": source.get("id", ""), "records": payload}
        if not isinstance(payload, dict):
            raise ValueError("API source must return a JSON object or array")
        return payload

    async def _collect_from_file(self, source: dict[str, Any]) -> dict[str, Any] | None:
        """
        Collect data from a file source.

        Args:
            source: Source configuration

        Returns:
            Collected data or None if collection failed
        """
        file_path = source.get("path", "")
        if not file_path:
            return None

        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix in {".json", ".geojson"}:
            with open(file_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("JSON source must contain an object")
            return payload
        if suffix == ".csv":
            frame = pd.read_csv(file_path)
            return {
                "source_id": source.get("id", ""),
                "records": frame.to_dict(orient="records"),
            }
        raise ValueError(f"Unsupported file source format: {suffix or '<none>'}")

    async def _collect_from_sensor(
        self, source: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Collect data from a sensor source.

        Args:
            source: Source configuration

        Returns:
            Collected data or None if collection failed
        """
        sensor_id = source.get("sensor_id", "")
        if not sensor_id:
            return None

        endpoint = source.get("url") or source.get("endpoint")
        if not sensor_id or not endpoint:
            raise ValueError("sensor sources require sensor_id and url")
        response = await asyncio.to_thread(
            requests.get,
            endpoint,
            params={"sensor_id": sensor_id, **source.get("params", {})},
            timeout=self.config["timeout"],
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("sensor source must return a JSON object")
        return payload

    async def _check_source_availability(self, source: dict[str, Any]) -> bool:
        """
        Check if a data source is available.

        Args:
            source: Source configuration

        Returns:
            True if the source is available
        """
        source_type = source.get("type", "")

        if source_type == "api":
            url = source.get("url", "")
            if not url:
                return False
            try:
                response = await asyncio.to_thread(
                    requests.head, url, timeout=self.config["timeout"]
                )
                return bool(response.ok)
            except requests.RequestException:
                return False

        elif source_type == "file":
            file_path = source.get("path", "")
            if not file_path:
                return False

            return os.path.isfile(file_path)

        elif source_type == "sensor":
            sensor_id = source.get("sensor_id", "")
            if not sensor_id:
                return False

            endpoint = source.get("url") or source.get("endpoint")
            if not sensor_id or not endpoint:
                return False
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    endpoint,
                    params={"sensor_id": sensor_id},
                    timeout=self.config["timeout"],
                )
                return bool(response.ok)
            except requests.RequestException:
                return False

        else:
            return False

    async def _process_dataset(
        self, data: dict[str, Any], dataset_info: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Process a collected dataset.

        Args:
            data: The collected data
            dataset_info: Information about the dataset

        Returns:
            Processed data or None if processing failed
        """
        try:
            # Process features if present
            if "features" in data:
                features = data["features"]

                # Simple processing - calculate statistics
                values = [
                    f["properties"]["value"]
                    for f in features
                    if "value" in f.get("properties", {})
                ]

                if values:
                    stats = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "stddev": (
                            sum((x - (sum(values) / len(values))) ** 2 for x in values)
                            / len(values)
                        )
                        ** 0.5,
                    }

                    # Add stats to processed data
                    processed_data = data.copy()
                    processed_data["stats"] = stats
                    processed_data["processed_timestamp"] = datetime.now().isoformat()

                    return processed_data

            # If no features or processing failed, return data with processing timestamp
            processed_data = data.copy()
            processed_data["processed_timestamp"] = datetime.now().isoformat()

            return processed_data

        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return None

    async def action_configure_source(
        self, source_id: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Action to configure a data source.

        Args:
            source_id: ID of the source to configure
            config: New configuration

        Returns:
            Result of the action
        """
        try:
            # Find the source in the config
            data_sources = self.config.get("data_sources", [])
            source_index = -1

            for i, source in enumerate(data_sources):
                if source.get("id") == source_id:
                    source_index = i
                    break

            if source_index == -1:
                # Source not found, add new source
                source = {"id": source_id}
                source.update(config)
                data_sources.append(source)
                source_index = len(data_sources) - 1
            else:
                # Update existing source
                data_sources[source_index].update(config)

            # Update agent config
            self.config["data_sources"] = data_sources

            # Update beliefs
            for key, value in config.items():
                self.state.update_belief(f"data_source.{source_id}.{key}", value)

            logger.info(f"Configured data source {source_id}")

            return {
                "success": True,
                "source_id": source_id,
                "message": f"Successfully configured source {source_id}",
            }

        except Exception as e:
            logger.error(f"Error configuring data source {source_id}: {str(e)}")

            return {"success": False, "source_id": source_id, "error": str(e)}

    async def action_get_collected_data(
        self, filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Action to get information about collected datasets.

        Args:
            filters: Optional filters to apply

        Returns:
            Result of the action with dataset information
        """
        filters = filters or {}

        # Filter datasets
        filtered_datasets = self.datasets

        if "source_id" in filters:
            filtered_datasets = [
                d for d in filtered_datasets if d["source_id"] == filters["source_id"]
            ]

        if "processed" in filters:
            filtered_datasets = [
                d for d in filtered_datasets if d["processed"] == filters["processed"]
            ]

        if "after" in filters:
            after_timestamp = filters["after"]
            filtered_datasets = [
                d for d in filtered_datasets if d["timestamp"] > after_timestamp
            ]

        if "before" in filters:
            before_timestamp = filters["before"]
            filtered_datasets = [
                d for d in filtered_datasets if d["timestamp"] < before_timestamp
            ]

        return {
            "success": True,
            "count": len(filtered_datasets),
            "datasets": filtered_datasets,
        }
