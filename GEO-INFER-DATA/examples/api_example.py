#!/usr/bin/env python3
"""
API example for GEO-INFER-DATA.

This example demonstrates how to use the REST API for data access
and management operations.

Usage:
    python api_example.py

Requirements:
    - GEO-INFER-DATA package installed
    - FastAPI and uvicorn for API server
"""

import asyncio
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Any

from geo_infer_data.api.rest_api import DataAPI


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataAPIClient:
    """Client for interacting with GEO-INFER-DATA API."""

    def __init__(self, base_url: str = "http://localhost:8001/v1"):
        self.base_url = base_url

    def health_check(self) -> dict[str, Any]:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_datasets(self, **filters) -> dict[str, Any]:
        """List available datasets."""
        response = requests.get(f"{self.base_url}/datasets", params=filters)
        response.raise_for_status()
        return response.json()

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Get dataset information."""
        response = requests.get(f"{self.base_url}/datasets/{dataset_id}")
        response.raise_for_status()
        return response.json()

    def create_dataset(self, dataset_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new dataset."""
        response = requests.post(f"{self.base_url}/datasets", json=dataset_data)
        response.raise_for_status()
        return response.json()

    def get_dataset_data(self, dataset_id: str, **params) -> dict[str, Any]:
        """Get dataset data."""
        response = requests.get(
            f"{self.base_url}/datasets/{dataset_id}/data", params=params
        )
        response.raise_for_status()
        return response.json()

    def search_datasets(self, **search_params) -> dict[str, Any]:
        """Search datasets."""
        response = requests.get(f"{self.base_url}/search", params=search_params)
        response.raise_for_status()
        return response.json()

    def ingest_multi_source(self, data: dict[str, Any]) -> dict[str, Any]:
        """Ingest data from multiple sources."""
        response = requests.post(f"{self.base_url}/data/ingest/multi-source", json=data)
        response.raise_for_status()
        return response.json()

    def get_storage_backends(self) -> dict[str, Any]:
        """Get storage backend information."""
        response = requests.get(f"{self.base_url}/storage/backends")
        response.raise_for_status()
        return response.json()

    def get_api_metrics(self) -> dict[str, Any]:
        """Get API performance metrics."""
        response = requests.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()


async def start_api_server():
    """Start the API server for testing."""
    logger.info("Starting API server for testing")

    api = DataAPI(config_path=None, host="localhost", port=8001, enable_cors=True)

    # Start server in background task
    import threading

    def run_server():
        api.start(reload=False)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    await asyncio.sleep(2)

    return api


async def main():
    """Main example function."""
    logger.info("Starting API example")

    # Start API server (runs in a daemon thread for the example's lifetime)
    await start_api_server()

    # Create API client
    client = DataAPIClient()

    try:
        # Test health check
        logger.info("Testing API health check")

        health = client.health_check()
        logger.info(f"API Health: {health['status']} - {health['message']}")

        # Test list datasets
        logger.info("Testing list datasets")

        datasets = client.list_datasets(limit=5)
        logger.info(f"Found {len(datasets)} datasets")

        # Test create dataset
        logger.info("Testing create dataset")

        dataset_data = {
            "title": "API Test Dataset",
            "description": "Dataset created via API example",
            "type": "vector",
            "format": "geojson",
            "metadata": {
                "title": "API Test Dataset",
                "description": "Test dataset for API functionality",
                "spatial": {
                    "bbox": [-122.5, 37.7, -122.3, 37.9],
                    "crs": {"epsg_code": "EPSG:4326"},
                },
                "temporal": {
                    "start": "2023-01-01T00:00:00Z",
                    "end": "2023-12-31T23:59:59Z",
                },
                "lineage": {
                    "source": "api_example",
                    "process": "automated_creation",
                    "created_by": "api_example",
                },
                "keywords": ["test", "api", "example"],
                "contact": {"organization": "Example Org", "email": "test@example.com"},
            },
        }

        created_dataset = client.create_dataset(dataset_data)
        logger.info(
            f"Created dataset: {created_dataset['title']} (ID: {created_dataset['id']})"
        )

        # Test get dataset
        logger.info("Testing get dataset")

        dataset_id = created_dataset["id"]
        retrieved_dataset = client.get_dataset(dataset_id)
        logger.info(f"Retrieved dataset: {retrieved_dataset['title']}")

        # Test search
        logger.info("Testing search functionality")

        search_results = client.search_datasets(q="test", type="vector")
        logger.info(f"Search found {search_results['total']} results")

        # Test multi-source ingestion
        logger.info("Testing multi-source data ingestion")

        ingestion_data = {
            "satellite": {
                "bbox": [-122.5, 37.7, -122.3, 37.9],
                "date_range": "2023-01-01/2023-01-31",
            },
            "sensors": {
                "time_range": "2023-01-01/2023-01-31",
                "sensor_types": ["temperature", "humidity"],
            },
        }

        ingestion_result = client.ingest_multi_source(ingestion_data)
        logger.info(
            f"Ingestion completed for {ingestion_result['ingestion_metadata']['sources_processed']} sources"
        )

        # Test storage backends
        logger.info("Testing storage backends")

        backends = client.get_storage_backends()
        logger.info(f"Available storage backends: {backends}")

        # Test API metrics
        logger.info("Testing API metrics")

        metrics = client.get_api_metrics()
        logger.info("API Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        # Save API interaction results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        api_results = {
            "health_check": health,
            "datasets_listed": len(datasets),
            "created_dataset": created_dataset,
            "search_results": search_results["total"],
            "ingestion_sources": ingestion_result["ingestion_metadata"][
                "sources_processed"
            ],
            "storage_backends": backends,
            "api_metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(output_dir / "api_results.json", "w") as f:
            import json

            json.dump(api_results, f, indent=2, default=str)

        logger.info(f"API results saved to {output_dir / 'api_results.json'}")

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        logger.info(
            "Note: API server may not be running. Start with: python -m geo_infer_data.api"
        )
    except Exception as e:
        logger.error(f"API example failed: {e}")
        raise

    logger.info("API example completed")


def demonstrate_api_usage():
    """Demonstrate API usage without starting server."""
    logger.info("Demonstrating API usage patterns")

    # Show example API calls
    example_calls = [
        {
            "method": "GET",
            "endpoint": "/health",
            "description": "Check API health status",
        },
        {
            "method": "GET",
            "endpoint": "/datasets?page=1&limit=10&type=vector",
            "description": "List datasets with filtering and pagination",
        },
        {
            "method": "POST",
            "endpoint": "/datasets",
            "description": "Create a new dataset",
            "body": {
                "title": "Environmental Monitoring Data",
                "type": "vector",
                "format": "geojson",
                "metadata": {
                    "spatial": {"bbox": [-122.5, 37.7, -122.3, 37.9]},
                    "temporal": {
                        "start": "2023-01-01T00:00:00Z",
                        "end": "2023-12-31T23:59:59Z",
                    },
                },
            },
        },
        {
            "method": "GET",
            "endpoint": "/datasets/{dataset_id}/data?format=geojson&bbox=-122.5,37.7,-122.3,37.9",
            "description": "Get dataset data with spatial filtering",
        },
        {
            "method": "POST",
            "endpoint": "/data/ingest/multi-source",
            "description": "Ingest data from multiple sources",
            "body": {
                "satellite": {"bbox": [-122.5, 37.7, -122.3, 37.9]},
                "sensors": {"time_range": "2023-01-01/2023-01-31"},
            },
        },
        {
            "method": "GET",
            "endpoint": "/search?q=temperature&bbox=-122.5,37.7,-122.3,37.9",
            "description": "Search datasets with spatial and text filters",
        },
    ]

    logger.info("API Usage Examples:")
    for call in example_calls:
        logger.info(f"  {call['method']} {call['endpoint']}")
        logger.info(f"    {call['description']}")
        if "body" in call:
            logger.info(f"    Body: {call['body']}")
        logger.info("")


if __name__ == "__main__":
    # Demonstrate API usage first
    demonstrate_api_usage()

    # Then run interactive example
    asyncio.run(main())
