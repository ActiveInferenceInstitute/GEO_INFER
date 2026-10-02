"""
API connectors for GEO-INFER-DATA.

This module provides comprehensive API connectivity for REST APIs,
GraphQL endpoints, and various web services that provide geospatial data.
"""

import logging
import asyncio
import math
import threading
from datetime import datetime
from typing import Any, cast
import time

import requests
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from urllib3.util.retry import Retry

from ..utils.identifiers import validate_sql_identifier


logger = logging.getLogger(__name__)


class APIConnector:
    """
    Universal API connector for geospatial web services.

    This class provides connectivity to REST APIs, GraphQL endpoints,
    and various web services with automatic pagination, rate limiting,
    and error handling.

    Args:
        base_url: Base URL for the API
        authentication: Authentication configuration
        rate_limiting: Rate limiting configuration
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Examples:
        >>> # REST API with API key
        >>> connector = APIConnector(
        ...     base_url='https://api.weather.com',
        ...     authentication={'api_key': 'your_key', 'type': 'header'},
        ...     rate_limiting={'requests_per_minute': 60}
        ... )
        >>>
        >>> # GraphQL API with token
        >>> connector = APIConnector(
        ...     base_url='https://api.graphql.com',
        ...     authentication={'token': 'your_token', 'type': 'bearer'},
        ...     timeout=30
        ... )
        >>>
        >>> # Query geospatial data
        >>> data = await connector.query_geospatial(
        ...     endpoint='/weather/stations',
        ...     params={'bbox': '-122.5,37.7,-122.3,37.9'}
        ... )
    """

    def __init__(
        self,
        base_url: str,
        authentication: dict[str, Any] | None = None,
        rate_limiting: dict[str, Any] | None = None,
        timeout: int = 30,
        retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.authentication = authentication or {}
        self.rate_limiting = rate_limiting or {}
        self.timeout = timeout
        self.retries = retries
        self._rate_interval = 0.0
        if "requests_per_minute" in self.rate_limiting:
            rate = self.rate_limiting["requests_per_minute"]
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(rate)
                or rate <= 0
            ):
                raise ValueError("requests_per_minute must be finite and positive")
            self._rate_interval = 60.0 / rate
        self._next_request_at = 0.0
        self._rate_lock = threading.Lock()
        self._session_lock = threading.Lock()

        self.session: Any = None
        self.request_count = 0
        self.last_request_time: float | None = None

        self._initialize_session()

        logger.info(f"Initialized API connector for {base_url}")

    def _initialize_session(self) -> None:
        """Initialize HTTP session with retry strategy."""
        # Configure retry strategy
        retry_strategy = Retry(  # type: ignore[call-arg]
            total=self.retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1,
        )

        # Create session with retry adapter
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set authentication headers
        if self.authentication:
            auth_type = self.authentication.get("type", "header")
            if auth_type == "header":
                api_key = self.authentication.get("api_key")
                if api_key:
                    self.session.headers.update({"X-API-Key": api_key})
            elif auth_type == "bearer":
                token = self.authentication.get("token")
                if token:
                    self.session.headers.update({"Authorization": f"Bearer {token}"})
            elif auth_type == "basic":
                from requests.auth import HTTPBasicAuth  # type: ignore[import-untyped]

                username = self.authentication.get("username")
                password = self.authentication.get("password")
                if username and password:
                    self.session.auth = HTTPBasicAuth(username, password)

    def _check_rate_limit(self) -> None:
        """Space actual starts under the Session lock, entirely in the worker."""
        with self._rate_lock:
            now = time.monotonic()
            scheduled = max(now, self._next_request_at)
            if scheduled > now:
                time.sleep(scheduled - now)
            self.last_request_time = time.monotonic()
            self._next_request_at = self.last_request_time + self._rate_interval
            self.request_count += 1

    def _request(self, **kwargs: Any) -> requests.Response:
        """Run blocking HTTP work in a worker; serialize shared Session use."""
        with self._session_lock:
            self._check_rate_limit()
            return self.session.request(**kwargs)

    async def query_endpoint(
        self,
        endpoint: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Query API endpoint with automatic retry and rate limiting.

        Args:
            endpoint: API endpoint path
            method: HTTP method
            params: Query parameters
            data: Request body data
            headers: Additional headers

        Returns:
            API response data
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = await asyncio.to_thread(
                self._request,
                method=method,
                url=url,
                params=params,
                json=data if data else None,
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

            # Handle different response formats
            content_type = response.headers.get("content-type", "")

            if "application/json" in content_type:
                return cast(dict[str, Any], response.json())
            elif "text/" in content_type:
                return {"text": response.text}
            else:
                return {"content": response.content}

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    async def query_geospatial(
        self,
        endpoint: str,
        spatial_filter: dict[str, Any] | None = None,
        temporal_filter: dict[str, Any] | None = None,
        pagination: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Query geospatial data from API with spatial and temporal filters.

        Args:
            endpoint: API endpoint
            spatial_filter: Spatial filtering options
            temporal_filter: Temporal filtering options
            pagination: Pagination options
            **kwargs: Additional query parameters

        Returns:
            List of geospatial records
        """
        logger.info(f"Querying geospatial data from {endpoint}")

        # Build query parameters
        params = kwargs.copy()

        if spatial_filter:
            if "bbox" in spatial_filter:
                bbox = spatial_filter["bbox"]
                if len(bbox) >= 4:
                    params["bbox"] = ",".join(map(str, bbox))
            if "geometry" in spatial_filter:
                params["geometry"] = spatial_filter["geometry"]

        if temporal_filter:
            if "start_date" in temporal_filter:
                params["start_date"] = temporal_filter["start_date"].isoformat()
            if "end_date" in temporal_filter:
                params["end_date"] = temporal_filter["end_date"].isoformat()

        if pagination:
            if "page" in pagination:
                params["page"] = pagination["page"]
            if "limit" in pagination:
                params["limit"] = pagination["limit"]

        # Execute query with pagination handling
        all_results = []
        page = 1

        while True:
            try:
                params["page"] = page
                response = await self.query_endpoint(endpoint, params=params)

                # Handle different response formats
                if isinstance(response, dict):
                    if "data" in response:
                        data = response["data"]
                    elif "results" in response:
                        data = response["results"]
                    elif "features" in response:
                        data = response["features"]
                    else:
                        data = [response]

                    if isinstance(data, list):
                        all_results.extend(data)

                        # Check if we have more pages
                        if pagination and "total_pages" in response:
                            if page >= response["total_pages"]:
                                break
                        elif len(data) == 0:
                            break
                        else:
                            page += 1
                            if (
                                pagination
                                and "max_pages" in pagination
                                and page > pagination["max_pages"]
                            ):
                                break
                    else:
                        all_results.append(data)
                        break
                else:
                    all_results.append(response)  # type: ignore[unreachable]
                    break

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to query page {page}: {e}")
                raise

        logger.info(f"Retrieved {len(all_results)} geospatial records")
        return all_results

    async def download_file(
        self, endpoint: str, local_path: str, params: dict[str, Any] | None = None
    ) -> str:
        """
        Download file from API endpoint.

        Args:
            endpoint: API endpoint for file download
            local_path: Local path to save file
            params: Query parameters

        Returns:
            Local file path
        """
        url = f"{self.base_url}{endpoint}"

        try:

            def download() -> None:
                with self._session_lock:
                    self._check_rate_limit()
                    with self.session.get(
                        url, params=params, timeout=self.timeout, stream=True
                    ) as response:
                        response.raise_for_status()
                        with open(local_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

            await asyncio.to_thread(download)

            logger.info(f"Downloaded file to {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"File download failed: {e}")
            raise

    async def close(self) -> None:
        """Close API connection."""
        if self.session:

            def close_session() -> None:
                with self._session_lock:
                    self.session.close()

            await asyncio.to_thread(close_session)
        logger.info("API connection closed")


class GraphQLConnector:
    """
    GraphQL API connector for complex geospatial queries.

    This class provides GraphQL connectivity with support for complex
    queries, mutations, and subscriptions for geospatial data.
    """

    def __init__(
        self,
        endpoint: str,
        authentication: dict[str, Any] | None = None,
        timeout: int = 30,
    ):
        self.endpoint = endpoint
        self.authentication = authentication
        self.timeout = timeout
        self.session: Any = None
        self._session_lock = threading.Lock()

        self._initialize_session()

        logger.info(f"Initialized GraphQL connector for {endpoint}")

    def _initialize_session(self) -> None:
        """Initialize GraphQL session."""
        self.session = requests.Session()

        if self.authentication:
            auth_type = self.authentication.get("type", "bearer")
            if auth_type == "bearer":
                token = self.authentication.get("token")
                if token:
                    self.session.headers.update({"Authorization": f"Bearer {token}"})
            elif auth_type == "header":
                api_key = self.authentication.get("api_key")
                if api_key:
                    self.session.headers.update({"X-API-Key": api_key})

    async def execute_query(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query response
        """
        payload = {"query": query, "variables": variables or {}}

        try:

            def post() -> requests.Response:
                with self._session_lock:
                    return self.session.post(
                        self.endpoint, json=payload, timeout=self.timeout
                    )

            response = await asyncio.to_thread(post)
            response.raise_for_status()

            result = response.json()

            if "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")
                raise ValueError(f"GraphQL query failed: {result['errors']}")

            return cast(dict[str, Any], result.get("data", {}))

        except Exception as e:
            logger.error(f"GraphQL query failed: {e}")
            raise

    async def query_geospatial_features(
        self,
        feature_type: str,
        spatial_filter: dict[str, Any] | None = None,
        temporal_filter: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query geospatial features using GraphQL.

        Args:
            feature_type: Type of feature to query
            spatial_filter: Spatial filtering criteria
            temporal_filter: Temporal filtering criteria
            fields: Fields to select
            limit: Maximum number of features

        Returns:
            List of geospatial features
        """
        query = self._build_features_query(
            feature_type=feature_type,
            spatial_filter=spatial_filter,
            temporal_filter=temporal_filter,
            fields=fields,
            limit=limit,
        )

        # Execute query
        result = await self.execute_query(query)

        features = result.get(feature_type, [])
        logger.info(f"Retrieved {len(features)} {feature_type} features")

        return cast(list[dict[str, Any]], features)

    @staticmethod
    def _coerce_iso_date(value: Any, label: str) -> str:
        """Coerce a temporal filter value to an ISO-8601 string.

        Accepts ``datetime``/``date`` objects and ISO-parseable strings;
        everything else is rejected so no raw value is ever interpolated.
        """
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat") and not isinstance(value, str):
            return value.isoformat()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).isoformat()
            except ValueError as exc:
                raise ValueError(
                    f"GraphQL {label} must be an ISO-8601 date/datetime, got {value!r}"
                ) from exc
        raise ValueError(
            f"GraphQL {label} must be a datetime or ISO-8601 string, "
            f"got {type(value).__name__}"
        )

    @staticmethod
    def _build_features_query(
        feature_type: str,
        spatial_filter: dict[str, Any] | None = None,
        temporal_filter: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        limit: int | None = None,
    ) -> str:
        """Build a GraphQL features query with validated names and coerced values.

        ``feature_type`` and every field name pass through
        :func:`validate_sql_identifier`; bbox coordinates are coerced with
        ``float()``, temporal values with ISO-date parsing, and the limit
        with ``int()``. Values that fail coercion raise ``ValueError``
        rather than being interpolated.
        """
        safe_feature_type = validate_sql_identifier(feature_type)
        selected_fields = fields or ["id", "geometry", "properties"]
        fields_str = ", ".join(
            validate_sql_identifier(field) for field in selected_fields
        )

        where_parts: list[str] = []
        if spatial_filter and "bbox" in spatial_filter:
            bbox = spatial_filter["bbox"]
            min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox[:4])
            where_parts.append(
                f"bbox: {{minLon: {min_lon}, minLat: {min_lat}, "
                f"maxLon: {max_lon}, maxLat: {max_lat}}}"
            )

        if temporal_filter:
            if "start_date" in temporal_filter:
                start = GraphQLConnector._coerce_iso_date(
                    temporal_filter["start_date"], "start_date"
                )
                where_parts.append(f'createdAfter: "{start}"')
            if "end_date" in temporal_filter:
                end = GraphQLConnector._coerce_iso_date(
                    temporal_filter["end_date"], "end_date"
                )
                where_parts.append(f'createdBefore: "{end}"')

        args: list[str] = []
        if where_parts:
            args.append(f"where: {{ {', '.join(where_parts)} }}")
        if limit:
            args.append(f"first: {int(limit)}")
        args_str = f"({', '.join(args)})" if args else ""

        return f"""
        query {{
            {safe_feature_type}{args_str} {{
                {fields_str}
            }}
        }}
        """

    async def close(self) -> None:
        """Close GraphQL connection."""
        if self.session:

            def close_session() -> None:
                with self._session_lock:
                    self.session.close()

            await asyncio.to_thread(close_session)
        logger.info("GraphQL connection closed")


class STACConnector:
    """
    SpatioTemporal Asset Catalog (STAC) connector.

    This class provides connectivity to STAC APIs for satellite imagery
    and other spatiotemporal data with advanced search capabilities.
    """

    def __init__(
        self,
        stac_url: str,
        authentication: dict[str, Any] | None = None,
        timeout: int = 60,
    ):
        self.stac_url = stac_url.rstrip("/")
        self.authentication = authentication
        self.timeout = timeout

        self.connector = APIConnector(
            base_url=stac_url, authentication=authentication, timeout=timeout
        )

        logger.info(f"Initialized STAC connector for {stac_url}")

    async def search_items(
        self,
        collections: list[str] | None = None,
        bbox: list[float] | None = None,
        datetime_range: str | None = None,
        properties: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Search STAC items with spatial and temporal filters.

        Args:
            collections: Collection IDs to search
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            datetime_range: DateTime range (ISO 8601)
            properties: Additional property filters
            limit: Maximum number of items

        Returns:
            STAC search results
        """
        endpoint = "/search"

        params: dict[str, Any] = {"limit": limit}

        if collections:
            params["collections"] = collections

        if bbox:
            params["bbox"] = bbox

        if datetime_range:
            params["datetime"] = datetime_range

        if properties:
            params.update(properties)

        try:
            result = await self.connector.query_endpoint(endpoint, params=params)

            # Handle STAC search response
            if "features" in result:
                items = result["features"]
            else:
                items = result.get("items", [])

            logger.info(f"STAC search returned {len(items)} items")
            return result

        except Exception as e:
            logger.error(f"STAC search failed: {e}")
            raise

    async def get_collection(self, collection_id: str) -> dict[str, Any]:
        """
        Get STAC collection metadata.

        Args:
            collection_id: Collection identifier

        Returns:
            Collection metadata
        """
        endpoint = f"/collections/{collection_id}"

        try:
            collection = await self.connector.query_endpoint(endpoint)
            return collection
        except Exception as e:
            logger.error(f"Failed to get collection {collection_id}: {e}")
            raise

    async def list_collections(self) -> list[dict[str, Any]]:
        """
        List available STAC collections.

        Returns:
            List of collection metadata
        """
        endpoint = "/collections"

        try:
            response = await self.connector.query_endpoint(endpoint)

            if "collections" in response:
                collections = response["collections"]
            else:
                collections = response.get("data", [])

            logger.info(f"Found {len(collections)} STAC collections")
            return cast(list[dict[str, Any]], collections)

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    async def download_item_assets(
        self,
        item: dict[str, Any],
        asset_keys: list[str] | None = None,
        download_dir: str = "./downloads",
    ) -> list[str]:
        """
        Download assets from STAC item.

        Args:
            item: STAC item metadata
            asset_keys: Specific assets to download
            download_dir: Download directory

        Returns:
            List of downloaded file paths
        """
        import os
        from pathlib import Path

        download_paths = []
        os.makedirs(download_dir, exist_ok=True)

        assets = item.get("assets", {})
        if asset_keys:
            assets = {k: v for k, v in assets.items() if k in asset_keys}

        for asset_key, asset_info in assets.items():
            asset_url = asset_info.get("href")
            if not asset_url:
                continue

            # Generate filename
            item_id = item.get("id", "unknown")
            filename = f"{item_id}_{asset_key}.{asset_info.get('type', 'dat')}"
            file_path = Path(download_dir) / filename

            try:
                await self.connector.download_file(asset_url, str(file_path))
                download_paths.append(str(file_path))
                logger.info(f"Downloaded asset: {asset_key} -> {file_path}")

            except Exception as e:
                logger.error(f"Failed to download asset {asset_key}: {e}")

        return download_paths

    async def close(self) -> None:
        """Close STAC connection."""
        await self.connector.close()
        logger.info("STAC connection closed")
