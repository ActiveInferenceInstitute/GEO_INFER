"""
Data Integration: scaffold for external data-source integration.

The built-in default sources (credit_bureau, property_database, weather_data,
claims_history) point at illustrative endpoints that are not reachable; there
is no live upstream bundled with this module. ``get_data`` refuses
unconfigured sources loudly: a source whose authentication is entirely unset
raises ``ValueError`` naming it (no network attempt is made), so callers
cannot mistake "integration never worked" for "no record". For configured
sources, expected transport failures (missing ``requests`` library,
unreachable endpoint) are logged and return ``None``. Configure credentials
(e.g. ``CREDIT_BUREAU_API_KEY``) or register explicit ``ExternalDataSource``
entries with real endpoints via ``add_data_source`` for any production use.
"""

import logging
import time
import os
from typing import Any, cast
from datetime import datetime
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class ExternalDataSource:
    """External data source configuration."""

    name: str
    source_type: str  # api, database, file, stream
    endpoint: str
    authentication: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    update_frequency: str = "daily"  # real_time, hourly, daily, weekly
    cache_duration: int = 3600  # seconds
    retry_attempts: int = 3
    timeout: int = 30  # seconds

    def get_cache_key(self) -> str:
        """Generate cache key for this data source."""
        return f"{self.name}_{self.source_type}_{self.endpoint}"


class DataIntegrationManager:
    """Manager for external data-source integration.

    Ships only illustrative default endpoints with unset credentials;
    ``get_data`` raises ``ValueError`` for them instead of attempting a
    network fetch. See the module docstring for the full failure semantics.
    """

    def __init__(self, data_sources: list[str] | None = None):
        """
        Initialize the data integration manager.

        Args:
            data_sources: List of data source names to initialize
        """
        self.logger = logging.getLogger(
            "geo_infer_insurance.underwriting.data_integration"
        )

        # Data source configurations
        self.data_sources: dict[str, ExternalDataSource] = {}
        self.data_cache: dict[str, dict[str, Any]] = {}
        self.cache_timestamps: dict[str, datetime] = {}

        # Initialize default data sources
        self._initialize_default_sources()

        # Initialize requested sources
        if data_sources:
            for source_name in data_sources:
                if source_name in self.data_sources:
                    self.data_sources[source_name] = self._configure_data_source(
                        source_name
                    )

        self.logger.info(
            f"Data integration manager initialized with {len(self.data_sources)} sources"
        )

    def _initialize_default_sources(self) -> None:
        """Initialize default data sources."""
        default_sources = {
            "credit_bureau": ExternalDataSource(
                name="credit_bureau",
                source_type="api",
                endpoint="https://api.creditbureau.com/v1/credit-score",
                authentication={"api_key": os.environ.get("CREDIT_BUREAU_API_KEY", "")},
                update_frequency="real_time",
                cache_duration=1800,  # 30 minutes
            ),
            "property_database": ExternalDataSource(
                name="property_database",
                source_type="api",
                endpoint="https://api.propertydb.com/v1/property-info",
                authentication={"api_key": os.environ.get("PROPERTY_DB_API_KEY", "")},
                update_frequency="hourly",
                cache_duration=3600,  # 1 hour
            ),
            "weather_data": ExternalDataSource(
                name="weather_data",
                source_type="api",
                endpoint="https://api.weather.com/v1/forecast",
                authentication={"api_key": os.environ.get("WEATHER_API_KEY", "")},
                update_frequency="real_time",
                cache_duration=900,  # 15 minutes
            ),
            "claims_history": ExternalDataSource(
                name="claims_history",
                source_type="database",
                endpoint="postgresql://claims_db:5432/claims",
                authentication={
                    "username": os.environ.get("CLAIMS_DB_USER", ""),
                    "password": os.environ.get("CLAIMS_DB_PASSWORD", ""),
                },
                update_frequency="daily",
                cache_duration=86400,  # 24 hours
            ),
        }

        self.data_sources.update(default_sources)

    def _configure_data_source(self, source_name: str) -> ExternalDataSource:
        """Configure specific data source."""
        base_config = self.data_sources[source_name]

        # Apply any source-specific configuration
        if source_name == "credit_bureau":
            base_config.parameters = {"include_history": True, "score_type": "fico"}
        elif source_name == "property_database":
            base_config.parameters = {"include_photos": False, "include_history": True}

        return base_config

    def get_data(
        self, source_name: str, query_parameters: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """
        Get data from external source.

        Args:
            source_name: Name of data source
            query_parameters: Query parameters for the request

        Returns:
            Retrieved data, or None if a configured source's fetch failed

        Raises:
            ValueError: If the source's credentials are entirely unset (the
                built-in illustrative defaults); no network attempt is made.
        """
        if source_name not in self.data_sources:
            self.logger.error(f"Unknown data source: {source_name}")
            return None

        source = self.data_sources[source_name]
        self._require_configured_source(source)
        cache_key = self._generate_cache_key(source, query_parameters)

        # Check cache
        if self._is_cache_valid(source, cache_key):
            self.logger.info(f"Using cached data for {source_name}")
            return self.data_cache[cache_key]

        try:
            # Fetch fresh data
            data = self._fetch_data_from_source(source, query_parameters)

            if data:
                # Cache the data
                self.data_cache[cache_key] = data
                self.cache_timestamps[cache_key] = datetime.now()

                self.logger.info(f"Data fetched from {source_name}")
                return data
            else:
                self.logger.warning(f"No data retrieved from {source_name}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to fetch data from {source_name}: {e}")
            return None

    def _require_configured_source(self, source: ExternalDataSource) -> None:
        """Raise a loud configuration error for sources with unset credentials.

        The built-in defaults ship illustrative endpoints with empty env-key
        authentication. Attempting them would either fail with DNS errors or,
        worse, silently return ``None``, so underwriting callers could not
        distinguish "no record" from "integration never worked". Unconfigured
        sources are refused before any network attempt instead.
        """
        configured = any(str(value).strip() for value in source.authentication.values())
        if not configured:
            raise ValueError(
                f"Data source {source.name!r} has no configured credentials; "
                "its built-in default is an illustrative endpoint with no live "
                "upstream. Set the source's credentials (e.g. "
                "CREDIT_BUREAU_API_KEY) or register an explicit "
                "ExternalDataSource via add_data_source() before fetching."
            )

    def _generate_cache_key(
        self,
        source: ExternalDataSource,
        query_parameters: dict[str, Any] | None = None,
    ) -> str:
        """Generate cache key for data request."""
        base_key = source.get_cache_key()

        if query_parameters:
            # Include query parameters in cache key
            sorted_params = sorted(query_parameters.items())
            params_str = json.dumps(sorted_params, sort_keys=True)
            return f"{base_key}_{hash(params_str)}"
        else:
            return base_key

    def _is_cache_valid(self, source: ExternalDataSource, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_timestamps:
            return False

        cache_time = self.cache_timestamps[cache_key]
        now = datetime.now()

        # Check if cache has expired
        age_seconds = (now - cache_time).total_seconds()

        return age_seconds < source.cache_duration

    def _fetch_data_from_source(
        self,
        source: ExternalDataSource,
        query_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Fetch data from specific source."""
        try:
            if source.source_type == "api":
                return self._fetch_from_api(source, query_parameters)
            elif source.source_type == "database":
                return self._fetch_from_database(source, query_parameters)
            elif source.source_type == "file":
                return self._fetch_from_file(source, query_parameters)
            elif source.source_type == "stream":
                return self._fetch_from_stream(source, query_parameters)
            else:
                self.logger.error(f"Unknown source type: {source.source_type}")
                return None

        except Exception as e:
            self.logger.error(f"Data fetch failed for {source.name}: {e}")
            return None

    def _fetch_from_api(
        self,
        source: ExternalDataSource,
        query_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Fetch data from API endpoint."""
        try:
            import requests
        except ImportError:
            self.logger.warning(
                f"requests not installed; cannot fetch API {source.name}"
            )
            return None
        try:
            # Prepare request
            url = source.endpoint
            headers = self._prepare_auth_headers(source)

            # Merge query parameters
            params = {**source.parameters}
            if query_parameters:
                params.update(query_parameters)

            # Make request with retry logic
            for attempt in range(source.retry_attempts):
                try:
                    response = requests.get(
                        url, headers=headers, params=params, timeout=source.timeout
                    )

                    if response.status_code == 200:
                        return cast(dict[str, Any], response.json())
                    else:
                        self.logger.warning(
                            f"API request failed with status {response.status_code}"
                        )

                except requests.RequestException as e:
                    self.logger.warning(
                        f"API request attempt {attempt + 1} failed: {e}"
                    )

                # Wait before retry
                if attempt < source.retry_attempts - 1:
                    time.sleep(2**attempt)  # Exponential backoff

            return None

        except Exception as e:
            self.logger.error(f"API fetch failed: {e}")
            return None

    def _fetch_from_database(
        self,
        source: ExternalDataSource,
        query_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Fetch data from a database using SQLAlchemy or psycopg2."""
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(
                source.endpoint,
                connect_args={
                    "user": source.authentication.get("username", ""),
                    "password": source.authentication.get("password", ""),
                },
            )
            query = (query_parameters or {}).get(
                "query", f"SELECT * FROM {source.name} LIMIT 100"
            )
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(query))
                rows = [dict(row._mapping) for row in result]
            self.logger.info(f"Database fetch for {source.name}: {len(rows)} rows")
            return {"source": source.name, "rows": rows, "row_count": len(rows)}
        except ImportError:
            self.logger.warning(
                f"sqlalchemy not installed; cannot fetch from database {source.name}"
            )
            return None
        except Exception as e:
            self.logger.error(f"Database fetch failed for {source.name}: {e}")
            return None

    def _fetch_from_file(
        self,
        source: ExternalDataSource,
        query_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Fetch data from a local file (JSON, CSV, or GeoJSON)."""
        import pathlib

        file_path = pathlib.Path(source.endpoint)
        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return None
        try:
            suffix = file_path.suffix.lower()
            if suffix == ".json":
                import json

                with open(file_path) as f:
                    data = json.load(f)
            elif suffix == ".csv":
                import csv

                with open(file_path) as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
            elif suffix in (".geojson",):
                import json

                with open(file_path) as f:
                    data = json.load(f)
            else:
                self.logger.warning(f"Unsupported file format: {suffix}")
                return None
            self.logger.info(f"File fetch for {source.name}: loaded {file_path.name}")
            return {"source": source.name, "file": str(file_path), "data": data}
        except Exception as e:
            self.logger.error(f"File fetch failed for {source.name}: {e}")
            return None

    def _fetch_from_stream(
        self,
        source: ExternalDataSource,
        query_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Fetch data from a streaming source (HTTP SSE or WebSocket snapshot)."""
        try:
            import requests

            timeout = (query_parameters or {}).get("timeout", 10)
            headers = {"Accept": "text/event-stream"}
            auth = source.authentication
            if auth.get("api_key"):
                headers["Authorization"] = f"Bearer {auth['api_key']}"

            resp = requests.get(
                source.endpoint, headers=headers, stream=True, timeout=timeout
            )
            resp.raise_for_status()

            # Consume up to N events
            max_events = (query_parameters or {}).get("max_events", 50)
            events = []
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    events.append(line[5:].strip())
                    if len(events) >= max_events:
                        break

            self.logger.info(f"Stream fetch for {source.name}: {len(events)} events")
            return {"source": source.name, "events": events, "event_count": len(events)}
        except ImportError:
            self.logger.warning(
                f"requests not installed; cannot fetch stream {source.name}"
            )
            return None
        except Exception as e:
            self.logger.error(f"Stream fetch failed for {source.name}: {e}")
            return None

    def _prepare_auth_headers(self, source: ExternalDataSource) -> dict[str, str]:
        """Prepare authentication headers for API requests."""
        headers = {"Content-Type": "application/json"}

        auth = source.authentication
        if "api_key" in auth:
            headers["Authorization"] = f"Bearer {auth['api_key']}"
        elif "username" in auth and "password" in auth:
            import base64

            credentials = f"{auth['username']}:{auth['password']}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        return headers

    def add_data_source(self, source: ExternalDataSource) -> bool:
        """Add new data source."""
        try:
            self.data_sources[source.name] = source
            self.logger.info(f"Data source added: {source.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add data source: {e}")
            return False

    def remove_data_source(self, source_name: str) -> bool:
        """Remove data source."""
        if source_name in self.data_sources:
            del self.data_sources[source_name]
            self.logger.info(f"Data source removed: {source_name}")
            return True
        return False

    def get_data_source_info(self, source_name: str) -> dict[str, Any] | None:
        """Get information about data source."""
        if source_name not in self.data_sources:
            return None

        source = self.data_sources[source_name]
        return {
            "name": source.name,
            "source_type": source.source_type,
            "endpoint": source.endpoint,
            "update_frequency": source.update_frequency,
            "cache_duration": source.cache_duration,
            "last_updated": self.cache_timestamps.get(source.get_cache_key()),
        }

    def clear_cache(self, source_name: str | None = None) -> None:
        """Clear data cache."""
        if source_name:
            # Clear cache for specific source
            keys_to_remove = [
                key
                for key in self.data_cache.keys()
                if key.startswith(f"{source_name}_")
            ]
            for key in keys_to_remove:
                del self.data_cache[key]
                del self.cache_timestamps[key]
        else:
            # Clear all cache
            self.data_cache.clear()
            self.cache_timestamps.clear()

        self.logger.info("Data cache cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache information."""
        return {
            "total_entries": len(self.data_cache),
            "oldest_entry": min(self.cache_timestamps.values()).isoformat()
            if self.cache_timestamps
            else None,
            "newest_entry": max(self.cache_timestamps.values()).isoformat()
            if self.cache_timestamps
            else None,
            "cache_size_mb": len(json.dumps(self.data_cache).encode("utf-8"))
            / (1024 * 1024),
        }

    def health_check(self) -> dict[str, Any]:
        """Perform health check on data integration."""
        health_status = {
            "status": "operational",
            "total_sources": len(self.data_sources),
            "cache_entries": len(self.data_cache),
            "timestamp": datetime.now().isoformat(),
        }

        # Check each data source
        source_status = {}
        for name, source in self.data_sources.items():
            try:
                # Test connectivity
                test_data = self.get_data(name, {"test": True})
                source_status[name] = "operational" if test_data else "error"
            except (KeyError, ValueError, OSError) as e:
                logger.warning("Health check failed for data source %s: %s", name, e)
                source_status[name] = "error"

        health_status["source_status"] = source_status

        # Determine overall status
        if any(status == "error" for status in source_status.values()):
            health_status["status"] = "degraded"

        return health_status

    def close_connections(self) -> None:
        """Close any open connections."""
        # Baseline for connection cleanup
        self.logger.info("Data integration connections closed")


# Convenience functions
def create_data_integration_manager(
    data_sources: list[str] | None = None,
) -> DataIntegrationManager:
    """Create a new data integration manager."""
    return DataIntegrationManager(data_sources)


def get_credit_score(
    ssn: str, data_manager: DataIntegrationManager | None = None
) -> int | None:
    """
    Get credit score from external credit bureau.

    Args:
        ssn: Social Security Number
        data_manager: Data integration manager instance

    Returns:
        Credit score or None if failed
    """
    if data_manager is None:
        data_manager = DataIntegrationManager()

    try:
        credit_data = data_manager.get_data("credit_bureau", {"ssn": ssn})
    except (KeyError, ValueError, OSError) as e:
        logger.warning("Credit bureau fetch failed: %s", e)
        return None
    return credit_data.get("credit_score") if credit_data else None


def get_property_history(
    property_id: str, data_manager: DataIntegrationManager | None = None
) -> dict[str, Any] | None:
    """
    Get property history from external database.

    Args:
        property_id: Property identifier
        data_manager: Data integration manager instance

    Returns:
        Property history data or None if failed
    """
    if data_manager is None:
        data_manager = DataIntegrationManager()

    try:
        property_data = data_manager.get_data(
            "property_database", {"property_id": property_id}
        )
    except (KeyError, ValueError, OSError) as e:
        logger.warning("Property database fetch failed: %s", e)
        return None
    return property_data if property_data else None
