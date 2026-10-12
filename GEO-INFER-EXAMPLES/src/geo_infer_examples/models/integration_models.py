#!/usr/bin/env python3
"""
GEO-INFER Integration Models

Comprehensive data models for cross-module integrations, workflow definitions,
execution contexts, and standardized result structures.
"""

from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from datetime import datetime
from pathlib import Path


class ModuleType(Enum):
    """Categories of GEO-INFER modules."""

    CORE_INFRASTRUCTURE = "core_infrastructure"
    DATA_PROCESSING = "data_processing"
    SPATIAL_TEMPORAL = "spatial_temporal"
    ANALYTICS_AI = "analytics_ai"
    DOMAIN_SPECIFIC = "domain_specific"
    USER_INTERFACE = "user_interface"
    OPERATIONS = "operations"


class DataFormat(Enum):
    """Supported data formats for inter-module communication."""

    GEOJSON = "geojson"
    SPATIAL_TEMPORAL_JSON = "spatial_temporal_json"
    RASTER_ARRAY = "raster_array"
    TIME_SERIES = "time_series"
    PREDICTION_RESULT = "prediction_result"
    HEALTH_RECORD = "health_record"
    AGRICULTURAL_OBSERVATION = "agricultural_observation"
    URBAN_FEATURE = "urban_feature"
    RISK_ASSESSMENT = "risk_assessment"


class IntegrationPattern(Enum):
    """Integration patterns between modules."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"
    PIPELINE = "pipeline"
    EVENT_DRIVEN = "event_driven"
    FEEDBACK_LOOP = "feedback_loop"
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"


@dataclass
class ModuleSpec:
    """Specification for a GEO-INFER module."""

    name: str
    module_type: ModuleType
    api_base_url: str
    version: str
    capabilities: list[str] = field(default_factory=list)
    supported_formats: list[DataFormat] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    optional_dependencies: list[str] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)
    health_endpoint: str = "/health"
    documentation_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "module_type": self.module_type.value,
            "api_base_url": self.api_base_url,
            "version": self.version,
            "capabilities": self.capabilities,
            "supported_formats": [f.value for f in self.supported_formats],
            "dependencies": self.dependencies,
            "optional_dependencies": self.optional_dependencies,
            "configuration": self.configuration,
            "health_endpoint": self.health_endpoint,
            "documentation_url": self.documentation_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleSpec":
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            module_type=ModuleType(data["module_type"]),
            api_base_url=data["api_base_url"],
            version=data["version"],
            capabilities=data.get("capabilities", []),
            supported_formats=[
                DataFormat(f) for f in data.get("supported_formats", [])
            ],
            dependencies=data.get("dependencies", []),
            optional_dependencies=data.get("optional_dependencies", []),
            configuration=data.get("configuration", {}),
            health_endpoint=data.get("health_endpoint", "/health"),
            documentation_url=data.get("documentation_url"),
        )


@dataclass
class ModuleConnection:
    """Defines connection between two modules."""

    source_module: str
    target_module: str
    pattern: IntegrationPattern
    data_format: DataFormat
    endpoint: str
    transformation: str | None = None  # Data transformation function
    validation: str | None = None  # Validation rules
    retry_policy: dict[str, Any] | None = None
    timeout: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_module": self.source_module,
            "target_module": self.target_module,
            "pattern": self.pattern.value,
            "data_format": self.data_format.value,
            "endpoint": self.endpoint,
            "transformation": self.transformation,
            "validation": self.validation,
            "retry_policy": self.retry_policy,
            "timeout": self.timeout,
        }


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""

    name: str
    module: str
    endpoint: str
    dependencies: list[str] = field(default_factory=list)
    input_mapping: dict[str, str] | None = None
    output_mapping: dict[str, str] | None = None
    condition: str | None = None
    timeout: int | None = None
    retry_count: int = 0
    optional: bool = False

    # Event-driven properties
    trigger_events: list[str] = field(default_factory=list)
    emits_events: list[str] = field(default_factory=list)

    # Feedback loop properties
    feedback_mapping: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "module": self.module,
            "endpoint": self.endpoint,
            "dependencies": self.dependencies,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping,
            "condition": self.condition,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "optional": self.optional,
            "trigger_events": self.trigger_events,
            "emits_events": self.emits_events,
            "feedback_mapping": self.feedback_mapping,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStep":
        """Create from dictionary representation."""
        return cls(**data)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""

    id: str
    name: str
    description: str
    steps: list[WorkflowStep]
    execution_strategy: str = "sequential"
    max_iterations: int | None = None  # For feedback loops
    convergence_threshold: float | None = None  # For feedback loops
    timeout: int | None = None
    retry_policy: dict[str, Any] | None = None
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "execution_strategy": self.execution_strategy,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "timeout": self.timeout,
            "retry_policy": self.retry_policy,
            "version": self.version,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowDefinition":
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=[WorkflowStep.from_dict(step) for step in data["steps"]],
            execution_strategy=data.get("execution_strategy", "sequential"),
            max_iterations=data.get("max_iterations"),
            convergence_threshold=data.get("convergence_threshold"),
            timeout=data.get("timeout"),
            retry_policy=data.get("retry_policy"),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    def copy(self) -> "WorkflowDefinition":
        """Create a deep copy of the workflow definition."""
        return WorkflowDefinition.from_dict(self.to_dict())


@dataclass
class ExecutionContext:
    """Context for workflow execution."""

    user_id: str | None = None
    session_id: str | None = None
    priority: int = 5  # 1-10 scale
    resilience_mode: bool = False
    debug_mode: bool = False
    resource_limits: dict[str, Any] | None = None
    environment: str = "production"
    custom_config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class SpatialTemporalData:
    """Standardized spatial-temporal data structure."""

    features: list[dict[str, Any]]
    temporal_range: tuple[datetime, datetime]
    spatial_bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    coordinate_system: str = "EPSG:4326"
    temporal_resolution: str | None = None
    spatial_resolution: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_geojson(self) -> dict[str, Any]:
        """Convert to GeoJSON format."""
        return {
            "type": "FeatureCollection",
            "features": self.features,
            "metadata": {
                "temporal_range": [
                    self.temporal_range[0].isoformat(),
                    self.temporal_range[1].isoformat(),
                ],
                "spatial_bounds": self.spatial_bounds,
                "coordinate_system": self.coordinate_system,
                "temporal_resolution": self.temporal_resolution,
                "spatial_resolution": self.spatial_resolution,
                **self.metadata,
            },
        }


@dataclass
class AnalysisResult:
    """Standardized analysis result structure."""

    data: dict[str, Any]
    confidence: float | None = None
    uncertainty: dict[str, Any] | None = None
    method: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    validation_results: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data": self.data,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "method": self.method,
            "parameters": self.parameters,
            "performance_metrics": self.performance_metrics,
            "validation_results": self.validation_results,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IntegrationResult:
    """Result of cross-module integration."""

    success: bool
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    execution_time: float | None = None
    module_results: dict[str, AnalysisResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "module_results": {k: v.to_dict() for k, v in self.module_results.items()},
        }

    def add_module_result(self, module_name: str, result: AnalysisResult) -> None:
        """Add result from a specific module."""
        self.module_results[module_name] = result

    def get_module_result(self, module_name: str) -> AnalysisResult | None:
        """Get result from a specific module."""
        return self.module_results.get(module_name)


@dataclass
class HealthSurveillanceData(SpatialTemporalData):
    """Specialized data structure for health surveillance."""

    case_data: list[dict[str, Any]] = field(default_factory=list)
    demographic_data: list[dict[str, Any]] = field(default_factory=list)
    environmental_factors: dict[str, Any] = field(default_factory=dict)
    disease_type: str | None = None
    severity_levels: list[str] | None = None

    def to_health_geojson(self) -> dict[str, Any]:
        """Convert to health-specific GeoJSON format."""
        geojson = self.to_geojson()
        geojson["metadata"].update(
            {
                "domain": "health_surveillance",
                "disease_type": self.disease_type,
                "severity_levels": self.severity_levels,
                "case_count": len(self.case_data),
                "demographic_count": len(self.demographic_data),
            }
        )
        return geojson


@dataclass
class AgriculturalData(SpatialTemporalData):
    """Specialized data structure for agricultural applications."""

    field_boundaries: list[dict[str, Any]] = field(default_factory=list)
    crop_types: list[str] = field(default_factory=list)
    growth_stages: dict[str, Any] = field(default_factory=dict)
    weather_data: list[dict[str, Any]] = field(default_factory=list)
    soil_properties: dict[str, Any] = field(default_factory=dict)
    management_practices: list[dict[str, Any]] = field(default_factory=list)

    def to_agricultural_geojson(self) -> dict[str, Any]:
        """Convert to agriculture-specific GeoJSON format."""
        geojson = self.to_geojson()
        geojson["metadata"].update(
            {
                "domain": "agriculture",
                "crop_types": self.crop_types,
                "field_count": len(self.field_boundaries),
                "weather_records": len(self.weather_data),
                "management_practices": len(self.management_practices),
            }
        )
        return geojson


@dataclass
class UrbanPlanningData(SpatialTemporalData):
    """Specialized data structure for urban planning."""

    zoning_data: list[dict[str, Any]] = field(default_factory=list)
    infrastructure: dict[str, Any] = field(default_factory=dict)
    demographic_data: list[dict[str, Any]] = field(default_factory=list)
    land_use: dict[str, Any] = field(default_factory=dict)
    community_input: list[dict[str, Any]] = field(default_factory=list)
    regulatory_constraints: dict[str, Any] = field(default_factory=dict)

    def to_urban_geojson(self) -> dict[str, Any]:
        """Convert to urban planning-specific GeoJSON format."""
        geojson = self.to_geojson()
        geojson["metadata"].update(
            {
                "domain": "urban_planning",
                "zoning_areas": len(self.zoning_data),
                "community_inputs": len(self.community_input),
                "infrastructure_types": list(self.infrastructure.keys()),
            }
        )
        return geojson


@dataclass
class ClimateData(SpatialTemporalData):
    """Specialized data structure for climate applications."""

    variables: list[str] = field(
        default_factory=list
    )  # temperature, precipitation, etc.
    scenarios: list[str] = field(default_factory=list)  # RCP scenarios
    models: list[str] = field(default_factory=list)  # Climate models used
    ensemble_data: bool = False
    downscaling_method: str | None = None
    bias_correction: str | None = None

    def to_climate_geojson(self) -> dict[str, Any]:
        """Convert to climate-specific GeoJSON format."""
        geojson = self.to_geojson()
        geojson["metadata"].update(
            {
                "domain": "climate",
                "variables": self.variables,
                "scenarios": self.scenarios,
                "models": self.models,
                "ensemble_data": self.ensemble_data,
                "downscaling_method": self.downscaling_method,
                "bias_correction": self.bias_correction,
            }
        )
        return geojson


class IntegrationPatterns:
    """Collection of common integration patterns and templates."""

    @staticmethod
    def create_health_surveillance_workflow() -> WorkflowDefinition:
        """Create a standard health surveillance workflow."""
        return WorkflowDefinition(
            id="health_surveillance_standard",
            name="Standard Health Surveillance",
            description="Complete health surveillance and outbreak detection workflow",
            steps=[
                WorkflowStep(
                    name="data_ingestion",
                    module="DATA",
                    endpoint="/ingest/health-records",
                    dependencies=[],
                ),
                WorkflowStep(
                    name="geocoding",
                    module="SPACE",
                    endpoint="/geocode/addresses",
                    dependencies=["data_ingestion"],
                ),
                WorkflowStep(
                    name="temporal_analysis",
                    module="TIME",
                    endpoint="/analyze/temporal-patterns",
                    dependencies=["geocoding"],
                ),
                WorkflowStep(
                    name="spatial_clustering",
                    module="SPACE",
                    endpoint="/analyze/spatial-clusters",
                    dependencies=["temporal_analysis"],
                ),
                WorkflowStep(
                    name="outbreak_detection",
                    module="HEALTH",
                    endpoint="/detect/outbreaks",
                    dependencies=["spatial_clustering"],
                ),
                WorkflowStep(
                    name="risk_assessment",
                    module="RISK",
                    endpoint="/assess/outbreak-risk",
                    dependencies=["outbreak_detection"],
                    optional=True,
                ),
                WorkflowStep(
                    name="alert_generation",
                    module="API",
                    endpoint="/alerts/generate",
                    dependencies=["outbreak_detection", "risk_assessment"],
                ),
            ],
            execution_strategy="sequential",
            tags=["health", "surveillance", "outbreak", "standard"],
        )

    @staticmethod
    def create_precision_agriculture_workflow() -> WorkflowDefinition:
        """Create a precision agriculture monitoring workflow."""
        return WorkflowDefinition(
            id="precision_agriculture_monitoring",
            name="Precision Agriculture Monitoring",
            description="Comprehensive crop monitoring and management workflow",
            steps=[
                WorkflowStep(
                    name="sensor_data_collection",
                    module="IOT",
                    endpoint="/collect/sensor-data",
                    dependencies=[],
                ),
                WorkflowStep(
                    name="satellite_data_processing",
                    module="SPACE",
                    endpoint="/process/satellite-imagery",
                    dependencies=[],
                ),
                WorkflowStep(
                    name="data_fusion",
                    module="DATA",
                    endpoint="/fuse/multi-source",
                    dependencies=[
                        "sensor_data_collection",
                        "satellite_data_processing",
                    ],
                ),
                WorkflowStep(
                    name="crop_health_analysis",
                    module="AG",
                    endpoint="/analyze/crop-health",
                    dependencies=["data_fusion"],
                ),
                WorkflowStep(
                    name="predictive_modeling",
                    module="AI",
                    endpoint="/predict/crop-yield",
                    dependencies=["crop_health_analysis"],
                ),
                WorkflowStep(
                    name="intervention_simulation",
                    module="SIM",
                    endpoint="/simulate/interventions",
                    dependencies=["predictive_modeling"],
                ),
                WorkflowStep(
                    name="recommendations",
                    module="AG",
                    endpoint="/generate/recommendations",
                    dependencies=["intervention_simulation"],
                ),
            ],
            execution_strategy="parallel",
            tags=["agriculture", "precision", "monitoring", "iot"],
        )

    @staticmethod
    def create_active_inference_workflow() -> WorkflowDefinition:
        """Create an active inference feedback loop workflow."""
        return WorkflowDefinition(
            id="active_inference_adaptive",
            name="Active Inference Adaptive System",
            description="Adaptive system using active inference principles",
            steps=[
                WorkflowStep(
                    name="observation_processing",
                    module="SPACE",
                    endpoint="/process/observations",
                    dependencies=[],
                    emits_events=["observations_ready"],
                ),
                WorkflowStep(
                    name="belief_update",
                    module="ACT",
                    endpoint="/update/beliefs",
                    dependencies=["observation_processing"],
                    feedback_mapping={"prior_beliefs": "posterior_beliefs"},
                    emits_events=["beliefs_updated"],
                ),
                WorkflowStep(
                    name="policy_selection",
                    module="ACT",
                    endpoint="/select/policy",
                    dependencies=["belief_update"],
                    emits_events=["action_selected"],
                ),
                WorkflowStep(
                    name="action_execution",
                    module="AGENT",
                    endpoint="/execute/action",
                    dependencies=["policy_selection"],
                    emits_events=["action_executed"],
                ),
                WorkflowStep(
                    name="outcome_evaluation",
                    module="BAYES",
                    endpoint="/evaluate/outcome",
                    dependencies=["action_execution"],
                    feedback_mapping={"observations": "new_observations"},
                ),
            ],
            execution_strategy="feedback_loop",
            max_iterations=10,
            convergence_threshold=0.001,
            tags=["active_inference", "adaptive", "feedback", "advanced"],
        )


class DataFormatConverter:
    """Utility class for converting between different data formats."""

    @staticmethod
    def convert_to_standard_format(
        data: dict[str, Any], source_format: DataFormat, target_format: DataFormat
    ) -> dict[str, Any]:
        """Convert data between different standardized formats."""
        if source_format == target_format:
            return data

        # Implement format conversions as needed
        conversion_map = {
            (
                DataFormat.GEOJSON,
                DataFormat.SPATIAL_TEMPORAL_JSON,
            ): DataFormatConverter._geojson_to_spatial_temporal,
            (
                DataFormat.SPATIAL_TEMPORAL_JSON,
                DataFormat.GEOJSON,
            ): DataFormatConverter._spatial_temporal_to_geojson,
            # Add more conversions as needed
        }

        converter = conversion_map.get((source_format, target_format))
        if converter:
            return converter(data)
        else:
            raise ValueError(
                f"No conversion available from {source_format} to {target_format}"
            )

    @staticmethod
    def _geojson_to_spatial_temporal(data: dict[str, Any]) -> dict[str, Any]:
        """Convert GeoJSON to spatial-temporal format."""
        # Implementation for GeoJSON to spatial-temporal conversion
        return {
            "features": data.get("features", []),
            "temporal_info": data.get("metadata", {}).get("temporal_range"),
            "spatial_bounds": data.get("metadata", {}).get("spatial_bounds"),
            "metadata": data.get("metadata", {}),
        }

    @staticmethod
    def _spatial_temporal_to_geojson(data: dict[str, Any]) -> dict[str, Any]:
        """Convert spatial-temporal format to GeoJSON."""
        # Implementation for spatial-temporal to GeoJSON conversion
        return {
            "type": "FeatureCollection",
            "features": data.get("features", []),
            "metadata": {
                "temporal_range": data.get("temporal_info"),
                "spatial_bounds": data.get("spatial_bounds"),
                **data.get("metadata", {}),
            },
        }


def load_workflow_from_file(file_path: str | Path) -> WorkflowDefinition:
    """Load workflow definition from YAML or JSON file."""
    file_path = Path(file_path)

    with open(file_path) as f:
        if file_path.suffix.lower() in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif file_path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return WorkflowDefinition.from_dict(data)


def save_workflow_to_file(
    workflow: WorkflowDefinition, file_path: str | Path, format: str = "yaml"
) -> None:
    """Save workflow definition to YAML or JSON file."""
    file_path = Path(file_path)
    data = workflow.to_dict()

    with open(file_path, "w") as f:
        if format.lower() in ["yaml", "yml"]:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Partial registry of pre-defined module specifications. Only modules with a
# verified spec (DATA, SPACE) are listed; treat this as sample data for the
# models layer, not a complete catalogue of GEO-INFER modules.
GEO_INFER_MODULES: dict = {
    "DATA": ModuleSpec(
        name="GEO-INFER-DATA",
        module_type=ModuleType.DATA_PROCESSING,
        api_base_url="http://localhost:8001",
        version="1.0.0",
        capabilities=["data_ingestion", "data_fusion", "quality_assurance", "storage"],
        supported_formats=[
            DataFormat.GEOJSON,
            DataFormat.SPATIAL_TEMPORAL_JSON,
            DataFormat.TIME_SERIES,
        ],
        dependencies=["OPS", "SEC"],
    ),
    "SPACE": ModuleSpec(
        name="GEO-INFER-SPACE",
        module_type=ModuleType.SPATIAL_TEMPORAL,
        api_base_url="http://localhost:8002",
        version="1.0.0",
        capabilities=[
            "spatial_analysis",
            "geocoding",
            "spatial_clustering",
            "h3_indexing",
        ],
        supported_formats=[
            DataFormat.GEOJSON,
            DataFormat.RASTER_ARRAY,
            DataFormat.SPATIAL_TEMPORAL_JSON,
        ],
        dependencies=["DATA", "MATH"],
    ),
}
