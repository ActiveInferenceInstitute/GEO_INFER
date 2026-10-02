"""
REST API for GEO-INFER-COG

This module implements REST API endpoints for the GEO-INFER-COG cognitive
geospatial processing module, providing HTTP interfaces for all major
functionality including natural language processing, cognitive reasoning,
knowledge extraction, decision support, and visualization services.

API Endpoints:
- NLP: Natural language processing for spatial content
- Reasoning: Cognitive reasoning and inference
- Knowledge: Knowledge extraction and management
- Decision Support: Spatial decision analysis
- Visualization: Human-centered visualization services
- System: Health monitoring and configuration

Integration Points:
- GEO-INFER-API: Standard API patterns and middleware
- GEO-INFER-SEC: Authentication and authorization
- GEO-INFER-MONITORING: API usage tracking and analytics
"""

from __future__ import annotations
from importlib.metadata import version as distribution_version

import logging
from typing import Any, cast
from datetime import datetime
import traceback

try:
    from flask import Flask, request, jsonify, Response as Response
    from flask_cors import CORS  # type: ignore[import-untyped]

    FLASK_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name not in {"flask", "flask_cors"}:
        raise
    FLASK_AVAILABLE = False
    Flask = None  # type: ignore[misc, assignment]
    request = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]
    CORS = None

from ..core.cognitive_engine import CognitiveProcessingEngine
from ..core.spatial_perception import SpatialPerceptionModel
from ..core.spatial_reasoning import SpatialReasoningEngine
from ..core.spatial_memory import SpatialMemoryModel
from ..spatial_language import SpatialLanguageProcessor
from ..visualization import HumanCenteredVisualizer
from ..decision import SpatialDecisionSupport
from ..models.user_profiles import UserCognitiveProfile, ProfileManager
from ..utils.validation import (
    validate_spatial_data,
    validate_user_profile,
)

logger = logging.getLogger(__name__)


def create_cog_api_app(config: dict[str, Any] | None = None) -> Flask:
    """
    Create and configure the GEO-INFER-COG REST API application.

    Args:
        config: Configuration parameters for the API

    Returns:
        Configured Flask application.
    """
    if not FLASK_AVAILABLE:
        raise ImportError(
            "COG REST API requires geo-infer-cog[api] (Flask and Flask-CORS)"
        )

    app: Any = Flask(__name__)

    # Enable CORS for cross-origin requests
    CORS(app)

    # API configuration
    app.config.update(
        {
            "MAX_CONTENT_LENGTH": 50 * 1024 * 1024,  # 50MB max request size
            "PROPAGATE_EXCEPTIONS": True,
            "JSON_SORT_KEYS": False,
        }
    )

    # Initialize core components
    app.cognitive_engine = None
    app.perception_model = None
    app.reasoning_engine = None
    app.memory_model = None
    app.language_processor = None
    app.visualizer = None
    app.decision_support = None
    app.profile_manager = None

    try:
        # Initialize core components with default configurations
        app.cognitive_engine = CognitiveProcessingEngine()
        app.perception_model = SpatialPerceptionModel()
        app.reasoning_engine = SpatialReasoningEngine()
        app.memory_model = SpatialMemoryModel()
        app.language_processor = SpatialLanguageProcessor()
        app.visualizer = HumanCenteredVisualizer()
        app.decision_support = SpatialDecisionSupport()
        app.profile_manager = ProfileManager()

        logger.info("All COG components initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing COG components: {str(e)}")
        # Continue with limited functionality

    # Register API routes
    register_api_routes(app)

    # Register error handlers
    register_error_handlers(app)

    # Register health check endpoint
    @app.route("/health", methods=["GET"])  # type: ignore[misc]
    def health_check() -> Any:
        """Health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": distribution_version("geo-infer-cog"),
                "components": {
                    "cognitive_engine": app.cognitive_engine is not None,
                    "perception_model": app.perception_model is not None,
                    "reasoning_engine": app.reasoning_engine is not None,
                    "memory_model": app.memory_model is not None,
                    "language_processor": app.language_processor is not None,
                    "visualizer": app.visualizer is not None,
                    "decision_support": app.decision_support is not None,
                    "profile_manager": app.profile_manager is not None,
                },
            }
        )

    return cast(Flask, app)


def register_api_routes(app: Any) -> None:
    """Register all API routes for the COG module."""

    # NLP Routes
    @app.route("/nlp/analyze", methods=["POST"])  # type: ignore[misc]
    def analyze_text() -> Any:
        """Analyze text for spatial content."""
        try:
            data = request.get_json()
            if not data or "text" not in data:
                return jsonify({"error": "Missing text field"}), 400

            text = data["text"]
            language = data.get("language", "en")

            # Initialize language processor if needed
            if not app.language_processor:
                app.language_processor = SpatialLanguageProcessor(language=language)

            # Extract spatial entities
            entities = app.language_processor.extract_spatial_entities(text)

            return jsonify(
                {
                    "analysis_id": f"nlp_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "text": text,
                    "entities": [entity.__dict__ for entity in entities],
                    "entity_count": len(entities),
                }
            )

        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/nlp/entities/extract", methods=["POST"])  # type: ignore[misc]
    def extract_entities() -> Any:
        """Extract spatial entities from text."""
        try:
            data = request.get_json()
            if not data or "text" not in data:
                return jsonify({"error": "Missing text field"}), 400

            text = data["text"]
            entity_types = data.get("entity_types", [])

            if not app.language_processor:
                app.language_processor = SpatialLanguageProcessor()

            entities = app.language_processor.extract_spatial_entities(text)

            # Filter by entity types if specified
            if entity_types:
                entities = [e for e in entities if e.entity_type in entity_types]

            return jsonify(
                {
                    "extraction_id": f"ent_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "entities": [entity.__dict__ for entity in entities],
                    "confidence_scores": {e.text: e.confidence for e in entities},
                }
            )

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/nlp/sentiment/analyze", methods=["POST"])  # type: ignore[misc]
    def analyze_sentiment() -> Any:
        """Analyze sentiment in spatial text."""
        try:
            data = request.get_json()
            if not data or "text" not in data:
                return jsonify({"error": "Missing text field"}), 400

            # Simple sentiment analysis (could be enhanced with ML models)
            text = data["text"].lower()
            positive_words = [
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "love",
                "like",
            ]
            negative_words = [
                "bad",
                "terrible",
                "awful",
                "hate",
                "dislike",
                "poor",
                "worst",
            ]

            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)

            if positive_count > negative_count:
                sentiment = "positive"
                score = min(1.0, positive_count / (positive_count + negative_count + 1))
            elif negative_count > positive_count:
                sentiment = "negative"
                score = -min(
                    1.0, negative_count / (positive_count + negative_count + 1)
                )
            else:
                sentiment = "neutral"
                score = 0.0

            return jsonify(
                {
                    "analysis_id": f"sent_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "sentiment": sentiment,
                    "score": score,
                    "confidence": 0.7,  # Simple rule-based confidence
                }
            )

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Reasoning Routes
    @app.route("/reasoning/infer", methods=["POST"])  # type: ignore[misc]
    def perform_inference() -> Any:
        """Perform cognitive inference on spatial data."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            reasoning_type = data.get("reasoning_type", "deductive")

            if not app.reasoning_engine:
                app.reasoning_engine = SpatialReasoningEngine(
                    reasoning_type=reasoning_type
                )

            # Use the caller's spatial evidence for reasoning.
            spatial_data = data.get("spatial_data", {})
            perception_result = data.get("perception_result", {})

            # Perform reasoning
            reasoning_result = app.reasoning_engine.reason_about_space(
                spatial_data, perception_result, None
            )

            return jsonify(
                {
                    "inference_id": f"inf_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "reasoning_type": reasoning_type,
                    "conclusions": [
                        conc.__dict__
                        for conc in reasoning_result.get("conclusions", [])
                    ],
                    "confidence_score": reasoning_result.get("confidence_score", 0.5),
                }
            )

        except Exception as e:
            logger.error(f"Error in inference: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Knowledge Management Routes
    @app.route("/knowledge/extract", methods=["POST"])  # type: ignore[misc]
    def extract_knowledge() -> Any:
        """Extract structured knowledge from data."""
        try:
            data = request.get_json()
            if not data or "source_data" not in data:
                return jsonify({"error": "Missing source_data field"}), 400

            source_data = data["source_data"]
            extraction_type = data.get("extraction_type", "entities")

            # Validate spatial data
            validation = validate_spatial_data(source_data)
            if not validation["valid"]:
                return jsonify({"error": f"Invalid data: {validation['errors']}"}), 400

            # Extract knowledge based on type
            if extraction_type == "entities":
                if not app.language_processor:
                    app.language_processor = SpatialLanguageProcessor()

                # Extract entities from text if available
                text_content = source_data.get("properties", {}).get("description", "")
                if text_content:
                    entities = app.language_processor.extract_spatial_entities(
                        text_content
                    )
                    extracted_knowledge = {
                        "entities": [entity.__dict__ for entity in entities],
                        "entity_count": len(entities),
                    }
                else:
                    extracted_knowledge = {"entities": [], "entity_count": 0}

            elif extraction_type == "relationships":
                # Extract spatial relationships
                extracted_knowledge = {"relationships": [], "relationship_count": 0}

            else:
                extracted_knowledge = {"extracted_knowledge": {}}

            return jsonify(
                {
                    "extraction_id": f"know_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "extraction_type": extraction_type,
                    "extracted_knowledge": extracted_knowledge,
                    "confidence_scores": {},
                }
            )

        except Exception as e:
            logger.error(f"Error extracting knowledge: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Decision Support Routes
    @app.route("/decision-support/analyze", methods=["POST"])  # type: ignore[misc]
    def analyze_decision() -> Any:
        """Analyze decision scenario."""
        try:
            data = request.get_json()
            if not data or "scenario" not in data:
                return jsonify({"error": "Missing scenario field"}), 400

            scenario = data["scenario"]
            criteria = data.get("criteria", [])
            alternatives = data.get("alternatives", [])

            if not app.decision_support:
                app.decision_support = SpatialDecisionSupport()

            # No stakeholder profiles are inferred without submitted evidence.
            stakeholder_profiles: list[UserCognitiveProfile] = []
            if not app.profile_manager:
                app.profile_manager = ProfileManager()

            # Analyze decision
            analysis_result = app.decision_support.analyze_decision(
                decision_problem=scenario,
                spatial_alternatives=alternatives,
                decision_criteria=criteria,
                stakeholder_profiles=stakeholder_profiles,
            )

            return jsonify(
                {
                    "analysis_id": f"dec_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "scenario": scenario,
                    "recommendations": analysis_result.get("recommendations", []),
                    "evaluation_scores": analysis_result.get("evaluations", {}),
                    "uncertainty_assessment": analysis_result.get(
                        "uncertainty_assessment", {}
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Error in decision analysis: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Visualization Routes
    @app.route("/visualization/create", methods=["POST"])  # type: ignore[misc]
    def create_visualization() -> Any:
        """Create cognitively optimized visualization."""
        try:
            data = request.get_json()
            if not data or "spatial_data" not in data:
                return jsonify({"error": "Missing spatial_data field"}), 400

            spatial_data = data["spatial_data"]
            user_profile_data = data.get("user_profile")
            task_context = data.get("task_context", "general_exploration")

            # Validate spatial data
            validation = validate_spatial_data(spatial_data)
            if not validation["valid"]:
                return (
                    jsonify({"error": f"Invalid spatial data: {validation['errors']}"}),
                    400,
                )

            # Create user profile if provided
            user_profile = None
            if user_profile_data:
                profile_validation = validate_user_profile(user_profile_data)
                if profile_validation["valid"]:
                    user_profile = UserCognitiveProfile.import_profile(
                        user_profile_data
                    )
                else:
                    return (
                        jsonify(
                            {
                                "error": f"Invalid user profile: {profile_validation['errors']}"
                            }
                        ),
                        400,
                    )

            if not app.visualizer:
                app.visualizer = HumanCenteredVisualizer()

            # Create visualization
            visualization_result = app.visualizer.create_optimized_map(
                spatial_data=spatial_data,
                user_cognitive_profile=user_profile,
                task_context=task_context,
            )

            return jsonify(
                {
                    "visualization_id": f"viz_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "visualization_specification": visualization_result.get(
                        "visualization_specification", {}
                    ),
                    "color_scheme": visualization_result.get("color_scheme", {}),
                    "adaptations_applied": visualization_result.get(
                        "adaptations_applied", []
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Cognitive Processing Routes
    @app.route("/cognitive/process", methods=["POST"])  # type: ignore[misc]
    def process_cognitive() -> Any:
        """Process spatial data through cognitive pipeline."""
        try:
            data = request.get_json()
            if not data or "spatial_data" not in data:
                return jsonify({"error": "Missing spatial_data field"}), 400

            spatial_data = data["spatial_data"]
            context = data.get("context", {})
            user_profile_data = data.get("user_profile")

            # Validate spatial data
            validation = validate_spatial_data(spatial_data)
            if not validation["valid"]:
                return (
                    jsonify({"error": f"Invalid spatial data: {validation['errors']}"}),
                    400,
                )

            # Create user profile if provided
            user_profile = None
            if user_profile_data:
                profile_validation = validate_user_profile(user_profile_data)
                if profile_validation["valid"]:
                    user_profile = UserCognitiveProfile.import_profile(
                        user_profile_data
                    )
                else:
                    return (
                        jsonify(
                            {
                                "error": f"Invalid user profile: {profile_validation['errors']}"
                            }
                        ),
                        400,
                    )

            if not app.cognitive_engine:
                app.cognitive_engine = CognitiveProcessingEngine()

            # Process through cognitive pipeline
            processing_result = app.cognitive_engine.process_spatial_input(
                spatial_data=spatial_data, context=context, user_profile=user_profile
            )

            decision_result = processing_result.get("decision_result", {})
            decisions = decision_result.get("decisions", [])
            confidences = [
                d.get("confidence_score", 0.0) for d in decisions if isinstance(d, dict)
            ]
            mean_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            return jsonify(
                {
                    "processing_id": f"cog_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": processing_result.get("processing_time", 0),
                    "cognitive_state": processing_result.get("cognitive_state", {}),
                    "confidence_score": mean_confidence,
                    "decision_recommendations": decisions,
                }
            )

        except Exception as e:
            logger.error(f"Error in cognitive processing: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # User Profile Routes
    @app.route("/profiles/<user_id>", methods=["GET"])  # type: ignore[misc]
    def get_user_profile(user_id: str) -> Any:
        """Get user cognitive profile."""
        try:
            if not app.profile_manager:
                app.profile_manager = ProfileManager()

            profile = app.profile_manager.get_profile(user_id)
            if profile:
                return jsonify(
                    {"user_id": user_id, "profile": profile.get_profile_summary()}
                )
            else:
                return jsonify({"error": f"Profile not found for user {user_id}"}), 404

        except Exception as e:
            logger.error(f"Error getting user profile: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/profiles/<user_id>", methods=["POST"])  # type: ignore[misc]
    def create_user_profile(user_id: str) -> Any:
        """Create or update user cognitive profile."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            if not app.profile_manager:
                app.profile_manager = ProfileManager()

            # Validate profile data
            validation = validate_user_profile(data)
            if not validation["valid"]:
                return (
                    jsonify({"error": f"Invalid profile data: {validation['errors']}"}),
                    400,
                )

            # Create or update profile
            profile = app.profile_manager.create_profile(user_id, data)

            return jsonify(
                {
                    "user_id": user_id,
                    "profile_created": True,
                    "profile_summary": profile.get_profile_summary(),
                }
            )

        except Exception as e:
            logger.error(f"Error creating user profile: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # System Management Routes
    @app.route("/system/status", methods=["GET"])  # type: ignore[misc]
    def get_system_status() -> Any:
        """Get system status and component health."""
        try:
            return jsonify(
                {
                    "status": "operational",
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        "cognitive_engine": app.cognitive_engine is not None,
                        "perception_model": app.perception_model is not None,
                        "reasoning_engine": app.reasoning_engine is not None,
                        "memory_model": app.memory_model is not None,
                        "language_processor": app.language_processor is not None,
                        "visualizer": app.visualizer is not None,
                        "decision_support": app.decision_support is not None,
                        "profile_manager": app.profile_manager is not None,
                    },
                    "version": distribution_version("geo-infer-cog"),
                }
            )

        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/system/metrics", methods=["GET"])  # type: ignore[misc]
    def get_system_metrics() -> Any:
        """Get system performance metrics."""
        try:
            metrics: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": 0,  # Would be calculated from start time
                "requests_processed": 0,  # Would be tracked
                "errors_encountered": 0,  # Would be tracked
                "component_metrics": {},
            }

            # Collect metrics from each component
            components = {
                "cognitive_engine": app.cognitive_engine,
                "perception_model": app.perception_model,
                "reasoning_engine": app.reasoning_engine,
                "memory_model": app.memory_model,
                "language_processor": app.language_processor,
                "visualizer": app.visualizer,
                "decision_support": app.decision_support,
                "profile_manager": app.profile_manager,
            }

            for name, component in components.items():
                if component and hasattr(component, "get_status"):
                    try:
                        metrics["component_metrics"][name] = component.get_status()
                    except Exception as e:
                        logger.warning(f"Error getting metrics for {name}: {str(e)}")
                        metrics["component_metrics"][name] = {"error": str(e)}

            return jsonify(metrics)

        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return jsonify({"error": str(e)}), 500


def register_error_handlers(app: Any) -> None:
    """Register error handlers for the API."""

    @app.errorhandler(400)  # type: ignore[misc]
    def bad_request(error: Any) -> Any:
        return jsonify({"error": "Bad Request", "message": str(error)}), 400

    @app.errorhandler(404)  # type: ignore[misc]
    def not_found(error: Any) -> Any:
        return jsonify({"error": "Not Found", "message": str(error)}), 404

    @app.errorhandler(405)  # type: ignore[misc]
    def method_not_allowed(error: Any) -> Any:
        return jsonify({"error": "Method Not Allowed", "message": str(error)}), 405

    @app.errorhandler(413)  # type: ignore[misc]
    def payload_too_large(error: Any) -> Any:
        return jsonify({"error": "Payload Too Large", "message": str(error)}), 413

    @app.errorhandler(500)  # type: ignore[misc]
    def internal_server_error(error: Any) -> Any:
        logger.error(f"Internal server error: {str(error)}")
        logger.error(traceback.format_exc())
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                }
            ),
            500,
        )

    @app.errorhandler(Exception)  # type: ignore[misc]
    def handle_exception(error: Any) -> Any:
        logger.error(f"Unhandled exception: {str(error)}")
        logger.error(traceback.format_exc())
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                }
            ),
            500,
        )


def run_api_server(
    host: str = "0.0.0.0", port: int = 8000, debug: bool = False
) -> None:
    """
    Run the COG API server.

    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        debug: Enable debug mode
    """
    if not FLASK_AVAILABLE:
        raise RuntimeError(
            "Flask is not installed. Install the API extra with: "
            "pip install 'geo-infer-cog[api]'"
        )

    app = create_cog_api_app()
    if app is None:
        raise RuntimeError("Failed to create the COG API application")

    logger.info(f"Starting COG API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
