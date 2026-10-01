"""
Spatial Language Processor for GEO-INFER-COG

This module implements natural language processing capabilities focused on
geographic references, spatial relations, and place descriptions. It provides
tools for extracting spatial information from text and understanding spatial
language in human communication.

Key Components:
- Geographic Named Entity Recognition (NER)
- Spatial Relation Extraction
- Place Description Interpretation
- Spatial Language Understanding
- Geocoding of informal descriptions
- Spatial concept mapping

Mathematical Foundations:
- Named entity recognition algorithms
- Relation extraction using dependency parsing
- Semantic role labeling for spatial expressions
- Fuzzy matching for place name resolution
- Vector space models for spatial concept similarity
"""

import re
import logging
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SpatialEntity:
    """Represents a spatial entity extracted from text."""

    text: str
    entity_type: str  # 'location', 'region', 'landmark', 'address', 'coordinate'
    confidence: float = 1.0
    coordinates: tuple[float, float] | None = None
    bounding_box: dict[str, float] | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    context: str = ""  # Surrounding text context

    def to_geojson(self) -> dict[str, Any]:
        """Convert to GeoJSON format."""
        feature = {
            "type": "Feature",
            "geometry": None,
            "properties": {
                "name": self.text,
                "entity_type": self.entity_type,
                "confidence": self.confidence,
                "extracted_from_text": True,
            },
        }

        # Add geometry if coordinates available
        if self.coordinates:
            feature["geometry"] = {
                "type": "Point",
                "coordinates": [
                    self.coordinates[1],
                    self.coordinates[0],
                ],  # GeoJSON is [lon, lat]
            }

        elif self.bounding_box:
            # Create polygon from bounding box
            min_lon, min_lat, max_lon, max_lat = self.bounding_box.values()
            feature["geometry"] = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]
                ],
            }

        return feature


@dataclass
class SpatialRelation:
    """Represents a spatial relationship extracted from text."""

    relation_type: str  # 'contains', 'adjacent', 'north_of', 'near', 'far_from', etc.
    source_entity: str
    target_entity: str
    confidence: float = 1.0
    direction: str | None = None  # 'north', 'south', 'east', 'west', etc.
    distance: str | None = None  # 'near', 'far', 'adjacent', etc.
    context: str = ""  # Text context where relation was found


class SpatialLanguageProcessor:
    """
    Natural language processing for spatial and geographic content.

    This processor handles:
    - Geographic named entity recognition and classification
    - Spatial relation extraction from text
    - Place description interpretation and geocoding
    - Spatial language understanding and normalization
    - Fuzzy matching for place name resolution

    The processor uses pattern-based and rule-based approaches to extract
    spatial information from unstructured text, supporting applications
    like geocoding informal descriptions and spatial search.
    """

    def __init__(
        self,
        language: str = "en",
        domain: str = "general",
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize spatial language processor.

        Args:
            language: Language code for processing ('en', 'es', 'fr', etc.)
            domain: Domain specialization ('general', 'urban', 'environmental', 'navigation')
            config: Additional configuration parameters
        """
        self.language = language
        self.domain = domain
        self.config = config or {}

        # Geographic entity patterns
        self.location_patterns = self._initialize_location_patterns()
        self.relation_patterns = self._initialize_relation_patterns()
        self.direction_patterns = self._initialize_direction_patterns()
        self.distance_patterns = self._initialize_distance_patterns()

        # Fuzzy matching for place names
        self.place_name_variants = self._initialize_place_variants()

        # Performance tracking
        self.processing_metrics = {
            "entities_extracted": 0,
            "relations_found": 0,
            "descriptions_processed": 0,
            "geocoding_attempts": 0,
            "successful_geocoding": 0,
        }

        logger.info(f"Spatial Language Processor initialized for language: {language}")

    def _initialize_location_patterns(self) -> dict[str, list[str]]:
        """Initialize patterns for location entity recognition."""
        return {
            "city": [
                r"\b([A-Z][a-z]+,\s*[A-Z]{2})\b",  # City, State
                r"\b([A-Z][a-z]+\s+City)\b",  # City
                r"\b([A-Z][a-z]+\s+Town)\b",  # Town
            ],
            "street": [
                r"\b(\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(Street|Avenue|Road|Drive|Lane|Way|Place|Boulevard))\b",
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(Street|Avenue|Road|Drive|Lane|Way|Place|Boulevard))\b",
            ],
            "landmark": [
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(Park|Lake|River|Mountain|Building|Center|Station|Airport|Hospital|School|University))\b",
                r"\b(the\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))\b",  # "the [landmark]"
            ],
            "region": [
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(County|Region|District|Area|Zone))\b",
                r"\b([A-Z][a-z]+\s+(Valley|Basin|Plateau|Desert|Forest|Mountains))\b",
            ],
            "coordinate": [
                r"\b(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\b",  # lat, lon
                r"\b(-?\d+\.?\d*)\s+(degrees?\s+)?([NS])\s*,?\s*(-?\d+\.?\d*)\s+(degrees?\s+)?([EW])\b",  # DD N, DD W format
            ],
        }

    def _initialize_relation_patterns(self) -> dict[str, list[str]]:
        """Initialize patterns for spatial relation extraction."""
        return {
            "contains": [
                r"\b(in|inside|within|contained\s+in)\s+([A-Z][a-z\s]+)\b",
                r"\b([A-Z][a-z\s]+)\s+(contains|includes|has)\s+([A-Z][a-z\s]+)\b",
            ],
            "adjacent": [
                r"\b(next\s+to|beside|adjacent\s+to|bordering)\s+([A-Z][a-z\s]+)\b",
                r"\b([A-Z][a-z\s]+)\s+(borders|adjoins|abuts)\s+([A-Z][a-z\s]+)\b",
            ],
            "near": [
                r"\b(near|close\s+to|proximate\s+to)\s+([A-Z][a-z\s]+)\b",
                r"\b([A-Z][a-z\s]+)\s+(near|close\s+to|proximate\s+to)\s+([A-Z][a-z\s]+)\b",
            ],
            "north_of": [
                r"\b(north\s+of|to\s+the\s+north\s+of)\s+([A-Z][a-z\s]+)\b",
                r"\b([A-Z][a-z\s]+)\s+(north\s+of|to\s+the\s+north\s+of)\s+([A-Z][a-z\s]+)\b",
            ],
            "south_of": [
                r"\b(south\s+of|to\s+the\s+south\s+of)\s+([A-Z][a-z\s]+)\b",
                r"\b([A-Z][a-z\s]+)\s+(south\s+of|to\s+the\s+south\s+of)\s+([A-Z][a-z\s]+)\b",
            ],
            "east_of": [
                r"\b(east\s+of|to\s+the\s+east\s+of)\s+([A-Z][a-z\s]+)\b",
                r"\b([A-Z][a-z\s]+)\s+(east\s+of|to\s+the\s+east\s+of)\s+([A-Z][a-z\s]+)\b",
            ],
            "west_of": [
                r"\b(west\s+of|to\s+the\s+west\s+of)\s+([A-Z][a-z\s]+)\b",
                r"\b([A-Z][a-z\s]+)\s+(west\s+of|to\s+the\s+west\s+of)\s+([A-Z][a-z\s]+)\b",
            ],
        }

    def _initialize_direction_patterns(self) -> dict[str, list[str]]:
        """Initialize patterns for direction expressions."""
        return {
            "north": [r"\b(north|northeast|northwest|northern|northwards)\b"],
            "south": [r"\b(south|southeast|southwest|southern|southwards)\b"],
            "east": [r"\b(east|northeast|southeast|eastern|eastwards)\b"],
            "west": [r"\b(west|northwest|southwest|western|westwards)\b"],
            "center": [r"\b(center|central|middle|downtown)\b"],
            "peripheral": [r"\b(outskirts|peripheral|edge|boundary)\b"],
        }

    def _initialize_distance_patterns(self) -> dict[str, list[str]]:
        """Initialize patterns for distance expressions."""
        return {
            "adjacent": [r"\b(adjacent|next\s+to|beside|touching)\b"],
            "near": [r"\b(near|close\s+to|nearby|proximate)\b"],
            "moderate": [r"\b(a\s+few\s+miles|several\s+blocks|moderate\s+distance)\b"],
            "far": [r"\b(far\s+from|distant|remote|far\s+away)\b"],
            "walking_distance": [
                r"\b(walking\s+distance|short\s+walk|few\s+minutes?\s+walk)\b"
            ],
            "driving_distance": [
                r"\b(driving\s+distance|short\s+drive|few\s+minutes?\s+drive)\b"
            ],
        }

    def _initialize_place_variants(self) -> dict[str, list[str]]:
        """Initialize common place name variants for fuzzy matching."""
        return {
            "new_york": [
                "new york",
                "nyc",
                "new york city",
                "manhattan",
                "brooklyn",
                "queens",
                "bronx",
                "staten island",
            ],
            "los_angeles": [
                "los angeles",
                "la",
                "l.a.",
                "los angeles county",
                "hollywood",
                "beverly hills",
            ],
            "chicago": ["chicago", "windy city", "chi-town"],
            "san_francisco": ["san francisco", "sf", "s.f.", "san fran", "bay area"],
            "london": ["london", "londinium", "the city"],
            "paris": ["paris", "city of light", "paname"],
            "tokyo": ["tokyo", "edo", "tokio"],
            "sydney": ["sydney", "harbour city"],
        }

    def extract_spatial_entities(self, text: str) -> list[SpatialEntity]:
        """
        Extract spatial entities from text using pattern matching.

        Args:
            text: Input text to process

        Returns:
            List of extracted spatial entities
        """
        entities = []

        # Extract location entities using patterns
        for entity_type, patterns in self.location_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(0)

                    # Determine confidence based on pattern specificity
                    confidence = self._calculate_entity_confidence(
                        entity_text, entity_type, pattern
                    )

                    entity = SpatialEntity(
                        text=entity_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        context=self._extract_context(text, match.start(), match.end()),
                    )

                    # Try to geocode if possible
                    coordinates = self._attempt_geocoding(entity_text, entity_type)
                    if coordinates:
                        entity.coordinates = coordinates
                        self.processing_metrics["successful_geocoding"] += 1

                    entities.append(entity)
                    self.processing_metrics["entities_extracted"] += 1

        # Remove duplicates while preserving highest confidence
        unique_entities = self._deduplicate_entities(entities)

        logger.info(f"Extracted {len(unique_entities)} spatial entities from text")
        return unique_entities

    def extract_spatial_relations(
        self, text: str, entities: list[SpatialEntity]
    ) -> list[SpatialRelation]:
        """
        Extract spatial relationships from text.

        Args:
            text: Input text to process
            entities: List of entities found in the text

        Returns:
            List of extracted spatial relations
        """
        relations = []

        # Get entity names for relation matching
        entity_names = {entity.text.lower(): entity.text for entity in entities}

        # Extract relations using patterns
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        # Identify source and target entities
                        source_text = groups[-2]  # Usually the first entity mentioned
                        target_text = groups[-1]  # Usually the second entity mentioned

                        # Normalize entity names
                        source_norm = source_text.lower().strip()
                        target_norm = target_text.lower().strip()

                        # Match to extracted entities
                        source_entity = entity_names.get(source_norm)
                        target_entity = entity_names.get(target_norm)

                        if source_entity and target_entity:
                            relation = SpatialRelation(
                                relation_type=relation_type,
                                source_entity=source_entity,
                                target_entity=target_entity,
                                confidence=self._calculate_relation_confidence(
                                    match.group(0), relation_type
                                ),
                                context=self._extract_context(
                                    text, match.start(), match.end()
                                ),
                            )

                            # Extract additional details (direction, distance)
                            direction = self._extract_direction_from_text(
                                match.group(0)
                            )
                            if direction:
                                relation.direction = direction

                            distance = self._extract_distance_from_text(match.group(0))
                            if distance:
                                relation.distance = distance

                            relations.append(relation)
                            self.processing_metrics["relations_found"] += 1

        logger.info(f"Extracted {len(relations)} spatial relations from text")
        return relations

    def process_place_description(self, description: str) -> dict[str, Any]:
        """
        Process and interpret a place description.

        Args:
            description: Text description of a place or location

        Returns:
            Structured interpretation of the description
        """
        self.processing_metrics["descriptions_processed"] += 1

        interpretation = {
            "original_description": description,
            "entities": [],
            "relations": [],
            "spatial_concepts": [],
            "uncertainty_indicators": [],
            "geocoding_candidates": [],
            "interpretation_confidence": 0.0,
        }

        # Extract entities
        entities = self.extract_spatial_entities(description)
        interpretation["entities"] = [entity.__dict__ for entity in entities]

        # Extract relations
        relations = self.extract_spatial_relations(description, entities)
        interpretation["relations"] = [relation.__dict__ for relation in relations]

        # Extract spatial concepts
        concepts = self._extract_spatial_concepts(description)
        interpretation["spatial_concepts"] = concepts

        # Identify uncertainty indicators
        uncertainty_indicators = self._identify_uncertainty_indicators(description)
        interpretation["uncertainty_indicators"] = uncertainty_indicators

        # Generate geocoding candidates
        candidates = self._generate_geocoding_candidates(entities, relations, concepts)
        interpretation["geocoding_candidates"] = candidates

        # Calculate overall interpretation confidence
        interpretation["interpretation_confidence"] = (
            self._calculate_interpretation_confidence(
                entities, relations, concepts, uncertainty_indicators
            )
        )

        logger.info(
            f"Processed place description with confidence {interpretation['interpretation_confidence']:.3f}"
        )
        return interpretation

    def _calculate_entity_confidence(
        self, entity_text: str, entity_type: str, pattern: str
    ) -> float:
        """Calculate confidence score for extracted entity."""
        base_confidence = 0.5

        # Adjust based on entity type
        type_confidence = {
            "city": 0.8,
            "street": 0.9,
            "landmark": 0.7,
            "region": 0.6,
            "coordinate": 0.95,
        }

        base_confidence = type_confidence.get(entity_type, 0.5)

        # Adjust based on pattern specificity
        if pattern.count(r"\b") > 2:  # More word boundaries = more specific
            base_confidence += 0.1

        # Adjust based on text length (longer = more specific)
        if len(entity_text) > 10:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _calculate_relation_confidence(
        self, relation_text: str, relation_type: str
    ) -> float:
        """Calculate confidence score for extracted relation."""
        base_confidence = 0.6

        # Adjust based on relation type specificity
        specific_relations = ["north_of", "south_of", "east_of", "west_of", "contains"]
        if relation_type in specific_relations:
            base_confidence = 0.8

        # Adjust based on text clarity
        if len(relation_text.split()) > 3:  # More words = clearer relation
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _extract_context(
        self, text: str, start_pos: int, end_pos: int, context_length: int = 50
    ) -> str:
        """Extract surrounding context for an entity or relation."""
        text_length = len(text)

        # Calculate context boundaries
        context_start = max(0, start_pos - context_length)
        context_end = min(text_length, end_pos + context_length)

        context = text[context_start:context_end]

        # Add ellipsis if context was truncated
        if context_start > 0:
            context = "..." + context
        if context_end < text_length:
            context = context + "..."

        return context.strip()

    def _deduplicate_entities(
        self, entities: list[SpatialEntity]
    ) -> list[SpatialEntity]:
        """Remove duplicate entities while preserving highest confidence."""
        unique_entities: dict[tuple[str, str], SpatialEntity] = {}

        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)

            if (
                key not in unique_entities
                or entity.confidence > unique_entities[key].confidence
            ):
                unique_entities[key] = entity

        return list(unique_entities.values())

    def _attempt_geocoding(
        self, entity_text: str, entity_type: str
    ) -> tuple[float, float] | None:
        """Attempt to geocode an entity text to coordinates."""
        self.processing_metrics["geocoding_attempts"] += 1

        # Simple geocoding based on known place variants
        entity_lower = entity_text.lower()

        for canonical_name, variants in self.place_name_variants.items():
            if entity_lower in [v.lower() for v in variants]:
                # Return coordinates for known places (simplified)
                coordinates_map = {
                    "new_york": (40.7128, -74.0060),
                    "los_angeles": (34.0522, -118.2437),
                    "chicago": (41.8781, -87.6298),
                    "san_francisco": (37.7749, -122.4194),
                    "london": (51.5074, -0.1278),
                    "paris": (48.8566, 2.3522),
                    "tokyo": (35.6762, 139.6503),
                    "sydney": (-33.8688, 151.2093),
                }

                if canonical_name in coordinates_map:
                    return coordinates_map[canonical_name]

        # For coordinates, try to parse directly
        if entity_type == "coordinate":
            coord_match = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", entity_text)
            if coord_match:
                lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)

        return None

    def _extract_direction_from_text(self, text: str) -> str | None:
        """Extract directional information from text."""
        text_lower = text.lower()

        for direction, patterns in self.direction_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return direction

        return None

    def _extract_distance_from_text(self, text: str) -> str | None:
        """Extract distance information from text."""
        text_lower = text.lower()

        for distance, patterns in self.distance_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return distance

        return None

    def _extract_spatial_concepts(self, description: str) -> list[dict[str, Any]]:
        """Extract spatial concepts from description."""
        concepts = []

        # Simple concept extraction based on keywords
        concept_keywords = {
            "urban": ["city", "urban", "metropolitan", "downtown", "suburb"],
            "rural": ["rural", "countryside", "village", "farmland", "agricultural"],
            "natural": ["forest", "mountain", "river", "lake", "park", "wilderness"],
            "commercial": ["shopping", "business", "commercial", "retail", "office"],
            "residential": ["residential", "housing", "neighborhood", "community"],
            "transportation": ["highway", "road", "street", "transport", "transit"],
        }

        description_lower = description.lower()

        for concept_type, keywords in concept_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    concepts.append(
                        {
                            "concept_type": concept_type,
                            "keyword": keyword,
                            "confidence": 0.7,  # Moderate confidence for keyword-based extraction
                        }
                    )
                    break  # Only add concept type once

        return concepts

    def _identify_uncertainty_indicators(self, description: str) -> list[str]:
        """Identify words/phrases indicating uncertainty in description."""
        uncertainty_words = [
            "approximately",
            "about",
            "around",
            "roughly",
            "somewhere",
            "maybe",
            "perhaps",
            "possibly",
            "could be",
            "might be",
            "near",
            "close to",
            "in the vicinity of",
            "sort of",
            "kind of",
        ]

        indicators = []
        description_lower = description.lower()

        for word in uncertainty_words:
            if word in description_lower:
                indicators.append(word)

        return indicators

    def _generate_geocoding_candidates(
        self,
        entities: list[SpatialEntity],
        relations: list[SpatialRelation],
        concepts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate geocoding candidates based on extracted information."""
        candidates = []

        for entity in entities:
            candidate = {
                "entity_text": entity.text,
                "entity_type": entity.entity_type,
                "confidence": entity.confidence,
                "candidate_coordinates": entity.coordinates,
                "candidate_bbox": entity.bounding_box,
                "geocoding_method": "pattern_matching"
                if entity.coordinates
                else "none",
            }

            # Add contextual information
            if entity.coordinates:
                candidate["geocoding_success"] = True
            else:
                candidate["geocoding_success"] = False
                candidate["failure_reason"] = "No matching pattern found"

            candidates.append(candidate)

        return candidates

    def _calculate_interpretation_confidence(
        self,
        entities: list[SpatialEntity],
        relations: list[SpatialRelation],
        concepts: list[dict[str, Any]],
        uncertainty_indicators: list[str],
    ) -> float:
        """Calculate overall confidence in description interpretation."""
        factors = []

        # Entity confidence
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            factors.append(avg_entity_confidence * 0.4)

        # Relation confidence
        if relations:
            avg_relation_confidence = sum(r.confidence for r in relations) / len(
                relations
            )
            factors.append(avg_relation_confidence * 0.3)

        # Concept confidence
        if concepts:
            avg_concept_confidence = sum(c["confidence"] for c in concepts) / len(
                concepts
            )
            factors.append(avg_concept_confidence * 0.2)

        # Uncertainty penalty
        uncertainty_penalty = len(uncertainty_indicators) * 0.1
        factors.append(max(0.0, 0.8 - uncertainty_penalty))

        return sum(factors) / len(factors) if factors else 0.3
