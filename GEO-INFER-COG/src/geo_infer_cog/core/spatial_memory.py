"""
Spatial Memory Model for GEO-INFER-COG

This module implements spatial memory models that simulate how humans store,
retrieve, and consolidate spatial knowledge. The models are based on cognitive
psychology research on spatial memory systems and provide the foundation for
long-term spatial knowledge management.

Key Components:
- Working memory for temporary spatial information
- Long-term memory for persistent spatial knowledge
- Episodic memory for spatial experiences
- Semantic memory for spatial concepts and relationships
- Memory consolidation and forgetting mechanisms

Mathematical Foundations:
- Baddeley working memory model (Baddeley, 1986)
- Memory consolidation theories (McGaugh, 2000)
- Forgetting curve models (Ebbinghaus, 1885)
- Cognitive map formation (Tolman, 1948)
- Spatial memory chunking and organization
"""

import itertools
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from ..utils.rng import resolve_rng

# Module-level monotonic counter backing memory item IDs; combined with an
# optional generator so IDs stay collision-free and reproducible.
_ITEM_ID_SEQUENCE: "itertools.count[int]" = itertools.count(1)


def _extract_bbox(spatial: dict[str, Any]) -> tuple | None:
    """Extract an axis-aligned bbox as (min_x, min_y, max_x, max_y) if present.

    Returns None when no valid 4-number bbox is available.
    """
    bbox = spatial.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        min_x, min_y, max_x, max_y = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if max_x < min_x or max_y < min_y:
        return None
    return (min_x, min_y, max_x, max_y)


def _bboxes_intersect(a: tuple, b: tuple) -> bool:
    """Check whether two (min_x, min_y, max_x, max_y) boxes intersect."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


logger = logging.getLogger(__name__)


@dataclass
class SpatialMemoryItem:
    """Represents an item stored in spatial memory."""

    item_id: str
    content: dict[str, Any]
    memory_type: str  # 'working', 'episodic', 'semantic', 'procedural'
    importance: float = 1.0
    confidence: float = 1.0
    access_count: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime | None = None
    decay_rate: float = 0.1  # Items decay over time
    spatial_context: dict[str, Any] = field(default_factory=dict)

    def calculate_retrieval_probability(self) -> float:
        """Calculate probability of successful retrieval."""
        # Base probability from importance and confidence
        base_prob = self.importance * self.confidence

        # Decay factor based on time since last access
        if self.last_access_time:
            time_since_access = (datetime.now() - self.last_access_time).total_seconds()
            decay_factor = math.exp(
                -self.decay_rate * time_since_access / 3600
            )  # Hours
        else:
            # Decay since creation for never-accessed items
            time_since_creation = (datetime.now() - self.creation_time).total_seconds()
            decay_factor = math.exp(-self.decay_rate * time_since_creation / 3600)

        # Access history factor (more accesses = easier retrieval)
        access_factor = 1.0 + (self.access_count * 0.1)

        return min(1.0, base_prob * decay_factor * access_factor)

    def update_access(self) -> None:
        """Update access statistics when item is retrieved."""
        self.access_count += 1
        self.last_access_time = datetime.now()

        # Boost importance with successful access
        self.importance = min(1.0, self.importance + 0.05)

    def decay_memory(self) -> None:
        """Apply memory decay over time."""
        time_since_creation = (datetime.now() - self.creation_time).total_seconds()

        # Apply forgetting curve (Ebbinghaus-style)
        decay_factor = math.exp(-self.decay_rate * time_since_creation / 3600)

        # Reduce confidence and importance
        self.confidence *= decay_factor
        self.importance *= decay_factor

        # Mark for potential removal if decayed too much
        if self.confidence < 0.1:
            self.decay_rate *= 1.5  # Increase decay rate for weak memories


class MemoryConsolidation:
    """Handles memory consolidation from working to long-term memory."""

    def __init__(
        self, consolidation_threshold: float = 0.7, consolidation_delay: int = 300
    ):
        """
        Initialize memory consolidation system.

        Args:
            consolidation_threshold: Minimum importance for consolidation (0-1)
            consolidation_delay: Minimum time in working memory before consolidation (seconds)
        """
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_delay = consolidation_delay
        self.pending_consolidation: list[SpatialMemoryItem] = []

    def check_for_consolidation(
        self, working_memory_items: list[SpatialMemoryItem]
    ) -> list[SpatialMemoryItem]:
        """Check which working memory items are ready for consolidation."""
        ready_for_consolidation = []

        for item in working_memory_items:
            # Check if item meets consolidation criteria
            time_in_memory = (datetime.now() - item.creation_time).total_seconds()

            if (
                item.importance >= self.consolidation_threshold
                and time_in_memory >= self.consolidation_delay
                and item.memory_type == "working"
            ):
                # Convert to long-term memory
                item.memory_type = "long_term"
                item.decay_rate *= 0.5  # Slower decay for long-term memories
                ready_for_consolidation.append(item)

        return ready_for_consolidation


class SpatialMemoryModel:
    """
    Comprehensive spatial memory model for geospatial knowledge management.

    This model simulates human spatial memory systems including:
    - Working memory for active spatial processing
    - Long-term memory for persistent spatial knowledge
    - Episodic memory for spatial experiences and events
    - Semantic memory for spatial concepts and relationships
    - Procedural memory for spatial skills and navigation strategies

    The model implements memory processes such as:
    - Encoding of spatial information
    - Storage and organization of spatial knowledge
    - Retrieval and reconstruction of spatial memories
    - Consolidation from working to long-term memory
    - Forgetting and memory decay
    - Interference and memory competition
    """

    def __init__(
        self,
        memory_types: list[str] | None = None,
        consolidation_strategy: str = "adaptive",
        config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize spatial memory model.

        Args:
            memory_types: Types of memory to model ('working', 'long_term', 'episodic', 'semantic', 'procedural')
            consolidation_strategy: Strategy for memory consolidation ('adaptive', 'threshold', 'time_based')
            config: Additional configuration parameters
            rng: Optional random generator for stochastic retrieval and ID
                suffixes. When omitted, a fixed-seed generator is used so the
                model is deterministic by default.
        """
        self.memory_types = list(
            memory_types
            if memory_types is not None
            else ["working", "long_term", "episodic"]
        )
        self.consolidation_strategy = consolidation_strategy
        self.config = config or {}

        # Random generator for retrieval sampling. Resolved through the
        # repo-wide resolve_rng pattern; None resolves to a fixed seed, so
        # retrieval outcomes are reproducible unless a caller supplies a
        # generator (or an unseeded one for fresh entropy).
        self._rng = resolve_rng(rng)
        self._rng_supplied = rng is not None

        # Memory storage systems
        self.memory_storage: dict[str, dict[str, SpatialMemoryItem]] = {
            mem_type: {} for mem_type in self.memory_types
        }

        # Memory capacity limits
        self.memory_capacities = {
            "working": self.config.get(
                "working_memory_capacity", 7
            ),  # Miller's magic number
            "long_term": self.config.get("long_term_capacity", 10000),
            "episodic": self.config.get("episodic_capacity", 5000),
            "semantic": self.config.get("semantic_capacity", 2000),
            "procedural": self.config.get("procedural_capacity", 100),
        }

        # Memory consolidation system
        self.consolidation_system = MemoryConsolidation(
            consolidation_threshold=self.config.get("consolidation_threshold", 0.7),
            consolidation_delay=self.config.get("consolidation_delay", 300),
        )

        # Memory decay parameters
        self.decay_parameters = {
            "working": {"rate": 0.2, "interval": 60},  # Fast decay, short interval
            "long_term": {"rate": 0.05, "interval": 3600},  # Slow decay, long interval
            "episodic": {
                "rate": 0.1,
                "interval": 1800,
            },  # Moderate decay, medium interval
            "semantic": {
                "rate": 0.02,
                "interval": 7200,
            },  # Very slow decay, long interval
            "procedural": {
                "rate": 0.01,
                "interval": 14400,
            },  # Extremely slow decay, very long interval
        }

        # Performance tracking
        self.memory_metrics = {
            "items_stored": 0,
            "items_retrieved": 0,
            "items_consolidated": 0,
            "items_forgotten": 0,
            "average_retrieval_time": 0.0,
        }

        # Memory organization and indexing
        self.spatial_index: dict[str, list[str]] = {}  # For spatial memory organization
        self.temporal_index: dict[
            str, list[str]
        ] = {}  # For temporal memory organization
        self.conceptual_index: dict[
            str, list[str]
        ] = {}  # For semantic memory organization

        logger.info(f"Spatial Memory Model initialized with types: {self.memory_types}")

    def store_spatial_memory(
        self,
        content: dict[str, Any],
        memory_type: str = "working",
        importance: float = 1.0,
        spatial_context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store spatial information in memory.

        Args:
            content: Spatial content to store
            memory_type: Type of memory to use
            importance: Importance weight (0-1)
            spatial_context: Spatial context information
            metadata: Additional metadata

        Returns:
            Memory item ID for later retrieval
        """
        if memory_type not in self.memory_types:
            raise ValueError(f"Unsupported memory type: {memory_type}")

        # Check capacity limits
        current_count = len(self.memory_storage[memory_type])
        if current_count >= self.memory_capacities[memory_type]:
            # Remove least important item if at capacity
            self._remove_least_important(memory_type)

        # Item IDs combine a module-level monotonic counter (collision-free
        # within the process) with a generator-derived suffix when a caller
        # supplied an rng.
        item_id = f"{memory_type}_{next(_ITEM_ID_SEQUENCE)}"
        if self._rng_supplied:
            item_id += f"_{int(self._rng.integers(0, 1_000_000))}"

        memory_item = SpatialMemoryItem(
            item_id=item_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            spatial_context=spatial_context or {},
        )

        # Store in memory system
        self.memory_storage[memory_type][item_id] = memory_item

        # Update indexes for efficient retrieval
        self._update_indexes(memory_item)

        self.memory_metrics["items_stored"] += 1

        logger.info(f"Spatial memory item {item_id} stored in {memory_type} memory")
        return item_id

    def retrieve_spatial_memory(
        self, item_id: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """
        Retrieve spatial information from memory.

        Args:
            item_id: ID of memory item to retrieve
            context: Retrieval context (spatial, temporal, semantic cues)

        Returns:
            Retrieved memory content or None if not found/accessible
        """
        start_time = datetime.now()

        # Search across all memory types
        for memory_type in self.memory_types:
            if item_id in self.memory_storage[memory_type]:
                item = self.memory_storage[memory_type][item_id]

                # Check retrieval probability
                retrieval_prob = item.calculate_retrieval_probability()

                if self._rng.random() < retrieval_prob:
                    # Successful retrieval
                    item.update_access()
                    self.memory_metrics["items_retrieved"] += 1

                    # Update retrieval time metrics
                    retrieval_time = (datetime.now() - start_time).total_seconds()
                    self.memory_metrics["average_retrieval_time"] = (
                        self.memory_metrics["average_retrieval_time"]
                        * (self.memory_metrics["items_retrieved"] - 1)
                        + retrieval_time
                    ) / self.memory_metrics["items_retrieved"]

                    logger.info(f"Spatial memory item {item_id} successfully retrieved")
                    return item.content
                else:
                    # Retrieval failure due to decay/interference
                    logger.warning(
                        f"Failed to retrieve memory item {item_id} (probability: {retrieval_prob:.3f})"
                    )
                    return None

        logger.warning(f"Memory item {item_id} not found")
        return None

    def update_memory(
        self,
        perception_result: dict[str, Any],
        reasoning_result: dict[str, Any],
        cognitive_state: Any,
    ) -> dict[str, Any]:
        """
        Update memory based on perception and reasoning results.

        Args:
            perception_result: Results from spatial perception processing
            reasoning_result: Results from spatial reasoning
            cognitive_state: Current cognitive state

        Returns:
            Dictionary containing memory update results
        """
        update_results = {
            "items_stored": 0,
            "items_consolidated": 0,
            "memory_cleanup_performed": False,
        }

        # Store perception results in working memory
        if "spatial_elements" in perception_result:
            for element in perception_result["spatial_elements"]:
                element_content = {
                    "type": "spatial_element",
                    "geometry": element.get("geometry", {}),
                    "saliency": element.get("visual_saliency", 0.0),
                    "perceptual_group": element.get("perceptual_group", ""),
                    "attention_weight": element.get("attention_weight", 0.0),
                }

                self.store_spatial_memory(
                    content=element_content,
                    memory_type="working",
                    importance=element.get("visual_saliency", 0.5),
                    spatial_context={"scale": element.get("scale_level", "medium")},
                )
                update_results["items_stored"] += 1

        # Store reasoning results
        if "conclusions" in reasoning_result:
            for conclusion in reasoning_result["conclusions"]:
                conclusion_content = {
                    "type": "spatial_reasoning",
                    "source_region": conclusion.source_region,
                    "target_region": conclusion.target_region,
                    "relation_type": conclusion.relation_type,
                    "confidence": conclusion.confidence,
                    "reasoning_path": conclusion.reasoning_path,
                }

                self.store_spatial_memory(
                    content=conclusion_content,
                    memory_type="working",
                    importance=conclusion.confidence,
                    spatial_context={"relation_type": conclusion.relation_type},
                )
                update_results["items_stored"] += 1

        # Perform memory consolidation
        consolidated_items = self._perform_memory_consolidation()
        update_results["items_consolidated"] = len(consolidated_items)

        # Perform memory cleanup (decay and forgetting)
        cleanup_results = self._perform_memory_cleanup()
        update_results["memory_cleanup_performed"] = (
            cleanup_results["items_removed"] > 0
        )

        return update_results

    def _perform_memory_consolidation(self) -> list[SpatialMemoryItem]:
        """Perform memory consolidation from working to long-term memory."""
        consolidated_items = []

        # Get working memory items ready for consolidation
        working_items = list(self.memory_storage["working"].values())
        ready_items = self.consolidation_system.check_for_consolidation(working_items)

        for item in ready_items:
            # Move to long-term memory
            del self.memory_storage["working"][item.item_id]
            self.memory_storage["long_term"][item.item_id] = item
            consolidated_items.append(item)

            self.memory_metrics["items_consolidated"] += 1

        return consolidated_items

    def _perform_memory_cleanup(self) -> dict[str, Any]:
        """Perform memory cleanup including decay and forgetting."""
        cleanup_results: dict[str, Any] = {
            "items_decayed": 0,
            "items_removed": 0,
            "memory_types_affected": [],
        }

        for memory_type in self.memory_types:
            if memory_type not in self.memory_storage:
                continue

            items_to_remove = []
            for item_id, item in self.memory_storage[memory_type].items():
                # Apply decay
                item.decay_memory()
                cleanup_results["items_decayed"] += 1

                # Check for removal (too decayed or at capacity)
                if item.confidence < 0.05:  # Very weak memories
                    items_to_remove.append(item_id)
                elif (
                    memory_type == "working"
                    and len(self.memory_storage[memory_type])
                    > self.memory_capacities[memory_type]
                ):
                    # Remove least important items when at capacity
                    items_to_remove.append(item_id)

            # Remove decayed items
            for item_id in items_to_remove:
                del self.memory_storage[memory_type][item_id]
                cleanup_results["items_removed"] += 1

            if items_to_remove:
                cleanup_results["memory_types_affected"].append(memory_type)

        return cleanup_results

    def _remove_least_important(self, memory_type: str) -> None:
        """Remove the least important item from memory when at capacity."""
        if (
            memory_type not in self.memory_storage
            or not self.memory_storage[memory_type]
        ):
            return

        # Find item with lowest importance
        items = list(self.memory_storage[memory_type].values())
        least_important = min(items, key=lambda x: x.importance)

        del self.memory_storage[memory_type][least_important.item_id]
        self.memory_metrics["items_forgotten"] += 1

        logger.info(
            f"Removed least important item {least_important.item_id} from {memory_type} memory"
        )

    def _update_indexes(self, memory_item: SpatialMemoryItem) -> None:
        """Update spatial, temporal, and conceptual indexes for efficient retrieval."""
        # Spatial index
        spatial_key = self._generate_spatial_key(memory_item)
        if spatial_key:
            if spatial_key not in self.spatial_index:
                self.spatial_index[spatial_key] = []
            self.spatial_index[spatial_key].append(memory_item.item_id)

        # Temporal index
        temporal_key = memory_item.creation_time.strftime("%Y%m%d_%H")
        if temporal_key not in self.temporal_index:
            self.temporal_index[temporal_key] = []
        self.temporal_index[temporal_key].append(memory_item.item_id)

        # Conceptual index (based on content type)
        content_type = memory_item.content.get("type", "unknown")
        if content_type not in self.conceptual_index:
            self.conceptual_index[content_type] = []
        self.conceptual_index[content_type].append(memory_item.item_id)

    def _generate_spatial_key(self, memory_item: SpatialMemoryItem) -> str | None:
        """Generate spatial index key for a memory item."""
        spatial_context = memory_item.spatial_context

        # Use scale and relation type for spatial organization
        scale = spatial_context.get("scale", "unknown")
        relation_type = spatial_context.get("relation_type", "unknown")

        return f"{scale}_{relation_type}"

    def search_memory(
        self,
        query: dict[str, Any],
        memory_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search memory using spatial, temporal, or conceptual criteria.

            query: Search query with criteria. Accepted keys:
                - "spatial_bounds": dict with an optional "bbox" ([min_x, min_y,
                  max_x, max_y]) and/or a "scale" label (small/medium/large).
                  When both the item's spatial_context and the query carry a
                  bbox, real bbox intersection is required; the "scale" label
                  is only compared when geometry is absent.
                - "temporal_range": (start, end) datetime tuple covering the
                  item creation time.
                - "content_type": exact match against item content "type".
                - "min_importance": minimum item importance threshold.
            memory_types: Memory types to search (default: all)
            limit: Maximum number of results to return

        Returns:
            List of matching memory items
        """
        search_types = memory_types or self.memory_types
        results: list[dict[str, Any]] = []

        for memory_type in search_types:
            if memory_type not in self.memory_storage:
                continue

            for item_id, item in self.memory_storage[memory_type].items():
                # Check if item matches query criteria
                if self._item_matches_query(item, query):
                    results.append(
                        {
                            "item_id": item_id,
                            "content": item.content,
                            "memory_type": memory_type,
                            "importance": item.importance,
                            "confidence": item.confidence,
                            "retrieval_probability": item.calculate_retrieval_probability(),
                        }
                    )

        # Sort by relevance and limit results
        results.sort(key=lambda x: x["retrieval_probability"], reverse=True)
        return results[:limit]

    def _item_matches_query(
        self, item: SpatialMemoryItem, query: dict[str, Any]
    ) -> bool:
        """Check if memory item matches search query."""
        # Spatial criteria
        if "spatial_bounds" in query:
            item_spatial = item.spatial_context
            query_bounds = query["spatial_bounds"]

            # Simple bounding box overlap check
            if not self._spatial_overlap(item_spatial, query_bounds):
                return False

        # Temporal criteria
        if "temporal_range" in query:
            item_time = item.creation_time
            query_start, query_end = query["temporal_range"]

            if not (query_start <= item_time <= query_end):
                return False

        # Content type criteria
        if "content_type" in query:
            item_content_type = item.content.get("type", "")
            if item_content_type != query["content_type"]:
                return False

        # Importance threshold
        if "min_importance" in query:
            if item.importance < query["min_importance"]:
                return False

        return True

    def _spatial_overlap(
        self, item_spatial: dict[str, Any], query_bounds: dict[str, Any]
    ) -> bool:
        """Check if item spatial context overlaps with query bounds.

        When both sides carry a ``bbox`` ([min_x, min_y, max_x, max_y]),
        real axis-aligned bbox intersection is required. The ``scale``
        label is only compared when geometry is absent from both sides;
        if exactly one side has geometry, there is no confirmed overlap.
        """
        item_bbox = _extract_bbox(item_spatial)
        query_bbox = _extract_bbox(query_bounds)

        if item_bbox is not None and query_bbox is not None:
            return _bboxes_intersect(item_bbox, query_bbox)

        if item_bbox is None and query_bbox is None:
            item_scale = item_spatial.get("scale", "medium")
            query_scale = query_bounds.get("scale", "medium")
            return bool(item_scale == query_scale)

        return False

    def get_memory_statistics(self) -> dict[str, Any]:
        """Get comprehensive memory system statistics."""
        stats: dict[str, Any] = {
            "memory_utilization": {},
            "access_patterns": {},
            "consolidation_status": {},
            "decay_analysis": {},
        }

        # Memory utilization by type
        for mem_type in self.memory_types:
            items = self.memory_storage.get(mem_type, {})
            capacity = self.memory_capacities[mem_type]
            utilization = len(items) / capacity if capacity > 0 else 0.0

            stats["memory_utilization"][mem_type] = {
                "used": len(items),
                "capacity": capacity,
                "utilization": utilization,
                "available": capacity - len(items),
            }

        # Access patterns
        total_accesses = sum(
            item.access_count
            for mem_type in self.memory_types
            for item in self.memory_storage.get(mem_type, {}).values()
        )

        if total_accesses > 0:
            stats["access_patterns"] = {
                "total_accesses": total_accesses,
                "average_accesses_per_item": total_accesses
                / sum(len(items) for items in self.memory_storage.values()),
            }

        # Consolidation status
        working_items = self.memory_storage.get("working", {})
        ready_for_consolidation = self.consolidation_system.check_for_consolidation(
            list(working_items.values())
        )

        stats["consolidation_status"] = {
            "working_memory_size": len(working_items),
            "ready_for_consolidation": len(ready_for_consolidation),
            "long_term_memory_size": len(self.memory_storage.get("long_term", {})),
        }

        # Decay analysis
        all_items = [
            item
            for mem_type in self.memory_types
            for item in self.memory_storage.get(mem_type, {}).values()
        ]

        if all_items:
            confidences = [item.confidence for item in all_items]
            importances = [item.importance for item in all_items]

            stats["decay_analysis"] = {
                "average_confidence": float(np.mean(confidences)),
                "average_importance": float(np.mean(importances)),
                "items_at_risk": len(
                    [item for item in all_items if item.confidence < 0.3]
                ),
            }

        return stats

    def export_memory_knowledge_graph(self) -> dict[str, Any]:
        """Export memory contents as a knowledge graph for analysis."""
        knowledge_graph: dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "memory_types": self.memory_types,
                "total_items": sum(
                    len(items) for items in self.memory_storage.values()
                ),
            },
        }

        # Create nodes for memory items
        for mem_type in self.memory_types:
            for item_id, item in self.memory_storage.get(mem_type, {}).items():
                node = {
                    "id": item_id,
                    "type": mem_type,
                    "content_type": item.content.get("type", "unknown"),
                    "importance": item.importance,
                    "confidence": item.confidence,
                    "access_count": item.access_count,
                    "creation_time": item.creation_time.isoformat(),
                }
                knowledge_graph["nodes"].append(node)

        # Create edges based on spatial relationships
        for mem_type in self.memory_types:
            for item in self.memory_storage.get(mem_type, {}).values():
                # Create edges between related items (simplified)
                content_type = item.content.get("type", "")

                if content_type == "spatial_element":
                    # Connect to related spatial reasoning items
                    for other_mem_type in self.memory_types:
                        for other_item in self.memory_storage.get(
                            other_mem_type, {}
                        ).values():
                            other_content_type = other_item.content.get("type", "")

                            if other_content_type == "spatial_reasoning":
                                edge = {
                                    "source": item.item_id,
                                    "target": other_item.item_id,
                                    "type": "spatial_relationship",
                                    "weight": min(
                                        item.importance, other_item.importance
                                    ),
                                }
                                knowledge_graph["edges"].append(edge)

        return knowledge_graph

    def update_model(
        self, training_data: dict[str, Any], learning_rate: float = 0.01
    ) -> dict[str, Any]:
        """Update memory model based on training data."""
        update_results = {
            "decay_rates_updated": False,
            "consolidation_parameters_updated": False,
            "performance_improvement": 0.0,
        }

        # Update decay rates based on retrieval performance
        if "retrieval_feedback" in training_data:
            feedback = training_data["retrieval_feedback"]

            if feedback.get("successful_retrieval_rate", 0) < 0.7:
                # Increase decay rates to simulate forgetting
                for mem_type in self.memory_types:
                    if mem_type in self.decay_parameters:
                        self.decay_parameters[mem_type]["rate"] *= 1.1
                update_results["decay_rates_updated"] = True

        # Update consolidation parameters
        if "consolidation_feedback" in training_data:
            feedback = training_data["consolidation_feedback"]

            if feedback.get("premature_consolidation_rate", 0) > 0.3:
                # Increase consolidation threshold to be more selective
                self.consolidation_system.consolidation_threshold = min(
                    0.9, self.consolidation_system.consolidation_threshold + 0.05
                )
                update_results["consolidation_parameters_updated"] = True

        return update_results

    def get_status(self) -> dict[str, Any]:
        """Get current status of the memory model."""
        return {
            "model_type": "spatial_memory",
            "status": "active",
            "memory_metrics": self.memory_metrics,
            "memory_utilization": {
                mem_type: len(self.memory_storage[mem_type])
                / self.memory_capacities[mem_type]
                for mem_type in self.memory_types
            },
            "consolidation_status": {
                "threshold": self.consolidation_system.consolidation_threshold,
                "delay": self.consolidation_system.consolidation_delay,
                "pending_items": len(self.consolidation_system.pending_consolidation),
            },
        }
