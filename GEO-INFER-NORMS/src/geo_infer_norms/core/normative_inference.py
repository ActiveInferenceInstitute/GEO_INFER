"""
Normative inference module for probabilistic reasoning about norms and regulations.

This module provides classes and methods for inferring, learning, and reasoning about
normative compliance using probabilistic approaches.
"""

from typing import (
    Any,
    cast,
)
from collections.abc import Callable
import datetime
import logging
import uuid
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class NormEvaluationError(RuntimeError):
    """Raised when a norm condition cannot be evaluated (evaluator fault).

    A raising condition callable is an evaluation failure, not a genuine
    compliance verdict; consumers must never fold it into non-compliance.
    """


class NormativeInference:
    """
    A class for probabilistic inference about norms and compliance.

    This class provides methods for inferring compliance with norms using
    probabilistic models, allowing for uncertainty in both norm definitions
    and observations.
    """

    def __init__(self) -> None:
        """Initialize a NormativeInference instance."""
        self.norms: dict[str, dict[str, Any]] = {}  # norm ID -> norm definitions
        self.observations: dict[
            str, dict[str, list[dict[str, Any]]]
        ] = {}  # entity ID -> behavior -> observations
        self.prior_beliefs: dict[tuple[str, str | None], float] = {}
        self.norm_relationships: dict[str, dict[str, Any]] = {}

    def add_norm(
        self,
        name: str,
        condition: Callable[[dict[str, Any]], bool],
        probability: float = 1.0,
        description: str = "",
        spatial_constraint: Polygon | None = None,
        temporal_constraint: tuple[datetime.datetime, datetime.datetime] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a norm to the inference engine.

        Args:
            name: Name of the norm
            condition: Function that takes observations and returns True if compliant
            probability: Prior probability of norm validity (0.0 to 1.0)
            description: Description of the norm
            spatial_constraint: Optional spatial constraint for the norm
            temporal_constraint: Optional temporal constraint (start_time, end_time)
            attributes: Optional additional attributes

        Returns:
            ID of the added norm
        """
        norm_id = str(uuid.uuid4())

        self.norms[norm_id] = {
            "id": norm_id,
            "name": name,
            "description": description,
            "condition": condition,
            "probability": probability,
            "spatial_constraint": spatial_constraint,
            "temporal_constraint": temporal_constraint,
            "attributes": attributes or {},
        }

        logger.info(f"Added norm: {name} with ID {norm_id}")
        return norm_id

    def add_norm_relationship(
        self,
        norm1_id: str,
        norm2_id: str,
        relationship_type: str,
        strength: float = 1.0,
    ) -> None:
        """
        Add a relationship between norms.

        Args:
            norm1_id: ID of the first norm
            norm2_id: ID of the second norm
            relationship_type: Type of relationship (e.g., 'supports', 'conflicts')
            strength: Strength of the relationship (0.0 to 1.0)
        """
        if norm1_id not in self.norms:
            raise KeyError(
                f"normative_inference.add_norm_relationship: norm ID {norm1_id!r} not found"
            )
        if norm2_id not in self.norms:
            raise KeyError(
                f"normative_inference.add_norm_relationship: norm ID {norm2_id!r} not found"
            )

        relationship_id = f"{norm1_id}_{relationship_type}_{norm2_id}"

        self.norm_relationships[relationship_id] = {
            "norm1_id": norm1_id,
            "norm2_id": norm2_id,
            "type": relationship_type,
            "strength": strength,
        }

        logger.info(
            f"Added {relationship_type} relationship between norms {norm1_id} and {norm2_id}"
        )

    def add_observation(
        self,
        entity_id: str,
        behavior: str,
        value: Any,
        timestamp: datetime.datetime | None = None,
        location: Point | None = None,
        certainty: float = 1.0,
    ) -> None:
        """
        Add an observation about an entity's behavior.

        Args:
            entity_id: ID of the entity
            behavior: Type of behavior observed
            value: Value of the behavior
            timestamp: Optional timestamp of the observation
            location: Optional location of the observation
            certainty: Certainty of the observation (0.0 to 1.0)
        """
        if entity_id not in self.observations:
            self.observations[entity_id] = {}

        if behavior not in self.observations[entity_id]:
            self.observations[entity_id][behavior] = []

        observation = {
            "value": value,
            "timestamp": timestamp or datetime.datetime.now(),
            "location": location,
            "certainty": certainty,
        }

        self.observations[entity_id][behavior].append(observation)
        logger.info(f"Added observation for entity {entity_id}: {behavior} = {value}")

    def set_prior_belief(
        self,
        norm_id: str,
        entity_id: str | None = None,
        compliance_probability: float = 0.5,
    ) -> None:
        """
        Set prior belief about norm compliance.

        Args:
            norm_id: ID of the norm
            entity_id: Optional entity ID (if None, sets for all entities)
            compliance_probability: Prior probability of compliance (0.0 to 1.0)
        """
        if norm_id not in self.norms:
            raise KeyError(
                f"normative_inference.set_prior_belief: norm ID {norm_id!r} not found"
            )

        key = (norm_id, entity_id)
        self.prior_beliefs[key] = compliance_probability

        if entity_id:
            logger.info(
                f"Set prior compliance belief for entity {entity_id}, norm {norm_id}: {compliance_probability}"
            )
        else:
            logger.info(
                f"Set prior compliance belief for all entities, norm {norm_id}: {compliance_probability}"
            )

    def get_latest_observation(
        self, entity_id: str, behavior: str
    ) -> dict[str, Any] | None:
        """
        Get the most recent observation for an entity's behavior.

        Args:
            entity_id: ID of the entity
            behavior: Type of behavior

        Returns:
            The most recent observation or None if not found
        """
        if (
            entity_id not in self.observations
            or behavior not in self.observations[entity_id]
        ):
            return None

        observations = self.observations[entity_id][behavior]
        if not observations:
            return None

        # Return the most recent observation
        return max(observations, key=lambda o: o["timestamp"])

    def get_entity_observations(self, entity_id: str) -> dict[str, Any]:
        """
        Get all latest observations for an entity.

        Args:
            entity_id: ID of the entity

        Returns:
            Dictionary mapping behaviors to their latest observed values
        """
        if entity_id not in self.observations:
            return {}

        result = {}
        for behavior, observations in self.observations[entity_id].items():
            if observations:
                latest = max(observations, key=lambda o: o["timestamp"])
                result[behavior] = latest["value"]

        return result

    def check_norm_compliance(
        self, norm_id: str, entity_id: str
    ) -> tuple[bool | str, float | None]:
        """
        Check if an entity complies with a norm.

        Args:
            norm_id: ID of the norm
            entity_id: ID of the entity

        Returns:
            Tuple of (compliant, certainty), or ('error', None) when the norm
            condition raises during evaluation. The ('error', None) state is an
            evaluation error and must never be confused with a genuine
            non-compliance verdict.
        """
        if norm_id not in self.norms:
            logger.warning(f"Cannot check compliance: norm {norm_id} not found")
            return False, 0.0

        norm = self.norms[norm_id]
        observations = self.get_entity_observations(entity_id)

        # If no observations, return uncertain compliance
        if not observations:
            return False, 0.0

        # Check spatial constraint if present
        if norm["spatial_constraint"] is not None:
            has_location = False
            within_constraint = False

            for obs_list in self.observations[entity_id].values():
                for obs in obs_list:
                    if obs["location"] is not None:
                        has_location = True
                        if norm["spatial_constraint"].contains(obs["location"]):
                            within_constraint = True
                            break

            if has_location and not within_constraint:
                return (
                    False,
                    1.0,
                )  # Certain non-compliance if outside spatial constraint

        # Check temporal constraint if present
        if norm["temporal_constraint"] is not None:
            start_time, end_time = norm["temporal_constraint"]
            has_timestamp = False
            within_timeframe = False

            for obs_list in self.observations[entity_id].values():
                for obs in obs_list:
                    if obs["timestamp"] is not None:
                        has_timestamp = True
                        if start_time <= obs["timestamp"] <= end_time:
                            within_timeframe = True
                            break

            if has_timestamp and not within_timeframe:
                return (
                    False,
                    1.0,
                )  # Certain non-compliance if outside temporal constraint

        # Check the norm condition
        try:
            compliant = norm["condition"](observations)
            # Determine certainty based on observation certainties
            certainty = min(
                [
                    obs[-1]["certainty"]
                    for obs in self.observations[entity_id].values()
                    if obs
                ]
            )
            return compliant, certainty
        except Exception as e:
            logger.error(f"Error checking norm compliance: {str(e)}")
            # An evaluation error is distinct from a genuine verdict: surface
            # ('error', None) rather than fabricating non-compliance.
            return "error", None

    def infer_compliance(
        self, entity_id: str, norm_id: str | None = None
    ) -> float | dict[str, float]:
        """
        Infer the probability of norm compliance for an entity.

        Args:
            entity_id: ID of the entity
            norm_id: Optional ID of the norm (if None, infers for all norms)

        Returns:
            If norm_id is provided: probability of compliance with that norm
            If norm_id is None: dictionary mapping norm IDs to compliance probabilities
        """
        if norm_id is not None:
            if norm_id not in self.norms:
                logger.warning(f"Cannot infer compliance: norm {norm_id} not found")
                return 0.0

            # Simple case: check compliance for a single norm
            verdict, certainty = self.check_norm_compliance(norm_id, entity_id)
            if verdict == "error":
                raise NormEvaluationError(
                    f"Norm '{norm_id}' condition raised during evaluation for "
                    f"entity '{entity_id}'; cannot infer a compliance probability"
                )

            # Get prior belief
            prior_key = (norm_id, entity_id)
            generic_prior_key = (norm_id, None)

            if prior_key in self.prior_beliefs:
                prior = self.prior_beliefs[prior_key]
            elif generic_prior_key in self.prior_beliefs:
                prior = self.prior_beliefs[generic_prior_key]
            else:
                prior = 0.5  # Default prior

            # Simple Bayesian update
            if verdict:
                # P(compliant|observation) = P(observation|compliant) * P(compliant) / P(observation)
                # With P(observation|compliant) = certainty
                probability = (certainty * prior) / (
                    (certainty * prior) + ((1 - certainty) * (1 - prior))
                )
            else:
                # P(non-compliant|observation) = P(observation|non-compliant) * P(non-compliant) / P(observation)
                probability = ((1 - certainty) * (1 - prior)) / (
                    ((1 - certainty) * (1 - prior)) + (certainty * prior)
                )
                probability = 1 - probability  # Convert to compliance probability

            return probability
        else:
            # Infer compliance for all norms
            result: dict[str, float] = {}
            for norm_id in self.norms:
                result[norm_id] = cast(float, self.infer_compliance(entity_id, norm_id))
            return result

    def infer_network_compliance(self, entity_id: str, norm_id: str) -> float:
        """
        Infer compliance using the network of norm relationships.

        This method takes into account how norms relate to each other when
        inferring compliance.

        Args:
            entity_id: ID of the entity
            norm_id: ID of the norm

        Returns:
            Probability of compliance considering norm relationships
        """
        if norm_id not in self.norms:
            logger.warning(f"Cannot infer network compliance: norm {norm_id} not found")
            return 0.0

        # Get direct compliance probability
        direct_probability = cast(float, self.infer_compliance(entity_id, norm_id))

        # Find related norms
        related_norms = {}
        for rel in self.norm_relationships.values():
            if rel["norm1_id"] == norm_id:
                related_norms[rel["norm2_id"]] = (rel["type"], rel["strength"])
            elif rel["norm2_id"] == norm_id:
                related_norms[rel["norm1_id"]] = (rel["type"], rel["strength"])

        if not related_norms:
            return direct_probability

        # Calculate influence from related norms
        influences = []
        for related_id, (rel_type, strength) in related_norms.items():
            related_prob = cast(float, self.infer_compliance(entity_id, related_id))

            if rel_type == "supports":
                # Supporting norms positively influence compliance
                influence = related_prob * strength
            elif rel_type == "conflicts":
                # Conflicting norms negatively influence compliance
                influence = (1 - related_prob) * strength
            else:
                # Unknown relationship type
                influence = 0.0

            influences.append(influence)

        # Combine direct probability with influences
        if influences:
            avg_influence = sum(influences) / len(influences)
            # Use a weighted average with direct probability having more weight
            final_probability = (direct_probability * 0.7) + (avg_influence * 0.3)
            # Ensure probability is in [0, 1]
            final_probability = max(0.0, min(1.0, final_probability))
            return cast(float, final_probability)
        else:
            return direct_probability

    def identify_norm_violations(
        self, entity_id: str, threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        """
        Identify potential norm violations for an entity.

        Args:
            entity_id: ID of the entity
            threshold: Compliance probability threshold for identifying violations

        Returns:
            List of potential violations with norm information and compliance probability
        """
        violations = []

        for norm_id, norm in self.norms.items():
            compliance_prob = cast(float, self.infer_compliance(entity_id, norm_id))

            if compliance_prob < threshold:
                violations.append(
                    {
                        "norm_id": norm_id,
                        "norm_name": norm["name"],
                        "compliance_probability": compliance_prob,
                        "description": norm["description"],
                        "severity": 1.0
                        - compliance_prob,  # Severity based on non-compliance probability
                    }
                )

        # Sort by severity (most severe first)
        violations.sort(key=lambda v: v["severity"], reverse=True)
        return violations

    def suggest_compliance_improvements(
        self, entity_id: str, improvement_threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Suggest improvements to increase compliance.

        Args:
            entity_id: ID of the entity
            improvement_threshold: Threshold for suggesting improvements

        Returns:
            List of suggested improvements with norm information and potential impact
        """
        suggestions = []

        # Get current observations
        observations = self.get_entity_observations(entity_id)

        # For each norm with low compliance, suggest improvements
        for norm_id, norm in self.norms.items():
            compliance_prob = cast(float, self.infer_compliance(entity_id, norm_id))

            if compliance_prob < improvement_threshold:
                # Simple heuristic: identify behaviors that might affect compliance
                relevant_behaviors = []

                if observations:
                    # Try changing each behavior to see if it improves compliance
                    for behavior, current_value in observations.items():
                        if isinstance(current_value, (int, float)):
                            # Try increasing and decreasing numeric values
                            test_observations = observations.copy()

                            # Test increase
                            test_observations[behavior] = (
                                current_value * 1.2
                            )  # 20% increase
                            try:
                                if norm["condition"](test_observations):
                                    relevant_behaviors.append(
                                        {
                                            "behavior": behavior,
                                            "current_value": current_value,
                                            "suggested_value": test_observations[
                                                behavior
                                            ],
                                            "change": "increase",
                                            "impact": "potentially increases compliance",
                                        }
                                    )
                                    continue
                            except Exception as exc:
                                logger.warning(
                                    "Norm condition raised during increase probe "
                                    "for behavior %s: %s",
                                    behavior,
                                    exc,
                                )

                            # Test decrease
                            test_observations[behavior] = (
                                current_value * 0.8
                            )  # 20% decrease
                            try:
                                if norm["condition"](test_observations):
                                    relevant_behaviors.append(
                                        {
                                            "behavior": behavior,
                                            "current_value": current_value,
                                            "suggested_value": test_observations[
                                                behavior
                                            ],
                                            "change": "decrease",
                                            "impact": "potentially increases compliance",
                                        }
                                    )
                            except Exception as exc:
                                logger.warning(
                                    "Norm condition raised during decrease probe "
                                    "for behavior %s: %s",
                                    behavior,
                                    exc,
                                )

                # Build a human-readable recommendation string from identified behaviors
                if relevant_behaviors:
                    behavior_hints = [
                        f"{b['change']} {b['behavior']} from {b['current_value']} to {b['suggested_value']:.4g}"
                        for b in relevant_behaviors
                    ]
                    recommendation = "To improve compliance: " + "; ".join(
                        behavior_hints
                    )
                else:
                    recommendation = (
                        f"Review norm '{norm['name']}' requirements and adjust "
                        f"relevant behaviors to meet compliance conditions"
                    )

                suggestions.append(
                    {
                        "norm_id": norm_id,
                        "norm_name": norm["name"],
                        "current_compliance": compliance_prob,
                        "recommendation": recommendation,
                        "relevant_behaviors": relevant_behaviors,
                    }
                )

        return suggestions


class SocialNormDiffusion:
    """
    A class for modeling the diffusion of social norms across populations.

    This class implements agent-based diffusion models to simulate how norms
    spread through social networks and spatial contexts.
    """

    def __init__(self) -> None:
        """Initialize a SocialNormDiffusion instance."""
        self.entities: dict[str, dict[str, Any]] = {}
        self.norms: dict[str, dict[str, Any]] = {}
        self.social_connections: dict[str, dict[str, float]] = {}
        self.spatial_locations: dict[str, Point] = {}
        self.adoption_state: dict[str, dict[str, bool]] = {}
        self.history: list[dict[str, Any]] = []

    def add_entity(
        self,
        entity_id: str,
        attributes: dict[str, Any],
        location: Point | None = None,
        adoption_threshold: float = 0.5,
    ) -> None:
        """
        Add an entity to the diffusion model.

        Args:
            entity_id: ID of the entity
            attributes: Entity attributes
            location: Optional spatial location
            adoption_threshold: Threshold for norm adoption (0.0 to 1.0)
        """
        self.entities[entity_id] = {
            "id": entity_id,
            "attributes": attributes,
            "adoption_threshold": adoption_threshold,
        }

        if location is not None:
            self.spatial_locations[entity_id] = location

    def add_norm(
        self,
        norm_id: str,
        name: str,
        initial_adopters: list[str] | None = None,
        spatial_factor: float = 0.5,
        network_factor: float = 0.5,
        content_factor: float = 0.0,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a norm to the diffusion model.

        Args:
            norm_id: ID of the norm
            name: Name of the norm
            initial_adopters: List of entity IDs who initially adopt the norm
            spatial_factor: Importance of spatial proximity (0.0 to 1.0)
            network_factor: Importance of social networks (0.0 to 1.0)
            content_factor: Importance of norm content (0.0 to 1.0)
            attributes: Norm attributes
        """
        self.norms[norm_id] = {
            "id": norm_id,
            "name": name,
            "spatial_factor": spatial_factor,
            "network_factor": network_factor,
            "content_factor": content_factor,
            "attributes": attributes or {},
        }

        # Initialize adoption state
        if norm_id not in self.adoption_state:
            self.adoption_state[norm_id] = {}

        initial_adopters = initial_adopters or []
        for entity_id in self.entities:
            self.adoption_state[norm_id][entity_id] = entity_id in initial_adopters

    def add_social_connection(
        self, entity1_id: str, entity2_id: str, strength: float = 1.0
    ) -> None:
        """
        Add a social connection between entities.

        Args:
            entity1_id: ID of the first entity
            entity2_id: ID of the second entity
            strength: Strength of the connection (0.0 to 1.0)
        """
        if entity1_id not in self.entities or entity2_id not in self.entities:
            logger.warning("Cannot add connection: one or both entities not found")
            return

        if entity1_id not in self.social_connections:
            self.social_connections[entity1_id] = {}

        if entity2_id not in self.social_connections:
            self.social_connections[entity2_id] = {}

        self.social_connections[entity1_id][entity2_id] = strength
        self.social_connections[entity2_id][entity1_id] = strength

    def calculate_adoption_probability(self, norm_id: str, entity_id: str) -> float:
        """
        Calculate the probability of an entity adopting a norm.

        Args:
            norm_id: ID of the norm
            entity_id: ID of the entity

        Returns:
            Probability of adoption (0.0 to 1.0)
        """
        if norm_id not in self.norms or entity_id not in self.entities:
            return 0.0

        norm = self.norms[norm_id]

        # Skip if already adopted
        if self.adoption_state[norm_id].get(entity_id, False):
            return 1.0

        # Social network influence
        network_influence = 0.0
        if norm["network_factor"] > 0 and entity_id in self.social_connections:
            connections = self.social_connections[entity_id]
            if connections:
                adopted_connections = sum(
                    strength
                    for neighbor, strength in connections.items()
                    if self.adoption_state[norm_id].get(neighbor, False)
                )
                total_strength = sum(connections.values())
                if total_strength > 0:
                    network_influence = adopted_connections / total_strength

        # Spatial influence
        spatial_influence = 0.0
        if norm["spatial_factor"] > 0 and entity_id in self.spatial_locations:
            entity_location = self.spatial_locations[entity_id]
            adopted_entities = [
                e_id
                for e_id, adopted in self.adoption_state[norm_id].items()
                if adopted and e_id in self.spatial_locations
            ]

            if adopted_entities:
                # Calculate average distance to adopted entities
                distances = []
                for adopted_id in adopted_entities:
                    adopted_location = self.spatial_locations[adopted_id]
                    distance = entity_location.distance(adopted_location)
                    distances.append(distance)

                if distances:
                    # Normalize distances
                    max_distance = max(distances)
                    if max_distance > 0:
                        normalized_distances = [d / max_distance for d in distances]
                        avg_normalized_distance = sum(normalized_distances) / len(
                            normalized_distances
                        )
                        spatial_influence = 1 - avg_normalized_distance

        # Content influence: Jaccard similarity between norm and entity attributes
        content_influence = 0.0
        if norm["content_factor"] > 0:
            norm_attrs = set(norm.get("attributes", {}).keys())
            entity_attrs = set(self.entities[entity_id].get("attributes", {}).keys())
            if norm_attrs or entity_attrs:
                intersection = norm_attrs & entity_attrs
                union = norm_attrs | entity_attrs
                content_influence = len(intersection) / len(union) if union else 0.0

        # Combine influences
        total_influence = (
            norm["network_factor"] * network_influence
            + norm["spatial_factor"] * spatial_influence
            + norm["content_factor"] * content_influence
        )

        # Normalize
        normalization_factor = (
            norm["network_factor"] + norm["spatial_factor"] + norm["content_factor"]
        )
        if normalization_factor > 0:
            total_influence /= normalization_factor

        return cast(float, total_influence)

    def simulate_step(self) -> dict[str, Any]:
        """
        Simulate one step of norm diffusion.

        Returns:
            A dictionary with changes made in this step
        """
        step_changes: dict[str, Any] = {
            "time_step": len(self.history) + 1,
            "norm_changes": {},
        }

        for norm_id in self.norms:
            norm_changes = []

            # Calculate adoption probabilities for each entity
            for entity_id in self.entities:
                if not self.adoption_state[norm_id].get(entity_id, False):
                    # Only consider non-adopters
                    adoption_prob = self.calculate_adoption_probability(
                        norm_id, entity_id
                    )
                    entity_threshold = self.entities[entity_id]["adoption_threshold"]

                    if adoption_prob >= entity_threshold:
                        # Entity adopts the norm
                        self.adoption_state[norm_id][entity_id] = True
                        norm_changes.append(entity_id)

            step_changes["norm_changes"][norm_id] = norm_changes

        self.history.append(step_changes)
        return step_changes

    def simulate(self, steps: int) -> list[dict[str, Any]]:
        """
        Simulate multiple steps of norm diffusion.

        Args:
            steps: Number of steps to simulate

        Returns:
            List of dictionaries with changes at each step
        """
        results = []

        for _ in range(steps):
            step_result = self.simulate_step()
            results.append(step_result)

            # Check if diffusion is complete (no changes)
            if all(
                len(changes) == 0 for changes in step_result["norm_changes"].values()
            ):
                break

        return results

    def get_adoption_summary(self) -> dict[str, dict[str, float]]:
        """
        Get a summary of norm adoption.

        Returns:
            Dictionary mapping norm IDs to adoption statistics
        """
        summary = {}

        for norm_id in self.norms:
            if norm_id in self.adoption_state:
                adopted_count = sum(
                    1 for adopted in self.adoption_state[norm_id].values() if adopted
                )
                total_count = len(self.adoption_state[norm_id])

                summary[norm_id] = {
                    "adopted_count": adopted_count,
                    "total_count": total_count,
                    "adoption_rate": (
                        adopted_count / total_count if total_count > 0 else 0
                    ),
                }

        return summary

    def get_adoption_history(self) -> dict[str, list[float]]:
        """
        Get the history of adoption rates over time.

        Returns:
            Dictionary mapping norm IDs to lists of adoption rates
        """
        history: dict[str, list[float]] = {norm_id: [] for norm_id in self.norms}

        # Calculate initial state
        for norm_id in self.norms:
            adopted_count = sum(
                1 for adopted in self.adoption_state[norm_id].values() if adopted
            )
            total_count = len(self.adoption_state[norm_id])
            adoption_rate = adopted_count / total_count if total_count > 0 else 0
            history[norm_id].append(adoption_rate)

        # Add rates from history
        current_adopted = {
            norm_id: sum(
                1 for adopted in self.adoption_state[norm_id].values() if adopted
            )
            for norm_id in self.norms
        }

        for step in self.history:
            for norm_id, changes in step["norm_changes"].items():
                current_adopted[norm_id] += len(changes)
                total_count = len(self.adoption_state[norm_id])
                adoption_rate = (
                    current_adopted[norm_id] / total_count if total_count > 0 else 0
                )
                history[norm_id].append(adoption_rate)

        return history
