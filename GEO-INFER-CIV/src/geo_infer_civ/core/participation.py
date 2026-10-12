"""
Civic participation modeling for GEO-INFER-CIV.

Provides engagement scoring, participation index calculation,
and demographic representation analysis for civic processes.
"""

import math
from typing import Any
from dataclasses import dataclass
from enum import Enum


class ParticipationMethod(Enum):
    """Methods through which citizens can participate."""

    SURVEY = "survey"
    PUBLIC_COMMENT = "public_comment"
    TOWN_HALL = "town_hall"
    WORKSHOP = "workshop"
    ONLINE_FORUM = "online_forum"
    BALLOT = "ballot"
    PETITION = "petition"
    MAP_ANNOTATION = "map_annotation"


@dataclass
class ParticipantRecord:
    """Record of a single participant's engagement."""

    participant_id: str
    method: ParticipationMethod
    timestamp: float
    demographic_group: str | None = None
    location: tuple[float, float] | None = None
    sentiment_score: float | None = None
    weight: float = 1.0


@dataclass
class EngagementScore:
    """Computed engagement score with breakdown."""

    overall_score: float
    method_scores: dict[str, float]
    temporal_consistency: float
    diversity_index: float
    reach_ratio: float


@dataclass
class RepresentationReport:
    """Report on demographic representation in participation."""

    group_proportions: dict[str, float]
    population_proportions: dict[str, float]
    representation_indices: dict[str, float]
    overall_representation_score: float
    underrepresented_groups: list[str]
    overrepresented_groups: list[str]


class ParticipationAnalyzer:
    """
    Analyzes civic participation patterns and computes engagement metrics.

    Provides methods for scoring engagement, calculating participation
    indices, and analyzing demographic representation in civic processes.
    """

    def __init__(
        self, method_weights: dict[ParticipationMethod, float] | None = None
    ) -> None:
        """
        Initialize the participation analyzer.

        Args:
            method_weights: Optional mapping of participation methods to
                their relative importance weights. Defaults to equal weighting.
        """
        self._method_weights = method_weights or {m: 1.0 for m in ParticipationMethod}
        self._records: list[ParticipantRecord] = []

    def add_record(self, record: ParticipantRecord) -> None:
        """
        Add a participation record to the analyzer.

        Args:
            record: A participant's engagement record.
        """
        self._records.append(record)

    def add_records(self, records: list[ParticipantRecord]) -> None:
        """
        Add multiple participation records.

        Args:
            records: List of participant engagement records.
        """
        self._records.extend(records)

    def clear_records(self) -> None:
        """Remove all stored participation records."""
        self._records.clear()

    def compute_engagement_score(
        self,
        target_population: int,
        time_window: tuple[float, float] | None = None,
    ) -> EngagementScore:
        """
        Compute an overall engagement score for the participation data.

        The score is composed of:
        - Method-weighted participation rates
        - Temporal consistency (how evenly participation is distributed over time)
        - Method diversity (Shannon entropy of methods used)
        - Reach ratio (unique participants / target population)

        Args:
            target_population: Total population that could participate.
            time_window: Optional (start, end) timestamps to filter records.

        Returns:
            EngagementScore with overall and component scores.

        Raises:
            ValueError: If target_population is not positive.
        """
        if target_population <= 0:
            raise ValueError("target_population must be positive")

        records = self._filter_by_time(time_window)
        if not records:
            return EngagementScore(
                overall_score=0.0,
                method_scores={},
                temporal_consistency=0.0,
                diversity_index=0.0,
                reach_ratio=0.0,
            )

        method_scores = self._compute_method_scores(records)
        temporal_consistency = self._compute_temporal_consistency(records, time_window)
        diversity_index = self._compute_method_diversity(records)
        unique_participants = len({r.participant_id for r in records})
        reach_ratio = min(unique_participants / target_population, 1.0)

        weighted_method_avg = 0.0
        total_weight = 0.0
        for method_name, score in method_scores.items():
            method_enum = ParticipationMethod(method_name)
            w = self._method_weights.get(method_enum, 1.0)
            weighted_method_avg += score * w
            total_weight += w
        if total_weight > 0:
            weighted_method_avg /= total_weight

        overall = (
            0.35 * weighted_method_avg
            + 0.20 * temporal_consistency
            + 0.20 * diversity_index
            + 0.25 * reach_ratio
        )

        return EngagementScore(
            overall_score=round(overall, 4),
            method_scores=method_scores,
            temporal_consistency=round(temporal_consistency, 4),
            diversity_index=round(diversity_index, 4),
            reach_ratio=round(reach_ratio, 4),
        )

    def compute_participation_index(
        self,
        target_population: int,
        baseline_rate: float = 0.10,
    ) -> float:
        """
        Compute a normalized participation index relative to a baseline.

        The index is 1.0 when participation matches the baseline rate,
        greater than 1.0 when exceeding it, and less than 1.0 when below.

        Args:
            target_population: Total population that could participate.
            baseline_rate: Expected baseline participation rate (0-1).

        Returns:
            Participation index as a float.

        Raises:
            ValueError: If baseline_rate is not in (0, 1] or target_population <= 0.
        """
        if target_population <= 0:
            raise ValueError("target_population must be positive")
        if not (0.0 < baseline_rate <= 1.0):
            raise ValueError("baseline_rate must be in (0, 1]")

        unique_participants = len({r.participant_id for r in self._records})
        actual_rate = unique_participants / target_population
        index = actual_rate / baseline_rate
        return round(index, 4)

    def analyze_representation(
        self,
        population_demographics: dict[str, float],
    ) -> RepresentationReport:
        """
        Analyze how well participation reflects the target population demographics.

        Uses the representation ratio for each group (participation proportion /
        population proportion). A ratio of 1.0 means perfect representation.

        Args:
            population_demographics: Mapping of demographic group names to their
                proportion in the total population. Values should sum to 1.0.

        Returns:
            RepresentationReport with per-group and overall metrics.

        Raises:
            ValueError: If population_demographics is empty.
        """
        if not population_demographics:
            raise ValueError("population_demographics must not be empty")

        records_with_demographics = [
            r for r in self._records if r.demographic_group is not None
        ]
        total_with_demo = len(records_with_demographics)

        group_counts: dict[str, int] = {}
        for r in records_with_demographics:
            if r.demographic_group is not None:
                group_counts[r.demographic_group] = (
                    group_counts.get(r.demographic_group, 0) + 1
                )

        group_proportions: dict[str, float] = {}
        if total_with_demo > 0:
            for group, count in group_counts.items():
                group_proportions[group] = round(count / total_with_demo, 4)

        representation_indices: dict[str, float] = {}
        for group, pop_prop in population_demographics.items():
            if pop_prop > 0:
                part_prop = group_proportions.get(group, 0.0)
                representation_indices[group] = round(part_prop / pop_prop, 4)
            else:
                representation_indices[group] = 0.0

        underrepresented = [g for g, ri in representation_indices.items() if ri < 0.8]
        overrepresented = [g for g, ri in representation_indices.items() if ri > 1.2]

        if representation_indices:
            vals = list(representation_indices.values())
            mean_ri = sum(vals) / len(vals)
            variance = sum((v - mean_ri) ** 2 for v in vals) / len(vals)
            overall_score = max(0.0, 1.0 - math.sqrt(variance))
        else:
            overall_score = 0.0

        return RepresentationReport(
            group_proportions=group_proportions,
            population_proportions=population_demographics,
            representation_indices=representation_indices,
            overall_representation_score=round(overall_score, 4),
            underrepresented_groups=underrepresented,
            overrepresented_groups=overrepresented,
        )

    def get_participation_summary(self) -> dict[str, Any]:
        """
        Return a summary of all participation data.

        Returns:
            Dictionary with counts by method, unique participants,
            total records, and average sentiment.
        """
        method_counts: dict[str, int] = {}
        sentiments: list[float] = []
        for r in self._records:
            method_counts[r.method.value] = method_counts.get(r.method.value, 0) + 1
            if r.sentiment_score is not None:
                sentiments.append(r.sentiment_score)

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None
        unique_ids = len({r.participant_id for r in self._records})

        return {
            "total_records": len(self._records),
            "unique_participants": unique_ids,
            "method_counts": method_counts,
            "average_sentiment": round(avg_sentiment, 4)
            if avg_sentiment is not None
            else None,
        }

    # ---- private helpers ----

    def _filter_by_time(
        self, time_window: tuple[float, float] | None
    ) -> list[ParticipantRecord]:
        if time_window is None:
            return list(self._records)
        start, end = time_window
        return [r for r in self._records if start <= r.timestamp <= end]

    def _compute_method_scores(
        self, records: list[ParticipantRecord]
    ) -> dict[str, float]:
        method_counts: dict[str, int] = {}
        for r in records:
            method_counts[r.method.value] = method_counts.get(r.method.value, 0) + 1
        total = len(records)
        return {m: round(c / total, 4) for m, c in method_counts.items()}

    def _compute_temporal_consistency(
        self,
        records: list[ParticipantRecord],
        time_window: tuple[float, float] | None,
    ) -> float:
        """Measure how evenly distributed participation is over time."""
        if len(records) < 2:
            return 1.0

        timestamps = sorted(r.timestamp for r in records)
        if time_window:
            start, end = time_window
        else:
            start, end = timestamps[0], timestamps[-1]

        span = end - start
        if span <= 0:
            return 1.0

        num_bins = min(10, len(records))
        bin_size = span / num_bins
        bins = [0] * num_bins
        for t in timestamps:
            idx = min(int((t - start) / bin_size), num_bins - 1)
            bins[idx] += 1

        total = sum(bins)
        if total == 0:
            return 0.0

        expected = total / num_bins
        chi_sq = sum((b - expected) ** 2 / expected for b in bins)
        max_chi_sq = (num_bins - 1) * total / expected if expected > 0 else 1.0
        consistency = max(0.0, 1.0 - chi_sq / max_chi_sq) if max_chi_sq > 0 else 1.0
        return consistency

    def _compute_method_diversity(self, records: list[ParticipantRecord]) -> float:
        """Compute Shannon entropy of participation methods, normalized to [0, 1]."""
        method_counts: dict[str, int] = {}
        for r in records:
            method_counts[r.method.value] = method_counts.get(r.method.value, 0) + 1

        total = len(records)
        if total == 0 or len(method_counts) <= 1:
            return 0.0

        entropy = 0.0
        for count in method_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(method_counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
