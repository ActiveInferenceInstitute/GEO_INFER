"""Tests for civic participation modeling."""

import pytest
from geo_infer_civ.core.participation import (
    ParticipationAnalyzer,
    ParticipationMethod,
    ParticipantRecord,
)


@pytest.fixture
def analyzer():
    return ParticipationAnalyzer()


@pytest.fixture
def sample_records():
    """Generate a set of participation records across methods and demographics."""
    records = []
    methods = list(ParticipationMethod)
    demographics = ["youth", "seniors", "families", "professionals"]
    for i in range(40):
        records.append(
            ParticipantRecord(
                participant_id=f"p_{i}",
                method=methods[i % len(methods)],
                timestamp=1000.0 + i * 100,
                demographic_group=demographics[i % len(demographics)],
                sentiment_score=0.5 + (i % 5) * 0.1,
            )
        )
    return records


class TestParticipationAnalyzer:
    def test_empty_engagement_score(self, analyzer):
        score = analyzer.compute_engagement_score(target_population=100)
        assert score.overall_score == 0.0
        assert score.reach_ratio == 0.0

    def test_engagement_score_positive(self, analyzer, sample_records):
        analyzer.add_records(sample_records)
        score = analyzer.compute_engagement_score(target_population=100)
        assert 0.0 < score.overall_score <= 1.0
        assert score.reach_ratio > 0.0
        assert score.diversity_index > 0.0
        assert len(score.method_scores) > 1

    def test_engagement_score_invalid_population(self, analyzer):
        with pytest.raises(ValueError, match="positive"):
            analyzer.compute_engagement_score(target_population=0)

    def test_reach_ratio_capped_at_one(self, analyzer):
        for i in range(200):
            analyzer.add_record(
                ParticipantRecord(
                    participant_id=f"p_{i}",
                    method=ParticipationMethod.SURVEY,
                    timestamp=1000.0,
                )
            )
        score = analyzer.compute_engagement_score(target_population=50)
        assert score.reach_ratio <= 1.0

    def test_participation_index_baseline(self, analyzer, sample_records):
        analyzer.add_records(sample_records)
        index = analyzer.compute_participation_index(
            target_population=100, baseline_rate=0.40
        )
        assert index == 1.0  # 40 unique participants / 100 pop / 0.40 baseline = 1.0

    def test_participation_index_above_baseline(self, analyzer, sample_records):
        analyzer.add_records(sample_records)
        index = analyzer.compute_participation_index(
            target_population=100, baseline_rate=0.10
        )
        assert index > 1.0

    def test_participation_index_invalid_baseline(self, analyzer):
        with pytest.raises(ValueError, match="baseline_rate"):
            analyzer.compute_participation_index(
                target_population=100, baseline_rate=0.0
            )

    def test_representation_analysis(self, analyzer, sample_records):
        analyzer.add_records(sample_records)
        report = analyzer.analyze_representation(
            population_demographics={
                "youth": 0.25,
                "seniors": 0.25,
                "families": 0.25,
                "professionals": 0.25,
            }
        )
        assert report.overall_representation_score > 0.0
        assert len(report.representation_indices) == 4
        for ri in report.representation_indices.values():
            assert ri > 0.0

    def test_representation_underrepresented(self, analyzer):
        # All participants from one group
        for i in range(20):
            analyzer.add_record(
                ParticipantRecord(
                    participant_id=f"p_{i}",
                    method=ParticipationMethod.SURVEY,
                    timestamp=1000.0,
                    demographic_group="youth",
                )
            )
        report = analyzer.analyze_representation(
            population_demographics={"youth": 0.5, "seniors": 0.5}
        )
        assert "seniors" in report.underrepresented_groups

    def test_representation_empty_demographics_error(self, analyzer):
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze_representation({})

    def test_summary(self, analyzer, sample_records):
        analyzer.add_records(sample_records)
        summary = analyzer.get_participation_summary()
        assert summary["total_records"] == 40
        assert summary["unique_participants"] == 40
        assert summary["average_sentiment"] is not None
        assert len(summary["method_counts"]) > 1

    def test_clear_records(self, analyzer, sample_records):
        analyzer.add_records(sample_records)
        analyzer.clear_records()
        assert analyzer.get_participation_summary()["total_records"] == 0

    def test_time_window_filtering(self, analyzer, sample_records):
        analyzer.add_records(sample_records)
        # Only records in first half of time range
        score = analyzer.compute_engagement_score(
            target_population=100,
            time_window=(1000.0, 2000.0),
        )
        assert score.overall_score > 0.0
        assert score.reach_ratio < 0.4  # Less than all 40 participants
