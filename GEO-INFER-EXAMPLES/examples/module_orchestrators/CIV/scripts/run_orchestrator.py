#!/usr/bin/env python3
"""GEO-INFER-CIV module orchestrator.

Runs one documented end-to-end CIV operation on synthetic data: score civic
engagement and demographic representation for a synthetic participatory-mapping
program, then track synthetic meeting attendance and analyze public comments.
All work goes through the real ``geo_infer_civ`` public API.
"""

from __future__ import annotations

import random
import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    from geo_infer_civ import (
        AttendanceTracker,
        CommentCategory,
        MeetingRecord,
        MeetingType,
        ParticipantRecord,
        ParticipationAnalyzer,
        ParticipationMethod,
        PublicComment,
        PublicCommentAnalyzer,
    )

    rng = random.Random(42)

    # Synthetic participation records for a fictional coastal county program.
    methods = list(ParticipationMethod)
    groups = ["north_town", "harbor_district", "river_valley", "unincorporated"]
    records = [
        ParticipantRecord(
            participant_id=f"resident-{i:03d}",
            method=methods[i % len(methods)],
            timestamp=1_700_000_000.0 + i * 86_400.0,
            demographic_group=groups[i % len(groups)],
            location=(-124.20 + rng.random() * 0.4, 41.74 + rng.random() * 0.5),
            sentiment_score=rng.uniform(-1.0, 1.0),
        )
        for i in range(90)
    ]

    analyzer = ParticipationAnalyzer()
    analyzer.add_records(records)
    engagement = analyzer.compute_engagement_score(target_population=500)
    representation = analyzer.analyze_representation(
        {
            "north_town": 0.40,
            "harbor_district": 0.25,
            "river_valley": 0.20,
            "unincorporated": 0.15,
        }
    )
    participation_index = analyzer.compute_participation_index(
        target_population=500, baseline_rate=0.10
    )

    # Synthetic meeting attendance across one season.
    tracker = AttendanceTracker()
    meetings = [
        MeetingRecord(
            meeting_id=f"council-{i:02d}",
            meeting_type=MeetingType.CITY_COUNCIL
            if i % 2 == 0
            else MeetingType.PLANNING_COMMISSION,
            date=1_700_000_000.0 + i * 1_209_600.0,
            registered_attendees=40 + 3 * i,
            actual_attendees=30 + 4 * i,
            public_comments_count=2 + i,
            duration_minutes=90.0,
        )
        for i in range(6)
    ]
    tracker.add_meetings(meetings)
    trend = tracker.compute_attendance_trend()

    # Synthetic public comments on a single shoreline-plan meeting.
    comment_analyzer = PublicCommentAnalyzer()
    categories = list(CommentCategory)
    comments = [
        PublicComment(
            comment_id=f"comment-{i:03d}",
            meeting_id="council-00",
            category=categories[i % len(categories)],
            word_count=40 + rng.randrange(220),
            timestamp=1_700_000_000.0 + i * 3_600.0,
            submitter_id=f"resident-{i % 30:03d}",
            sentiment_score=rng.uniform(-1.0, 1.0),
        )
        for i in range(24)
    ]
    comment_analyzer.add_comments(comments)
    comment_analysis = comment_analyzer.analyze(meeting_id="council-00")

    return {
        "operation": "civic_engagement_scoring",
        "participation_records": len(records),
        "engagement_score": {
            "overall_score": engagement.overall_score,
            "method_scores": engagement.method_scores,
            "temporal_consistency": engagement.temporal_consistency,
            "diversity_index": engagement.diversity_index,
            "reach_ratio": engagement.reach_ratio,
        },
        "participation_index": participation_index,
        "representation": {
            "overall_representation_score": representation.overall_representation_score,
            "underrepresented_groups": representation.underrepresented_groups,
            "overrepresented_groups": representation.overrepresented_groups,
            "representation_indices": representation.representation_indices,
        },
        "attendance_trend": {
            "average_attendance": trend.average_attendance,
            "attendance_rate": trend.attendance_rate,
            "trend_direction": trend.trend_direction,
            "trend_slope": trend.trend_slope,
            "meeting_count": trend.meeting_count,
        },
        "comment_analysis": {
            "total_comments": comment_analysis.total_comments,
            "category_distribution": comment_analysis.category_distribution,
            "average_word_count": comment_analysis.average_word_count,
            "average_sentiment": comment_analysis.average_sentiment,
            "engagement_depth_score": comment_analysis.engagement_depth_score,
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("CIV", _operation))
