"""
Community engagement metrics for GEO-INFER-CIV.

Provides meeting attendance tracking, public comment analysis,
and voter turnout modeling for civic engagement processes.
"""

import math
from typing import Any
from dataclasses import dataclass
from enum import Enum


class MeetingType(Enum):
    """Types of civic meetings."""

    CITY_COUNCIL = "city_council"
    PLANNING_COMMISSION = "planning_commission"
    PUBLIC_HEARING = "public_hearing"
    TOWN_HALL = "town_hall"
    WORKSHOP = "workshop"
    COMMUNITY_FORUM = "community_forum"
    BUDGET_HEARING = "budget_hearing"


class CommentCategory(Enum):
    """Categories for public comments."""

    SUPPORT = "support"
    OPPOSITION = "opposition"
    QUESTION = "question"
    SUGGESTION = "suggestion"
    CONCERN = "concern"
    NEUTRAL = "neutral"


@dataclass
class MeetingRecord:
    """Record of attendance at a civic meeting."""

    meeting_id: str
    meeting_type: MeetingType
    date: float
    registered_attendees: int
    actual_attendees: int
    public_comments_count: int = 0
    duration_minutes: float = 60.0
    location: tuple[float, float] | None = None
    topic: str | None = None


@dataclass
class PublicComment:
    """A single public comment record."""

    comment_id: str
    meeting_id: str
    category: CommentCategory
    word_count: int
    timestamp: float
    submitter_id: str | None = None
    topic: str | None = None
    sentiment_score: float = 0.0


@dataclass
class AttendanceTrend:
    """Attendance trend analysis result."""

    average_attendance: float
    attendance_rate: float
    trend_direction: str
    trend_slope: float
    peak_meeting_type: str
    lowest_meeting_type: str
    meeting_count: int


@dataclass
class CommentAnalysis:
    """Analysis of public comments."""

    total_comments: int
    category_distribution: dict[str, float]
    average_word_count: float
    average_sentiment: float
    unique_submitters: int
    topics: dict[str, int]
    engagement_depth_score: float


class AttendanceTracker:
    """
    Tracks and analyzes attendance patterns for civic meetings.

    Computes attendance rates, trends over time, and identifies
    patterns across different meeting types and topics.
    """

    def __init__(self) -> None:
        self._meetings: list[MeetingRecord] = []

    def add_meeting(self, meeting: MeetingRecord) -> None:
        """
        Add a meeting record.

        Args:
            meeting: A civic meeting attendance record.
        """
        self._meetings.append(meeting)

    def add_meetings(self, meetings: list[MeetingRecord]) -> None:
        """
        Add multiple meeting records.

        Args:
            meetings: List of civic meeting records.
        """
        self._meetings.extend(meetings)

    def compute_attendance_trend(
        self,
        meeting_type: MeetingType | None = None,
    ) -> AttendanceTrend:
        """
        Compute attendance trend analysis.

        Uses linear regression on attendance numbers over time to
        determine the trend direction and slope.

        Args:
            meeting_type: Optional filter for a specific meeting type.

        Returns:
            AttendanceTrend with computed metrics.

        Raises:
            ValueError: If no meeting records are available.
        """
        meetings = self._meetings
        if meeting_type:
            meetings = [m for m in meetings if m.meeting_type == meeting_type]

        if not meetings:
            raise ValueError("No meeting records available for analysis")

        sorted_meetings = sorted(meetings, key=lambda m: m.date)

        total_actual = sum(m.actual_attendees for m in sorted_meetings)
        total_registered = sum(m.registered_attendees for m in sorted_meetings)
        avg_attendance = total_actual / len(sorted_meetings)
        attendance_rate = (
            total_actual / total_registered if total_registered > 0 else 0.0
        )

        # Linear regression for trend
        n = len(sorted_meetings)
        if n >= 2:
            x_vals = list(range(n))
            y_vals = [m.actual_attendees for m in sorted_meetings]
            x_mean = sum(x_vals) / n
            y_mean = sum(y_vals) / n

            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
            denominator = sum((x - x_mean) ** 2 for x in x_vals)
            slope = numerator / denominator if denominator != 0 else 0.0
        else:
            slope = 0.0

        if slope > 0.5:
            direction = "increasing"
        elif slope < -0.5:
            direction = "decreasing"
        else:
            direction = "stable"

        # Per-type averages
        type_averages: dict[str, float] = {}
        type_counts: dict[str, tuple[int, int]] = {}
        for m in meetings:
            t = m.meeting_type.value
            if t not in type_counts:
                type_counts[t] = (0, 0)
            current = type_counts[t]
            type_counts[t] = (current[0] + m.actual_attendees, current[1] + 1)

        for t, (total, count) in type_counts.items():
            type_averages[t] = total / count

        if type_averages:
            peak_type = max(type_averages, key=lambda k: type_averages[k])
            lowest_type = min(type_averages, key=lambda k: type_averages[k])
        else:
            peak_type = "none"
            lowest_type = "none"

        return AttendanceTrend(
            average_attendance=round(avg_attendance, 2),
            attendance_rate=round(attendance_rate, 4),
            trend_direction=direction,
            trend_slope=round(slope, 4),
            peak_meeting_type=peak_type,
            lowest_meeting_type=lowest_type,
            meeting_count=len(meetings),
        )

    def get_meeting_effectiveness(self, meeting_id: str) -> dict[str, Any]:
        """
        Compute effectiveness metrics for a specific meeting.

        Effectiveness considers attendance rate, duration, and
        public comment engagement.

        Args:
            meeting_id: The meeting identifier.

        Returns:
            Dictionary with effectiveness metrics.

        Raises:
            ValueError: If the meeting is not found.
        """
        meeting = None
        for m in self._meetings:
            if m.meeting_id == meeting_id:
                meeting = m
                break

        if meeting is None:
            raise ValueError(f"Meeting {meeting_id} not found")

        attendance_rate = (
            meeting.actual_attendees / meeting.registered_attendees
            if meeting.registered_attendees > 0
            else 0.0
        )

        comment_rate = (
            meeting.public_comments_count / meeting.actual_attendees
            if meeting.actual_attendees > 0
            else 0.0
        )

        # Effectiveness: weighted combination of attendance rate and comment engagement
        effectiveness = 0.6 * attendance_rate + 0.4 * min(comment_rate, 1.0)

        return {
            "meeting_id": meeting.meeting_id,
            "meeting_type": meeting.meeting_type.value,
            "attendance_rate": round(attendance_rate, 4),
            "comment_rate": round(comment_rate, 4),
            "effectiveness_score": round(effectiveness, 4),
            "actual_attendees": meeting.actual_attendees,
            "registered_attendees": meeting.registered_attendees,
        }


class PublicCommentAnalyzer:
    """
    Analyzes public comments submitted to civic processes.

    Computes distribution of comment categories, sentiment analysis,
    and engagement depth metrics.
    """

    def __init__(self) -> None:
        self._comments: list[PublicComment] = []

    def add_comment(self, comment: PublicComment) -> None:
        """
        Add a public comment record.

        Args:
            comment: A public comment record.
        """
        self._comments.append(comment)

    def add_comments(self, comments: list[PublicComment]) -> None:
        """
        Add multiple public comment records.

        Args:
            comments: List of public comment records.
        """
        self._comments.extend(comments)

    def analyze(
        self,
        meeting_id: str | None = None,
    ) -> CommentAnalysis:
        """
        Analyze public comments, optionally filtered by meeting.

        Engagement depth score is computed from the diversity of
        categories, average comment length, and unique submitter ratio.

        Args:
            meeting_id: Optional filter to analyze comments for a specific meeting.

        Returns:
            CommentAnalysis with computed metrics.
        """
        comments = self._comments
        if meeting_id:
            comments = [c for c in comments if c.meeting_id == meeting_id]

        if not comments:
            return CommentAnalysis(
                total_comments=0,
                category_distribution={},
                average_word_count=0.0,
                average_sentiment=0.0,
                unique_submitters=0,
                topics={},
                engagement_depth_score=0.0,
            )

        # Category distribution
        cat_counts: dict[str, int] = {}
        for c in comments:
            cat_counts[c.category.value] = cat_counts.get(c.category.value, 0) + 1
        total = len(comments)
        cat_distribution = {k: round(v / total, 4) for k, v in cat_counts.items()}

        # Average word count and sentiment
        avg_word_count = sum(c.word_count for c in comments) / total
        avg_sentiment = sum(c.sentiment_score for c in comments) / total

        # Unique submitters
        unique_submitters = len({c.submitter_id for c in comments if c.submitter_id})

        # Topics
        topic_counts: dict[str, int] = {}
        for c in comments:
            if c.topic:
                topic_counts[c.topic] = topic_counts.get(c.topic, 0) + 1

        # Engagement depth score
        category_diversity = self._shannon_entropy(list(cat_counts.values()))
        max_category_entropy = (
            math.log2(len(CommentCategory)) if len(CommentCategory) > 1 else 1.0
        )
        normalized_diversity = (
            category_diversity / max_category_entropy
            if max_category_entropy > 0
            else 0.0
        )

        length_score = min(avg_word_count / 200.0, 1.0)
        submitter_ratio = unique_submitters / total if total > 0 else 0.0

        engagement_depth = (
            0.4 * normalized_diversity + 0.3 * length_score + 0.3 * submitter_ratio
        )

        return CommentAnalysis(
            total_comments=total,
            category_distribution=cat_distribution,
            average_word_count=round(avg_word_count, 2),
            average_sentiment=round(avg_sentiment, 4),
            unique_submitters=unique_submitters,
            topics=topic_counts,
            engagement_depth_score=round(engagement_depth, 4),
        )

    @staticmethod
    def _shannon_entropy(counts: list[int]) -> float:
        """Compute Shannon entropy from a list of counts."""
        total = sum(counts)
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy


class VoterTurnoutModel:
    """
    Models and predicts voter turnout for civic elections and measures.

    Uses historical turnout data to compute trends and predict
    future turnout based on contextual factors.
    """

    def __init__(self) -> None:
        self._turnout_records: list[dict[str, Any]] = []

    def add_election(
        self,
        election_id: str,
        eligible_voters: int,
        actual_voters: int,
        election_type: str,
        date: float,
        is_contested: bool = True,
        media_coverage_score: float = 0.5,
    ) -> None:
        """
        Add a historical election turnout record.

        Args:
            election_id: Identifier for the election.
            eligible_voters: Number of eligible voters.
            actual_voters: Number who actually voted.
            election_type: Type of election (e.g., "general", "primary", "special").
            date: Timestamp of the election.
            is_contested: Whether the election was contested.
            media_coverage_score: Media attention level (0-1).
        """
        self._turnout_records.append(
            {
                "election_id": election_id,
                "eligible_voters": eligible_voters,
                "actual_voters": actual_voters,
                "turnout_rate": actual_voters / eligible_voters
                if eligible_voters > 0
                else 0.0,
                "election_type": election_type,
                "date": date,
                "is_contested": is_contested,
                "media_coverage_score": media_coverage_score,
            }
        )

    def compute_average_turnout(
        self,
        election_type: str | None = None,
    ) -> float:
        """
        Compute average turnout rate across all or filtered elections.

        Args:
            election_type: Optional filter for election type.

        Returns:
            Average turnout rate as a float in [0, 1].

        Raises:
            ValueError: If no election records are available.
        """
        records = self._turnout_records
        if election_type:
            records = [r for r in records if r["election_type"] == election_type]

        if not records:
            raise ValueError("No election records available")

        return float(round(sum(r["turnout_rate"] for r in records) / len(records), 4))

    def predict_turnout(
        self,
        eligible_voters: int,
        election_type: str,
        is_contested: bool = True,
        media_coverage_score: float = 0.5,
    ) -> dict[str, Any]:
        """
        Predict voter turnout for an upcoming election.

        Estimates turnout as the historical baseline rate for the election
        type, plus fixed offsets for contestedness and media coverage. No
        statistical fitting is performed.

        Args:
            eligible_voters: Number of eligible voters.
            election_type: Type of upcoming election.
            is_contested: Whether the election is contested.
            media_coverage_score: Expected media coverage (0-1).

        Returns:
            Dictionary with predicted turnout rate and confidence.

        Raises:
            ValueError: If no historical data is available.
        """
        if not self._turnout_records:
            raise ValueError("No historical data available for prediction")

        # Baseline: average for this election type
        type_records = [
            r for r in self._turnout_records if r["election_type"] == election_type
        ]
        if type_records:
            base_rate = sum(r["turnout_rate"] for r in type_records) / len(type_records)
        else:
            base_rate = sum(r["turnout_rate"] for r in self._turnout_records) / len(
                self._turnout_records
            )

        # Adjustment factors
        contest_adjustment = 0.0
        contested_records = [r for r in self._turnout_records if r["is_contested"]]
        uncontested_records = [
            r for r in self._turnout_records if not r["is_contested"]
        ]

        if contested_records and uncontested_records:
            contested_avg = sum(r["turnout_rate"] for r in contested_records) / len(
                contested_records
            )
            uncontested_avg = sum(r["turnout_rate"] for r in uncontested_records) / len(
                uncontested_records
            )
            contest_effect = contested_avg - uncontested_avg
            if not is_contested:
                contest_adjustment = -abs(contest_effect)

        media_adjustment = (media_coverage_score - 0.5) * 0.10

        predicted_rate = max(
            0.0, min(1.0, base_rate + contest_adjustment + media_adjustment)
        )

        # Confidence based on available data
        n = len(type_records) if type_records else len(self._turnout_records)
        confidence = min(1.0, n / 10.0)

        predicted_voters = int(predicted_rate * eligible_voters)

        return {
            "predicted_turnout_rate": round(predicted_rate, 4),
            "predicted_voters": predicted_voters,
            "eligible_voters": eligible_voters,
            "confidence": round(confidence, 4),
            "base_rate": round(base_rate, 4),
            "adjustments": {
                "contest_effect": round(contest_adjustment, 4),
                "media_effect": round(media_adjustment, 4),
            },
        }

    def get_turnout_summary(self) -> dict[str, Any]:
        """
        Get a summary of all turnout data.

        Returns:
            Dictionary with overall and per-type turnout statistics.
        """
        if not self._turnout_records:
            return {"total_elections": 0, "overall_average_turnout": 0.0, "by_type": {}}

        type_rates: dict[str, list[float]] = {}
        for r in self._turnout_records:
            t = r["election_type"]
            if t not in type_rates:
                type_rates[t] = []
            type_rates[t].append(r["turnout_rate"])

        by_type = {}
        for t, rates in type_rates.items():
            by_type[t] = {
                "count": len(rates),
                "average_turnout": round(sum(rates) / len(rates), 4),
                "min_turnout": round(min(rates), 4),
                "max_turnout": round(max(rates), 4),
            }

        all_rates = [r["turnout_rate"] for r in self._turnout_records]
        return {
            "total_elections": len(self._turnout_records),
            "overall_average_turnout": round(sum(all_rates) / len(all_rates), 4),
            "by_type": by_type,
        }
