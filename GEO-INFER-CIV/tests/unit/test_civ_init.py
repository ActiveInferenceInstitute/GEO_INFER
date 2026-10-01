"""Tests for GEO-INFER-CIV module initialization and imports."""

from pathlib import Path
import tomllib


class TestCivImports:
    def test_import_module(self):
        import geo_infer_civ

        project = tomllib.loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        )["project"]
        assert geo_infer_civ.__version__ == project["version"]

    def test_import_participation(self):
        from geo_infer_civ import (
            ParticipationAnalyzer,
            ParticipationMethod,
            ParticipantRecord,
        )

        assert all(
            isinstance(exported, type)
            for exported in (
                ParticipationMethod,
                ParticipantRecord,
            )
        )

        assert ParticipationAnalyzer is not None
        analyzer = ParticipationAnalyzer()
        assert analyzer is not None

    def test_import_engagement(self):
        from geo_infer_civ import (
            AttendanceTracker,
            PublicCommentAnalyzer,
            VoterTurnoutModel,
        )

        assert AttendanceTracker is not None
        assert PublicCommentAnalyzer is not None
        assert VoterTurnoutModel is not None

    def test_import_policy(self):
        from geo_infer_civ import (
            CostBenefitAnalyzer,
            StakeholderImpactAnalyzer,
            EquityAnalyzer,
        )

        assert CostBenefitAnalyzer is not None
        assert StakeholderImpactAnalyzer is not None
        assert EquityAnalyzer is not None

    def test_import_enums(self):
        from geo_infer_civ import (
            ParticipationMethod,
            MeetingType,
            CommentCategory,
            ImpactLevel,
            PolicyDomain,
        )

        assert len(ParticipationMethod) > 0
        assert len(MeetingType) > 0
        assert len(CommentCategory) > 0
        assert len(ImpactLevel) > 0
        assert len(PolicyDomain) > 0

    def test_core_module_imports(self):
        from geo_infer_civ.core import (
            ParticipationAnalyzer,
            AttendanceTracker,
            CostBenefitAnalyzer,
        )

        assert ParticipationAnalyzer is not None
        assert AttendanceTracker is not None
        assert CostBenefitAnalyzer is not None

    def test_import_result_dataclasses(self):
        from geo_infer_civ import (
            AttendanceTrend,
            CommentAnalysis,
            CostBenefitResult,
            EquityScore,
        )

        for cls in (AttendanceTrend, CommentAnalysis, CostBenefitResult, EquityScore):
            assert cls is not None
