## GEO-INFER-CIV — Participatory Mapping and Civic Engagement

**Purpose.** GEO-INFER-CIV empowers communities with participatory mapping, citizen science, and collaborative geospatial decision-making tools. Its README defines the module as the civic-technology vertical: the machinery through which public participation data — meetings, comments, turnout, representation — becomes analyzable geospatial evidence.

**Public API.** The `geo_infer_civ` package exports analytical components from its `core` subpackage. Participation: `ParticipationAnalyzer` with `ParticipationMethod`, `ParticipantRecord`, `EngagementScore`, and `RepresentationReport`. Engagement: `AttendanceTracker`, `PublicCommentAnalyzer`, `VoterTurnoutModel`, and meeting records via `MeetingRecord`, `MeetingType`, `PublicComment`, and `CommentCategory`. Policy analysis: `CostBenefitAnalyzer`, `StakeholderImpactAnalyzer`, and `EquityAnalyzer` with supporting types `CostBenefitItem`, `StakeholderImpact`, `ImpactLevel`, and `PolicyDomain`. The package is deliberately lean — `core/` is its only code subpackage, matching the README's analysis-first scope.

The tests exercise participation, engagement, and policy analysis. The shared repository runner selects this module with `--module CIV`.

**Theme role.** Domain sciences: CIV is the social/civic band, extending the framework's inference stack to human institutions and demonstrating citizen-science data as a first-class geospatial source.
