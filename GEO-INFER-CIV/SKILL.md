---
name: geo-infer-civ
description: Civic engagement analytics and policy appraisal. Use when analyzing meeting attendance, public comments, voter turnout, participation indices, demographic representation, or cost-benefit / stakeholder / equity analysis of policy proposals.
prerequisites:
  recommended:
    - geo-infer-space
    - geo-infer-data
    - geo-infer-econ
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-CIV

## Instructions

### Core Capabilities

- **Attendance tracking** (`AttendanceTracker`): registered vs. actual attendance across meeting types, linear trend analysis, per-meeting effectiveness.
- **Public comment analysis** (`PublicCommentAnalyzer`): comment categorization, Shannon-entropy engagement depth, unique submitter ratios.
- **Voter turnout modeling** (`VoterTurnoutModel`): historical turnout summary and prediction with contestedness / media-coverage adjustments.
- **Participation analysis** (`ParticipationAnalyzer`): engagement scoring across eight participation methods, participation index relative to a baseline, demographic representation reports.
- **Policy appraisal** (`CostBenefitAnalyzer`, `StakeholderImpactAnalyzer`, `EquityAnalyzer`): discounted NPV, benefit-cost ratio, IRR, payback period, risk-adjusted NPV, stakeholder impact matrices, and Gini / disparate-impact equity scoring.

### Key Imports

```python
from geo_infer_civ import (
    AttendanceTracker,
    PublicCommentAnalyzer,
    VoterTurnoutModel,
    ParticipationAnalyzer,
    ParticipantRecord,
    ParticipationMethod,
    CostBenefitAnalyzer,
    CostBenefitItem,
    StakeholderImpactAnalyzer,
    EquityAnalyzer,
)
```

The full public surface also includes the enums (`MeetingType`, `CommentCategory`, `ImpactLevel`, `PolicyDomain`), the record dataclasses (`MeetingRecord`, `PublicComment`), and the result dataclasses (`AttendanceTrend`, `CommentAnalysis`, `EngagementScore`, `RepresentationReport`, `CostBenefitResult`, `StakeholderImpact`, `EquityScore`).

## Examples

```python
from geo_infer_civ import (
    CostBenefitAnalyzer,
    CostBenefitItem,
    ParticipantRecord,
    ParticipationAnalyzer,
    ParticipationMethod,
)

# Participation analysis
analyzer = ParticipationAnalyzer()
analyzer.add_records(
    [
        ParticipantRecord("p01", ParticipationMethod.TOWN_HALL, 1705276800.0, "group_a"),
        ParticipantRecord("p02", ParticipationMethod.SURVEY, 1705363200.0, "group_a"),
        ParticipantRecord("p03", ParticipationMethod.PUBLIC_COMMENT, 1705449600.0, "group_b"),
    ]
)
score = analyzer.compute_engagement_score(target_population=5000)
index = analyzer.compute_participation_index(target_population=10_000, baseline_rate=0.10)
report = analyzer.analyze_representation({"group_a": 0.5, "group_b": 0.5})

# Cost-benefit appraisal
cba = CostBenefitAnalyzer(discount_rate=0.05)
cba.add_items(
    [
        CostBenefitItem("construction", 250_000.0, is_benefit=False),
        CostBenefitItem("flood avoidance", 900_000.0, is_benefit=True, time_horizon_years=10),
    ]
)
result = cba.analyze()
print(result.net_present_value, result.benefit_cost_ratio, result.payback_period_years)
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog CIV` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- `compute_participation_index` returns `actual_rate / baseline_rate`: 1.0 means participation exactly at the baseline, values above 1.0 mean the baseline was exceeded. It is not bounded to [0, 1].
- `CostBenefitResult.internal_rate_of_return` is `nan` when NPV never crosses zero (all-cost or all-benefit item sets); `payback_period_years` is `inf` when break-even is never reached within the modeled horizon.

### Integrations

- No cross-module `geo_infer_*` imports exist in the code today; `geo-infer-space`, `geo-infer-data`, and `geo-infer-econ` are recommended companions only. The core runs on the standard library alone.
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module CIV`
