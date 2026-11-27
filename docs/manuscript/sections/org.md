## GEO-INFER-ORG — Organizational Structures and Governance

GEO-INFER-ORG models organizational structures, governance frameworks, and community processes for geospatial initiatives (per its `README.md`). Its package `geo_infer_org` contains `core/` and `__init__.py`, with data structures and decision algorithms in the owning package.

The public interface, verified from `__init__.py`, exports components across structural, decision and collaboration concerns. Structure: `OrganizationModel`, `OrgUnit`, `Role`, `Resource`, `OrgStructureType`, `RoleLevel`, and `OrgMetrics` for representing hierarchies and their measured properties. Decision-making: `VotingEngine`, `ConsensusModel`, `VotingMethod`, `DecisionStatus`, `Vote`, `Proposal`, and `VotingResult` for executing and recording collective choices. Collaboration: `CollaborationNetwork`, `TeamFormation`, `CollaborationEdge`, `CollaborationType`, `TeamMember`, `NetworkMetrics`, and `TeamFormationResult` for analyzing and composing teams as graphs.

The tests exercise structural models, voting and consensus, and collaboration networks. Organizational effectiveness requires evidence outside these software fixtures.

In the root README's Module Themes, ORG belongs to Governance, Risk & Domain alongside METAGOV, NORMS, PEP, REQ, SEC, and RISK. Its role is the organizational layer beneath the governance theme: METAGOV designs governance regimes and NORMS encodes rules, while ORG supplies the concrete organizational objects — units, roles, votes, teams — that such regimes operate on.
