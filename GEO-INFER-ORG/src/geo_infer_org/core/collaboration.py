"""
Collaboration networks for GEO-INFER-ORG.

Provides team formation analysis, knowledge sharing metrics,
and coordination scoring for organizational collaboration.
"""

import logging
from typing import Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CollaborationType(Enum):
    """Types of collaboration interactions."""

    KNOWLEDGE_SHARE = "knowledge_share"
    TASK_COORDINATION = "task_coordination"
    DECISION_MAKING = "decision_making"
    MENTORING = "mentoring"
    PEER_REVIEW = "peer_review"
    JOINT_PROJECT = "joint_project"


@dataclass
class CollaborationEdge:
    """An edge in the collaboration network between two entities."""

    source_id: str
    target_id: str
    collaboration_type: CollaborationType
    strength: float = 1.0
    frequency: int = 1
    timestamp: float = 0.0


@dataclass
class TeamMember:
    """A member with skills that can be assigned to teams."""

    member_id: str
    name: str
    skills: list[str]
    capacity: float = 1.0
    unit_id: str | None = None
    location: tuple[float, float] | None = None


@dataclass
class NetworkMetrics:
    """Metrics for a collaboration network."""

    node_count: int
    edge_count: int
    density: float
    avg_degree: float
    clustering_coefficient: float
    connected_components: int
    most_central_nodes: list[str]


@dataclass
class TeamFormationResult:
    """Result of a team formation optimization."""

    team_members: list[str]
    skill_coverage: float
    team_diversity: float
    coordination_cost: float
    overall_score: float


class CollaborationNetwork:
    """
    Models and analyzes collaboration networks within organizations.

    Represents collaboration as a weighted directed graph and computes
    network metrics including density, centrality, and clustering.
    """

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._edges: list[CollaborationEdge] = []
        self._adjacency: dict[str, dict[str, float]] = {}

    def add_node(self, node_id: str) -> None:
        """
        Add a node to the collaboration network.

        Args:
            node_id: Identifier for the node (person, team, or unit).
        """
        self._nodes.add(node_id)
        self._adjacency.setdefault(node_id, {})

    def add_edge(self, edge: CollaborationEdge) -> None:
        """
        Add a collaboration edge to the network.

        Args:
            edge: The collaboration interaction edge.
        """
        self._nodes.add(edge.source_id)
        self._nodes.add(edge.target_id)
        self._adjacency.setdefault(edge.source_id, {})
        self._adjacency.setdefault(edge.target_id, {})

        # Accumulate strength for repeated edges
        current = self._adjacency[edge.source_id].get(edge.target_id, 0.0)
        self._adjacency[edge.source_id][edge.target_id] = current + edge.strength

        self._edges.append(edge)
        logger.debug(
            "Collaboration edge added: %s -> %s (%s, strength=%.2f)",
            edge.source_id,
            edge.target_id,
            edge.collaboration_type.value,
            edge.strength,
        )

    def compute_metrics(self) -> NetworkMetrics:
        """
        Compute network-level metrics.

        Returns:
            NetworkMetrics with density, average degree, clustering, etc.
        """
        n = len(self._nodes)
        e = len(self._edges)

        if n == 0:
            return NetworkMetrics(
                node_count=0,
                edge_count=0,
                density=0.0,
                avg_degree=0.0,
                clustering_coefficient=0.0,
                connected_components=0,
                most_central_nodes=[],
            )

        # Density (for directed graph)
        max_edges = n * (n - 1) if n > 1 else 1
        density = e / max_edges

        # Degree (undirected view)
        degree: dict[str, int] = {node: 0 for node in self._nodes}
        neighbors: dict[str, set[str]] = {node: set() for node in self._nodes}
        for edge in self._edges:
            degree[edge.source_id] += 1
            degree[edge.target_id] += 1
            neighbors[edge.source_id].add(edge.target_id)
            neighbors[edge.target_id].add(edge.source_id)

        avg_degree = sum(degree.values()) / n if n > 0 else 0.0

        # Clustering coefficient (average local)
        clustering_coefficients = []
        for node in self._nodes:
            nbrs = neighbors[node]
            k = len(nbrs)
            if k < 2:
                clustering_coefficients.append(0.0)
                continue
            # Count edges between neighbors
            links_between = 0
            nbr_list = list(nbrs)
            for i in range(len(nbr_list)):
                for j in range(i + 1, len(nbr_list)):
                    if nbr_list[j] in neighbors[nbr_list[i]]:
                        links_between += 1
            possible = k * (k - 1) / 2
            clustering_coefficients.append(
                links_between / possible if possible > 0 else 0.0
            )

        avg_clustering = (
            sum(clustering_coefficients) / len(clustering_coefficients)
            if clustering_coefficients
            else 0.0
        )

        # Connected components (undirected)
        visited: set[str] = set()
        components = 0
        for node in self._nodes:
            if node not in visited:
                components += 1
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        stack.extend(neighbors[current] - visited)

        # Most central nodes (by degree)
        sorted_by_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)
        top_central = [node_id for node_id, _ in sorted_by_degree[: min(5, n)]]

        return NetworkMetrics(
            node_count=n,
            edge_count=e,
            density=round(density, 4),
            avg_degree=round(avg_degree, 2),
            clustering_coefficient=round(avg_clustering, 4),
            connected_components=components,
            most_central_nodes=top_central,
        )

    def compute_betweenness_centrality(self) -> dict[str, float]:
        """
        Compute betweenness centrality for each node.

        Uses unweighted Brandes' algorithm: BFS shortest-path counting with
        dependency accumulation from every source, then normalization.

        Documented contract: the graph is treated as undirected and edge
        weights are ignored; the directed normalization factor is applied to
        the undirected per-source accumulation, which matches
        ``networkx.betweenness_centrality`` for the same graph exactly
        (pinned by TestBetweennessVsNetworkx).

        Returns:
            Mapping of node_id to betweenness centrality score.
        """
        centrality: dict[str, float] = {node: 0.0 for node in self._nodes}
        nodes_list = list(self._nodes)

        # Build undirected neighbor map
        neighbors: dict[str, set[str]] = {node: set() for node in self._nodes}
        for edge in self._edges:
            neighbors[edge.source_id].add(edge.target_id)
            neighbors[edge.target_id].add(edge.source_id)

        for source in nodes_list:
            # BFS from source
            stack: list[str] = []
            predecessors: dict[str, list[str]] = {node: [] for node in self._nodes}
            sigma: dict[str, int] = {node: 0 for node in self._nodes}
            sigma[source] = 1
            dist: dict[str, int] = {node: -1 for node in self._nodes}
            dist[source] = 0

            queue = [source]
            while queue:
                current = queue.pop(0)
                stack.append(current)
                for neighbor in neighbors[current]:
                    if dist[neighbor] < 0:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
                    if dist[neighbor] == dist[current] + 1:
                        sigma[neighbor] += sigma[current]
                        predecessors[neighbor].append(current)

            # Accumulate dependencies
            delta: dict[str, float] = {node: 0.0 for node in self._nodes}
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != source:
                    centrality[w] += delta[w]

        # Normalization: although this factor is the directed one, it is
        # correct here. The per-source accumulation above runs over the
        # undirected adjacency, so every unordered pair is counted twice
        # (once from each endpoint); networkx applies the same 1/2 rescale
        # for undirected graphs, so the results match exactly. Pinned by
        # TestBetweennessVsNetworkx.
        n = len(self._nodes)
        norm = (n - 1) * (n - 2) if n > 2 else 1
        return {node: round(val / norm, 6) for node, val in centrality.items()}

    def get_knowledge_flow(self) -> dict[str, dict[str, float]]:
        """
        Analyze knowledge flow patterns in the network.

        Looks at KNOWLEDGE_SHARE type edges to identify knowledge
        sources and sinks.

        Returns:
            Dictionary with "sources" (outbound knowledge) and
            "sinks" (inbound knowledge) scores.
        """
        outbound: dict[str, float] = {}
        inbound: dict[str, float] = {}

        for edge in self._edges:
            if edge.collaboration_type == CollaborationType.KNOWLEDGE_SHARE:
                outbound[edge.source_id] = (
                    outbound.get(edge.source_id, 0.0) + edge.strength
                )
                inbound[edge.target_id] = (
                    inbound.get(edge.target_id, 0.0) + edge.strength
                )

        return {
            "sources": {
                k: round(v, 4)
                for k, v in sorted(outbound.items(), key=lambda x: x[1], reverse=True)
            },
            "sinks": {
                k: round(v, 4)
                for k, v in sorted(inbound.items(), key=lambda x: x[1], reverse=True)
            },
        }


class TeamFormation:
    """
    Optimizes team formation based on skill requirements and constraints.

    Given a pool of available members and required skills, selects
    optimal team compositions that maximize coverage and minimize cost.
    """

    def __init__(self) -> None:
        self._members: dict[str, TeamMember] = {}

    def add_member(self, member: TeamMember) -> None:
        """
        Add a candidate member to the pool.

        Args:
            member: Team member with skills and capacity.
        """
        self._members[member.member_id] = member

    def add_members(self, members: list[TeamMember]) -> None:
        """
        Add multiple candidate members.

        Args:
            members: List of team members.
        """
        for m in members:
            self._members[m.member_id] = m

    def form_team(
        self,
        required_skills: list[str],
        max_size: int = 10,
        prefer_diverse_units: bool = True,
    ) -> TeamFormationResult:
        """
        Form an optimal team for the given skill requirements.

        Uses a greedy algorithm that iteratively selects the member
        covering the most uncovered skills, weighted by the member's
        available capacity, with diversity bonuses.

        Args:
            required_skills: List of skills the team must cover.
            max_size: Maximum team size.
            prefer_diverse_units: Bonus for members from different org units.

        Returns:
            TeamFormationResult with the selected team and quality metrics.

        Raises:
            ValueError: If no members are available.
        """
        if not self._members:
            raise ValueError("No members available for team formation")

        required_set = set(required_skills)
        covered: set[str] = set()
        selected: list[str] = []
        selected_units: set[str] = set()

        available = dict(self._members)

        for _ in range(min(max_size, len(available))):
            if covered >= required_set:
                break

            best_id = None
            best_score = -1.0

            for mid, member in available.items():
                if mid in selected:
                    continue

                member_skills = set(member.skills)
                new_coverage = len(member_skills & required_set - covered)
                diversity_bonus = 0.0
                if (
                    prefer_diverse_units
                    and member.unit_id
                    and member.unit_id not in selected_units
                ):
                    diversity_bonus = 0.5

                # Capacity weights the contribution: a member at full
                # availability contributes fully; a loaded member is
                # discounted proportionally to their remaining capacity.
                score = (new_coverage + diversity_bonus) * max(member.capacity, 0.0)

                if score > best_score:
                    best_score = score
                    best_id = mid

            if best_id is None or best_score <= 0:
                break

            selected.append(best_id)
            member = available[best_id]
            covered.update(set(member.skills) & required_set)
            if member.unit_id:
                selected_units.add(member.unit_id)

        # Compute metrics
        skill_coverage = len(covered) / len(required_set) if required_set else 1.0

        # Team diversity: number of unique units / team size
        units = {
            self._members[mid].unit_id for mid in selected if self._members[mid].unit_id
        }
        team_diversity = len(units) / len(selected) if selected else 0.0

        # Coordination cost: scales with team size squared (communication overhead)
        coordination_cost = (
            len(selected) * (len(selected) - 1) / 2.0 if len(selected) > 1 else 0.0
        )
        normalized_cost = (
            coordination_cost / (max_size * (max_size - 1) / 2.0)
            if max_size > 1
            else 0.0
        )

        overall = (
            0.5 * skill_coverage + 0.3 * team_diversity + 0.2 * (1.0 - normalized_cost)
        )

        logger.info(
            "Team formed: members=%s coverage=%.2f diversity=%.2f",
            selected,
            skill_coverage,
            team_diversity,
        )
        return TeamFormationResult(
            team_members=selected,
            skill_coverage=round(skill_coverage, 4),
            team_diversity=round(team_diversity, 4),
            coordination_cost=round(normalized_cost, 4),
            overall_score=round(overall, 4),
        )

    def compute_skill_gap(self, required_skills: list[str]) -> dict[str, Any]:
        """
        Identify skill gaps in the available member pool.

        Args:
            required_skills: Skills that need to be covered.

        Returns:
            Dictionary with covered skills, missing skills, and coverage ratio.
        """
        all_skills: set[str] = set()
        for member in self._members.values():
            all_skills.update(member.skills)

        required_set = set(required_skills)
        covered = all_skills & required_set
        missing = required_set - all_skills

        # Skill frequency
        skill_freq: dict[str, int] = {}
        for member in self._members.values():
            for skill in member.skills:
                if skill in required_set:
                    skill_freq[skill] = skill_freq.get(skill, 0) + 1

        return {
            "required": sorted(required_set),
            "covered": sorted(covered),
            "missing": sorted(missing),
            "coverage_ratio": round(len(covered) / len(required_set), 4)
            if required_set
            else 1.0,
            "skill_frequency": skill_freq,
        }
