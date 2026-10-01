"""
Governance structures for GEO-INFER-ORG.

Provides decision protocols, voting mechanisms,
and consensus models for organizational governance.
"""

import logging
import math
from typing import Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class VotingMethod(Enum):
    """Available voting methods."""

    SIMPLE_MAJORITY = "simple_majority"
    SUPERMAJORITY = "supermajority"
    UNANIMOUS = "unanimous"
    RANKED_CHOICE = "ranked_choice"
    WEIGHTED = "weighted"
    APPROVAL = "approval"


class DecisionStatus(Enum):
    """Status of a governance decision."""

    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    TABLED = "tabled"


@dataclass
class Vote:
    """A single vote cast by a participant."""

    voter_id: str
    choice: str
    weight: float = 1.0
    rank: list[str] | None = None  # For ranked choice
    approvals: list[str] | None = None  # For approval voting
    timestamp: float = 0.0


@dataclass
class Proposal:
    """A governance proposal to be decided upon."""

    proposal_id: str
    title: str
    description: str
    proposer_id: str
    options: list[str]
    status: DecisionStatus = DecisionStatus.PROPOSED
    voting_method: VotingMethod = VotingMethod.SIMPLE_MAJORITY
    quorum_fraction: float = 0.5
    eligible_voters: int = 0
    votes: list[Vote] = field(default_factory=list)


@dataclass
class VotingResult:
    """Result of a voting process."""

    proposal_id: str
    winner: str | None
    vote_counts: dict[str, float]
    total_votes: int
    quorum_met: bool
    method: VotingMethod
    rounds: list[dict[str, float]] | None = None


class VotingEngine:
    """
    Implements multiple voting mechanisms for governance decisions.

    Supports simple majority, supermajority, unanimous, ranked-choice,
    weighted, and approval voting methods.
    """

    def __init__(self) -> None:
        self._proposals: dict[str, Proposal] = {}

    def create_proposal(self, proposal: Proposal) -> None:
        """
        Register a new proposal for voting.

        Args:
            proposal: The governance proposal.

        Raises:
            ValueError: If proposal_id already exists or options are empty.
        """
        if proposal.proposal_id in self._proposals:
            raise ValueError(f"Proposal {proposal.proposal_id} already exists")
        if len(proposal.options) < 2:
            raise ValueError("Proposal must have at least 2 options")
        self._proposals[proposal.proposal_id] = proposal
        logger.info(
            "Proposal created: %s (method=%s)",
            proposal.proposal_id,
            proposal.voting_method.value,
        )

    def cast_vote(self, proposal_id: str, vote: Vote) -> None:
        """
        Cast a vote on a proposal.

        Args:
            proposal_id: The proposal to vote on.
            vote: The vote to cast.

        Raises:
            KeyError: If the proposal doesn't exist.
            ValueError: If voter already voted or choice is invalid.
        """
        if proposal_id not in self._proposals:
            raise KeyError(f"Proposal {proposal_id} not found")

        proposal = self._proposals[proposal_id]

        # Check for duplicate voter
        existing_voter_ids = {v.voter_id for v in proposal.votes}
        if vote.voter_id in existing_voter_ids:
            raise ValueError(f"Voter {vote.voter_id} has already voted")

        # Validate choice based on method
        if proposal.voting_method not in (
            VotingMethod.RANKED_CHOICE,
            VotingMethod.APPROVAL,
        ):
            if vote.choice not in proposal.options:
                raise ValueError(f"Invalid choice: {vote.choice}")

        proposal.votes.append(vote)
        logger.debug("Vote cast on %s by %s", proposal_id, vote.voter_id)

    def tally(self, proposal_id: str) -> VotingResult:
        """
        Tally votes for a proposal using its configured voting method.

        Args:
            proposal_id: The proposal to tally.

        Returns:
            VotingResult with the outcome.

        Raises:
            KeyError: If the proposal doesn't exist.
        """
        if proposal_id not in self._proposals:
            raise KeyError(f"Proposal {proposal_id} not found")

        proposal = self._proposals[proposal_id]
        method = proposal.voting_method

        quorum_met = True
        if proposal.eligible_voters > 0:
            participation = len(proposal.votes) / proposal.eligible_voters
            quorum_met = participation >= proposal.quorum_fraction

        if method == VotingMethod.SIMPLE_MAJORITY:
            return self._tally_simple_majority(proposal, quorum_met)
        elif method == VotingMethod.SUPERMAJORITY:
            return self._tally_supermajority(proposal, quorum_met)
        elif method == VotingMethod.UNANIMOUS:
            return self._tally_unanimous(proposal, quorum_met)
        elif method == VotingMethod.RANKED_CHOICE:
            return self._tally_ranked_choice(proposal, quorum_met)
        elif method == VotingMethod.WEIGHTED:
            return self._tally_weighted(proposal, quorum_met)
        elif method == VotingMethod.APPROVAL:
            return self._tally_approval(proposal, quorum_met)
        else:
            raise ValueError(f"Unknown voting method: {method}")

    def _tally_simple_majority(
        self, proposal: Proposal, quorum_met: bool
    ) -> VotingResult:
        """Tally a simple-majority vote.

        Semantics: plurality wins. The option with the most votes is the
        winner (no strict >50% requirement); ties resolve deterministically
        by the declared option order via ``max``. If quorum is not met,
        there are no votes, or the top option has zero votes, there is
        no winner.
        """
        counts: dict[str, float] = {opt: 0 for opt in proposal.options}
        for vote in proposal.votes:
            if vote.choice in counts:
                counts[vote.choice] += 1

        total = len(proposal.votes)
        winner = (
            max(counts, key=lambda k: counts[k]) if total > 0 and quorum_met else None
        )
        if winner and counts[winner] == 0:
            winner = None

        return VotingResult(
            proposal_id=proposal.proposal_id,
            winner=winner,
            vote_counts=counts,
            total_votes=total,
            quorum_met=quorum_met,
            method=VotingMethod.SIMPLE_MAJORITY,
        )

    def _tally_supermajority(
        self, proposal: Proposal, quorum_met: bool, threshold: float = 2.0 / 3.0
    ) -> VotingResult:
        counts: dict[str, float] = {opt: 0 for opt in proposal.options}
        for vote in proposal.votes:
            if vote.choice in counts:
                counts[vote.choice] += 1

        total = len(proposal.votes)
        winner = None
        if total > 0 and quorum_met:
            best = max(counts, key=lambda k: counts[k])
            if counts[best] / total >= threshold:
                winner = best

        return VotingResult(
            proposal_id=proposal.proposal_id,
            winner=winner,
            vote_counts=counts,
            total_votes=total,
            quorum_met=quorum_met,
            method=VotingMethod.SUPERMAJORITY,
        )

    def _tally_unanimous(self, proposal: Proposal, quorum_met: bool) -> VotingResult:
        counts: dict[str, float] = {opt: 0 for opt in proposal.options}
        for vote in proposal.votes:
            if vote.choice in counts:
                counts[vote.choice] += 1

        total = len(proposal.votes)
        winner = None
        if total > 0 and quorum_met:
            for opt, count in counts.items():
                if count == total:
                    winner = opt
                    break

        return VotingResult(
            proposal_id=proposal.proposal_id,
            winner=winner,
            vote_counts=counts,
            total_votes=total,
            quorum_met=quorum_met,
            method=VotingMethod.UNANIMOUS,
        )

    def _tally_ranked_choice(
        self, proposal: Proposal, quorum_met: bool
    ) -> VotingResult:
        """Instant-runoff ranked choice voting."""
        if not quorum_met or not proposal.votes:
            return VotingResult(
                proposal_id=proposal.proposal_id,
                winner=None,
                vote_counts={opt: 0 for opt in proposal.options},
                total_votes=len(proposal.votes),
                quorum_met=quorum_met,
                method=VotingMethod.RANKED_CHOICE,
                rounds=[],
            )

        active_candidates = set(proposal.options)
        ballots = []
        for vote in proposal.votes:
            if vote.rank:
                ballots.append(list(vote.rank))
            else:
                ballots.append([vote.choice])

        rounds: list[dict[str, float]] = []
        total = len(ballots)

        while len(active_candidates) > 1:
            round_counts: dict[str, float] = {c: 0 for c in active_candidates}
            for ballot in ballots:
                for choice in ballot:
                    if choice in active_candidates:
                        round_counts[choice] += 1
                        break

            rounds.append(dict(round_counts))

            # Majority check: a candidate with strictly more than half of
            # all ballots wins immediately.
            best = max(round_counts, key=lambda k: round_counts[k])
            if round_counts[best] > total / 2.0:
                logger.debug("IRV winner %s in round %d", best, len(rounds))
                return VotingResult(
                    proposal_id=proposal.proposal_id,
                    winner=best,
                    vote_counts=round_counts,
                    total_votes=total,
                    quorum_met=quorum_met,
                    method=VotingMethod.RANKED_CHOICE,
                    rounds=rounds,
                )

            # Standard IRV eliminates exactly ONE candidate per round: the
            # lowest-scoring candidate, with a deterministic tie-break
            # (lexicographically first candidate id among the tied).
            min_count = min(round_counts.values())
            tied = sorted(c for c, cnt in round_counts.items() if cnt == min_count)
            eliminated = tied[0]
            active_candidates.discard(eliminated)
            logger.debug(
                "IRV round %d: eliminated %s (tied: %s)", len(rounds), eliminated, tied
            )

        # One candidate remains after the eliminations. Because zero-vote
        # candidates are always strict-minimum and eliminated first, the
        # survivor carries the transferred active votes; the old behavior
        # of re-scanning the last round (which could resurrect an
        # already-eliminated candidate) is gone.
        winner = next(iter(active_candidates)) if active_candidates else None

        return VotingResult(
            proposal_id=proposal.proposal_id,
            winner=winner,
            vote_counts=rounds[-1] if rounds else {},
            total_votes=total,
            quorum_met=quorum_met,
            method=VotingMethod.RANKED_CHOICE,
            rounds=rounds,
        )

    def _tally_weighted(self, proposal: Proposal, quorum_met: bool) -> VotingResult:
        counts: dict[str, float] = {opt: 0.0 for opt in proposal.options}
        for vote in proposal.votes:
            if vote.choice in counts:
                counts[vote.choice] += vote.weight

        total = len(proposal.votes)
        winner = None
        if total > 0 and quorum_met:
            winner = max(counts, key=lambda k: counts[k])
            if counts[winner] == 0:
                winner = None

        return VotingResult(
            proposal_id=proposal.proposal_id,
            winner=winner,
            vote_counts={k: round(v, 4) for k, v in counts.items()},
            total_votes=total,
            quorum_met=quorum_met,
            method=VotingMethod.WEIGHTED,
        )

    def _tally_approval(self, proposal: Proposal, quorum_met: bool) -> VotingResult:
        counts: dict[str, float] = {opt: 0 for opt in proposal.options}
        for vote in proposal.votes:
            approvals = vote.approvals or [vote.choice]
            for choice in approvals:
                if choice in counts:
                    counts[choice] += 1

        total = len(proposal.votes)
        winner = (
            max(counts, key=lambda k: counts[k]) if total > 0 and quorum_met else None
        )
        if winner and counts[winner] == 0:
            winner = None

        return VotingResult(
            proposal_id=proposal.proposal_id,
            winner=winner,
            vote_counts=counts,
            total_votes=total,
            quorum_met=quorum_met,
            method=VotingMethod.APPROVAL,
        )


class ConsensusModel:
    """
    Models consensus-building processes for organizational decisions.

    Participants rate options round by round (the caller drives the
    rounds: collect ratings via :meth:`submit_rating`, score them with
    :meth:`compute_consensus`, and use :meth:`check_convergence` against
    the previous round's scores to decide when agreement has stabilized
    within ``convergence_threshold``). This class does not loop on its own.
    """

    def __init__(self, convergence_threshold: float = 0.05) -> None:
        """
        Initialize the consensus model.

        Args:
            convergence_threshold: Minimum change between rounds to continue.
        """
        self._convergence_threshold = convergence_threshold
        self._options: list[str] = []
        self._ratings: dict[
            str, dict[str, float]
        ] = {}  # participant -> option -> rating

    def set_options(self, options: list[str]) -> None:
        """
        Set the options to build consensus on.

        Args:
            options: List of option names.
        """
        self._options = list(options)

    def submit_rating(self, participant_id: str, ratings: dict[str, float]) -> None:
        """
        Submit participant ratings for each option (scale 0-10).

        Args:
            participant_id: The participant identifier.
            ratings: Mapping of option name to rating (0-10).

        Raises:
            ValueError: If ratings don't cover all options or are out of range.
        """
        for opt in self._options:
            if opt not in ratings:
                raise ValueError(f"Missing rating for option: {opt}")
        for opt, val in ratings.items():
            if not (0.0 <= val <= 10.0):
                raise ValueError(f"Rating for {opt} must be in [0, 10], got {val}")

        self._ratings[participant_id] = ratings

    def compute_consensus(self) -> dict[str, Any]:
        """
        Compute the current consensus state.

        Returns:
            Dictionary with:
            - option_scores: average rating per option
            - consensus_level: 0 (no consensus) to 1 (full agreement)
            - recommendation: the highest-scoring option
            - spread: standard deviation of ratings per option

        Raises:
            ValueError: If no ratings have been submitted.
        """
        if not self._ratings:
            raise ValueError("No ratings submitted")

        n_participants = len(self._ratings)
        option_totals: dict[str, float] = {opt: 0.0 for opt in self._options}
        option_values: dict[str, list[float]] = {opt: [] for opt in self._options}

        for participant_ratings in self._ratings.values():
            for opt, val in participant_ratings.items():
                if opt in option_totals:
                    option_totals[opt] += val
                    option_values[opt].append(val)

        option_scores = {
            opt: round(total / n_participants, 4)
            for opt, total in option_totals.items()
        }

        # Spread (standard deviation) per option
        spread: dict[str, float] = {}
        for opt, values in option_values.items():
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                spread[opt] = round(math.sqrt(variance), 4)
            else:
                spread[opt] = 0.0

        # Consensus level: inverse of average spread, normalized
        avg_spread = sum(spread.values()) / len(spread) if spread else 0.0
        # Max possible spread on 0-10 scale is 5.0 (half the range)
        consensus_level = max(0.0, 1.0 - avg_spread / 5.0)

        recommendation = max(option_scores, key=lambda k: option_scores[k])

        return {
            "option_scores": option_scores,
            "consensus_level": round(consensus_level, 4),
            "recommendation": recommendation,
            "spread": spread,
            "participant_count": n_participants,
        }

    def check_convergence(self, previous_scores: dict[str, float]) -> bool:
        """
        Check if consensus has converged compared to previous scores.

        Args:
            previous_scores: Option scores from the previous round.

        Returns:
            True if the change is below the convergence threshold.
        """
        current = self.compute_consensus()["option_scores"]
        total_change = sum(
            abs(current.get(opt, 0) - previous_scores.get(opt, 0))
            for opt in self._options
        )
        avg_change = total_change / len(self._options) if self._options else 0.0
        return avg_change < self._convergence_threshold
