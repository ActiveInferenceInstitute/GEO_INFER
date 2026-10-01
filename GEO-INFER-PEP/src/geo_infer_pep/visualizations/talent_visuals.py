"""Talent Acquisition Data Visualization functions."""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..models.talent_models import Candidate, CandidateStatus
from ..talent.transformer import convert_candidates_to_dataframe

logger = logging.getLogger(__name__)

# Default Talent output directory; created on write by the plot functions below.
DEFAULT_TALENT_VISUALS_DIR = Path("visualizations_output/talent")


def plot_candidate_pipeline_by_status(
    candidates: list[Candidate], output_dir: Path = DEFAULT_TALENT_VISUALS_DIR
) -> str | None:
    """
    Generates a bar chart of candidates by their current status in the pipeline.
    """
    if not candidates:
        logger.info("No candidate data for pipeline status plot.")
        return None

    df = convert_candidates_to_dataframe(candidates)
    if df.empty or "status" not in df.columns:
        logger.info("Candidate data is empty or 'status' column missing.")
        return None

    plt.figure(figsize=(12, 7))
    # Ensure the order of CandidateStatus enum is used for a logical flow if desired
    status_order = [status.value for status in CandidateStatus]
    sns.countplot(
        data=df,
        y="status",
        order=df["status"].value_counts().reindex(status_order, fill_value=0).index,
        hue="status",
        palette="magma",
        legend=False,
    )
    plt.title("Candidate Pipeline by Status")
    plt.xlabel("Number of Candidates")
    plt.ylabel("Status")
    plt.tight_layout()

    file_path = output_dir / "candidate_pipeline_status.png"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path)
        logger.info(f"Saved candidate pipeline status plot to: {file_path}")
        plt.close()
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        plt.close()
        return None


def plot_time_to_hire_distribution(
    hired_candidates_with_tth_days: list[int],
    output_dir: Path = DEFAULT_TALENT_VISUALS_DIR,
) -> str | None:
    """
    Generates a histogram for Time to Hire distribution.
    Expects a list of integers representing TTH in days.
    """
    if not hired_candidates_with_tth_days:
        logger.info("No Time to Hire data for distribution plot.")
        return None

    plt.figure(figsize=(10, 6))
    sns.histplot(hired_candidates_with_tth_days, kde=True, bins=15)
    plt.title("Distribution of Time to Hire (Days)")
    plt.xlabel("Days to Hire")
    plt.ylabel("Number of Hires")
    plt.tight_layout()

    file_path = output_dir / "time_to_hire_distribution.png"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path)
        logger.info(f"Saved Time to Hire distribution plot to: {file_path}")
        plt.close()
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        plt.close()
        return None


# Add more Talent visualization functions here, e.g.:
# - Offer acceptance rate over time (line chart)
# - Candidate source effectiveness (bar chart)
# - Pipeline conversion rates (funnel chart)
