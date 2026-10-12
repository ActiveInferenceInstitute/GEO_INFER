"""CRM Data Visualization functions."""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..models.crm_models import Customer
from ..crm.transformer import convert_customers_to_dataframe

logger = logging.getLogger(__name__)

# Default output directory; created on write by the plot functions below.
DEFAULT_OUTPUT_DIR = Path("visualizations_output")


def plot_customer_distribution_by_status(
    customers: list[Customer], output_dir: Path = DEFAULT_OUTPUT_DIR
) -> str | None:
    """
    Generates a bar chart of customer distribution by status.
    Saves the plot to a file and returns the file path.
    """
    if not customers:
        logger.info("No customer data to plot distribution by status.")
        return None

    df = convert_customers_to_dataframe(customers)
    if df.empty or "status" not in df.columns:
        logger.info("Customer data is empty or 'status' column missing for plotting.")
        return None

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="status", order=df["status"].value_counts().index)
    plt.title("Customer Distribution by Status")
    plt.xlabel("Status")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = output_dir / "customer_status_distribution.png"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path)
        logger.info(f"Saved customer status distribution plot to: {file_path}")
        plt.close()  # Close the plot to free memory
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        plt.close()
        return None


def plot_customer_distribution_by_source(
    customers: list[Customer], output_dir: Path = DEFAULT_OUTPUT_DIR
) -> str | None:
    """
    Generates a bar chart of customer distribution by source.
    Saves the plot to a file and returns the file path.
    """
    if not customers:
        logger.info("No customer data to plot distribution by source.")
        return None

    df = convert_customers_to_dataframe(customers)
    if df.empty or "source" not in df.columns:
        logger.info("Customer data is empty or 'source' column missing for plotting.")
        return None

    plt.figure(figsize=(12, 7))
    sns.countplot(
        data=df,
        y="source",
        order=df["source"].value_counts().index,
        hue="source",
        palette="viridis",
        legend=False,
    )
    plt.title("Customer Distribution by Source")
    plt.xlabel("Number of Customers")
    plt.ylabel("Source")
    plt.tight_layout()

    file_path = output_dir / "customer_source_distribution.png"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path)
        logger.info(f"Saved customer source distribution plot to: {file_path}")
        plt.close()
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        plt.close()
        return None


# Add more CRM visualization functions here, e.g.:
# - Sales pipeline funnel chart
# - Customer acquisition cost over time
# - Lead conversion rates by channel (pie chart or bar chart)

# Example conceptual usage
