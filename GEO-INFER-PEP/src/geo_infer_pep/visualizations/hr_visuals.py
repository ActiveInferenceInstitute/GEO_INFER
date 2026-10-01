"""HR Data Visualization functions."""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..models.hr_models import Employee, EmploymentStatus
from ..hr.transformer import convert_employees_to_dataframe

logger = logging.getLogger(__name__)

# Default HR output directory; created on write by the plot functions below.
DEFAULT_HR_VISUALS_DIR = Path("visualizations_output/hr")


def plot_headcount_by_department(
    employees: list[Employee], output_dir: Path = DEFAULT_HR_VISUALS_DIR
) -> str | None:
    """
    Generates a bar chart of active employee headcount by department.
    Saves the plot and returns its path.
    """
    if not employees:
        logger.info("No employee data for headcount by department plot.")
        return None

    df = convert_employees_to_dataframe(employees)
    active_df = df[df["employment_status"] == EmploymentStatus.ACTIVE]

    if active_df.empty or "department" not in active_df.columns:
        logger.info("No active employee data or 'department' column missing.")
        return None

    plt.figure(figsize=(12, 7))
    sns.countplot(
        data=active_df,
        y="department",
        order=active_df["department"].value_counts().index,
        hue="department",
        palette="crest",
        legend=False,
    )
    plt.title("Active Employee Headcount by Department")
    plt.xlabel("Number of Active Employees")
    plt.ylabel("Department")
    plt.tight_layout()

    file_path = output_dir / "headcount_by_department.png"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path)
        logger.info(f"Saved headcount by department plot to: {file_path}")
        plt.close()
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        plt.close()
        return None


def plot_gender_distribution(
    employees: list[Employee], output_dir: Path = DEFAULT_HR_VISUALS_DIR
) -> str | None:
    """
    Generates a pie chart for gender distribution of active employees.
    (Consider ethical implications and alternatives for diversity visualization).
    """
    if not employees:
        logger.info("No employee data for gender distribution plot.")
        return None

    df = convert_employees_to_dataframe(employees)
    active_df = df[df["employment_status"] == EmploymentStatus.ACTIVE]

    if active_df.empty or "gender" not in active_df.columns:
        logger.info("No active employee data or 'gender' column missing.")
        return None

    gender_counts = active_df["gender"].value_counts()
    if gender_counts.empty:
        logger.info("No gender data to plot.")
        return None

    plt.figure(figsize=(8, 8))
    plt.pie(
        gender_counts,
        labels=gender_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Gender Distribution of Active Employees")
    plt.tight_layout()

    file_path = output_dir / "gender_distribution.png"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path)
        logger.info(f"Saved gender distribution plot to: {file_path}")
        plt.close()
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        plt.close()
        return None


# Add more HR visualization functions here, e.g.:
# - Tenure distribution (histogram)
# - Attrition trends over time (line chart)
# - Compensation distribution by job level (box plot)
