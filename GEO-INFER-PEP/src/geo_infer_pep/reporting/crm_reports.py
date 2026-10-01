"""CRM Reporting functions."""

import logging
from typing import Any
from ..models.crm_models import Customer
from ..crm.transformer import (
    convert_customers_to_dataframe,
)  # Assuming this function exists

logger = logging.getLogger(__name__)


def generate_customer_segmentation_report(customers: list[Customer]) -> dict[str, Any]:
    """
    Generates a report on customer segmentation.
    Example: Counts customers by status, source, or custom tags.
    """
    if not customers:
        return {"message": "No customer data to report."}

    df = convert_customers_to_dataframe(customers)
    if df.empty:
        return {"message": "Customer data is empty after conversion to DataFrame."}

    report = {}

    if "status" in df.columns:
        report["customers_by_status"] = df["status"].value_counts().to_dict()

    if "source" in df.columns:
        report["customers_by_source"] = df["source"].value_counts().to_dict()

    # Example: Segmentation by a common tag like 'VIP_CUSTOMER' (created during enrichment)
    if "tags" in df.columns:
        # Explode tags if they are lists, then count
        # This assumes 'tags' column contains lists of strings
        try:
            all_tags = df["tags"].explode()
            report["customers_by_tag"] = all_tags.value_counts().to_dict()
            if "VIP_CUSTOMER" in all_tags.values:
                report["vip_customer_count"] = int(
                    all_tags[all_tags == "VIP_CUSTOMER"].count()
                )
            else:
                report["vip_customer_count"] = 0
        except Exception as e:
            logger.warning(f"Could not process tags for reporting: {e}")
            report["tags_processing_error"] = str(e)

    report["total_customers"] = len(df)
    logger.info("Generated customer segmentation report.")
    return report


def generate_lead_conversion_report(customers: list[Customer]) -> dict[str, Any]:
    """
    Generates a report on lead conversion rates.
    Requires 'status' and potentially 'created_at' or 'updated_at' fields.
    (This is a simplified example)
    """
    if not customers:
        return {"message": "No customer data for lead conversion report."}

    df = convert_customers_to_dataframe(customers)
    if df.empty:
        return {"message": "Customer data is empty after conversion to DataFrame."}

    report = {}
    if "status" not in df.columns:
        return {
            "message": "'status' column missing, cannot generate lead conversion report."
        }

    total_leads = df[df["status"] == "lead"].shape[0]
    converted_customers = df[df["status"] == "active_customer"].shape[
        0
    ]  # Simplified definition of "converted"

    report["total_identified_leads"] = total_leads
    report["total_converted_customers"] = converted_customers

    if total_leads > 0:
        report["lead_to_customer_conversion_rate"] = (
            converted_customers / total_leads
        ) * 100
    else:
        report["lead_to_customer_conversion_rate"] = 0.0

    logger.info("Generated lead conversion report.")
    return report


def get_quarterly_metrics(
    quarter: str, year: int, customers: list[Customer] | None = None
) -> dict[str, Any]:
    """Calculate CRM quarterly metrics from customer records."""
    if not customers:
        return {
            "quarter": quarter,
            "year": year,
            "message": "No customer data available for metrics calculation",
            "new_leads_acquired": 0,
            "customers_converted": 0,
            "churned_customers": 0,
            "conversion_rate_percent": None,
        }

    customer_df = convert_customers_to_dataframe(customers)
    if customer_df.empty or "status" not in customer_df:
        return {
            "quarter": quarter,
            "year": year,
            "message": "Customer data has no usable status records",
            "new_leads_acquired": 0,
            "customers_converted": 0,
            "churned_customers": 0,
            "conversion_rate_percent": None,
        }

    leads = int((customer_df["status"] == "lead").sum())
    converted = int((customer_df["status"] == "active_customer").sum())
    churned = int((customer_df["status"] == "churned").sum())
    return {
        "quarter": quarter,
        "year": year,
        "new_leads_acquired": leads,
        "customers_converted": converted,
        "churned_customers": churned,
        "conversion_rate_percent": (converted / leads * 100) if leads else None,
    }


# Add more CRM-specific reporting functions here, e.g.:
# - Sales pipeline analysis
# - Customer activity summary
# - Churn rate analysis
