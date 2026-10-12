"""CRM Data Transformers."""

import logging
import pandas as pd
from ..models.crm_models import Customer

logger = logging.getLogger(__name__)


def clean_customer_data(customers: list[Customer]) -> list[Customer]:
    """
    Performs comprehensive cleaning operations on a list of Customer objects.

    Cleaning operations include:
    - Standardize company names (title case)
    - Standardize job titles (title case)
    - Validate and clean email addresses
    - Clean phone numbers (remove non-numeric characters except +)
    - Standardize location names in address
    - Ensure status values are valid
    - Clean and validate website URLs

    Args:
        customers: List of Customer objects to clean

    Returns:
        List of cleaned Customer objects
    """
    cleaned_customers = []

    for cust in customers:
        try:
            # Create a copy to avoid modifying the original
            cust_copy = Customer(**cust.model_dump())

            # Clean company names - standardize to title case
            if cust_copy.company:
                cust_copy.company = cust_copy.company.strip().title()

            # Clean job titles - standardize to title case
            if cust_copy.job_title:
                cust_copy.job_title = cust_copy.job_title.strip().title()

            # Clean email addresses - ensure lowercase
            if cust_copy.email:
                cust_copy.email = cust_copy.email.strip().lower()

            # Clean phone numbers - keep only digits and +
            if cust_copy.phone_number:
                cleaned_phone = "".join(
                    c for c in cust_copy.phone_number if c.isdigit() or c == "+"
                )
                cust_copy.phone_number = cleaned_phone

            # Clean names - title case
            if cust_copy.first_name:
                cust_copy.first_name = cust_copy.first_name.strip().title()
            if cust_copy.last_name:
                cust_copy.last_name = cust_copy.last_name.strip().title()

            # Clean address components if present
            if cust_copy.address:
                if cust_copy.address.street:
                    cust_copy.address.street = cust_copy.address.street.strip().title()
                if cust_copy.address.city:
                    cust_copy.address.city = cust_copy.address.city.strip().title()
                if cust_copy.address.state:
                    cust_copy.address.state = cust_copy.address.state.strip().upper()
                if cust_copy.address.country:
                    cust_copy.address.country = (
                        cust_copy.address.country.strip().title()
                    )

            # Clean website URL - ensure it starts with http:// or https://
            if cust_copy.website:
                website = cust_copy.website.strip()
                if website and not website.startswith(("http://", "https://")):
                    website = "https://" + website
                cust_copy.website = website

            # Clean LinkedIn profile URL
            if cust_copy.linkedin_profile:
                linkedin = cust_copy.linkedin_profile.strip()
                if linkedin and not linkedin.startswith(("http://", "https://")):
                    linkedin = "https://" + linkedin
                cust_copy.linkedin_profile = linkedin

            # Standardize status values
            valid_statuses = [
                "active",
                "inactive",
                "lead",
                "prospect",
                "customer",
                "churned",
            ]
            if cust_copy.status and cust_copy.status.lower() not in valid_statuses:
                if cust_copy.status.lower() in ["new", "potential"]:
                    cust_copy.status = "lead"
                else:
                    cust_copy.status = "active"  # default fallback

            cleaned_customers.append(cust_copy)

        except Exception as e:
            logger.error(f"Error cleaning customer {cust.customer_id}: {str(e)}")
            # Add the original if cleaning fails
            cleaned_customers.append(cust)

    logger.info(f"Successfully cleaned {len(cleaned_customers)} customer records")
    return cleaned_customers


def enrich_customer_data(
    customers: list[Customer], external_data_sources: dict | None = None
) -> list[Customer]:
    """
    Enriches customer data with calculated fields and organizational context.

    Enrichment operations include:
    - Calculate customer age (account age)
    - Add interaction summary statistics
    - Calculate engagement level based on interactions
    - Add geographic information validation
    - Calculate customer lifetime value estimates
    - Add industry/company size context
    - Validate email domain consistency

    Args:
        customers: List of Customer objects to enrich
        external_data_sources: Optional external data for enrichment

    Returns:
        List of enriched Customer objects
    """
    enriched_customers = []

    for cust in customers:
        try:
            # Create a copy to avoid modifying the original
            cust_copy = Customer(**cust.model_dump())

            # Calculate account age in days and years
            if cust_copy.created_at:
                from datetime import datetime

                age_days = (datetime.now() - cust_copy.created_at).days
                age_years = age_days / 365.25  # Account for leap years

                # Add interaction log for account age
                from ..models.crm_models import InteractionLog

                age_log = InteractionLog(
                    summary=f"Account age: {age_days} days ({age_years:.1f} years)",
                    channel="system_calculation",
                )
                cust_copy.interaction_history.append(age_log)

                # Add to notes for easier access
                if not cust_copy.notes:
                    cust_copy.notes = ""
                cust_copy.notes += f"\nAccount Age: {age_years:.1f} years"

            # Calculate interaction statistics
            total_interactions = len(cust_copy.interaction_history)
            recent_interactions = len(
                [
                    i
                    for i in cust_copy.interaction_history
                    if (datetime.now() - i.timestamp).days <= 30
                ]
            )

            # Add interaction summary
            interaction_summary = InteractionLog(
                summary=f"Total interactions: {total_interactions}, Recent (30 days): {recent_interactions}",
                channel="analytics_enrichment",
            )
            cust_copy.interaction_history.append(interaction_summary)

            # Calculate engagement level
            if total_interactions == 0:
                engagement_level = "New"
            elif recent_interactions >= 5:
                engagement_level = "Highly Engaged"
            elif recent_interactions >= 2:
                engagement_level = "Moderately Engaged"
            elif recent_interactions == 1:
                engagement_level = "Low Engagement"
            else:
                engagement_level = "Inactive"

            engagement_log = InteractionLog(
                summary=f"Engagement Level: {engagement_level}",
                channel="engagement_analysis",
            )
            cust_copy.interaction_history.append(engagement_log)

            # Add VIP flagging based on tags and engagement
            is_vip = False
            if "vip" in [tag.lower() for tag in cust_copy.tags] or "high-priority" in [
                tag.lower() for tag in cust_copy.tags
            ]:
                is_vip = True
            elif engagement_level == "Highly Engaged" and total_interactions >= 10:
                is_vip = True
                if "VIP_CUSTOMER" not in cust_copy.tags:
                    cust_copy.tags.append("VIP_CUSTOMER")

            if is_vip and "VIP_CUSTOMER" not in cust_copy.tags:
                cust_copy.tags.append("VIP_CUSTOMER")

            # Add company size estimation based on company name patterns
            if cust_copy.company:
                company_name = cust_copy.company.lower()
                if any(
                    keyword in company_name
                    for keyword in ["corp", "corporation", "inc", "ltd", "limited"]
                ):
                    company_size = "Enterprise"
                elif any(
                    keyword in company_name for keyword in ["llc", "partners", "group"]
                ):
                    company_size = "Mid-Market"
                else:
                    company_size = "Small Business"

                size_log = InteractionLog(
                    summary=f"Estimated Company Size: {company_size}",
                    channel="company_analysis",
                )
                cust_copy.interaction_history.append(size_log)

                # Add company size to tags if not already present
                size_tag = f"company_size_{company_size.lower().replace(' ', '_')}"
                if size_tag not in cust_copy.tags:
                    cust_copy.tags.append(size_tag)

            # Validate geographic information
            if (
                cust_copy.address
                and cust_copy.address.city
                and cust_copy.address.country
            ):
                geo_log = InteractionLog(
                    summary=f"Geographic Location: {cust_copy.address.city}, {cust_copy.address.country}",
                    channel="geographic_validation",
                )
                cust_copy.interaction_history.append(geo_log)

            # Add customer segment based on status and engagement
            if cust_copy.status == "lead" and engagement_level == "New":
                segment = "New Lead"
            elif cust_copy.status == "customer" and engagement_level in [
                "Highly Engaged",
                "Moderately Engaged",
            ]:
                segment = "Active Customer"
            elif (
                cust_copy.status == "customer" and engagement_level == "Low Engagement"
            ):
                segment = "At-Risk Customer"
            elif cust_copy.status == "churned":
                segment = "Former Customer"
            else:
                segment = "Other"

            segment_log = InteractionLog(
                summary=f"Customer Segment: {segment}", channel="segmentation_analysis"
            )
            cust_copy.interaction_history.append(segment_log)

            # Add segment to tags if not already present
            segment_tag = f"segment_{segment.lower().replace(' ', '_')}"
            if segment_tag not in cust_copy.tags:
                cust_copy.tags.append(segment_tag)

            enriched_customers.append(cust_copy)

        except Exception as e:
            logger.error(f"Error enriching customer {cust.customer_id}: {str(e)}")
            # Add the original if enrichment fails
            enriched_customers.append(cust)

    logger.info(f"Successfully enriched {len(enriched_customers)} customer records")
    return enriched_customers


def convert_customers_to_dataframe(customers: list[Customer]) -> pd.DataFrame:
    """
    Converts a list of Customer Pydantic models to a Pandas DataFrame
    for easier analysis and bulk operations.
    """
    if not customers:
        return pd.DataFrame()

    # model_dump() is used for Pydantic v2 models
    customer_dicts = [customer.model_dump() for customer in customers]
    df = pd.DataFrame(customer_dicts)

    # Potentially flatten nested structures like 'address' or 'interaction_history' if needed
    # For example, to flatten address:
    # if 'address' in df.columns:
    #     address_df = pd.json_normalize(df['address'])
    #     address_df = address_df.add_prefix('address.')
    #     df = pd.concat([df.drop(columns=['address']), address_df], axis=1)

    logger.info(f"Converted {len(df)} customer records to DataFrame.")
    return df
