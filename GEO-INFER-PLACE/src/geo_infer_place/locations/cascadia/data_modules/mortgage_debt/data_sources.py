"""
Cascadian Mortgage Data Sources

This module is responsible for fetching and processing mortgage data from the
Home Mortgage Disclosure Act (HMDA) database.
"""

from importlib.resources import files

import logging
import os
import pandas as pd
import numpy as np
import requests
import io
import json

logger = logging.getLogger(__name__)


class CascadianMortgageDataSources:
    """Handles fetching and processing of HMDA mortgage data."""

    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), "data", "mortgages")
        os.makedirs(self.data_dir, exist_ok=True)

        config_path = str(
            files("geo_infer_place.locations.cascadia.config").joinpath(
                "data_urls.json"
            )
        )
        try:
            with open(config_path) as f:
                self.config = json.load(f).get("mortgage_debt", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse mortgage_debt config: {e}")
            self.config = {}

        self.hmda_bulk_url_template = self.config.get("hmda_bulk_url")

    def _fetch_hmda_data(self, year: int) -> pd.DataFrame:
        """
        Fetches HMDA LAR data using the Data Browser API (filtered) instead of bulk files.
        """
        # We need to fetch by state or even county to keep it small.
        # Del Norte (CA) is 06015. Curry (OR) is 41015, Josephine (OR) is 41033.
        # Let's try getting data for specific counties if possible, or state subset.
        # The Data Browser API supports 'counties' parameter.

        # Common counties in Cascadia region of interest:
        # Del Norte: 06015
        # Curry: 41015
        # Josephine: 41033
        # Coos: 41011
        # Jackson: 41029
        # Humboldt: 06023
        target_counties = ["06015", "41015", "41033", "41011", "41029", "06023"]

        # Fallback to state level if needing broader coverage, but let's try counties first for speed.

        api_url = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

        # Chunking counties to avoid URL length issues or timeouts
        counties_str = ",".join(target_counties)

        file_path = os.path.join(self.data_dir, f"hmda_{year}_cascadia_subset.csv")

        if os.path.exists(file_path):
            logger.info(f"Loading cached HMDA data from {file_path}")
            return pd.read_csv(file_path, dtype={"census_tract": str})

        params = {
            "counties": counties_str,
            "years": str(year),
            "actions_taken": "1",  # Loan Originated
            "loan_purposes": "1,31,32",  # Home purchase, Refinancing
            "loan_products": "1,2,3",  # Conventional, FHA, VA
        }

        logger.info(f"Fetching filtered HMDA data for counties {counties_str}...")

        try:
            response = requests.get(api_url, params=params, timeout=120)
            response.raise_for_status()

            # Use chunks if large, but filtered should be OK.
            df = pd.read_csv(io.StringIO(response.text), dtype={"census_tract": str})

            if not df.empty:
                df.to_csv(file_path, index=False)
                logger.info(
                    f"Successfully downloaded and cached filtered HMDA data: {len(df)} records."
                )
                return df
            else:
                logger.warning("HMDA API returned empty data.")

        except Exception as e:
            logger.error(f"Error downloading filtered HMDA data: {e}")

        return pd.DataFrame()

    def fetch_all_mortgage_data(self, year: int = 2022) -> pd.DataFrame:
        """
        Loads all available HMDA mortgage data for target counties.

        Args:
            year: The year to fetch data for. Defaults to 2022.

        Returns:
            A DataFrame containing aggregated mortgage data by census tract.
        """
        logger.info(f"Fetching mortgage data for year {year}.")

        hmda_df = self._fetch_hmda_data(year)

        if hmda_df.empty:
            logger.warning("No mortgage data found; returning an empty typed result.")
            return pd.DataFrame(columns=["census_tract", "loan_to_value_ratio"])

        # Clean and process the data
        # Select only relevant columns - ensure they exist
        required_cols = ["census_tract", "loan_amount", "property_value", "income"]
        available_cols = [c for c in required_cols if c in hmda_df.columns]

        if len(available_cols) < len(required_cols):
            logger.warning(f"HMDA data missing columns. Found: {available_cols}")
            return pd.DataFrame(columns=["census_tract", "loan_to_value_ratio"])

        hmda_df = hmda_df[required_cols].copy()

        # Convert to numeric, coercing errors
        for col in ["loan_amount", "property_value", "income"]:
            hmda_df[col] = pd.to_numeric(hmda_df[col], errors="coerce")

        # Drop rows where key financial data is missing
        hmda_df.dropna(
            subset=["loan_amount", "property_value", "census_tract"], inplace=True
        )

        # Filter out nonsensical values
        hmda_df = hmda_df[hmda_df["loan_amount"] > 1000]
        hmda_df = hmda_df[hmda_df["property_value"] > 1000]

        if hmda_df.empty:
            logger.warning("No valid mortgage records after cleaning.")
            return pd.DataFrame(columns=["census_tract", "loan_to_value_ratio"])

        logger.info(
            f"Aggregating HMDA data by census tract ({len(hmda_df)} records)..."
        )

        # Aggregate by census tract
        agg_df = (
            hmda_df.groupby("census_tract")
            .agg(
                total_loan_volume=("loan_amount", "sum"),
                average_loan_amount=("loan_amount", "mean"),
                average_property_value=("property_value", "mean"),
                average_income=("income", "mean"),
                number_of_loans=("loan_amount", "count"),
            )
            .reset_index()
        )

        # Calculate Loan to Value Ratio: Sum(Loan Amount) / Sum(Property Value) per tract
        tract_sums = (
            hmda_df.groupby("census_tract")[["loan_amount", "property_value"]]
            .sum()
            .reset_index()
        )
        agg_df["loan_to_value_ratio"] = (
            tract_sums["loan_amount"] / tract_sums["property_value"]
        )

        agg_df["loan_to_value_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

        logger.info(
            f"Successfully processed and aggregated mortgage data for {len(agg_df)} census tracts."
        )
        return agg_df
