#!/usr/bin/env python3
"""
Command-line interface for GEO-INFER-SEC.

This module provides command-line access to security and privacy
features for geospatial data.
"""

import sys
import json
import argparse
import logging
import datetime
from typing import cast
from pathlib import Path
import pandas as pd
import geopandas as gpd

from geo_infer_sec.core.anonymization import GeospatialAnonymizer
from geo_infer_sec.core.compliance import (
    ComplianceFramework,
    ComplianceRegime,
    create_gdpr_validators,
)
from geo_infer_sec.core.encryption import GeospatialEncryption
from geo_infer_sec.utils.security_utils import (
    generate_secure_token,
    check_pii_columns,
    validate_spatial_bounds,
)
from geo_infer_sec.models.risk_assessment import (
    RiskAssessment,
    create_common_geospatial_risks,
)


logger = logging.getLogger("geo_infer_sec")


def configure_logging(verbose: bool = False) -> None:
    """Configure CLI logging after argument parsing."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(level)


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the command-line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="GEO-INFER-SEC: Security and privacy tools for geospatial data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Anonymize command
    anonymize_parser = subparsers.add_parser(
        "anonymize", help="Anonymize geospatial data"
    )
    anonymize_parser.add_argument("input_file", help="Input GeoJSON or Shapefile")
    anonymize_parser.add_argument("output_file", help="Output file path")
    anonymize_parser.add_argument(
        "--method",
        choices=["perturbation", "k-anonymity", "geographic-masking"],
        default="perturbation",
        help="Anonymization method to use",
    )
    anonymize_parser.add_argument(
        "--epsilon",
        type=float,
        default=100.0,
        help="Maximum distance (in meters) for perturbation",
    )
    anonymize_parser.add_argument(
        "--k", type=int, default=5, help="K value for k-anonymity"
    )
    anonymize_parser.add_argument(
        "--h3-resolution",
        type=int,
        default=9,
        help="H3 resolution for k-anonymity (0-15)",
    )
    anonymize_parser.add_argument(
        "--admin-boundaries", help="Admin boundaries file for geographic masking"
    )
    anonymize_parser.add_argument(
        "--attribute-cols", help="Comma-separated list of attribute columns to preserve"
    )

    # Encrypt command
    encrypt_parser = subparsers.add_parser(
        "encrypt", help="Encrypt sensitive geospatial data"
    )
    encrypt_parser.add_argument("input_file", help="Input GeoJSON or CSV file")
    encrypt_parser.add_argument("output_file", help="Output file path")
    encrypt_parser.add_argument("--password", help="Password for encryption")
    encrypt_parser.add_argument("--key-file", help="File to save the encryption key")
    encrypt_parser.add_argument(
        "--columns", help="Comma-separated list of columns to encrypt"
    )
    encrypt_parser.add_argument(
        "--encrypt-geometry",
        action="store_true",
        help="Whether to encrypt geometry coordinates",
    )

    # Decrypt command
    decrypt_parser = subparsers.add_parser(
        "decrypt", help="Decrypt encrypted geospatial data"
    )
    decrypt_parser.add_argument("input_file", help="Input encrypted file")
    decrypt_parser.add_argument("output_file", help="Output file path")
    decrypt_parser.add_argument("--password", help="Password for decryption")
    decrypt_parser.add_argument("--key-file", help="File with the encryption key")
    decrypt_parser.add_argument(
        "--columns", help="Comma-separated list of columns to decrypt"
    )

    # Check compliance command
    compliance_parser = subparsers.add_parser(
        "check-compliance", help="Check data compliance with privacy regulations"
    )
    compliance_parser.add_argument("input_file", help="Input GeoJSON or CSV file")
    compliance_parser.add_argument(
        "--output-file", help="Output file for compliance report"
    )
    compliance_parser.add_argument(
        "--regimes",
        default="gdpr",
        help="Comma-separated list of compliance regimes to check (gdpr, ccpa, hipaa)",
    )
    compliance_parser.add_argument(
        "--format",
        choices=["json", "html", "text"],
        default="json",
        help="Output format for the compliance report",
    )

    # Audit command
    audit_parser = subparsers.add_parser(
        "audit", help="Audit a file for security and privacy concerns"
    )
    audit_parser.add_argument("input_file", help="Input file to audit")
    audit_parser.add_argument("--output-file", help="Output file for audit report")
    audit_parser.add_argument(
        "--check-pii", action="store_true", help="Check for potential PII columns"
    )
    audit_parser.add_argument(
        "--check-bounds", action="store_true", help="Validate spatial bounds"
    )
    audit_parser.add_argument(
        "--detect-outliers",
        action="store_true",
        help="Detect potential outliers in the data",
    )

    # Risk assessment command
    risk_parser = subparsers.add_parser(
        "risk-assessment", help="Generate security risk assessment"
    )
    risk_parser.add_argument("--output-file", help="Output file for risk report")
    risk_parser.add_argument(
        "--name",
        default="Geospatial Security Risk Assessment",
        help="Name of the risk assessment",
    )
    risk_parser.add_argument(
        "--format",
        choices=["json", "html", "text"],
        default="html",
        help="Output format for the risk report",
    )

    # Generate token command
    token_parser = subparsers.add_parser(
        "generate-token", help="Generate a secure token"
    )
    token_parser.add_argument(
        "--length", type=int, default=32, help="Token length in bytes"
    )

    # Help
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser


def load_geospatial_data(file_path: str) -> gpd.GeoDataFrame:
    """
    Load geospatial data from a file.

    Args:
        file_path: Path to the data file (GeoJSON, Shapefile, etc.)

    Returns:
        GeoDataFrame with the data
    """
    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == ".geojson" or file_ext == ".json":
            return gpd.read_file(file_path)
        elif file_ext == ".shp":
            return gpd.read_file(file_path)
        elif file_ext == ".csv":
            # For CSV, we'll try to parse as a regular DataFrame first
            df = pd.read_csv(file_path)

            # Check if it has geometry columns
            if "latitude" in df.columns and "longitude" in df.columns:
                from shapely.geometry import Point

                geometry = [
                    Point(lon, lat) for lon, lat in zip(df.longitude, df.latitude)
                ]
                return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            elif "lat" in df.columns and "lon" in df.columns:
                from shapely.geometry import Point

                geometry = [Point(lon, lat) for lon, lat in zip(df.lon, df.lat)]
                return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            else:
                # Return as a regular DataFrame
                return df
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def save_geospatial_data(gdf: gpd.GeoDataFrame | pd.DataFrame, file_path: str) -> None:
    """
    Save geospatial data to a file.

    Args:
        gdf: GeoDataFrame or DataFrame to save
        file_path: Path to save the data
    """
    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == ".geojson" or file_ext == ".json":
            if isinstance(gdf, gpd.GeoDataFrame):
                gdf.to_file(file_path, driver="GeoJSON")
            else:
                with open(file_path, "w") as f:
                    json.dump(gdf.to_dict(orient="records"), f, indent=2)
        elif file_ext == ".shp":
            if isinstance(gdf, gpd.GeoDataFrame):
                gdf.to_file(file_path)
            else:
                raise ValueError("Cannot save regular DataFrame as Shapefile")
        elif file_ext == ".csv":
            gdf.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {file_ext}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        raise


def command_anonymize(args: argparse.Namespace) -> bool:
    """
    Execute the anonymize command.

    Args:
        args: Command-line arguments

    Returns:
        True on success, False on handled failure.
    """
    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    try:
        gdf = load_geospatial_data(args.input_file)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

    if not isinstance(gdf, gpd.GeoDataFrame):
        logger.error("Input file must contain geospatial data")
        return False

    # Create anonymizer
    anonymizer = GeospatialAnonymizer()

    # Apply anonymization
    logger.info(f"Applying {args.method} anonymization")

    try:
        if args.method == "perturbation":
            result = anonymizer.location_perturbation(gdf, epsilon=args.epsilon)
            logger.info(f"Applied perturbation with epsilon={args.epsilon} meters")

        elif args.method == "k-anonymity":
            result = anonymizer.spatial_k_anonymity(
                gdf, k=args.k, h3_resolution=args.h3_resolution
            )
            logger.info(
                f"Applied k-anonymity with k={args.k}, h3_resolution={args.h3_resolution}"
            )

        elif args.method == "geographic-masking":
            if not args.admin_boundaries:
                logger.error("Admin boundaries file is required for geographic masking")
                return False

            if not args.attribute_cols:
                logger.error(
                    "Attribute columns must be specified for geographic masking"
                )
                return False

            # Load admin boundaries
            admin_boundaries = load_geospatial_data(args.admin_boundaries)

            # Get attribute columns
            attribute_cols = args.attribute_cols.split(",")

            result = anonymizer.geographic_masking(
                gdf, attribute_cols=attribute_cols, admin_boundaries=admin_boundaries
            )
            logger.info(
                f"Applied geographic masking with {len(admin_boundaries)} admin areas"
            )

        else:
            logger.error(f"Unknown anonymization method: {args.method}")
            return False
    except Exception as e:
        logger.error(f"Error during anonymization: {str(e)}")
        return False

    # Save results
    logger.info(f"Saving anonymized data to {args.output_file}")
    save_geospatial_data(result, args.output_file)
    return True


def command_encrypt(args: argparse.Namespace) -> bool:
    """
    Execute the encrypt command.

    Args:
        args: Command-line arguments

    Returns:
        True on success, False on handled failure.
    """
    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    try:
        df = load_geospatial_data(args.input_file)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

    # Initialize encryptor
    if args.password:
        logger.info("Creating encryption key from password")
        encryptor = GeospatialEncryption.from_password(args.password)
    else:
        logger.info("Generating new encryption key")
        encryptor = GeospatialEncryption()

    # Get columns to encrypt
    if args.columns:
        encrypt_columns = args.columns.split(",")
        logger.info(f"Will encrypt columns: {', '.join(encrypt_columns)}")
    else:
        # If no columns specified, encrypt all non-geometry columns
        encrypt_columns = [col for col in df.columns if col != "geometry"]
        logger.info(
            f"No columns specified, will encrypt all {len(encrypt_columns)} non-geometry columns"
        )

    try:
        # Encrypt data
        result = encryptor.encrypt_geodataframe(
            df,
            sensitive_columns=encrypt_columns,
            encrypt_coordinates=args.encrypt_geometry,
        )

        logger.info(f"Encrypted {len(encrypt_columns)} columns")
        if args.encrypt_geometry:
            logger.info("Encrypted geometry coordinates")

        # Save key if requested
        if args.key_file:
            with open(args.key_file, "wb") as f:
                f.write(encryptor.get_key())
            logger.info(f"Saved encryption key to {args.key_file}")

        # Save encrypted data
        logger.info(f"Saving encrypted data to {args.output_file}")
        save_geospatial_data(result, args.output_file)
        return True

    except Exception as e:
        logger.error(f"Error during encryption: {str(e)}")
        return False


def command_decrypt(args: argparse.Namespace) -> bool:
    """
    Execute the decrypt command.

    Args:
        args: Command-line arguments

    Returns:
        True on success, False on handled failure.
    """
    # Load input data
    logger.info(f"Loading encrypted data from {args.input_file}")
    try:
        df = load_geospatial_data(args.input_file)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

    # Get encryption key
    if args.key_file:
        logger.info(f"Loading encryption key from {args.key_file}")
        with open(args.key_file, "rb") as f:
            key = f.read()
        encryptor = GeospatialEncryption(key)
    elif args.password:
        logger.info("Creating encryption key from password")
        # Note: This requires the same salt used for encryption
        # In a real application, we would store and retrieve the salt
        # For this example, we'll just show the approach
        encryptor = GeospatialEncryption.from_password(args.password)
    else:
        logger.error("Either a key file or password must be provided")
        return False

    # Get columns to decrypt
    if args.columns:
        decrypt_columns = args.columns.split(",")
        logger.info(f"Will decrypt columns: {', '.join(decrypt_columns)}")
    else:
        # If no columns specified, look for columns with _encrypted suffix
        decrypt_columns = [col for col in df.columns if col.endswith("_encrypted")]
        logger.info(
            f"No columns specified, will decrypt {len(decrypt_columns)} columns with _encrypted suffix"
        )

    try:
        # Check if we need to decrypt geometry
        geometry_col = None
        if "encrypted_geometry" in df.columns:
            geometry_col = "encrypted_geometry"

        # Decrypt data
        result = encryptor.decrypt_geodataframe(
            df, encrypted_columns=decrypt_columns, geometry_col=cast(str, geometry_col)
        )

        logger.info(f"Decrypted {len(decrypt_columns)} columns")
        if geometry_col:
            logger.info("Decrypted geometry coordinates")

        # Save decrypted data
        logger.info(f"Saving decrypted data to {args.output_file}")
        save_geospatial_data(result, args.output_file)
        return True

    except Exception as e:
        logger.error(f"Error during decryption: {str(e)}")
        return False


def command_check_compliance(args: argparse.Namespace) -> bool:
    """
    Execute the check-compliance command.

    Args:
        args: Command-line arguments

    Returns:
        True on success, False on handled failure.
    """
    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    try:
        df = load_geospatial_data(args.input_file)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

    # Create compliance framework
    framework = ComplianceFramework()

    # Add rules based on specified regimes
    regimes = args.regimes.split(",")
    logger.info(f"Checking compliance with regimes: {', '.join(regimes)}")

    for regime in regimes:
        if regime.lower() == "gdpr":
            # Add GDPR rules
            gdpr_rules = create_gdpr_validators()
            for rule in gdpr_rules.values():
                framework.add_rule(rule)
                logger.info(f"Added GDPR rule: {rule.name}")

    try:
        # Convert to GeoDataFrame if needed
        if not isinstance(df, gpd.GeoDataFrame) and isinstance(df, pd.DataFrame):
            from shapely.geometry import Point

            if not {"latitude", "longitude"}.issubset(df.columns):
                raise ValueError(
                    "tabular compliance input must contain latitude and longitude columns"
                )
            geometry = [
                Point(lon, lat) for lat, lon in zip(df["latitude"], df["longitude"])
            ]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
        else:
            gdf = df

        # Check compliance
        regime_enums = [
            getattr(ComplianceRegime, regime.upper())
            for regime in regimes
            if hasattr(ComplianceRegime, regime.upper())
        ]

        violations = framework.check_geodataframe_compliance(
            gdf, data_reference=args.input_file, regimes=regime_enums
        )

        # Generate report
        if len(violations) == 0:
            logger.info("No compliance violations found")
        else:
            logger.warning(f"Found {len(violations)} compliance violations")

        report = framework.generate_compliance_report(
            output_format=args.format,
            output_file=args.output_file if args.output_file else None,
        )

        # Print report if no output file specified
        if not args.output_file:
            if args.format == "json":
                print(json.dumps(json.loads(cast(str, report)), indent=2))
            else:
                print(report)
        else:
            logger.info(f"Saved compliance report to {args.output_file}")

    except Exception as e:
        logger.error(f"Error during compliance check: {str(e)}")
        return False

    return True


def command_audit(args: argparse.Namespace) -> bool:
    """
    Execute the audit command.

    Args:
        args: Command-line arguments

    Returns:
        True on success, False on handled failure.
    """
    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    try:
        df = load_geospatial_data(args.input_file)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

    audit_results = {
        "file": args.input_file,
        "timestamp": datetime.datetime.now(datetime.UTC)
        .replace(tzinfo=None)
        .isoformat(),
        "record_count": len(df),
        "columns": list(df.columns),
        "issues": [],
    }

    # Check for PII
    if args.check_pii:
        logger.info("Checking for potential PII columns")
        pii_columns = check_pii_columns(df)

        if pii_columns:
            audit_results["issues"].append(
                {
                    "type": "pii_columns",
                    "severity": "medium",
                    "description": f"Found {len(pii_columns)} potential PII columns",
                    "details": pii_columns,
                }
            )
            logger.warning(f"Found potential PII columns: {', '.join(pii_columns)}")

    # Check spatial bounds
    if args.check_bounds and isinstance(df, gpd.GeoDataFrame):
        logger.info("Validating spatial bounds")
        valid_bounds = validate_spatial_bounds(df)

        if not valid_bounds:
            audit_results["issues"].append(
                {
                    "type": "invalid_bounds",
                    "severity": "high",
                    "description": "Found geometries outside valid WGS84 bounds",
                    "details": None,
                }
            )
            logger.warning("Found geometries outside valid WGS84 bounds")

    # Detect outliers
    if args.detect_outliers and isinstance(df, gpd.GeoDataFrame):
        from geo_infer_sec.utils.security_utils import detect_outliers

        logger.info("Detecting potential outliers")

        outlier_result = detect_outliers(df)
        outlier_count = outlier_result["is_outlier"].sum()

        if outlier_count > 0:
            audit_results["issues"].append(
                {
                    "type": "outliers",
                    "severity": "medium",
                    "description": f"Found {outlier_count} potential outliers",
                    "details": {
                        "count": int(outlier_count),
                        "percentage": float(outlier_count) / len(df) * 100,
                    },
                }
            )
            logger.warning(
                f"Found {outlier_count} potential outliers ({outlier_count / len(df) * 100:.1f}%)"
            )

    # Save or print audit results
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(audit_results, f, indent=2)
        logger.info(f"Saved audit report to {args.output_file}")
    else:
        print(json.dumps(audit_results, indent=2))

    # Log summary
    issue_count = len(audit_results["issues"])
    if issue_count == 0:
        logger.info("Audit completed: No issues found")
    else:
        logger.info(f"Audit completed: Found {issue_count} issues")

    return True


def command_risk_assessment(args: argparse.Namespace) -> bool:
    """
    Execute the risk-assessment command.

    Args:
        args: Command-line arguments
    """
    logger.info("Generating security risk assessment")

    # Create risk assessment
    assessment = RiskAssessment(
        name=args.name,
        description="Security risk assessment for geospatial data and applications",
    )

    # Add common risks
    common_risks = create_common_geospatial_risks()
    for risk in common_risks:
        assessment.add_risk(risk)

    logger.info(f"Added {len(common_risks)} common geospatial security risks")

    # Generate report
    report = assessment.generate_risk_report(format=args.format)

    # Save or print report
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(report)
        logger.info(f"Saved risk assessment report to {args.output_file}")
    else:
        print(report)

    logger.info("Risk assessment completed")

    return True


def command_generate_token(args: argparse.Namespace) -> bool:
    """
    Execute the generate-token command.

    Args:
        args: Command-line arguments
    """
    token = generate_secure_token(args.length)
    print(f"Secure token: {token}")
    return True


def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    if args.verbose:
        logger.debug("Verbose output enabled")

    # Execute the selected command; handlers report success so scripted
    # pipelines can detect failure via the exit code.
    handlers = {
        "anonymize": command_anonymize,
        "encrypt": command_encrypt,
        "decrypt": command_decrypt,
        "check-compliance": command_check_compliance,
        "audit": command_audit,
        "risk-assessment": command_risk_assessment,
        "generate-token": command_generate_token,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return
    sys.exit(0 if handler(args) else 1)


if __name__ == "__main__":
    main()
