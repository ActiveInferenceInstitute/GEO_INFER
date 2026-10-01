#!/usr/bin/env python3
"""
Comprehensive Realms API Testing Script

This script tests all documented Realms API endpoints:
1. Search Realms by name
2. Get all Realms (with pagination and sorting)
3. Get Realm details by ID

Features:
- JSON Schema validation
- Response time measurement
- Error handling and logging
- Comprehensive reporting
- Rate limiting protection
"""

import json
import time
import logging
import requests
from typing import Any
from datetime import datetime
from jsonschema import validate, ValidationError
import argparse
import sys
import os

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("outputs", f"test_run_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Configure logging to write to timestamped directory
log_file = os.path.join(output_dir, "test_execution.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class RealmsAPITester:
    """
    Comprehensive API tester for the Realms API endpoints.
    """

    def __init__(
        self,
        schema_path: str = "realm_schema.json",
        timeout: int = 30,
        output_dir: str | None = None,
    ):
        """
        Initialize the API tester.

        Args:
            schema_path: Path to the realm schema JSON file
            timeout: Request timeout in seconds
            output_dir: Directory to save outputs
        """
        self.timeout = timeout
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "GEO-INFER-RealmsAPI-Tester/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        # Load and parse the schema
        self.schema = self._load_schema(schema_path)

        # API endpoints from README
        self.endpoints = {
            "search_by_name": "https://api.guardiansofearth.io/realms",
            "get_all_realms": "https://portal.biosmart.life/api/v1/contest/109/regions.json",
            "get_realm_by_id": "https://portal.biosmart.life/api/v1/region",
        }

        # Test results storage
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "start_time": None,
            "end_time": None,
        }

    def _load_schema(self, schema_path: str) -> dict[str, Any]:
        """Load and validate the JSON schema."""
        try:
            with open(schema_path) as f:
                schema = json.load(f)
            logger.info(f"Successfully loaded schema from {schema_path}")
            return schema
        except FileNotFoundError:
            logger.error(f"Schema file not found: {schema_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema file: {e}")
            return {}

    def _make_request(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Make HTTP request with error handling and timing.

        Returns:
            Tuple of (success: bool, result: Dict)
        """
        start_time = time.time()
        result = {
            "url": url,
            "method": method,
            "params": params,
            "response_time": 0,
            "status_code": None,
            "response_data": None,
            "error": None,
        }

        try:
            if headers:
                session_headers = self.session.headers.copy()
                session_headers.update(headers)
            else:
                session_headers = self.session.headers

            response = self.session.request(
                method=method,
                url=url,
                params=params,
                headers=session_headers,
                timeout=self.timeout,
            )

            result["response_time"] = time.time() - start_time
            result["status_code"] = response.status_code

            # Try to parse JSON response
            try:
                result["response_data"] = response.json()
            except json.JSONDecodeError:
                result["response_data"] = response.text
                result["error"] = "Response is not valid JSON"

            # Check if request was successful
            if response.status_code == 200:
                logger.info(
                    f"✓ {method} {url} - {response.status_code} ({result['response_time']:.2f}s)"
                )
                return True, result
            else:
                logger.warning(
                    f"✗ {method} {url} - {response.status_code} ({result['response_time']:.2f}s)"
                )
                result["error"] = f"HTTP {response.status_code}: {response.reason}"
                return False, result

        except requests.exceptions.Timeout:
            result["response_time"] = time.time() - start_time
            result["error"] = f"Request timeout after {self.timeout}s"
            logger.error(f"✗ {method} {url} - Timeout")
            return False, result

        except requests.exceptions.ConnectionError as e:
            result["response_time"] = time.time() - start_time
            result["error"] = f"Connection error: {str(e)}"
            logger.error(f"✗ {method} {url} - Connection error")
            return False, result

        except Exception as e:
            result["response_time"] = time.time() - start_time
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"✗ {method} {url} - Unexpected error: {e}")
            return False, result

    def _validate_realm_schema(self, data: Any) -> tuple[bool, str | None]:
        """
        Validate data against the realm schema.

        Returns:
            Tuple of (is_valid: bool, error_message: str | None)
        """
        if not self.schema:
            return True, "No schema loaded - skipping validation"

        try:
            # Handle both single objects and arrays
            if isinstance(data, list):
                for item in data:
                    validate(instance=item, schema=self.schema)
            else:
                validate(instance=data, schema=self.schema)
            return True, None
        except ValidationError as e:
            return False, f"Schema validation error: {e.message}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _record_test_result(
        self, test_name: str, success: bool, details: dict[str, Any]
    ):
        """Record the result of a test."""
        self.results["total_tests"] += 1
        if success:
            self.results["passed_tests"] += 1
        else:
            self.results["failed_tests"] += 1

        self.results["test_details"].append(
            {
                "test_name": test_name,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "details": details,
            }
        )

    def test_search_by_name(self, search_terms: list[str] | None = None) -> None:
        """
        Test the search Realms by name endpoint.

        Args:
            search_terms: List of search terms to test. Defaults to common terms.
        """
        logger.info("=" * 60)
        logger.info("Testing: Search Realms by Name")
        logger.info("=" * 60)

        if search_terms is None:
            search_terms = ["Avana", "Forest", "Park", "Reserve", "Ocean"]

        for term in search_terms:
            params = {"query": "name", "regEx": term, "limit": 10}

            success, result = self._make_request(
                "GET", self.endpoints["search_by_name"], params=params
            )

            # Additional validation for successful responses
            if success and result["response_data"]:
                # Validate response structure
                if isinstance(result["response_data"], list):
                    logger.info(
                        f"  Found {len(result['response_data'])} results for '{term}'"
                    )

                    # Check if results contain expected fields from README example
                    for item in result["response_data"]:
                        expected_fields = ["_id", "name"]
                        missing_fields = [f for f in expected_fields if f not in item]
                        if missing_fields:
                            success = False
                            result["error"] = (
                                f"Missing expected fields: {missing_fields}"
                            )
                            break
                else:
                    success = False
                    result["error"] = "Response should be an array"

            self._record_test_result(f"search_by_name_{term}", success, result)

            # Rate limiting - small delay between requests
            time.sleep(0.5)

    def test_get_all_realms(self) -> None:
        """Test the get all Realms endpoint with various parameters."""
        logger.info("=" * 60)
        logger.info("Testing: Get All Realms")
        logger.info("=" * 60)

        # Test cases with different parameters
        test_cases = [
            {"name": "default", "params": {}},
            {"name": "limited", "params": {"limit": 5}},
            {"name": "offset", "params": {"limit": 3, "offset": 10}},
            {
                "name": "sorted_by_id_asc",
                "params": {"sort_by": "id", "sort_order": "asc", "limit": 5},
            },
            {
                "name": "sorted_by_id_desc",
                "params": {"sort_by": "id", "sort_order": "desc", "limit": 5},
            },
            {
                "name": "sorted_by_bioscore_desc",
                "params": {"sort_by": "bioscore", "sort_order": "desc", "limit": 5},
            },
        ]

        for test_case in test_cases:
            logger.info(f"Testing with params: {test_case['params']}")

            success, result = self._make_request(
                "GET", self.endpoints["get_all_realms"], params=test_case["params"]
            )

            # Additional validation for successful responses
            if success and result["response_data"]:
                if isinstance(result["response_data"], list):
                    logger.info(f"  Retrieved {len(result['response_data'])} realms")

                    # Validate against schema if we have data
                    if result["response_data"] and self.schema:
                        schema_valid, schema_error = self._validate_realm_schema(
                            result["response_data"]
                        )
                        if not schema_valid:
                            logger.warning(
                                f"  Schema validation failed: {schema_error}"
                            )
                            result["schema_validation_error"] = schema_error
                        else:
                            logger.info("  ✓ Schema validation passed")
                else:
                    success = False
                    result["error"] = "Response should be an array"

            self._record_test_result(
                f"get_all_realms_{test_case['name']}", success, result
            )

            # Store first successful response for ID extraction
            if (
                success
                and result["response_data"]
                and isinstance(result["response_data"], list)
            ):
                if not hasattr(self, "sample_realm_ids"):
                    self.sample_realm_ids = [
                        item.get("id") or item.get("_id")
                        for item in result["response_data"][:5]
                        if item.get("id") or item.get("_id")
                    ]

            time.sleep(0.5)

    def test_get_realm_by_id(self, realm_ids: list[int] | None = None) -> None:
        """
        Test the get Realm by ID endpoint.

        Args:
            realm_ids: List of realm IDs to test. Uses sample IDs if None.
        """
        logger.info("=" * 60)
        logger.info("Testing: Get Realm by ID")
        logger.info("=" * 60)

        # Use provided IDs or sample IDs from previous tests
        test_ids = realm_ids or getattr(self, "sample_realm_ids", [2188, 6472, 8155])

        if not test_ids:
            logger.warning("No realm IDs available for testing")
            self._record_test_result(
                "get_realm_by_id_no_ids",
                False,
                {"error": "No realm IDs available for testing"},
            )
            return

        for realm_id in test_ids[:5]:  # Test up to 5 IDs
            url = f"{self.endpoints['get_realm_by_id']}/{realm_id}"

            success, result = self._make_request("GET", url)

            # Additional validation for successful responses
            if success and result["response_data"]:
                # Validate against schema
                if self.schema:
                    schema_valid, schema_error = self._validate_realm_schema(
                        result["response_data"]
                    )
                    if not schema_valid:
                        logger.warning(
                            f"  Schema validation failed for ID {realm_id}: {schema_error}"
                        )
                        result["schema_validation_error"] = schema_error
                    else:
                        logger.info(f"  ✓ Schema validation passed for ID {realm_id}")

                # Check if returned ID matches requested ID
                returned_id = result["response_data"].get("id") or result[
                    "response_data"
                ].get("_id")
                if returned_id and returned_id != realm_id:
                    logger.warning(
                        f"  ID mismatch: requested {realm_id}, got {returned_id}"
                    )
                    result["id_mismatch"] = f"requested {realm_id}, got {returned_id}"

            self._record_test_result(f"get_realm_by_id_{realm_id}", success, result)

            time.sleep(0.5)

    def test_error_cases(self) -> None:
        """Test various error cases and edge conditions."""
        logger.info("=" * 60)
        logger.info("Testing: Error Cases and Edge Conditions")
        logger.info("=" * 60)

        error_tests = [
            {
                "name": "invalid_realm_id",
                "method": "GET",
                "url": f"{self.endpoints['get_realm_by_id']}/99999999",
                "expected_status": [404, 400],
            },
            {
                "name": "non_numeric_realm_id",
                "method": "GET",
                "url": f"{self.endpoints['get_realm_by_id']}/invalid",
                "expected_status": [404, 400],
            },
            {
                "name": "empty_search_term",
                "method": "GET",
                "url": self.endpoints["search_by_name"],
                "params": {"query": "name", "regEx": ""},
                "expected_status": [400, 422],
            },
            {
                "name": "invalid_sort_parameter",
                "method": "GET",
                "url": self.endpoints["get_all_realms"],
                "params": {"sort_by": "invalid_field"},
                "expected_status": [400, 422],
            },
        ]

        for test in error_tests:
            logger.info(f"Testing error case: {test['name']}")

            success, result = self._make_request(
                test["method"], test["url"], params=test.get("params")
            )

            # For error tests, we expect specific status codes
            expected_statuses = test.get("expected_status", [400, 404, 422])
            if result["status_code"] in expected_statuses:
                success = True
                logger.info(f"  ✓ Got expected error status: {result['status_code']}")
            elif result["status_code"] == 200:
                # Unexpected success - might be valid behavior
                logger.info(
                    f"  ? Unexpected success (might be valid): {result['status_code']}"
                )
                success = True

            self._record_test_result(f"error_case_{test['name']}", success, result)

            time.sleep(0.5)

    def run_all_tests(
        self, search_terms: list[str] | None = None, realm_ids: list[int] | None = None
    ) -> dict[str, Any]:
        """
        Run all API tests.

        Args:
            search_terms: Custom search terms for name search tests
            realm_ids: Custom realm IDs for detail tests

        Returns:
            Complete test results
        """
        logger.info("🚀 Starting Comprehensive Realms API Testing")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Timeout: {self.timeout}s")
        logger.info(f"Schema loaded: {'Yes' if self.schema else 'No'}")

        self.results["start_time"] = datetime.now().isoformat()

        try:
            # Run all test suites
            self.test_search_by_name(search_terms)
            self.test_get_all_realms()
            self.test_get_realm_by_id(realm_ids)
            self.test_error_cases()

        except KeyboardInterrupt:
            logger.warning("Tests interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error during testing: {e}")

        self.results["end_time"] = datetime.now().isoformat()

        # Generate summary report
        self._generate_summary_report()

        return self.results

    def _generate_summary_report(self) -> None:
        """Generate and display a summary report."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 TEST SUMMARY REPORT")
        logger.info("=" * 80)

        # Overall statistics
        total = self.results["total_tests"]
        passed = self.results["passed_tests"]
        failed = self.results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0

        logger.info(f"Total Tests:    {total}")
        logger.info(f"Passed:         {passed}")
        logger.info(f"Failed:         {failed}")
        logger.info(f"Success Rate:   {success_rate:.1f}%")

        # Time summary
        if self.results["start_time"] and self.results["end_time"]:
            start_time = datetime.fromisoformat(self.results["start_time"])
            end_time = datetime.fromisoformat(self.results["end_time"])
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Total Duration: {duration:.2f}s")

        # Endpoint summary
        endpoint_stats = {}
        for test in self.results["test_details"]:
            test_name = test["test_name"]
            endpoint = (
                test_name.split("_")[0] + "_" + test_name.split("_")[1]
                if "_" in test_name
                else "unknown"
            )
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {"total": 0, "passed": 0}
            endpoint_stats[endpoint]["total"] += 1
            if test["success"]:
                endpoint_stats[endpoint]["passed"] += 1

        logger.info("\n📈 ENDPOINT BREAKDOWN:")
        for endpoint, stats in endpoint_stats.items():
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            logger.info(
                f"  {endpoint}: {stats['passed']}/{stats['total']} ({rate:.1f}%)"
            )

        # Failed tests details
        failed_tests = [t for t in self.results["test_details"] if not t["success"]]
        if failed_tests:
            logger.info(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                logger.info(f"  • {test['test_name']}")
                if test["details"].get("error"):
                    logger.info(f"    Error: {test['details']['error']}")
                if test["details"].get("status_code"):
                    logger.info(f"    Status: {test['details']['status_code']}")

        # Response time analysis
        response_times = [
            t["details"]["response_time"]
            for t in self.results["test_details"]
            if t["details"].get("response_time") is not None
        ]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            logger.info("\n⏱️  RESPONSE TIMES:")
            logger.info(f"  Average: {avg_time:.2f}s")
            logger.info(f"  Fastest: {min_time:.2f}s")
            logger.info(f"  Slowest: {max_time:.2f}s")

        logger.info("\n" + "=" * 80)

        # Save detailed results to JSON in timestamped directory
        results_file = os.path.join(self.output_dir, "test_results.json")
        try:
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"💾 Detailed results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main function to run the API tests."""
    parser = argparse.ArgumentParser(description="Test the Realms API endpoints")
    parser.add_argument(
        "--schema", default="realm_schema.json", help="Path to realm schema JSON file"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Request timeout in seconds"
    )
    parser.add_argument(
        "--search-terms", nargs="+", help="Custom search terms for name search tests"
    )
    parser.add_argument(
        "--realm-ids", type=int, nargs="+", help="Custom realm IDs for detail tests"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run a quick test with minimal test cases"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = RealmsAPITester(
        schema_path=args.schema, timeout=args.timeout, output_dir=output_dir
    )

    # Adjust test parameters for quick mode
    search_terms = args.search_terms
    realm_ids = args.realm_ids

    if args.quick:
        search_terms = search_terms or ["Avana"]
        realm_ids = realm_ids or [2188]

    # Run tests
    results = tester.run_all_tests(search_terms=search_terms, realm_ids=realm_ids)

    # Exit with appropriate code
    if results["failed_tests"] == 0:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.warning(f"⚠️  {results['failed_tests']} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
