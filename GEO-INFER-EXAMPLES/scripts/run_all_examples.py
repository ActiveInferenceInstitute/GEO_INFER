#!/usr/bin/env python3
"""
Comprehensive Integration Example Runner and Assessment Tool

This script systematically runs all GEO-INFER integration examples,
assesses their success, and generates detailed reports.
"""

import sys
import os
import time
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExampleResult:
    """Result of running a single example."""

    name: str
    category: str
    path: str
    status: str  # 'success', 'failure', 'skipped', 'error'
    execution_time: float
    modules_used: list[str] = field(default_factory=list)
    integration_patterns: list[str] = field(default_factory=list)
    output_summary: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    complexity_level: int = 1  # 1-5 scale


@dataclass
class AssessmentReport:
    """Comprehensive assessment report for all examples."""

    total_examples: int
    successful_examples: int
    failed_examples: int
    skipped_examples: int
    average_execution_time: float
    total_execution_time: float
    examples: list[ExampleResult] = field(default_factory=list)
    integration_coverage: dict[str, int] = field(default_factory=dict)
    module_usage_stats: dict[str, int] = field(default_factory=dict)
    pattern_usage_stats: dict[str, int] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class IntegrationExampleRunner:
    """Comprehensive runner for all integration examples."""

    def __init__(self, examples_dir: Path):
        self.examples_dir = examples_dir
        self.logger = self._setup_logging()
        self.results: list[ExampleResult] = []

        # Known integration patterns
        self.integration_patterns = {
            "linear_pipeline": "Sequential processing chain",
            "parallel_processing": "Independent parallel analyses",
            "feedback_loop": "Adaptive feedback systems",
            "event_driven": "Real-time event responses",
            "hub_and_spoke": "Centralized coordination",
            "iot_driven": "IoT sensor-based workflows",
            "multi_domain": "Cross-domain analysis",
            "simulation_based": "Simulation and optimization",
        }

        # Known GEO-INFER modules
        self.geo_infer_modules = [
            "DATA",
            "SPACE",
            "TIME",
            "AI",
            "HEALTH",
            "AG",
            "IOT",
            "RISK",
            "API",
            "APP",
            "BAYES",
            "ACT",
            "AGENT",
            "SIM",
            "ECON",
            "CIV",
            "NORMS",
            "ORG",
            "BIO",
            "COMMS",
            "LOG",
            "MATH",
            "OPS",
            "SEC",
            "COG",
            "ART",
            "GIT",
            "PEP",
            "REQ",
            "TEST",
            "INTRA",
            "ANT",
            "SPM",
        ]

        # Example configurations. `modules` lists only GEO-INFER modules whose
        # public APIs the example actually imports; the remaining integration
        # examples are self-contained synthetic simulations.
        self.example_configs = {
            "basic_integration_demo": {
                "modules": ["SPACE", "TIME", "MATH"],
                "pattern": "linear_pipeline",
                "complexity": 2,
                "expected_duration": 5,
            },
            "disease_surveillance_pipeline": {
                "modules": [],
                "pattern": "simulation_based",
                "complexity": 5,
                "expected_duration": 10,
            },
            "precision_farming_system": {
                "modules": [],
                "pattern": "simulation_based",
                "complexity": 4,
                "expected_duration": 8,
            },
            "spatial_microbiome_soil_climate": {
                "modules": [],
                "pattern": "simulation_based",
                "complexity": 4,
                "expected_duration": 12,
            },
        }

    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging."""
        logger = logging.getLogger("example_runner")
        logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = self.examples_dir.parent / "logs" / "example_runner.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def discover_examples(self) -> list[dict[str, Any]]:
        """Discover all runnable examples."""
        examples = []

        # Define known examples with their paths
        known_examples = [
            {
                "name": "basic_integration_demo",
                "category": "getting_started",
                "script_path": "examples/getting_started/basic_integration_demo/scripts/run_example.py",
                "description": "Basic 4-module integration demonstration",
            },
            {
                "name": "disease_surveillance_pipeline",
                "category": "health_integration",
                "script_path": "examples/health_integration/disease_surveillance_pipeline/scripts/run_surveillance_pipeline.py",
                "description": "Comprehensive 8-module disease surveillance system",
            },
            {
                "name": "precision_farming_system",
                "category": "agriculture_integration",
                "script_path": "examples/agriculture_integration/precision_farming_system/scripts/run_example.py",
                "description": "IoT-driven precision agriculture system",
            },
            {
                "name": "spatial_microbiome_soil_climate",
                "category": "climate_integration",
                "script_path": "examples/climate_integration/spatial_microbiome_soil_climate/scripts/run_example.py",
                "description": "Climate-microbiome-economic analysis system",
            },
        ]

        # Check which examples actually exist
        for example in known_examples:
            script_path = self.examples_dir.parent / example["script_path"]
            if script_path.exists():
                example["full_path"] = script_path
                examples.append(example)
                self.logger.info(f"Found example: {example['name']}")
            else:
                self.logger.warning(f"Example script not found: {script_path}")

        self.logger.info(f"Discovered {len(examples)} runnable examples")
        return examples

    def run_example(self, example_config: dict[str, Any]) -> ExampleResult:
        """Run a single example and assess its success."""
        example_name = example_config["name"]
        self.logger.info(f"Running example: {example_name}")

        start_time = time.time()
        result = ExampleResult(
            name=example_name,
            category=example_config["category"],
            path=str(example_config["full_path"]),
            status="skipped",
            execution_time=0.0,
        )

        try:
            # Get expected configuration
            config = self.example_configs.get(example_name, {})
            result.modules_used = config.get("modules", [])
            result.integration_patterns = [config.get("pattern", "unknown")]
            result.complexity_level = config.get("complexity", 1)

            # Run the example
            script_path = example_config["full_path"]
            working_dir = script_path.parent.parent

            # Set environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.examples_dir.parent)

            # Execute with timeout
            process = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
                env=env,
            )

            result.execution_time = time.time() - start_time

            if process.returncode == 0:
                result.status = "success"
                result.output_summary = self._parse_output_summary(
                    process.stdout, example_name
                )
                self.logger.info(
                    f"✅ {example_name} completed successfully in {result.execution_time:.2f}s"
                )
            else:
                result.status = "failure"
                result.error_message = process.stderr or "Non-zero return code"
                self.logger.warning(f"❌ {example_name} failed: {result.error_message}")

        except subprocess.TimeoutExpired:
            result.status = "failure"
            result.error_message = "Execution timeout (60 seconds)"
            result.execution_time = time.time() - start_time
            self.logger.warning(f"⏰ {example_name} timed out")

        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
            self.logger.error(f"💥 {example_name} error: {e}")

        return result

    def _parse_output_summary(self, stdout: str, example_name: str) -> dict[str, Any]:
        """Parse the structured JSON status/result envelope printed by examples.

        Examples that follow the repository convention end their stdout with a
        JSON object such as ``{"status": "ok", ...}`` (see
        examples/getting_started/basic_integration_demo). The last such object
        in the output is decoded; its ``status`` field and scalar top-level
        entries form the summary.
        """
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        trailing_blank: list[dict[str, Any]] = []
        for match in re.finditer(r"\{", stdout):
            try:
                candidate, end = decoder.raw_decode(stdout[match.start() :])
            except ValueError:
                continue
            if isinstance(candidate, dict):
                candidates.append(candidate)
                if not stdout[match.start() + end :].strip():
                    trailing_blank.append(candidate)
        # Prefer an envelope that runs to the end of the output (the printed
        # JSON envelope is the final statement); fall back to the last decoded
        # JSON object when trailing prose follows it.
        envelope = (
            trailing_blank[-1]
            if trailing_blank
            else (candidates[-1] if candidates else None)
        )
        if envelope is None:
            self.logger.warning(f"No JSON envelope found in output of {example_name}")
            return {}

        summary: dict[str, Any] = {}
        if "status" in envelope:
            summary["status"] = envelope["status"]
        for key, value in envelope.items():
            if key == "status":
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                summary[key] = value
        return summary

    def run_all_examples(self) -> AssessmentReport:
        """Run all examples and generate comprehensive assessment."""
        self.logger.info("Starting comprehensive example assessment...")

        examples = self.discover_examples()
        total_start_time = time.time()

        # Run each example
        for example_config in examples:
            result = self.run_example(example_config)
            self.results.append(result)

        total_execution_time = time.time() - total_start_time

        # Generate assessment report
        report = self._generate_assessment_report(total_execution_time)

        # Save detailed results
        self._save_results(report)

        return report

    def _generate_assessment_report(
        self, total_execution_time: float
    ) -> AssessmentReport:
        """Generate comprehensive assessment report."""
        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status in ["failure", "error"]]
        skipped = [r for r in self.results if r.status == "skipped"]

        # Calculate statistics
        avg_execution_time = (
            sum(r.execution_time for r in successful) / len(successful)
            if successful
            else 0.0
        )

        # Module usage statistics
        module_usage = {}
        for result in self.results:
            for module in result.modules_used:
                module_usage[module] = module_usage.get(module, 0) + 1

        # Pattern usage statistics
        pattern_usage = {}
        for result in self.results:
            for pattern in result.integration_patterns:
                pattern_usage[pattern] = pattern_usage.get(pattern, 0) + 1

        # Integration coverage by category
        integration_coverage = {}
        for result in self.results:
            category = result.category
            integration_coverage[category] = integration_coverage.get(category, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(successful, failed, skipped)

        return AssessmentReport(
            total_examples=len(self.results),
            successful_examples=len(successful),
            failed_examples=len(failed),
            skipped_examples=len(skipped),
            average_execution_time=avg_execution_time,
            total_execution_time=total_execution_time,
            examples=self.results,
            integration_coverage=integration_coverage,
            module_usage_stats=module_usage,
            pattern_usage_stats=pattern_usage,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        successful: list[ExampleResult],
        failed: list[ExampleResult],
        skipped: list[ExampleResult],
    ) -> list[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        # Success rate analysis
        total = len(successful) + len(failed) + len(skipped)
        success_rate = len(successful) / total if total > 0 else 0

        if success_rate >= 0.8:
            recommendations.append(
                "✅ HIGH SUCCESS RATE: Examples are working well. "
                "Focus on expanding integration patterns and adding advanced features."
            )
        elif success_rate >= 0.5:
            recommendations.append(
                "⚠️ MODERATE SUCCESS RATE: Some examples need attention. "
                "Review failed examples and improve error handling."
            )
        else:
            recommendations.append(
                "🚨 LOW SUCCESS RATE: Priority focus needed on example reliability. "
                "Address fundamental integration issues and dependencies."
            )

        # Performance analysis
        if successful:
            avg_time = sum(r.execution_time for r in successful) / len(successful)
            if avg_time > 30:
                recommendations.append(
                    "⚡ OPTIMIZE PERFORMANCE: Examples are taking longer than expected. "
                    "Consider optimization and provide progress indicators."
                )
            elif avg_time < 1:
                recommendations.append(
                    "🚀 EXCELLENT PERFORMANCE: Examples execute quickly. "
                    "Consider adding more complex scenarios to demonstrate capabilities."
                )

        # Module coverage analysis
        module_usage = {}
        for result in successful:
            for module in result.modules_used:
                module_usage[module] = module_usage.get(module, 0) + 1

        if len(module_usage) < 10:
            recommendations.append(
                "📊 EXPAND MODULE COVERAGE: Only using limited modules. "
                "Add examples showcasing more GEO-INFER modules for comprehensive coverage."
            )

        # Integration pattern diversity
        pattern_usage = {}
        for result in successful:
            for pattern in result.integration_patterns:
                pattern_usage[pattern] = pattern_usage.get(pattern, 0) + 1

        if len(pattern_usage) < 5:
            recommendations.append(
                "🔄 DIVERSIFY INTEGRATION PATTERNS: Limited pattern variety. "
                "Implement more integration patterns to showcase ecosystem flexibility."
            )

        # Failure analysis
        if failed:
            recommendations.append(
                f"🔧 FIX FAILED EXAMPLES: {len(failed)} examples failed. "
                "Review error messages and address dependency or implementation issues."
            )

        return recommendations

    def _save_results(self, report: AssessmentReport) -> None:
        """Save comprehensive results to files."""
        output_dir = self.examples_dir.parent / "assessment_results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_file = output_dir / f"comprehensive_assessment_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self._report_to_dict(report), f, indent=2, default=str)

        # Save human-readable summary
        summary_file = output_dir / f"assessment_summary_{timestamp}.md"
        self._generate_summary_report(report, summary_file)

        # Save latest results (overwrite)
        latest_json = output_dir / "latest_comprehensive_assessment.json"
        with open(latest_json, "w") as f:
            json.dump(self._report_to_dict(report), f, indent=2, default=str)

        latest_summary = output_dir / "latest_assessment_summary.md"
        self._generate_summary_report(report, latest_summary)

        self.logger.info(f"Results saved to {output_dir}")

    def _report_to_dict(self, report: AssessmentReport) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "total_examples": report.total_examples,
            "successful_examples": report.successful_examples,
            "failed_examples": report.failed_examples,
            "skipped_examples": report.skipped_examples,
            "average_execution_time": report.average_execution_time,
            "total_execution_time": report.total_execution_time,
            "examples": [
                {
                    "name": ex.name,
                    "category": ex.category,
                    "path": ex.path,
                    "status": ex.status,
                    "execution_time": ex.execution_time,
                    "modules_used": ex.modules_used,
                    "integration_patterns": ex.integration_patterns,
                    "output_summary": ex.output_summary,
                    "error_message": ex.error_message,
                    "complexity_level": ex.complexity_level,
                }
                for ex in report.examples
            ],
            "integration_coverage": report.integration_coverage,
            "module_usage_stats": report.module_usage_stats,
            "pattern_usage_stats": report.pattern_usage_stats,
            "recommendations": report.recommendations,
            "timestamp": report.timestamp,
        }

    def _generate_summary_report(
        self, report: AssessmentReport, output_file: Path
    ) -> None:
        """Generate human-readable summary report."""
        with open(output_file, "w") as f:
            f.write("# GEO-INFER Integration Examples - Comprehensive Assessment\n\n")
            f.write(f"**Generated**: {report.timestamp}\n\n")

            # Executive Summary
            f.write("## 📊 Executive Summary\n\n")
            f.write(f"- **Total Examples**: {report.total_examples}\n")
            f.write(
                f"- **Successful**: {report.successful_examples} ({report.successful_examples / report.total_examples * 100:.1f}%)\n"
            )
            f.write(
                f"- **Failed**: {report.failed_examples} ({report.failed_examples / report.total_examples * 100:.1f}%)\n"
            )
            f.write(
                f"- **Skipped**: {report.skipped_examples} ({report.skipped_examples / report.total_examples * 100:.1f}%)\n"
            )
            f.write(
                f"- **Average Execution Time**: {report.average_execution_time:.2f} seconds\n"
            )
            f.write(
                f"- **Total Assessment Time**: {report.total_execution_time:.2f} seconds\n\n"
            )

            # Success Rate Analysis
            success_rate = report.successful_examples / report.total_examples * 100
            if success_rate >= 80:
                status_emoji = "🟢"
                status_text = "EXCELLENT"
            elif success_rate >= 60:
                status_emoji = "🟡"
                status_text = "GOOD"
            else:
                status_emoji = "🔴"
                status_text = "NEEDS IMPROVEMENT"

            f.write(
                f"### Overall Status: {status_emoji} {status_text} ({success_rate:.1f}% success rate)\n\n"
            )

            # Key Recommendations
            f.write("## 🎯 Key Recommendations\n\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n\n")

            # Module Usage Statistics
            f.write("## 📊 Module Usage Statistics\n\n")
            f.write("| Module | Usage Count | Coverage |\n")
            f.write("|--------|-------------|----------|\n")
            for module, count in sorted(
                report.module_usage_stats.items(), key=lambda x: x[1], reverse=True
            ):
                coverage = count / report.total_examples * 100
                f.write(f"| {module} | {count} | {coverage:.1f}% |\n")
            f.write("\n")

            # Integration Pattern Usage
            f.write("## 🔄 Integration Pattern Usage\n\n")
            f.write("| Pattern | Usage Count | Description |\n")
            f.write("|---------|-------------|-------------|\n")
            for pattern, count in sorted(
                report.pattern_usage_stats.items(), key=lambda x: x[1], reverse=True
            ):
                description = self.integration_patterns.get(pattern, "Custom pattern")
                f.write(f"| {pattern} | {count} | {description} |\n")
            f.write("\n")

            # Detailed Results by Status
            f.write("## 📋 Detailed Example Results\n\n")

            for status in ["success", "failure", "error", "skipped"]:
                examples_with_status = [
                    r for r in report.examples if r.status == status
                ]
                if not examples_with_status:
                    continue

                status_emoji = {
                    "success": "✅",
                    "failure": "❌",
                    "error": "💥",
                    "skipped": "⏭️",
                }[status]
                f.write(
                    f"### {status_emoji} {status.title()} Examples ({len(examples_with_status)})\n\n"
                )

                for example in examples_with_status:
                    f.write(f"#### {example.name}\n")
                    f.write(f"- **Category**: {example.category}\n")
                    f.write(f"- **Execution Time**: {example.execution_time:.2f}s\n")
                    f.write(
                        f"- **Modules Used**: {', '.join(example.modules_used) if example.modules_used else 'None detected'}\n"
                    )
                    f.write(
                        f"- **Integration Patterns**: {', '.join(example.integration_patterns) if example.integration_patterns else 'None detected'}\n"
                    )
                    f.write(f"- **Complexity Level**: {example.complexity_level}/5\n")

                    if example.output_summary:
                        f.write(f"- **Output Summary**: {example.output_summary}\n")

                    if example.error_message:
                        f.write(f"- **Error**: {example.error_message}\n")

                    f.write("\n")

            f.write("---\n")
            f.write(
                "*This report was generated automatically by the GEO-INFER Examples Assessment Tool*\n"
            )


def main():
    """Main function to run comprehensive example assessment."""
    print("🚀 GEO-INFER Integration Examples - Comprehensive Assessment")
    print("=" * 70)

    # Setup
    examples_dir = Path(__file__).parent.parent / "examples"
    runner = IntegrationExampleRunner(examples_dir)

    # Run assessment
    try:
        report = runner.run_all_examples()

        # Display summary
        print("\n📊 Assessment Complete!")
        print(f"Total Examples: {report.total_examples}")
        print(
            f"Successful: {report.successful_examples} ({report.successful_examples / report.total_examples * 100:.1f}%)"
        )
        print(
            f"Failed: {report.failed_examples} ({report.failed_examples / report.total_examples * 100:.1f}%)"
        )
        print(
            f"Skipped: {report.skipped_examples} ({report.skipped_examples / report.total_examples * 100:.1f}%)"
        )
        print(f"Total Time: {report.total_execution_time:.2f} seconds")

        print("\n🎯 Top Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"{i}. {rec}")

        print("\n📁 Detailed results saved to: assessment_results/")

        return 0 if report.failed_examples == 0 else 1

    except Exception as e:
        print(f"❌ Assessment failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
