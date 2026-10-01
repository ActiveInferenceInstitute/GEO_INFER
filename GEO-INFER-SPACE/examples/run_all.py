#!/usr/bin/env python3
"""
Run All Examples - GEO-INFER-SPACE

This script orchestrates the execution of all example files in the examples directory.
It provides comprehensive error handling, timing, and summary reporting.

Usage:
    uv run python examples/run_all.py
    uv run python examples/run_all.py --quick  # Run only quick examples
"""

import sys
import time
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Any

EXAMPLES_DIR = Path(__file__).parent

# ============================================================================
# Configuration
# ============================================================================

EXAMPLES = [
    {
        "name": "Multiple Dispatch Demo",
        "file": "multiple_dispatch_demo.py",
        "function": "demonstrate_multiple_dispatch",
        "quick": True,
        "description": "Demonstrates unified backend dispatching to H3/SRAI",
    },
    {
        "name": "H3 Integration Examples",
        "file": "h3_integration_examples.py",
        "function": "main",
        "quick": True,
        "description": "H3 integration with other SPACE modules",
    },
    {
        "name": "H3 Comprehensive Examples",
        "file": "h3_comprehensive_examples.py",
        "function": None,  # Uses __main__ block
        "quick": False,
        "description": "Comprehensive H3 operations with visualizations",
    },
    {
        "name": "H3 Advanced Applications",
        "file": "h3_advanced_applications.py",
        "function": None,  # Uses __main__ block
        "quick": False,
        "description": "ML, disaster response, and performance optimization",
    },
    {
        "name": "Nested Orchestrator Examples",
        "file": "nested_orchestrator_examples.py",
        "function": None,  # Uses __main__ block
        "quick": False,
        "description": "Complex nested H3 system orchestration",
    },
]


# ============================================================================
# Example Runner
# ============================================================================


class ExampleRunner:
    """Orchestrates example execution with error handling and reporting."""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results: list[dict[str, Any]] = []
        self.start_time = datetime.now()

    def run_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Run a single example and capture results."""
        result = {
            "name": example["name"],
            "file": example["file"],
            "status": "pending",
            "duration_seconds": 0,
            "error": None,
            "output": None,
        }

        file_path = EXAMPLES_DIR / example["file"]

        if not file_path.exists():
            result["status"] = "skipped"
            result["error"] = f"File not found: {file_path}"
            return result

        print(f"\n{'=' * 60}")
        print(f"🚀 Running: {example['name']}")
        print(f"   File: {example['file']}")
        print(f"   Description: {example['description']}")
        print(f"{'=' * 60}")

        start = time.time()

        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(
                example["file"].replace(".py", ""), file_path
            )
            module = importlib.util.module_from_spec(spec)

            # Execute the module
            spec.loader.exec_module(module)

            # If a specific function is specified, call it
            if example.get("function"):
                func = getattr(module, example["function"], None)
                if func:
                    func()

            result["status"] = "success"
            print(f"\n✅ {example['name']} completed successfully")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"\n❌ {example['name']} failed: {e}")

        result["duration_seconds"] = time.time() - start
        return result

    def run_all(self) -> dict[str, Any]:
        """Run all configured examples."""
        print("\n" + "=" * 70)
        print("🎯 GEO-INFER-SPACE Examples Runner")
        print("=" * 70)
        print(f"Started: {self.start_time.isoformat()}")
        print(f"Quick mode: {self.quick_mode}")
        print(f"Total examples: {len(EXAMPLES)}")

        examples_to_run = [
            ex for ex in EXAMPLES if not self.quick_mode or ex.get("quick", False)
        ]

        print(f"Examples to run: {len(examples_to_run)}")
        print("=" * 70)

        for example in examples_to_run:
            result = self.run_example(example)
            self.results.append(result)

        return self.generate_summary()

    def generate_summary(self) -> dict[str, Any]:
        """Generate execution summary."""
        total_time = (datetime.now() - self.start_time).total_seconds()

        success_count = sum(1 for r in self.results if r["status"] == "success")
        failed_count = sum(1 for r in self.results if r["status"] == "failed")
        skipped_count = sum(1 for r in self.results if r["status"] == "skipped")

        summary = {
            "total_examples": len(self.results),
            "successful": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_duration_seconds": total_time,
            "results": self.results,
            "timestamp": datetime.now().isoformat(),
        }

        print("\n" + "=" * 70)
        print("📊 EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total examples: {len(self.results)}")
        print(f"  ✅ Successful: {success_count}")
        print(f"  ❌ Failed: {failed_count}")
        print(f"  ⏭️  Skipped: {skipped_count}")
        print(f"Total duration: {total_time:.2f} seconds")
        print("=" * 70)

        # Show individual results
        print("\nDetailed Results:")
        for result in self.results:
            status_icon = {"success": "✅", "failed": "❌", "skipped": "⏭️"}.get(
                result["status"], "❓"
            )

            print(
                f"  {status_icon} {result['name']}: {result['status']} ({result['duration_seconds']:.2f}s)"
            )

            if result["error"]:
                print(f"      Error: {result['error'][:100]}...")

        print("=" * 70)

        if failed_count == 0:
            print("\n🎉 All examples completed successfully!")
        else:
            print(
                f"\n⚠️  {failed_count} example(s) failed. Check the logs above for details."
            )

        return summary


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the example runner."""
    quick_mode = "--quick" in sys.argv or "-q" in sys.argv

    runner = ExampleRunner(quick_mode=quick_mode)
    summary = runner.run_all()

    # Exit with error code if any examples failed
    if summary["failed"] > 0:
        sys.exit(1)

    return summary


if __name__ == "__main__":
    main()
