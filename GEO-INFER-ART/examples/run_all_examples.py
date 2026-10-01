#!/usr/bin/env python
"""
Script to run all GEO-INFER-ART examples.
"""

import os
import sys
import subprocess


def find_example_scripts() -> list[str]:
    """Find all Python example scripts in the examples directory.

    Returns:
        List of paths to example scripts
    """
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    example_scripts = []

    for root, _, files in os.walk(examples_dir):
        for file in files:
            if file.endswith(".py") and file != os.path.basename(__file__):
                example_scripts.append(os.path.join(root, file))

    return sorted(example_scripts)


def run_example(script_path: str) -> tuple[int, str]:
    """Run an example script and return its exit code.

    Args:
        script_path: Path to the script to run

    Returns:
        Tuple of (exit_code, output)
    """
    print(f"\n\n{'=' * 80}")
    print(f"Running example: {os.path.basename(script_path)}")
    print(f"{'=' * 80}")

    try:
        # Run the script as a separate process
        result = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True, check=False
        )

        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode, result.stdout

    except Exception as e:
        print(f"Error running example {script_path}: {str(e)}")
        return 1, str(e)


def run_all_examples() -> int:
    """Run all example scripts.

    Returns:
        0 if all examples succeeded, 1 otherwise
    """
    example_scripts = find_example_scripts()

    print(f"Found {len(example_scripts)} example scripts:")
    for script in example_scripts:
        print(f"  - {os.path.basename(script)}")

    # Add the project directory to the path so modules can be imported
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    # Run all examples
    results = []
    for script in example_scripts:
        exit_code, _ = run_example(script)
        results.append((script, exit_code))

    # Print summary
    print("\n\nSummary:")
    print("-" * 40)
    success_count = sum(1 for _, code in results if code == 0)
    fail_count = len(results) - success_count

    print(f"{success_count} examples succeeded")
    print(f"{fail_count} examples failed")

    if fail_count > 0:
        print("\nFailed examples:")
        for script, code in results:
            if code != 0:
                print(f"  - {os.path.basename(script)} (exit code: {code})")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_examples())
