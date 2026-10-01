#!/usr/bin/env python3

"""
Command-line interface for GEO-INFER-GIT

This module provides a command-line interface for interacting with
the GEO-INFER-GIT repository management functionality.
"""

import os
import sys
import argparse
import logging
import yaml
from typing import Any
import json

from geo_infer_git.core.repo_manager import RepoManager

# Configure logger
logger = logging.getLogger("geo_infer_git.cli")


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the CLI.

    Args:
        verbose: Whether to use verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce verbosity of some modules
    logging.getLogger("git").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_repo_list(file_path: str) -> list[dict[str, Any]]:
    """
    Load repository list from file.

    Args:
        file_path: Path to YAML or JSON file containing repository list

    Returns:
        List of repository configurations
    """
    if not os.path.exists(file_path):
        logger.error(f"Repository list file not found: {file_path}")
        sys.exit(1)

    try:
        with open(file_path) as f:
            if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                repos = yaml.safe_load(f)
            elif file_path.endswith(".json"):
                repos = json.load(f)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                sys.exit(1)

        if not isinstance(repos, list):
            logger.error(f"Repository list must be a list, got {type(repos)}")
            sys.exit(1)

        # Validate each repository entry
        for idx, repo in enumerate(repos):
            if not isinstance(repo, dict):
                logger.error(f"Repository entry must be a dictionary, got {type(repo)}")
                sys.exit(1)

            if "url" not in repo:
                logger.error(f"Repository entry {idx} is missing required 'url' field")
                sys.exit(1)

        return repos
    except Exception as e:
        logger.error(f"Failed to load repository list: {str(e)}")
        sys.exit(1)


def clone_command(args: argparse.Namespace) -> None:
    """
    Handle the 'clone' command.

    Args:
        args: Command-line arguments
    """
    logger.info(f"Executing clone command with args: {args}")

    # Create repo manager
    manager = RepoManager(config_path=args.config)

    # Handle different input methods
    if args.file:
        # Load repositories from file
        repos = load_repo_list(args.file)
    elif args.url:
        # Single repository from URL
        repos = [{"url": args.url, "name": args.name}]
    else:
        logger.error("No repository source specified (--file or --url)")
        sys.exit(1)

    # Clone repositories
    results = manager.clone_repositories(repos, parallel=not args.sequential)

    # Print results
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"Cloned {success_count}/{len(results)} repositories")

    for name, success in results.items():
        status = "Success" if success else "Failed"
        logger.info(f"{name}: {status}")

    # Exit with error if any failed
    if success_count < len(results):
        sys.exit(1)


def sync_command(args: argparse.Namespace) -> None:
    """
    Handle the 'sync' command.

    Args:
        args: Command-line arguments
    """
    logger.info(f"Executing sync command with args: {args}")

    # Create repo manager
    manager = RepoManager(config_path=args.config)

    # Get repos to sync
    repos = args.repos if args.repos else None

    # Sync repositories
    results = manager.sync_repositories(repo_names=repos)

    # Print results
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"Synced {success_count}/{len(results)} repositories")

    for name, success in results.items():
        status = "Success" if success else "Failed"
        logger.info(f"{name}: {status}")

    # Exit with error if any failed
    if success_count < len(results):
        sys.exit(1)


def status_command(args: argparse.Namespace) -> None:
    """
    Handle the 'status' command.

    Args:
        args: Command-line arguments
    """
    logger.info(f"Executing status command with args: {args}")

    # Create repo manager
    manager = RepoManager(config_path=args.config)

    # Get repos to check
    repos = args.repos if args.repos else None

    # Check repository status
    results = manager.check_repo_status(repo_names=repos)

    # Print results
    for name, status in results.items():
        if "error" in status:
            logger.error(f"{name}: Error - {status['error']}")
            continue

        branch = status.get("branch", "unknown")
        is_dirty = status.get("is_dirty", False)
        commits_behind = status.get("commits_behind", 0)
        commits_ahead = status.get("commits_ahead", 0)

        state = "dirty" if is_dirty else "clean"

        if args.json:
            print(json.dumps({name: status}, indent=2))
        else:
            logger.info(
                f"{name}: branch={branch}, state={state}, behind={commits_behind}, ahead={commits_ahead}"
            )
            if is_dirty and status.get("untracked_files"):
                logger.info(f"  Untracked files: {len(status['untracked_files'])}")


def branch_command(args: argparse.Namespace) -> None:
    """
    Handle the 'branch' command.

    Args:
        args: Command-line arguments
    """
    logger.info(f"Executing branch command with args: {args}")

    # Create repo manager
    manager = RepoManager(config_path=args.config)

    # Get repos to work with
    repos = args.repos if args.repos else None

    # Handle subcommands
    if args.create:
        # Create branch
        branch_name = args.create
        results = manager.create_branch(branch_name, repo_names=repos)

        # Print results
        success_count = sum(1 for v in results.values() if v)
        logger.info(
            f"Created branch {branch_name} in {success_count}/{len(results)} repositories"
        )

    elif args.checkout:
        # Checkout branch
        branch_name = args.checkout
        results = manager.checkout_branch(branch_name, repo_names=repos)

        # Print results
        success_count = sum(1 for v in results.values() if v)
        logger.info(
            f"Checked out branch {branch_name} in {success_count}/{len(results)} repositories"
        )

    # Exit with error if any failed
    if "results" in locals() and success_count < len(results):
        sys.exit(1)


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="GEO-INFER-GIT - Repository management for geospatial projects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Global arguments
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--config", "-c", default=None, help="Path to configuration file"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Clone command
    clone_parser = subparsers.add_parser("clone", help="Clone repositories")
    clone_parser.add_argument(
        "--file", "-f", help="Path to repository list file (YAML or JSON)"
    )
    clone_parser.add_argument("--url", "-u", help="URL of repository to clone")
    clone_parser.add_argument("--name", "-n", help="Name for the cloned repository")
    clone_parser.add_argument(
        "--sequential", action="store_true", help="Clone repositories sequentially"
    )

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync (pull) repositories")
    sync_parser.add_argument(
        "repos", nargs="*", help="Repositories to sync (all if not specified)"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check repository status")
    status_parser.add_argument(
        "repos", nargs="*", help="Repositories to check (all if not specified)"
    )
    status_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    # Branch command
    branch_parser = subparsers.add_parser("branch", help="Branch operations")
    branch_parser.add_argument("--create", help="Create a new branch")
    branch_parser.add_argument("--checkout", help="Checkout an existing branch")
    branch_parser.add_argument(
        "repos", nargs="*", help="Repositories to operate on (all if not specified)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Handle commands
    if args.command == "clone":
        clone_command(args)
    elif args.command == "sync":
        sync_command(args)
    elif args.command == "status":
        status_command(args)
    elif args.command == "branch":
        branch_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
