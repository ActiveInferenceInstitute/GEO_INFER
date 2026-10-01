#!/usr/bin/env python3
"""
Main entry point for GEO-INFER-GIT repository cloning functionality.

This script handles the cloning of GitHub repositories based on configured
target repositories and users.
"""

import os
import argparse
import logging
from typing import Any

# Import the required modules
from .utils.config_loader import (
    load_clone_config,
    load_target_repos_config,
    load_target_users_config,
)
from .core.github_api import GitHubAPI
from .core.repo_cloner import RepoCloner

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GEO-INFER-GIT Repository Cloning Tool"
    )

    parser.add_argument(
        "--config-dir", type=str, help="Directory containing configuration files"
    )

    parser.add_argument(
        "--output-dir", type=str, help="Directory to store cloned repositories"
    )

    parser.add_argument(
        "--clone-repos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clone repositories from target_repos.yaml (use --no-clone-repos to disable)",
    )

    parser.add_argument(
        "--clone-users",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clone repositories from users in target_users.yaml (use --no-clone-users to disable)",
    )

    parser.add_argument(
        "--github-token", type=str, help="GitHub API token for authentication"
    )

    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clone repositories in parallel (use --no-parallel to disable)",
    )

    parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum number of parallel workers"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--generate-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a report after cloning (use --no-generate-report to disable)",
    )

    return parser.parse_args()


def create_gitignore_entry(output_dir: str) -> None:
    """Add output directory to the containing repository's .gitignore.

    Walks up from the output directory to find an enclosing git working
    tree. When the output directory is not inside any repository, nothing
    is written.
    """
    search_dir = os.path.abspath(output_dir)
    repo_root = None
    while True:
        parent = os.path.dirname(search_dir)
        if os.path.isdir(os.path.join(search_dir, ".git")):
            repo_root = search_dir
            break
        if parent == search_dir:
            break
        search_dir = parent

    if repo_root is None:
        logger.debug(
            f"Output directory {output_dir} is not inside a git repository; "
            "skipping .gitignore update"
        )
        return

    gitignore_path = os.path.join(repo_root, ".gitignore")
    relative_entry = os.path.relpath(os.path.abspath(output_dir), repo_root)

    # Create .gitignore if it doesn't exist
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write(f"{relative_entry}\n")
        logger.info(f"Created .gitignore with entry for {relative_entry}")
        return

    # Check if entry already exists
    with open(gitignore_path) as f:
        content = f.read()

    # Add entry if not found
    if relative_entry not in content:
        with open(gitignore_path, "a") as f:
            f.write(f"\n# GEO-INFER-GIT cloned repositories\n{relative_entry}\n")
        logger.info(f"Added {relative_entry} to .gitignore")


def generate_report(
    results: dict[str, Any], output_dir: str, format: str = "markdown"
) -> str:
    """Generate a report of cloning results."""
    report_filename = os.path.join(output_dir, "clone_report.md")

    with open(report_filename, "w") as f:
        f.write("# GitHub Repository Cloning Report\n\n")

        # Summary statistics
        f.write("## Summary\n\n")
        total_repos = results.get("total_repos", 0)
        success_repos = results.get("success_repos", 0)
        success_rate = (success_repos / total_repos * 100) if total_repos > 0 else 0

        f.write(f"- **Total repositories attempted**: {total_repos}\n")
        f.write(f"- **Successfully cloned repositories**: {success_repos}\n")
        f.write(f"- **Success rate**: {success_rate:.2f}%\n")
        f.write(f"- **Output directory**: {os.path.abspath(output_dir)}\n\n")

        # Target repositories results
        if "target_repos" in results:
            f.write("## Target Repositories\n\n")
            for repo in results["target_repos"]:
                owner = repo.get("owner", "")
                repo_name = repo.get("repo", "")
                success = repo.get("success", False)
                status = "✅ Success" if success else "❌ Failed"

                f.write(
                    f"- [{owner}/{repo_name}](https://github.com/{owner}/{repo_name}) - {status}\n"
                )

            f.write("\n")

        # Target users results
        if "target_users" in results:
            f.write("## User Repositories\n\n")
            for user in results["target_users"]:
                username = user.get("username", "")
                f.write(f"### {username}\n\n")

                repos = user.get("repos", [])
                if not repos:
                    f.write("No repositories cloned.\n\n")
                    continue

                for repo in repos:
                    repo_name = repo.get("name", "")
                    success = repo.get("success", False)
                    status = "✅ Success" if success else "❌ Failed"

                    f.write(
                        f"- [{username}/{repo_name}](https://github.com/{username}/{repo_name}) - {status}\n"
                    )

                f.write("\n")

    logger.info(f"Generated report at {report_filename}")
    return report_filename


def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    config_dir = args.config_dir
    clone_config = load_clone_config(config_dir)

    # Override configuration with command line arguments
    if args.output_dir:
        clone_config.output_dir = args.output_dir

    if args.github_token:
        clone_config.github_token = args.github_token

    clone_config.concurrency_enabled = args.parallel
    if args.max_workers:
        clone_config.max_workers = args.max_workers

    # Initialize GitHub API client
    github_api = GitHubAPI(
        token=clone_config.github_token,
        api_url=clone_config.github_api_url,
        wait_on_rate_limit=clone_config.github_wait_on_rate_limit,
        max_retries=clone_config.github_max_retries,
        retry_delay=clone_config.github_retry_delay,
    )

    # Initialize repository cloner
    repo_cloner = RepoCloner(clone_config)

    # Add cloned repositories directory to .gitignore
    create_gitignore_entry(clone_config.output_dir)

    # Collect results for report
    results: dict[str, Any] = {
        "total_repos": 0,
        "success_repos": 0,
        "target_repos": [],
        "target_users": [],
    }

    try:
        # Clone specific repositories
        if args.clone_repos:
            target_repos = load_target_repos_config(config_dir)
            logger.info(f"Cloning {len(target_repos)} target repositories")

            for repo_info in target_repos:
                owner = repo_info.owner
                repo = repo_info.repo
                branch = repo_info.branch
                if not owner or not repo:
                    logger.warning(
                        f"Skipping invalid repository configuration: {repo_info}"
                    )
                    continue

                results["total_repos"] += 1
                try:
                    success = repo_cloner.clone_repository(owner, repo, branch)
                    if success:
                        results["success_repos"] += 1

                    results["target_repos"].append(
                        {"owner": owner, "repo": repo, "success": success}
                    )
                except Exception as e:
                    logger.error(f"Error cloning repository {owner}/{repo}: {e}")
                    results["target_repos"].append(
                        {
                            "owner": owner,
                            "repo": repo,
                            "success": False,
                            "error": str(e),
                        }
                    )

        if args.clone_users:
            target_users = load_target_users_config(config_dir)
            logger.info(f"Cloning repositories for {len(target_users)} users")

            for user_info in target_users:
                username = user_info.username
                if not username:
                    logger.warning(f"Skipping user with no username: {user_info}")
                    continue

                include_repos = user_info.include_repos
                exclude_repos = user_info.exclude_repos
                max_repos = user_info.max_repos

                try:
                    user_repos = github_api.get_user_repositories(
                        username, include_repos, exclude_repos, max_repos
                    )

                    logger.info(
                        f"Found {len(user_repos)} repositories for user {username}"
                    )

                    success_count, total_count = (
                        repo_cloner.clone_repositories_for_user(username, user_repos)
                    )

                    results["total_repos"] += total_count
                    results["success_repos"] += success_count

                    user_result = {"username": username, "repos": []}

                    for repo in user_repos:
                        repo_name = getattr(repo, "name", "")
                        repo_path = os.path.join(
                            clone_config.output_dir,
                            username,
                            repo_name,
                        )
                        # A clone is only successful when a real git repo
                        # exists (has .git), not merely an empty/partial dir
                        # left behind by an interrupted clone.
                        success = os.path.isdir(repo_path) and os.path.isdir(
                            os.path.join(repo_path, ".git")
                        )

                        user_result["repos"].append(
                            {"name": repo_name, "success": success}
                        )

                    results["target_users"].append(user_result)
                except Exception as e:
                    logger.error(f"Error processing user {username}: {e}")
                    results["target_users"].append(
                        {"username": username, "error": str(e), "repos": []}
                    )

        # Generate report
        if args.generate_report:
            report_file = generate_report(
                results, clone_config.output_dir, clone_config.report_format
            )
            logger.info(f"Cloning report generated: {report_file}")

        # Print summary
        success_rate = (
            (results["success_repos"] / results["total_repos"] * 100)
            if results["total_repos"] > 0
            else 0
        )
        logger.info(
            f"Cloning complete: {results['success_repos']}/{results['total_repos']} repositories cloned successfully ({success_rate:.2f}%)"
        )

    except KeyboardInterrupt:
        logger.info("Operation canceled by user.")
        raise SystemExit(1) from None
    finally:
        # Clean up resources
        repo_cloner.close()


if __name__ == "__main__":
    main()
