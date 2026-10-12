#!/usr/bin/env python3

"""
GitHub API client for GEO-INFER-GIT.

This module provides functionality to interact with the GitHub API
for repository discovery, metadata retrieval, and rate limit management.
"""

import os
import time
import logging
from typing import Any, cast
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class GitHubRepository:
    """GitHub repository information."""

    name: str
    full_name: str
    owner: str
    description: str
    url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    language: str
    stars: int
    forks: int
    size: int
    created_at: str
    updated_at: str
    pushed_at: str
    archived: bool
    private: bool
    fork: bool
    topics: list[str]

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "GitHubRepository":
        """Create repository object from GitHub API response."""
        return cls(
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            owner=data.get("owner", {}).get("login", ""),
            description=data.get("description", ""),
            url=data.get("html_url", ""),
            clone_url=data.get("clone_url", ""),
            ssh_url=data.get("ssh_url", ""),
            default_branch=data.get("default_branch", "main"),
            language=data.get("language", ""),
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            size=data.get("size", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            pushed_at=data.get("pushed_at", ""),
            archived=data.get("archived", False),
            private=data.get("private", False),
            fork=data.get("fork", False),
            topics=data.get("topics", []),
        )


@dataclass
class RateLimit:
    """GitHub API rate limit information."""

    limit: int
    remaining: int
    reset_time: int
    used: int

    @property
    def reset_datetime(self) -> float:
        """Get reset time as datetime."""
        return self.reset_time


class GitHubAPI:
    """
    GitHub API client for repository operations.

    Provides methods for:
    - Repository discovery and metadata retrieval
    - Rate limit management
    - User and organization repository listing
    - Repository filtering and selection
    """

    def __init__(
        self,
        token: str | None = None,
        api_url: str = "https://api.github.com",
        wait_on_rate_limit: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize GitHub API client.

        Args:
            token: GitHub personal access token
            api_url: GitHub API base URL
            wait_on_rate_limit: Whether to wait when rate limit exceeded
            max_retries: Maximum number of retries for requests
            retry_delay: Delay between retries in seconds
        """
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.api_url = api_url.rstrip("/")
        self.wait_on_rate_limit = wait_on_rate_limit
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update(
            {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "GEO-INFER-GIT/1.0",
            }
        )

        if self.token:
            self.session.headers.update({"Authorization": f"token {self.token}"})

    def _make_request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> requests.Response:
        """
        Make a request to GitHub API with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.api_url}{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)

                # Handle rate limiting
                if (
                    response.status_code == 403
                    and "X-RateLimit-Remaining" in response.headers
                ):
                    remaining = int(response.headers.get("X-RateLimit-Remaining", "0"))
                    if remaining == 0:
                        reset_time = int(response.headers.get("X-RateLimit-Reset", "0"))
                        if self.wait_on_rate_limit:
                            wait_time = max(0, reset_time - int(time.time())) + 1
                            logger.warning(
                                f"Rate limit exceeded. Waiting {wait_time} seconds..."
                            )
                            time.sleep(wait_time)
                            continue
                        else:
                            raise requests.RequestException("Rate limit exceeded")

                response.raise_for_status()
                return response

            except requests.RequestException as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Request failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise

                wait_time = self.retry_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)

    def get_rate_limit(self) -> RateLimit:
        """
        Get current rate limit status.

        Returns:
            RateLimit object with current rate limit information
        """
        response = self._make_request("GET", "/rate_limit")
        data = response.json()

        core = data.get("rate", {})
        return RateLimit(
            limit=core.get("limit", 0),
            remaining=core.get("remaining", 0),
            reset_time=core.get("reset", 0),
            used=core.get("used", 0),
        )

    def get_user_repositories(
        self,
        username: str,
        include_repos: list[str] | None = None,
        exclude_repos: list[str] | None = None,
        max_repos: int = 100,
    ) -> list[GitHubRepository]:
        """
        Get repositories for a specific user.

        Args:
            username: GitHub username
            include_repos: List of repository names to include (empty for all)
            exclude_repos: List of repository names to exclude
            max_repos: Maximum number of repositories to return

        Returns:
            List of GitHubRepository objects
        """
        repositories: list[GitHubRepository] = []
        page = 1
        per_page = min(100, max_repos)

        while len(repositories) < max_repos:
            endpoint = f"/users/{username}/repos"
            params = {
                "sort": "updated",
                "direction": "desc",
                "per_page": per_page,
                "page": page,
            }

            response = self._make_request("GET", endpoint, params=params)
            repos_data = response.json()

            if not repos_data:
                break

            for repo_data in repos_data:
                repo = GitHubRepository.from_api_response(repo_data)
                repo_name = repo.name

                # Check include/exclude filters
                if include_repos and repo_name not in include_repos:
                    continue

                if exclude_repos and repo_name in exclude_repos:
                    continue

                repositories.append(repo)

                if len(repositories) >= max_repos:
                    break

            page += 1

            # Safety check to prevent infinite loops
            if page > 10:
                logger.warning(f"Reached maximum pages (10) for user {username}")
                break

        logger.info(f"Found {len(repositories)} repositories for user {username}")
        return repositories

    def get_organization_repositories(
        self,
        org_name: str,
        include_repos: list[str] | None = None,
        exclude_repos: list[str] | None = None,
        max_repos: int = 100,
    ) -> list[GitHubRepository]:
        """
        Get repositories for a specific organization.

        Args:
            org_name: GitHub organization name
            include_repos: List of repository names to include (empty for all)
            exclude_repos: List of repository names to exclude
            max_repos: Maximum number of repositories to return

        Returns:
            List of GitHubRepository objects
        """
        repositories: list[GitHubRepository] = []
        page = 1
        per_page = min(100, max_repos)

        while len(repositories) < max_repos:
            endpoint = f"/orgs/{org_name}/repos"
            params = {
                "sort": "updated",
                "direction": "desc",
                "per_page": per_page,
                "page": page,
            }

            response = self._make_request("GET", endpoint, params=params)
            repos_data = response.json()

            if not repos_data:
                break

            for repo_data in repos_data:
                repo = GitHubRepository.from_api_response(repo_data)
                repo_name = repo.name

                # Check include/exclude filters
                if include_repos and repo_name not in include_repos:
                    continue

                if exclude_repos and repo_name in exclude_repos:
                    continue

                repositories.append(repo)

                if len(repositories) >= max_repos:
                    break

            page += 1

            # Safety check to prevent infinite loops
            if page > 10:
                logger.warning(
                    f"Reached maximum pages (10) for organization {org_name}"
                )
                break

        logger.info(
            f"Found {len(repositories)} repositories for organization {org_name}"
        )
        return repositories

    def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """
        Get information about a specific repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            GitHubRepository object

        Raises:
            requests.RequestException: If repository not found or API error
        """
        endpoint = f"/repos/{owner}/{repo}"
        response = self._make_request("GET", endpoint)
        repo_data = response.json()

        return GitHubRepository.from_api_response(repo_data)

    def search_repositories(
        self,
        query: str,
        language: str | None = None,
        stars: str | None = None,
        forks: str | None = None,
        size: str | None = None,
        followers: str | None = None,
        license: str | None = None,
        sort: str = "updated",
        order: str = "desc",
        per_page: int = 30,
        max_results: int = 100,
    ) -> list[GitHubRepository]:
        """
        Search for repositories using GitHub's search API.

        Args:
            query: Search query
            language: Programming language filter
            stars: Stars filter (e.g., ">=100")
            forks: Forks filter (e.g., ">=10")
            size: Size filter in kilobytes (e.g., ">=1000")
            followers: Followers filter (e.g., ">=100")
            license: License filter (e.g., "mit")
            sort: Sort field (updated, created, stars, forks)
            order: Sort order (asc, desc)
            per_page: Results per page (max 100)
            max_results: Maximum total results

        Returns:
            List of GitHubRepository objects
        """
        repositories: list[GitHubRepository] = []
        page = 1

        while len(repositories) < max_results:
            endpoint = "/search/repositories"
            query_str = str(query)
            # Add optional filters
            filters = []
            if language:
                filters.append(f"language:{language}")
            if stars:
                filters.append(f"stars:{stars}")
            if forks:
                filters.append(f"forks:{forks}")
            if size:
                filters.append(f"size:{size}")
            if followers:
                filters.append(f"followers:{followers}")
            if license:
                filters.append(f"license:{license}")

            if filters:
                query_str = query_str + " " + " ".join(filters)

            params: dict[str, Any] = {
                "q": query_str,
                "sort": sort,
                "order": order,
                "per_page": min(per_page, 100),
                "page": page,
            }

            response = self._make_request("GET", endpoint, params=params)
            search_data = response.json()

            items = search_data.get("items", [])
            if not items:
                break

            for item in items:
                repo = GitHubRepository.from_api_response(item)
                repositories.append(repo)

                if len(repositories) >= max_results:
                    break

            page += 1

            # Check if we've reached the end
            if page > search_data.get("total_count", 0) // per_page + 1:
                break

            # Safety check
            if page > 10:
                logger.warning("Reached maximum search pages (10)")
                break

        logger.info(f"Search returned {len(repositories)} repositories")
        return repositories

    def filter_repositories(
        self,
        repositories: list[GitHubRepository],
        min_stars: int = 0,
        max_size: int | None = None,
        languages: list[str] | None = None,
        exclude_forks: bool = False,
        exclude_archived: bool = False,
        has_topics: list[str] | None = None,
    ) -> list[GitHubRepository]:
        """
        Filter repositories based on various criteria.

        Args:
            repositories: List of repositories to filter
            min_stars: Minimum number of stars
            max_size: Maximum repository size in KB
            languages: List of programming languages to include
            exclude_forks: Whether to exclude forked repositories
            exclude_archived: Whether to exclude archived repositories
            has_topics: List of topics that must be present

        Returns:
            Filtered list of GitHubRepository objects
        """
        filtered = []

        for repo in repositories:
            # Filter by stars
            if repo.stars < min_stars:
                continue

            # Filter by size
            if max_size and repo.size > max_size:
                continue

            # Filter by language
            if (
                languages
                and repo.language
                and repo.language.lower() not in [lang.lower() for lang in languages]
            ):
                continue

            # Filter out forks
            if exclude_forks and repo.fork:
                continue

            # Filter out archived repositories
            if exclude_archived and repo.archived:
                continue

            # Filter by topics
            if has_topics:
                repo_topics = [topic.lower() for topic in repo.topics]
                required_topics = [topic.lower() for topic in has_topics]
                if not all(topic in repo_topics for topic in required_topics):
                    continue

            filtered.append(repo)

        logger.info(
            f"Filtered {len(repositories)} repositories down to {len(filtered)}"
        )
        return filtered

    def get_repository_languages(self, owner: str, repo: str) -> dict[str, int]:
        """
        Get programming languages used in a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dictionary mapping language names to bytes of code
        """
        endpoint = f"/repos/{owner}/{repo}/languages"
        response = self._make_request("GET", endpoint)
        return cast(dict[str, int], response.json())

    def get_repository_topics(self, owner: str, repo: str) -> list[str]:
        """
        Get topics for a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            List of topic names
        """
        endpoint = f"/repos/{owner}/{repo}/topics"
        response = self._make_request("GET", endpoint)
        data = response.json()
        return cast(list[str], data.get("names", []))

    def check_repository_exists(self, owner: str, repo: str) -> bool:
        """
        Check if a repository exists.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            True if repository exists, False otherwise
        """
        try:
            self.get_repository(owner, repo)
            return True
        except requests.RequestException:
            return False

    def get_repository_contributors(
        self, owner: str, repo: str, max_contributors: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get top contributors for a repository.

        Args:
            owner: Repository owner
            repo: Repository name
            max_contributors: Maximum number of contributors to return

        Returns:
            List of contributor information dictionaries
        """
        endpoint = f"/repos/{owner}/{repo}/contributors"
        params = {"per_page": min(max_contributors, 100)}

        response = self._make_request("GET", endpoint, params=params)
        return cast(list[dict[str, Any]], response.json())[:max_contributors]

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
