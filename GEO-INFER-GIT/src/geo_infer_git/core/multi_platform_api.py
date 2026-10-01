#!/usr/bin/env python3

"""
Multi-platform Git API client for GEO-INFER-GIT.

This module provides a unified interface for interacting with multiple
Git platforms including GitHub, GitLab, Bitbucket, and local repositories.
"""

import os
import time
import logging
from typing import Any, Protocol, cast
from dataclasses import dataclass
from pathlib import Path
import requests

from .github_api import GitHubAPI
from ..utils.error_handler import NetworkError

logger = logging.getLogger(__name__)


@dataclass
class GitLabRepository:
    """GitLab repository information."""

    id: int
    name: str
    full_name: str
    owner: str
    description: str
    url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    visibility: str
    stars: int
    forks: int
    size: int
    created_at: str
    updated_at: str
    last_activity_at: str
    archived: bool
    topics: list[str]

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "GitLabRepository":
        """Create repository object from GitLab API response."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            full_name=data.get("path_with_namespace", ""),
            owner=data.get("namespace", {}).get("name", ""),
            description=data.get("description", ""),
            url=data.get("web_url", ""),
            clone_url=data.get("http_url_to_repo", ""),
            ssh_url=data.get("ssh_url_to_repo", ""),
            default_branch=data.get("default_branch", "main"),
            visibility=data.get("visibility", "private"),
            stars=0,  # GitLab doesn't have stars
            forks=data.get("forks_count", 0),
            size=data.get("statistics", {}).get("repository_size", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("last_activity_at", ""),
            last_activity_at=data.get("last_activity_at", ""),
            archived=data.get("archived", False),
            topics=data.get("topics", []) if data.get("topics") else [],
        )


@dataclass
class BitbucketRepository:
    """Bitbucket repository information."""

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
    is_private: bool
    topics: list[str]

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "BitbucketRepository":
        """Create repository object from Bitbucket API response."""
        return cls(
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            owner=data.get("owner", {}).get("username", ""),
            description=data.get("description", ""),
            url=data.get("links", {}).get("html", {}).get("href", ""),
            clone_url=data.get("links", {}).get("clone", [{}])[0].get("href", ""),
            ssh_url=data.get("links", {})
            .get("clone", [{}])[-1]
            .get("href", ""),  # SSH is usually last
            default_branch=data.get("mainbranch", {}).get("name", "main"),
            language=data.get("language", ""),
            stars=0,  # Bitbucket doesn't have stars in the same way
            forks=data.get("forks_count", 0),
            size=data.get("size", 0),
            created_at=data.get("created_on", ""),
            updated_at=data.get("updated_on", ""),
            is_private=data.get("is_private", False),
            topics=[],  # Bitbucket doesn't have topics in basic API
        )


@dataclass
class LocalRepository:
    """Local Git repository information."""

    name: str
    path: str
    description: str
    default_branch: str
    remote_urls: list[str]
    size: int
    last_modified: str

    @classmethod
    def from_path(cls, path: Path) -> "LocalRepository":
        """Create repository object from local path."""
        try:
            import git

            repo = git.Repo(path)
            name = path.name

            # Get remote URLs
            remote_urls = []
            for remote in repo.remotes:
                remote_urls.extend([url for url in remote.urls])

            # Get description from README if available
            description = ""
            readme_files = ["README.md", "README.txt", "README.rst"]
            for readme in readme_files:
                readme_path = path / readme
                if readme_path.exists():
                    try:
                        with open(readme_path, encoding="utf-8") as f:
                            content = f.read()
                            # Extract first non-empty line as description
                            lines = [
                                line.strip()
                                for line in content.split("\n")
                                if line.strip()
                            ]
                            if lines:
                                description = lines[0][:200]  # Limit to 200 chars
                                break
                    except Exception as exc:
                        logger.warning(
                            "Could not extract description from %s: %s",
                            readme_path,
                            exc,
                        )

            return cls(
                name=name,
                path=str(path),
                description=description,
                default_branch=repo.active_branch.name if repo.heads else "main",
                remote_urls=remote_urls,
                size=path.stat().st_size if path.exists() else 0,
                last_modified=str(path.stat().st_mtime) if path.exists() else "",
            )

        except Exception as e:
            logger.warning(f"Error reading local repository {path}: {e}")
            return cls(
                name=path.name,
                path=str(path),
                description="",
                default_branch="main",
                remote_urls=[],
                size=path.stat().st_size if path.exists() else 0,
                last_modified=str(path.stat().st_mtime) if path.exists() else "",
            )


class PlatformAPI(Protocol):
    """Protocol for Git platform API clients."""

    def get_user_repositories(self, username: str, **kwargs: Any) -> list[Any]:
        """Get repositories for a user."""
        ...

    def get_repository(self, owner: str, repo: str) -> Any:
        """Get information about a specific repository."""
        ...

    def check_credentials(self) -> bool:
        """Check if credentials are valid."""
        ...


class GitLabAPI:
    """
    GitLab API client for repository operations.

    Provides methods for:
    - Repository discovery and metadata retrieval
    - User and group repository listing
    - Authentication management
    """

    def __init__(
        self,
        token: str | None = None,
        api_url: str = "https://gitlab.com/api/v4",
        wait_on_rate_limit: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize GitLab API client.

        Args:
            token: GitLab access token
            api_url: GitLab API base URL
            wait_on_rate_limit: Whether to wait when rate limit exceeded
            max_retries: Maximum number of retries for requests
            retry_delay: Delay between retries in seconds
        """
        self.token = token or os.environ.get("GITLAB_TOKEN", "")
        self.api_url = api_url.rstrip("/")
        self.wait_on_rate_limit = wait_on_rate_limit
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Setup session
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "GEO-INFER-GIT/1.0"}
        )

        if self.token:
            self.session.headers.update({"Private-Token": self.token})

    def _make_request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> requests.Response:
        """Make a request to GitLab API with error handling."""
        url = f"{self.api_url}{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)

                # Handle rate limiting
                if response.status_code == 429:
                    if self.wait_on_rate_limit and "Retry-After" in response.headers:
                        wait_time = int(response.headers["Retry-After"])
                        logger.warning(
                            f"Rate limit exceeded. Waiting {wait_time} seconds..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise NetworkError("Rate limit exceeded")

                response.raise_for_status()
                return response

            except requests.RequestException as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Request failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise

                wait_time = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)

    def get_user_repositories(
        self,
        username: str,
        include_repos: list[str] | None = None,
        exclude_repos: list[str] | None = None,
        max_repos: int = 100,
    ) -> list[GitLabRepository]:
        """Get repositories for a GitLab user."""
        repositories: list[GitLabRepository] = []
        page = 1
        per_page = min(100, max_repos)

        while len(repositories) < max_repos:
            endpoint = f"/users/{username}/projects"
            params = {
                "per_page": per_page,
                "page": page,
                "order_by": "last_activity_at",
                "sort": "desc",
            }

            response = self._make_request("GET", endpoint, params=params)
            repos_data = response.json()

            if not repos_data:
                break

            for repo_data in repos_data:
                repo = GitLabRepository.from_api_response(repo_data)
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

            if page > 10:
                logger.warning(f"Reached maximum pages (10) for user {username}")
                break

        logger.info(
            f"Found {len(repositories)} repositories for GitLab user {username}"
        )
        return repositories

    def get_repository(self, owner: str, repo: str) -> GitLabRepository:
        """Get information about a specific GitLab repository."""
        # Try multiple ways to find the repository
        try:
            # Method 1: By project path (owner/repo)
            endpoint = f"/projects/{owner}%2F{repo}"
            response = self._make_request("GET", endpoint)
            repo_data = response.json()
            return GitLabRepository.from_api_response(repo_data)

        except requests.RequestException:
            # Method 2: Search by name
            endpoint = "/projects"
            params = {"search": repo, "per_page": 1}

            response = self._make_request("GET", endpoint, params=params)
            projects = response.json()

            if projects:
                return GitLabRepository.from_api_response(projects[0])
            else:
                raise requests.RequestException(f"Repository {owner}/{repo} not found")

    def check_credentials(self) -> bool:
        """Check if GitLab credentials are valid."""
        try:
            response = self._make_request("GET", "/user")
            return bool(response.status_code == 200)
        except requests.RequestException:
            return False


class BitbucketAPI:
    """
    Bitbucket API client for repository operations.

    Provides methods for:
    - Repository discovery and metadata retrieval
    - User and workspace repository listing
    - Authentication management
    """

    def __init__(
        self,
        username: str | None = None,
        app_password: str | None = None,
        api_url: str = "https://api.bitbucket.org/2.0",
        wait_on_rate_limit: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize Bitbucket API client.

        Args:
            username: Bitbucket username
            app_password: Bitbucket app password
            api_url: Bitbucket API base URL
            wait_on_rate_limit: Whether to wait when rate limit exceeded
            max_retries: Maximum number of retries for requests
            retry_delay: Delay between retries in seconds
        """
        self.username = username or os.environ.get("BITBUCKET_USERNAME", "")
        self.app_password = app_password or os.environ.get("BITBUCKET_APP_PASSWORD", "")
        self.api_url = api_url.rstrip("/")
        self.wait_on_rate_limit = wait_on_rate_limit
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Setup session
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "GEO-INFER-GIT/1.0"}
        )

        if self.username and self.app_password:
            # Basic auth for Bitbucket
            auth_string = f"{self.username}:{self.app_password}"
            import base64

            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            self.session.headers.update({"Authorization": f"Basic {encoded_auth}"})

    def _make_request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> requests.Response:
        """Make a request to Bitbucket API with error handling."""
        url = f"{self.api_url}{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Request failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise

                wait_time = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)

    def get_user_repositories(
        self,
        username: str,
        include_repos: list[str] | None = None,
        exclude_repos: list[str] | None = None,
        max_repos: int = 100,
    ) -> list[BitbucketRepository]:
        """Get repositories for a Bitbucket user."""
        repositories: list[BitbucketRepository] = []
        page = 1
        per_page = min(100, max_repos)

        while len(repositories) < max_repos:
            endpoint = f"/repositories/{username}"
            params = {"pagelen": per_page, "page": page, "sort": "-updated_on"}

            response = self._make_request("GET", endpoint, params=params)
            repos_data = response.json()

            values = repos_data.get("values", [])
            if not values:
                break

            for repo_data in values:
                repo = BitbucketRepository.from_api_response(repo_data)
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

            if page > 10:
                logger.warning(f"Reached maximum pages (10) for user {username}")
                break

        logger.info(
            f"Found {len(repositories)} repositories for Bitbucket user {username}"
        )
        return repositories

    def get_repository(self, owner: str, repo: str) -> BitbucketRepository:
        """Get information about a specific Bitbucket repository."""
        endpoint = f"/repositories/{owner}/{repo}"
        response = self._make_request("GET", endpoint)
        repo_data = response.json()

        return BitbucketRepository.from_api_response(repo_data)

    def check_credentials(self) -> bool:
        """Check if Bitbucket credentials are valid."""
        try:
            response = self._make_request("GET", "/user")
            return bool(response.status_code == 200)
        except requests.RequestException:
            return False


class LocalGitAPI:
    """
    Local Git repository API for managing local repositories.

    Provides methods for:
    - Discovering local Git repositories
    - Getting repository information
    - Managing local repository operations
    """

    def __init__(self, base_paths: list[str] | None = None) -> None:
        """
        Initialize local Git API client.

        Args:
            base_paths: List of base directories to search for repositories
        """
        raw_paths: list[str] = base_paths or ["."]
        self.base_paths: list[Path] = [Path(p).resolve() for p in raw_paths]

    def discover_repositories(self, max_depth: int = 3) -> list[LocalRepository]:
        """
        Discover Git repositories in base paths.

        Args:
            max_depth: Maximum directory depth to search

        Returns:
            List of LocalRepository objects
        """
        repositories = []

        for base_path in self.base_paths:
            if not base_path.exists():
                continue

            # Search for .git directories
            for git_dir in base_path.rglob(".git"):
                if git_dir.is_dir():
                    repo_path = git_dir.parent

                    # Skip if too deep
                    relative_path = repo_path.relative_to(base_path)
                    depth = len(relative_path.parts)
                    if depth > max_depth:
                        continue

                    repo = LocalRepository.from_path(repo_path)
                    repositories.append(repo)

        logger.info(f"Discovered {len(repositories)} local repositories")
        return repositories

    def get_repository(self, path: str) -> LocalRepository:
        """Get information about a local repository."""
        repo_path = Path(path)
        return LocalRepository.from_path(repo_path)

    def check_repository(self, path: str) -> bool:
        """Check if a path contains a valid Git repository."""
        repo_path = Path(path)
        git_dir = repo_path / ".git"

        return git_dir.exists() and git_dir.is_dir()


class MultiPlatformAPI:
    """
    Unified API client for multiple Git platforms.

    Provides a single interface for working with GitHub, GitLab, Bitbucket,
    and local repositories.
    """

    def __init__(
        self, platform_configs: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """
        Initialize multi-platform API client.

        Args:
            platform_configs: Configuration for each platform
        """
        self.platform_configs = platform_configs or {}
        self.clients: dict[str, Any] = {}

        # Initialize platform clients
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize API clients for each platform."""
        # GitHub
        if "github" in self.platform_configs:
            github_config = self.platform_configs["github"]
            self.clients["github"] = GitHubAPI(**github_config)

        # GitLab
        if "gitlab" in self.platform_configs:
            gitlab_config = self.platform_configs["gitlab"]
            self.clients["gitlab"] = GitLabAPI(**gitlab_config)

        # Bitbucket
        if "bitbucket" in self.platform_configs:
            bitbucket_config = self.platform_configs["bitbucket"]
            self.clients["bitbucket"] = BitbucketAPI(**bitbucket_config)

        # Local
        if "local" in self.platform_configs:
            local_config = self.platform_configs["local"]
            self.clients["local"] = LocalGitAPI(**local_config)

    def get_user_repositories(
        self, platform: str, username: str, **kwargs: Any
    ) -> list[Any]:
        """
        Get repositories for a user across platforms.

        Args:
            platform: Platform name (github, gitlab, bitbucket, local)
            username: Username for the platform
            **kwargs: Additional arguments for the platform API

        Returns:
            List of repository objects
        """
        if platform not in self.clients:
            raise ValueError(f"Platform '{platform}' not configured")

        client = cast(PlatformAPI, self.clients[platform])

        if platform == "github":
            return client.get_user_repositories(username, **kwargs)
        elif platform == "gitlab":
            return client.get_user_repositories(username, **kwargs)
        elif platform == "bitbucket":
            return client.get_user_repositories(username, **kwargs)
        elif platform == "local":
            # For local, username is ignored, just discover repositories
            local_client: Any = self.clients[platform]
            return cast(list[Any], local_client.discover_repositories(**kwargs))
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def get_repository(self, platform: str, owner: str, repo: str) -> Any:
        """
        Get repository information across platforms.

        Args:
            platform: Platform name
            owner: Repository owner
            repo: Repository name

        Returns:
            Repository object
        """
        if platform not in self.clients:
            raise ValueError(f"Platform '{platform}' not configured")

        client = self.clients[platform]

        if platform in ["github", "gitlab", "bitbucket"]:
            return client.get_repository(owner, repo)
        elif platform == "local":
            # For local, owner/repo is interpreted as path
            return client.get_repository(f"{owner}/{repo}")
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def check_credentials(self, platform: str) -> bool:
        """
        Check if credentials are valid for a platform.

        Args:
            platform: Platform name

        Returns:
            True if credentials are valid
        """
        if platform not in self.clients:
            return False

        client = cast(PlatformAPI, self.clients[platform])

        if platform in ["github", "gitlab", "bitbucket"]:
            return client.check_credentials()
        elif platform == "local":
            return True  # Local doesn't need credentials
        else:
            return False

    def get_supported_platforms(self) -> list[str]:
        """Get list of supported platforms."""
        return list(self.clients.keys())


def create_platform_api(config: dict[str, Any]) -> MultiPlatformAPI:
    """
    Create a multi-platform API client from configuration.

    Args:
        config: Platform configuration

    Returns:
        MultiPlatformAPI instance
    """
    platform_configs = {}

    # GitHub configuration
    if "github" in config:
        github_config = config["github"]
        platform_configs["github"] = {
            "token": github_config.get("token"),
            "api_url": github_config.get("api_url", "https://api.github.com"),
            "wait_on_rate_limit": github_config.get("wait_on_rate_limit", True),
            "max_retries": github_config.get("max_retries", 3),
            "retry_delay": github_config.get("retry_delay", 1.0),
        }

    # GitLab configuration
    if "gitlab" in config:
        gitlab_config = config["gitlab"]
        platform_configs["gitlab"] = {
            "token": gitlab_config.get("token"),
            "api_url": gitlab_config.get("api_url", "https://gitlab.com/api/v4"),
            "wait_on_rate_limit": gitlab_config.get("wait_on_rate_limit", True),
            "max_retries": gitlab_config.get("max_retries", 3),
            "retry_delay": gitlab_config.get("retry_delay", 1.0),
        }

    # Bitbucket configuration
    if "bitbucket" in config:
        bitbucket_config = config["bitbucket"]
        platform_configs["bitbucket"] = {
            "username": bitbucket_config.get("username"),
            "app_password": bitbucket_config.get("app_password"),
            "api_url": bitbucket_config.get("api_url", "https://api.bitbucket.org/2.0"),
            "wait_on_rate_limit": bitbucket_config.get("wait_on_rate_limit", True),
            "max_retries": bitbucket_config.get("max_retries", 3),
            "retry_delay": bitbucket_config.get("retry_delay", 1.0),
        }

    # Local configuration
    if "local" in config:
        local_config = config["local"]
        platform_configs["local"] = {
            "base_paths": local_config.get("base_paths", ["."])
        }

    return MultiPlatformAPI(platform_configs)
