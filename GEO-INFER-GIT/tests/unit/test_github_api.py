#!/usr/bin/env python3

"""
Unit tests for GitHub API client.
"""

import pytest
import requests
from unittest.mock import Mock, patch
import time

from geo_infer_git.core.github_api import GitHubAPI, GitHubRepository, RateLimit


class TestGitHubRepository:
    """Test GitHubRepository dataclass."""

    def test_from_api_response(self):
        """Test creating repository from API response."""
        api_data = {
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "owner": {"login": "owner"},
            "description": "Test repository",
            "html_url": "https://github.com/owner/test-repo",
            "clone_url": "https://github.com/owner/test-repo.git",
            "ssh_url": "git@github.com:owner/test-repo.git",
            "default_branch": "main",
            "language": "Python",
            "stargazers_count": 100,
            "forks_count": 20,
            "size": 1024,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z",
            "pushed_at": "2023-01-03T00:00:00Z",
            "archived": False,
            "private": False,
            "fork": False,
            "topics": ["python", "api"],
        }

        repo = GitHubRepository.from_api_response(api_data)

        assert repo.name == "test-repo"
        assert repo.full_name == "owner/test-repo"
        assert repo.owner == "owner"
        assert repo.description == "Test repository"
        assert repo.url == "https://github.com/owner/test-repo"
        assert repo.clone_url == "https://github.com/owner/test-repo.git"
        assert repo.ssh_url == "git@github.com:owner/test-repo.git"
        assert repo.default_branch == "main"
        assert repo.language == "Python"
        assert repo.stars == 100
        assert repo.forks == 20
        assert repo.size == 1024
        assert repo.archived is False
        assert repo.private is False
        assert repo.fork is False
        assert repo.topics == ["python", "api"]

    def test_from_api_response_missing_fields(self):
        """Test creating repository from API response with missing fields."""
        api_data = {
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "owner": {"login": "owner"},
        }

        repo = GitHubRepository.from_api_response(api_data)

        assert repo.name == "test-repo"
        assert repo.description == ""
        assert repo.stars == 0
        assert repo.forks == 0
        assert repo.size == 0


class TestRateLimit:
    """Test RateLimit dataclass."""

    def test_properties(self):
        """Test RateLimit properties."""
        rate_limit = RateLimit(
            limit=5000, remaining=4999, reset_time=1234567890, used=1
        )

        assert rate_limit.limit == 5000
        assert rate_limit.remaining == 4999
        assert rate_limit.reset_time == 1234567890
        assert rate_limit.used == 1
        assert rate_limit.reset_datetime == 1234567890


class TestGitHubAPI:
    """Test GitHubAPI class."""

    def test_initialization(self):
        """Test GitHubAPI initialization."""
        api = GitHubAPI(token="test_token")

        assert api.token == "test_token"
        assert api.api_url == "https://api.github.com"
        assert api.wait_on_rate_limit is True
        assert api.max_retries == 3
        assert api.retry_delay == 1.0
        assert api.session is not None

    def test_initialization_with_environment_token(self):
        """Test GitHubAPI initialization with environment token."""
        with patch.dict("os.environ", {"GITHUB_TOKEN": "env_token"}):
            api = GitHubAPI()
            assert api.token == "env_token"

    @patch("requests.Session.request")
    def test_make_request_success(self, mock_request):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_request.return_value = mock_response

        api = GitHubAPI()
        response = api._make_request("GET", "/test")

        assert response.status_code == 200
        assert response.json() == {"test": "data"}
        mock_request.assert_called_once()

    @patch("requests.Session.request")
    def test_make_request_rate_limit_wait(self, mock_request):
        """Test API request with rate limit handling."""
        # First call returns 403 with rate limit headers
        mock_response_403 = Mock()
        mock_response_403.status_code = 403
        mock_response_403.headers = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time()) + 5),
        }

        # Second call returns 200
        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"test": "data"}

        mock_request.side_effect = [mock_response_403, mock_response_200]

        api = GitHubAPI(wait_on_rate_limit=True)
        response = api._make_request("GET", "/test")

        assert response.status_code == 200
        assert mock_request.call_count == 2

    @patch("requests.Session.request")
    def test_make_request_rate_limit_no_wait(self, mock_request):
        """Test API request with rate limit when not waiting."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time()) + 5),
        }
        mock_request.return_value = mock_response

        api = GitHubAPI(wait_on_rate_limit=False)

        with pytest.raises(requests.RequestException):
            api._make_request("GET", "/test")

    @patch("requests.Session.request")
    def test_make_request_retry_on_failure(self, mock_request):
        """Test API request retry on failure."""
        # First two calls fail, third succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = requests.RequestException(
            "Server error"
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"test": "data"}

        mock_request.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success,
        ]

        api = GitHubAPI(max_retries=2, retry_delay=0.1)
        response = api._make_request("GET", "/test")

        assert response.status_code == 200
        assert mock_request.call_count == 3

    @patch("requests.Session.request")
    def test_get_rate_limit(self, mock_request):
        """Test getting rate limit information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "rate": {"limit": 5000, "remaining": 4999, "reset": 1234567890, "used": 1}
        }
        mock_request.return_value = mock_response

        api = GitHubAPI()
        rate_limit = api.get_rate_limit()

        assert rate_limit.limit == 5000
        assert rate_limit.remaining == 4999
        assert rate_limit.reset_time == 1234567890
        assert rate_limit.used == 1

    @patch("requests.Session.request")
    def test_get_user_repositories(self, mock_request):
        """Test getting user repositories."""
        # Mock API responses for pagination
        mock_response_1 = Mock()
        mock_response_1.json.return_value = [
            {
                "name": "repo1",
                "full_name": "user/repo1",
                "owner": {"login": "user"},
                "description": "Repository 1",
            }
        ]

        mock_response_2 = Mock()
        mock_response_2.json.return_value = []  # Empty for end of pagination

        mock_request.side_effect = [mock_response_1, mock_response_2]

        api = GitHubAPI()
        repos = api.get_user_repositories("testuser", max_repos=1)

        assert len(repos) == 1
        assert repos[0].name == "repo1"
        assert repos[0].owner == "user"

    @patch("requests.Session.request")
    def test_get_repository(self, mock_request):
        """Test getting specific repository."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "owner": {"login": "owner"},
            "description": "Test repository",
        }
        mock_request.return_value = mock_response

        api = GitHubAPI()
        repo = api.get_repository("owner", "test-repo")

        assert repo.name == "test-repo"
        assert repo.owner == "owner"

    @patch("requests.Session.request")
    def test_search_repositories(self, mock_request):
        """Test repository search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "total_count": 1,
            "items": [
                {
                    "name": "search-repo",
                    "full_name": "owner/search-repo",
                    "owner": {"login": "owner"},
                }
            ],
        }
        mock_request.return_value = mock_response

        api = GitHubAPI()
        repos = api.search_repositories("test query", max_results=1)

        assert len(repos) == 1
        assert repos[0].name == "search-repo"

    @patch("requests.Session.request")
    def test_filter_repositories(self, mock_request):
        """Test repository filtering."""
        # Create test repositories
        repos = [
            GitHubRepository(
                name="repo1",
                full_name="owner/repo1",
                owner="owner",
                description="",
                url="",
                clone_url="",
                ssh_url="",
                default_branch="main",
                language="Python",
                stars=50,
                forks=5,
                size=1000,
                created_at="",
                updated_at="",
                pushed_at="",
                archived=False,
                private=False,
                fork=False,
                topics=["python"],
            ),
            GitHubRepository(
                name="repo2",
                full_name="owner/repo2",
                owner="owner",
                description="",
                url="",
                clone_url="",
                ssh_url="",
                default_branch="main",
                language="JavaScript",
                stars=10,
                forks=2,
                size=500,
                created_at="",
                updated_at="",
                pushed_at="",
                archived=False,
                private=False,
                fork=True,
                topics=["js"],
            ),
        ]

        api = GitHubAPI()

        # Test filtering by minimum stars
        filtered = api.filter_repositories(repos, min_stars=20)
        assert len(filtered) == 1
        assert filtered[0].name == "repo1"

        # Test filtering out forks
        filtered = api.filter_repositories(repos, exclude_forks=True)
        assert len(filtered) == 1
        assert filtered[0].name == "repo1"

    def test_get_repository_languages(self):
        """Test getting repository languages."""
        with patch("requests.Session.request") as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = {"Python": 1000, "JavaScript": 500}
            mock_request.return_value = mock_response

            api = GitHubAPI()
            languages = api.get_repository_languages("owner", "repo")

            assert languages == {"Python": 1000, "JavaScript": 500}

    def test_get_repository_topics(self):
        """Test getting repository topics."""
        with patch("requests.Session.request") as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = {"names": ["python", "api"]}
            mock_request.return_value = mock_response

            api = GitHubAPI()
            topics = api.get_repository_topics("owner", "repo")

            assert topics == ["python", "api"]

    def test_check_repository_exists(self):
        """Test checking if repository exists."""
        with patch.object(GitHubAPI, "get_repository") as mock_get_repo:
            mock_get_repo.return_value = GitHubRepository(
                name="test",
                full_name="owner/test",
                owner="owner",
                description="",
                url="",
                clone_url="",
                ssh_url="",
                default_branch="main",
                language="",
                stars=0,
                forks=0,
                size=0,
                created_at="",
                updated_at="",
                pushed_at="",
                archived=False,
                private=False,
                fork=False,
                topics=[],
            )

            api = GitHubAPI()
            exists = api.check_repository_exists("owner", "test")

            assert exists is True

    def test_check_repository_exists_not_found(self):
        """Test checking non-existent repository."""
        with patch.object(GitHubAPI, "get_repository") as mock_get_repo:
            mock_get_repo.side_effect = requests.RequestException("Not found")

            api = GitHubAPI()
            exists = api.check_repository_exists("owner", "nonexistent")

            assert exists is False

    def test_close(self):
        """Test API client cleanup."""
        api = GitHubAPI()
        api.close()

        # Should not raise any exceptions
        assert True
