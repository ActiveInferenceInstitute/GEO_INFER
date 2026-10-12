#!/usr/bin/env python3

"""
Unit tests for MultiPlatformAPI: GitLab/Bitbucket response mapping, request
construction against a stubbed HTTP session, LocalGitAPI discovery, and the
MultiPlatformAPI facade / create_platform_api config mapping.
"""

from unittest.mock import Mock

import pytest
import requests

from geo_infer_git.core.multi_platform_api import (
    BitbucketAPI,
    BitbucketRepository,
    GitLabAPI,
    GitLabRepository,
    LocalGitAPI,
    LocalRepository,
    MultiPlatformAPI,
    create_platform_api,
)

GITLAB_PAYLOAD = {
    "id": 42,
    "name": "test-repo",
    "path_with_namespace": "owner/test-repo",
    "namespace": {"name": "owner"},
    "description": "Test repository",
    "web_url": "https://gitlab.example/owner/test-repo",
    "http_url_to_repo": "https://gitlab.example/owner/test-repo.git",
    "ssh_url_to_repo": "git@gitlab.example:owner/test-repo.git",
    "default_branch": "main",
    "visibility": "public",
    "forks_count": 7,
    "statistics": {"repository_size": 2048},
    "created_at": "2026-01-01T00:00:00Z",
    "last_activity_at": "2026-02-01T00:00:00Z",
    "archived": False,
    "topics": ["geo"],
}

BITBUCKET_PAYLOAD = {
    "name": "test-repo",
    "full_name": "owner/test-repo",
    "owner": {"username": "owner"},
    "description": "Test repository",
    "links": {
        "html": {"href": "https://bitbucket.example/owner/test-repo"},
        "clone": [
            {"name": "https", "href": "https://bitbucket.example/owner/test-repo.git"},
            {"name": "ssh", "href": "git@bitbucket.example:owner/test-repo.git"},
        ],
    },
    "mainbranch": {"name": "main"},
    "language": "python",
    "forks_count": 3,
    "size": 4096,
    "created_on": "2026-01-01T00:00:00Z",
    "updated_on": "2026-02-01T00:00:00Z",
    "is_private": False,
}


def make_response(payload):
    """A 200 response stub with JSON payload."""
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.headers = {}
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class TestGitLabRepositoryMapping:
    """Test GitLabRepository.from_api_response field mapping."""

    def test_maps_api_response_fields(self):
        repo = GitLabRepository.from_api_response(GITLAB_PAYLOAD)

        assert repo.id == 42
        assert repo.name == "test-repo"
        assert repo.full_name == "owner/test-repo"
        assert repo.owner == "owner"
        assert repo.description == "Test repository"
        assert repo.url == "https://gitlab.example/owner/test-repo"
        assert repo.clone_url == "https://gitlab.example/owner/test-repo.git"
        assert repo.ssh_url == "git@gitlab.example:owner/test-repo.git"
        assert repo.default_branch == "main"
        assert repo.visibility == "public"
        assert repo.forks == 7
        assert repo.size == 2048
        assert repo.archived is False
        assert repo.topics == ["geo"]

    def test_missing_fields_use_defaults(self):
        repo = GitLabRepository.from_api_response({})

        assert repo.id == 0
        assert repo.name == ""
        assert repo.default_branch == "main"
        assert repo.visibility == "private"
        assert repo.stars == 0
        assert repo.topics == []


class TestBitbucketRepositoryMapping:
    """Test BitbucketRepository.from_api_response field mapping."""

    def test_maps_api_response_fields(self):
        repo = BitbucketRepository.from_api_response(BITBUCKET_PAYLOAD)

        assert repo.name == "test-repo"
        assert repo.full_name == "owner/test-repo"
        assert repo.owner == "owner"
        assert repo.url == "https://bitbucket.example/owner/test-repo"
        assert repo.clone_url == "https://bitbucket.example/owner/test-repo.git"
        assert repo.ssh_url == "git@bitbucket.example:owner/test-repo.git"
        assert repo.default_branch == "main"
        assert repo.is_private is False

    def test_missing_fields_use_defaults(self):
        repo = BitbucketRepository.from_api_response({})

        assert repo.name == ""
        assert repo.default_branch == "main"
        assert repo.is_private is False
        assert repo.topics == []


class TestGitLabAPI:
    """Test GitLabAPI request construction against a stubbed session."""

    @pytest.fixture
    def client(self):
        return GitLabAPI(
            token="gl-token",
            api_url="https://gitlab.example/api/v4",
            max_retries=0,
            retry_delay=0,
        )

    def test_token_sets_private_token_header(self):
        client = GitLabAPI(token="gl-token")

        assert client.session.headers["Private-Token"] == "gl-token"

    def test_no_token_leaves_private_token_header_unset(self, monkeypatch):
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        client = GitLabAPI()

        assert "Private-Token" not in client.session.headers

    def test_get_user_repositories_builds_paged_request(self, client, monkeypatch):
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        session = Mock()
        session.request.side_effect = [
            make_response([GITLAB_PAYLOAD]),
            make_response([]),
        ]
        client.session = session

        repos = client.get_user_repositories("alice", max_repos=5)

        assert len(repos) == 1
        assert repos[0].name == "test-repo"
        assert session.request.call_count == 2
        first_args, first_kwargs = session.request.call_args_list[0]
        assert first_args == (
            "GET",
            "https://gitlab.example/api/v4/users/alice/projects",
        )
        assert first_kwargs["params"]["page"] == 1
        assert first_kwargs["params"]["order_by"] == "last_activity_at"
        assert first_kwargs["params"]["sort"] == "desc"
        second_kwargs = session.request.call_args_list[1].kwargs
        assert second_kwargs["params"]["page"] == 2

    def test_include_filter_drops_non_matching_repos(self, client):
        session = Mock()
        session.request.side_effect = [
            make_response([GITLAB_PAYLOAD]),
            make_response([]),
        ]
        client.session = session

        repos = client.get_user_repositories(
            "alice", include_repos=["other-repo"], max_repos=5
        )

        assert repos == []

    def test_exclude_filter_drops_matching_repos(self, client):
        session = Mock()
        session.request.side_effect = [
            make_response([GITLAB_PAYLOAD]),
            make_response([]),
        ]
        client.session = session

        repos = client.get_user_repositories(
            "alice", exclude_repos=["test-repo"], max_repos=5
        )

        assert repos == []

    def test_get_repository_encodes_project_path(self, client):
        session = Mock()
        session.request.return_value = make_response(GITLAB_PAYLOAD)
        client.session = session

        repo = client.get_repository("owner", "test-repo")

        assert repo.name == "test-repo"
        assert session.request.call_args.args == (
            "GET",
            "https://gitlab.example/api/v4/projects/owner%2Ftest-repo",
        )

    def test_get_repository_falls_back_to_search(self, client):
        session = Mock()
        session.request.side_effect = [
            requests.ConnectionError("404"),
            make_response([GITLAB_PAYLOAD]),
        ]
        client.session = session

        repo = client.get_repository("owner", "test-repo")

        assert repo.name == "test-repo"
        search_kwargs = session.request.call_args.kwargs
        assert search_kwargs["params"]["search"] == "test-repo"

    def test_check_credentials_true_on_200(self, client):
        session = Mock()
        session.request.return_value = make_response({"login": "alice"})
        client.session = session

        assert client.check_credentials() is True

    def test_check_credentials_false_on_request_failure(self, client):
        session = Mock()
        session.request.side_effect = requests.ConnectionError("boom")
        client.session = session

        assert client.check_credentials() is False


class TestBitbucketAPI:
    """Test BitbucketAPI request construction against a stubbed session."""

    @pytest.fixture
    def client(self):
        return BitbucketAPI(
            username="bb-user",
            app_password="bb-pass",
            api_url="https://api.bitbucket.example/2.0",
            max_retries=0,
            retry_delay=0,
        )

    def test_basic_auth_header_from_credentials(self):
        import base64

        client = BitbucketAPI(username="bb-user", app_password="bb-pass")

        expected = base64.b64encode(b"bb-user:bb-pass").decode()
        assert client.session.headers["Authorization"] == f"Basic {expected}"

    def test_no_auth_header_without_credentials(self, monkeypatch):
        monkeypatch.delenv("BITBUCKET_USERNAME", raising=False)
        monkeypatch.delenv("BITBUCKET_APP_PASSWORD", raising=False)
        client = BitbucketAPI()

        assert "Authorization" not in client.session.headers

    def test_get_user_repositories_builds_paged_request(self, client):
        session = Mock()
        session.request.side_effect = [
            make_response({"values": [BITBUCKET_PAYLOAD]}),
            make_response({"values": []}),
        ]
        client.session = session

        repos = client.get_user_repositories("bb-user", max_repos=5)

        assert len(repos) == 1
        assert repos[0].name == "test-repo"
        first_args, first_kwargs = session.request.call_args_list[0]
        assert first_args == (
            "GET",
            "https://api.bitbucket.example/2.0/repositories/bb-user",
        )
        assert first_kwargs["params"]["page"] == 1
        assert first_kwargs["params"]["sort"] == "-updated_on"

    def test_get_repository_builds_endpoint(self, client):
        session = Mock()
        session.request.return_value = make_response(BITBUCKET_PAYLOAD)
        client.session = session

        repo = client.get_repository("owner", "test-repo")

        assert repo.name == "test-repo"
        assert session.request.call_args.args == (
            "GET",
            "https://api.bitbucket.example/2.0/repositories/owner/test-repo",
        )

    def test_check_credentials_false_on_request_failure(self, client):
        session = Mock()
        session.request.side_effect = requests.ConnectionError("boom")
        client.session = session

        assert client.check_credentials() is False


class TestLocalGitAPI:
    """Test LocalGitAPI discovery over tmp_path repositories."""

    def test_discover_repositories_finds_nested_git_dirs(self, tmp_path):
        (tmp_path / "repo-a" / ".git").mkdir(parents=True)
        (tmp_path / "group" / "repo-b" / ".git").mkdir(parents=True)
        (tmp_path / "not-a-repo").mkdir()

        api = LocalGitAPI(base_paths=[str(tmp_path)])
        repos = api.discover_repositories(max_depth=3)

        assert sorted(r.name for r in repos) == ["repo-a", "repo-b"]

    def test_max_depth_excludes_deep_repositories(self, tmp_path):
        (tmp_path / "group" / "repo-b" / ".git").mkdir(parents=True)

        api = LocalGitAPI(base_paths=[str(tmp_path)])

        assert api.discover_repositories(max_depth=1) == []

    def test_discover_skips_missing_base_paths(self, tmp_path):
        api = LocalGitAPI(base_paths=[str(tmp_path / "missing")])

        assert api.discover_repositories() == []

    def test_check_repository(self, tmp_path):
        git_dir = tmp_path / "repo-a" / ".git"
        git_dir.mkdir(parents=True)
        api = LocalGitAPI(base_paths=[str(tmp_path)])

        assert api.check_repository(str(tmp_path / "repo-a")) is True
        assert api.check_repository(str(tmp_path / "not-a-repo")) is False


class TestMultiPlatformAPI:
    """Test the MultiPlatformAPI facade dispatch."""

    def test_unconfigured_platform_get_user_repositories_raises(self):
        api = MultiPlatformAPI({})

        with pytest.raises(ValueError, match="not configured"):
            api.get_user_repositories("gitlab", "alice")

    def test_unconfigured_platform_get_repository_raises(self):
        api = MultiPlatformAPI({})

        with pytest.raises(ValueError, match="not configured"):
            api.get_repository("gitlab", "owner", "repo")

    def test_local_platform_discovers_repositories(self, tmp_path):
        (tmp_path / "repo-a" / ".git").mkdir(parents=True)
        api = MultiPlatformAPI({"local": {"base_paths": [str(tmp_path)]}})

        assert api.get_supported_platforms() == ["local"]
        assert api.check_credentials("local") is True

        repos = api.get_user_repositories("local", "ignored")
        assert [r.name for r in repos] == ["repo-a"]

    def test_check_credentials_unconfigured_platform_returns_false(self):
        api = MultiPlatformAPI({})

        assert api.check_credentials("gitlab") is False


class TestCreatePlatformAPI:
    """Test create_platform_api configuration mapping."""

    def test_builds_all_configured_clients(self, tmp_path):
        config = {
            "gitlab": {
                "token": "gl-token",
                "api_url": "https://gitlab.example/api/v4",
            },
            "bitbucket": {"username": "bb-user", "app_password": "bb-pass"},
            "local": {"base_paths": [str(tmp_path)]},
        }

        api = create_platform_api(config)

        assert set(api.get_supported_platforms()) == {"gitlab", "bitbucket", "local"}
        assert api.clients["gitlab"].token == "gl-token"
        assert api.clients["gitlab"].api_url == "https://gitlab.example/api/v4"
        assert api.clients["bitbucket"].username == "bb-user"
        assert api.clients["local"].base_paths == [tmp_path.resolve()]

    def test_unconfigured_platforms_are_absent(self):
        api = create_platform_api({})

        assert api.get_supported_platforms() == []


class TestLocalRepositoryMapping:
    """Test LocalRepository.from_path against a real repository."""

    def test_maps_repository_fields(self, tmp_path):
        import subprocess

        repo_dir = tmp_path / "local-repo"
        repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        (repo_dir / "README.md").write_text("# Local repo header line\n")

        repo = LocalRepository.from_path(repo_dir)

        assert repo.name == "local-repo"
        assert repo.description == "# Local repo header line"
        assert repo.remote_urls == []
        assert repo.path == str(repo_dir)

    def test_non_repository_uses_defaults(self, tmp_path):
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()

        repo = LocalRepository.from_path(plain_dir)

        assert repo.name == "plain"
        assert repo.default_branch == "main"
        assert repo.description == ""
