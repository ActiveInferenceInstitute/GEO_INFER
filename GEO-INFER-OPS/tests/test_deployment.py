"""
Tests for deployment management.
"""

import subprocess
from unittest.mock import patch, MagicMock, mock_open

import pytest
from kubernetes.client.rest import ApiException
from kubernetes.client.models.v1_deployment import V1Deployment
from kubernetes.client.models.v1_service import V1Service
from kubernetes.client.models.v1_config_map import V1ConfigMap
from kubernetes.client.models.v1_secret import V1Secret
from kubernetes.client.models.v1_pod_list import V1PodList
from kubernetes.client.models.v1_pod import V1Pod
from kubernetes.client.models.v1_container_status import V1ContainerStatus

from geo_infer_ops.core.deployment import DeploymentManager
from geo_infer_ops.core.config import (
    Config,
    DeploymentConfig,
    DockerConfig,
    KubernetesConfig,
)


@pytest.fixture
def mock_config():
    """Fixture providing a mock configuration."""
    return Config(
        deployment=DeploymentConfig(
            docker=DockerConfig(registry="test-registry", tag="test-tag"),
            kubernetes=KubernetesConfig(namespace="test-namespace"),
        )
    )


@pytest.fixture
def mock_k8s_client():
    """Fixture providing mock Kubernetes clients."""
    with (
        patch("kubernetes.client.AppsV1Api") as mock_apps,
        patch("kubernetes.client.CoreV1Api") as mock_core,
        patch("kubernetes.client.NetworkingV1Api") as mock_networking,
    ):
        yield {
            "apps": mock_apps.return_value,
            "core": mock_core.return_value,
            "networking": mock_networking.return_value,
        }


@pytest.fixture
def deployment_manager(mock_config):
    """Fixture providing a deployment manager instance."""
    with patch("geo_infer_ops.core.deployment.get_config") as mock_get_config:
        mock_get_config.return_value = mock_config
        manager = DeploymentManager()
        yield manager


def test_build_docker_image_success(deployment_manager):
    """Test successful Docker image build."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        result = deployment_manager.build_docker_image()
        assert result is True
        mock_run.assert_called_once_with(
            ["docker", "build", "-t", "test-tag", "."],
            check=True,
            timeout=1800,
        )


def test_build_docker_image_failure(deployment_manager):
    """Test failed Docker image build."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["docker", "build"])
        result = deployment_manager.build_docker_image()
        assert result is False


def test_push_docker_image_success(deployment_manager):
    """Test successful Docker image push."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        result = deployment_manager.push_docker_image()
        assert result is True
        mock_run.assert_called_once_with(
            ["docker", "push", "test-registry/test-tag"],
            check=True,
            timeout=900,
        )


def test_push_docker_image_failure(deployment_manager):
    """Test failed Docker image push."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["docker", "push"])
        result = deployment_manager.push_docker_image()
        assert result is False


def test_deploy_kubernetes_success(deployment_manager, mock_k8s_client):
    """Test successful Kubernetes deployment."""
    # Mock manifest files
    manifest_content = """
    kind: Deployment
    metadata:
      name: test-deployment
    spec:
      replicas: 1
    """
    with (
        patch("builtins.open", mock_open(read_data=manifest_content)),
        patch("pathlib.Path.glob") as mock_glob,
    ):
        mock_glob.return_value = ["test.yml"]

        # Mock Kubernetes API calls
        mock_k8s_client[
            "apps"
        ].create_namespaced_deployment.return_value = V1Deployment()

        result = deployment_manager.deploy_kubernetes()
        assert result is True
        mock_k8s_client["apps"].create_namespaced_deployment.assert_called_once()


def test_deploy_kubernetes_failure(deployment_manager, mock_k8s_client):
    """Test failed Kubernetes deployment."""
    with patch("builtins.open", mock_open(read_data="invalid yaml")):
        result = deployment_manager.deploy_kubernetes()
        assert result is False


def test_get_deployment_status_success(deployment_manager, mock_k8s_client):
    """Test successful deployment status retrieval."""
    mock_deployment = MagicMock(spec=V1Deployment)
    mock_deployment.metadata.name = "test-deployment"
    mock_deployment.spec.replicas = 3
    mock_deployment.status.available_replicas = 3
    mock_deployment.status.ready_replicas = 3
    mock_deployment.status.updated_replicas = 3
    mock_deployment.status.conditions = []

    mock_k8s_client["apps"].read_namespaced_deployment.return_value = mock_deployment

    status = deployment_manager.get_deployment_status("test-deployment")
    assert status["name"] == "test-deployment"
    assert status["replicas"] == 3
    assert status["available_replicas"] == 3
    assert status["ready_replicas"] == 3
    assert status["updated_replicas"] == 3
    assert status["conditions"] == []


def test_get_deployment_status_failure(deployment_manager, mock_k8s_client):
    """Test failed deployment status retrieval."""
    mock_k8s_client["apps"].read_namespaced_deployment.side_effect = ApiException(
        status=404
    )
    status = deployment_manager.get_deployment_status("test-deployment")
    assert status == {}


def test_scale_deployment_success(deployment_manager, mock_k8s_client):
    """Test successful deployment scaling."""
    mock_k8s_client["apps"].patch_namespaced_deployment_scale.return_value = MagicMock()
    result = deployment_manager.scale_deployment("test-deployment", 5)
    assert result is True
    mock_k8s_client["apps"].patch_namespaced_deployment_scale.assert_called_once()


def test_scale_deployment_failure(deployment_manager, mock_k8s_client):
    """Test failed deployment scaling."""
    mock_k8s_client[
        "apps"
    ].patch_namespaced_deployment_scale.side_effect = ApiException(status=404)
    result = deployment_manager.scale_deployment("test-deployment", 5)
    assert result is False


def test_get_pods_success(deployment_manager, mock_k8s_client):
    """Test successful pod listing."""
    mock_pod = MagicMock(spec=V1Pod)
    mock_pod.metadata.name = "test-pod"
    mock_pod.status.phase = "Running"
    mock_pod.status.pod_ip = "10.0.0.1"
    mock_pod.spec.node_name = "test-node"
    mock_pod.status.container_statuses = [MagicMock(spec=V1ContainerStatus)]

    mock_pod_list = MagicMock(spec=V1PodList)
    mock_pod_list.items = [mock_pod]

    mock_k8s_client["core"].list_namespaced_pod.return_value = mock_pod_list

    pods = deployment_manager.get_pods()
    assert len(pods) == 1
    assert pods[0]["name"] == "test-pod"
    assert pods[0]["status"] == "Running"
    assert pods[0]["ip"] == "10.0.0.1"
    assert pods[0]["node"] == "test-node"


def test_get_pods_failure(deployment_manager, mock_k8s_client):
    """Test failed pod listing."""
    mock_k8s_client["core"].list_namespaced_pod.side_effect = ApiException(status=404)
    pods = deployment_manager.get_pods()
    assert pods == []


def test_apply_deployment_update(deployment_manager, mock_k8s_client):
    """Test updating existing deployment."""
    mock_k8s_client["apps"].create_namespaced_deployment.side_effect = ApiException(
        status=409
    )
    mock_k8s_client["apps"].replace_namespaced_deployment.return_value = V1Deployment()

    manifest = {"kind": "Deployment", "metadata": {"name": "test-deployment"}}

    deployment_manager._apply_deployment(manifest)
    mock_k8s_client["apps"].replace_namespaced_deployment.assert_called_once()


def test_apply_service_update(deployment_manager, mock_k8s_client):
    """Test updating existing service."""
    mock_k8s_client["core"].create_namespaced_service.side_effect = ApiException(
        status=409
    )
    mock_k8s_client["core"].replace_namespaced_service.return_value = V1Service()

    manifest = {"kind": "Service", "metadata": {"name": "test-service"}}

    deployment_manager._apply_service(manifest)
    mock_k8s_client["core"].replace_namespaced_service.assert_called_once()


def test_apply_configmap_update(deployment_manager, mock_k8s_client):
    """Test updating existing configmap."""
    mock_k8s_client["core"].create_namespaced_config_map.side_effect = ApiException(
        status=409
    )
    mock_k8s_client["core"].replace_namespaced_config_map.return_value = V1ConfigMap()

    manifest = {"kind": "ConfigMap", "metadata": {"name": "test-configmap"}}

    deployment_manager._apply_configmap(manifest)
    mock_k8s_client["core"].replace_namespaced_config_map.assert_called_once()


def test_apply_secret_update(deployment_manager, mock_k8s_client):
    """Test updating existing secret."""
    mock_k8s_client["core"].create_namespaced_secret.side_effect = ApiException(
        status=409
    )
    mock_k8s_client["core"].replace_namespaced_secret.return_value = V1Secret()

    manifest = {"kind": "Secret", "metadata": {"name": "test-secret"}}

    deployment_manager._apply_secret(manifest)
    mock_k8s_client["core"].replace_namespaced_secret.assert_called_once()


def test_pod_lifecycle_management(deployment_manager, mock_k8s_client):
    """Test pod lifecycle management operations."""
    mock_pod = MagicMock(spec=V1Pod)
    mock_pod.metadata.name = "test-pod"
    mock_pod.status.phase = "Running"

    # Test pod deletion
    mock_k8s_client["core"].delete_namespaced_pod.return_value = mock_pod
    result = deployment_manager.delete_pod("test-pod")
    assert result is True
    mock_k8s_client["core"].delete_namespaced_pod.assert_called_once()

    # Test pod recreation
    mock_k8s_client["core"].create_namespaced_pod.return_value = mock_pod
    result = deployment_manager.recreate_pod("test-pod")
    assert result is True
    mock_k8s_client["core"].create_namespaced_pod.assert_called_once()


def test_deployment_rollback(deployment_manager, mock_k8s_client):
    """Test deployment rollback functionality."""
    mock_deployment = MagicMock(spec=V1Deployment)
    mock_deployment.metadata.name = "test-deployment"
    mock_deployment.spec.template.spec.containers[0].image = "test-image:old"

    # Mock rollback operation
    mock_k8s_client[
        "apps"
    ].create_namespaced_deployment_rollback.return_value = mock_deployment

    result = deployment_manager.rollback_deployment("test-deployment", "test-revision")
    assert result is True
    mock_k8s_client["apps"].create_namespaced_deployment_rollback.assert_called_once()


def test_resource_quota_handling(deployment_manager, mock_k8s_client):
    """Test resource quota validation and handling."""
    mock_deployment = MagicMock(spec=V1Deployment)
    mock_deployment.metadata.name = "test-deployment"
    mock_deployment.spec.template.spec.containers[0].resources = {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "200m", "memory": "256Mi"},
    }

    # Test resource quota validation
    result = deployment_manager.validate_resource_quota(mock_deployment)
    assert result is True

    # Test resource quota exceeded
    mock_k8s_client["core"].create_namespaced_resource_quota.side_effect = ApiException(
        status=403
    )
    result = deployment_manager.apply_resource_quota(mock_deployment)
    assert result is False


def test_health_check_monitoring(deployment_manager, mock_k8s_client):
    """Test health check monitoring functionality."""
    mock_pod = MagicMock(spec=V1Pod)
    mock_pod.metadata.name = "test-pod"
    mock_pod.status.conditions = [
        MagicMock(type="Ready", status="True"),
        MagicMock(type="Initialized", status="True"),
    ]

    # Test health check success
    mock_k8s_client["core"].read_namespaced_pod.return_value = mock_pod
    result = deployment_manager.check_pod_health("test-pod")
    assert result["healthy"] is True
    assert result["ready"] is True
    assert result["initialized"] is True

    # Test health check failure
    mock_pod.status.conditions[0].status = "False"
    result = deployment_manager.check_pod_health("test-pod")
    assert result["healthy"] is False
    assert result["ready"] is False


def test_network_policy_validation(deployment_manager, mock_k8s_client):
    """Test network policy validation and application."""
    mock_policy = {
        "kind": "NetworkPolicy",
        "metadata": {"name": "test-policy"},
        "spec": {
            "podSelector": {"matchLabels": {"app": "test"}},
            "policyTypes": ["Ingress", "Egress"],
        },
    }

    # Test network policy creation
    mock_k8s_client[
        "networking"
    ].create_namespaced_network_policy.return_value = MagicMock()
    result = deployment_manager.apply_network_policy(mock_policy)
    assert result is True
    mock_k8s_client["networking"].create_namespaced_network_policy.assert_called_once()

    # Test network policy validation
    invalid_policy = mock_policy.copy()
    del invalid_policy["spec"]["podSelector"]
    result = deployment_manager.validate_network_policy(invalid_policy)
    assert result is False


def test_deployment_scale_with_resource_limits(deployment_manager, mock_k8s_client):
    """Test deployment scaling with resource limits consideration."""
    mock_deployment = MagicMock(spec=V1Deployment)
    mock_deployment.metadata.name = "test-deployment"
    mock_deployment.spec.replicas = 1
    mock_deployment.spec.template.spec.containers[0].resources = {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "200m", "memory": "256Mi"},
    }

    # Test scaling up with sufficient resources
    mock_k8s_client["apps"].patch_namespaced_deployment_scale.return_value = MagicMock()
    result = deployment_manager.scale_deployment_with_resources("test-deployment", 3)
    assert result is True

    # Test scaling up with insufficient resources
    mock_k8s_client[
        "apps"
    ].patch_namespaced_deployment_scale.side_effect = ApiException(status=403)
    result = deployment_manager.scale_deployment_with_resources("test-deployment", 5)
    assert result is False


def test_pod_eviction_handling(deployment_manager, mock_k8s_client):
    """Test pod eviction handling and recovery."""
    mock_pod = MagicMock(spec=V1Pod)
    mock_pod.metadata.name = "test-pod"
    mock_pod.status.phase = "Running"

    # Test pod eviction
    mock_k8s_client["core"].create_namespaced_binding.return_value = MagicMock()
    result = deployment_manager.evict_pod("test-pod")
    assert result is True

    # Test pod recovery after eviction
    mock_k8s_client["core"].create_namespaced_pod.return_value = mock_pod
    result = deployment_manager.recover_evicted_pod("test-pod")
    assert result is True

    # Test failed pod recovery
    mock_k8s_client["core"].create_namespaced_pod.side_effect = ApiException(status=403)
    result = deployment_manager.recover_evicted_pod("test-pod")
    assert result is False


def test_deployment_update_strategy(deployment_manager, mock_k8s_client):
    """Test deployment update strategy handling."""
    mock_deployment = MagicMock(spec=V1Deployment)
    mock_deployment.metadata.name = "test-deployment"
    mock_deployment.spec.strategy.type = "RollingUpdate"
    mock_deployment.spec.strategy.rolling_update = MagicMock()
    mock_deployment.spec.strategy.rolling_update.max_surge = "25%"
    mock_deployment.spec.strategy.rolling_update.max_unavailable = "25%"

    # Test update strategy validation
    result = deployment_manager.validate_update_strategy(mock_deployment)
    assert result is True

    # Test update strategy modification
    mock_k8s_client["apps"].patch_namespaced_deployment.return_value = mock_deployment
    result = deployment_manager.update_deployment_strategy(
        "test-deployment", max_surge="50%", max_unavailable="0"
    )
    assert result is True
    mock_k8s_client["apps"].patch_namespaced_deployment.assert_called_once()
