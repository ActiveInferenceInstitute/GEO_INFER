#!/usr/bin/env python3

"""
Integration example: GEO-INFER-GIT with GEO-INFER-AI

This example demonstrates how GEO-INFER-GIT can be used to automatically
clone and manage AI/ML repositories containing geospatial models,
algorithms, and research code for integration with the GEO-INFER-AI module.

Requires network access to the GitHub API. Set GITHUB_TOKEN to raise rate
limits (optional).
"""

import logging
from pathlib import Path

from geo_infer_git.core.github_api import GitHubAPI, GitHubRepository
from geo_infer_git.core.repo_cloner import RepoCloner
from geo_infer_git.utils.config_loader import CloneConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_ai_research_integration() -> tuple[GitHubAPI, RepoCloner]:
    """
    Set up integration between GEO-INFER-GIT and GEO-INFER-AI for
    automated AI/ML research repository acquisition and management.
    """
    # GitHub API client (uses GITHUB_TOKEN from the environment when set)
    api_client = GitHubAPI()

    # Configuration for cloning operations
    clone_config = CloneConfig(
        output_dir="./ai_research_models",
        concurrency_enabled=True,
        max_workers=4,
        default_branch="main",
        clone_depth=1,
    )

    # Initialize repository cloner
    cloner = RepoCloner(clone_config)

    return api_client, cloner


def discover_ai_research_repositories(
    api_client: GitHubAPI, cloner: RepoCloner
) -> tuple[dict[str, bool], list[GitHubRepository]]:
    """
    Discover and clone AI research repositories containing geospatial models.

    This function demonstrates how to use GEO-INFER-GIT to find and
    clone repositories with AI/ML models and algorithms for geospatial
    analysis, suitable for integration with GEO-INFER-AI.
    """
    logger.info("Discovering AI research repositories...")

    # Search for AI/geospatial repositories
    search_queries = [
        "geospatial AI models",
        "spatial machine learning",
        "GIS deep learning",
        "remote sensing neural networks",
        "geographic data science",
    ]

    ai_repositories: list[GitHubRepository] = []

    for query in search_queries:
        # Search GitHub for relevant repositories
        search_results = api_client.search_repositories(
            query,
            language="Python",  # Focus on Python implementations
            stars=">=10",  # Quality filter
            max_results=20,
        )

        # Filter for AI/ML focused repositories
        filtered_results = api_client.filter_repositories(
            search_results,
            min_stars=5,
            languages=["Python", "R", "Julia"],
            exclude_forks=True,
            has_topics=[
                "machine-learning",
                "deep-learning",
                "neural-networks",
                "computer-vision",
            ],
        )

        ai_repositories.extend(filtered_results)

    # Remove duplicates based on full_name
    seen: set[str] = set()
    unique_repositories: list[GitHubRepository] = []
    for repo in ai_repositories:
        if repo.full_name not in seen:
            seen.add(repo.full_name)
            unique_repositories.append(repo)

    logger.info(f"Found {len(unique_repositories)} unique AI research repositories")

    # Clone selected repositories
    logger.info("Cloning AI research repositories...")
    repositories_to_clone: list[tuple[str, str, str]] = []

    for repo in unique_repositories[:15]:  # Clone top 15
        repositories_to_clone.append((repo.owner, repo.name, repo.default_branch))

    # Clone repositories
    clone_results = cloner.clone_multiple_repositories(repositories_to_clone)

    # Report results
    success_count = sum(1 for success in clone_results.values() if success)
    total_count = len(clone_results)

    logger.info(f"Cloned {success_count}/{total_count} AI research repositories")

    return clone_results, unique_repositories


def analyze_model_compatibility(
    repositories: list[GitHubRepository],
) -> dict[str, object]:
    """
    Analyze cloned repositories for model compatibility with GEO-INFER-AI.

    This demonstrates how to analyze cloned AI repositories to determine
    their compatibility with geospatial AI workflows in GEO-INFER-AI.
    """

    logger.info("Analyzing model compatibility...")

    models_dir = Path("./ai_research_models")
    compatibility_report: dict[str, object] = {
        "tensorflow_models": [],
        "pytorch_models": [],
        "scikit_learn_models": [],
        "other_frameworks": [],
        "geospatial_specific": [],
        "total_analyzed": 0,
    }

    for repo_path in models_dir.rglob("*"):
        if repo_path.is_dir() and (repo_path / ".git").exists():
            repo_name = repo_path.name

            # Analyze repository for AI frameworks and geospatial focus
            framework_detected = None
            geospatial_focus = False

            # Check for common AI framework files
            framework_indicators = {
                "tensorflow": ["tensorflow", "tf", "keras"],
                "pytorch": ["torch", "pytorch", "nn.Module"],
                "scikit_learn": ["sklearn", "scikit-learn", "RandomForest", "SVM"],
            }

            for indicator_file in repo_path.rglob("*.py"):
                try:
                    with open(indicator_file, encoding="utf-8") as f:
                        content = f.read().lower()

                        # Check for geospatial indicators
                        geospatial_terms = [
                            "gis",
                            "geospatial",
                            "spatial",
                            "coordinate",
                            "latitude",
                            "longitude",
                        ]
                        if any(term in content for term in geospatial_terms):
                            geospatial_focus = True

                        # Check for AI frameworks
                        for framework, indicators in framework_indicators.items():
                            if any(indicator in content for indicator in indicators):
                                framework_detected = framework
                                break

                        if framework_detected:
                            break

                except Exception:
                    continue

            # Categorize repository
            if framework_detected:
                if geospatial_focus:
                    compatibility_report["geospatial_specific"].append(
                        {
                            "repository": repo_name,
                            "framework": framework_detected,
                            "geospatial_focus": True,
                        }
                    )

                compatibility_report[f"{framework_detected}_models"].append(
                    {
                        "repository": repo_name,
                        "framework": framework_detected,
                        "geospatial_focus": geospatial_focus,
                    }
                )

                compatibility_report["total_analyzed"] += 1

    logger.info(f"Analyzed {compatibility_report['total_analyzed']} AI repositories")

    return compatibility_report


def integrate_with_ai_module(
    clone_results: dict[str, bool], compatibility_report: dict[str, object]
) -> dict[str, object]:
    """
    Demonstrate integration with GEO-INFER-AI module.

    This shows how cloned AI repositories can be integrated
    with model training and inference workflows in GEO-INFER-AI.
    """

    logger.info("Integrating with GEO-INFER-AI workflows...")

    models_dir = Path("./ai_research_models")

    # Example integration points with GEO-INFER-AI:
    integration_points = {
        "model_discovery": {
            "tensorflow_models": len(compatibility_report["tensorflow_models"]),
            "pytorch_models": len(compatibility_report["pytorch_models"]),
            "scikit_learn_models": len(compatibility_report["scikit_learn_models"]),
            "geospatial_models": len(compatibility_report["geospatial_specific"]),
        },
        "framework_support": {
            "tensorflow": True,
            "pytorch": True,
            "scikit_learn": True,
            "keras": True,
            "xgboost": True,
        },
        "deployment_ready": {
            "containerized_models": 0,
            "api_endpoints": 0,
            "documentation": 0,
        },
    }

    # Scan for deployment-ready models
    for repo_path in models_dir.rglob("*"):
        if repo_path.is_dir() and (repo_path / ".git").exists():
            # Check for deployment indicators
            if (repo_path / "Dockerfile").exists():
                integration_points["deployment_ready"]["containerized_models"] += 1

            if (repo_path / "requirements.txt").exists():
                integration_points["deployment_ready"]["api_endpoints"] += 1

            if (repo_path / "README.md").exists():
                integration_points["deployment_ready"]["documentation"] += 1

    integration_report = {
        "models_directory": str(models_dir),
        "integration_points": integration_points,
        "total_repositories": len(clone_results),
        "successful_clones": sum(1 for success in clone_results.values() if success),
        "ai_integration_ready": True,
    }

    return integration_report


def create_model_catalog(compatibility_report: dict[str, object]) -> dict[str, object]:
    """
    Create a catalog of available AI models for GEO-INFER-AI integration.

    This demonstrates how to create a structured catalog of AI models
    that can be used by GEO-INFER-AI for model selection and deployment.
    """

    logger.info("Creating AI model catalog...")

    catalog: dict[str, object] = {
        "metadata": {
            "created_at": str(Path("./ai_research_models").stat().st_ctime)
            if Path("./ai_research_models").exists()
            else "",
            "total_models": sum(
                len(models)
                for models in compatibility_report.values()
                if isinstance(models, list)
            ),
            "frameworks_supported": list(
                set(
                    model.get("framework")
                    for category in compatibility_report.values()
                    if isinstance(category, list)
                    for model in category
                )
            ),
        },
        "models": {},
    }

    # Organize models by framework
    framework_categories = [
        "tensorflow_models",
        "pytorch_models",
        "scikit_learn_models",
        "geospatial_specific",
    ]

    for category in framework_categories:
        models = compatibility_report.get(category, [])
        if models:
            framework = (
                models[0].get("framework")
                if models
                else category.replace("_models", "")
            )

            catalog["models"][framework] = {
                "count": len(models),
                "repositories": [model["repository"] for model in models],
                "geospatial_focus": any(
                    model.get("geospatial_focus", False) for model in models
                ),
            }

    # Save catalog
    catalog_file = Path("./ai_model_catalog.json")
    try:
        import json

        with open(catalog_file, "w") as f:
            json.dump(catalog, f, indent=2)
        logger.info(f"Model catalog saved to {catalog_file}")
    except Exception as e:
        logger.warning(f"Failed to save model catalog: {e}")

    return catalog


def main() -> dict[str, object] | None:
    """
    Main integration example demonstrating GEO-INFER-GIT with GEO-INFER-AI.
    """

    logger.info("Starting GEO-INFER-GIT and GEO-INFER-AI integration example")

    try:
        # Set up integration components
        api_client, cloner = setup_ai_research_integration()

        # Discover and clone AI research repositories
        clone_results, repositories = discover_ai_research_repositories(
            api_client, cloner
        )

        # Analyze model compatibility
        compatibility_report = analyze_model_compatibility(repositories)

        # Create integration with AI workflows
        integration_report = integrate_with_ai_module(
            clone_results, compatibility_report
        )

        # Create model catalog
        model_catalog = create_model_catalog(compatibility_report)

        # Generate final report
        final_report = {
            "integration": integration_report,
            "compatibility": compatibility_report,
            "model_catalog": model_catalog,
            "clone_results": clone_results,
        }

        logger.info("AI integration example completed successfully")
        logger.info(f"Final report: {final_report}")

        return final_report

    except Exception as e:
        logger.error(f"AI integration example failed: {e}")
        raise
    finally:
        # Clean up resources
        if "cloner" in locals():
            cloner.close()
        if "api_client" in locals():
            api_client.close()


if __name__ == "__main__":
    main()
