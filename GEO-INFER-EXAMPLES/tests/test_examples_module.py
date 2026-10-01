"""
Unit tests for GEO-INFER-EXAMPLES module.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geo_infer_examples import __version__


class TestExamplesModule:
    """Test basic module functionality."""

    def test_module_import(self):
        """Test that module can be imported."""
        import geo_infer_examples

        assert geo_infer_examples is not None

    def test_version(self):
        """Test version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_examples_directory_exists(self):
        """Test that examples directory exists."""
        examples_dir = Path(__file__).parent.parent / "examples"
        assert examples_dir.exists()
        assert examples_dir.is_dir()

    def test_docs_directory_exists(self):
        """Test that docs directory exists."""
        docs_dir = Path(__file__).parent.parent / "docs"
        assert docs_dir.exists()
        assert docs_dir.is_dir()

    def test_readme_exists(self):
        """Test that README exists."""
        readme = Path(__file__).parent.parent / "README.md"
        assert readme.exists()

    def test_pyproject_exists(self):
        """Test that pyproject.toml declares the module package."""
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject.exists()


class TestExamplesStructure:
    """Test examples directory structure."""

    def test_integration_examples_exist(self):
        """Test that integration examples exist."""
        examples_dir = Path(__file__).parent.parent / "examples"
        assert (examples_dir / "getting_started").exists()

    def test_module_orchestrators_exist(self):
        """Test that module orchestrators exist."""
        orchestrators_dir = (
            Path(__file__).parent.parent / "examples" / "module_orchestrators"
        )
        assert orchestrators_dir.exists()
        assert orchestrators_dir.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
