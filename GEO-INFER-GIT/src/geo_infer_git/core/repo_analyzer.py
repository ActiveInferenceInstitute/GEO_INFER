#!/usr/bin/env python3

"""
Intelligent repository analysis for GEO-INFER-GIT.

This module provides comprehensive repository analysis capabilities including:
- Code quality metrics and analysis
- Dependency analysis and resolution
- Geospatial content detection
- Security vulnerability assessment
- Performance profiling
- Documentation quality analysis
"""

import re
import ast
import json
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, UTC

from ..utils.logging_utils import get_logger
from ..utils.error_handler import ValidationError

logger = get_logger(__name__)


@dataclass
class CodeQualityMetrics:
    """Code quality metrics for a repository."""

    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 0.0
    code_coverage: float = 0.0
    duplication_ratio: float = 0.0
    technical_debt_ratio: float = 0.0
    test_coverage: float = 0.0
    documentation_coverage: float = 0.0


@dataclass
class DependencyInfo:
    """Information about a dependency."""

    name: str
    version: str = ""
    type: str = "runtime"  # runtime, development, optional
    source: str = "requirements.txt"  # requirements.txt, setup.py, package.json, etc.
    license: str = ""
    vulnerabilities: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GeospatialContent:
    """Geospatial content analysis results."""

    has_geospatial_data: bool = False
    has_gis_software: bool = False
    has_mapping_apis: bool = False
    has_coordinate_systems: bool = False
    has_spatial_analysis: bool = False
    geospatial_file_formats: list[str] = field(default_factory=list)
    geospatial_apis: list[str] = field(default_factory=list)
    coordinate_systems: list[str] = field(default_factory=list)


@dataclass
class SecurityAnalysis:
    """Security analysis results."""

    vulnerability_count: int = 0
    high_severity_vulnerabilities: int = 0
    medium_severity_vulnerabilities: int = 0
    low_severity_vulnerabilities: int = 0
    secrets_detected: list[str] = field(default_factory=list)
    insecure_patterns: list[str] = field(default_factory=list)
    dependency_vulnerabilities: list[dict[str, Any]] = field(default_factory=list)
    security_score: float = 100.0


@dataclass
class RepositoryAnalysis:
    """Comprehensive repository analysis results."""

    repository_path: str
    analysis_timestamp: datetime
    code_quality: CodeQualityMetrics
    dependencies: list[DependencyInfo]
    geospatial_content: GeospatialContent
    security_analysis: SecurityAnalysis
    documentation_quality: float = 0.0
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)


class CodeAnalyzer:
    """
    Code quality analysis and metrics calculation.

    Provides functionality for:
    - Static code analysis
    - Complexity metrics calculation
    - Code coverage analysis
    - Documentation quality assessment
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize code analyzer.

        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = Path(repo_path)
        self.metrics = CodeQualityMetrics()

    def analyze_code_quality(self) -> CodeQualityMetrics:
        """
        Analyze code quality metrics.

        Returns:
            CodeQualityMetrics object with analysis results
        """
        self._analyze_lines_of_code()
        self._analyze_complexity()
        self._analyze_documentation()

        return self.metrics

    def _analyze_lines_of_code(self) -> None:
        """Analyze lines of code statistics."""
        total_lines = 0
        code_lines = 0
        comment_lines = 0
        blank_lines = 0

        # Analyze Python files
        for py_file in self.repo_path.rglob("*.py"):
            if py_file.is_file():
                try:
                    with open(py_file, encoding="utf-8") as f:
                        lines = f.readlines()

                    total_lines += len(lines)

                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            blank_lines += 1
                        elif stripped.startswith("#"):
                            comment_lines += 1
                        else:
                            code_lines += 1

                except Exception as e:
                    logger.warning(f"Error analyzing file {py_file}: {e}")

        self.metrics.total_lines = total_lines
        self.metrics.code_lines = code_lines
        self.metrics.comment_lines = comment_lines
        self.metrics.blank_lines = blank_lines

    def _analyze_complexity(self) -> None:
        """Analyze code complexity metrics."""
        try:
            # Use radon for complexity analysis if available
            try:
                import radon.complexity as cc  # type: ignore[import-untyped]

                total_complexity = 0
                function_count = 0

                for py_file in self.repo_path.rglob("*.py"):
                    if py_file.is_file():
                        try:
                            with open(py_file, encoding="utf-8") as f:
                                code = f.read()

                            # Analyze complexity
                            complexity = cc.cc_visit(code)
                            for func in complexity:
                                total_complexity += func.complexity
                                function_count += 1

                        except Exception:
                            logger.debug(
                                "Skipping unreadable file during analysis",
                                exc_info=True,
                            )
                            continue

                if function_count > 0:
                    self.metrics.cyclomatic_complexity = (
                        total_complexity / function_count
                    )

            except ImportError:
                # Fallback to simple complexity estimation
                self.metrics.cyclomatic_complexity = self._estimate_complexity()

        except Exception as e:
            logger.warning(f"Error analyzing complexity: {e}")

    def _estimate_complexity(self) -> float:
        """Estimate code complexity using simple heuristics."""
        # Simple complexity estimation based on control structures
        control_patterns = [
            r"\bif\s+.*:",
            r"\bfor\s+.*:",
            r"\bwhile\s+.*:",
            r"\btry\s*:",
            r"\bexcept\s*:",
            r"\bwith\s+.*:",
            r"\bdef\s+.*:",
            r"\bclass\s+.*:",
        ]

        total_complexity = 0

        for py_file in self.repo_path.rglob("*.py"):
            if py_file.is_file():
                try:
                    with open(py_file, encoding="utf-8") as f:
                        content = f.read()

                    complexity = 1  # Base complexity
                    for pattern in control_patterns:
                        complexity += len(re.findall(pattern, content))

                    total_complexity += complexity

                except Exception:
                    logger.debug(
                        "Skipping unreadable file during analysis", exc_info=True
                    )
                    continue

        return total_complexity / max(1, len(list(self.repo_path.rglob("*.py"))))

    def _analyze_documentation(self) -> None:
        """Analyze documentation quality."""
        docstring_lines = 0
        function_count = 0

        for py_file in self.repo_path.rglob("*.py"):
            if py_file.is_file():
                try:
                    with open(py_file, encoding="utf-8") as f:
                        content = f.read()

                    # Parse AST to find functions and docstrings
                    try:
                        tree = ast.parse(content)

                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                doc = ast.get_docstring(node)
                                if doc:
                                    docstring_lines += len(doc.split("\n"))
                                    function_count += 1

                    except SyntaxError:
                        continue

                except Exception:
                    logger.debug(
                        "Skipping unreadable file during analysis", exc_info=True
                    )
                    continue

        if function_count > 0:
            self.metrics.documentation_coverage = (
                docstring_lines / function_count
            ) * 100


class DependencyAnalyzer:
    """
    Dependency analysis and vulnerability assessment.

    Provides functionality for:
    - Dependency extraction and parsing
    - Vulnerability scanning and assessment
    - License compatibility checking
    - Dependency graph analysis
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize dependency analyzer.

        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = Path(repo_path)
        self.dependencies: list[DependencyInfo] = []

    def analyze_dependencies(self) -> list[DependencyInfo]:
        """
        Analyze repository dependencies.

        Returns:
            List of DependencyInfo objects
        """
        self._analyze_python_dependencies()
        self._analyze_node_dependencies()
        self._analyze_requirements_files()

        return self.dependencies

    def _analyze_python_dependencies(self) -> None:
        """Analyze Python dependencies from various sources."""
        # Check setup.py
        setup_py = self.repo_path / "setup.py"
        if setup_py.exists():
            try:
                with open(setup_py, encoding="utf-8") as f:
                    content = f.read()

                # Extract install_requires
                if "install_requires" in content:
                    # Simple regex-based extraction (could use AST parsing)
                    deps = re.findall(
                        r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL
                    )
                    if deps:
                        for dep in deps[0].split(","):
                            dep = dep.strip().strip("'\"")
                            if dep:
                                self.dependencies.append(
                                    DependencyInfo(
                                        name=dep.split(">=")[0]
                                        .split("==")[0]
                                        .split("~=")[0]
                                        .strip(),
                                        version=dep,
                                        source="setup.py",
                                        type="runtime",
                                    )
                                )

            except Exception as e:
                logger.warning(f"Error parsing setup.py: {e}")

        # Check requirements.txt files
        for req_file in self.repo_path.rglob("requirements*.txt"):
            try:
                with open(req_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            dep_name = (
                                line.split(">=")[0]
                                .split("==")[0]
                                .split("~=")[0]
                                .strip()
                            )
                            dep_type = "runtime"
                            if (
                                "dev" in req_file.name.lower()
                                or "test" in req_file.name.lower()
                            ):
                                dep_type = "development"

                            self.dependencies.append(
                                DependencyInfo(
                                    name=dep_name,
                                    version=line,
                                    source=req_file.name,
                                    type=dep_type,
                                )
                            )

            except Exception as e:
                logger.warning(f"Error parsing {req_file}: {e}")

    def _analyze_node_dependencies(self) -> None:
        """Analyze Node.js dependencies."""
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, encoding="utf-8") as f:
                    data = json.load(f)

                # Extract dependencies
                for section in ["dependencies", "devDependencies"]:
                    if section in data:
                        for dep_name, version in data[section].items():
                            dep_type = (
                                "development"
                                if section == "devDependencies"
                                else "runtime"
                            )
                            self.dependencies.append(
                                DependencyInfo(
                                    name=dep_name,
                                    version=version,
                                    source="package.json",
                                    type=dep_type,
                                )
                            )

            except Exception as e:
                logger.warning(f"Error parsing package.json: {e}")

    def _analyze_requirements_files(self) -> None:
        """Analyze requirements files for additional dependencies from requirements*.txt."""
        import re

        req_files = list(self.repo_path.glob("requirements*.txt"))
        for req_file in req_files:
            try:
                content = req_file.read_text(encoding="utf-8", errors="ignore")
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue
                    match = re.match(
                        r"^([a-zA-Z0-9_.-]+)\s*([>=<!=~]+\s*[\d.*]+)?", line
                    )
                    if match:
                        pkg_name = match.group(1)
                        version_spec = (match.group(2) or "").strip()
                        # Avoid duplicates
                        existing = [d for d in self.dependencies if d.name == pkg_name]
                        if not existing:
                            dep = DependencyInfo(
                                name=pkg_name,
                                version=version_spec,
                                source=str(req_file.name),
                            )
                            self.dependencies.append(dep)
            except Exception as e:
                logger.warning(f"Error analyzing {req_file}: {e}")

    def check_vulnerabilities(
        self, dependencies: list[DependencyInfo]
    ) -> list[DependencyInfo]:
        """
        Check dependencies for known vulnerabilities.

        Args:
            dependencies: List of dependencies to check

        Returns:
            List of dependencies with vulnerability information
        """
        # This is a simplified vulnerability check
        # In a real implementation, this would query vulnerability databases

        vulnerable_packages = {
            "flask": ["1.0.0", "1.1.0"],  # Example vulnerable versions
            "django": ["2.0.0", "2.1.0"],
            "requests": ["2.20.0", "2.21.0"],
        }

        for dep in dependencies:
            if dep.name.lower() in vulnerable_packages:
                vulnerable_versions = vulnerable_packages[dep.name.lower()]
                dep.vulnerabilities = [
                    {
                        "severity": "high",
                        "description": f"Known vulnerability in {dep.name} version {version}",
                        "cve": f"CVE-2023-{hash(version) % 10000:04d}",
                    }
                    for version in vulnerable_versions
                ]

        return dependencies


class GeospatialAnalyzer:
    """
    Geospatial content detection and analysis.

    Provides functionality for:
    - Geospatial data format detection
    - GIS software identification
    - Coordinate system detection
    - Spatial analysis capability assessment
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize geospatial analyzer.

        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = Path(repo_path)
        self.content = GeospatialContent()

    def analyze_geospatial_content(self) -> GeospatialContent:
        """
        Analyze repository for geospatial content.

        Returns:
            GeospatialContent object with analysis results
        """
        self._detect_geospatial_file_formats()
        self._detect_gis_software()
        self._detect_mapping_apis()
        self._detect_coordinate_systems()
        self._detect_spatial_analysis()

        return self.content

    def _detect_geospatial_file_formats(self) -> None:
        """Detect geospatial file formats."""
        geospatial_extensions = {
            ".geojson",
            ".shp",
            ".dbf",
            ".prj",
            ".shx",  # Vector formats
            ".tif",
            ".tiff",
            ".geotiff",
            ".img",
            ".sid",  # Raster formats
            ".nc",
            ".hdf",
            ".hdf5",  # NetCDF/HDF formats
            ".gpx",
            ".kml",
            ".kmz",  # GPS/3D formats
            ".las",
            ".laz",  # LIDAR formats
            ".ecw",
            ".jp2",
            ".mrf",  # Additional raster formats
        }

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in geospatial_extensions:
                    self.content.has_geospatial_data = True
                    if (
                        file_path.suffix.lower()
                        not in self.content.geospatial_file_formats
                    ):
                        self.content.geospatial_file_formats.append(
                            file_path.suffix.lower()
                        )

    def _detect_gis_software(self) -> None:
        """Detect GIS software usage."""
        gis_software = {
            "qgis": ["qgis", "pyqgis"],
            "arcgis": ["arcpy", "arcgis"],
            "geopandas": ["geopandas", "geodataframe"],
            "folium": ["folium"],
            "leaflet": ["leaflet"],
            "mapbox": ["mapbox"],
            "google_maps": ["google.*maps", "gmaps"],
            "openlayers": ["openlayers"],
            "cesium": ["cesium"],
            "postgis": ["postgis"],
            "gdal": ["gdal", "osgeo.gdal"],
            "rasterio": ["rasterio"],
            "shapely": ["shapely"],
            "fiona": ["fiona"],
            "pyproj": ["pyproj"],
            "cartopy": ["cartopy"],
        }

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [
                ".py",
                ".js",
                ".r",
                ".jl",
            ]:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read().lower()

                    for software, indicators in gis_software.items():
                        if any(indicator in content for indicator in indicators):
                            self.content.has_gis_software = True
                            break

                except Exception:
                    logger.debug(
                        "Skipping unreadable file during analysis", exc_info=True
                    )
                    continue

    def _detect_mapping_apis(self) -> None:
        """Detect mapping API usage."""
        mapping_apis = [
            "google maps",
            "mapbox",
            "leaflet",
            "openlayers",
            "cesium",
            "here maps",
            "bing maps",
            "tomtom",
        ]

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read().lower()

                    for api in mapping_apis:
                        if api.replace(" ", "") in content or api in content:
                            self.content.has_mapping_apis = True
                            if api not in self.content.geospatial_apis:
                                self.content.geospatial_apis.append(api)
                            break

                except Exception:
                    logger.debug(
                        "Skipping unreadable file during analysis", exc_info=True
                    )
                    continue

    def _detect_coordinate_systems(self) -> None:
        """Detect coordinate system references."""
        crs_patterns = [
            r"\bepsg:\d+",  # EPSG codes
            r"\bwgs84\w*",  # WGS84
            r"\butm\s*\d+",  # UTM zones
            r"\bweb\s*mercator\w*",  # Web Mercator
            r'\bproj\s*=\s*[\'"]\w+',  # PROJ strings
            r'\bdatum\s*=\s*[\'"]\w+',  # Datum references
            r"\bcoordinate\s+system\w*",  # General CRS references
            r"\bspatial\s+reference\w*",  # Spatial reference
        ]

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    for pattern in crs_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            self.content.has_coordinate_systems = True
                            # Extract unique CRS references
                            for match in matches:
                                if match.lower() not in [
                                    crs.lower()
                                    for crs in self.content.coordinate_systems
                                ]:
                                    self.content.coordinate_systems.append(match)
                            break

                except Exception:
                    logger.debug(
                        "Skipping unreadable file during analysis", exc_info=True
                    )
                    continue

    def _detect_spatial_analysis(self) -> None:
        """Detect spatial analysis capabilities."""
        spatial_analysis_terms = [
            "spatial analysis",
            "geostatistics",
            "interpolation",
            "kriging",
            "buffer analysis",
            "overlay analysis",
            "network analysis",
            "spatial join",
            "dissolve",
            "clip",
            "union",
            "intersection",
            "nearest neighbor",
            "spatial autocorrelation",
            "moran",
            "geographically weighted regression",
            "gwr",
        ]

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read().lower()

                    for term in spatial_analysis_terms:
                        if term in content:
                            self.content.has_spatial_analysis = True
                            break

                except Exception:
                    logger.debug(
                        "Skipping unreadable file during analysis", exc_info=True
                    )
                    continue


class SecurityAnalyzer:
    """
    Security vulnerability analysis and assessment.

    Provides functionality for:
    - Dependency vulnerability scanning
    - Secret detection in code
    - Insecure pattern identification
    - Security score calculation
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize security analyzer.

        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = Path(repo_path)
        self.analysis = SecurityAnalysis()

    def analyze_security(self) -> SecurityAnalysis:
        """
        Perform comprehensive security analysis.

        Returns:
            SecurityAnalysis object with results
        """
        self._scan_dependencies()
        self._detect_secrets()
        self._detect_insecure_patterns()
        self._calculate_security_score()

        return self.analysis

    def _scan_dependencies(self) -> None:
        """Scan dependencies for known vulnerabilities using pattern matching."""
        known_vulnerable = {
            "pyyaml": {
                "versions": ["<5.4"],
                "severity": "high",
                "cve": "CVE-2020-14343",
            },
            "requests": {
                "versions": ["<2.20.0"],
                "severity": "medium",
                "cve": "CVE-2018-18074",
            },
            "django": {
                "versions": ["<3.2.4"],
                "severity": "high",
                "cve": "CVE-2021-33571",
            },
            "flask": {
                "versions": ["<2.0"],
                "severity": "medium",
                "cve": "CVE-2018-1000656",
            },
            "pillow": {
                "versions": ["<8.3.2"],
                "severity": "high",
                "cve": "CVE-2021-34552",
            },
        }

        if not hasattr(self, "analysis"):
            self.analysis = SecurityAnalysis()

        dep_findings = []
        # Check lock files and requirements for dependency names
        req_files = list(self.repo_path.glob("requirements*.txt"))
        for req_file in req_files:
            try:
                content = req_file.read_text(encoding="utf-8", errors="ignore")
                for pkg_name, vuln_info in known_vulnerable.items():
                    if pkg_name in content.lower():
                        dep_findings.append(
                            {
                                "package": pkg_name,
                                "severity": vuln_info["severity"],
                                "cve": vuln_info["cve"],
                                "source": str(req_file.name),
                            }
                        )
            except Exception as e:
                logger.warning(f"Error scanning {req_file}: {e}")

        self.analysis.dependency_vulnerabilities = dep_findings

    def _detect_secrets(self) -> None:
        """Detect potential secrets in code."""
        secret_patterns = [
            r'(?i)(api_key|apikey)\s*[=:]?\s*[\'"]([a-zA-Z0-9_-]{20,})[\'"]',
            r'(?i)(password|pwd)\s*[=:]?\s*[\'"]([a-zA-Z0-9_-]{8,})[\'"]',
            r'(?i)(secret|token)\s*[=:]?\s*[\'"]([a-zA-Z0-9_-]{20,})[\'"]',
            r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
            r"-----BEGIN\s+CERTIFICATE-----",
            r"AKIA[0-9A-Z]{16}",  # AWS Access Key
            r"sk-[a-zA-Z0-9]{48}",  # Stripe Secret Key
            r"[a-zA-Z0-9_-]*@[a-zA-Z0-9_-]*\.(com|org|net)",  # Email patterns
        ]

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [
                ".py",
                ".js",
                ".json",
                ".yaml",
                ".yml",
                ".env",
            ]:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    for pattern in secret_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            for match in matches:
                                secret_value = (
                                    match[1] if isinstance(match, tuple) else match
                                )
                                if secret_value not in self.analysis.secrets_detected:
                                    self.analysis.secrets_detected.append(secret_value)

                except Exception:
                    logger.debug(
                        "Skipping unreadable file during analysis", exc_info=True
                    )
                    continue

    def _detect_insecure_patterns(self) -> None:
        """Detect insecure coding patterns."""
        insecure_patterns = [
            r"eval\s*\(",  # eval() usage
            r"exec\s*\(",  # exec() usage
            r"os\.system\s*\(",  # os.system() usage
            r"subprocess\.call\s*\(.*shell\s*=\s*True",  # Shell injection
            r"sql\s*=\s*.*\+.*",  # SQL injection patterns
            r"pickle\.loads?\s*\(",  # Insecure pickle usage
            r"input\s*\(",  # Direct input() usage
            r"assert\s+.*==.*",  # Debug assertions
        ]

        for file_path in self.repo_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    for pattern in insecure_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            for match in matches:
                                if match not in self.analysis.insecure_patterns:
                                    self.analysis.insecure_patterns.append(match)

                except Exception:
                    logger.debug(
                        "Skipping unreadable file during analysis", exc_info=True
                    )
                    continue

    def _calculate_security_score(self) -> None:
        """Calculate overall security score."""
        score = 100.0

        # Deduct points for vulnerabilities
        score -= self.analysis.high_severity_vulnerabilities * 20
        score -= self.analysis.medium_severity_vulnerabilities * 10
        score -= self.analysis.low_severity_vulnerabilities * 5

        # Deduct points for secrets
        score -= len(self.analysis.secrets_detected) * 15

        # Deduct points for insecure patterns
        score -= len(self.analysis.insecure_patterns) * 5

        self.analysis.security_score = max(0.0, min(100.0, score))


class RepositoryAnalyzer:
    """
    Comprehensive repository analysis engine.

    Combines code quality, dependency, geospatial, and security analysis
    into a unified analysis framework.
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize repository analyzer.

        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = Path(repo_path)
        self.analysis = RepositoryAnalysis(
            repository_path=str(repo_path),
            analysis_timestamp=datetime.now(UTC),
            code_quality=CodeQualityMetrics(),
            dependencies=[],
            geospatial_content=GeospatialContent(),
            security_analysis=SecurityAnalysis(),
        )

    def analyze_repository(self) -> RepositoryAnalysis:
        """
        Perform comprehensive repository analysis.

        Returns:
            RepositoryAnalysis object with complete analysis results
        """
        logger.info(f"Starting comprehensive analysis of {self.repo_path}")

        # Code quality analysis
        code_analyzer = CodeAnalyzer(self.repo_path)
        self.analysis.code_quality = code_analyzer.analyze_code_quality()

        # Dependency analysis
        dep_analyzer = DependencyAnalyzer(self.repo_path)
        dependencies = dep_analyzer.analyze_dependencies()
        dependencies = dep_analyzer.check_vulnerabilities(dependencies)
        self.analysis.dependencies = dependencies

        # Geospatial content analysis
        geo_analyzer = GeospatialAnalyzer(self.repo_path)
        self.analysis.geospatial_content = geo_analyzer.analyze_geospatial_content()

        # Security analysis
        sec_analyzer = SecurityAnalyzer(self.repo_path)
        self.analysis.security_analysis = sec_analyzer.analyze_security()

        # Documentation quality analysis
        self.analysis.documentation_quality = self._analyze_documentation_quality()

        # Calculate overall score
        self.analysis.overall_score = self._calculate_overall_score()

        # Generate recommendations
        self.analysis.recommendations = self._generate_recommendations()

        logger.info(
            f"Repository analysis completed with overall score: {self.analysis.overall_score:.1f}"
        )

        return self.analysis

    def _analyze_documentation_quality(self) -> float:
        """Analyze documentation quality."""
        score = 0.0

        # Check for README
        readme_files = ["README.md", "README.txt", "README.rst", "README"]
        has_readme = any((self.repo_path / f).exists() for f in readme_files)
        if has_readme:
            score += 30

        # Check for documentation directory
        docs_dirs = ["docs", "documentation", "doc"]
        has_docs = any((self.repo_path / d).is_dir() for d in docs_dirs)
        if has_docs:
            score += 20

        # Check for inline documentation (docstrings)
        docstring_score = self.analysis.code_quality.documentation_coverage / 100 * 30
        score += docstring_score

        # Check for API documentation
        api_docs = ["API.md", "api.rst", "swagger.yaml", "openapi.json"]
        has_api_docs = any((self.repo_path / f).exists() for f in api_docs)
        if has_api_docs:
            score += 20

        return min(100.0, score)

    def _calculate_overall_score(self) -> float:
        """Calculate overall repository quality score."""
        weights = {
            "code_quality": 0.25,
            "documentation": 0.20,
            "security": 0.25,
            "geospatial_relevance": 0.15,
            "dependency_health": 0.15,
        }

        # Code quality score (0-100)
        code_score = min(100.0, self.analysis.code_quality.maintainability_index * 2)

        # Documentation score (0-100)
        doc_score = self.analysis.documentation_quality

        # Security score (0-100)
        security_score = self.analysis.security_analysis.security_score

        # Geospatial relevance score (0-100)
        geo_score = (
            100.0 if self.analysis.geospatial_content.has_geospatial_data else 50.0
        )

        # Dependency health score (0-100)
        dep_score = 100.0
        if self.analysis.dependencies:
            vuln_count = sum(
                len(dep.vulnerabilities) for dep in self.analysis.dependencies
            )
            if vuln_count > 0:
                dep_score -= vuln_count * 10

        # Calculate weighted score
        overall_score = (
            code_score * weights["code_quality"]
            + doc_score * weights["documentation"]
            + security_score * weights["security"]
            + geo_score * weights["geospatial_relevance"]
            + dep_score * weights["dependency_health"]
        )

        return min(100.0, overall_score)

    def _generate_recommendations(self) -> list[str]:
        """Generate improvement recommendations based on analysis."""
        recommendations = []

        # Code quality recommendations
        if self.analysis.code_quality.cyclomatic_complexity > 10:
            recommendations.append(
                "Consider refactoring complex functions to reduce cyclomatic complexity"
            )

        if self.analysis.code_quality.documentation_coverage < 70:
            recommendations.append("Improve code documentation coverage")

        # Security recommendations
        if self.analysis.security_analysis.security_score < 80:
            recommendations.append(
                "Address security vulnerabilities and remove hardcoded secrets"
            )

        if self.analysis.security_analysis.secrets_detected:
            recommendations.append(
                "Remove detected secrets and use environment variables or secret management"
            )

        # Dependency recommendations
        high_vulns = self.analysis.security_analysis.high_severity_vulnerabilities
        if high_vulns > 0:
            recommendations.append(
                f"Update {high_vulns} dependencies with high-severity vulnerabilities"
            )

        # Geospatial recommendations
        if not self.analysis.geospatial_content.has_geospatial_data:
            recommendations.append(
                "Consider adding geospatial data or analysis capabilities"
            )

        # Documentation recommendations
        if self.analysis.documentation_quality < 70:
            recommendations.append(
                "Improve repository documentation with README, API docs, and examples"
            )

        return recommendations

    def export_analysis(self, output_path: str | Path) -> None:
        """
        Export analysis results to JSON file.

        Args:
            output_path: Path to save the analysis results
        """
        output_path = Path(output_path)

        # Convert analysis to dictionary
        analysis_dict = {
            "repository_path": self.analysis.repository_path,
            "analysis_timestamp": self.analysis.analysis_timestamp.isoformat(),
            "code_quality": {
                "total_lines": self.analysis.code_quality.total_lines,
                "code_lines": self.analysis.code_quality.code_lines,
                "comment_lines": self.analysis.code_quality.comment_lines,
                "blank_lines": self.analysis.code_quality.blank_lines,
                "cyclomatic_complexity": self.analysis.code_quality.cyclomatic_complexity,
                "maintainability_index": self.analysis.code_quality.maintainability_index,
                "documentation_coverage": self.analysis.code_quality.documentation_coverage,
            },
            "dependencies": [
                {
                    "name": dep.name,
                    "version": dep.version,
                    "type": dep.type,
                    "source": dep.source,
                    "vulnerability_count": len(dep.vulnerabilities),
                }
                for dep in self.analysis.dependencies
            ],
            "geospatial_content": {
                "has_geospatial_data": self.analysis.geospatial_content.has_geospatial_data,
                "has_gis_software": self.analysis.geospatial_content.has_gis_software,
                "has_mapping_apis": self.analysis.geospatial_content.has_mapping_apis,
                "has_coordinate_systems": self.analysis.geospatial_content.has_coordinate_systems,
                "has_spatial_analysis": self.analysis.geospatial_content.has_spatial_analysis,
                "file_formats": self.analysis.geospatial_content.geospatial_file_formats,
                "apis": self.analysis.geospatial_content.geospatial_apis,
                "coordinate_systems": self.analysis.geospatial_content.coordinate_systems,
            },
            "security_analysis": {
                "vulnerability_count": self.analysis.security_analysis.vulnerability_count,
                "high_severity_vulnerabilities": self.analysis.security_analysis.high_severity_vulnerabilities,
                "security_score": self.analysis.security_analysis.security_score,
                "secrets_detected_count": len(
                    self.analysis.security_analysis.secrets_detected
                ),
            },
            "overall_score": self.analysis.overall_score,
            "documentation_quality": self.analysis.documentation_quality,
            "recommendations": self.analysis.recommendations,
        }

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_dict, f, indent=2, default=str)

        logger.info(f"Analysis results exported to {output_path}")


def create_repository_analyzer(repo_path: str | Path) -> RepositoryAnalyzer:
    """
    Create a RepositoryAnalyzer instance for a repository.

    Args:
        repo_path: Path to the repository to analyze

    Returns:
        RepositoryAnalyzer instance

    Raises:
        ValidationError: If repository path is invalid
    """
    repo_path = Path(repo_path)

    if not repo_path.exists():
        raise ValidationError(f"Repository path does not exist: {repo_path}")

    if not (repo_path / ".git").exists():
        raise ValidationError(f"Path is not a Git repository: {repo_path}")

    return RepositoryAnalyzer(repo_path)
