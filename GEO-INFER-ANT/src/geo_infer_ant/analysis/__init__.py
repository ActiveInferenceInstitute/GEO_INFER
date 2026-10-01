"""
GEO-INFER-ANT Analysis Tools

This module provides comprehensive analysis tools for understanding emergent
behavior patterns in swarm intelligence systems, including spatial pattern
recognition, temporal analysis, interaction networks, and complexity measures.

Analysis Components:
- SwarmPatternAnalyzer: Main pattern analysis engine
- Spatial pattern detection and classification
- Temporal dynamics and synchronization analysis
- Interaction network analysis and metrics
- Information theory measures (mutual information, transfer entropy)
- Complexity analysis (fractal dimension, Lyapunov exponents)
- Emergent phenomenon detection and interpretation

Integration Points:
- GEO-INFER-SPACE: Spatial analysis and pattern recognition
- GEO-INFER-MATH: Statistical and mathematical analysis tools
- GEO-INFER-TIME: Temporal pattern analysis

Example:
    >>> from geo_infer_ant.analysis import SwarmPatternAnalyzer
    >>>
    >>> # Analyze spatial patterns in agent trajectories
    >>> analyzer = SwarmPatternAnalyzer(
    ...     analysis_types=['spatial_patterns', 'interaction_networks'],
    ...     statistical_methods=['cluster_analysis', 'network_analysis']
    ... )
    >>>
    >>> # Analyze agent movement patterns
    >>> spatial_results = analyzer.analyze_spatial_patterns(
    ...     agent_trajectories=trajectory_data,
    ...     pattern_types=['clustering', 'flocking', 'migration']
    ... )
    >>>
    >>> # Detect emergent behaviors
    >>> emergence_results = analyzer.detect_emergence(
    ...     individual_behaviors=agent_actions,
    ...     collective_outcomes=system_behavior
    ... )
"""

import logging

from .patterns import SwarmPatternAnalyzer, AnalysisConfiguration
from .metrics import SwarmPerformanceMetrics, PerformanceConfiguration

logger = logging.getLogger(__name__)

# Export main classes and functions
__all__ = [
    # Pattern Analysis
    "SwarmPatternAnalyzer",
    "AnalysisConfiguration",
    # Performance Metrics
    "SwarmPerformanceMetrics",
    "PerformanceConfiguration",
]
