"""
GEO-INFER-ANT Algorithms Module

This module contains implementations of swarm intelligence and optimization algorithms
for the GEO-INFER-ANT framework, including classical and modern variants.

Algorithms:
- AntColonyOptimization: Ant Colony Optimization for combinatorial problems
- ParticleSwarmOptimization: Particle Swarm Optimization for continuous problems
- ArtificialBeeColony: Artificial Bee Colony algorithm

Integration Points:
- GEO-INFER-SPACE: Spatial optimization and constraint handling
- GEO-INFER-MATH: Mathematical optimization foundations
- GEO-INFER-TIME: Temporal optimization and scheduling

Example:
    >>> from geo_infer_ant.algorithms import AntColonyOptimization, ParticleSwarmOptimization
    >>>
    >>> # Solve traveling salesman problem with ACO
    >>> aco = AntColonyOptimization(number_of_ants=50, max_iterations=100)
    >>> aco.initialize_problem(city_coordinates, distance_matrix)
    >>> result = aco.solve()
    >>> optimal_route = result.best_solution
    >>>
    >>> # Optimize continuous function with PSO
    >>> pso = ParticleSwarmOptimization(swarm_size=100, dimensions=2)
    >>> optimal_point = pso.optimize(objective_function, bounds)
"""

import logging

from .aco import AntColonyOptimization, ACOParameters, OptimizationResult
from .pso import ParticleSwarmOptimization, PSOParameters
from .abc import ArtificialBeeColony, ABCParameters

logger = logging.getLogger(__name__)

# Export main classes and functions
__all__ = [
    # Ant Colony Optimization
    "AntColonyOptimization",
    "ACOParameters",
    "OptimizationResult",
    # Particle Swarm Optimization
    "ParticleSwarmOptimization",
    "PSOParameters",
    # Artificial Bee Colony
    "ArtificialBeeColony",
    "ABCParameters",
]
