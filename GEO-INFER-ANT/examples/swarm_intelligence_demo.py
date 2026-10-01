#!/usr/bin/env python3
"""
Comprehensive Swarm Intelligence Demonstration for GEO-INFER-ANT

This example demonstrates the complete functionality of the GEO-INFER-ANT module,
including swarm agent creation, population dynamics, stigmergic communication,
optimization algorithms, and emergent behavior analysis.

The demonstration simulates an environmental monitoring scenario where a swarm
of intelligent agents coordinate to monitor air quality, detect anomalies,
and optimize their monitoring coverage using various swarm intelligence techniques.

Usage:
    python swarm_intelligence_demo.py

This will run a complete simulation and generate analysis results.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta

import numpy as np

# There is deliberately no mock fallback: a broken environment must fail
# loudly instead of "demonstrating" with simulated components.
import geo_infer_ant
from geo_infer_ant.core.agent_base import SwarmAgent
from geo_infer_ant.core.population import AgentPopulation
from geo_infer_ant.core.stigmergy import PheromoneSystem
from geo_infer_ant.core.digital_stigmergy import DigitalStigmergy
from geo_infer_ant.algorithms.aco import AntColonyOptimization
from geo_infer_ant.algorithms.pso import ParticleSwarmOptimization
from geo_infer_ant.algorithms.abc import ArtificialBeeColony
from geo_infer_ant.applications.environmental import EnvironmentalMonitoringSwarm
from geo_infer_ant.analysis.patterns import SwarmPatternAnalyzer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(
    "GEO-INFER-ANT %s modules imported successfully",
    getattr(geo_infer_ant, "__version__", "unknown"),
)


def generate_sample_data():
    """Generate sample data for the demonstration."""
    logger.info("Generating sample environmental monitoring data")

    # Generate agent trajectories (simulate movement over time)
    n_agents = 50
    n_time_steps = 100
    trajectories = []

    for agent in range(n_agents):
        # Start positions (clustered in 3 groups)
        if agent < n_agents // 3:
            start_pos = np.array([-8, -8]) + np.random.normal(0, 1, 2)
        elif agent < 2 * n_agents // 3:
            start_pos = np.array([0, 0]) + np.random.normal(0, 1, 2)
        else:
            start_pos = np.array([8, 8]) + np.random.normal(0, 1, 2)

        # Simulate movement (biased toward center for clustering demonstration)
        agent_trajectory = [start_pos]
        current_pos = start_pos.copy()

        for _ in range(n_time_steps):
            # Movement with social attraction
            center_attraction = -0.1 * (
                current_pos - np.array([0, 0])
            )  # Attraction to center
            random_movement = np.random.normal(0, 0.5, 2)
            noise = np.random.normal(0, 0.1, 2)

            current_pos += center_attraction + random_movement + noise
            agent_trajectory.append(current_pos.copy())

        trajectories.append(np.array(agent_trajectory))

    # Generate communication data
    communication_data = []
    for _ in range(100):  # 100 communication events
        comm = {
            "from": f"agent_{np.random.randint(0, n_agents)}",
            "to": f"agent_{np.random.randint(0, n_agents)}",
            "type": np.random.choice(["status_update", "alert", "coordination"]),
            "timestamp": datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
            "location": np.random.uniform(-10, 10, 2),
        }
        communication_data.append(comm)

    # Generate sensor readings
    sensor_readings = []
    for _ in range(200):  # 200 sensor readings
        reading = {
            "agent_id": f"agent_{np.random.randint(0, n_agents)}",
            "sensor_type": np.random.choice(["temperature", "humidity", "pm25", "no2"]),
            "value": (
                np.random.normal(20, 5)
                if np.random.random() < 0.9
                else np.random.normal(35, 2)
            ),  # Normal + some anomalies
            "location": np.random.uniform(-10, 10, 2),
            "timestamp": datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
            "quality_score": np.random.uniform(0.7, 1.0),
        }
        sensor_readings.append(reading)

    return {
        "trajectories": trajectories,
        "communication_data": communication_data,
        "sensor_readings": sensor_readings,
        "environmental_conditions": {
            "temperature": 22.0,
            "humidity": 65.0,
            "wind_speed": 5.0,
            "air_quality_index": 45.0,
        },
    }


async def demonstrate_swarm_agents():
    """Demonstrate individual swarm agent functionality."""
    logger.info("=== Demonstrating Swarm Agents ===")

    # Create sample agents
    agents = []
    for i in range(5):
        agent = SwarmAgent(
            agent_id=f"demo_agent_{i}",
            position=np.random.uniform(-10, 10, 2),
            sensory_range=100.0,
            active_inference_enabled=True,
        )
        agents.append(agent)

    logger.info(f"Created {len(agents)} swarm agents")

    # Demonstrate agent perception and decision making
    for agent in agents:
        # Simulate environmental perception
        sensory_input = await agent.perceive_environment(
            spatial_context={"position": agent.position},
            environmental_signals={"temperature": 22.0, "food_nearby": True},
            social_signals={"nearby_agents": 3},
        )

        # Agent decision making
        motivations = {"energy_conservation": 0.6, "task_completion": 0.8}
        decision = agent.make_decision(sensory_input, motivations)

        logger.info(
            f"Agent {agent.agent_id} at {agent.position}: decided to {decision.action_type}"
        )

    return agents


async def demonstrate_population_dynamics():
    """Demonstrate agent population dynamics."""
    logger.info("=== Demonstrating Population Dynamics ===")

    # Create agent population
    population = AgentPopulation(
        population_size=100,
        agent_types=["worker", "scout", "soldier"],
        spatial_distribution="clustered",
        behavioral_heterogeneity="stochastic",
    )

    # Initialize environment
    environment = population.initialize_environment(
        spatial_bounds={"min_lat": -10, "max_lat": 10, "min_lng": -10, "max_lng": 10},
        resource_distribution={
            "food": {
                "centers": [np.array([0, 0]), np.array([5, 5]), np.array([-5, -5])],
                "max_density": 1.0,
                "decay_rate": 0.1,
            }
        },
    )

    logger.info(f"Initialized population with {len(population.agents)} agents")
    logger.info(f"Environment bounds: {environment.spatial_bounds}")

    # Configure behavioral rules
    population.set_behavioral_rules(
        foraging_rules={"target_preference": "nearest"},
        communication_rules={"frequency": "adaptive"},
        adaptation_rules={"learning_rate": 0.1},
    )

    # Run short simulation
    sample_data = generate_sample_data()
    trajectories = sample_data["trajectories"][:10]  # Use subset for demo

    # Analyze population patterns
    analyzer = SwarmPatternAnalyzer()
    spatial_analysis = analyzer.analyze_spatial_patterns(
        agent_trajectories=trajectories, pattern_types=["clustering", "flocking"]
    )

    logger.info(
        f"Population analysis: {spatial_analysis['interpretation']['pattern_summary']}"
    )

    return population, spatial_analysis


async def demonstrate_stigmergic_communication():
    """Demonstrate stigmergic communication systems."""
    logger.info("=== Demonstrating Stigmergic Communication ===")

    # Initialize pheromone system
    pheromone_system = PheromoneSystem(
        spatial_resolution="h3_r8",
        pheromone_types=["trail", "food", "alarm"],
        bounds={"min_lat": -10, "max_lat": 10, "min_lng": -10, "max_lng": 10},
    )

    logger.info(
        f"Pheromone system initialized with {len(pheromone_system.pheromone_types)} types"
    )

    # Simulate pheromone deposition
    agents = [
        SwarmAgent(f"comm_agent_{i}", np.random.uniform(-10, 10, 2)) for i in range(10)
    ]

    for agent in agents:
        # Deposit trail pheromones
        await pheromone_system.deposit_pheromone(
            agent_id=agent.agent_id,
            pheromone_type="trail",
            location=agent.position,
            intensity=1.0,
            metadata={"agent_type": "worker"},
        )

        # Deposit food pheromones at resource locations
        await pheromone_system.deposit_pheromone(
            agent_id=agent.agent_id,
            pheromone_type="food",
            location=agent.position + np.random.normal(0, 2, 2),
            intensity=2.0,
            metadata={"resource_quality": np.random.uniform(0.5, 1.0)},
        )

    # Simulate pheromone diffusion
    await pheromone_system.diffuse_pheromones(
        time_step=60.0,
        environmental_conditions={"wind_speed": 3.0, "temperature": 25.0},
    )

    # Test pheromone sensing
    test_location = np.array([0.0, 0.0])
    sensed_pheromones = await pheromone_system.sense_pheromones(
        location=test_location, sensory_range=5.0, pheromone_types=["trail", "food"]
    )

    logger.info(f"Pheromones sensed at {test_location}: {sensed_pheromones}")

    # Initialize digital stigmergy
    digital_stigmergy = DigitalStigmergy(
        communication_medium="iot_network",
        information_types=["sensor_data", "alerts", "coordination"],
        persistence_model="temporal_decay",
    )

    logger.info(
        f"Digital stigmergy initialized with {len(digital_stigmergy.information_types)} information types"
    )

    # Contribute digital information
    for agent in agents:
        await digital_stigmergy.contribute_information(
            agent_id=agent.agent_id,
            information_type="sensor_data",
            content={
                "temperature": np.random.normal(22, 3),
                "humidity": np.random.uniform(50, 80),
                "air_quality": np.random.uniform(0.6, 1.0),
            },
            location=agent.position,
            credibility_score=0.8,
        )

    # Query digital information
    nearby_info = await digital_stigmergy.query_stigmergy(
        agent_id="demo_agent",
        query_type="environmental_data",
        spatial_bounds={"min_lat": -5, "max_lat": 5, "min_lng": -5, "max_lng": 5},
        credibility_threshold=0.6,
    )

    logger.info(
        f"Digital stigmergy query returned {len(nearby_info)} information traces"
    )

    return pheromone_system, digital_stigmergy


async def demonstrate_optimization_algorithms():
    """Demonstrate swarm optimization algorithms."""
    logger.info("=== Demonstrating Optimization Algorithms ===")

    # Test Ant Colony Optimization
    logger.info("Testing Ant Colony Optimization...")
    aco = AntColonyOptimization(number_of_ants=30, max_iterations=50, variant="ACS")

    # Create sample TSP problem
    n_cities = 10
    city_positions = np.random.uniform(-10, 10, (n_cities, 2))

    # Calculate distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distance_matrix[i, j] = np.linalg.norm(
                city_positions[i] - city_positions[j]
            )

    # Initialize and solve
    aco.initialize_problem(city_positions.tolist(), distance_matrix)
    aco_result = aco.solve()

    logger.info(
        f"ACO Result: Best fitness = {aco_result.best_fitness}, Solution length = {len(aco_result.best_solution)}"
    )

    # Test Particle Swarm Optimization
    logger.info("Testing Particle Swarm Optimization...")
    pso = ParticleSwarmOptimization(
        swarm_size=50, dimensions=2, bounds=[(-10, 10), (-10, 10)], max_iterations=100
    )

    # Define objective function (Rastrigin function)
    def rastrigin_function(x):
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    optimal_point = pso.optimize(rastrigin_function)
    optimal_value = rastrigin_function(optimal_point)

    logger.info(
        f"PSO Result: Optimal point = {optimal_point}, Optimal value = {optimal_value}"
    )

    # Test Artificial Bee Colony
    logger.info("Testing Artificial Bee Colony...")
    abc = ArtificialBeeColony(
        colony_size=50, dimensions=2, bounds=[(-5, 5), (-5, 5)], max_iterations=100
    )

    optimal_solution = abc.optimize(rastrigin_function)
    optimal_fitness = rastrigin_function(optimal_solution)

    logger.info(
        f"ABC Result: Optimal solution = {optimal_solution}, Optimal fitness = {optimal_fitness}"
    )

    return {
        "aco": aco_result,
        "pso": {"optimal_point": optimal_point, "optimal_value": optimal_value},
        "abc": {
            "optimal_solution": optimal_solution,
            "optimal_fitness": optimal_fitness,
        },
    }


async def demonstrate_environmental_monitoring():
    """Demonstrate environmental monitoring application."""
    logger.info("=== Demonstrating Environmental Monitoring ===")

    # Initialize monitoring swarm
    monitoring_swarm = EnvironmentalMonitoringSwarm(
        swarm_size=100,
        monitoring_objectives=["air_quality", "biodiversity"],
        spatial_coverage={
            "min_lat": 35,
            "max_lat": 40,
            "min_lng": -120,
            "max_lng": -115,
        },
        adaptive_sampling=True,
    )

    logger.info(
        f"Environmental monitoring swarm initialized with {monitoring_swarm.swarm_size} agents"
    )

    # Deploy agents
    deployment_plan = await monitoring_swarm.deploy_agents(
        environmental_priorities={"air_quality": 0.8, "biodiversity": 0.6},
        logistical_constraints={"max_range": 1000.0},
    )

    logger.info(
        f"Agent deployment completed: {len(deployment_plan['agents'])} agents deployed"
    )
    logger.info(f"Expected coverage: {deployment_plan['coverage_achieved']:.2%}")

    # Generate sample sensor data
    sample_data = generate_sample_data()
    sensor_readings = sample_data["sensor_readings"]

    # Process collective intelligence
    assessment = await monitoring_swarm.process_collective_intelligence(
        individual_measurements=sensor_readings,
        spatial_interpolation="kriging",
        uncertainty_quantification="bayesian",
        anomaly_detection="statistical",
    )

    logger.info(
        f"Environmental assessment completed: {len(assessment['recommendations'])} recommendations"
    )
    logger.info(f"Anomalies detected: {len(assessment.get('anomaly_detection', []))}")

    # Get monitoring status
    status = monitoring_swarm.get_monitoring_status()
    logger.info(
        f"System efficiency: {status['performance_metrics']['monitoring_efficiency']:.2%}"
    )

    return monitoring_swarm, assessment


async def demonstrate_pattern_analysis():
    """Demonstrate pattern analysis and emergence detection."""
    logger.info("=== Demonstrating Pattern Analysis ===")

    # Generate sample data
    sample_data = generate_sample_data()
    trajectories = sample_data["trajectories"]
    communication_data = sample_data["communication_data"]

    # Initialize pattern analyzer
    analyzer = SwarmPatternAnalyzer(
        analysis_types=[
            "spatial_patterns",
            "interaction_networks",
            "emergent_phenomena",
        ],
        statistical_methods=[
            "cluster_analysis",
            "network_analysis",
            "information_theory",
        ],
    )

    logger.info("Pattern analyzer initialized")

    # Analyze spatial patterns
    spatial_analysis = analyzer.analyze_spatial_patterns(
        agent_trajectories=trajectories,
        pattern_types=["clustering", "flocking", "migration"],
        spatial_scale=1000.0,
    )

    logger.info(
        f"Spatial analysis: {spatial_analysis['interpretation']['pattern_summary']}"
    )

    # Analyze interaction networks
    interaction_analysis = analyzer.analyze_interactions(
        communication_data=communication_data,
        network_metrics=["centrality", "clustering", "density"],
    )

    network_props = interaction_analysis.get("network_structure", {}).get(
        "network_properties", {}
    )
    logger.info(
        f"Network density: {network_props.get('density', 0):.3f}, "
        f"Clustering: {network_props.get('clustering_coefficient', 0):.3f}"
    )

    # Detect emergent phenomena
    emergence_analysis = analyzer.detect_emergence(
        individual_behaviors=communication_data,
        collective_outcomes={"network_formation": network_props},
        information_measures=["mutual_information"],
        complexity_measures=["fractal_dimension"],
    )

    logger.info(
        f"Emergence detection: {'Positive' if emergence_analysis['emergence_detected'] else 'Negative'}"
    )

    return analyzer, spatial_analysis, interaction_analysis, emergence_analysis


async def run_complete_demonstration():
    """Run the complete GEO-INFER-ANT demonstration."""
    logger.info("Starting Complete GEO-INFER-ANT Demonstration")
    logger.info("=" * 60)

    start_time = datetime.now()
    results = {}

    try:
        # 1. Swarm Agents
        agents = await demonstrate_swarm_agents()
        results["swarm_agents"] = {"count": len(agents)}

        # 2. Population Dynamics
        population, spatial_analysis = await demonstrate_population_dynamics()
        results["population_dynamics"] = {
            "population_size": population.population_size,
            "spatial_patterns": spatial_analysis["interpretation"]["pattern_summary"],
        }

        # 3. Stigmergic Communication
        (
            pheromone_system,
            digital_stigmergy,
        ) = await demonstrate_stigmergic_communication()
        results["stigmergic_communication"] = {
            "pheromone_types": len(pheromone_system.pheromone_types),
            "digital_types": len(digital_stigmergy.information_types),
        }

        # 4. Optimization Algorithms
        optimization_results = await demonstrate_optimization_algorithms()
        results["optimization_algorithms"] = {
            "aco_fitness": optimization_results["aco"].best_fitness,
            "pso_optimal": optimization_results["pso"]["optimal_value"],
            "abc_fitness": optimization_results["abc"]["optimal_fitness"],
        }

        # 5. Environmental Monitoring
        monitoring_swarm, assessment = await demonstrate_environmental_monitoring()
        results["environmental_monitoring"] = {
            "swarm_size": monitoring_swarm.swarm_size,
            "anomalies_detected": len(assessment.get("anomaly_detection", [])),
            "recommendations": len(assessment["recommendations"]),
        }

        # 6. Pattern Analysis
        (
            analyzer,
            spatial_analysis,
            interaction_analysis,
            emergence_analysis,
        ) = await demonstrate_pattern_analysis()
        results["pattern_analysis"] = {
            "emergence_detected": emergence_analysis["emergence_detected"],
            "network_density": interaction_analysis.get("network_structure", {})
            .get("network_properties", {})
            .get("density", 0),
            "spatial_clusters": spatial_analysis.get("patterns_detected", {})
            .get("clustering", {})
            .get("n_clusters", 0),
        }

        # Generate comprehensive summary
        total_time = datetime.now() - start_time

        summary = {
            "demonstration_completed": True,
            "total_execution_time": total_time.total_seconds(),
            "components_tested": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "system_status": "All components operational",
        }

        # Save results
        output_file = f"swarm_intelligence_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Total execution time: {total_time}")
        logger.info(f"Components tested: {len(results)}")

        for component, component_results in results.items():
            logger.info(f"  {component}: {component_results}")

        return summary

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        error_summary = {
            "demonstration_completed": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }

        # Save error report
        error_file = f"swarm_intelligence_demo_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, "w") as f:
            json.dump(error_summary, f, indent=2)

        logger.error(f"Error report saved to: {error_file}")
        return error_summary


async def main():
    """Main demonstration function."""
    logger.info("GEO-INFER-ANT Swarm Intelligence Demonstration")
    logger.info(
        "This demonstrates the complete functionality of the swarm intelligence framework"
    )
    logger.info(f"Using GEO-INFER-ANT version {geo_infer_ant.__version__}")
    # Run complete demonstration
    results = await run_complete_demonstration()

    # Print final summary
    if results["demonstration_completed"]:
        print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉")
        print(f"Total time: {results['total_execution_time']:.2f} seconds")
        print(f"Components tested: {results['components_tested']}")

        print("\n📊 RESULTS SUMMARY:")
        for component in results["results"]:
            print(f"  {component}: ✓")

        print("📁 Results saved to file for detailed analysis")
        print("🚀 The GEO-INFER-ANT swarm intelligence framework is ready for use!")

    else:
        print(f"\n❌ DEMONSTRATION FAILED: {results.get('error', 'Unknown error')}")
        print("Check the error log for details")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
