"""
Basic Agent-Based Model example using GEO-INFER-SIM.

This example demonstrates how to create and run a simple agent-based model.
"""

import numpy as np
from geo_infer_sim.paradigms.abm import AgentBasedModel, Agent
from geo_infer_sim.core.simulation_engine import SimulationEngine, SimulationConfig


def main() -> None:
    """Run basic ABM example."""
    print("GEO-INFER-SIM: Basic Agent-Based Model Example")
    print("=" * 50)

    # Create agents with a dedicated Generator (deterministic, no global
    # NumPy seeding — see the module's isolated-Generator RNG pattern).
    rng = np.random.default_rng(42)
    abm = AgentBasedModel()
    for i in range(10):
        agent = Agent(
            agent_id=f"agent_{i}",
            position=rng.normal(0.0, 10.0, size=2),  # Random positions
            properties={"type": "mobile", "energy": 100.0},
        )
        abm.add_agent(agent)

    print(f"Created ABM with {len(abm.agents)} agents")

    # Run simulation
    config = SimulationConfig(time_step=1.0, max_time=10.0)
    engine = SimulationEngine(config)

    def step_func(time, state):
        # Update ABM
        abm.step(time_step=config.time_step)

        # Return updated state
        return abm.get_state()

    # Initialize and run
    engine.initialize(abm.get_state())
    results = engine.run(step_func)

    print("\nSimulation Results:")
    print(f"  Status: {results['status']}")
    print(f"  Final time: {results['final_time']}")
    print(f"  Duration: {results['duration_seconds']:.2f}s")
    print(f"  State history entries: {len(results['state_history'])}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
