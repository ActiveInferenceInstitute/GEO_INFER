"""
Example demonstrating module-specific simulation methods.

This example shows how to use the comprehensive module simulation methods
that are exactly named after each GEO-INFER module.
"""

from geo_infer_sim.module_simulations import ModuleSimulations, ModuleSimulationConfig


def main() -> None:
    """Run module simulation examples."""
    print("GEO-INFER-SIM: Module Simulation Methods Example")
    print("=" * 60)

    # Initialize module simulations
    config = ModuleSimulationConfig(
        time_horizon=50.0,
        time_step=1.0,
        random_seed=42,
    )
    sims = ModuleSimulations(config)

    # Example 1: Simulate ACT module
    print("\n1. Simulating ACT (Active Inference) module...")
    act_results = sims.simulate_act()
    print(f"   Final beliefs: {act_results['final_beliefs']}")
    print(f"   Free energy history length: {len(act_results['free_energy_history'])}")

    # Example 2: Simulate SPACE module
    print("\n2. Simulating SPACE (Spatial Analysis) module...")
    space_results = sims.simulate_space()
    print(f"   Total operations: {space_results['total_operations']}")
    print(f"   Total indices created: {space_results['total_indices']}")

    # Example 3: Simulate AG module
    print("\n3. Simulating AG (Agriculture) module...")
    ag_results = sims.simulate_ag()
    print(f"   Crop growth history length: {len(ag_results['crop_growth_history'])}")
    print(f"   Yield history length: {len(ag_results['yield_history'])}")

    # Example 4: Simulate HEALTH module
    print("\n4. Simulating HEALTH module...")
    health_results = sims.simulate_health()
    print(f"   Peak cases: {health_results['peak_cases']}")
    print(f"   Cases history length: {len(health_results['cases_history'])}")

    # Example 5: Simulate AI module
    print("\n5. Simulating AI (Artificial Intelligence) module...")
    ai_results = sims.simulate_ai()
    print(f"   Final accuracy: {ai_results['final_accuracy']:.3f}")
    print(f"   Loss history length: {len(ai_results['loss_history'])}")

    # Example 6: Simulate AGENT module
    print("\n6. Simulating AGENT module...")
    agent_results = sims.simulate_agent(agent_count=20)
    print(f"   Position history length: {len(agent_results['position_history'])}")
    print(f"   Final positions shape: {agent_results['final_positions'].shape}")

    # Example 7: Simulate DATA module
    print("\n7. Simulating DATA module...")
    data_results = sims.simulate_data()
    print(f"   Total processed volume: {data_results['total_processed']:.2f}")
    print(f"   Average quality: {data_results['avg_quality']:.3f}")

    # Example 8: Simulate ECON module
    print("\n8. Simulating ECON (Economics) module...")
    econ_results = sims.simulate_econ()
    print(f"   GDP history length: {len(econ_results['gdp_history'])}")
    print(f"   Market history length: {len(econ_results['market_history'])}")

    # Example 9: Simulate TIME module
    print("\n9. Simulating TIME (Temporal Analysis) module...")
    time_results = sims.simulate_time()
    print(f"   Forecast history length: {len(time_results['forecast_history'])}")
    print(f"   Pattern detection length: {len(time_results['pattern_detection'])}")

    # Example 10: Simulate RISK module
    print("\n10. Simulating RISK module...")
    risk_results = sims.simulate_risk()
    print(f"   Average risk score: {risk_results['avg_risk']:.3f}")
    print(f"   Risk scores length: {len(risk_results['risk_scores'])}")

    # Example 11: Simulate all modules programmatically
    print("\n11. Simulating all modules programmatically...")
    module_names = [
        "act",
        "ag",
        "ai",
        "agent",
        "ant",
        "api",
        "app",
        "art",
        "bayes",
        "bio",
        "civ",
        "cog",
        "comms",
        "data",
        "econ",
        "git",
        "health",
        "intra",
        "iot",
        "math",
        "norms",
        "ops",
        "org",
        "pep",
        "req",
        "sec",
        "sim",
        "space",
        "spm",
        "time",
        "risk",
        "log",
        "place",
        "test",
        "examples",
    ]

    results_summary = {}
    for module_name in module_names[:10]:  # Simulate first 10 for demo
        try:
            method = getattr(sims, f"simulate_{module_name}")
            result = method()
            results_summary[module_name.upper()] = {
                "status": "success",
                "module": result.get("module", module_name.upper()),
            }
        except Exception as e:
            results_summary[module_name.upper()] = {
                "status": "error",
                "error": str(e),
            }

    print(
        f"\n   Simulated {len([r for r in results_summary.values() if r['status'] == 'success'])} modules successfully"
    )

    print("\n" + "=" * 60)
    print("Module simulation examples completed successfully!")
    print("\nAll GEO-INFER modules have corresponding simulation methods:")
    print("  - simulate_act(), simulate_ag(), simulate_ai(), ...")
    print("  - Each method simulates the core behavior of its module")
    print("  - Results include module-specific metrics and histories")


if __name__ == "__main__":
    main()
