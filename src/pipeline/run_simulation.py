"""
Pipeline entry point for running the Case 01 simulation layer.
"""

from src.simulation import SubscriptionEcosystemSimulator


def main() -> None:
    """Run the simulation pipeline."""
    simulator = SubscriptionEcosystemSimulator()
    outputs = simulator.run(export_outputs=True)

    master_table = outputs["simulation_master_table"]
    churn_rate = master_table["will_churn_30d"].mean()

    print("\nSimulation completed successfully.")
    print(f"Users simulated      : {len(master_table):,}")
    print(f"Observed churn rate  : {churn_rate:.2%}")
    print("Export location      : data/raw/")


if __name__ == "__main__":
    main()