"""
Pipeline entry point for running the Case 01 modeling benchmark.
"""

from src.modeling import ChurnModelTrainer


def main() -> None:
    """Run the modeling benchmark pipeline."""
    trainer = ChurnModelTrainer()
    artifacts = trainer.run(export_outputs=True)

    print("\nModeling benchmark completed successfully.")
    print(f"Champion model        : {artifacts.champion_model_name}")
    print(f"Benchmark rows        : {len(artifacts.benchmark_results):,}")
    print(f"Prediction rows       : {len(artifacts.prediction_table):,}")
    print("Export location       : reports/")


if __name__ == "__main__":
    main()