"""
Pipeline entry point for building the Case 01 feature table.
"""

from src.features.feature_builder import ChurnFeatureBuilder


def main() -> None:
    """Run the feature engineering pipeline."""
    builder = ChurnFeatureBuilder()
    feature_table = builder.build_feature_table(export_output=True)
    summary = builder.summarize_feature_table(feature_table)

    print("\nFeature engineering completed successfully.")
    print(f"Rows                  : {summary['n_rows']:,}")
    print(f"Columns               : {summary['n_columns']}")
    print(f"Model feature count   : {summary['n_model_features']}")
    print(f"Observed churn rate   : {summary['target_rate']:.2%}")
    print("Export location       : data/features/feature_table.csv")


if __name__ == "__main__":
    main()