"""
Pipeline entry point for running the Case 01 explainability layer.
"""

from src.explainability import ChurnShapExplainer


def main() -> None:
    """Run the explainability pipeline."""
    explainer = ChurnShapExplainer()
    outputs = explainer.run(export_outputs=True)

    global_df = outputs["shap_global_importance"]
    local_df = outputs["shap_local_top_drivers"]

    print("\nExplainability pipeline completed successfully.")
    print(f"Global importance rows : {len(global_df):,}")
    print(f"Local driver rows      : {len(local_df):,}")
    print("Export location        : reports/ and assets/")


if __name__ == "__main__":
    main()