from __future__ import annotations

import pandas as pd

from src.decisioning.decision_engine import RetentionDecisionEngine
from src.io_utils import save_csv
from src.logging_utils import get_logger
from pathlib import Path


logger = get_logger("case01.decisioning")


def main():

    logger.info("Starting decisioning pipeline.")

    predictions = pd.read_csv("reports/model_predictions.csv")

    shap_local = pd.read_csv("reports/shap_local_top_drivers.csv")

    feature_table = pd.read_csv("data/features/feature_table.csv")

    engine = RetentionDecisionEngine()

    outputs = engine.run(
        predictions=predictions,
        shap_local=shap_local,
        feature_table=feature_table,
    )

    save_csv(
    outputs["decision_table"],
    Path("reports/decision_table.csv"),
    )

    save_csv(
        outputs["tier_summary"],
        Path("reports/decision_tier_summary.csv"),
    )

    save_csv(
        outputs["playbook_summary"],
        Path("reports/decision_playbook_summary.csv"),
    )

    logger.info("Decisioning pipeline completed successfully.")

    print("\nDecisioning completed successfully.")
    print(f"Users scored: {len(outputs['decision_table']):,}")


if __name__ == "__main__":
    main()