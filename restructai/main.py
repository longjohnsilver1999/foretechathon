"""Command-line entry point for data generation, training and demo inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_generation import save_synthetic_dataset
from src.feature_engineering import BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.inference import predict_msme_risk
from src.train_models import print_evaluation_report, train_and_evaluate

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "synthetic_msme_data.csv"


def _sample_borrower() -> dict[str, object]:
    data = pd.read_csv(DATASET_PATH)
    candidate = data.dropna(subset=BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES).sort_values(
        ["delayed_emi_count_3m", "revenue_growth_3m"], ascending=[False, True]
    ).iloc[len(data) // 40]
    return {column: candidate[column] for column in BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES}


def run_pipeline(rows: int) -> None:
    frame = save_synthetic_dataset(DATASET_PATH, n_borrowers=rows)
    print(f"Generated {len(frame):,} MSME observations; observed stress prevalence {frame['default_90d'].mean():.2%}.")
    artifact = train_and_evaluate(DATASET_PATH, PROJECT_DIR)
    print_evaluation_report(artifact, PROJECT_DIR / "reports" / "model_comparison.csv")
    result = predict_msme_risk(_sample_borrower())
    print("\nSAMPLE MSME RISK ANALYSIS")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RestructAI 90-day MSME financial-stress model")
    parser.add_argument("--rows", type=int, default=12_000, help="Number of synthetic borrowers")
    parser.add_argument("--predict-json", type=Path, help="Score a borrower JSON file using the saved model")
    args = parser.parse_args()
    if args.predict_json:
        borrower = json.loads(args.predict_json.read_text(encoding="utf-8"))
        print(json.dumps(predict_msme_risk(borrower), indent=2))
    else:
        run_pipeline(args.rows)
