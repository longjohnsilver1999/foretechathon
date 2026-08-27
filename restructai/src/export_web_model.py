"""Export the selected Logistic Regression pipeline for exact browser-side demo inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def export_web_model(model_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Write model parameters needed to reproduce the selected LR score in TypeScript."""

    artifact = joblib.load(model_path)
    if artifact["model_name"] != "Logistic Regression":
        raise ValueError("The browser exporter currently supports the selected Logistic Regression pipeline only")
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]
    encoder = categorical_pipeline.named_steps["encoder"]
    estimator = pipeline.named_steps["classifier"]
    calibrator = artifact["calibrator"]
    comparison_path = Path(model_path).resolve().parents[1] / "reports" / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path)[
        ["Model", "ROC_AUC", "PR_AUC", "Recall", "Precision", "F1", "Brier_Score"]
    ].to_dict(orient="records")

    exported = {
        "model_name": artifact["model_name"],
        "threshold": artifact["threshold"],
        "calibration_method": artifact["calibration_method"],
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "numeric_features": list(preprocessor.transformers_[0][2]),
        "numeric_imputer_statistics": imputer.statistics_.tolist(),
        "missing_indicator_indices": imputer.indicator_.features_.tolist(),
        "numeric_scaler_mean": scaler.mean_.tolist(),
        "numeric_scaler_scale": scaler.scale_.tolist(),
        "categorical_features": list(preprocessor.transformers_[1][2]),
        "categorical_categories": [category.tolist() for category in encoder.categories_],
        "coefficients": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "calibration_coefficient": float(calibrator.model.coef_[0][0]) if calibrator.method == "sigmoid" else 1.0,
        "calibration_intercept": float(calibrator.model.intercept_[0]) if calibrator.method == "sigmoid" else 0.0,
        "dataset_prevalence": artifact["dataset_prevalence"],
        "test_metrics": artifact["test_metrics"],
        "model_benchmark": comparison,
        "notice": "Synthetic-data methodology demonstration; not validated for real credit decisions.",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(exported, indent=2, default=_serializable), encoding="utf-8")
    return exported


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    model = Path(__file__).resolve().parents[1] / "models" / "risk_model.joblib"
    destination = root / "app" / "risk-model-artifact.json"
    result = export_web_model(model, destination)
    print(f"Exported {result['model_name']} with {len(result['coefficients'])} parameters to {destination}")
