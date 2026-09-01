"""Export every fitted benchmark model for exact browser-side inference."""

from __future__ import annotations

import json
import math
import re
import tempfile
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


def _export_preprocessor(pipeline: Any) -> dict[str, Any]:
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]
    encoder = categorical_pipeline.named_steps["encoder"]
    return {
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "numeric_features": list(preprocessor.transformers_[0][2]),
        "numeric_imputer_statistics": imputer.statistics_.tolist(),
        "missing_indicator_indices": imputer.indicator_.features_.tolist(),
        "numeric_scaler_mean": scaler.mean_.tolist(),
        "numeric_scaler_scale": scaler.scale_.tolist(),
        "categorical_features": list(preprocessor.transformers_[1][2]),
        "categorical_categories": [category.tolist() for category in encoder.categories_],
    }


def _export_calibrator(calibrator: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"method": calibrator.method}
    if calibrator.method == "sigmoid":
        result.update(
            coefficient=float(calibrator.model.coef_[0][0]),
            intercept=float(calibrator.model.intercept_[0]),
        )
    elif calibrator.method == "isotonic":
        result.update(
            x_thresholds=calibrator.model.X_thresholds_.tolist(),
            y_thresholds=calibrator.model.y_thresholds_.tolist(),
        )
    return result


def _xgboost_base_score(estimator: Any) -> float:
    config = json.loads(estimator.get_booster().save_config())
    encoded = config["learner"]["learner_model_param"]["base_score"]
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(encoded))
    if match is None:
        raise ValueError(f"Could not parse XGBoost base score: {encoded}")
    probability = min(max(float(match.group(0)), 1e-9), 1 - 1e-9)
    return math.log(probability / (1 - probability))


def _export_estimator(name: str, estimator: Any) -> dict[str, Any]:
    if name == "Logistic Regression":
        return {
            "kind": "logistic",
            "coefficients": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]),
        }
    if name == "XGBoost":
        trees = [json.loads(tree) for tree in estimator.get_booster().get_dump(dump_format="json")]
        return {
            "kind": "xgboost",
            "base_margin": _xgboost_base_score(estimator),
            "trees": trees,
        }
    if name == "CatBoost":
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "catboost.json"
            estimator.save_model(model_path, format="json")
            raw = json.loads(model_path.read_text(encoding="utf-8"))
        trees = []
        for tree in raw["oblivious_trees"]:
            splits = []
            for split in tree["splits"]:
                if split["split_type"] != "FloatFeature":
                    raise ValueError("Browser export only supports CatBoost float-feature splits")
                splits.append(
                    {
                        "feature_index": int(split["float_feature_index"]),
                        "border": float(split["border"]),
                    }
                )
            trees.append({"splits": splits, "leaf_values": tree["leaf_values"]})
        scale_and_bias = raw.get("scale_and_bias", [1.0, [0.0]])
        bias = scale_and_bias[1][0] if isinstance(scale_and_bias[1], list) else scale_and_bias[1]
        return {
            "kind": "catboost",
            "scale": float(scale_and_bias[0]),
            "bias": float(bias),
            "trees": trees,
        }
    raise ValueError(f"Unsupported browser model: {name}")


def export_web_model(model_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Write preprocessing, calibration and estimator parameters for all live models."""

    artifact = joblib.load(model_path)
    model_artifacts = artifact.get("models")
    if not model_artifacts:
        model_artifacts = {
            artifact["model_name"]: {
                "pipeline": artifact["pipeline"],
                "calibrator": artifact["calibrator"],
                "threshold": artifact["threshold"],
                "calibration_method": artifact["calibration_method"],
                "test_metrics": artifact["test_metrics"],
            }
        }

    first_pipeline = next(iter(model_artifacts.values()))["pipeline"]
    preprocessor_export = _export_preprocessor(first_pipeline)
    models = []
    for name, model_artifact in model_artifacts.items():
        pipeline = model_artifact["pipeline"]
        current_preprocessor = _export_preprocessor(pipeline)
        if current_preprocessor != preprocessor_export:
            raise ValueError(f"{name} preprocessing differs from the shared browser contract")
        models.append(
            {
                "name": name,
                "threshold": float(model_artifact["threshold"]),
                "calibration": _export_calibrator(model_artifact["calibrator"]),
                "test_metrics": model_artifact["test_metrics"],
                "estimator": _export_estimator(name, pipeline.named_steps["classifier"]),
            }
        )

    comparison_path = Path(model_path).resolve().parents[1] / "reports" / "model_comparison.csv"
    comparison_columns = [
        "Model", "ROC_AUC", "PR_AUC", "Recall", "Precision", "F1", "Brier_Score",
        "Threshold", "Calibration", "CV_ROC_AUC", "CV_PR_AUC", "CV_Recall", "CV_F1",
    ]
    comparison = pd.read_csv(comparison_path)[comparison_columns].to_dict(orient="records")

    exported = {
        "schema_version": 2,
        "default_model": artifact["model_name"],
        "dataset_rows": int(artifact.get("dataset_rows", 12_000)),
        "test_rows": int(artifact.get("test_rows", 2_400)),
        "dataset_prevalence": artifact["dataset_prevalence"],
        "preprocessor": preprocessor_export,
        "models": models,
        "model_benchmark": comparison,
        "notice": "Operational decision support using synthetic training data; not validated for autonomous lending decisions.",
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
    print(f"Exported {len(result['models'])} live models to {destination}")
