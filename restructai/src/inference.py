"""Load the persisted pipeline and score one MSME without retraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from .explainability import explain_single_prediction
from .feature_engineering import MODEL_FEATURES, engineer_features
from .risk_scoring import early_warning_flags, risk_category


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run training first.")
    artifact = joblib.load(path)
    required = {"pipeline", "calibrator", "threshold", "model_name"}
    if not required.issubset(artifact):
        raise ValueError("Persisted model artifact is incomplete")
    return artifact


def predict_msme_risk(
    msme_data: Mapping[str, Any],
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Predict serious financial stress/default within 90 days for one borrower."""

    default_path = Path(__file__).resolve().parents[1] / "models" / "risk_model.joblib"
    artifact = load_artifact(model_path or default_path)
    engineered = engineer_features(pd.DataFrame([dict(msme_data)]))
    probability_raw = artifact["pipeline"].predict_proba(engineered[MODEL_FEATURES])[:, 1]
    probability = float(artifact["calibrator"].predict(probability_raw)[0])
    probability = min(max(probability, 0.0), 1.0)
    score = round(probability * 100, 1)
    threshold = float(artifact["threshold"])
    explanation = explain_single_prediction(artifact, engineered[MODEL_FEATURES])
    return {
        "probability_of_stress": probability,
        "risk_score": score,
        "risk_category": risk_category(score),
        "classification": int(probability >= threshold),
        "threshold": threshold,
        "model_name": artifact["model_name"],
        "calibration_method": artifact["calibration_method"],
        "top_risk_factors": explanation["top_risk_factors"],
        "protective_factors": explanation["protective_factors"],
        "warning_flags": early_warning_flags(engineered.iloc[0]),
        "decision_support_notice": "Prototype decision support only; not an automatic credit decision.",
    }
