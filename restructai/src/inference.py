"""Load the persisted pipeline and score one MSME without retraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .advisor_input import build_model_record
from .explainability import explain_single_prediction
from .feature_engineering import BASE_NUMERIC_FEATURES, MODEL_FEATURES, engineer_features
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
    frame = pd.DataFrame([dict(msme_data)])
    missing = sorted(set(BASE_NUMERIC_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required borrower fields: {', '.join(missing)}")
    for field in BASE_NUMERIC_FEATURES:
        value = frame.iloc[0][field]
        if pd.isna(value):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(float(value)):
            raise ValueError(f"{field} must be numeric or missing")
    for field in [
        "business_age_years", "employee_count", "monthly_revenue", "monthly_operating_expenses",
        "average_bank_balance", "minimum_bank_balance", "cash_flow_volatility", "cash_runway_months",
        "accounts_receivable", "receivable_days", "outstanding_loan_amount", "interest_rate", "current_emi",
        "remaining_tenure_months", "total_existing_debt", "number_of_active_loans", "delayed_emi_count_3m",
        "delayed_emi_count_6m", "missed_emi_count", "average_payment_delay_days", "maximum_payment_delay_days",
        "gst_monthly_turnover", "gst_vs_bank_credit_difference",
    ]:
        value = frame.iloc[0][field]
        if not pd.isna(value) and float(value) < 0:
            raise ValueError(f"{field} cannot be negative")
    if not pd.isna(frame.iloc[0]["monthly_revenue"]) and float(frame.iloc[0]["monthly_revenue"]) <= 0:
        raise ValueError("monthly_revenue must be greater than zero")
    engineered = engineer_features(frame)
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


def predict_advisor_risk(
    advisor_data: Mapping[str, Any],
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and score the same 26-signal contract used by the browser advisor."""

    return predict_msme_risk(build_model_record(advisor_data), model_path=model_path)
