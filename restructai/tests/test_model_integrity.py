"""Regression tests for the saved model, browser export and advisor contract."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.advisor_input import build_model_record, validate_advisor_input
from src.evaluate_models import classification_metrics
from src.feature_engineering import MODEL_FEATURES, engineer_features
from src.inference import predict_advisor_risk

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
MODEL_PATH = PROJECT / "models" / "risk_model.joblib"
WEB_ARTIFACT_PATH = ROOT / "app" / "risk-model-artifact.json"
DATA_PATH = PROJECT / "data" / "synthetic_msme_data.csv"

STRESSED = {
    "industry": "Textile",
    "state": "Gujarat",
    "business_type": "Private Limited",
    "business_age_years": 7,
    "employee_count": 28,
    "monthly_revenue": 1_200_000,
    "monthly_operating_expenses": 1_020_000,
    "free_cash_flow": 128_000,
    "average_bank_balance": 720_000,
    "revenue_growth_3m": -0.19,
    "cash_flow_volatility": 0.42,
    "receivable_days": 72,
    "receivable_days_change": 19,
    "overdue_receivables_ratio": 0.29,
    "outstanding_loan_amount": 5_600_000,
    "interest_rate": 12.4,
    "current_emi": 178_000,
    "remaining_tenure_months": 34,
    "total_existing_debt": 7_100_000,
    "number_of_active_loans": 3,
    "delayed_emi_count_3m": 2,
    "delayed_emi_count_6m": 3,
    "missed_emi_count": 0,
    "average_payment_delay_days": 14,
    "gst_turnover_growth": -0.16,
    "gst_vs_bank_credit_difference": 0.11,
}

STABLE = {
    **STRESSED,
    "monthly_revenue": 1_550_000,
    "monthly_operating_expenses": 1_030_000,
    "free_cash_flow": 390_000,
    "average_bank_balance": 3_200_000,
    "revenue_growth_3m": 0.08,
    "cash_flow_volatility": 0.17,
    "receivable_days": 36,
    "receivable_days_change": -2,
    "overdue_receivables_ratio": 0.08,
    "outstanding_loan_amount": 3_700_000,
    "current_emi": 128_000,
    "total_existing_debt": 4_400_000,
    "number_of_active_loans": 1,
    "delayed_emi_count_3m": 0,
    "delayed_emi_count_6m": 0,
    "missed_emi_count": 0,
    "average_payment_delay_days": 2,
    "gst_turnover_growth": 0.07,
    "gst_vs_bank_credit_difference": 0.03,
}

CRITICAL = {
    **STRESSED,
    "monthly_revenue": 950_000,
    "monthly_operating_expenses": 910_000,
    "free_cash_flow": 62_000,
    "average_bank_balance": 360_000,
    "revenue_growth_3m": -0.31,
    "cash_flow_volatility": 0.59,
    "receivable_days": 92,
    "receivable_days_change": 31,
    "overdue_receivables_ratio": 0.43,
    "current_emi": 212_000,
    "total_existing_debt": 8_400_000,
    "number_of_active_loans": 4,
    "delayed_emi_count_3m": 3,
    "delayed_emi_count_6m": 6,
    "missed_emi_count": 1,
    "average_payment_delay_days": 29,
    "gst_turnover_growth": -0.28,
    "gst_vs_bank_credit_difference": 0.18,
}


def _score_web_export(advisor: dict[str, object]) -> float:
    exported = json.loads(WEB_ARTIFACT_PATH.read_text(encoding="utf-8"))
    row = engineer_features(pd.DataFrame([build_model_record(advisor)])).iloc[0]

    missing: list[bool] = []
    numeric: list[float] = []
    for index, name in enumerate(exported["numeric_features"]):
        value = row[name]
        is_missing = bool(pd.isna(value))
        missing.append(is_missing)
        imputed = exported["numeric_imputer_statistics"][index] if is_missing else float(value)
        numeric.append(
            (imputed - exported["numeric_scaler_mean"][index])
            / (exported["numeric_scaler_scale"][index] or 1)
        )

    for indicator_index, feature_index in enumerate(exported["missing_indicator_indices"]):
        scaler_index = len(exported["numeric_features"]) + indicator_index
        raw = 1.0 if missing[feature_index] else 0.0
        numeric.append(
            (raw - exported["numeric_scaler_mean"][scaler_index])
            / (exported["numeric_scaler_scale"][scaler_index] or 1)
        )

    categorical: list[float] = []
    for feature_index, name in enumerate(exported["categorical_features"]):
        value = str(row[name])
        categorical.extend(
            1.0 if value == category else 0.0
            for category in exported["categorical_categories"][feature_index]
        )

    transformed = np.asarray(numeric + categorical)
    coefficients = np.asarray(exported["coefficients"])
    assert len(transformed) == len(coefficients) == len(exported["feature_names"])
    logit = float(exported["intercept"] + transformed @ coefficients)
    raw_probability = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    if exported["calibration_method"] == "sigmoid":
        clipped = float(np.clip(raw_probability, 1e-6, 1 - 1e-6))
        calibrated_logit = (
            exported["calibration_coefficient"] * np.log(clipped / (1 - clipped))
            + exported["calibration_intercept"]
        )
        return float(1 / (1 + np.exp(-np.clip(calibrated_logit, -30, 30))))
    return float(raw_probability)


@pytest.mark.parametrize("advisor", [STABLE, STRESSED, CRITICAL])
def test_browser_export_matches_persisted_pipeline(advisor: dict[str, object]) -> None:
    python_probability = predict_advisor_risk(advisor, model_path=MODEL_PATH)["probability_of_stress"]
    assert _score_web_export(advisor) == pytest.approx(python_probability, abs=1e-10)


def test_scenario_risk_ordering_and_buckets() -> None:
    stable = predict_advisor_risk(STABLE, model_path=MODEL_PATH)
    stressed = predict_advisor_risk(STRESSED, model_path=MODEL_PATH)
    critical = predict_advisor_risk(CRITICAL, model_path=MODEL_PATH)
    assert stable["probability_of_stress"] < stressed["probability_of_stress"] < critical["probability_of_stress"]
    assert stable["risk_category"] == "Low"
    assert stressed["classification"] == 1
    assert critical["risk_category"] == "Critical"


def test_advisor_validation_rejects_impossible_inputs() -> None:
    invalid = {
        **STRESSED,
        "monthly_revenue": -1,
        "total_existing_debt": 2_000_000,
        "delayed_emi_count_6m": 1,
    }
    issues = validate_advisor_input(invalid)
    assert any("monthly_revenue" in issue for issue in issues)
    with pytest.raises(ValueError):
        build_model_record(invalid)


def test_saved_metrics_reproduce_on_untouched_test_partition() -> None:
    artifact = joblib.load(MODEL_PATH)
    data = engineer_features(pd.read_csv(DATA_PATH))
    _, x_test, _, y_test = train_test_split(
        data[MODEL_FEATURES],
        data["default_90d"].astype(int),
        test_size=0.20,
        stratify=data["default_90d"].astype(int),
        random_state=42,
    )
    raw = artifact["pipeline"].predict_proba(x_test)[:, 1]
    calibrated = artifact["calibrator"].predict(raw)
    reproduced = classification_metrics(y_test.to_numpy(), calibrated, float(artifact["threshold"]))
    for metric, expected in artifact["test_metrics"].items():
        assert reproduced[metric] == pytest.approx(expected, abs=1e-12)


def test_synthetic_dataset_contract() -> None:
    data = pd.read_csv(DATA_PATH)
    assert len(data) == 12_000
    assert data["borrower_id"].is_unique
    assert 0.06 <= data["default_90d"].mean() <= 0.18
    assert set(data["default_90d"].unique()) == {0, 1}
