"""Core financial, preprocessing and inference contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import MODEL_FEATURES, engineer_features, safe_divide
from src.inference import predict_msme_risk
from src.preprocessing import build_preprocessor
from src.risk_scoring import early_warning_flags, risk_category


def borrower(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "industry": "Textile",
        "state": "Gujarat",
        "business_type": "Private Limited",
        "business_age_years": 7.0,
        "employee_count": 28,
        "monthly_revenue": 1_200_000.0,
        "revenue_3m_avg": 1_260_000.0,
        "revenue_6m_avg": 1_350_000.0,
        "revenue_growth_3m": -0.12,
        "revenue_growth_6m": -0.07,
        "revenue_volatility": 0.24,
        "monthly_operating_expenses": 920_000.0,
        "fixed_expenses": 420_000.0,
        "variable_expenses": 500_000.0,
        "expense_growth": 0.08,
        "operating_cash_flow": 280_000.0,
        "free_cash_flow": 190_000.0,
        "average_bank_balance": 1_150_000.0,
        "minimum_bank_balance": 310_000.0,
        "cash_flow_volatility": 0.31,
        "cash_runway_months": 1.25,
        "accounts_receivable": 2_280_000.0,
        "receivable_days": 57.0,
        "receivable_days_change": 13.0,
        "overdue_receivables_ratio": 0.24,
        "outstanding_loan_amount": 4_800_000.0,
        "interest_rate": 12.1,
        "current_emi": 165_000.0,
        "remaining_tenure_months": 34,
        "total_existing_debt": 6_100_000.0,
        "number_of_active_loans": 2,
        "delayed_emi_count_3m": 1,
        "delayed_emi_count_6m": 2,
        "missed_emi_count": 0,
        "average_payment_delay_days": 8.0,
        "maximum_payment_delay_days": 18.0,
        "gst_monthly_turnover": 1_150_000.0,
        "gst_turnover_growth": -0.10,
        "gst_vs_bank_credit_difference": 0.07,
    }
    base.update(overrides)
    return base


def test_dscr_and_emi_burden_calculation() -> None:
    row = engineer_features(pd.DataFrame([borrower()])).iloc[0]
    assert row["dscr"] == pytest.approx(190_000 / 165_000)
    assert row["emi_to_revenue"] == pytest.approx(165_000 / 1_200_000)
    assert row["emi_to_free_cash_flow"] == pytest.approx(165_000 / 190_000)


def test_feature_engineering_and_warning_flags() -> None:
    row = engineer_features(
        pd.DataFrame([borrower(free_cash_flow=120_000, revenue_growth_3m=-0.22, receivable_days=74, delayed_emi_count_3m=2)])
    ).iloc[0]
    assert np.isfinite(row["debt_burden"])
    warnings = early_warning_flags(row)
    assert any("Debt-service" in warning for warning in warnings)
    assert any("Revenue" in warning for warning in warnings)
    assert any("Receivable" in warning for warning in warnings)


def test_missing_value_preprocessing() -> None:
    data = pd.DataFrame([borrower(), borrower(revenue_3m_avg=np.nan, industry=None)])
    engineered = engineer_features(data)
    transformed = build_preprocessor().fit_transform(engineered[MODEL_FEATURES])
    assert transformed.shape[0] == 2
    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert np.isfinite(values).all()


def test_no_division_by_zero() -> None:
    result = safe_divide(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    assert np.isfinite(result).all()
    engineered = engineer_features(pd.DataFrame([borrower(monthly_revenue=0, current_emi=0, free_cash_flow=0)]))
    assert np.isfinite(engineered.select_dtypes(include=[np.number]).fillna(0).to_numpy()).all()


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0, "Low"), (30, "Low"), (31, "Moderate"), (60, "Moderate"), (61, "High"), (80, "High"), (81, "Critical"), (100, "Critical")],
)
def test_risk_buckets(score: float, expected: str) -> None:
    assert risk_category(score) == expected


def test_persisted_prediction_output() -> None:
    model_path = Path(__file__).resolve().parents[1] / "models" / "risk_model.joblib"
    if not model_path.exists():
        pytest.skip("Training artifact is created by the pipeline run")
    result = predict_msme_risk(borrower(), model_path=model_path)
    assert 0 <= result["probability_of_stress"] <= 1
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_category"] in {"Low", "Moderate", "High", "Critical"}
    assert result["classification"] in {0, 1}
    assert isinstance(result["warning_flags"], list)
    assert isinstance(result["top_risk_factors"], list)


def test_inference_rejects_negative_financial_values() -> None:
    with pytest.raises(ValueError, match="monthly_revenue"):
        predict_msme_risk(borrower(monthly_revenue=-1))
