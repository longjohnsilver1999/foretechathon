"""Financially meaningful feature engineering shared by training and inference."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

EPSILON = 1e-6

CATEGORICAL_FEATURES = ["industry", "state", "business_type"]

BASE_NUMERIC_FEATURES = [
    "business_age_years",
    "employee_count",
    "monthly_revenue",
    "revenue_3m_avg",
    "revenue_6m_avg",
    "revenue_growth_3m",
    "revenue_growth_6m",
    "revenue_volatility",
    "monthly_operating_expenses",
    "fixed_expenses",
    "variable_expenses",
    "expense_growth",
    "operating_cash_flow",
    "free_cash_flow",
    "average_bank_balance",
    "minimum_bank_balance",
    "cash_flow_volatility",
    "cash_runway_months",
    "accounts_receivable",
    "receivable_days",
    "receivable_days_change",
    "overdue_receivables_ratio",
    "outstanding_loan_amount",
    "interest_rate",
    "current_emi",
    "remaining_tenure_months",
    "total_existing_debt",
    "number_of_active_loans",
    "delayed_emi_count_3m",
    "delayed_emi_count_6m",
    "missed_emi_count",
    "average_payment_delay_days",
    "maximum_payment_delay_days",
    "gst_monthly_turnover",
    "gst_turnover_growth",
    "gst_vs_bank_credit_difference",
]

ENGINEERED_FEATURES = [
    "dscr",
    "emi_to_revenue",
    "emi_to_free_cash_flow",
    "debt_to_revenue",
    "operating_margin",
    "cash_runway",
    "receivables_to_revenue",
    "expense_to_revenue",
    "revenue_momentum",
    "revenue_deterioration",
    "receivable_stress",
    "repayment_delinquency_score",
    "liquidity_stress",
    "debt_burden",
    "cashflow_volatility_index",
]

MODEL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_FEATURES
MODEL_FEATURES = MODEL_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def safe_divide(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
    fill_value: float = 0.0,
) -> pd.Series | np.ndarray | float:
    """Divide without producing infinities or division-by-zero failures."""

    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(num.shape, den.shape), fill_value, dtype=float)
    np.divide(num, den, out=result, where=np.abs(den) > EPSILON)
    result = np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index)
    if result.ndim == 0:
        return float(result)
    return result


def _ensure_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required borrower fields: {', '.join(missing)}")


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add ratios and stress indicators using only information known at prediction time."""

    _ensure_columns(data, BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    frame = data.copy()

    frame["dscr"] = safe_divide(frame["free_cash_flow"], frame["current_emi"])
    frame["emi_to_revenue"] = safe_divide(frame["current_emi"], frame["monthly_revenue"])
    frame["emi_to_free_cash_flow"] = safe_divide(
        frame["current_emi"], frame["free_cash_flow"].clip(lower=EPSILON)
    )
    frame["debt_to_revenue"] = safe_divide(
        frame["total_existing_debt"], frame["monthly_revenue"] * 12
    )
    frame["operating_margin"] = safe_divide(
        frame["operating_cash_flow"], frame["monthly_revenue"]
    )
    frame["cash_runway"] = safe_divide(
        frame["average_bank_balance"], frame["monthly_operating_expenses"]
    )
    frame["receivables_to_revenue"] = safe_divide(
        frame["accounts_receivable"], frame["monthly_revenue"]
    )
    frame["expense_to_revenue"] = safe_divide(
        frame["monthly_operating_expenses"], frame["monthly_revenue"]
    )
    frame["revenue_momentum"] = (
        0.65 * frame["revenue_growth_3m"] + 0.35 * frame["revenue_growth_6m"]
    )
    frame["revenue_deterioration"] = (
        (-frame["revenue_growth_3m"]).clip(lower=0)
        + 0.5 * (-frame["revenue_growth_6m"]).clip(lower=0)
    )
    frame["receivable_stress"] = (
        frame["receivable_days"] / 60
        + frame["receivable_days_change"].clip(lower=0) / 30
        + frame["overdue_receivables_ratio"]
    )
    frame["repayment_delinquency_score"] = (
        frame["delayed_emi_count_3m"]
        + 0.5 * frame["delayed_emi_count_6m"]
        + 3.0 * frame["missed_emi_count"]
        + frame["average_payment_delay_days"] / 15
    )
    frame["liquidity_stress"] = (
        (1.25 - frame["dscr"]).clip(lower=0)
        + (2.0 - frame["cash_runway"]).clip(lower=0) / 2
        + safe_divide(
            (frame["monthly_operating_expenses"] - frame["minimum_bank_balance"]).clip(lower=0),
            frame["monthly_operating_expenses"],
        )
    )
    frame["debt_burden"] = (
        frame["emi_to_revenue"] * 2.5
        + frame["debt_to_revenue"]
        + frame["number_of_active_loans"] / 5
    )
    frame["cashflow_volatility_index"] = (
        0.55 * frame["cash_flow_volatility"] + 0.45 * frame["revenue_volatility"]
    )

    frame[ENGINEERED_FEATURES] = frame[ENGINEERED_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    return frame

