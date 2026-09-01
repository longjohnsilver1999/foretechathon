"""Validated 26-signal advisor input and deterministic model-record expansion."""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping

import numpy as np

ADVISOR_FIELDS = [
    "industry",
    "state",
    "business_type",
    "business_age_years",
    "employee_count",
    "monthly_revenue",
    "monthly_operating_expenses",
    "free_cash_flow",
    "average_bank_balance",
    "revenue_growth_3m",
    "cash_flow_volatility",
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
    "gst_turnover_growth",
    "gst_vs_bank_credit_difference",
]

RANGES: dict[str, tuple[float, float]] = {
    "business_age_years": (0.1, 100),
    "employee_count": (1, 1_000_000),
    "monthly_revenue": (1, 10_000_000_000),
    "monthly_operating_expenses": (1, 10_000_000_000),
    "free_cash_flow": (-10_000_000_000, 10_000_000_000),
    "average_bank_balance": (0, 100_000_000_000),
    "revenue_growth_3m": (-1, 3),
    "cash_flow_volatility": (0, 1),
    "receivable_days": (0, 365),
    "receivable_days_change": (-365, 365),
    "overdue_receivables_ratio": (0, 1),
    "outstanding_loan_amount": (1, 1_000_000_000_000),
    "interest_rate": (0, 40),
    "current_emi": (1, 10_000_000_000),
    "remaining_tenure_months": (1, 360),
    "total_existing_debt": (1, 1_000_000_000_000),
    "number_of_active_loans": (1, 20),
    "delayed_emi_count_3m": (0, 3),
    "delayed_emi_count_6m": (0, 6),
    "missed_emi_count": (0, 6),
    "average_payment_delay_days": (0, 365),
    "gst_turnover_growth": (-1, 3),
    "gst_vs_bank_credit_difference": (0, 1),
}

INTEGER_FIELDS = {
    "employee_count",
    "remaining_tenure_months",
    "number_of_active_loans",
    "delayed_emi_count_3m",
    "delayed_emi_count_6m",
    "missed_emi_count",
}


def validate_advisor_input(data: Mapping[str, Any]) -> list[str]:
    """Return all actionable input-contract errors without scoring invalid data."""

    issues: list[str] = []
    missing = [field for field in ADVISOR_FIELDS if field not in data]
    if missing:
        issues.append(f"Missing advisor fields: {', '.join(missing)}")
        return issues

    for field in ["industry", "state", "business_type"]:
        if not isinstance(data[field], str) or not data[field].strip():
            issues.append(f"{field} must be a non-empty string")

    for field, (minimum, maximum) in RANGES.items():
        value = data[field]
        if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(float(value)):
            issues.append(f"{field} must be a finite number")
            continue
        numeric = float(value)
        if not minimum <= numeric <= maximum:
            issues.append(f"{field} must be between {minimum:g} and {maximum:g}")
        if field in INTEGER_FIELDS and not numeric.is_integer():
            issues.append(f"{field} must be a whole number")

    if not issues:
        if float(data["total_existing_debt"]) < float(data["outstanding_loan_amount"]):
            issues.append("total_existing_debt cannot be lower than outstanding_loan_amount")
        if float(data["delayed_emi_count_6m"]) < float(data["delayed_emi_count_3m"]):
            issues.append("delayed_emi_count_6m cannot be lower than delayed_emi_count_3m")
    return issues


def build_model_record(data: Mapping[str, Any]) -> dict[str, Any]:
    """Expand the web advisor's 26 inputs into the raw record expected by the model."""

    issues = validate_advisor_input(data)
    if issues:
        raise ValueError("; ".join(issues))

    values = dict(data)
    revenue = float(values["monthly_revenue"])
    expenses = float(values["monthly_operating_expenses"])
    growth_3m = float(values["revenue_growth_3m"])
    growth_6m = growth_3m * 0.65
    volatility = float(values["cash_flow_volatility"])
    average_balance = float(values["average_bank_balance"])
    receivable_days = float(values["receivable_days"])
    gst_variance = float(values["gst_vs_bank_credit_difference"])
    average_delay = float(values["average_payment_delay_days"])

    values.update(
        {
            "revenue_3m_avg": revenue / max(1 + growth_3m * 0.45, 0.3),
            "revenue_6m_avg": revenue / max(1 + growth_6m * 0.5, 0.3),
            "revenue_growth_6m": growth_6m,
            "revenue_volatility": float(np.clip(volatility * 0.74, 0.02, 0.78)),
            "fixed_expenses": expenses * 0.47,
            "variable_expenses": expenses * 0.53,
            "expense_growth": max(0.02, -growth_3m * 0.42),
            "operating_cash_flow": revenue - expenses,
            "minimum_bank_balance": average_balance * 0.28,
            "cash_runway_months": average_balance / expenses,
            "accounts_receivable": revenue * receivable_days / 30,
            "maximum_payment_delay_days": average_delay * 1.9,
            "gst_monthly_turnover": revenue * (1 - min(gst_variance, 0.4)),
        }
    )
    return values

