"""Generate a realistic, correlated synthetic Indian MSME credit-risk dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .feature_engineering import engineer_features

RANDOM_STATE = 42

INDUSTRIES = np.array(
    [
        "Manufacturing",
        "Textile",
        "Retail",
        "Food Processing",
        "Logistics",
        "Auto Components",
        "Services",
        "Construction",
        "Electronics",
        "Chemicals",
    ]
)
STATES = np.array(
    ["Maharashtra", "Gujarat", "Tamil Nadu", "Karnataka", "Delhi", "Rajasthan", "Telangana", "Uttar Pradesh"]
)
BUSINESS_TYPES = np.array(["Proprietorship", "Partnership", "Private Limited", "LLP"])


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def _amortizing_emi(principal: np.ndarray, annual_rate: np.ndarray, months: np.ndarray) -> np.ndarray:
    rate = annual_rate / 1200
    factor = np.power(1 + rate, months)
    return principal * rate * factor / np.maximum(factor - 1, 1e-6)


def _calibrated_intercept(linear_predictor: np.ndarray, target_rate: float) -> float:
    """Find an intercept that gives the desired average event probability."""

    low, high = -15.0, 5.0
    for _ in range(80):
        midpoint = (low + high) / 2
        if _sigmoid(linear_predictor + midpoint).mean() > target_rate:
            high = midpoint
        else:
            low = midpoint
    return (low + high) / 2


def generate_synthetic_msme_data(
    n_borrowers: int = 12_000,
    random_state: int = RANDOM_STATE,
    target_default_rate: float = 0.12,
) -> pd.DataFrame:
    """Create cross-sectional borrower observations with plausible financial relationships."""

    if n_borrowers < 1_000:
        raise ValueError("Use at least 1,000 borrowers for a meaningful imbalanced-risk sample")
    rng = np.random.default_rng(random_state)
    distress = rng.normal(0, 1, n_borrowers)
    scale = rng.lognormal(mean=np.log(1_350_000), sigma=0.82, size=n_borrowers)
    base_revenue = np.clip(scale, 200_000, 25_000_000)

    industry = rng.choice(INDUSTRIES, n_borrowers, p=[.16, .12, .14, .10, .09, .09, .13, .07, .06, .04])
    state = rng.choice(STATES, n_borrowers, p=[.19, .15, .13, .12, .10, .09, .10, .12])
    business_type = rng.choice(BUSINESS_TYPES, n_borrowers, p=[.39, .22, .29, .10])
    business_age = np.clip(rng.gamma(2.4, 3.1, n_borrowers) - 0.35 * distress, 0.5, 38)
    employee_count = np.clip(
        np.round((base_revenue / 135_000) ** 0.72 + rng.normal(4, 5, n_borrowers)), 2, 750
    ).astype(int)

    revenue_growth_6m = np.clip(rng.normal(0.045 - 0.055 * distress, 0.13, n_borrowers), -0.48, 0.55)
    revenue_growth_3m = np.clip(
        revenue_growth_6m + rng.normal(-0.035 * np.maximum(distress, 0), 0.09, n_borrowers), -0.58, 0.62
    )
    revenue_6m_avg = base_revenue
    revenue_3m_avg = base_revenue * (1 + 0.55 * revenue_growth_3m)
    monthly_revenue = np.clip(base_revenue * (1 + revenue_growth_3m), 90_000, None)
    revenue_volatility = np.clip(rng.beta(2.0, 9.0, n_borrowers) + 0.06 * np.maximum(distress, 0), 0.02, 0.78)

    expense_ratio = np.clip(
        rng.normal(0.73 + 0.035 * distress - 0.03 * revenue_growth_3m, 0.075, n_borrowers), 0.47, 1.13
    )
    monthly_operating_expenses = monthly_revenue * expense_ratio
    fixed_share = np.clip(rng.normal(0.47, 0.11, n_borrowers), 0.22, 0.78)
    fixed_expenses = monthly_operating_expenses * fixed_share
    variable_expenses = monthly_operating_expenses - fixed_expenses
    expense_growth = np.clip(rng.normal(0.035 + 0.035 * distress, 0.085, n_borrowers), -0.20, 0.45)
    operating_cash_flow = monthly_revenue - monthly_operating_expenses

    receivable_days = np.clip(rng.normal(39 + 9 * distress, 15, n_borrowers), 4, 135)
    receivable_days_change = np.clip(rng.normal(2 + 7 * distress, 11, n_borrowers), -32, 62)
    overdue_ratio = np.clip(rng.beta(1.7, 8.5, n_borrowers) + 0.06 * np.maximum(distress, 0), 0, 0.88)
    accounts_receivable = monthly_revenue * receivable_days / 30
    working_capital_drag = monthly_revenue * (
        np.maximum(receivable_days_change, 0) / 210 + 0.07 * overdue_ratio
    )
    free_cash_flow = operating_cash_flow - working_capital_drag

    raw_runway = np.clip(rng.lognormal(mean=np.log(2.6), sigma=0.55, size=n_borrowers) - 0.48 * distress, 0.08, 12)
    average_bank_balance = monthly_operating_expenses * raw_runway
    minimum_bank_balance = np.maximum(
        average_bank_balance * np.clip(rng.normal(0.34 - 0.04 * distress, 0.14, n_borrowers), 0.02, 0.75), 1_000
    )
    cash_flow_volatility = np.clip(0.55 * revenue_volatility + rng.beta(1.8, 8, n_borrowers) + 0.04 * np.maximum(distress, 0), 0.02, 0.95)
    cash_runway_months = average_bank_balance / np.maximum(monthly_operating_expenses, 1)

    debt_to_annual_revenue = np.clip(rng.lognormal(np.log(0.46), 0.55, n_borrowers) + 0.11 * np.maximum(distress, 0), 0.05, 2.35)
    total_existing_debt = monthly_revenue * 12 * debt_to_annual_revenue
    outstanding_loan_amount = total_existing_debt * rng.uniform(0.58, 0.96, n_borrowers)
    interest_rate = np.clip(rng.normal(10.7 + 0.65 * np.maximum(distress, 0), 1.25, n_borrowers), 7.5, 18.5)
    remaining_tenure_months = rng.integers(18, 85, n_borrowers)
    current_emi = _amortizing_emi(outstanding_loan_amount, interest_rate, remaining_tenure_months)
    number_of_active_loans = np.clip(rng.poisson(1.1 + 0.32 * _sigmoid(distress), n_borrowers) + 1, 1, 7)

    delay_intensity = np.exp(-1.25 + 0.62 * distress + 2.0 * np.maximum(-revenue_growth_3m, 0))
    delayed_emi_count_3m = np.clip(rng.poisson(delay_intensity), 0, 3)
    delayed_emi_count_6m = np.clip(delayed_emi_count_3m + rng.poisson(delay_intensity * 0.75), 0, 6)
    missed_emi_count = np.clip(rng.binomial(2, _sigmoid(-3.0 + 0.75 * distress + 0.45 * delayed_emi_count_3m)), 0, 2)
    average_payment_delay_days = np.clip(
        rng.gamma(1.5, 2.2, n_borrowers) + 5.0 * delayed_emi_count_3m + 8 * missed_emi_count, 0, 75
    )
    maximum_payment_delay_days = np.clip(
        average_payment_delay_days + rng.gamma(1.7, 4.2, n_borrowers), 0, 110
    )

    gst_monthly_turnover = monthly_revenue * np.clip(rng.normal(0.985 - 0.018 * distress, 0.055, n_borrowers), 0.70, 1.12)
    gst_turnover_growth = np.clip(revenue_growth_3m + rng.normal(0, 0.045, n_borrowers), -0.62, 0.60)
    gst_vs_bank_credit_difference = np.clip(
        np.abs(gst_monthly_turnover - monthly_revenue) / np.maximum(monthly_revenue, 1), 0, 0.40
    )

    frame = pd.DataFrame(
        {
            "borrower_id": [f"MSME-{index:06d}" for index in range(1, n_borrowers + 1)],
            "industry": industry,
            "state": state,
            "business_type": business_type,
            "business_age_years": business_age.round(1),
            "employee_count": employee_count,
            "monthly_revenue": monthly_revenue.round(2),
            "revenue_3m_avg": revenue_3m_avg.round(2),
            "revenue_6m_avg": revenue_6m_avg.round(2),
            "revenue_growth_3m": revenue_growth_3m,
            "revenue_growth_6m": revenue_growth_6m,
            "revenue_volatility": revenue_volatility,
            "monthly_operating_expenses": monthly_operating_expenses.round(2),
            "fixed_expenses": fixed_expenses.round(2),
            "variable_expenses": variable_expenses.round(2),
            "expense_growth": expense_growth,
            "operating_cash_flow": operating_cash_flow.round(2),
            "free_cash_flow": free_cash_flow.round(2),
            "average_bank_balance": average_bank_balance.round(2),
            "minimum_bank_balance": minimum_bank_balance.round(2),
            "cash_flow_volatility": cash_flow_volatility,
            "cash_runway_months": cash_runway_months,
            "accounts_receivable": accounts_receivable.round(2),
            "receivable_days": receivable_days,
            "receivable_days_change": receivable_days_change,
            "overdue_receivables_ratio": overdue_ratio,
            "outstanding_loan_amount": outstanding_loan_amount.round(2),
            "interest_rate": interest_rate,
            "current_emi": current_emi.round(2),
            "remaining_tenure_months": remaining_tenure_months,
            "total_existing_debt": total_existing_debt.round(2),
            "number_of_active_loans": number_of_active_loans,
            "delayed_emi_count_3m": delayed_emi_count_3m,
            "delayed_emi_count_6m": delayed_emi_count_6m,
            "missed_emi_count": missed_emi_count,
            "average_payment_delay_days": average_payment_delay_days,
            "maximum_payment_delay_days": maximum_payment_delay_days,
            "gst_monthly_turnover": gst_monthly_turnover.round(2),
            "gst_turnover_growth": gst_turnover_growth,
            "gst_vs_bank_credit_difference": gst_vs_bank_credit_difference,
        }
    )

    engineered = engineer_features(frame)
    risk_signal = (
        1.25 * (1.10 - engineered["dscr"]).clip(-1, 2.5)
        + 1.05 * engineered["revenue_deterioration"] / 0.20
        + 0.90 * (engineered["emi_to_revenue"] - 0.11) / 0.10
        + 0.70 * (engineered["receivable_days"] - 45) / 32
        + 0.85 * (1.6 - engineered["cash_runway"]) / 1.6
        + 0.55 * engineered["repayment_delinquency_score"]
        + 0.62 * (engineered["debt_to_revenue"] - 0.55)
        + 0.55 * engineered["cashflow_volatility_index"]
        + 0.36 * engineered["gst_vs_bank_credit_difference"] / 0.10
        - 0.13 * np.log1p(engineered["business_age_years"])
        + rng.normal(0, 1.05, n_borrowers)
    ).to_numpy()
    intercept = _calibrated_intercept(risk_signal, target_default_rate)
    event_probability = _sigmoid(risk_signal + intercept)
    engineered["default_90d"] = rng.binomial(1, event_probability)

    # Missingness is added only after target generation, so it cannot cause label leakage.
    missing_columns = [
        "revenue_3m_avg",
        "revenue_growth_3m",
        "average_bank_balance",
        "minimum_bank_balance",
        "receivable_days_change",
        "gst_turnover_growth",
        "gst_vs_bank_credit_difference",
    ]
    for column in missing_columns:
        mask = rng.random(n_borrowers) < rng.uniform(0.008, 0.025)
        engineered.loc[mask, column] = np.nan

    return engineered


def save_synthetic_dataset(
    output_path: str | Path,
    n_borrowers: int = 12_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate and save the competition dataset, creating parent folders as needed."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_msme_data(n_borrowers=n_borrowers, random_state=random_state)
    frame.to_csv(output, index=False)
    return frame


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "data" / "synthetic_msme_data.csv"
    dataset = save_synthetic_dataset(destination)
    print(f"Saved {len(dataset):,} borrowers to {destination}")
    print(f"Observed 90-day stress prevalence: {dataset['default_90d'].mean():.2%}")
