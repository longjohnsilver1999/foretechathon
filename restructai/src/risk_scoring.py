"""Probability calibration, risk buckets and rule-assisted early warnings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    """Small serializable probability calibrator fitted on validation predictions."""

    method: str = "none"
    model: object | None = None

    def fit(self, probability: np.ndarray, target: np.ndarray) -> "ProbabilityCalibrator":
        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        target = np.asarray(target, dtype=int)
        if self.method == "sigmoid":
            logits = np.log(probability / (1 - probability)).reshape(-1, 1)
            self.model = LogisticRegression(random_state=42).fit(logits, target)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(probability, target)
        elif self.method != "none":
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "sigmoid" and self.model is not None:
            logits = np.log(probability / (1 - probability)).reshape(-1, 1)
            return self.model.predict_proba(logits)[:, 1]
        if self.method == "isotonic" and self.model is not None:
            return np.asarray(self.model.predict(probability), dtype=float)
        return probability


def risk_category(score: float) -> str:
    """Map a 0–100 prototype stress score to a policy bucket."""

    if not 0 <= score <= 100:
        raise ValueError("Risk score must be between 0 and 100")
    if score <= 30:
        return "Low"
    if score <= 60:
        return "Moderate"
    if score <= 80:
        return "High"
    return "Critical"


def early_warning_flags(row: pd.Series) -> list[str]:
    """Return transparent operational alerts that complement, not replace, ML."""

    warnings: list[str] = []
    if row.get("dscr", np.inf) < 1.0:
        warnings.append("Debt-service coverage is below a sustainable level (DSCR < 1).")
    if row.get("revenue_growth_3m", 0) < -0.15:
        warnings.append("Revenue has declined by more than 15% over three months.")
    if row.get("receivable_days", 0) > 60 or row.get("receivable_days_change", 0) > 15:
        warnings.append("Receivable cycle is elevated or deteriorating.")
    if row.get("cash_runway", row.get("cash_runway_months", np.inf)) < 2:
        warnings.append("Liquidity runway is below two months.")
    if row.get("emi_to_free_cash_flow", 0) > 0.8:
        warnings.append("EMI consumes more than 80% of free cash flow.")
    if row.get("delayed_emi_count_3m", 0) >= 2 or row.get("missed_emi_count", 0) > 0:
        warnings.append("Repeated EMI delays or missed instalments detected.")
    if row.get("cashflow_volatility_index", 0) > 0.35:
        warnings.append("Significant cash-flow volatility detected.")
    return warnings


def optimize_threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose an operating threshold that explicitly prioritizes distressed recall."""

    rows: list[dict[str, float]] = []
    target = np.asarray(target, dtype=int)
    for threshold in np.arange(0.10, 0.71, 0.02):
        prediction = (probability >= threshold).astype(int)
        tp = int(((prediction == 1) & (target == 1)).sum())
        fp = int(((prediction == 1) & (target == 0)).sum())
        tn = int(((prediction == 0) & (target == 0)).sum())
        fn = int(((prediction == 0) & (target == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        fpr = fp / max(fp + tn, 1)
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": fpr,
                "false_negative_rate": 1 - recall,
                "policy_utility": 0.50 * recall + 0.30 * f1 + 0.20 * precision,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["recall"] >= 0.72]
    winner = (eligible if not eligible.empty else table).sort_values(
        ["policy_utility", "false_positive_rate"], ascending=[False, True]
    ).iloc[0]
    return float(winner["threshold"]), table
