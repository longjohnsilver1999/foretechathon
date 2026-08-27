"""Global and local model explanations with mathematically honest contribution units."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _humanize(name: str) -> str:
    cleaned = name.replace("numeric__", "").replace("categorical__", "")
    return cleaned.replace("_", " ").title()


def _feature_names(pipeline: Any) -> np.ndarray:
    return np.asarray(pipeline.named_steps["preprocessor"].get_feature_names_out(), dtype=str)


def generate_global_explanations(
    pipeline: Any,
    model_name: str,
    sample: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Create SHAP plots for tree models, or coefficient importance for the baseline."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transformed = pipeline.named_steps["preprocessor"].transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = _feature_names(pipeline)
    estimator = pipeline.named_steps["classifier"]

    if model_name in {"XGBoost", "CatBoost"}:
        import shap

        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        importance = np.abs(values).mean(axis=0)
        order = np.argsort(importance)[-15:]

        fig, axis = plt.subplots(figsize=(9, 6.5))
        axis.barh([_humanize(names[index]) for index in order], importance[order], color="#096149")
        axis.set(title=f"{model_name} global SHAP importance", xlabel="Mean absolute SHAP value (model-output units)")
        fig.tight_layout()
        fig.savefig(output / "shap_global_importance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        shap.summary_plot(values, transformed, feature_names=[_humanize(name) for name in names], show=False, max_display=15)
        plt.title(f"{model_name} SHAP summary")
        plt.tight_layout()
        plt.savefig(output / "shap_summary_plot.png", dpi=180, bbox_inches="tight")
        plt.close()
    else:
        coefficients = np.asarray(estimator.coef_[0])
        import shap

        masker = shap.maskers.Independent(transformed, max_samples=len(transformed))
        linear_explainer = shap.LinearExplainer(estimator, masker)
        values = np.asarray(linear_explainer.shap_values(transformed))
        importance = np.abs(values).mean(axis=0)
        order = np.argsort(importance)[-15:]
        fig, axis = plt.subplots(figsize=(9, 6.5))
        axis.barh([_humanize(names[index]) for index in order], importance[order], color="#096149")
        axis.set(title="Logistic Regression global SHAP importance", xlabel="Mean absolute SHAP value (log-odds units)")
        fig.tight_layout()
        fig.savefig(output / "shap_global_importance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        shap.summary_plot(values, transformed, feature_names=[_humanize(name) for name in names], show=False, max_display=15)
        plt.title("Logistic Regression SHAP summary")
        plt.tight_layout()
        plt.savefig(output / "shap_summary_plot.png", dpi=180, bbox_inches="tight")
        plt.close()

    return pd.DataFrame({"feature": [_humanize(name) for name in names], "importance": importance}).sort_values(
        "importance", ascending=False
    )


def explain_single_prediction(artifact: dict[str, Any], borrower: pd.DataFrame, top_n: int = 5) -> dict[str, list[dict[str, float | str]]]:
    """Return local risk/protective contributions in log-odds or standardized units, not percentages."""

    pipeline = artifact["pipeline"]
    model_name = artifact["model_name"]
    transformed = pipeline.named_steps["preprocessor"].transform(borrower)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = _feature_names(pipeline)
    estimator = pipeline.named_steps["classifier"]

    if model_name in {"XGBoost", "CatBoost"}:
        import shap

        values = shap.TreeExplainer(estimator).shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        contribution = np.asarray(values)[0]
        unit = "SHAP model-output contribution"
    else:
        contribution = np.asarray(estimator.coef_[0]) * np.asarray(transformed)[0]
        unit = "standardized log-odds contribution"

    risk_indices = np.argsort(contribution)[::-1]
    protective_indices = np.argsort(contribution)
    risk = [
        {"feature": _humanize(names[index]), "impact": float(contribution[index]), "unit": unit}
        for index in risk_indices[:top_n]
        if contribution[index] > 0
    ]
    protective = [
        {"feature": _humanize(names[index]), "impact": float(contribution[index]), "unit": unit}
        for index in protective_indices[:top_n]
        if contribution[index] < 0
    ]
    return {"top_risk_factors": risk, "protective_factors": protective}
