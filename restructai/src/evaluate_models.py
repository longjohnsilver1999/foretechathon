"""Model metrics, EDA and competition-ready visual reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"green": "#096149", "lime": "#b7ce45", "sand": "#d9a96d", "ink": "#17372d", "grey": "#b7c2bc"}


def classification_metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute imbalance-aware discrimination, classification and calibration metrics."""

    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "ROC_AUC": roc_auc_score(target, probability),
        "PR_AUC": average_precision_score(target, probability),
        "Precision": precision_score(target, prediction, zero_division=0),
        "Recall": recall_score(target, prediction, zero_division=0),
        "F1": f1_score(target, prediction, zero_division=0),
        "Brier_Score": brier_score_loss(target, probability),
    }


def generate_eda_figures(data: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate focused EDA charts with direct credit-risk interpretation value."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    data["default_90d"].value_counts(normalize=True).sort_index().plot.bar(
        ax=axes[0, 0], color=[PALETTE["green"], PALETTE["sand"]]
    )
    axes[0, 0].set(title="90-day stress class distribution", xlabel="Default / serious stress", ylabel="Share of borrowers")
    missing = data.isna().mean().sort_values(ascending=False).head(12)
    missing.plot.barh(ax=axes[0, 1], color=PALETTE["grey"])
    axes[0, 1].set(title="Highest missing-value rates", xlabel="Missing share", ylabel="")
    industry_rate = data.groupby("industry", observed=True)["default_90d"].mean().sort_values()
    industry_rate.plot.barh(ax=axes[1, 0], color=PALETTE["green"])
    axes[1, 0].set(title="Observed stress rate by industry", xlabel="90-day stress rate", ylabel="")
    age_band = pd.cut(data["business_age_years"], [0, 2, 5, 10, 20, 50], include_lowest=True)
    data.assign(age_band=age_band).groupby("age_band", observed=True)["default_90d"].mean().plot.bar(
        ax=axes[1, 1], color=PALETTE["lime"]
    )
    axes[1, 1].set(title="Stress rate by business age", xlabel="Business age band (years)", ylabel="90-day stress rate")
    fig.tight_layout()
    fig.savefig(output / "eda_portfolio_overview.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    relationships = [
        ("dscr", "DSCR vs observed stress"),
        ("emi_to_revenue", "EMI burden vs observed stress"),
        ("receivable_days", "Receivable days vs observed stress"),
        ("revenue_growth_3m", "Three-month revenue growth vs observed stress"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (feature, title) in zip(axes.flat, relationships):
        plot_data = data[[feature, "default_90d"]].dropna().copy()
        lower, upper = plot_data[feature].quantile([0.01, 0.99])
        plot_data[feature] = plot_data[feature].clip(lower, upper)
        sns.boxplot(data=plot_data, x="default_90d", y=feature, ax=axis, color="#c9ddd4", showfliers=False)
        axis.set(title=title, xlabel="90-day stress outcome", ylabel=feature.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output / "eda_financial_stress_relationships.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    core = [
        "default_90d", "dscr", "emi_to_revenue", "debt_to_revenue", "operating_margin",
        "cash_runway", "receivable_days", "revenue_growth_3m", "repayment_delinquency_score",
        "cashflow_volatility_index",
    ]
    fig, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(data[core].corr(), cmap="RdYlGn", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Correlation of core financial-risk indicators")
    fig.tight_layout()
    fig.savefig(output / "eda_correlation_matrix.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.histplot(data=data, x="monthly_revenue", bins=50, log_scale=True, ax=axes[0], color=PALETTE["green"])
    axes[0].set_title("Monthly revenue distribution (log scale)")
    sns.boxplot(data=data, x="default_90d", y=data["emi_to_revenue"].clip(upper=data["emi_to_revenue"].quantile(.99)), ax=axes[1], color="#e7c798", showfliers=False)
    axes[1].set_title("EMI burden outlier view")
    fig.tight_layout()
    fig.savefig(output / "eda_distributions_and_outliers.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def generate_model_figures(
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
    selected_model: str,
    selected_probability: np.ndarray,
    threshold: float,
    threshold_table: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Create ROC, PR, calibration, threshold, confusion and risk-bucket charts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(target, probability)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(target, probability):.3f})")
        precision, recall, _ = precision_recall_curve(target, probability)
        axes[1].plot(recall, precision, label=f"{name} ({average_precision_score(target, probability):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#9ba49f")
    axes[0].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(np.mean(target), ls="--", color="#9ba49f", label="Portfolio prevalence")
    axes[1].set(title="Precision–Recall curves", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "model_roc_and_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    observed, predicted = calibration_curve(target, selected_probability, n_bins=10, strategy="quantile")
    axes[0].plot(predicted, observed, marker="o", color=PALETTE["green"], label=selected_model)
    axes[0].plot([0, 1], [0, 1], "--", color="#9ba49f", label="Perfect calibration")
    axes[0].set(title="Calibration curve", xlabel="Mean predicted probability", ylabel="Observed stress rate")
    axes[0].legend(frameon=False)
    axes[1].plot(threshold_table["threshold"], threshold_table["precision"], label="Precision", color=PALETTE["sand"])
    axes[1].plot(threshold_table["threshold"], threshold_table["recall"], label="Recall", color=PALETTE["green"])
    axes[1].plot(threshold_table["threshold"], threshold_table["false_positive_rate"], label="False positive rate", color="#8e99a0")
    axes[1].axvline(threshold, ls="--", color=PALETTE["ink"], label=f"Selected {threshold:.2f}")
    axes[1].set(title="Operational threshold trade-offs", xlabel="Classification threshold", ylabel="Rate")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "calibration_and_threshold_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    prediction = (selected_probability >= threshold).astype(int)
    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    ConfusionMatrixDisplay(confusion_matrix(target, prediction), display_labels=["No stress", "Stress"]).plot(
        ax=axis, colorbar=False, cmap="Greens"
    )
    axis.set_title(f"{selected_model} confusion matrix at {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(output / "selected_model_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    risk_score = selected_probability * 100
    bucket = pd.cut(risk_score, [-0.1, 30, 60, 80, 100], labels=["Low", "Moderate", "High", "Critical"])
    risk_frame = pd.DataFrame({"risk_score": risk_score, "risk_bucket": bucket, "default_90d": target})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(risk_frame, x="risk_score", hue="default_90d", bins=35, stat="density", common_norm=False, ax=axes[0], palette=[PALETTE["green"], PALETTE["sand"]])
    axes[0].set(title="Predicted 90-day stress-score distribution", xlabel="Stress score (0–100)")
    bucket_rate = risk_frame.groupby("risk_bucket", observed=False)["default_90d"].mean()
    bucket_rate.plot.bar(ax=axes[1], color=["#94c4ad", "#d7d879", "#e1ad70", "#bc6957"])
    axes[1].set(title="Observed stress rate by risk bucket", xlabel="Prototype risk bucket", ylabel="Observed stress rate")
    fig.tight_layout()
    fig.savefig(output / "risk_distribution_and_bucket_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
