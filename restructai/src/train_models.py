"""Train, compare, calibrate and persist the three formal credit-risk models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .evaluate_models import classification_metrics, generate_eda_figures, generate_model_figures
from .explainability import generate_global_explanations
from .feature_engineering import CATEGORICAL_FEATURES, MODEL_FEATURES, engineer_features
from .preprocessing import build_preprocessor
from .risk_scoring import ProbabilityCalibrator, optimize_threshold

RANDOM_STATE = 42


def _model_definitions(scale_pos_weight: float) -> dict[str, Pipeline]:
    """Return only the three intentionally selected benchmark classifiers."""

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_200,
                        class_weight="balanced",
                        C=0.85,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=280,
                        max_depth=4,
                        learning_rate=0.045,
                        subsample=0.82,
                        colsample_bytree=0.78,
                        min_child_weight=4,
                        reg_alpha=0.15,
                        reg_lambda=2.0,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "CatBoost": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    CatBoostClassifier(
                        iterations=320,
                        depth=6,
                        learning_rate=0.045,
                        l2_leaf_reg=5.0,
                        auto_class_weights="Balanced",
                        loss_function="Logloss",
                        eval_metric="AUC",
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=-1,
                        random_seed=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _selection_score(metrics: dict[str, float]) -> float:
    return (
        0.35 * metrics["PR_AUC"]
        + 0.18 * metrics["ROC_AUC"]
        + 0.24 * metrics["Recall"]
        + 0.15 * metrics["F1"]
        + 0.08 * (1 - metrics["Brier_Score"])
    )


def _choose_calibration(raw_probability: np.ndarray, target: np.ndarray) -> tuple[ProbabilityCalibrator, pd.DataFrame]:
    fit_probability, eval_probability, fit_target, eval_target = train_test_split(
        raw_probability,
        target,
        test_size=0.5,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    rows: list[dict[str, float | str]] = []
    calibrators: dict[str, ProbabilityCalibrator] = {}
    for method in ["none", "sigmoid", "isotonic"]:
        calibrator = ProbabilityCalibrator(method=method).fit(fit_probability, fit_target)
        calibrated = calibrator.predict(eval_probability)
        rows.append({"method": method, "brier_score": brier_score_loss(eval_target, calibrated)})
        calibrators[method] = calibrator
    table = pd.DataFrame(rows).sort_values("brier_score")
    best_method = str(table.iloc[0]["method"])
    raw_brier = float(table.loc[table["method"] == "none", "brier_score"].iloc[0])
    best_brier = float(table.iloc[0]["brier_score"])
    if best_method != "none" and raw_brier - best_brier < 0.0005:
        best_method = "none"
    final = ProbabilityCalibrator(method=best_method).fit(raw_probability, target)
    return final, table


def train_and_evaluate(
    dataset_path: str | Path,
    project_dir: str | Path,
) -> dict[str, Any]:
    """Run the full leakage-safe benchmark and return the persisted model artifact."""

    project = Path(project_dir)
    reports = project / "reports"
    figures = reports / "figures"
    models_dir = project / "models"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(dataset_path)
    data = engineer_features(data)
    if not 0.06 <= data["default_90d"].mean() <= 0.18:
        raise ValueError("Synthetic target prevalence is outside the intended imbalanced-risk range")
    generate_eda_figures(data, figures)

    features = data[MODEL_FEATURES]
    target = data["default_90d"].astype(int)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features, target, test_size=0.20, stratify=target, random_state=RANDOM_STATE
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val, y_train_val, test_size=0.20, stratify=y_train_val, random_state=RANDOM_STATE
    )

    positive = int(y_train.sum())
    scale_pos_weight = (len(y_train) - positive) / max(positive, 1)
    models = _model_definitions(scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"ROC_AUC": "roc_auc", "PR_AUC": "average_precision", "Recall": "recall", "F1": "f1"}

    fitted: dict[str, Pipeline] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    validation_metrics: dict[str, dict[str, float]] = {}
    cv_metrics: dict[str, dict[str, float]] = {}

    for name, pipeline in models.items():
        cv_result = cross_validate(
            clone(pipeline), x_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
        )
        cv_metrics[name] = {
            metric: float(np.mean(cv_result[f"test_{metric}"])) for metric in scoring
        }
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_validation)[:, 1]
        fitted[name] = pipeline
        validation_probabilities[name] = probability
        validation_metrics[name] = classification_metrics(y_validation.to_numpy(), probability, 0.5)

    selected_name = max(validation_metrics, key=lambda name: _selection_score(validation_metrics[name]))
    selected_pipeline = fitted[selected_name]
    calibrator, calibration_table = _choose_calibration(
        validation_probabilities[selected_name], y_validation.to_numpy()
    )
    calibrated_validation = calibrator.predict(validation_probabilities[selected_name])
    threshold, threshold_table = optimize_threshold(y_validation.to_numpy(), calibrated_validation)

    test_probabilities = {name: pipeline.predict_proba(x_test)[:, 1] for name, pipeline in fitted.items()}
    raw_test_rows: list[dict[str, float | str]] = []
    for name, probability in test_probabilities.items():
        row: dict[str, float | str] = {"Model": name}
        row.update(classification_metrics(y_test.to_numpy(), probability, 0.5))
        row.update({f"CV_{metric}": value for metric, value in cv_metrics[name].items()})
        raw_test_rows.append(row)
    comparison = pd.DataFrame(raw_test_rows).sort_values("PR_AUC", ascending=False)
    comparison.to_csv(reports / "model_comparison.csv", index=False)

    calibrated_test = calibrator.predict(test_probabilities[selected_name])
    final_metrics = classification_metrics(y_test.to_numpy(), calibrated_test, threshold)
    threshold_table.to_csv(reports / "threshold_analysis.csv", index=False)
    calibration_table.to_csv(reports / "calibration_comparison.csv", index=False)
    generate_model_figures(
        y_test.to_numpy(),
        test_probabilities,
        selected_name,
        calibrated_test,
        threshold,
        threshold_table,
        figures,
    )

    importance = generate_global_explanations(
        selected_pipeline,
        selected_name,
        x_train.sample(min(500, len(x_train)), random_state=RANDOM_STATE),
        figures,
    )
    importance.to_csv(reports / "global_feature_importance.csv", index=False)

    selection_reason = (
        f"{selected_name} achieved the strongest validation utility when PR-AUC, distressed-class recall, "
        f"F1, ROC-AUC and probability quality were weighted for the early-warning use case. "
        f"The {calibrator.method} calibration choice was retained using held-out validation Brier score."
    )
    feature_names = selected_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    artifact: dict[str, Any] = {
        "pipeline": selected_pipeline,
        "calibrator": calibrator,
        "threshold": threshold,
        "model_name": selected_name,
        "feature_names": feature_names,
        "categorical_metadata": {column: sorted(data[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL_FEATURES},
        "selection_reason": selection_reason,
        "validation_metrics": validation_metrics,
        "test_metrics": final_metrics,
        "calibration_method": calibrator.method,
        "dataset_prevalence": float(data["default_90d"].mean()),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(artifact, models_dir / "risk_model.joblib")

    summary = {
        "selected_model": selected_name,
        "selection_reason": selection_reason,
        "operational_threshold": threshold,
        "calibration_method": calibrator.method,
        "dataset_rows": len(data),
        "dataset_prevalence": float(data["default_90d"].mean()),
        "test_metrics": final_metrics,
        "top_global_features": importance.head(8).to_dict(orient="records"),
    }
    (reports / "model_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return artifact


def print_evaluation_report(artifact: dict[str, Any], comparison_path: str | Path) -> None:
    comparison = pd.read_csv(comparison_path)
    print("=" * 49)
    print("RESTRUCTAI — MODEL EVALUATION")
    print("=" * 49)
    for _, row in comparison.iterrows():
        print(f"\n{row['Model']}")
        for metric in ["ROC_AUC", "PR_AUC", "Recall", "Precision", "F1", "Brier_Score"]:
            print(f"{metric.replace('_', ' ').title()}: {row[metric]:.4f}")
    print("\n" + "=" * 49)
    print(f"SELECTED MODEL: {artifact['model_name']}")
    print("=" * 49)
    print(f"Reason for selection: {artifact['selection_reason']}")
    print(f"Optimal operational threshold: {artifact['threshold']:.2f}")
    print("=" * 49)

