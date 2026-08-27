# RestructAI — 90-Day MSME Financial-Stress Engine

RestructAI is a production-style, hackathon-friendly credit-risk project for Indian MSMEs. It predicts the probability that a borrower will experience serious repayment stress or default within the next 90 days. It is a decision-support system, not an automatic loan approval or rejection system.

This phase deliberately stops before EMI restructuring optimization. Its output is a calibrated probability, risk score, early-warning assessment and explanation that a later cash-flow and EMI scenario engine can consume.

## Pipeline

```text
MSME Data
    ↓
Data Cleaning
    ↓
Financial Feature Engineering
    ↓
Logistic Regression / XGBoost / CatBoost
    ↓
5-fold Stratified Cross-Validation
    ↓
Validation-Based Model Selection
    ↓
Probability Calibration
    ↓
Threshold Optimization
    ↓
SHAP / Coefficient Explainability
    ↓
Early-Warning Rules + Risk Score
```

The formal benchmark intentionally contains only three classifiers:

> Logistic Regression represents the interpretable baseline, while XGBoost and CatBoost represent high-performing nonlinear gradient-boosting methods suited to structured financial data. The comparison intentionally focuses on predictive performance, explainability and practical deployability rather than exhaustively testing every algorithm.

No SVM, KNN, neural network, Naive Bayes, Random Forest, LightGBM or other classifier is included in the formal comparison.

## Project structure

```text
restructai/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic_msme_data.csv
├── notebooks/exploratory_analysis.ipynb
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── risk_scoring.py
│   ├── explainability.py
│   └── inference.py
├── models/risk_model.joblib
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── threshold_analysis.csv
│   └── model_selection_summary.json
├── tests/test_core_logic.py
├── requirements.txt
└── main.py
```

## Synthetic-data methodology

The generator creates one cross-sectional record per borrower, so borrower-level temporal leakage cannot occur. Financial variables are correlated through a latent stress factor and explicit accounting relationships:

- Revenue level drives employee count, GST turnover and operating scale.
- Operating expenses are derived from revenue and a stress-sensitive expense ratio.
- Operating and free cash flow follow from revenue, expenses and working-capital drag.
- Receivables are derived from monthly revenue and receivable days.
- Bank balances and cash runway are linked to operating expenses.
- Total debt is tied to annualized revenue; EMI is calculated with the amortizing-loan formula.
- Repayment delays become more likely as the latent stress state and revenue deterioration increase.

The `default_90d` target is sampled probabilistically from pre-outcome indicators including DSCR, revenue deterioration, EMI burden, receivable stress, liquidity, delinquency, leverage, cash-flow volatility and GST mismatch. Gaussian noise prevents a perfectly deterministic label. The intercept is numerically calibrated to a target prevalence near 12%, creating realistic class imbalance without using post-default information.

Small amounts of missingness are added after the target is generated. Median/frequent-value imputation is fitted only inside training pipelines and cross-validation folds.

## Financial features

Important engineered indicators include:

- **DSCR:** free cash flow divided by current EMI.
- **EMI burden:** EMI relative to revenue and free cash flow.
- **Debt to revenue:** total debt relative to annualized monthly revenue.
- **Operating margin:** operating cash flow relative to revenue.
- **Revenue deterioration:** weighted negative three- and six-month growth.
- **Receivable stress:** receivable days, deterioration and overdue share.
- **Liquidity stress:** DSCR shortfall, short runway and low minimum balance.
- **Repayment delinquency:** delayed/missed EMIs and payment-delay days.
- **Cash-flow volatility:** combined revenue and cash-flow instability.

All ratios use denominator safeguards, and the exact same feature function and fitted preprocessing pipeline are reused during inference.

## Leakage-safe modelling

The dataset is split into stratified training, validation and test samples using `random_state=42`.

- Five-fold stratified cross-validation is performed only on the training sample.
- Model selection uses validation metrics, not the final test sample.
- Calibration choice is compared on a held-out part of validation data, then fitted on full validation probabilities.
- The operational threshold is optimized separately from probability estimation and prioritizes recall for distressed borrowers.
- The test sample remains untouched until final comparison and reporting.

Class weighting and each boosting library's native imbalance mechanism are used. SMOTE is intentionally not applied; using it outside training folds would create leakage risk, and class weighting provides a simpler auditable baseline.

## Evaluation and model selection

Accuracy is not used as the deciding metric. The project reports:

- Precision and distressed-class recall
- F1 score
- ROC-AUC
- PR-AUC
- Brier score and calibration curve
- Confusion matrix
- Precision/recall/false-positive/false-negative threshold trade-offs

The selection utility gives the greatest weight to PR-AUC and distressed-class recall, while also considering F1, ROC-AUC and Brier score. The code determines the winner from actual validation results; it does not assume that either boosting model must win.

## Explainability and early warnings

For a selected tree model, the project generates a SHAP summary and global SHAP-importance chart. A Logistic Regression winner instead receives standardized coefficient importance. Local explanations report model-output or log-odds contributions—not invented probability percentages.

The rule-assisted early-warning layer flags operational conditions such as DSCR below 1, revenue decline above 15%, receivable days above 60, cash runway below two months, EMI/free-cash-flow burden above 0.8, repeated payment delays and high cash-flow volatility. These rules support interpretation and do not replace the model.

## Running the project

From this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py --rows 12000
.venv\Scripts\python -m pytest -q
```

The complete run generates the dataset, trains the three models, evaluates calibration and thresholds, writes report charts, persists the selected pipeline and prints a sample inference.

Single-borrower inference is available through:

```python
from src.inference import predict_msme_risk

risk_result = predict_msme_risk(borrower_data)
```

It returns a calibrated probability, 0–100 score, Low/Moderate/High/Critical category, operational classification, configurable threshold, model factors and warning flags. Bucket boundaries are prototype policy thresholds and must be calibrated using each lender's risk appetite and governed outcome data.

## Limitations

> Results produced using synthetic data demonstrate the technical methodology and should not be interpreted as validated real-world credit-risk performance. A production model would require appropriately governed historical lender data.

Additional limitations include synthetic portfolio mix, simplified GST/bank-credit reconciliation, cross-sectional rather than account-level time-series modelling, and no fairness or macroeconomic stability validation. Production deployment would require governance, monitoring, drift detection, bias testing, adverse-action review and human credit oversight.

## Next phase interface

```text
Cash-Flow Forecasting Engine
        ↓
Default Risk Model (`predict_msme_risk`)
        ↓
EMI Scenario Generator
        ↓
Constrained Optimization Engine
        ↓
Monte Carlo Stress Testing
        ↓
Recommended Restructuring Plan
```

