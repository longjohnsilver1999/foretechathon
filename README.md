# RestructAI

RestructAI is an explainable 90-day financial-stress prediction system for Indian MSMEs. The project combines a calibrated Python credit-risk pipeline with an interactive web demonstration that runs the exported fitted model directly in the browser.

## What is included

- A correlated 12,000-borrower synthetic MSME dataset with approximately 12% stress prevalence.
- Leakage-safe feature engineering and train/validation/test splitting.
- A formal five-fold benchmark of Logistic Regression, XGBoost and CatBoost only.
- Probability calibration, threshold optimization, SHAP/coefficient explanations and early-warning rules.
- Persisted single-borrower inference through `predict_msme_risk(data)`.
- An interactive risk dashboard with editable borrower inputs, scenario presets and downloadable analysis.
- A cash-flow-based EMI restructuring studio that compares tenure extension, rate-plus-tenure and short-bridge scenarios, including monthly relief, DSCR, lifetime cost and a live model what-if.
- A borrower-aware Savings Coach that explains when to approach the lender and how the selected plan changes the repayment discussion.

The modelling project, methodology, limitations and commands are documented in [restructai/README.md](restructai/README.md).

All displayed performance numbers come from the checked-in experiment outputs. Results use synthetic data to demonstrate methodology and must not be interpreted as validated real-world credit-risk performance or automatic credit decisions.

The restructuring plans are illustrative amortisation scenarios. Borrowers must continue their contractual payments until a lender formally approves revised terms in writing.
