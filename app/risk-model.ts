import artifact from "./risk-model-artifact.json";

export type ModelName = "Logistic Regression" | "CatBoost" | "XGBoost";

export type ModelMetrics = {
  ROC_AUC: number;
  PR_AUC: number;
  Precision: number;
  Recall: number;
  F1: number;
  Brier_Score: number;
};

type CalibrationExport = {
  method: "none" | "sigmoid" | "isotonic";
  coefficient?: number;
  intercept?: number;
  x_thresholds?: number[];
  y_thresholds?: number[];
};

type XGBoostNode = {
  nodeid: number;
  split?: string;
  split_condition?: number;
  yes?: number;
  no?: number;
  missing?: number;
  leaf?: number;
  children?: XGBoostNode[];
};

type EstimatorExport =
  | { kind: "logistic"; coefficients: number[]; intercept: number }
  | { kind: "xgboost"; base_margin: number; trees: XGBoostNode[] }
  | { kind: "catboost"; scale: number; bias: number; trees: Array<{ splits: Array<{ feature_index: number; border: number }>; leaf_values: number[] }> };

type ModelExport = {
  name: ModelName;
  threshold: number;
  calibration: CalibrationExport;
  test_metrics: ModelMetrics;
  estimator: EstimatorExport;
};

type WebArtifact = {
  schema_version: number;
  default_model: ModelName;
  dataset_rows: number;
  test_rows: number;
  dataset_prevalence: number;
  preprocessor: {
    feature_names: string[];
    numeric_features: string[];
    numeric_imputer_statistics: number[];
    missing_indicator_indices: number[];
    numeric_scaler_mean: number[];
    numeric_scaler_scale: number[];
    categorical_features: string[];
    categorical_categories: string[][];
  };
  models: ModelExport[];
  model_benchmark: Array<ModelMetrics & {
    Model: ModelName;
    Threshold: number;
    Calibration: string;
    CV_ROC_AUC: number;
    CV_PR_AUC: number;
    CV_Recall: number;
    CV_F1: number;
  }>;
  notice: string;
};

const webArtifact = artifact as unknown as WebArtifact;

export type BorrowerInput = {
  industry: string;
  state: string;
  business_type: string;
  business_age_years: number;
  employee_count: number;
  monthly_revenue: number;
  monthly_operating_expenses: number;
  free_cash_flow: number;
  average_bank_balance: number;
  revenue_growth_3m: number;
  cash_flow_volatility: number;
  receivable_days: number;
  receivable_days_change: number;
  overdue_receivables_ratio: number;
  outstanding_loan_amount: number;
  interest_rate: number;
  current_emi: number;
  remaining_tenure_months: number;
  total_existing_debt: number;
  number_of_active_loans: number;
  delayed_emi_count_3m: number;
  delayed_emi_count_6m: number;
  missed_emi_count: number;
  average_payment_delay_days: number;
  gst_turnover_growth: number;
  gst_vs_bank_credit_difference: number;
};

type FeatureRecord = Record<string, number | string>;

export type Driver = {
  feature: string;
  impact: number;
  direction: "risk" | "protective";
};

export type RiskResult = {
  modelName: ModelName;
  calibrationMethod: CalibrationExport["method"];
  driverMethod: "log-odds contribution" | "probability sensitivity";
  probability: number;
  score: number;
  category: "Low" | "Moderate" | "High" | "Critical";
  classification: number;
  threshold: number;
  dscr: number;
  emiBurden: number;
  cashRunway: number;
  drivers: Driver[];
  warnings: string[];
};

export const MODEL_NAME = webArtifact.default_model;
export const AVAILABLE_MODELS = webArtifact.models.map((model) => model.name);
export const PORTFOLIO_PREVALENCE = webArtifact.dataset_prevalence;
export const DATASET_ROWS = webArtifact.dataset_rows;
export const TEST_ROWS = webArtifact.test_rows;
export const MODEL_BENCHMARK = webArtifact.model_benchmark;
export const MODEL_METRICS = webArtifact.models.find((model) => model.name === MODEL_NAME)!.test_metrics;

export function getModelMetrics(modelName: ModelName): ModelMetrics {
  const model = webArtifact.models.find((candidate) => candidate.name === modelName);
  if (!model) throw new RangeError(`Unknown risk model: ${modelName}`);
  return model.test_metrics;
}

export function validateBorrower(input: BorrowerInput): string[] {
  const issues: string[] = [];
  const requiredText: Array<[keyof BorrowerInput, string]> = [
    ["industry", "Industry"],
    ["state", "State"],
    ["business_type", "Business type"],
  ];
  requiredText.forEach(([key, label]) => {
    if (typeof input[key] !== "string" || !String(input[key]).trim()) issues.push(`${label} is required.`);
  });

  const range = (key: keyof BorrowerInput, label: string, minimum: number, maximum: number) => {
    const value = input[key];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      issues.push(`${label} is required.`);
    } else if (value < minimum || value > maximum) {
      issues.push(`${label} must be between ${minimum.toLocaleString("en-IN")} and ${maximum.toLocaleString("en-IN")}.`);
    }
  };

  range("business_age_years", "Business age", 0.1, 100);
  range("employee_count", "Employees", 1, 1_000_000);
  range("monthly_revenue", "Monthly revenue", 1, 10_000_000_000);
  range("monthly_operating_expenses", "Operating expenses", 1, 10_000_000_000);
  range("free_cash_flow", "Free cash flow", -10_000_000_000, 10_000_000_000);
  range("average_bank_balance", "Average bank balance", 0, 100_000_000_000);
  range("revenue_growth_3m", "Three-month revenue growth", -1, 3);
  range("cash_flow_volatility", "Cash-flow volatility", 0, 1);
  range("receivable_days", "Receivable days", 0, 365);
  range("receivable_days_change", "Change in receivable days", -365, 365);
  range("overdue_receivables_ratio", "Overdue receivables", 0, 1);
  range("outstanding_loan_amount", "Outstanding loan", 1, 1_000_000_000_000);
  range("interest_rate", "Interest rate", 0, 40);
  range("current_emi", "Current monthly EMI", 1, 10_000_000_000);
  range("remaining_tenure_months", "Remaining tenure", 1, 360);
  range("total_existing_debt", "Total existing debt", 1, 1_000_000_000_000);
  range("number_of_active_loans", "Active loans", 1, 20);
  range("delayed_emi_count_3m", "Delayed EMIs in three months", 0, 3);
  range("delayed_emi_count_6m", "Delayed EMIs in six months", 0, 6);
  range("missed_emi_count", "Missed EMIs", 0, 6);
  range("average_payment_delay_days", "Average payment delay", 0, 365);
  range("gst_turnover_growth", "GST turnover growth", -1, 3);
  range("gst_vs_bank_credit_difference", "GST-to-bank variance", 0, 1);

  const integerFields: Array<[keyof BorrowerInput, string]> = [
    ["employee_count", "Employees"],
    ["remaining_tenure_months", "Remaining tenure"],
    ["number_of_active_loans", "Active loans"],
    ["delayed_emi_count_3m", "Delayed EMIs in three months"],
    ["delayed_emi_count_6m", "Delayed EMIs in six months"],
    ["missed_emi_count", "Missed EMIs"],
  ];
  integerFields.forEach(([key, label]) => {
    const value = input[key];
    if (typeof value === "number" && Number.isFinite(value) && !Number.isInteger(value)) issues.push(`${label} must be a whole number.`);
  });
  if (Number.isFinite(input.total_existing_debt) && Number.isFinite(input.outstanding_loan_amount) && input.total_existing_debt < input.outstanding_loan_amount) {
    issues.push("Total existing debt cannot be lower than the outstanding loan.");
  }
  if (Number.isFinite(input.delayed_emi_count_6m) && Number.isFinite(input.delayed_emi_count_3m) && input.delayed_emi_count_6m < input.delayed_emi_count_3m) {
    issues.push("Six-month delayed EMIs cannot be lower than the three-month count.");
  }
  return issues;
}

const safeDivide = (numerator: number, denominator: number) =>
  Math.abs(denominator) > 1e-6 && Number.isFinite(numerator / denominator)
    ? numerator / denominator
    : 0;

const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));

const humanize = (name: string) =>
  name
    .replace("numeric__", "")
    .replace("categorical__", "")
    .replace("missingindicator_", "Missing ")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

export function buildFeatureRecord(input: BorrowerInput): FeatureRecord {
  const operatingCashFlow = input.monthly_revenue - input.monthly_operating_expenses;
  const revenueGrowth6m = input.revenue_growth_3m * 0.65;
  const revenue3mAvg = input.monthly_revenue / Math.max(1 + input.revenue_growth_3m * 0.45, 0.3);
  const revenue6mAvg = input.monthly_revenue / Math.max(1 + revenueGrowth6m * 0.5, 0.3);
  const minimumBankBalance = input.average_bank_balance * 0.28;
  const accountsReceivable = input.monthly_revenue * input.receivable_days / 30;
  const gstMonthlyTurnover = input.monthly_revenue * (1 - Math.min(input.gst_vs_bank_credit_difference, 0.4));
  const dscr = safeDivide(input.free_cash_flow, input.current_emi);
  const emiToRevenue = safeDivide(input.current_emi, input.monthly_revenue);
  const emiToFreeCashFlow = safeDivide(input.current_emi, Math.max(input.free_cash_flow, 1e-6));
  const debtToRevenue = safeDivide(input.total_existing_debt, input.monthly_revenue * 12);
  const operatingMargin = safeDivide(operatingCashFlow, input.monthly_revenue);
  const cashRunway = safeDivide(input.average_bank_balance, input.monthly_operating_expenses);
  const receivablesToRevenue = safeDivide(accountsReceivable, input.monthly_revenue);
  const expenseToRevenue = safeDivide(input.monthly_operating_expenses, input.monthly_revenue);
  const revenueMomentum = 0.65 * input.revenue_growth_3m + 0.35 * revenueGrowth6m;
  const revenueDeterioration = Math.max(-input.revenue_growth_3m, 0) + 0.5 * Math.max(-revenueGrowth6m, 0);
  const receivableStress = input.receivable_days / 60 + Math.max(input.receivable_days_change, 0) / 30 + input.overdue_receivables_ratio;
  const repaymentDelinquencyScore = input.delayed_emi_count_3m + 0.5 * input.delayed_emi_count_6m + 3 * input.missed_emi_count + input.average_payment_delay_days / 15;
  const liquidityStress = Math.max(1.25 - dscr, 0) + Math.max(2 - cashRunway, 0) / 2 + safeDivide(Math.max(input.monthly_operating_expenses - minimumBankBalance, 0), input.monthly_operating_expenses);
  const debtBurden = emiToRevenue * 2.5 + debtToRevenue + input.number_of_active_loans / 5;
  const revenueVolatility = Math.min(Math.max(input.cash_flow_volatility * 0.74, 0.02), 0.78);
  const cashflowVolatilityIndex = 0.55 * input.cash_flow_volatility + 0.45 * revenueVolatility;

  return {
    ...input,
    revenue_3m_avg: revenue3mAvg,
    revenue_6m_avg: revenue6mAvg,
    revenue_growth_6m: revenueGrowth6m,
    revenue_volatility: revenueVolatility,
    fixed_expenses: input.monthly_operating_expenses * 0.47,
    variable_expenses: input.monthly_operating_expenses * 0.53,
    expense_growth: Math.max(0.02, -input.revenue_growth_3m * 0.42),
    operating_cash_flow: operatingCashFlow,
    minimum_bank_balance: minimumBankBalance,
    cash_runway_months: cashRunway,
    accounts_receivable: accountsReceivable,
    maximum_payment_delay_days: input.average_payment_delay_days * 1.9,
    gst_monthly_turnover: gstMonthlyTurnover,
    dscr,
    emi_to_revenue: emiToRevenue,
    emi_to_free_cash_flow: emiToFreeCashFlow,
    debt_to_revenue: debtToRevenue,
    operating_margin: operatingMargin,
    cash_runway: cashRunway,
    receivables_to_revenue: receivablesToRevenue,
    expense_to_revenue: expenseToRevenue,
    revenue_momentum: revenueMomentum,
    revenue_deterioration: revenueDeterioration,
    receivable_stress: receivableStress,
    repayment_delinquency_score: repaymentDelinquencyScore,
    liquidity_stress: liquidityStress,
    debt_burden: debtBurden,
    cashflow_volatility_index: cashflowVolatilityIndex,
  };
}

function transformForModel(features: FeatureRecord) {
  const preprocessor = webArtifact.preprocessor;
  const numericMissing: boolean[] = [];
  const numeric = preprocessor.numeric_features.map((name, index) => {
    const value = features[name];
    const missing = typeof value !== "number" || !Number.isFinite(value);
    numericMissing.push(missing);
    const imputed = missing ? preprocessor.numeric_imputer_statistics[index] : value;
    return (imputed - preprocessor.numeric_scaler_mean[index]) / (preprocessor.numeric_scaler_scale[index] || 1);
  });

  preprocessor.missing_indicator_indices.forEach((featureIndex, indicatorIndex) => {
    const raw = numericMissing[featureIndex] ? 1 : 0;
    const scalerIndex = preprocessor.numeric_features.length + indicatorIndex;
    numeric.push((raw - preprocessor.numeric_scaler_mean[scalerIndex]) / (preprocessor.numeric_scaler_scale[scalerIndex] || 1));
  });

  const categorical: number[] = [];
  preprocessor.categorical_features.forEach((name, featureIndex) => {
    const value = String(features[name] ?? "");
    preprocessor.categorical_categories[featureIndex].forEach((category) => categorical.push(value === category ? 1 : 0));
  });
  return [...numeric, ...categorical];
}

function evaluateXGBoostTree(node: XGBoostNode, transformed: number[]): number {
  if (typeof node.leaf === "number") return node.leaf;
  if (!node.split || typeof node.split_condition !== "number") throw new Error("Invalid XGBoost tree node");
  const featureIndex = Number(node.split.replace(/^f/, ""));
  // XGBoost's DMatrix stores inputs as float32. Reproduce that cast before
  // applying learned split thresholds so browser and Python branches match.
  const value = Math.fround(transformed[featureIndex]);
  const nextId = !Number.isFinite(value)
    ? node.missing
    : value < node.split_condition ? node.yes : node.no;
  const child = node.children?.find((candidate) => candidate.nodeid === nextId);
  if (!child) throw new Error("XGBoost tree branch is missing");
  return evaluateXGBoostTree(child, transformed);
}

function rawModelProbability(model: ModelExport, transformed: number[]): number {
  const estimator = model.estimator;
  if (estimator.kind === "logistic") {
    if (transformed.length !== estimator.coefficients.length) throw new Error("Risk model feature contract is inconsistent");
    const margin = transformed.reduce((total, value, index) => total + value * estimator.coefficients[index], estimator.intercept);
    return sigmoid(margin);
  }
  if (estimator.kind === "xgboost") {
    const margin = estimator.trees.reduce(
      (total, tree) => total + evaluateXGBoostTree(tree, transformed),
      estimator.base_margin,
    );
    return sigmoid(margin);
  }
  const treeTotal = estimator.trees.reduce((total, tree) => {
    let leafIndex = 0;
    tree.splits.forEach((split, depth) => {
      if (transformed[split.feature_index] > split.border) leafIndex |= 1 << depth;
    });
    return total + tree.leaf_values[leafIndex];
  }, 0);
  return sigmoid(estimator.scale * treeTotal + estimator.bias);
}

function interpolateIsotonic(value: number, xs: number[], ys: number[]): number {
  if (!xs.length || xs.length !== ys.length) return value;
  if (value <= xs[0]) return ys[0];
  if (value >= xs[xs.length - 1]) return ys[ys.length - 1];
  let low = 0;
  let high = xs.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (value < xs[middle]) high = middle;
    else low = middle;
  }
  const distance = xs[high] - xs[low];
  return distance <= 1e-12 ? ys[low] : ys[low] + (value - xs[low]) / distance * (ys[high] - ys[low]);
}

function calibrateProbability(rawProbability: number, calibration: CalibrationExport): number {
  const clipped = Math.max(1e-6, Math.min(1 - 1e-6, rawProbability));
  if (calibration.method === "sigmoid") {
    const logit = Math.log(clipped / (1 - clipped));
    return sigmoid((calibration.coefficient ?? 1) * logit + (calibration.intercept ?? 0));
  }
  if (calibration.method === "isotonic") {
    return interpolateIsotonic(clipped, calibration.x_thresholds ?? [], calibration.y_thresholds ?? []);
  }
  return clipped;
}

function modelDrivers(model: ModelExport, transformed: number[], probability: number) {
  const featureNames = webArtifact.preprocessor.feature_names;
  if (model.estimator.kind === "logistic") {
    return model.estimator.coefficients.map((coefficient, index) => ({
      feature: humanize(featureNames[index]),
      impact: coefficient * transformed[index],
    })).filter((driver) => !driver.feature.startsWith("Missing "));
  }
  return transformed.map((value, index) => {
    if (Math.abs(value) < 1e-12) return { feature: humanize(featureNames[index]), impact: 0 };
    const withoutFeature = [...transformed];
    withoutFeature[index] = 0;
    const baselineProbability = calibrateProbability(rawModelProbability(model, withoutFeature), model.calibration);
    return {
      feature: humanize(featureNames[index]),
      impact: probability - baselineProbability,
    };
  }).filter((driver) => !driver.feature.startsWith("Missing "));
}

export function scoreBorrower(input: BorrowerInput, modelName: ModelName = MODEL_NAME): RiskResult {
  const validationIssues = validateBorrower(input);
  if (validationIssues.length) throw new RangeError(validationIssues.join(" "));
  const model = webArtifact.models.find((candidate) => candidate.name === modelName);
  if (!model) throw new RangeError(`Unknown risk model: ${modelName}`);
  const features = buildFeatureRecord(input);
  const transformed = transformForModel(features);
  const rawProbability = rawModelProbability(model, transformed);
  const calibratedProbability = calibrateProbability(rawProbability, model.calibration);
  const probability = Math.max(0, Math.min(1, calibratedProbability));
  const score = Math.round(probability * 1000) / 10;
  const category = score <= 30 ? "Low" : score <= 60 ? "Moderate" : score <= 80 ? "High" : "Critical";
  const threshold = model.threshold;

  const contributions = modelDrivers(model, transformed, probability);
  const risk = contributions.filter((driver) => driver.impact > 0).sort((a, b) => b.impact - a.impact).slice(0, 5).map((driver) => ({ ...driver, direction: "risk" as const }));
  const protective = contributions.filter((driver) => driver.impact < 0).sort((a, b) => a.impact - b.impact).slice(0, 3).map((driver) => ({ ...driver, direction: "protective" as const }));

  const dscr = Number(features.dscr);
  const emiBurden = Number(features.emi_to_free_cash_flow);
  const cashRunway = Number(features.cash_runway);
  const warnings: string[] = [];
  if (dscr < 1) warnings.push("Debt-service coverage is below 1.0x");
  if (input.revenue_growth_3m < -0.15) warnings.push("Revenue has fallen more than 15% in three months");
  if (input.receivable_days > 60 || input.receivable_days_change > 15) warnings.push("Receivable cycle is elevated or deteriorating");
  if (cashRunway < 2) warnings.push("Cash runway is below two months");
  if (emiBurden > 0.8) warnings.push("EMI consumes over 80% of free cash flow");
  if (input.delayed_emi_count_3m >= 2 || input.missed_emi_count > 0) warnings.push("Repeated EMI delays or missed payments detected");
  if (Number(features.cashflow_volatility_index) > 0.35) warnings.push("Cash-flow volatility is materially elevated");

  return {
    modelName,
    calibrationMethod: model.calibration.method,
    driverMethod: model.estimator.kind === "logistic" ? "log-odds contribution" : "probability sensitivity",
    probability,
    score,
    category,
    classification: Number(probability >= threshold),
    threshold,
    dscr,
    emiBurden,
    cashRunway,
    drivers: [...risk, ...protective],
    warnings,
  };
}
