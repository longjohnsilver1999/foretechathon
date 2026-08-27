import artifact from "./risk-model-artifact.json";

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

export const MODEL_METRICS = artifact.test_metrics;
export const MODEL_NAME = artifact.model_name;
export const PORTFOLIO_PREVALENCE = artifact.dataset_prevalence;
export const MODEL_BENCHMARK = artifact.model_benchmark;

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
  const numericMissing: boolean[] = [];
  const numeric = artifact.numeric_features.map((name, index) => {
    const value = features[name];
    const missing = typeof value !== "number" || !Number.isFinite(value);
    numericMissing.push(missing);
    const imputed = missing ? artifact.numeric_imputer_statistics[index] : value;
    return (imputed - artifact.numeric_scaler_mean[index]) / (artifact.numeric_scaler_scale[index] || 1);
  });

  artifact.missing_indicator_indices.forEach((featureIndex, indicatorIndex) => {
    const raw = numericMissing[featureIndex] ? 1 : 0;
    const scalerIndex = artifact.numeric_features.length + indicatorIndex;
    numeric.push((raw - artifact.numeric_scaler_mean[scalerIndex]) / (artifact.numeric_scaler_scale[scalerIndex] || 1));
  });

  const categorical: number[] = [];
  artifact.categorical_features.forEach((name, featureIndex) => {
    const value = String(features[name] ?? "");
    artifact.categorical_categories[featureIndex].forEach((category) => categorical.push(value === category ? 1 : 0));
  });
  return [...numeric, ...categorical];
}

export function scoreBorrower(input: BorrowerInput): RiskResult {
  const features = buildFeatureRecord(input);
  const transformed = transformForModel(features);
  if (transformed.length !== artifact.coefficients.length) {
    throw new Error("Risk model feature contract is inconsistent");
  }
  const logit = transformed.reduce((total, value, index) => total + value * artifact.coefficients[index], artifact.intercept);
  const rawProbability = sigmoid(logit);
  const calibratedProbability = artifact.calibration_method === "sigmoid"
    ? sigmoid(artifact.calibration_coefficient * Math.log(rawProbability / (1 - rawProbability)) + artifact.calibration_intercept)
    : rawProbability;
  const probability = Math.max(0, Math.min(1, calibratedProbability));
  const score = Math.round(probability * 1000) / 10;
  const category = score <= 30 ? "Low" : score <= 60 ? "Moderate" : score <= 80 ? "High" : "Critical";
  const threshold = artifact.threshold;

  const contributions = artifact.coefficients.map((coefficient, index) => ({
    feature: humanize(artifact.feature_names[index]),
    impact: coefficient * transformed[index],
  })).filter((driver) => !driver.feature.startsWith("Missing "));
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
