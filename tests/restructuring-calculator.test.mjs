import assert from "node:assert/strict";
import test from "node:test";

import {
  TARGET_DSCR,
  buildRestructuringPlan,
  calculateEmi,
  generateRestructuringPlans,
  getRecommendedPlan,
  getRestructuringTiming,
  monthsRequiredForEmi,
} from "../app/restructuring.ts";

const stressedBorrower = {
  industry: "Textile",
  state: "Gujarat",
  business_type: "Private Limited",
  business_age_years: 7,
  employee_count: 28,
  monthly_revenue: 1_200_000,
  monthly_operating_expenses: 1_020_000,
  free_cash_flow: 128_000,
  average_bank_balance: 720_000,
  revenue_growth_3m: -0.19,
  cash_flow_volatility: 0.42,
  receivable_days: 72,
  receivable_days_change: 19,
  overdue_receivables_ratio: 0.29,
  outstanding_loan_amount: 5_600_000,
  interest_rate: 12.4,
  current_emi: 178_000,
  remaining_tenure_months: 34,
  total_existing_debt: 7_100_000,
  number_of_active_loans: 3,
  delayed_emi_count_3m: 2,
  delayed_emi_count_6m: 3,
  missed_emi_count: 0,
  average_payment_delay_days: 14,
  gst_turnover_growth: -0.16,
  gst_vs_bank_credit_difference: 0.11,
};

test("standard amortisation formula returns the expected EMI", () => {
  assert.ok(Math.abs(calculateEmi(100_000, 12, 12) - 8_884.88) < 0.01);
  assert.equal(calculateEmi(120_000, 0, 12), 10_000);
});

test("cash-flow-fit tenure reaches the target DSCR", () => {
  const cap = stressedBorrower.free_cash_flow / TARGET_DSCR;
  const months = monthsRequiredForEmi(
    stressedBorrower.outstanding_loan_amount,
    stressedBorrower.interest_rate,
    cap,
  );
  assert.ok(calculateEmi(stressedBorrower.outstanding_loan_amount, stressedBorrower.interest_rate, months) <= cap);
  assert.ok(calculateEmi(stressedBorrower.outstanding_loan_amount, stressedBorrower.interest_rate, months - 1) > cap);
});

test("generated plans are distinct and expose cost, relief and feasibility", () => {
  const plans = generateRestructuringPlans(stressedBorrower);
  assert.deepEqual(plans.map((plan) => plan.id), ["cash-flow-fit", "rate-tenure", "short-bridge"]);
  assert.equal(new Set(plans.map((plan) => `${plan.annualRate}-${plan.totalTenureMonths}-${plan.moratoriumMonths}`)).size, 3);
  plans.forEach((plan) => {
    assert.ok(plan.emi > 0);
    assert.ok(plan.totalRepayment > stressedBorrower.outstanding_loan_amount);
    assert.ok(Number.isFinite(plan.projectedDscr));
  });
  const recommended = getRecommendedPlan(plans);
  assert.equal(recommended.id, "cash-flow-fit");
  assert.equal(recommended.feasibility, "Comfortable");
  assert.ok(recommended.monthlyRelief > 0);
  assert.ok(recommended.projectedDscr >= TARGET_DSCR);
});

test("moratorium capitalises interest and increases the repayment base", () => {
  const plan = buildRestructuringPlan(
    stressedBorrower,
    { annualRate: 12.4, totalTenureMonths: 84, moratoriumMonths: 3 },
    { id: "test", name: "Test", description: "Test", assumption: "Test" },
  );
  assert.ok(plan.capitalizedPrincipal > stressedBorrower.outstanding_loan_amount);
  assert.equal(plan.moratoriumMonths, 3);
  assert.ok(plan.totalInterest > 0);
});

test("critical cash flow is escalated when EMI-only restructuring cannot restore viability", () => {
  const criticalBorrower = {
    ...stressedBorrower,
    free_cash_flow: 62_000,
    current_emi: 212_000,
    missed_emi_count: 1,
  };
  const plans = generateRestructuringPlans(criticalBorrower);
  assert.ok(plans.every((plan) => plan.feasibility === "Not feasible"));

  const timing = getRestructuringTiming(criticalBorrower, {
    probability: 0.99,
    score: 99,
    category: "Critical",
    classification: 1,
    threshold: 0.36,
    dscr: criticalBorrower.free_cash_flow / criticalBorrower.current_emi,
    emiBurden: criticalBorrower.current_emi / criticalBorrower.free_cash_flow,
    cashRunway: 0.4,
    drivers: [],
    warnings: [],
  });
  assert.equal(timing.label, "Act today");
  assert.match(timing.summary, /EMI-only relief may be insufficient/);
});
