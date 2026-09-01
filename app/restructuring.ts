import type { BorrowerInput, RiskResult } from "./risk-model";

export const TARGET_DSCR = 1.2;
export const MAX_PLANNING_TENURE = 120;

export type PlanTerms = {
  annualRate: number;
  totalTenureMonths: number;
  moratoriumMonths: number;
};

export type PlanFeasibility = "Comfortable" | "Tight" | "Not feasible";

export type RestructuringPlan = PlanTerms & {
  id: string;
  name: string;
  description: string;
  assumption: string;
  emi: number;
  monthlyRelief: number;
  projectedDscr: number;
  affordableEmiCap: number;
  requiredMonthlyCashFlow: number;
  cashFlowGap: number;
  emiCoverageGap: number;
  dscrGap: number;
  tenureCapReached: boolean;
  feasibilitySummary: string;
  feasibilityReasons: string[];
  capitalizedPrincipal: number;
  totalRepayment: number;
  totalInterest: number;
  changeVsCurrentSchedule: number;
  feasibility: PlanFeasibility;
};

export type RestructuringTiming = {
  label: string;
  tone: "monitor" | "prepare" | "urgent" | "critical";
  summary: string;
  trigger: string;
  steps: Array<{ when: string; title: string; detail: string }>;
};

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(Math.max(value, minimum), maximum);

const safeDivide = (numerator: number, denominator: number) =>
  denominator > 0 && Number.isFinite(numerator / denominator) ? numerator / denominator : 0;

export function calculateEmi(principal: number, annualRate: number, months: number): number {
  if (principal <= 0 || months <= 0) return 0;
  const monthlyRate = Math.max(annualRate, 0) / 1200;
  if (monthlyRate === 0) return principal / months;
  const growth = Math.pow(1 + monthlyRate, months);
  return principal * monthlyRate * growth / (growth - 1);
}

export function capitalizeDuringMoratorium(principal: number, annualRate: number, months: number): number {
  const monthlyRate = Math.max(annualRate, 0) / 1200;
  return principal * Math.pow(1 + monthlyRate, Math.max(Math.round(months), 0));
}

export function monthsRequiredForEmi(principal: number, annualRate: number, payment: number): number {
  if (principal <= 0) return 0;
  if (payment <= 0) return Number.POSITIVE_INFINITY;
  const monthlyRate = Math.max(annualRate, 0) / 1200;
  if (monthlyRate === 0) return Math.ceil(principal / payment);
  if (payment <= principal * monthlyRate) return Number.POSITIVE_INFINITY;
  return Math.ceil(-Math.log(1 - principal * monthlyRate / payment) / Math.log(1 + monthlyRate));
}

export function buildRestructuringPlan(
  borrower: BorrowerInput,
  terms: PlanTerms,
  metadata: Pick<RestructuringPlan, "id" | "name" | "description" | "assumption">,
): RestructuringPlan {
  const moratoriumMonths = clamp(Math.round(terms.moratoriumMonths), 0, 12);
  const totalTenureMonths = clamp(
    Math.round(terms.totalTenureMonths),
    moratoriumMonths + 1,
    MAX_PLANNING_TENURE,
  );
  const annualRate = clamp(terms.annualRate, 0, 40);
  const repaymentMonths = totalTenureMonths - moratoriumMonths;
  const capitalizedPrincipal = capitalizeDuringMoratorium(
    borrower.outstanding_loan_amount,
    annualRate,
    moratoriumMonths,
  );
  const emi = calculateEmi(capitalizedPrincipal, annualRate, repaymentMonths);
  const affordableEmiCap = Math.max(borrower.free_cash_flow, 0) / TARGET_DSCR;
  const projectedDscr = safeDivide(Math.max(borrower.free_cash_flow, 0), emi);
  const totalRepayment = emi * repaymentMonths;
  const currentRemainingRepayment = borrower.current_emi * borrower.remaining_tenure_months;
  const monthlyRelief = borrower.current_emi - emi;
  const feasibility: PlanFeasibility = borrower.free_cash_flow <= 0 || emi > borrower.free_cash_flow
    ? "Not feasible"
    : emi <= affordableEmiCap
      ? "Comfortable"
      : "Tight";
  const availableCashFlow = Math.max(borrower.free_cash_flow, 0);
  const requiredMonthlyCashFlow = emi * TARGET_DSCR;
  const cashFlowGap = Math.max(requiredMonthlyCashFlow - availableCashFlow, 0);
  const emiCoverageGap = Math.max(emi - availableCashFlow, 0);
  const dscrGap = Math.max(TARGET_DSCR - projectedDscr, 0);
  const tenureCapReached = totalTenureMonths === MAX_PLANNING_TENURE;
  const feasibilityReasons: string[] = [];

  if (borrower.free_cash_flow <= 0) {
    feasibilityReasons.push("The business has no positive monthly free cash flow available for debt service.");
  } else if (emiCoverageGap > 0) {
    feasibilityReasons.push(
      `The proposed EMI exceeds available monthly free cash flow by ₹${Math.round(emiCoverageGap).toLocaleString("en-IN")}.`,
    );
    feasibilityReasons.push(
      `Projected DSCR is ${projectedDscr.toFixed(2)}x, below the 1.00x level needed to cover one full instalment.`,
    );
  } else if (cashFlowGap > 0) {
    feasibilityReasons.push(
      `The EMI is covered, but projected DSCR is ${projectedDscr.toFixed(2)}x, below the ${TARGET_DSCR.toFixed(2)}x planning target.`,
    );
  }

  if (cashFlowGap > 0) {
    feasibilityReasons.push(
      `${TARGET_DSCR.toFixed(2)}x coverage requires ₹${Math.round(requiredMonthlyCashFlow).toLocaleString("en-IN")} of monthly free cash flow, a gap of ₹${Math.round(cashFlowGap).toLocaleString("en-IN")}.`,
    );
  }

  if (tenureCapReached && cashFlowGap > 0) {
    feasibilityReasons.push(
      `The plan already uses the ${MAX_PLANNING_TENURE}-month planning cap, so tenure extension within this tool cannot close the gap.`,
    );
  }

  if (moratoriumMonths > 0 && feasibility === "Not feasible") {
    const capitalizedInterest = Math.max(capitalizedPrincipal - borrower.outstanding_loan_amount, 0);
    feasibilityReasons.push(
      `The payment pause capitalizes about ₹${Math.round(capitalizedInterest).toLocaleString("en-IN")} of interest before repayment begins.`,
    );
  }

  const feasibilitySummary = feasibility === "Comfortable"
    ? `${projectedDscr.toFixed(2)}x DSCR meets the ${TARGET_DSCR.toFixed(2)}x target.`
    : feasibility === "Tight"
      ? `₹${Math.round(cashFlowGap).toLocaleString("en-IN")} more monthly free cash flow is needed for the ${TARGET_DSCR.toFixed(2)}x target.`
      : borrower.free_cash_flow <= 0
        ? "No positive monthly free cash flow is available to service an EMI."
        : `The EMI is ₹${Math.round(emiCoverageGap).toLocaleString("en-IN")} above available monthly free cash flow.`;

  return {
    ...metadata,
    annualRate,
    totalTenureMonths,
    moratoriumMonths,
    emi,
    monthlyRelief,
    projectedDscr,
    affordableEmiCap,
    requiredMonthlyCashFlow,
    cashFlowGap,
    emiCoverageGap,
    dscrGap,
    tenureCapReached,
    feasibilitySummary,
    feasibilityReasons,
    capitalizedPrincipal,
    totalRepayment,
    totalInterest: Math.max(totalRepayment - borrower.outstanding_loan_amount, 0),
    changeVsCurrentSchedule: totalRepayment - currentRemainingRepayment,
    feasibility,
  };
}

export function generateRestructuringPlans(borrower: BorrowerInput): RestructuringPlan[] {
  const affordableEmiCap = Math.max(borrower.free_cash_flow, 0) / TARGET_DSCR;
  const minimumTenure = Math.min(
    Math.max(borrower.remaining_tenure_months + 12, borrower.remaining_tenure_months),
    MAX_PLANNING_TENURE,
  );
  const cashFlowMonths = monthsRequiredForEmi(
    borrower.outstanding_loan_amount,
    borrower.interest_rate,
    affordableEmiCap,
  );
  const cashFlowTenure = clamp(
    Number.isFinite(cashFlowMonths) ? Math.max(cashFlowMonths, minimumTenure) : MAX_PLANNING_TENURE,
    borrower.remaining_tenure_months,
    MAX_PLANNING_TENURE,
  );

  const lowerRate = Math.max(borrower.interest_rate - 1, 0);
  const lowerRateMonths = monthsRequiredForEmi(
    borrower.outstanding_loan_amount,
    lowerRate,
    affordableEmiCap,
  );
  const lowerRateTenure = clamp(
    Number.isFinite(lowerRateMonths) ? Math.max(lowerRateMonths, minimumTenure) : MAX_PLANNING_TENURE,
    borrower.remaining_tenure_months,
    MAX_PLANNING_TENURE,
  );

  const bridgeMonths = 3;
  const bridgePrincipal = capitalizeDuringMoratorium(
    borrower.outstanding_loan_amount,
    borrower.interest_rate,
    bridgeMonths,
  );
  const bridgeRepaymentMonths = monthsRequiredForEmi(
    bridgePrincipal,
    borrower.interest_rate,
    affordableEmiCap,
  );
  const bridgeTenure = clamp(
    Number.isFinite(bridgeRepaymentMonths)
      ? Math.max(bridgeRepaymentMonths + bridgeMonths, minimumTenure + bridgeMonths)
      : MAX_PLANNING_TENURE,
    borrower.remaining_tenure_months,
    MAX_PLANNING_TENURE,
  );

  return [
    buildRestructuringPlan(
      borrower,
      { annualRate: borrower.interest_rate, totalTenureMonths: cashFlowTenure, moratoriumMonths: 0 },
      {
        id: "cash-flow-fit",
        name: "Cash-flow fit",
        description: `Extends tenure until EMI targets at least ${TARGET_DSCR.toFixed(1)}x cash coverage.`,
        assumption: "Same rate; lender accepts a longer amortisation schedule.",
      },
    ),
    buildRestructuringPlan(
      borrower,
      { annualRate: lowerRate, totalTenureMonths: lowerRateTenure, moratoriumMonths: 0 },
      {
        id: "rate-tenure",
        name: "Rate + tenure",
        description: "Pairs a one-point rate concession with the shortest cash-flow-fit tenure.",
        assumption: "Requires lender approval for a 1.0 percentage-point rate reduction.",
      },
    ),
    buildRestructuringPlan(
      borrower,
      { annualRate: borrower.interest_rate, totalTenureMonths: bridgeTenure, moratoriumMonths: bridgeMonths },
      {
        id: "short-bridge",
        name: "3-month bridge",
        description: "Defers principal-and-interest payments briefly, then amortises the capitalised balance.",
        assumption: "Interest accrues during the pause, so this usually has the highest total cost.",
      },
    ),
  ];
}

export function shouldRestructure(borrower: BorrowerInput, risk: RiskResult): boolean {
  return risk.category !== "Low"
    || risk.dscr < TARGET_DSCR
    || borrower.delayed_emi_count_3m > 0
    || borrower.missed_emi_count > 0;
}

export function getRecommendedPlan(plans: RestructuringPlan[]): RestructuringPlan {
  const cashFlowFit = plans.find((plan) => plan.id === "cash-flow-fit");
  if (cashFlowFit && cashFlowFit.feasibility !== "Not feasible") return cashFlowFit;
  return [...plans].sort((left, right) => {
    const feasibilityRank = { Comfortable: 0, Tight: 1, "Not feasible": 2 };
    const feasibilityDifference = feasibilityRank[left.feasibility] - feasibilityRank[right.feasibility];
    if (feasibilityDifference) return feasibilityDifference;
    return left.emi - right.emi;
  })[0];
}

export function getRestructuringTiming(borrower: BorrowerInput, risk: RiskResult): RestructuringTiming {
  const commonSteps = [
    {
      when: "Today",
      title: "Build the evidence pack",
      detail: "Collect six months of bank statements, GST returns, receivable ageing and a 13-week cash-flow forecast.",
    },
    {
      when: "Within 7 days",
      title: "Open the lender discussion",
      detail: "Share the current stress signals, the cash-flow-fit plan and one alternative with transparent cost assumptions.",
    },
    {
      when: "Before changing payment",
      title: "Obtain written approval",
      detail: "Continue the contractual EMI until the lender formally approves a revised repayment schedule.",
    },
    {
      when: "Every month",
      title: "Re-run the early warning check",
      detail: "Monitor DSCR, runway, collections and payment conduct; escalate again if the plan stops fitting cash flow.",
    },
  ];

  if (borrower.missed_emi_count > 0 || risk.category === "Critical") {
    return {
      label: "Act today",
      tone: "critical",
      summary: "EMI-only relief may be insufficient. Request an immediate lender-led viability and resolution review.",
      trigger: "Critical model risk or an existing missed EMI",
      steps: commonSteps,
    };
  }
  if (risk.category === "High" || risk.dscr < 1 || borrower.delayed_emi_count_3m >= 2) {
    return {
      label: "Start this week",
      tone: "urgent",
      summary: "Begin before another due date. The current repayment schedule is already consuming more cash than the business safely generates.",
      trigger: "High risk, DSCR below 1.0x or repeated EMI delays",
      steps: commonSteps,
    };
  }
  if (risk.category === "Moderate" || risk.dscr < TARGET_DSCR || borrower.delayed_emi_count_3m > 0) {
    return {
      label: "Prepare within 30 days",
      tone: "prepare",
      summary: "Prepare a proposal now and approach the lender if coverage or collections weaken further.",
      trigger: `Moderate risk, DSCR below ${TARGET_DSCR.toFixed(1)}x or a recent delay`,
      steps: commonSteps,
    };
  }
  return {
    label: "Monitor — do not restructure now",
    tone: "monitor",
    summary: "The current schedule is covered. Avoid adding interest cost unless the early-warning triggers deteriorate.",
    trigger: `Reconsider if DSCR falls below ${TARGET_DSCR.toFixed(1)}x, runway falls below two months or an EMI is delayed`,
    steps: [commonSteps[0], commonSteps[3]],
  };
}
