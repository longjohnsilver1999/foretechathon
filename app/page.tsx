"use client";

import { type CSSProperties, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Bot,
  Calculator,
  CalendarClock,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Coins,
  Database,
  Download,
  FlaskConical,
  Info,
  Landmark,
  LayoutDashboard,
  Menu,
  MessageCircle,
  PiggyBank,
  RefreshCw,
  ShieldCheck,
  Send,
  SlidersHorizontal,
  Sparkles,
  TrendingDown,
  TrendingUp,
  WalletCards,
  X,
} from "lucide-react";

import {
  type BorrowerInput,
  MODEL_BENCHMARK,
  MODEL_METRICS,
  MODEL_NAME,
  PORTFOLIO_PREVALENCE,
  scoreBorrower,
  validateBorrower,
} from "./risk-model";
import {
  type PlanTerms,
  buildRestructuringPlan,
  generateRestructuringPlans,
  getRecommendedPlan,
  getRestructuringTiming,
  shouldRestructure,
} from "./restructuring";

type InputTab = "financial" | "debt" | "conduct" | "business";
type ChatMessage = { id: number; role: "assistant" | "user"; text: string };

const QUICK_PROMPTS = [
  "Compare restructuring plans",
  "When should I restructure?",
  "How much EMI relief?",
  "Why this risk score?",
];

const LOGISTIC_BENCHMARK = MODEL_BENCHMARK.find((model) => model.Model === "Logistic Regression");
const CATBOOST_BENCHMARK = MODEL_BENCHMARK.find((model) => model.Model === "CatBoost");

const stressedBorrower: BorrowerInput = {
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

const scenarioPresets: Record<string, Partial<BorrowerInput>> = {
  Stable: {
    monthly_revenue: 1_550_000,
    monthly_operating_expenses: 1_030_000,
    free_cash_flow: 390_000,
    average_bank_balance: 3_200_000,
    revenue_growth_3m: 0.08,
    cash_flow_volatility: 0.17,
    receivable_days: 36,
    receivable_days_change: -2,
    overdue_receivables_ratio: 0.08,
    outstanding_loan_amount: 3_700_000,
    current_emi: 128_000,
    total_existing_debt: 4_400_000,
    number_of_active_loans: 1,
    delayed_emi_count_3m: 0,
    delayed_emi_count_6m: 0,
    missed_emi_count: 0,
    average_payment_delay_days: 2,
    gst_turnover_growth: 0.07,
    gst_vs_bank_credit_difference: 0.03,
  },
  Stressed: stressedBorrower,
  Critical: {
    monthly_revenue: 950_000,
    monthly_operating_expenses: 910_000,
    free_cash_flow: 62_000,
    average_bank_balance: 360_000,
    revenue_growth_3m: -0.31,
    cash_flow_volatility: 0.59,
    receivable_days: 92,
    receivable_days_change: 31,
    overdue_receivables_ratio: 0.43,
    current_emi: 212_000,
    total_existing_debt: 8_400_000,
    number_of_active_loans: 4,
    delayed_emi_count_3m: 3,
    delayed_emi_count_6m: 6,
    missed_emi_count: 1,
    average_payment_delay_days: 29,
    gst_turnover_growth: -0.28,
    gst_vs_bank_credit_difference: 0.18,
  },
};

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const money = (value: number) => `₹${Math.round(value).toLocaleString("en-IN")}`;
const percentagePoints = (value: number) => `${Math.abs(value * 100).toFixed(1)} pp`;

export default function Home() {
  const [draft, setDraft] = useState<BorrowerInput>(stressedBorrower);
  const [borrower, setBorrower] = useState<BorrowerInput>(stressedBorrower);
  const [tab, setTab] = useState<InputTab>("financial");
  const [activePreset, setActivePreset] = useState("Stressed");
  const [activeSection, setActiveSection] = useState("analysis");
  const [mobileNav, setMobileNav] = useState(false);
  const [toast, setToast] = useState("");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [selectedPlanId, setSelectedPlanId] = useState("cash-flow-fit");
  const [planOverrides, setPlanOverrides] = useState<PlanTerms | null>(null);
  const nextMessageId = useRef(2);
  const chatLogRef = useRef<HTMLDivElement>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { id: 1, role: "assistant", text: "Hi, I’m your Savings Coach. I use the current borrower score and cash-flow plan to suggest practical next steps. Ask me for an action plan, EMI relief, or ways to improve runway." },
  ]);
  const result = useMemo(() => scoreBorrower(borrower), [borrower]);
  const validationIssues = useMemo(() => validateBorrower(draft), [draft]);
  const planOptions = useMemo(() => generateRestructuringPlans(borrower), [borrower]);
  const recommendedPlan = useMemo(() => getRecommendedPlan(planOptions), [planOptions]);
  const selectedBasePlan = planOptions.find((plan) => plan.id === selectedPlanId) ?? recommendedPlan;
  const selectedPlan = useMemo(() => planOverrides
    ? buildRestructuringPlan(
        borrower,
        planOverrides,
        {
          id: selectedBasePlan.id,
          name: `Custom ${selectedBasePlan.name}`,
          description: "A live what-if plan using the terms selected below.",
          assumption: "Illustrative terms only; the lender must validate and approve the final schedule.",
        },
      )
    : selectedBasePlan,
  [borrower, planOverrides, selectedBasePlan]);
  const projectedResult = useMemo(
    () => scoreBorrower({ ...borrower, current_emi: Math.max(selectedPlan.emi, 1) }),
    [borrower, selectedPlan.emi],
  );
  const restructuringTiming = useMemo(() => getRestructuringTiming(borrower, result), [borrower, result]);
  const restructuringNeeded = useMemo(() => shouldRestructure(borrower, result), [borrower, result]);
  const modelProbabilityImprovement = result.probability - projectedResult.probability;
  const savingsPlan = useMemo(() => {
    const emiDiscussionTarget = result.category === "Low" ? borrower.current_emi : recommendedPlan.emi;
    const monthlyRelief = Math.max(borrower.current_emi - emiDiscussionTarget, 0);
    const reserveTarget = borrower.monthly_operating_expenses * 2;
    const reserveGap = Math.max(reserveTarget - borrower.average_bank_balance, 0);
    const daysToRecover = Math.min(Math.max(borrower.receivable_days - 45, 0), 15);
    const collectionRelease = borrower.monthly_revenue * daysToRecover / 30;
    return { emiDiscussionTarget, monthlyRelief, reserveTarget, reserveGap, daysToRecover, collectionRelease };
  }, [borrower, recommendedPlan.emi, result.category]);
  const downloadHref = useMemo(() => {
    const lines = [
      "RESTRUCTAI — MSME 90-DAY RISK ANALYSIS",
      "",
      `Predicted stress probability: ${percent(result.probability)}`,
      `Stress score: ${result.score} / 100`,
      `Risk category: ${result.category}`,
      `Operational threshold: ${percent(result.threshold)}`,
      `Selected model: ${MODEL_NAME} with sigmoid calibration`,
      "",
      "Early-warning signals:",
      ...result.warnings.map((warning) => `- ${warning}`),
      "",
      "Main model drivers:",
      ...result.drivers.slice(0, 5).map((driver) => `- ${driver.feature}: ${driver.impact > 0 ? "+" : ""}${driver.impact.toFixed(3)} log-odds contribution`),
      "",
      "Savings conversation plan:",
      `- Timing recommendation: ${restructuringTiming.label}`,
      `- Selected option: ${selectedPlan.name}`,
      `- Illustrative EMI: ${money(selectedPlan.emi)} per month`,
      `- Potential monthly EMI relief: ${money(Math.max(selectedPlan.monthlyRelief, 0))}`,
      `- Proposed rate / total tenure / moratorium: ${selectedPlan.annualRate.toFixed(1)}% / ${selectedPlan.totalTenureMonths} months / ${selectedPlan.moratoriumMonths} months`,
      `- Projected DSCR: ${selectedPlan.projectedDscr.toFixed(2)}x`,
      `- Estimated total interest: ${money(selectedPlan.totalInterest)}`,
      `- Two-month operating reserve gap: ${money(savingsPlan.reserveGap)}`,
      "",
      "Decision support only. Synthetic-data methodology demonstration; not validated for automatic credit decisions.",
    ];
    return `data:text/plain;charset=utf-8,${encodeURIComponent(lines.join("\n"))}`;
  }, [result, restructuringTiming.label, selectedPlan, savingsPlan.reserveGap]);

  useEffect(() => {
    if (!chatOpen) return;
    chatLogRef.current?.scrollTo({ top: chatLogRef.current.scrollHeight });
  }, [chatMessages, chatOpen]);

  const updateNumber = (key: keyof BorrowerInput, value: string) => {
    const parsed = value.trim() === "" ? Number.NaN : Number(value);
    setDraft((current) => ({ ...current, [key]: parsed }));
    setActivePreset("Custom");
  };

  const updatePercent = (key: keyof BorrowerInput, value: string) => {
    const parsed = value.trim() === "" ? Number.NaN : Number(value) / 100;
    setDraft((current) => ({ ...current, [key]: parsed }));
    setActivePreset("Custom");
  };

  const updateText = (key: keyof BorrowerInput, value: string) => {
    setDraft((current) => ({ ...current, [key]: value }));
    setActivePreset("Custom");
  };

  const choosePreset = (name: string) => {
    const next = { ...stressedBorrower, ...scenarioPresets[name] };
    const nextRecommended = getRecommendedPlan(generateRestructuringPlans(next));
    setDraft(next);
    setBorrower(next);
    setActivePreset(name);
    setSelectedPlanId(nextRecommended.id);
    setPlanOverrides(null);
    setToast(`${name} scenario loaded and rescored.`);
    window.setTimeout(() => setToast(""), 2200);
  };

  const runAnalysis = () => {
    if (validationIssues.length) {
      setToast(validationIssues[0]);
      window.setTimeout(() => setToast(""), 3200);
      return;
    }
    const nextRecommended = getRecommendedPlan(generateRestructuringPlans(draft));
    setBorrower(draft);
    setSelectedPlanId(nextRecommended.id);
    setPlanOverrides(null);
    setToast("90-day stress analysis updated.");
    window.setTimeout(() => setToast(""), 2200);
  };

  const confirmDownload = () => {
    setToast("Risk analysis downloaded.");
    window.setTimeout(() => setToast(""), 2200);
  };

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ block: "start" });
    setActiveSection(id);
    setMobileNav(false);
  };

  const choosePlan = (id: string) => {
    setSelectedPlanId(id);
    setPlanOverrides(null);
  };

  const updatePlanTerms = (changes: Partial<PlanTerms>) => {
    setPlanOverrides((current) => ({
      annualRate: current?.annualRate ?? selectedPlan.annualRate,
      totalTenureMonths: current?.totalTenureMonths ?? selectedPlan.totalTenureMonths,
      moratoriumMonths: current?.moratoriumMonths ?? selectedPlan.moratoriumMonths,
      ...changes,
    }));
  };

  const resetToRecommendedPlan = () => {
    setSelectedPlanId(recommendedPlan.id);
    setPlanOverrides(null);
  };

  const coachResponse = (question: string) => {
    const normalized = question.toLowerCase();
    const leadingDrivers = result.drivers.filter((driver) => driver.direction === "risk").slice(0, 2).map((driver) => driver.feature).join(" and ");
    const caution = " Treat this as a lender-discussion aid, not a binding restructuring offer.";
    if (/logistic|catboost|xgboost|model/.test(normalized)) {
      const figures = MODEL_BENCHMARK.map((model) => `${model.Model}: ROC–AUC ${model.ROC_AUC.toFixed(4)}, PR–AUC ${model.PR_AUC.toFixed(4)}`).join("; ");
      return `The three models are producing different results. ${figures}. Logistic Regression was selected for borrower scoring because it had the strongest validation utility; the other two remain portfolio benchmarks.`;
    }
    if (/when|timing|too late|now/.test(normalized)) {
      return `${restructuringTiming.label}. ${restructuringTiming.summary} Trigger: ${restructuringTiming.trigger}. Keep paying the contractual EMI until the lender approves revised terms in writing.${caution}`;
    }
    if (/restruct|plan|tenure|moratorium|refinance|option|compare/.test(normalized)) {
      if (!restructuringNeeded) return `The current schedule is covered, so restructuring is not recommended now. Monitor monthly and reconsider if DSCR falls below 1.2x, runway drops below two months or an EMI is delayed.${caution}`;
      return `The current best discussion anchor is ${recommendedPlan.name}: about ${money(recommendedPlan.emi)} per month for ${recommendedPlan.totalTenureMonths} months at ${recommendedPlan.annualRate.toFixed(1)}%, producing ${recommendedPlan.projectedDscr.toFixed(2)}x projected DSCR. It may relieve about ${money(Math.max(recommendedPlan.monthlyRelief, 0))} per month, while estimated total interest becomes ${money(recommendedPlan.totalInterest)}. Compare it with the rate-concession and short-bridge options in the Restructuring Studio.${caution}`;
    }
    if (/emi|relief|payment|instalment/.test(normalized)) {
      if (result.category === "Low") return `Current EMI coverage looks manageable. Keep the EMI at ${money(borrower.current_emi)} while preserving at least two months of operating expenses as cash. Avoid extending tenure unless cash flow weakens.`;
      return `The ${recommendedPlan.name} option produces an illustrative EMI near ${money(recommendedPlan.emi)} per month, about ${money(Math.max(recommendedPlan.monthlyRelief, 0))} below the current EMI. It targets ${recommendedPlan.projectedDscr.toFixed(2)}x cash coverage over ${recommendedPlan.totalTenureMonths} months. Review the added interest cost before choosing.${caution}`;
    }
    if (/cash|runway|reserve|saving/.test(normalized)) {
      return `A two-month operating reserve is ${money(savingsPlan.reserveTarget)}. The current gap is about ${money(savingsPlan.reserveGap)}. Ring-fence collections, pause nonessential capex and sweep a fixed share of weekly receipts into a protected operating reserve.${caution}`;
    }
    if (/receivable|collection|invoice/.test(normalized)) {
      return savingsPlan.daysToRecover > 0
        ? `Reducing receivable days by ${savingsPlan.daysToRecover} days could release roughly ${money(savingsPlan.collectionRelease)} of working capital. Prioritize the largest overdue invoices, offer selective early-payment discounts and escalate disputed invoices this week.${caution}`
        : `Receivable days are already near the healthy planning level. Keep weekly ageing reviews and avoid broad discounts that reduce margin.`;
    }
    if (/why|risk|score|driver/.test(normalized)) {
      return `This borrower is ${result.category.toLowerCase()} risk at ${percent(result.probability)}. The leading model pressures are ${leadingDrivers || "the combined financial signals"}; the rule layer also found ${result.warnings.length} early-warning signal${result.warnings.length === 1 ? "" : "s"}. Start with the drivers that can release cash fastest.`;
    }
    const planLead = result.category === "Low"
      ? `Maintain the current EMI, preserve the ${result.cashRunway.toFixed(1)}-month runway and review the score monthly.`
      : `1) ${restructuringTiming.label}: discuss the ${recommendedPlan.name} EMI of about ${money(recommendedPlan.emi)}. 2) Target ${savingsPlan.daysToRecover || 10} fewer receivable days. 3) Build the ${money(savingsPlan.reserveGap)} reserve gap in stages.`;
    return `${planLead} The current score is ${result.category.toLowerCase()} at ${percent(result.probability)}.${caution}`;
  };

  const sendChat = (prompt: string) => {
    const question = prompt.trim();
    if (!question) return;
    const userId = nextMessageId.current++;
    const assistantId = nextMessageId.current++;
    setChatMessages((current) => [
      ...current,
      { id: userId, role: "user", text: question },
      { id: assistantId, role: "assistant", text: coachResponse(question) },
    ]);
    setChatInput("");
    setChatOpen(true);
  };

  const scoreDegrees = `${result.score * 3.6}deg`;
  const categoryClass = result.category.toLowerCase();
  const maxDriverImpact = Math.max(...result.drivers.map((driver) => Math.abs(driver.impact)), 0.1);

  return (
    <main className="risk-app">
      <aside className={`risk-sidebar ${mobileNav ? "mobile-open" : ""}`} id="primary-sidebar">
        <button className="brand-mark" onClick={() => scrollTo("analysis")} aria-label="RestructAI home"><PiggyBank size={21} /></button>
        <nav aria-label="Primary navigation">
          <button className={`nav-item ${activeSection === "analysis" ? "active" : ""}`} onClick={() => scrollTo("analysis")} data-tooltip="Risk analysis" aria-label="Risk analysis" aria-current={activeSection === "analysis" ? "page" : undefined}><LayoutDashboard size={19} /></button>
          <button className={`nav-item ${activeSection === "restructure" ? "active" : ""}`} onClick={() => scrollTo("restructure")} data-tooltip="Restructuring studio" aria-label="Restructuring studio" aria-current={activeSection === "restructure" ? "page" : undefined}><Landmark size={19} /></button>
          <button className={`nav-item ${activeSection === "drivers" ? "active" : ""}`} onClick={() => scrollTo("drivers")} data-tooltip="Risk drivers" aria-label="Risk drivers" aria-current={activeSection === "drivers" ? "page" : undefined}><Activity size={19} /></button>
          <button className={`nav-item ${activeSection === "model-evidence" ? "active" : ""}`} onClick={() => scrollTo("model-evidence")} data-tooltip="Model evidence" aria-label="Model evidence" aria-current={activeSection === "model-evidence" ? "page" : undefined}><BarChart3 size={19} /></button>
          <button className={`nav-item ${activeSection === "methodology" ? "active" : ""}`} onClick={() => scrollTo("methodology")} data-tooltip="Methodology" aria-label="Methodology" aria-current={activeSection === "methodology" ? "page" : undefined}><Database size={19} /></button>
          <button className="nav-item" onClick={() => setChatOpen(true)} data-tooltip="Savings Coach" aria-label="Open Savings Coach"><Bot size={19} /></button>
        </nav>
        <button className="avatar" aria-label="Analyst account">SK</button>
      </aside>

      <section className="risk-workspace">
        <header className="risk-topbar">
          <button className="mobile-menu" onClick={() => setMobileNav(!mobileNav)} aria-label="Toggle navigation" aria-controls="primary-sidebar" aria-expanded={mobileNav}>{mobileNav ? <X size={20} /> : <Menu size={20} />}</button>
          <div className="risk-wordmark"><PiggyBank size={17} /> RESTRUCT<span>AI</span></div>
          <div className="top-actions">
            <span className="secure"><PiggyBank size={13} /> Savings mode active</span>
            <span className="model-chip"><FlaskConical size={13} /> Synthetic data • v1.0</span>
          </div>
        </header>

        <div className="risk-content" id="analysis">
          <div className="risk-hero">
            <div>
              <p className="eyebrow">SAVE CASH • PROTECT CREDIT • GROW AGAIN</p>
              <h1>Protect cash today. Preserve credit tomorrow.</h1>
              <p>Spot repayment pressure early, understand what is driving it and turn the result into practical money-saving actions for the business.</p>
            </div>
            <div className="decision-badge"><PiggyBank size={22} /><span><b>Savings-first guidance</b><small>Protect runway before default</small></span></div>
          </div>

          <div className="scenario-strip" aria-label="Borrower scenario presets">
            <span>Try a scenario</span>
            {Object.keys(scenarioPresets).map((name) => (
              <button key={name} className={activePreset === name ? "selected" : ""} onClick={() => choosePreset(name)}>
                {name === "Stable" ? <TrendingUp size={13} /> : name === "Critical" ? <AlertTriangle size={13} /> : <TrendingDown size={13} />}
                {name}
              </button>
            ))}
            {activePreset === "Custom" && <span className="custom-pill">Custom inputs</span>}
          </div>

          <section className="analysis-grid">
            <form className="borrower-card" onSubmit={(event) => { event.preventDefault(); runAnalysis(); }}>
              <div className="card-heading">
                <div><p className="eyebrow">BORROWER INPUT</p><h2>Financial snapshot</h2></div>
                <span><CheckCircle2 size={13} /> 26 signals</span>
              </div>
              <div className="input-tabs" role="tablist" aria-label="Input groups">
                <button type="button" id="tab-financial" aria-controls="borrower-input-panel" aria-selected={tab === "financial"} className={tab === "financial" ? "active" : ""} onClick={() => setTab("financial")} role="tab">Cash flow</button>
                <button type="button" id="tab-debt" aria-controls="borrower-input-panel" aria-selected={tab === "debt"} className={tab === "debt" ? "active" : ""} onClick={() => setTab("debt")} role="tab">Debt</button>
                <button type="button" id="tab-conduct" aria-controls="borrower-input-panel" aria-selected={tab === "conduct"} className={tab === "conduct" ? "active" : ""} onClick={() => setTab("conduct")} role="tab">Conduct</button>
                <button type="button" id="tab-business" aria-controls="borrower-input-panel" aria-selected={tab === "business"} className={tab === "business" ? "active" : ""} onClick={() => setTab("business")} role="tab">Business</button>
              </div>

              <div className="risk-form-grid" id="borrower-input-panel" role="tabpanel" aria-labelledby={`tab-${tab}`}>
                {tab === "financial" && <>
                  <NumberField label="Monthly revenue" value={draft.monthly_revenue} prefix="₹" min={1} max={10_000_000_000} onChange={(value) => updateNumber("monthly_revenue", value)} />
                  <NumberField label="Operating expenses" value={draft.monthly_operating_expenses} prefix="₹" min={1} max={10_000_000_000} onChange={(value) => updateNumber("monthly_operating_expenses", value)} />
                  <NumberField label="Free cash flow" value={draft.free_cash_flow} prefix="₹" min={-10_000_000_000} max={10_000_000_000} onChange={(value) => updateNumber("free_cash_flow", value)} />
                  <NumberField label="Average bank balance" value={draft.average_bank_balance} prefix="₹" min={0} max={100_000_000_000} onChange={(value) => updateNumber("average_bank_balance", value)} />
                  <NumberField label="3-month revenue growth" value={draft.revenue_growth_3m * 100} suffix="%" step="0.5" min={-100} max={300} onChange={(value) => updatePercent("revenue_growth_3m", value)} />
                  <NumberField label="Cash-flow volatility" value={draft.cash_flow_volatility * 100} suffix="%" step="1" min={0} max={100} onChange={(value) => updatePercent("cash_flow_volatility", value)} />
                </>}
                {tab === "debt" && <>
                  <NumberField label="Outstanding loan" value={draft.outstanding_loan_amount} prefix="₹" min={1} max={1_000_000_000_000} onChange={(value) => updateNumber("outstanding_loan_amount", value)} />
                  <NumberField label="Current monthly EMI" value={draft.current_emi} prefix="₹" min={1} max={10_000_000_000} onChange={(value) => updateNumber("current_emi", value)} />
                  <NumberField label="Total existing debt" value={draft.total_existing_debt} prefix="₹" min={1} max={1_000_000_000_000} onChange={(value) => updateNumber("total_existing_debt", value)} />
                  <NumberField label="Interest rate" value={draft.interest_rate} suffix="%" step="0.1" min={0} max={40} onChange={(value) => updateNumber("interest_rate", value)} />
                  <NumberField label="Remaining tenure" value={draft.remaining_tenure_months} suffix="months" min={1} max={360} onChange={(value) => updateNumber("remaining_tenure_months", value)} />
                  <NumberField label="Active loans" value={draft.number_of_active_loans} min={1} max={20} onChange={(value) => updateNumber("number_of_active_loans", value)} />
                </>}
                {tab === "conduct" && <>
                  <NumberField label="Receivable days" value={draft.receivable_days} suffix="days" min={0} max={365} onChange={(value) => updateNumber("receivable_days", value)} />
                  <NumberField label="Change in receivable days" value={draft.receivable_days_change} suffix="days" min={-365} max={365} onChange={(value) => updateNumber("receivable_days_change", value)} />
                  <NumberField label="Overdue receivables" value={draft.overdue_receivables_ratio * 100} suffix="%" min={0} max={100} onChange={(value) => updatePercent("overdue_receivables_ratio", value)} />
                  <NumberField label="Delayed EMIs (3m)" value={draft.delayed_emi_count_3m} min={0} max={3} onChange={(value) => updateNumber("delayed_emi_count_3m", value)} />
                  <NumberField label="Delayed EMIs (6m)" value={draft.delayed_emi_count_6m} min={0} max={6} onChange={(value) => updateNumber("delayed_emi_count_6m", value)} />
                  <NumberField label="Missed EMIs" value={draft.missed_emi_count} min={0} max={6} onChange={(value) => updateNumber("missed_emi_count", value)} />
                  <NumberField label="Average payment delay" value={draft.average_payment_delay_days} suffix="days" min={0} max={365} onChange={(value) => updateNumber("average_payment_delay_days", value)} />
                </>}
                {tab === "business" && <>
                  <SelectField label="Industry" value={draft.industry} options={["Manufacturing", "Textile", "Retail", "Food Processing", "Logistics", "Auto Components", "Services", "Construction", "Electronics", "Chemicals"]} onChange={(value) => updateText("industry", value)} />
                  <SelectField label="State" value={draft.state} options={["Maharashtra", "Gujarat", "Tamil Nadu", "Karnataka", "Delhi", "Rajasthan", "Telangana", "Uttar Pradesh"]} onChange={(value) => updateText("state", value)} />
                  <SelectField label="Business type" value={draft.business_type} options={["Proprietorship", "Partnership", "Private Limited", "LLP"]} onChange={(value) => updateText("business_type", value)} />
                  <NumberField label="Business age" value={draft.business_age_years} suffix="years" step="0.1" min={0.1} max={100} onChange={(value) => updateNumber("business_age_years", value)} />
                  <NumberField label="Employees" value={draft.employee_count} min={1} max={1_000_000} onChange={(value) => updateNumber("employee_count", value)} />
                  <NumberField label="GST turnover growth" value={draft.gst_turnover_growth * 100} suffix="%" min={-100} max={300} onChange={(value) => updatePercent("gst_turnover_growth", value)} />
                  <NumberField label="GST-to-bank variance" value={draft.gst_vs_bank_credit_difference * 100} suffix="%" min={0} max={100} onChange={(value) => updatePercent("gst_vs_bank_credit_difference", value)} />
                </>}
              </div>

              {validationIssues.length > 0 && <div className="validation-panel" role="alert"><AlertTriangle size={14} /><span><b>Check borrower inputs</b>{validationIssues.slice(0, 2).join(" ")}{validationIssues.length > 2 ? ` +${validationIssues.length - 2} more.` : ""}</span></div>}
              <div className="form-note"><Info size={13} /><span>Derived ratios—including DSCR, EMI burden and liquidity stress—are calculated automatically.</span></div>
              <button className="run-button" type="submit" disabled={validationIssues.length > 0}><Sparkles size={16} /> Run 90-day risk analysis <ArrowRight size={16} /></button>
            </form>

            <section className={`risk-result-card ${categoryClass}`} aria-live="polite">
              <div className="result-topline"><span>MODEL RESULT</span><span className={`risk-pill ${categoryClass}`}>{result.category} risk</span></div>
              <div className="score-layout">
                <div className="score-ring" style={{ "--score-angle": scoreDegrees } as CSSProperties}>
                  <div><strong>{Math.round(result.score)}</strong><span>/100</span></div>
                </div>
                <div className="probability-copy">
                  <span>Predicted 90-day stress probability</span>
                  <strong>{percent(result.probability)}</strong>
                  <p>{result.classification ? "Above" : "Below"} the operational alert threshold of {percent(result.threshold)}.</p>
                </div>
              </div>

              <div className="risk-ratio-grid">
                <div><span>DSCR</span><strong className={result.dscr < 1 ? "negative" : "positive"}>{result.dscr.toFixed(2)}x</strong><small>{result.dscr < 1 ? "Unsustainable" : "Covered"}</small></div>
                <div><span>EMI / free cash flow</span><strong className={result.emiBurden > 0.8 ? "negative" : "positive"}>{result.emiBurden.toFixed(2)}x</strong><small>{result.emiBurden > 0.8 ? "High burden" : "Manageable"}</small></div>
                <div><span>Cash runway</span><strong className={result.cashRunway < 2 ? "negative" : "positive"}>{result.cashRunway.toFixed(1)} mo</strong><small>{result.cashRunway < 2 ? "Thin buffer" : "Adequate"}</small></div>
              </div>

              <div className="savings-preview">
                <div><WalletCards size={15} /><span>EMI discussion anchor<b>{money(savingsPlan.emiDiscussionTarget)} / mo</b></span></div>
                <div><Coins size={15} /><span>Potential monthly relief<b>{money(savingsPlan.monthlyRelief)}</b></span></div>
              </div>
              <button className="plan-jump" type="button" onClick={() => scrollTo("restructure")}><Calculator size={15} /> Compare realistic restructuring plans <ArrowRight size={14} /></button>

              <div className="warning-panel">
                <div className="warning-heading"><AlertTriangle size={15} /><span>Early-warning signals</span><b>{result.warnings.length}</b></div>
                {result.warnings.length ? result.warnings.map((warning) => <p key={warning}><span>!</span>{warning}</p>) : <p className="all-clear"><Check size={13} /> No rule-assisted warnings detected.</p>}
              </div>
              <a className="download-button" href={downloadHref} download="restructai-risk-analysis.txt" onClick={confirmDownload}><Download size={15} /> Download risk analysis</a>
              <small className="result-disclaimer">Calibrated probability • Synthetic-data methodology demo</small>
            </section>
          </section>

          <section className="restructure-studio section-anchor" id="restructure">
            <div className="section-heading restructure-heading">
              <div><p className="eyebrow">INTERACTIVE EMI RESTRUCTURING</p><h2>See how—and when—to change the schedule</h2><p>Compare cash-flow-based repayment structures, adjust the terms and see affordability, lifetime cost and the model&apos;s EMI-only what-if response update together.</p></div>
              <span className="explain-badge"><Calculator size={13} /> Amortisation + live model simulation</span>
            </div>

            <div className={`timing-banner ${restructuringTiming.tone}`}>
              <div className="timing-icon"><CalendarClock size={22} /></div>
              <div><span>WHEN TO RESTRUCTURE</span><h3>{restructuringTiming.label}</h3><p>{restructuringTiming.summary}</p></div>
              <div className="timing-trigger"><b>Trigger</b><span>{restructuringTiming.trigger}</span></div>
            </div>

            <div className="plan-options" role="group" aria-label="Restructuring plan options">
              {planOptions.map((plan) => (
                <button key={plan.id} className={`plan-option ${selectedBasePlan.id === plan.id ? "selected" : ""}`} onClick={() => choosePlan(plan.id)} aria-pressed={selectedBasePlan.id === plan.id}>
                  <span className="plan-option-top"><i className={plan.feasibility.toLowerCase().replace(" ", "-")}>{plan.feasibility}</i>{restructuringNeeded && plan.id === recommendedPlan.id && <b>Recommended</b>}</span>
                  <strong>{plan.name}</strong>
                  <small>{plan.description}</small>
                  <span className="plan-emi">{money(plan.emi)}<i>/ month</i></span>
                  <span className="plan-terms">{plan.annualRate.toFixed(1)}% • {plan.totalTenureMonths} months • {plan.moratoriumMonths ? `${plan.moratoriumMonths}-month pause` : "no pause"}</span>
                </button>
              ))}
            </div>

            <div className="plan-workbench">
              <div className="plan-controls">
                <div className="workbench-title"><div><p className="eyebrow">LIVE PLAN BUILDER</p><h3>{selectedPlan.name}</h3></div><button onClick={resetToRecommendedPlan}><RefreshCw size={13} /> Reset best fit</button></div>
                <p className="plan-assumption">{selectedPlan.assumption}</p>

                <label className="plan-slider"><span><b>Interest rate</b><output>{selectedPlan.annualRate.toFixed(1)}%</output></span><input aria-label="Plan interest rate" type="range" min={Math.max(0, borrower.interest_rate - 3)} max={Math.min(40, borrower.interest_rate + 2)} step="0.1" value={selectedPlan.annualRate} onChange={(event) => updatePlanTerms({ annualRate: Number(event.target.value) })} /><small>Lender pricing decision; lower rates are not guaranteed.</small></label>
                <label className="plan-slider"><span><b>Total revised tenure</b><output>{selectedPlan.totalTenureMonths} months</output></span><input aria-label="Plan total tenure" type="range" min={Math.min(borrower.remaining_tenure_months, 120)} max="120" step="1" value={selectedPlan.totalTenureMonths} onChange={(event) => updatePlanTerms({ totalTenureMonths: Number(event.target.value) })} /><small>Longer tenure reduces EMI but normally raises total interest.</small></label>
                <label className="plan-select"><span><b>Payment moratorium</b><small>Interest is capitalised during the pause.</small></span><select aria-label="Plan moratorium" value={selectedPlan.moratoriumMonths} onChange={(event) => updatePlanTerms({ moratoriumMonths: Number(event.target.value) })}><option value="0">No pause</option><option value="3">3 months</option><option value="6">6 months</option></select></label>
              </div>

              <div className="plan-outcome" aria-live="polite">
                <div className="outcome-head"><div><p className="eyebrow">SELECTED PLAN OUTCOME</p><h3>{money(selectedPlan.emi)} <span>/ month</span></h3></div><span className={`feasibility ${selectedPlan.feasibility.toLowerCase().replace(" ", "-")}`}>{selectedPlan.feasibility}</span></div>
                <div className="outcome-grid">
                  <div><span>Monthly relief</span><b className={selectedPlan.monthlyRelief >= 0 ? "good" : "bad"}>{selectedPlan.monthlyRelief >= 0 ? money(selectedPlan.monthlyRelief) : `+${money(Math.abs(selectedPlan.monthlyRelief))}`}</b><small>vs current {money(borrower.current_emi)}</small></div>
                  <div><span>Projected DSCR</span><b className={selectedPlan.projectedDscr >= 1.2 ? "good" : "bad"}>{selectedPlan.projectedDscr.toFixed(2)}x</b><small>1.20x planning target</small></div>
                  <div><span>Estimated total interest</span><b>{money(selectedPlan.totalInterest)}</b><small>over revised schedule</small></div>
                  <div><span>Change in remaining cash outflow</span><b className={selectedPlan.changeVsCurrentSchedule <= 0 ? "good" : "cost"}>{selectedPlan.changeVsCurrentSchedule >= 0 ? "+" : "−"}{money(Math.abs(selectedPlan.changeVsCurrentSchedule))}</b><small>vs current EMI × tenure</small></div>
                </div>
                <div className="model-whatif">
                  <div className="model-whatif-icon"><SlidersHorizontal size={17} /></div>
                  <div><span>MODEL WHAT-IF • EMI CHANGED, OTHER INPUTS HELD CONSTANT</span><p><b>{percent(result.probability)}</b><ArrowRight size={14} /><strong>{percent(projectedResult.probability)}</strong></p><small>{modelProbabilityImprovement > 0 ? `${percentagePoints(modelProbabilityImprovement)} lower projected stress probability` : "No modelled probability improvement from these terms"}. This is a scenario simulation, not a promised outcome.</small></div>
                </div>
                {selectedPlan.feasibility === "Not feasible" && <div className="not-feasible-note"><AlertTriangle size={15} /><span><b>EMI-only restructuring is insufficient.</b> The proposed EMI still exceeds current free cash flow. The lender should assess business viability, working-capital support and deeper corrective action.</span></div>}
              </div>
            </div>

            <div className="restructure-timeline">
              <div className="timeline-title"><CalendarClock size={18} /><div><b>Action timeline</b><span>What to do before and after the lender conversation</span></div></div>
              <div className="timeline-steps">{restructuringTiming.steps.map((step, index) => <div key={`${step.when}-${step.title}`}><i>{index + 1}</i><span>{step.when}</span><b>{step.title}</b><p>{step.detail}</p></div>)}</div>
            </div>
            <div className="lender-approval-note"><Landmark size={16} /><span><b>Do not change payments on your own.</b> These are cash-flow scenarios, not sanctioned terms. Continue the contractual EMI until the lender issues a written revised schedule; restructuring can affect classification, pricing and credit history.</span></div>
          </section>

          <section className="drivers-section section-anchor" id="drivers">
            <div className="section-heading">
              <div><p className="eyebrow">LOCAL EXPLAINABILITY</p><h2>What moved this borrower&apos;s score</h2><p>Signed standardized log-odds contributions from the selected model—not invented percentage impacts.</p></div>
              <span className="explain-badge"><Sparkles size={13} /> Exact model contributions</span>
            </div>
            <div className="drivers-grid">
              <div className="driver-list">
                {result.drivers.map((driver, index) => (
                  <div className="driver-row" key={`${driver.feature}-${index}`}>
                    <span className={`driver-icon ${driver.direction}`}>{driver.direction === "risk" ? <TrendingUp size={13} /> : <TrendingDown size={13} />}</span>
                    <div><span><b>{driver.feature}</b><small>{driver.direction === "risk" ? "Increases risk" : "Reduces risk"}</small></span><div className="driver-track"><i className={driver.direction} style={{ width: `${Math.max(8, Math.abs(driver.impact) / maxDriverImpact * 100)}%` }}></i></div></div>
                    <strong className={driver.direction}>{driver.impact > 0 ? "+" : ""}{driver.impact.toFixed(2)}</strong>
                  </div>
                ))}
              </div>
              <aside className="explanation-card">
                <div className="icon-tile"><CircleGauge size={20} /></div>
                <p className="eyebrow">MODEL INTERPRETATION</p>
                <h3>{result.category === "Low" ? "Current signals indicate resilience." : "Liquidity and repayment pressure need attention."}</h3>
                <p>{result.category === "Low" ? "Cash coverage and conduct signals keep the estimated probability below the early-warning threshold." : `The ${result.category.toLowerCase()} score is driven by the combined pattern across cash coverage, revenue momentum, receivables and repayment conduct.`}</p>
                <div><ShieldCheck size={15} /><span>Rules explain operational alerts; they do not replace the ML estimate.</span></div>
              </aside>
            </div>
          </section>

          <section className="model-evidence section-anchor" id="model-evidence">
            <div className="section-heading">
              <div><p className="eyebrow">ACTUAL TEST RESULTS</p><h2>Selected on evidence, not assumption</h2><p>All metrics below were produced by the validated 12,000-borrower synthetic experiment.</p></div>
              <span className="winner-chip"><CheckCircle2 size={14} /> {MODEL_NAME} selected</span>
            </div>
            <div className="metric-cards">
              <MetricCard label="ROC–AUC" value={MODEL_METRICS.ROC_AUC} note="Discrimination" />
              <MetricCard label="PR–AUC" value={MODEL_METRICS.PR_AUC} note="Imbalanced performance" />
              <MetricCard label="Stress recall" value={MODEL_METRICS.Recall} note="At 0.36 threshold" />
              <MetricCard label="Brier score" value={MODEL_METRICS.Brier_Score} note="Lower is better" inverse />
            </div>

            <div className="benchmark-card">
              <div className="benchmark-head"><span>Formal benchmark</span><span>Uncalibrated 0.50 classification threshold for consistent comparison</span></div>
              <div className="benchmark-table" role="table" aria-label="Model benchmark metrics">
                <div className="benchmark-row header" role="row"><span>Model</span><span>ROC–AUC</span><span>PR–AUC</span><span>Recall</span><span>F1</span><span>Brier</span></div>
                {MODEL_BENCHMARK.map((model) => <div className={`benchmark-row ${model.Model === MODEL_NAME ? "winner" : ""}`} role="row" key={model.Model}><span>{model.Model}{model.Model === MODEL_NAME && <b>Selected</b>}</span><span>{model.ROC_AUC.toFixed(4)}</span><span>{model.PR_AUC.toFixed(4)}</span><span>{model.Recall.toFixed(4)}</span><span>{model.F1.toFixed(4)}</span><span>{model.Brier_Score.toFixed(4)}</span></div>)}
              </div>
            </div>
            <div className="model-clarity"><CheckCircle2 size={18} /><div><b>Verified: the models are not returning the same figures.</b><p>Logistic Regression and CatBoost looked identical only because ROC–AUC was rounded to three decimals. Their full values are {LOGISTIC_BENCHMARK?.ROC_AUC.toFixed(4)} and {CATBOOST_BENCHMARK?.ROC_AUC.toFixed(4)}. The live borrower score uses the selected {MODEL_NAME} pipeline only.</p></div></div>
          </section>

          <section className="methodology-section section-anchor" id="methodology">
            <div className="method-copy">
              <p className="eyebrow">METHODOLOGY</p>
              <h2>Built for an auditable competition story.</h2>
              <p>One reusable pipeline carries the borrower from raw inputs through financial features, preprocessing, calibrated probability, threshold classification and human-readable alerts.</p>
              <div className="method-facts"><span><Database size={15} /><b>12,000</b><small>synthetic MSMEs</small></span><span><Activity size={15} /><b>{percent(PORTFOLIO_PREVALENCE)}</b><small>stress prevalence</small></span><span><RefreshCw size={15} /><b>5-fold</b><small>stratified CV</small></span></div>
            </div>
            <div className="pipeline-flow">
              {["MSME inputs", "Financial features", "Three-model benchmark", "Sigmoid calibration", "Risk score + warnings"].map((step, index) => <div key={step}><span>{index + 1}</span><p>{step}</p>{index < 4 && <ChevronRight size={15} />}</div>)}
            </div>
          </section>

          <footer>
            <div className="footer-brand"><div className="brand-mark small"><PiggyBank size={16} /></div><span>RESTRUCTAI</span></div>
            <p>Results use synthetic data to demonstrate methodology and are not validated real-world credit-risk performance.</p>
            <span>Explainable • Calibrated • Human-reviewed</span>
          </footer>
        </div>
      </section>
      <button className={`chat-launcher ${chatOpen ? "open" : ""}`} onClick={() => setChatOpen(!chatOpen)} aria-label={chatOpen ? "Close Savings Coach" : "Open Savings Coach"} aria-expanded={chatOpen} aria-controls="savings-coach"><MessageCircle size={18} /><span>Savings Coach</span></button>
      {chatOpen && <aside className="coach-panel" id="savings-coach" aria-label="Savings Coach">
        <div className="coach-head"><div className="coach-avatar"><PiggyBank size={19} /></div><div><b>Savings Coach</b><span>Borrower-aware guidance</span></div><button onClick={() => setChatOpen(false)} aria-label="Close Savings Coach"><X size={17} /></button></div>
        <div className="coach-context"><span className={categoryClass}>{result.category} risk</span><b>{percent(result.probability)}</b><small>Selected: {selectedPlan.name}</small></div>
        <div className="coach-messages" role="log" aria-live="polite" ref={chatLogRef}>
          {chatMessages.map((message) => <div className={`coach-message ${message.role}`} key={message.id}>{message.text}</div>)}
        </div>
        <div className="coach-prompts">{QUICK_PROMPTS.map((prompt) => <button key={prompt} onClick={() => sendChat(prompt)}>{prompt}</button>)}</div>
        <form className="coach-form" onSubmit={(event) => { event.preventDefault(); sendChat(chatInput); }}><input value={chatInput} onChange={(event) => setChatInput(event.target.value)} placeholder="Ask about EMI, cash or collections" aria-label="Message Savings Coach" /><button type="submit" aria-label="Send message" disabled={!chatInput.trim()}><Send size={16} /></button></form>
        <p className="coach-note"><ShieldCheck size={12} /> Illustrative guidance. Confirm terms with the lender.</p>
      </aside>}
      {toast && <div className="toast" role="status"><CheckCircle2 size={16} />{toast}</div>}
    </main>
  );
}

function NumberField({ label, value, onChange, prefix, suffix, step = "1", min, max }: { label: string; value: number; onChange: (value: string) => void; prefix?: string; suffix?: string; step?: string; min?: number; max?: number }) {
  return <label className="risk-field"><span>{label}</span><div>{prefix && <i>{prefix}</i>}<input aria-label={label} type="number" step={step} min={min} max={max} required value={Number.isFinite(value) ? value : ""} onChange={(event) => onChange(event.target.value)} />{suffix && <b>{suffix}</b>}</div></label>;
}

function SelectField({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return <label className="risk-field"><span>{label}</span><div><select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option key={option}>{option}</option>)}</select></div></label>;
}

function MetricCard({ label, value, note, inverse = false }: { label: string; value: number; note: string; inverse?: boolean }) {
  return <div className="metric-card"><div><span>{label}</span><Info size={12} /></div><strong>{value.toFixed(4)}</strong><p>{note}</p><i className={inverse ? "inverse" : ""} style={{ width: `${Math.min((inverse ? 1 - value : value) * 100, 100)}%` }}></i></div>;
}
