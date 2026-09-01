"use client";

import { type CSSProperties, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Database,
  Download,
  FlaskConical,
  Info,
  LayoutDashboard,
  LockKeyhole,
  Menu,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
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

type InputTab = "financial" | "debt" | "conduct" | "business";

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

export default function Home() {
  const [draft, setDraft] = useState<BorrowerInput>(stressedBorrower);
  const [borrower, setBorrower] = useState<BorrowerInput>(stressedBorrower);
  const [tab, setTab] = useState<InputTab>("financial");
  const [activePreset, setActivePreset] = useState("Stressed");
  const [activeSection, setActiveSection] = useState("analysis");
  const [mobileNav, setMobileNav] = useState(false);
  const [toast, setToast] = useState("");
  const result = useMemo(() => scoreBorrower(borrower), [borrower]);
  const validationIssues = useMemo(() => validateBorrower(draft), [draft]);
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
      "Decision support only. Synthetic-data methodology demonstration; not validated for automatic credit decisions.",
    ];
    return `data:text/plain;charset=utf-8,${encodeURIComponent(lines.join("\n"))}`;
  }, [result]);

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
    setDraft(next);
    setBorrower(next);
    setActivePreset(name);
    setToast(`${name} scenario loaded and rescored.`);
    window.setTimeout(() => setToast(""), 2200);
  };

  const runAnalysis = () => {
    if (validationIssues.length) {
      setToast(validationIssues[0]);
      window.setTimeout(() => setToast(""), 3200);
      return;
    }
    setBorrower(draft);
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

  const scoreDegrees = `${result.score * 3.6}deg`;
  const categoryClass = result.category.toLowerCase();
  const maxDriverImpact = Math.max(...result.drivers.map((driver) => Math.abs(driver.impact)), 0.1);

  return (
    <main className="risk-app">
      <aside className={`risk-sidebar ${mobileNav ? "mobile-open" : ""}`} id="primary-sidebar">
        <button className="brand-mark" onClick={() => scrollTo("analysis")} aria-label="RestructAI home">R</button>
        <nav aria-label="Primary navigation">
          <button className={`nav-item ${activeSection === "analysis" ? "active" : ""}`} onClick={() => scrollTo("analysis")} data-tooltip="Risk analysis" aria-label="Risk analysis" aria-current={activeSection === "analysis" ? "page" : undefined}><LayoutDashboard size={19} /></button>
          <button className={`nav-item ${activeSection === "drivers" ? "active" : ""}`} onClick={() => scrollTo("drivers")} data-tooltip="Risk drivers" aria-label="Risk drivers" aria-current={activeSection === "drivers" ? "page" : undefined}><Activity size={19} /></button>
          <button className={`nav-item ${activeSection === "model-evidence" ? "active" : ""}`} onClick={() => scrollTo("model-evidence")} data-tooltip="Model evidence" aria-label="Model evidence" aria-current={activeSection === "model-evidence" ? "page" : undefined}><BarChart3 size={19} /></button>
          <button className={`nav-item ${activeSection === "methodology" ? "active" : ""}`} onClick={() => scrollTo("methodology")} data-tooltip="Methodology" aria-label="Methodology" aria-current={activeSection === "methodology" ? "page" : undefined}><Database size={19} /></button>
        </nav>
        <button className="avatar" aria-label="Analyst account">SK</button>
      </aside>

      <section className="risk-workspace">
        <header className="risk-topbar">
          <button className="mobile-menu" onClick={() => setMobileNav(!mobileNav)} aria-label="Toggle navigation" aria-controls="primary-sidebar" aria-expanded={mobileNav}>{mobileNav ? <X size={20} /> : <Menu size={20} />}</button>
          <div className="risk-wordmark">RESTRUCT<span>AI</span></div>
          <div className="top-actions">
            <span className="secure"><LockKeyhole size={13} /> Local model demo</span>
            <span className="model-chip"><FlaskConical size={13} /> Synthetic data • v1.0</span>
          </div>
        </header>

        <div className="risk-content" id="analysis">
          <div className="risk-hero">
            <div>
              <p className="eyebrow">90-DAY EARLY-WARNING ENGINE</p>
              <h1>See repayment stress before it becomes default.</h1>
              <p>Model an MSME&apos;s probability of serious financial stress using cash flow, debt burden, repayment conduct and GST activity.</p>
            </div>
            <div className="decision-badge"><ShieldCheck size={18} /><span><b>Decision support</b><small>Not an automatic credit decision</small></span></div>
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

              <div className="warning-panel">
                <div className="warning-heading"><AlertTriangle size={15} /><span>Early-warning signals</span><b>{result.warnings.length}</b></div>
                {result.warnings.length ? result.warnings.map((warning) => <p key={warning}><span>!</span>{warning}</p>) : <p className="all-clear"><Check size={13} /> No rule-assisted warnings detected.</p>}
              </div>
              <a className="download-button" href={downloadHref} download="restructai-risk-analysis.txt" onClick={confirmDownload}><Download size={15} /> Download risk analysis</a>
              <small className="result-disclaimer">Calibrated probability • Synthetic-data methodology demo</small>
            </section>
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
                {MODEL_BENCHMARK.map((model) => <div className={`benchmark-row ${model.Model === MODEL_NAME ? "winner" : ""}`} role="row" key={model.Model}><span>{model.Model}{model.Model === MODEL_NAME && <b>Selected</b>}</span><span>{model.ROC_AUC.toFixed(3)}</span><span>{model.PR_AUC.toFixed(3)}</span><span>{model.Recall.toFixed(3)}</span><span>{model.F1.toFixed(3)}</span><span>{model.Brier_Score.toFixed(3)}</span></div>)}
              </div>
            </div>
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
            <div className="footer-brand"><div className="brand-mark small">R</div><span>RESTRUCTAI</span></div>
            <p>Results use synthetic data to demonstrate methodology and are not validated real-world credit-risk performance.</p>
            <span>Explainable • Calibrated • Human-reviewed</span>
          </footer>
        </div>
      </section>
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
  return <div className="metric-card"><div><span>{label}</span><Info size={12} /></div><strong>{value.toFixed(3)}</strong><p>{note}</p><i className={inverse ? "inverse" : ""} style={{ width: `${Math.min((inverse ? 1 - value : value) * 100, 100)}%` }}></i></div>;
}
