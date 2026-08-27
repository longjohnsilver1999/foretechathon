"use client";

import { useMemo, useState } from "react";
import {
  ArrowRight,
  BarChart3,
  Building2,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Download,
  FileText,
  Gauge,
  IndianRupee,
  LayoutDashboard,
  LockKeyhole,
  Menu,
  PencilLine,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  WalletCards,
  X,
} from "lucide-react";

type BusinessData = {
  revenue: number;
  expenses: number;
  outstanding: number;
  currentEmi: number;
  annualRate: number;
  months: number;
};

type Plan = {
  id: string;
  label: string;
  title: string;
  description: string;
  emi: number;
  tenure: number;
  extraInterest: number;
  fit: number;
  note: string;
};

const initialData: BusinessData = {
  revenue: 725000,
  expenses: 599800,
  outstanding: 1935000,
  currentEmi: 91000,
  annualRate: 11.75,
  months: 24,
};

const months = ["SEP", "OCT", "NOV", "DEC", "JAN", "FEB"];
const seasonality = [0.97, 0.86, 0.82, 0.89, 0.96, 1.03];

function calculateEmi(principal: number, annualRate: number, tenure: number) {
  const monthlyRate = annualRate / 1200;
  if (!monthlyRate) return principal / tenure;
  const factor = (1 + monthlyRate) ** tenure;
  return (principal * monthlyRate * factor) / (factor - 1);
}

function formatMoney(value: number, compact = false) {
  if (compact && Math.abs(value) >= 100000) {
    return `₹${(value / 100000).toFixed(1)}L`;
  }
  return `₹${Math.round(value).toLocaleString("en-IN")}`;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export default function Home() {
  const [data, setData] = useState(initialData);
  const [draft, setDraft] = useState(initialData);
  const [selectedPlan, setSelectedPlan] = useState(0);
  const [downturn, setDownturn] = useState(0);
  const [dataModal, setDataModal] = useState(false);
  const [reviewModal, setReviewModal] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [toast, setToast] = useState("");

  const plans = useMemo<Plan[]>(() => {
    const currentInterest = Math.max(data.currentEmi * data.months - data.outstanding, 0);
    const balancedTenure = data.months + 12;
    const reliefTenure = data.months + 24;
    const balancedEmi = calculateEmi(data.outstanding, data.annualRate, balancedTenure);
    const reliefEmi = calculateEmi(data.outstanding, data.annualRate + 0.25, reliefTenure);
    const starterEmi = balancedEmi * 0.72;
    const stepUpEmi = (balancedEmi * balancedTenure - starterEmi * 6) / (balancedTenure - 6);

    return [
      {
        id: "balanced",
        label: "Balanced relief",
        title: "Reduce EMI, extend by 12 months",
        description: "Keep more cash in the business during your slower season without stretching the loan too far.",
        emi: balancedEmi,
        tenure: balancedTenure,
        extraInterest: Math.max(balancedEmi * balancedTenure - data.outstanding - currentInterest, 0),
        fit: 94,
        note: "Best balance of monthly relief and total borrowing cost",
      },
      {
        id: "max-relief",
        label: "Maximum relief",
        title: "Lower EMI, extend by 24 months",
        description: "Create the largest monthly buffer now, with a longer repayment runway and higher total interest.",
        emi: reliefEmi,
        tenure: reliefTenure,
        extraInterest: Math.max(reliefEmi * reliefTenure - data.outstanding - currentInterest, 0),
        fit: 87,
        note: "Strongest cash-flow protection during volatile months",
      },
      {
        id: "step-up",
        label: "Seasonal step-up",
        title: "Start low, step up after 6 months",
        description: "Pay less through the slow season, then increase repayments when your collections usually recover.",
        emi: starterEmi,
        tenure: balancedTenure,
        extraInterest: Math.max(stepUpEmi * (balancedTenure - 6) + starterEmi * 6 - data.outstanding - currentInterest, 0),
        fit: 82,
        note: `EMI steps up to ${formatMoney(stepUpEmi)} from month 7`,
      },
    ];
  }, [data]);

  const plan = plans[selectedPlan];
  const cashBeforeDebt = data.revenue - data.expenses;
  const currentSurplus = cashBeforeDebt - data.currentEmi;
  const breathingRoom = data.currentEmi - plan.emi;
  const dscr = cashBeforeDebt / data.currentEmi;
  const healthScore = Math.round(clamp(18 + dscr * 30, 35, 91));
  const proposedDscr = cashBeforeDebt / plan.emi;

  const forecast = useMemo(
    () => seasonality.map((factor) => {
      const revenue = data.revenue * factor * (1 - downturn / 100);
      return {
        current: revenue - data.expenses - data.currentEmi,
        proposed: revenue - data.expenses - plan.emi,
      };
    }),
    [data, downturn, plan.emi]
  );

  const showToast = (message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(""), 2600);
  };

  const saveData = () => {
    const sanitized = Object.fromEntries(
      Object.entries(draft).map(([key, value]) => [key, Math.max(Number(value) || 0, 0)])
    ) as unknown as BusinessData;
    setData(sanitized);
    setSelectedPlan(0);
    setDataModal(false);
    showToast("Business data updated. Your plans have been recalculated.");
  };

  const downloadProposal = () => {
    const text = [
      "MORROWAI — EMI RESTRUCTURING PROPOSAL",
      "Generated for Suresh Textiles Pvt Ltd",
      "",
      `Recommended plan: ${plan.title}`,
      `Outstanding principal: ${formatMoney(data.outstanding)}`,
      `Current EMI: ${formatMoney(data.currentEmi)}`,
      `Proposed EMI: ${formatMoney(plan.emi)}`,
      `Proposed tenure: ${plan.tenure} months`,
      `Monthly relief: ${formatMoney(breathingRoom)}`,
      `Estimated additional interest: ${formatMoney(plan.extraInterest)}`,
      `Projected DSCR: ${proposedDscr.toFixed(2)}x`,
      "",
      "Rationale",
      `The proposed structure improves monthly debt-service coverage from ${dscr.toFixed(2)}x to ${proposedDscr.toFixed(2)}x and creates additional working-capital headroom before the forecast seasonal dip.`,
      "",
      "This scenario is indicative and subject to lender approval and verification of financial documents.",
    ].join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "MorrowAI-restructuring-proposal.txt";
    link.click();
    URL.revokeObjectURL(link.href);
    showToast("Lender proposal downloaded.");
  };

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setMobileNav(false);
  };

  return (
    <main className="app-shell">
      <aside className={`sidebar ${mobileNav ? "mobile-open" : ""}`}>
        <button className="brand-mark" onClick={() => scrollTo("overview")} aria-label="MorrowAI home">M</button>
        <nav aria-label="Primary navigation">
          <button className="nav-item active" onClick={() => scrollTo("overview")} data-tooltip="Overview" aria-label="Overview"><LayoutDashboard size={19} /></button>
          <button className="nav-item" onClick={() => scrollTo("forecast")} data-tooltip="Cash-flow scenarios" aria-label="Cash-flow scenarios"><BarChart3 size={19} /></button>
          <button className="nav-item" onClick={() => scrollTo("plans")} data-tooltip="Restructuring plans" aria-label="Restructuring plans"><WalletCards size={19} /></button>
          <button className="nav-item" onClick={() => scrollTo("readiness")} data-tooltip="Lender readiness" aria-label="Lender readiness"><FileText size={19} /></button>
        </nav>
        <button className="avatar" aria-label="Account for Suresh Kumar">SK</button>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileNav(!mobileNav)} aria-label="Toggle navigation"><Menu size={20} /></button>
          <div className="wordmark">MORROW<span>AI</span></div>
          <div className="top-actions">
            <span className="secure"><LockKeyhole size={13} /> Bank-grade secure</span>
            <button><CircleHelp size={15} /> Need help?</button>
          </div>
        </header>

        <div className="content" id="overview">
          <div className="page-heading">
            <div>
              <p className="eyebrow">GOOD MORNING, SURESH</p>
              <h1>Your business can breathe easier.</h1>
              <p className="subtitle">We found a safer repayment path based on your latest cash flow.</p>
            </div>
            <button className="secondary-button" onClick={() => { setDraft(data); setDataModal(true); }}><PencilLine size={14} /> Update business data</button>
          </div>

          <section className="health-card">
            <div className="health-copy">
              <span className="status-pill"><TrendingDown size={12} /> Action recommended</span>
              <p className="label">REPAYMENT HEALTH</p>
              <div className="score-line"><strong>{healthScore}</strong><span>/100</span></div>
              <p>Your current EMI is putting pressure on working capital. Restructuring now could prevent a missed payment during the October dip.</p>
              <div className="health-stats">
                <span><small>Current DSCR</small><b>{dscr.toFixed(2)}x</b></span>
                <span><small>Cash after EMI</small><b>{formatMoney(currentSurplus)}</b></span>
              </div>
            </div>
            <div className="health-chart" aria-label="Six month cash after EMI forecast">
              <div className="chart-top"><span>Monthly cash after EMI</span><strong>{formatMoney(currentSurplus)}</strong></div>
              <div className="mini-chart">
                {forecast.map((item, index) => {
                  const value = item.current;
                  const height = clamp(Math.abs(value) / 1100, 8, 72);
                  return <div className="mini-bar-wrap" key={months[index]}><div className={`mini-bar ${value < 0 ? "negative" : ""}`} style={{ height: `${height}%` }}></div></div>;
                })}
                <div className="safety-rule"><span>SAFETY LINE</span></div>
              </div>
              <div className="month-row">{months.map((month) => <span key={month}>{month}</span>)}</div>
              <p className="chart-insight"><Sparkles size={13} /> AI flags October as your highest-risk month</p>
            </div>
          </section>

          <section id="plans" className="section-block">
            <div className="section-title">
              <div><p className="eyebrow">YOUR BEST MATCH</p><h2>Recommended restructuring plan</h2></div>
              <span className="confidence"><CheckCircle2 size={14} /> {plan.fit}% fit for your business</span>
            </div>

            <section className="plan-card">
              <div className="plan-main">
                <span className="plan-tag">{plan.label}</span>
                <h3>{plan.title}</h3>
                <p>{plan.description}</p>
                <div className="metrics">
                  <div><span>NEW MONTHLY EMI</span><strong>{formatMoney(plan.emi)}</strong><small>{formatMoney(breathingRoom)} less</small></div>
                  <div><span>NEW TENURE</span><strong>{plan.tenure} months</strong><small>+{plan.tenure - data.months} months</small></div>
                  <div><span>EXTRA INTEREST</span><strong>{formatMoney(plan.extraInterest)}</strong><small>over full tenure</small></div>
                </div>
              </div>
              <div className="plan-action">
                <Gauge size={25} />
                <p>Estimated monthly breathing room</p>
                <strong>+ {formatMoney(breathingRoom)}</strong>
                <button onClick={() => setReviewModal(true)}>Review this plan <ArrowRight size={16} /></button>
                <small>No commitment • Takes 2 minutes</small>
              </div>
            </section>

            <div className="alternative-row" aria-label="Alternative restructuring plans">
              {plans.map((item, index) => (
                <button key={item.id} className={`alternative-card ${index === selectedPlan ? "selected" : ""}`} onClick={() => setSelectedPlan(index)}>
                  <span>{index === 0 ? "RECOMMENDED" : "ALTERNATIVE 0" + index}</span>
                  <div><strong>{item.label}</strong><small>{formatMoney(item.emi)}/mo</small></div>
                  <p>{item.note}</p>
                  <ChevronRight size={17} />
                </button>
              ))}
            </div>
          </section>

          <section className="section-block scenario-section" id="forecast">
            <div className="section-title scenario-heading">
              <div><p className="eyebrow">STRESS TEST</p><h2>See how the plan holds up</h2><p>Test a revenue dip and compare your monthly buffer before committing.</p></div>
              <div className="stress-control">
                <label htmlFor="downturn">Revenue downturn <strong>{downturn}%</strong></label>
                <input id="downturn" type="range" min="0" max="25" step="5" value={downturn} onChange={(event) => setDownturn(Number(event.target.value))} />
                <div><span>Normal</span><span>Severe</span></div>
              </div>
            </div>
            <div className="forecast-card">
              <div className="forecast-legend"><span><i className="current-dot"></i> Current EMI</span><span><i className="proposed-dot"></i> With {plan.label.toLowerCase()}</span><b>₹0 safety line</b></div>
              <div className="forecast-grid">
                {forecast.map((item, index) => {
                  const currentHeight = clamp(Math.abs(item.current) / 1400, 5, 80);
                  const proposedHeight = clamp(Math.abs(item.proposed) / 1400, 5, 80);
                  return (
                    <div className="forecast-month" key={months[index]}>
                      <div className="bar-space">
                        <div className={`forecast-bar current-bar ${item.current < 0 ? "below" : ""}`} style={{ height: `${currentHeight}%` }}><span>{formatMoney(item.current, true)}</span></div>
                        <div className={`forecast-bar proposed-bar ${item.proposed < 0 ? "below" : ""}`} style={{ height: `${proposedHeight}%` }}><span>{formatMoney(item.proposed, true)}</span></div>
                      </div>
                      <small>{months[index]}</small>
                    </div>
                  );
                })}
              </div>
              <div className={`stress-result ${forecast.some(item => item.proposed < 0) ? "warning" : "safe"}`}>
                <ShieldCheck size={21} />
                <div><strong>{forecast.some(item => item.proposed < 0) ? "This scenario needs more relief" : "Your selected plan stays above water"}</strong><span>{forecast.some(item => item.proposed < 0) ? "Try the maximum relief option or reduce the downturn assumption." : `Even with a ${downturn}% downturn, the model protects more working capital in every forecast month.`}</span></div>
              </div>
            </div>
          </section>

          <section className="insight-grid section-block" id="readiness">
            <div className="ai-insight-card">
              <div className="icon-tile"><Sparkles size={21} /></div>
              <p className="eyebrow">WHY THIS PLAN</p>
              <h2>Built around your business rhythm, not just your loan.</h2>
              <p>MorrowAI weighed cash-flow stability, seasonal collections, debt coverage and total borrowing cost across 48 scenarios.</p>
              <ul>
                <li><Check size={15} /> Protects the October–December low season</li>
                <li><Check size={15} /> Raises projected DSCR to {proposedDscr.toFixed(2)}x</li>
                <li><Check size={15} /> Keeps added interest below the maximum-relief option</li>
              </ul>
            </div>
            <div className="readiness-card">
              <div className="readiness-top"><div><p className="eyebrow">LENDER READINESS</p><h3>Your proposal is nearly ready</h3></div><div className="progress-ring">75%</div></div>
              <div className="check-list">
                <div><span className="check done"><Check size={13} /></span><p><strong>Cash-flow analysis</strong><small>12 months reviewed</small></p></div>
                <div><span className="check done"><Check size={13} /></span><p><strong>Loan repayment history</strong><small>No missed EMIs detected</small></p></div>
                <div><span className="check done"><Check size={13} /></span><p><strong>Recommended structure</strong><small>{plan.label} selected</small></p></div>
                <button onClick={() => setReviewModal(true)}><span className="check"><FileText size={13} /></span><p><strong>Business documents</strong><small>Add GST returns and bank statement</small></p><ChevronRight size={16} /></button>
              </div>
              <button className="outline-action" onClick={() => setReviewModal(true)}>Open lender packet <ArrowRight size={15} /></button>
            </div>
          </section>

          <footer>
            <div className="footer-brand"><div className="brand-mark small">M</div><span>MORROWAI</span></div>
            <p>Indicative scenarios only. Final terms are subject to lender approval and document verification.</p>
            <span>Encrypted • Consent-led • RBI-aware</span>
          </footer>
        </div>
      </section>

      {dataModal && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setDataModal(false); }}>
          <section className="modal data-modal" role="dialog" aria-modal="true" aria-labelledby="data-title">
            <button className="modal-close" onClick={() => setDataModal(false)} aria-label="Close"><X size={19} /></button>
            <div className="modal-icon"><Building2 size={20} /></div>
            <p className="eyebrow">BUSINESS SNAPSHOT</p>
            <h2 id="data-title">Update your numbers</h2>
            <p className="modal-intro">We use these figures only to model repayment scenarios. Your recommendations update instantly.</p>
            <div className="form-grid">
              <label><span>Average monthly revenue</span><div><IndianRupee size={14} /><input type="number" value={draft.revenue} onChange={e => setDraft({ ...draft, revenue: Number(e.target.value) })} /></div></label>
              <label><span>Monthly operating expenses</span><div><IndianRupee size={14} /><input type="number" value={draft.expenses} onChange={e => setDraft({ ...draft, expenses: Number(e.target.value) })} /></div></label>
              <label><span>Outstanding loan principal</span><div><IndianRupee size={14} /><input type="number" value={draft.outstanding} onChange={e => setDraft({ ...draft, outstanding: Number(e.target.value) })} /></div></label>
              <label><span>Current monthly EMI</span><div><IndianRupee size={14} /><input type="number" value={draft.currentEmi} onChange={e => setDraft({ ...draft, currentEmi: Number(e.target.value) })} /></div></label>
              <label><span>Annual interest rate</span><div><input type="number" step="0.05" value={draft.annualRate} onChange={e => setDraft({ ...draft, annualRate: Number(e.target.value) })} /><b>%</b></div></label>
              <label><span>Remaining tenure</span><div><input type="number" value={draft.months} onChange={e => setDraft({ ...draft, months: Number(e.target.value) })} /><b>months</b></div></label>
            </div>
            <div className="modal-actions"><button className="ghost-button" onClick={() => setDataModal(false)}>Cancel</button><button className="primary-button" onClick={saveData}>Recalculate my plans <ArrowRight size={15} /></button></div>
          </section>
        </div>
      )}

      {reviewModal && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setReviewModal(false); }}>
          <section className="modal review-modal" role="dialog" aria-modal="true" aria-labelledby="review-title">
            <button className="modal-close" onClick={() => setReviewModal(false)} aria-label="Close"><X size={19} /></button>
            <div className="review-header"><span className="plan-tag">{plan.label}</span><h2 id="review-title">Your lender-ready plan</h2><p>A clear, evidence-backed request you can take to your lender.</p></div>
            <div className="review-summary">
              <div><span>Current EMI</span><strong>{formatMoney(data.currentEmi)}</strong></div><ArrowRight size={20} />
              <div className="highlight"><span>Proposed EMI</span><strong>{formatMoney(plan.emi)}</strong></div>
            </div>
            <div className="review-details">
              <div><span>Monthly cash released</span><strong>{formatMoney(breathingRoom)}</strong></div>
              <div><span>New tenure</span><strong>{plan.tenure} months</strong></div>
              <div><span>Projected DSCR</span><strong>{proposedDscr.toFixed(2)}x</strong></div>
              <div><span>Estimated extra interest</span><strong>{formatMoney(plan.extraInterest)}</strong></div>
            </div>
            <div className="ai-note"><Sparkles size={18} /><p><strong>AI rationale</strong><span>This structure creates enough headroom for the seasonal low while avoiding the higher lifetime cost of a 24-month extension.</span></p></div>
            <div className="document-note"><FileText size={18} /><p><strong>Complete your lender packet</strong><span>Attach your latest 6-month bank statement and GST returns before submission.</span></p></div>
            <div className="modal-actions"><button className="ghost-button" onClick={() => setReviewModal(false)}>Back</button><button className="primary-button" onClick={downloadProposal}><Download size={15} /> Download proposal</button></div>
          </section>
        </div>
      )}

      {toast && <div className="toast" role="status"><CheckCircle2 size={17} /> {toast}</div>}
    </main>
  );
}
