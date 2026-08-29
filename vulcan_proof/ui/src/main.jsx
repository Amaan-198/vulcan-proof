import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const EVIDENCE_ORDER = ["weight", "serial", "sealed", "packing", "geotag", "otp", "signature", "ack", "vack"];
const DEMO_ORDER_LIMIT = 5;

const navItems = [
  { id: "order", label: "Order", short: "01" },
  { id: "plan", label: "Plan", short: "02" },
  { id: "package", label: "Package", short: "03" },
  { id: "report", label: "Report", short: "04" },
];

const categoryColors = {
  Electronics: "blue",
  Jewellery: "violet",
  Apparel: "orange",
  Home: "teal",
  FMCG: "slate",
};

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function money(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const amount = Number(value);
  const sign = amount < 0 ? "-" : "";
  return `${sign}₹${Math.abs(amount).toLocaleString("en-IN", { maximumFractionDigits: 2, minimumFractionDigits: 2 })}`;
}

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function pretty(value) {
  return String(value || "—").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function App() {
  const [screen, setScreen] = useState("order");
  const [orders, setOrders] = useState([]);
  const [category, setCategory] = useState("Electronics");
  const [query, setQuery] = useState("");
  const [plansOnly, setPlansOnly] = useState(true);
  const [packageReadyOnly, setPackageReadyOnly] = useState(false);
  const [selectedId, setSelectedId] = useState("");
  const [plan, setPlan] = useState(null);
  const [pkg, setPkg] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    Promise.all([
      getJson(`/orders?category=${encodeURIComponent(category)}&limit=${DEMO_ORDER_LIMIT}&plans_only=${plansOnly}&package_ready_only=${packageReadyOnly}`),
      getJson("/report/kappa"),
      getJson("/demo/script").catch(() => null),
    ])
      .then(async ([orderData, reportData, script]) => {
        if (!active) return;
        setReport(reportData);
        const scriptId = script?.beats?.find((beat) => beat.beat === 1)?.order_id;
        const preferredId = category === "Electronics" ? scriptId : orderData.orders?.[0]?.order_id;
        let visibleOrders = orderData.orders || [];
        if (category === "Electronics" && scriptId && !visibleOrders.some((order) => order.order_id === scriptId)) {
          const featured = await getJson(`/orders?query=${encodeURIComponent(scriptId)}&limit=1&plans_only=${plansOnly}&package_ready_only=${packageReadyOnly}`);
          visibleOrders = [...(featured.orders || []), ...visibleOrders];
        }
        setOrders(visibleOrders);
        setSelectedId(preferredId && visibleOrders.some((order) => order.order_id === preferredId) ? preferredId : visibleOrders[0]?.order_id || "");
        setLoading(false);
      })
      .catch((reason) => {
        if (!active) return;
        setError(reason.message);
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [category, plansOnly, packageReadyOnly]);

  const filteredOrders = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return orders;
    return orders.filter((order) => `${order.order_id} ${order.merchant_id}`.toLowerCase().includes(needle));
  }, [orders, query]);

  const selectedOrder = orders.find((order) => order.order_id === selectedId) || null;

  async function openPlan(orderId = selectedId) {
    if (!orderId) return;
    setSelectedId(orderId);
    setScreen("plan");
    setDetailLoading(true);
    setError("");
    try {
      setPlan(await getJson(`/order/${encodeURIComponent(orderId)}/plan`));
    } catch (reason) {
      setError(reason.message);
    } finally {
      setDetailLoading(false);
    }
  }

  async function openPackage() {
    if (!selectedId) return;
    if (plan?.order?.order_id === selectedId && plan.package_available !== true) {
      setScreen("plan");
      setError("");
      return;
    }
    setScreen("package");
    setDetailLoading(true);
    setError("");
    try {
      setPkg(await getJson(`/order/${encodeURIComponent(selectedId)}/dispute-package`));
    } catch (reason) {
      setError(reason.message);
      setPkg(null);
    } finally {
      setDetailLoading(false);
    }
  }

  function navigate(next) {
    if (next === "plan" && !plan) {
      openPlan();
      return;
    }
    if (next === "package") {
      if (!plan || plan.order?.order_id !== selectedId) {
        if (selectedId) openPlan(selectedId);
        else setScreen("order");
        return;
      }
      openPackage();
      return;
    }
    setScreen(next);
  }

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="brand-mark" aria-label="Vulcan Proof">
          <span className="brand-flame">V</span>
          <span className="brand-word">Vulcan</span>
        </div>
        <button
          className={`demo-rail-cta ${screen === "demo" ? "active" : ""}`}
          onClick={() => setScreen("demo")}
          type="button"
        >
          <span className="demo-rail-kicker">INTERACTIVE</span>
          <span className="demo-rail-title">▶ Live demo</span>
        </button>
        <div className="rail-label">WORKSPACE</div>
        <nav className="rail-nav" aria-label="Demo sections">
          {navItems.map((item) => (
            <button
              className={`rail-item ${screen === item.id ? "active" : ""}`}
              key={item.id}
              onClick={() => navigate(item.id)}
              type="button"
            >
              <span className="rail-number">{item.short}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="rail-bottom">
          <div className="rail-status-dot" />
          <div>
            <div className="rail-status-title">Demo environment</div>
            <div className="rail-status-copy">Canonical test slice</div>
          </div>
        </div>
      </aside>

      <main className={`main-shell ${screen === "demo" ? "demo-active" : ""}`}>
        <header className="topbar">
          <div className="crumbs"><span>Risk operations</span><span className="crumb-slash">/</span><strong>Proof desk</strong></div>
          <div className="topbar-right">
            <span className="status-pill"><span className="status-dot" />SIMULATOR · TEST</span>
            <span className="avatar">RP</span>
          </div>
        </header>

        <div className="content-wrap">
          {screen === "demo" ? <LiveDemo /> : <>
          {error && <div className="error-banner" role="alert"><span>!</span>{error}<button onClick={() => setError("")} type="button">Dismiss</button></div>}
          {loading ? <LoadingState /> : (
            <>
              {screen === "order" && (
                <OrderScreen
                  category={category}
                  setCategory={setCategory}
                  query={query}
                  setQuery={setQuery}
                  plansOnly={plansOnly}
                  setPlansOnly={setPlansOnly}
                  packageReadyOnly={packageReadyOnly}
                  setPackageReadyOnly={setPackageReadyOnly}
                  orders={filteredOrders}
                  selectedId={selectedId}
                  setSelectedId={setSelectedId}
                  selectedOrder={selectedOrder}
                  openPlan={() => openPlan()}
                />
              )}
              {screen === "plan" && (
                <PlanScreen plan={plan} loading={detailLoading} openPackage={openPackage} goBack={() => setScreen("order")} />
              )}
              {screen === "package" && (
                <PackageScreen pkg={pkg} loading={detailLoading} goBack={() => setScreen("plan")} />
              )}
              {screen === "report" && <ReportScreen report={report} />}
            </>
          )}
          </>}
        </div>
        <footer className="app-footer">
          <span className="footer-rule" />
          <span>Simulator result · production calibration requires Razorpay dispute history</span>
          <span className="footer-right">Vulcan Proof · Phase 5</span>
        </footer>
      </main>
    </div>
  );
}

function LiveDemo() {
  const [phase, setPhase] = useState("intro");
  const [visibleCards, setVisibleCards] = useState(1);
  const [visibleArrows, setVisibleArrows] = useState(0);
  const [advancing, setAdvancing] = useState(false);

  useEffect(() => {
    if (!advancing) return undefined;
    const nextCard = window.setTimeout(() => {
      setVisibleCards((count) => count + 1);
      setAdvancing(false);
    }, 560);
    return () => window.clearTimeout(nextCard);
  }, [advancing]);

  function startSimulation() {
    setVisibleCards(1);
    setVisibleArrows(0);
    setAdvancing(false);
    setPhase("cards");
  }

  function advanceCards() {
    if (advancing) return;
    if (visibleCards === 4) {
      setPhase("reveal");
      return;
    }
    setVisibleArrows(visibleCards);
    setAdvancing(true);
  }

  function restartSimulation() {
    setVisibleCards(1);
    setVisibleArrows(0);
    setAdvancing(false);
    setPhase("intro");
  }

  return (
    <section className={`live-demo live-demo-${phase}`} aria-label="Vulcan Proof live simulation">
      {phase === "intro" && (
        <div className="demo-intro">
          <div className="demo-live-pill"><span />LIVE SIMULATION · VULCAN PROOF</div>
          <h1>Watch live how Vulcan Proof can make the difference between winning or losing a ₹43,907 dispute</h1>
          <p>A customer buys an electronics order. Razorpay Vulcan correctly approves the payment. 62 days later, the customer files a chargeback — “I never received it.”</p>
          <button className="demo-primary-button" onClick={startSimulation} type="button">Start simulation →</button>
          <div className="demo-metadata">Order #0074677 · Electronics · merchant_005802</div>
        </div>
      )}

      {phase === "cards" && (
        <div className="demo-cards-phase">
          <div className="demo-step-hint" aria-live="polite">{visibleCards === 4 ? "All stages shown" : `Step ${visibleCards} of 4 — click Next to continue`}</div>
          <div className="demo-cards-row">
            <DemoOrderCard visible />
            <DemoArrow visible={visibleArrows >= 1} />
            <DemoPaymentCard visible={visibleCards >= 2} />
            <DemoArrow visible={visibleArrows >= 2} />
            <DemoRiskCard visible={visibleCards >= 3} />
            <DemoArrow visible={visibleArrows >= 3} />
            <DemoPlanCard visible={visibleCards >= 4} />
          </div>
          <button className="demo-primary-button demo-next-button" disabled={advancing} onClick={advanceCards} type="button">
            {visibleCards === 4 ? "Fast-forward 62 days →" : advancing ? "Revealing…" : "Next →"}
          </button>
        </div>
      )}

      {phase === "reveal" && (
        <div className="demo-reveal">
          <div className="demo-time-jump">62 DAYS LATER</div>
          <h1>And the customer raised a bank dispute fraud...</h1>
          <div className="chargeback-box">
            <strong>CHARGEBACK FILED · VISA 13.1 · NON-RECEIPT</strong>
            <em>“I never received this item. I am requesting a full refund.”</em>
          </div>
          <button className="demo-dark-button" onClick={() => setPhase("result")} type="button">Show result →</button>
        </div>
      )}

      {phase === "result" && <DemoResult onRestart={restartSimulation} />}
    </section>
  );
}

function DemoCard({ accent, label, title, visible, children, badge, badgeTone = "blue" }) {
  return (
    <article className={`demo-card demo-card-${accent} ${visible ? "is-visible" : ""}`} aria-hidden={!visible}>
      <div className="demo-card-accent" />
      <div className="demo-card-body">
        <div className="demo-card-label">{label}</div>
        <h2>{title}</h2>
        <div className="demo-card-content">{children}</div>
        <div className={`demo-card-badge demo-badge-${badgeTone}`}>{badge}</div>
      </div>
    </article>
  );
}

function DemoArrow({ visible }) {
  return <div className={`demo-arrow ${visible ? "is-visible" : ""}`} aria-hidden="true"><span /></div>;
}

function DemoOrderCard({ visible }) {
  return (
    <DemoCard accent="blue" label="01 / ORDER" title="Customer order" visible={visible} badge="Order confirmed">
      <div className="demo-field-list">
        <DemoField label="Customer" value="Raj Kumar" />
        <DemoField label="Order" value="#0074677" />
        <DemoField label="Product" value="Samsung Galaxy S25" />
        <DemoField label="Category" value="Electronics" />
        <DemoField label="Amount" value="₹43,907.15" prominent />
        <DemoField label="Payment" value="Full prepaid" />
        <DemoField label="Date" value="26 Aug 2026" />
      </div>
    </DemoCard>
  );
}

function DemoField({ label, value, prominent = false }) {
  return <div className={`demo-field ${prominent ? "prominent" : ""}`}><span>{label}</span><strong>{value}</strong></div>;
}

function DemoPaymentCard({ visible }) {
  return (
    <DemoCard accent="green" label="02 / PAYMENT" title="Razorpay Vulcan" visible={visible} badge="✓ Payment approved" badgeTone="green">
      <div className="demo-risk-score"><span className="demo-score-check">✓</span><strong>0.04</strong><small>LOW RISK</small></div>
      <div className="demo-note-box">No fraud signals detected. Payment processed normally.</div>
    </DemoCard>
  );
}

function DemoRiskCard({ visible }) {
  const risks = [
    ["Non-Receipt", 48, "red"],
    ["Not-As-Described", 29, "orange"],
    ["Empty Box", 23, "yellow"],
  ];
  return (
    <DemoCard accent="orange" label="03 / RISK" title="Vulcan Proof" visible={visible} badge="⚠ Dispute risk detected" badgeTone="orange">
      <div className="demo-exposure"><strong>31%</strong><span>exposure</span></div>
      <div className="demo-exposure-note">ELEVATED · Evidence window open</div>
      <div className="demo-risk-bars">
        {risks.map(([name, value, tone]) => <div className="demo-risk-row" key={name}><div><span>{name}</span><strong>{value}%</strong></div><i><b className={`risk-fill-${tone}`} style={{ width: `${value}%` }} /></i></div>)}
      </div>
    </DemoCard>
  );
}

function DemoPlanCard({ visible }) {
  const evidence = [
    ["Weight", "₹4.22", true],
    ["Serial no.", "₹3.38", true],
    ["Packing video", "₹2.13", true],
    ["Geotag", "—", false],
    ["OTP", "—", false],
  ];
  return (
    <DemoCard accent="blue" label="04 / PLAN" title="Evidence plan" visible={visible} badge="Plan EV: ₹0.47">
      <div className="demo-plan-note">Capture before dispatch:</div>
      <div className="demo-plan-list">
        {evidence.map(([name, value, selected]) => <div className={`demo-plan-row ${selected ? "selected" : ""}`} key={name}><span className="demo-plan-check">{selected ? "✓" : ""}</span><span>{name}</span><strong>{value}</strong></div>)}
      </div>
    </DemoCard>
  );
}

function DemoResult({ onRestart }) {
  const story = [
    ["Told the merchant to record the package weight before shipping.", "Merchant did. When the dispute claimed tampering — there was nothing to stand on."],
    ["Told the merchant to photograph the serial number.", "Merchant did. The exact device was linked to this order in seconds."],
    ["Told the merchant to record the packing on video.", "Merchant did. The “empty box” claim collapsed the moment it was submitted."],
  ];
  return (
    <div className="demo-result">
      <section className="demo-outcome-zone">
        <div><div className="demo-result-label">DISPUTE OUTCOME · ORDER #0074677</div><h1>Merchant won the dispute.</h1><p>Vulcan Proof’s recommendations matched every requirement of the dispute.</p></div>
        <div className="demo-protected"><strong>PROTECTED</strong><b>₹43,907</b><span>recovered</span></div>
      </section>
      <section className="demo-story-zone">
        <div className="demo-result-label">What Vulcan Proof did</div>
        {story.map(([title, copy]) => <div className="demo-story-row" key={title}><span>✓</span><div><strong>{title}</strong><p>{copy}</p></div></div>)}
      </section>
      <section className="demo-contrast-zone">
        <strong>Without Vulcan Proof</strong>
        <p>No one told the merchant what to collect. So nothing was collected. When the dispute arrived 62 days later, all they had was a “DELIVERED” status. That’s not enough. ₹43,907 — gone.</p>
      </section>
      <button className="demo-restart-button" onClick={onRestart} type="button">← Restart simulation</button>
    </div>
  );
}

function LoadingState() {
  return <div className="loading-state"><div className="spinner" /><span>Loading the stored test slice…</span></div>;
}

function PageIntro({ eyebrow, title, copy, action }) {
  return (
    <div className="page-intro">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        {copy && <p>{copy}</p>}
      </div>
      {action}
    </div>
  );
}

function OrderScreen({ category, setCategory, query, setQuery, plansOnly, setPlansOnly, packageReadyOnly, setPackageReadyOnly, orders, selectedId, setSelectedId, selectedOrder, openPlan }) {
  const categories = ["Electronics", "Jewellery", "Apparel", "Home", "FMCG"];
  return (
    <>
      <PageIntro
        eyebrow="01 / ORDER"
        title="Decision desk"
        copy="Pick a test order to inspect the evidence plan that was selected for its context."
        action={<div className="intro-context"><span className="context-label">WORLD</span><span className="context-value">κ = 0.6 · Seed 1</span></div>}
      />
      <div className="order-layout">
        <section className="panel order-browser">
          <div className="panel-heading compact-heading">
            <div><h2>Test orders</h2><span className="muted">{packageReadyOnly ? "Stored orders with dispute packages" : plansOnly ? "Stored orders with evidence plans" : "Canonical Phase 3 slice"}</span></div>
            <span className="result-count">{orders.length} {packageReadyOnly ? "package-ready" : plansOnly ? "plan examples" : "shown"}</span>
          </div>
          <div className="filter-row">
            <div className="search-field"><span className="search-icon">⌕</span><input aria-label="Search orders" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search order or merchant" /></div>
            <select aria-label="Filter category" value={category} onChange={(event) => setCategory(event.target.value)}>
              {categories.map((item) => <option key={item}>{item}</option>)}
            </select>
            <label className="plan-filter"><input aria-label="Show evidence plans only" type="checkbox" checked={plansOnly} onChange={(event) => { setPlansOnly(event.target.checked); if (!event.target.checked) setPackageReadyOnly(false); }} /><span>Plans only</span></label>
            <label className="plan-filter"><input aria-label="Show package-ready examples only" type="checkbox" checked={packageReadyOnly} disabled={!plansOnly} onChange={(event) => setPackageReadyOnly(event.target.checked)} /><span>Package ready</span></label>
          </div>
          <div className="order-table-wrap">
            <table className="order-table">
              <thead><tr><th>Order</th><th>Category</th><th>Value</th><th>Tier</th></tr></thead>
              <tbody>
                {orders.map((order) => (
                  <tr className={selectedId === order.order_id ? "selected" : ""} onClick={() => setSelectedId(order.order_id)} key={order.order_id}>
                    <td><div className="order-cell"><span className="row-radio">{selectedId === order.order_id ? "✓" : ""}</span><span>{order.order_id.replace("order_1_", "#")}</span></div><small>{order.merchant_id} · {order.has_plan ? <><span className="order-status plan-status">evidence plan</span><span className={`order-status ${order.package_available ? "ready-status" : "plan-only-status"}`}>{order.package_available ? "package ready" : "plan only"}</span></> : <span className="order-status empty-status">no evidence selected</span>}</small></td>
                    <td><span className={`category-tag ${categoryColors[order.category] || "slate"}`}>{order.category}</span></td>
                    <td className="money-cell">{money(order.order_value)}</td>
                    <td><span className="tier-label">{pretty(order.eligible_tier)}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!orders.length && <div className="empty-table">{packageReadyOnly ? `No package-ready examples are available for ${category} in this world. Turn off “Package ready” to inspect plan examples.` : plansOnly ? `No stored evidence-bearing examples are available for ${category} in this world. Turn off “Plans only” to inspect the full slice.` : "No matching orders in this slice."}</div>}
          </div>
        </section>

        <section className="order-summary-column">
          <div className="selected-card panel">
            <div className="selected-card-top"><span className="selected-label">SELECTED ORDER</span><span className="live-mark"><span />stored</span></div>
            {selectedOrder ? <>
              <div className="order-id-large">{selectedOrder.order_id.replace("order_1_", "#")}</div>
              <div className="order-meta-line"><span>{selectedOrder.merchant_id}</span><span>·</span><span>Day {selectedOrder.decision_date}</span></div>
              <div className="value-block"><span>Order value</span><strong>{money(selectedOrder.order_value)}</strong></div>
              <div className="summary-grid">
                <SummaryField label="Category" value={selectedOrder.category} />
                <SummaryField label="Payment tier" value={pretty(selectedOrder.eligible_tier)} />
                <SummaryField label="Split" value="Test" />
                <SummaryField label="Source" value="Phase 3" />
              </div>
              <button className="primary-button full-button" onClick={openPlan} type="button">Review evidence plan <span>→</span></button>
            </> : <div className="blank-selection">Select an order from the table.</div>}
          </div>
          <div className="note-card">
            <div className="note-icon">i</div>
            <div><strong>Read the plan in context</strong><p>Every selection is tied to a stored test order and its observed purchase-time features.</p></div>
          </div>
        </section>
      </div>
    </>
  );
}

function SummaryField({ label, value }) {
  return <div className="summary-field"><span>{label}</span><strong>{value}</strong></div>;
}

function PlanScreen({ plan, loading, openPackage, goBack }) {
  if (loading || !plan) return <div className="detail-loading"><div className="spinner" /><span>Preparing the evidence readout…</span></div>;
  const stages = plan.stages || {};
  const selected = plan.evidence?.filter((item) => item.selected) || [];
  const exposure = Number(stages.exposure_probability || 0);
  const riskLabel = exposure >= 0.02 ? "Elevated exposure" : "Baseline exposure";
  return (
    <>
      <PageIntro
        eyebrow="02 / PLAN"
        title="Evidence plan"
        copy="The stored Arm 5 plan, with model diagnostics and per-evidence refusal reasons."
        action={<button className="secondary-button" onClick={goBack} type="button">← Change order</button>}
      />
      <div className="plan-order-strip panel">
        <div className="plan-order-main"><span className="selected-label">ORDER</span><strong>{plan.order.order_id.replace("order_1_", "#")}</strong><span className={`category-tag ${categoryColors[plan.order.category] || "slate"}`}>{plan.order.category}</span></div>
        <div className="plan-order-value"><span>Value</span><strong>{money(plan.order.order_value)}</strong></div>
        <div className="plan-order-risk"><span>Exposure</span><strong>{pct(exposure, 2)}</strong><em className={exposure >= 0.02 ? "risk-high" : "risk-normal"}>{riskLabel}</em></div>
      </div>
      <div className="plan-summary-grid">
        <section className="panel recommendation-card">
          <div className="card-kicker">RECOMMENDATION</div>
          <div className="recommendation-title">{selected.length ? selected.map((item) => item.label).join(" + ") : "No additional evidence"}</div>
          <p>{selected.length ? "Selected from the stored plan for this order context." : "The plan keeps the pre-dispatch workflow clear for this order context."}</p>
          <div className="recommendation-footer"><span>Plan EV</span><strong>{money(plan.plan.ev)}</strong></div>
        </section>
        <section className="panel type-card">
          <div className="card-kicker">ESTIMATED DISPUTE MIX</div>
          <div className="type-bars">
            {Object.entries(stages.dispute_type_probabilities || {}).map(([name, value]) => <div className="type-row" key={name}><span>{name}</span><div className="mini-track"><i style={{ width: `${Math.max(Number(value) * 100, 3)}%` }} /></div><strong>{pct(value, 0)}</strong></div>)}
          </div>
          <div className="type-footnote">Conditional on exposure</div>
        </section>
        <section className="panel comparison-card">
          <div className="card-kicker">POLICY COMPARISON</div>
          <div className="comparison-row"><div><span className="compare-label">Arm 5 · stored</span><strong>{selected.length ? selected.map((item) => item.label).join(" + ") : "Empty"}</strong></div><span className="compare-arrow">→</span></div>
          <div className="comparison-row muted-row"><div><span className="compare-label">Arm 4 · tuned rule</span><strong>{plan.comparison.arm4.evidence?.length ? plan.comparison.arm4.evidence.map(pretty).join(" + ") : "Empty"}</strong></div></div>
          <div className="comparison-note">Same order · paired readout</div>
        </section>
      </div>

      <section className="panel evidence-panel">
        <div className="panel-heading"><div><div className="card-kicker">EVIDENCE DECISION</div><h2>What made the cut</h2></div><span className="small-note">Selected items are highlighted; other rows show the stored reason code.</span></div>
        <div className="evidence-list">
          {(plan.evidence || []).map((item) => <EvidenceRow item={item} key={item.name} />)}
        </div>
      </section>
      <div className="plan-bottom-row">
        <div className="diagnostic-line"><span className="diagnostic-dot" />Model recomputation matches stored plan: <strong>{plan.comparison.model_recomputed_mask_matches_stored ? "yes" : "stored mask retained"}</strong></div>
        {plan.package_available ? <button className="primary-button" onClick={openPackage} type="button">Open dispute package <span>→</span></button> : <div className="package-unavailable"><strong>Package not available</strong><span>No opened dispute with captured evidence for this order.</span></div>}
      </div>
    </>
  );
}

function EvidenceRow({ item }) {
  const value = item.incremental_ev ?? item.standalone_ev;
  const magnitude = Math.min(Math.abs(Number(value || 0)) / 20, 1) * 100;
  return (
    <div className={`evidence-row ${item.selected ? "is-selected" : ""}`}>
      <div className="evidence-name"><span className={`evidence-check ${item.selected ? "checked" : ""}`}>{item.selected ? "✓" : ""}</span><div><strong>{item.label}</strong><span>{pretty(item.window)} · {item.available ? "available" : "not available"}</span></div></div>
      <div className="evidence-bar"><div className="evidence-track"><i className={Number(value) < 0 ? "negative" : ""} style={{ width: `${Math.max(magnitude, value !== null && value !== undefined ? 4 : 0)}%` }} /></div><span>{value === null || value === undefined ? "—" : money(value)}</span></div>
      <div className="evidence-slot"><span>API slot</span><strong>{item.api_slot}</strong></div>
      <ReasonBadge item={item} />
    </div>
  );
}

function ReasonBadge({ item }) {
  const labels = { SELECTED: "Selected", UNAVAILABLE: "Unavailable", INADMISSIBLE: "Not admissible", NO_SUPPORT: "Low support", NEGATIVE_STANDALONE: "Negative value", NEGATIVE_INCREMENTAL: "Overlap" };
  return <span className={`reason-badge ${item.selected ? "selected-badge" : item.reason === "NO_SUPPORT" ? "warning-badge" : ""}`}>{labels[item.reason] || pretty(item.reason)}</span>;
}

function PackageScreen({ pkg, loading, goBack }) {
  if (loading) return <div className="detail-loading"><div className="spinner" /><span>Opening the stored dispute package…</span></div>;
  if (!pkg) return <div className="detail-loading"><span>No dispute package is available for this order.</span><button className="secondary-button" onClick={goBack} type="button">← Back to plan</button></div>;
  return (
    <>
      <PageIntro eyebrow="03 / PACKAGE" title="Dispute package" copy="A compact handoff view for the materialised evidence bound to this order." action={<button className="secondary-button" onClick={goBack} type="button">← Back to plan</button>} />
      <div className="package-header panel"><div><span className="selected-label">STORED OUTCOME</span><div className="package-order-id">{pkg.order_id.replace("order_1_", "#")}</div><div className="order-meta-line">{pkg.category} <span>·</span> {pkg.dispute_type} dispute <span>·</span> Arm 5</div></div><div className="package-value"><span>Order value</span><strong>{money(pkg.order_value)}</strong></div></div>
      <div className="package-layout">
        <section className="panel package-items"><div className="panel-heading"><div><div className="card-kicker">CAPTURED ARTIFACTS</div><h2>API-ready evidence</h2></div><span className="captured-count"><span />{pkg.items?.length || 0} captured</span></div>
          {pkg.items?.length ? <div className="package-list">{pkg.items.map((item) => <div className="package-item" key={item.evidence}><span className="package-check">✓</span><div className="package-item-main"><strong>{item.label}</strong><span>{pretty(item.window)} · requested and captured</span></div><div className="package-item-slot"><span>Mapped to</span><strong>{item.api_slot}</strong></div></div>)}</div> : <div className="package-empty">No evidence materialised in the stored outcome.</div>}
        </section>
        <section className="panel provenance-card"><div className="card-kicker">PROVENANCE</div><h2>Bound to this order</h2><div className="provenance-line"><span className="timeline-dot" /><div><strong>{pkg.provenance.bound_to_order.replace("order_1_", "#")}</strong><span>Order binding</span></div></div><div className="provenance-line"><span className="timeline-dot" /><div><strong>Day {pkg.provenance.decision_day}</strong><span>Decision point</span></div></div><div className="provenance-line"><span className="timeline-dot last" /><div><strong>Phase 3 outcome</strong><span>Stored simulator artifact</span></div></div><div className="provenance-foot">The package preserves timing and order binding for review.</div></section>
      </div>
      <div className="package-footer-line"><span className="success-check">✓</span> Ready for a dispute review handoff <span className="footer-divider" /> <span>{pkg.dispute_type} · API slots mapped</span></div>
    </>
  );
}

function ReportScreen({ report }) {
  if (!report) return <div className="detail-loading"><div className="spinner" /><span>Loading the validation status…</span></div>;
  const deferred = report.production_results_available !== true;
  const smoke = report.smoke_simulator_result || {};
  return (
    <>
      <PageIntro eyebrow="04 / REPORT" title="Validation report" copy="A concise readout of what is available for this buildathon demo." action={<span className={`scope-pill ${deferred ? "deferred" : "complete"}`}>{deferred ? "Smoke scope" : "Extended scope"}</span>} />
      <section className={`validation-card panel ${deferred ? "deferred-card" : "complete-card"}`}><div className="validation-icon">{deferred ? "…" : "✓"}</div><div className="validation-copy"><div className="card-kicker">PHASE 4 STATUS</div><h2>{deferred ? "Production-scale robustness validation is deferred" : "Extended validation is available"}</h2><p>{report.message || "The completed Phase 4 artefacts are available for review."}</p></div><div className="validation-side"><span>Result scope</span><strong>{deferred ? "Buildathon smoke" : "Extended validation"}</strong><em>{deferred ? "No production result surfaced" : "Artefacts complete"}</em></div></section>
      <div className="report-grid">
        <section className="panel smoke-card"><div className="panel-heading"><div><div className="card-kicker">SMOKE VALIDATION</div><h2>Simulator comparison</h2></div><span className="mini-status">available</span></div>{deferred ? <><div className="smoke-metric"><span>Arm 5 − Arm 4 net / 1,000</span><strong>{money(smoke.arm5_minus_arm4_net_per_1000)}</strong></div><div className="ci-range"><span>95% interval</span><strong>{money(smoke.ci_low)} <i>to</i> {money(smoke.ci_high)}</strong></div><div className="metric-caption">Smoke simulator validation at κ = {smoke.kappa ?? "—"}. This is not a production measurement.</div></> : <ExtendedReport report={report} />}</section>
        <section className="panel anchor-card"><div className="card-kicker">REAL-DATA ANCHOR</div><h2>Olist detection</h2><p>Public marketplace data is used for detection only; it has no dispute or evidence fields.</p><div className="anchor-stat-grid"><AnchorStat label="PR-AUC" value={report.phase0?.pr_auc} /><AnchorStat label="ROC-AUC" value={report.phase0?.roc_auc} /><AnchorStat label="Top-decile lift" value={report.phase0?.top_decile_lift} /></div><div className="olist-footer">Olist public dataset · Brazil 2016–18 · no chargeback or evidence data · detection only</div></section>
      </div>
      <div className="report-note"><span className="note-icon">i</span><span>Phase 5 keeps validation scope visible. A missing extended result is not converted into a numeric κ*.</span></div>
    </>
  );
}

function ExtendedReport({ report }) {
  return <div className="extended-readout"><div className="smoke-metric"><span>κ*</span><strong>{report.kappa_star ?? "not found on [0, 1]"}</strong></div><div className="ci-range"><span>Verdict</span><strong>{pretty(report.verdict)}</strong></div><div className="metric-caption">Extended validation artefacts have passed the Phase 4 completeness gate.</div></div>;
}

function AnchorStat({ label, value }) {
  return <div className="anchor-stat"><span>{label}</span><strong>{value === undefined ? "—" : Number(value).toFixed(3)}</strong></div>;
}

createRoot(document.getElementById("root")).render(<App />);
