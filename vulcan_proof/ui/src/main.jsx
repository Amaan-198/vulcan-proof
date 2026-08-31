import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { BadgeCheck, Barcode, Check, CheckCheck, CheckCircle2, KeyRound, LoaderCircle, MapPin, Package, PackageCheck, Signature, Weight, X } from "lucide-react";
import "./styles.css";

const EVIDENCE_ORDER = ["weight", "serial", "sealed", "packing", "geotag", "otp", "signature", "ack", "vack"];
const DISPUTE_TYPE_LABELS = {
  NR: "Non-Receipt",
  NAD: "Not-As-Described",
  EB: "Empty Box",
};
const EVIDENCE_DISPLAY_LABELS = {
  ack: "Acknowledgement",
  vack: "Verified Acknowledgement",
};
const DEMO_ORDER_LIMIT = 36;
const FEATURED_DEMO = {
  orderId: "#0107201",
  amount: "₹79,439.43",
  amountRounded: "₹79,439",
  merchantId: "merchant_003633",
  payment: "Card",
  disputeType: "Non-Receipt (NR)",
  exposureCount: "~49",
  exposureUnit: "per 10,000 orders",
  fraudScore: "0.04",
  riskMix: [
    ["Non-Receipt (NR)", 38, "red"],
    ["Not-As-Described (NAD)", 46, "orange"],
    ["Empty Box (EB)", 16, "yellow"],
  ],
  evidence: ["Sealed packaging", "Verified acknowledgement"],
};

const navItems = [
  { id: "order", label: "Order", short: "01" },
  { id: "plan", label: "Plan", short: "02" },
  { id: "package", label: "Package", short: "03" },
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

function wholeMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const amount = Number(value);
  const sign = amount < 0 ? "-" : "";
  return `${sign}₹${Math.abs(amount).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

function perTenThousandOrders(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `~${Math.round(Number(value) * 10000).toLocaleString("en-IN")} per 10,000 orders`;
}

function perTenThousandMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `~${wholeMoney(Number(value) * 10000)} per 10,000 orders`;
}

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function pretty(value) {
  return String(value || "—").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function displayDisputeType(value) {
  const code = String(value || "").trim().toUpperCase();
  const label = DISPUTE_TYPE_LABELS[code];
  return label ? `${label} (${code})` : pretty(value);
}

function displayEvidenceLabel(value) {
  const normalized = String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
  return EVIDENCE_DISPLAY_LABELS[normalized] || pretty(value);
}

const EVIDENCE_ICONS = {
  weight: Weight,
  serial: Barcode,
  sealed: PackageCheck,
  packing: Package,
  geotag: MapPin,
  otp: KeyRound,
  signature: Signature,
  ack: CheckCheck,
  vack: BadgeCheck,
};

function evidenceIconKey(value) {
  const normalized = String(value || "").trim().toLowerCase().replace(/[_-]+/g, " ").replace(/\s+/g, " ");
  if (normalized.startsWith("verified acknowledgement")) return "vack";
  if (normalized.startsWith("sealed")) return "sealed";
  if (normalized.startsWith("packing")) return "packing";
  if (normalized.startsWith("serial")) return "serial";
  if (normalized === "ack" || normalized.startsWith("acknowledgement")) return "ack";
  if (normalized.startsWith("otp")) return "otp";
  if (normalized.startsWith("signature")) return "signature";
  if (normalized.startsWith("geotag")) return "geotag";
  if (normalized.startsWith("weight")) return "weight";
  return normalized;
}

function EvidenceIcon({ value }) {
  const Icon = EVIDENCE_ICONS[evidenceIconKey(value)];
  return Icon ? <Icon className="evidence-type-icon" size={14} strokeWidth={1.8} aria-hidden="true" focusable="false" /> : null;
}

function displayApiSlot(value) {
  const slot = String(value ?? "—");
  return slot === "others" ? "General documentation" : slot;
}

function ApiSlotValue({ value }) {
  const slot = String(value ?? "—");
  const isRawApiSlot = slot !== "—" && slot !== "others";
  return <strong className={isRawApiSlot ? "api-slot-chip" : ""}>{displayApiSlot(slot)}</strong>;
}

function displayOrderId(value) {
  return String(value || "—").replace(/^order_\d+_/, "#");
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
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [orderTableReady, setOrderTableReady] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    let revealTimer;
    setOrderTableReady(false);
    Promise.all([
      getJson(`/orders?category=${encodeURIComponent(category)}&limit=${DEMO_ORDER_LIMIT}&plans_only=${plansOnly}&package_ready_only=${packageReadyOnly}`),
      getJson("/demo/script").catch(() => null),
    ])
      .then(async ([orderData, script]) => {
        if (!active) return;
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
        revealTimer = window.setTimeout(() => {
          if (active) setOrderTableReady(true);
        }, 680);
      })
      .catch((reason) => {
        if (!active) return;
        setError(reason.message);
        setLoading(false);
        setOrderTableReady(true);
      });
    return () => {
      active = false;
      if (revealTimer) window.clearTimeout(revealTimer);
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

  const orderView = (
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
      tableLoading={loading || !orderTableReady}
      openPlan={() => openPlan()}
    />
  );

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
          {loading && screen !== "order" ? <LoadingState /> : (
            <div className="workspace-screen" key={screen}>
              {screen === "order" && orderView}
              {screen === "plan" && (
                <PlanScreen plan={plan} loading={detailLoading} openPackage={openPackage} goBack={() => setScreen("order")} />
              )}
              {screen === "package" && (
                <PackageScreen pkg={pkg} loading={detailLoading} goBack={() => setScreen("plan")} />
              )}
            </div>
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
  const [outgoingPhase, setOutgoingPhase] = useState("");
  const [phaseTransitionKey, setPhaseTransitionKey] = useState(0);
  const [visibleCards, setVisibleCards] = useState(1);
  const [visibleArrows, setVisibleArrows] = useState(0);
  const [advancing, setAdvancing] = useState(false);
  const [incomingReady, setIncomingReady] = useState(true);
  const phaseExitTimer = useRef(null);

  useEffect(() => {
    if (!advancing) return undefined;
    const nextCard = window.setTimeout(() => {
      setVisibleCards((count) => count + 1);
      setAdvancing(false);
    }, 560);
    return () => window.clearTimeout(nextCard);
  }, [advancing]);

  useEffect(() => () => {
    if (phaseExitTimer.current) window.clearTimeout(phaseExitTimer.current);
  }, []);

  function transitionTo(nextPhase) {
    if (nextPhase === phase) return;
    if (phaseExitTimer.current) window.clearTimeout(phaseExitTimer.current);
    setOutgoingPhase(phase);
    setIncomingReady(false);
    setPhase(nextPhase);
    setPhaseTransitionKey((key) => key + 1);
    phaseExitTimer.current = window.setTimeout(() => {
      setOutgoingPhase("");
      setIncomingReady(true);
    }, 560);
  }

  function startSimulation() {
    setVisibleCards(1);
    setVisibleArrows(0);
    setAdvancing(false);
    transitionTo("cards");
  }

  function advanceCards() {
    if (advancing) return;
    if (visibleCards === 4) {
      transitionTo("reveal");
      return;
    }
    setVisibleArrows(visibleCards);
    setAdvancing(true);
  }

  function restartSimulation() {
    setVisibleCards(1);
    setVisibleArrows(0);
    setAdvancing(false);
    transitionTo("intro");
  }

  function renderPhase(currentPhase) {
    return (
      <>
      {currentPhase === "intro" && (
        <div className="demo-intro">
          <div className="demo-live-pill"><span />LIVE SIMULATION · VULCAN PROOF</div>
          <h1>Watch live how Vulcan Proof can make the difference in a {FEATURED_DEMO.amountRounded} dispute</h1>
          <p>A customer buys an electronics order. Razorpay Vulcan correctly approves the payment. 62 days later, the customer files a {FEATURED_DEMO.disputeType} dispute — “I never received it.”</p>
          <button className="demo-primary-button" onClick={startSimulation} type="button">Start simulation →</button>
          <div className="demo-metadata">Order {FEATURED_DEMO.orderId} · Electronics · {FEATURED_DEMO.merchantId}</div>
        </div>
      )}

      {currentPhase === "cards" && (
        <div className="demo-cards-phase">
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

      {currentPhase === "reveal" && (
        <div className="demo-reveal">
          <div className="demo-time-jump">62 DAYS LATER</div>
          <h1>And the customer filed a {FEATURED_DEMO.disputeType} dispute.</h1>
          <div className="chargeback-box">
            <strong>DISPUTE FILED · VISA 13.1 · {FEATURED_DEMO.disputeType.toUpperCase()}</strong>
            <em>“I never received this item. I am requesting a full refund.”</em>
          </div>
          <button className="demo-dark-button" onClick={() => transitionTo("result")} type="button">Show result →</button>
        </div>
      )}

      {currentPhase === "result" && <DemoResult onRestart={restartSimulation} />}
      </>
    );
  }

  return (
    <section className={`live-demo live-demo-${phase}`} aria-label="Vulcan Proof live simulation">
      <div className="demo-stage-stack">
        {outgoingPhase && <div className={`demo-stage demo-stage-${outgoingPhase} demo-stage-exiting`} aria-hidden="true">{renderPhase(outgoingPhase)}</div>}
        <div className={`demo-stage demo-stage-entering ${incomingReady ? "" : "demo-stage-waiting"}`} key={phaseTransitionKey}>{renderPhase(phase)}</div>
      </div>
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
        {badge && <div className={`demo-card-badge demo-badge-${badgeTone}`}>{badge}</div>}
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
        <DemoField label="Order" value={FEATURED_DEMO.orderId} />
        <DemoField label="Product" value="Consumer electronics" />
        <DemoField label="Category" value="Electronics" />
        <DemoField label="Amount" value={FEATURED_DEMO.amount} prominent />
        <DemoField label="Payment" value={FEATURED_DEMO.payment} />
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
      <div className="demo-risk-score"><span className="demo-score-check">✓</span><span className="demo-score-label">Fraud score</span><strong>{FEATURED_DEMO.fraudScore}</strong><small>LOW RISK</small></div>
      <div className="demo-note-box">No fraud signals detected. Payment processed normally.</div>
    </DemoCard>
  );
}

function DemoRiskCard({ visible }) {
  return (
    <DemoCard accent="orange" label="03 / RISK" title="Vulcan Proof" visible={visible} badge="⚠ Dispute risk detected" badgeTone="orange">
      <div className="demo-exposure"><strong>{FEATURED_DEMO.exposureCount}</strong><span>{FEATURED_DEMO.exposureUnit}<br />estimated exposure</span></div>
      <div className="demo-exposure-note">MODEL ESTIMATE · Evidence window open</div>
      <div className="demo-risk-bars">
        {FEATURED_DEMO.riskMix.map(([name, value, tone]) => <div className="demo-risk-row" key={name}><div><span>{name}</span><strong>{value}%</strong></div><i><b className={`risk-fill-${tone}`} style={{ width: `${value}%` }} /></i></div>)}
      </div>
    </DemoCard>
  );
}

function DemoPlanCard({ visible }) {
  const evidence = [
    [FEATURED_DEMO.evidence[0], "selected", true],
    [FEATURED_DEMO.evidence[1], "selected", true],
    ["Weight", "—", false],
    ["Serial no.", "—", false],
    ["Geotag", "—", false],
  ];
  return (
    <DemoCard accent="blue" label="04 / PLAN" title="Evidence plan" visible={visible}>
      <div className="demo-plan-note">Capture before dispatch:</div>
      <div className="demo-plan-list">
        {evidence.map(([name, value, selected]) => <div className={`demo-plan-row ${selected ? "selected" : ""}`} key={name}><span className="demo-plan-check">{selected ? "✓" : ""}</span><span className="demo-plan-name"><EvidenceIcon value={name} /><span>{name}</span></span><strong>{value}</strong></div>)}
      </div>
    </DemoCard>
  );
}

function DemoResult({ onRestart }) {
  const withProof = [
    "Evaluated the order before dispatch — not after a dispute was filed.",
    "Weighed the cost of each evidence type against its likely payoff.",
    "Recommended sealed packaging evidence before dispatch.",
    "Recommended verified acknowledgement for the order.",
    "The dispute needed exactly this evidence — and Vulcan Proof had it ready.",
  ];
  const withoutProof = [
    "No evaluation before dispatch.",
    "No evidence plan in place.",
    "Just a \"DELIVERED\" status when the dispute arrived.",
    "Nothing to contest with.",
    `${FEATURED_DEMO.amountRounded} order left undefended.`,
  ];
  return (
    <div className="demo-result">
      <section className="demo-outcome-zone">
        <div><div className="demo-result-label">DISPUTE OUTCOME · ORDER {FEATURED_DEMO.orderId}</div><h1>Merchant won the dispute.</h1></div>
        <div className="demo-protected"><strong>PACKAGE READY</strong><b>{FEATURED_DEMO.amountRounded}</b><span>for review</span></div>
      </section>
      <section className="demo-comparison-zone">
        <div className="demo-comparison-column demo-comparison-positive">
          <div className="demo-comparison-heading"><div className="demo-result-label">WITH VULCAN PROOF</div></div>
          {withProof.map((item) => <div className="demo-comparison-row" key={item}><span className="comparison-icon"><Check size={14} strokeWidth={2.4} aria-hidden="true" focusable="false" /></span><strong>{item}</strong></div>)}
        </div>
        <div className="demo-comparison-column demo-comparison-negative">
          <div className="demo-comparison-heading"><div className="demo-result-label">WITHOUT VULCAN PROOF</div></div>
          {withoutProof.map((item) => <div className="demo-comparison-row" key={item}><span className="comparison-icon"><X size={14} strokeWidth={2.4} aria-hidden="true" focusable="false" /></span><strong>{item}</strong></div>)}
        </div>
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

function OrderScreen({ category, setCategory, query, setQuery, plansOnly, setPlansOnly, packageReadyOnly, setPackageReadyOnly, orders, selectedId, setSelectedId, selectedOrder, tableLoading, openPlan }) {
  const categories = ["Electronics", "Jewellery", "Apparel", "Home", "FMCG"];
  return (
    <>
      <PageIntro
        eyebrow="01 / ORDER"
        title="Decision desk"
        copy="Pick a test order to inspect the evidence plan that was selected for its context."
        action={<div className="intro-context"><span className="context-label">WORLD</span><span className="context-value">κ = 0.6 · Seed 2</span></div>}
      />
      <div className="order-layout">
        <section className="panel order-browser">
          <div className="panel-heading compact-heading">
            <div><h2>Test orders</h2><span className="muted">{packageReadyOnly ? "Stored orders with dispute packages" : plansOnly ? "Stored orders with evidence plans" : "Canonical Phase 3 slice"}</span></div>
            <span className="result-count">{tableLoading ? "checking" : `${orders.length} ${packageReadyOnly ? "package-ready" : plansOnly ? "plan examples" : "shown"}`}</span>
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
            {tableLoading ? <div className="order-table-loading" role="status" aria-live="polite"><LoaderCircle className="order-table-loading-icon" size={22} strokeWidth={1.8} /><strong>Checking stored orders</strong><span>Preparing the canonical test slice</span></div> : <>
              <table className="order-table">
                <thead><tr><th>Order</th><th>Category</th><th>Value</th><th>Tier</th></tr></thead>
                <tbody>
                  {orders.map((order) => (
                    <tr className={selectedId === order.order_id ? "selected" : ""} onClick={() => setSelectedId(order.order_id)} key={order.order_id}>
                      <td><div className="order-cell"><span className="row-radio">{selectedId === order.order_id ? "✓" : ""}</span><span>{displayOrderId(order.order_id)}</span></div><small>{order.merchant_id} · {order.has_plan ? <><span className="order-status plan-status">evidence plan</span><span className={`order-status ${order.package_available ? "ready-status" : "plan-only-status"}`}>{order.package_available ? "package ready" : "plan only"}</span></> : <span className="order-status empty-status">no evidence selected</span>}</small></td>
                      <td><span className={`category-tag ${categoryColors[order.category] || "slate"}`}>{order.category}</span></td>
                      <td className="money-cell">{money(order.order_value)}</td>
                      <td><span className="tier-label">{pretty(order.eligible_tier)}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!orders.length && <div className="empty-table">{packageReadyOnly ? `No package-ready examples are available for ${category} in this world. Turn off “Package ready” to inspect plan examples.` : plansOnly ? `No stored evidence-bearing examples are available for ${category} in this world. Turn off “Plans only” to inspect the full slice.` : "No matching orders in this slice."}</div>}
            </>}
          </div>
        </section>

        <section className="order-summary-column">
          <div className="selected-card panel">
            <div className="selected-card-top"><span className="selected-label">SELECTED ORDER</span><span className="live-mark"><span />stored</span></div>
            {selectedOrder ? <div className="selected-order-content" key={selectedOrder.order_id}>
              <div className="order-id-large">{displayOrderId(selectedOrder.order_id)}</div>
              <div className="order-meta-line"><span>{selectedOrder.merchant_id}</span></div>
              <div className="value-block"><span>Order value</span><strong>{money(selectedOrder.order_value)}</strong></div>
              <div className="summary-grid">
                <SummaryField label="Category" value={selectedOrder.category} />
                <SummaryField label="Payment tier" value={pretty(selectedOrder.eligible_tier)} />
                <SummaryField label="Split" value="Test" />
                <SummaryField label="Source" value="Phase 3" />
              </div>
              <button className="primary-button full-button" onClick={openPlan} type="button">Review evidence plan <span>→</span></button>
            </div> : <div className="blank-selection">Select an order from the table.</div>}
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
  const [recomputing, setRecomputing] = useState(true);

  useEffect(() => {
    setRecomputing(true);
    const verificationTimer = window.setTimeout(() => setRecomputing(false), 720);
    return () => window.clearTimeout(verificationTimer);
  }, [plan?.order?.order_id]);

  if (loading || !plan) return <div className="detail-loading"><div className="spinner" /><span>Preparing the evidence readout…</span></div>;
  const stages = plan.stages || {};
  const selected = plan.evidence?.filter((item) => item.selected) || [];
  const exposure = Number(stages.exposure_probability || 0);
  return (
    <>
      <PageIntro
        eyebrow="02 / PLAN"
        title="Evidence plan"
        copy="The stored evidence plan, with model diagnostics and per-evidence refusal reasons."
        action={<button className="secondary-button" onClick={goBack} type="button">← Change order</button>}
      />
      <div className="plan-order-strip panel">
        <div className="plan-order-main"><span className="selected-label">ORDER</span><strong>{displayOrderId(plan.order.order_id)}</strong><span className={`category-tag ${categoryColors[plan.order.category] || "slate"}`}>{plan.order.category}</span></div>
        <div className="plan-order-value"><span>Value</span><strong>{money(plan.order.order_value)}</strong></div>
        <div className="plan-order-risk"><span>Estimated risk</span><strong>{perTenThousandOrders(exposure)}</strong></div>
      </div>
      <div className="plan-summary-grid">
        <section className="panel recommendation-card">
          <div className="card-kicker">RECOMMENDATION</div>
          <div className="recommendation-title">{selected.length ? selected.map((item) => displayEvidenceLabel(item.name || item.label)).join(" + ") : "No additional evidence"}</div>
          <p>{selected.length ? "Selected from the stored plan for this order context." : "The plan keeps the pre-dispatch workflow clear for this order context."}</p>
          <div className="recommendation-footer"><div className="recommendation-metric"><span>Estimated value added</span><strong>{perTenThousandMoney(plan.plan.ev)}</strong></div><div className="recommendation-metric"><span>Value protected on this order</span><strong>{money(plan.order.order_value)}</strong></div></div>
        </section>
        <section className="panel type-card">
          <div className="card-kicker">ESTIMATED DISPUTE MIX</div>
          <div className="type-bars">
            {Object.entries(stages.dispute_type_probabilities || {}).map(([name, value]) => <div className="type-row" key={name}><span>{displayDisputeType(name)}</span><div className="mini-track"><i style={{ width: `${Math.max(Number(value) * 100, 3)}%` }} /></div><strong>{pct(value, 0)}</strong></div>)}
          </div>
        </section>
        <section className="panel comparison-card">
          <div className="card-kicker">POLICY COMPARISON</div>
          <div className="comparison-row"><div><span className="compare-label">Stored plan</span><strong>{selected.length ? selected.map((item) => displayEvidenceLabel(item.name || item.label)).join(" + ") : "Empty"}</strong></div><span className="compare-arrow">→</span></div>
          <div className="comparison-row muted-row"><div><span className="compare-label">Tuned rule (Arm 4)</span><strong>{plan.comparison.arm4.evidence?.length ? plan.comparison.arm4.evidence.map(displayEvidenceLabel).join(" + ") : "Collected nothing"}</strong></div></div>
          <div className="comparison-note">Same order · two policies side by side</div>
        </section>
      </div>

      <section className="panel evidence-panel">
        <div className="panel-heading"><div><div className="card-kicker">EVIDENCE DECISION</div><h2>What made the cut</h2></div><span className="small-note">Selected items are highlighted; other rows show why each item was or wasn't included.</span></div>
        <div className="evidence-list">
          <div className="evidence-table-header"><span>Evidence</span><span>API slot</span><span>Status</span></div>
          {(plan.evidence || []).map((item) => <EvidenceRow item={item} key={item.name} />)}
        </div>
      </section>
      <div className="plan-bottom-row">
        <div className={`diagnostic-line ${recomputing ? "is-checking" : "is-confirmed"}`} role="status" aria-live="polite">
          <span className="diagnostic-icon">{recomputing ? <LoaderCircle size={15} strokeWidth={1.9} aria-hidden="true" focusable="false" /> : <CheckCircle2 size={15} strokeWidth={2.1} aria-hidden="true" focusable="false" />}</span>
          {recomputing ? "Checking stored plan match…" : <>Model recomputation matches stored plan: <strong>{plan.comparison.model_recomputed_mask_matches_stored ? "yes" : "stored mask retained"}</strong></>}
        </div>
        {plan.package_available ? <button className="primary-button" onClick={openPackage} type="button">Open dispute package <span>→</span></button> : <div className="package-unavailable"><strong>Package not available</strong><span>No opened dispute with captured evidence for this order.</span></div>}
      </div>
    </>
  );
}

function EvidenceRow({ item }) {
  return (
    <div className={`evidence-row ${item.selected ? "is-selected" : ""}`}>
      <div className="evidence-name"><span className={`evidence-check ${item.selected ? "checked" : ""}`}>{item.selected ? "✓" : ""}</span><div><div className="evidence-type-name"><EvidenceIcon value={item.name || item.label} /><strong>{displayEvidenceLabel(item.name || item.label)}</strong></div><span>{pretty(item.window)} · {item.available ? "available" : "not available"}</span></div></div>
      <div className="evidence-slot"><span>API slot</span><strong>{displayApiSlot(item.api_slot)}</strong></div>
      <ReasonBadge item={item} />
    </div>
  );
}

function ReasonBadge({ item }) {
  const labels = { SELECTED: "Selected", UNAVAILABLE: "Unavailable", INADMISSIBLE: "Not admissible", NO_SUPPORT: "Low support", NEGATIVE_STANDALONE: "Not cost-effective", NEGATIVE_INCREMENTAL: "Overlap" };
  return <span className={`reason-badge ${item.selected ? "selected-badge" : item.reason === "NO_SUPPORT" ? "warning-badge" : ""}`}>{labels[item.reason] || pretty(item.reason)}</span>;
}

function PackageScreen({ pkg, loading, goBack }) {
  if (loading) return <div className="detail-loading"><div className="spinner" /><span>Opening the stored dispute package…</span></div>;
  if (!pkg) return <div className="detail-loading"><span>No dispute package is available for this order.</span><button className="secondary-button" onClick={goBack} type="button">← Back to plan</button></div>;
  return (
    <>
      <PageIntro eyebrow="03 / PACKAGE" title="Dispute package" copy="A compact handoff view for the materialised evidence bound to this order." action={<button className="secondary-button" onClick={goBack} type="button">← Back to plan</button>} />
      <div className="package-header panel"><div><span className="selected-label">DISPUTE CASE</span><div className="package-order-id">{displayOrderId(pkg.order_id)}</div><div className="order-meta-line">{pkg.category} <span>·</span> {displayDisputeType(pkg.dispute_type)} dispute</div></div><div className="package-value"><span>Order value</span><strong>{money(pkg.order_value)}</strong></div></div>
      <div className="package-layout">
        <section className="panel package-items"><div className="panel-heading"><div><div className="card-kicker">CAPTURED ARTIFACTS</div><h2>API-ready evidence</h2></div><span className="captured-count"><span />{pkg.items?.length || 0} captured</span></div>
          {pkg.items?.length ? <div className="package-list">{pkg.items.map((item) => <div className="package-item" key={item.evidence}><span className="package-check">✓</span><div className="package-item-main"><div className="evidence-type-name"><EvidenceIcon value={item.evidence || item.label} /><strong>{displayEvidenceLabel(item.evidence || item.label)}</strong></div><span>{pretty(item.window)} · requested and captured</span></div><div className="package-item-slot"><span>Mapped to</span><ApiSlotValue value={item.api_slot} /></div></div>)}</div> : <div className="package-empty">No evidence materialised in the stored outcome.</div>}
        </section>
        <section className="panel provenance-card"><div className="card-kicker">PROVENANCE</div><h2>Bound to this order</h2><div className="provenance-line"><span className="timeline-dot" /><div><strong>{displayOrderId(pkg.provenance.bound_to_order)}</strong><span>Order binding</span></div></div><div className="provenance-line"><span className="timeline-dot last" /><div><strong>System-generated decision</strong><span>Computed automatically from order data</span></div></div></section>
      </div>
      <div className="package-footer-line"><span className="success-check">✓</span> Ready for a dispute review handoff <span className="footer-divider" /> <span>{displayDisputeType(pkg.dispute_type)} · API slots mapped</span></div>
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
