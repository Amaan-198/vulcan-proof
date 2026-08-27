import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const EVIDENCE_ORDER = ["weight", "serial", "sealed", "packing", "geotag", "otp", "signature", "ack", "vack"];

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
      getJson(`/orders?category=${encodeURIComponent(category)}&limit=36`),
      getJson("/report/kappa"),
      getJson("/demo/script").catch(() => null),
    ])
      .then(async ([orderData, reportData, script]) => {
        if (!active) return;
        setReport(reportData);
        const scriptId = script?.beats?.find((beat) => beat.beat === 1)?.order_id;
        const preferredId = category === "Electronics" ? scriptId : orderData.orders?.[0]?.order_id;
        const firstId = preferredId || orderData.orders?.[0]?.order_id || "";
        let visibleOrders = orderData.orders || [];
        if (category === "Electronics" && scriptId && !visibleOrders.some((order) => order.order_id === scriptId)) {
          const featured = await getJson(`/orders?query=${encodeURIComponent(scriptId)}&limit=1`);
          visibleOrders = [...(featured.orders || []), ...visibleOrders];
        }
        setOrders(visibleOrders);
        setSelectedId(firstId);
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
  }, [category]);

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

      <main className="main-shell">
        <header className="topbar">
          <div className="crumbs"><span>Risk operations</span><span className="crumb-slash">/</span><strong>Proof desk</strong></div>
          <div className="topbar-right">
            <span className="status-pill"><span className="status-dot" />SIMULATOR · TEST</span>
            <span className="avatar">RP</span>
          </div>
        </header>

        <div className="content-wrap">
          {error && <div className="error-banner" role="alert"><span>!</span>{error}<button onClick={() => setError("")} type="button">Dismiss</button></div>}
          {loading ? <LoadingState /> : (
            <>
              {screen === "order" && (
                <OrderScreen
                  category={category}
                  setCategory={setCategory}
                  query={query}
                  setQuery={setQuery}
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

function OrderScreen({ category, setCategory, query, setQuery, orders, selectedId, setSelectedId, selectedOrder, openPlan }) {
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
            <div><h2>Test orders</h2><span className="muted">Canonical Phase 3 slice</span></div>
            <span className="result-count">{orders.length} shown</span>
          </div>
          <div className="filter-row">
            <div className="search-field"><span className="search-icon">⌕</span><input aria-label="Search orders" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search order or merchant" /></div>
            <select aria-label="Filter category" value={category} onChange={(event) => setCategory(event.target.value)}>
              {categories.map((item) => <option key={item}>{item}</option>)}
            </select>
          </div>
          <div className="order-table-wrap">
            <table className="order-table">
              <thead><tr><th>Order</th><th>Category</th><th>Value</th><th>Tier</th></tr></thead>
              <tbody>
                {orders.map((order) => (
                  <tr className={selectedId === order.order_id ? "selected" : ""} onClick={() => setSelectedId(order.order_id)} key={order.order_id}>
                    <td><div className="order-cell"><span className="row-radio">{selectedId === order.order_id ? "✓" : ""}</span><span>{order.order_id.replace("order_1_", "#")}</span></div><small>{order.merchant_id}</small></td>
                    <td><span className={`category-tag ${categoryColors[order.category] || "slate"}`}>{order.category}</span></td>
                    <td className="money-cell">{money(order.order_value)}</td>
                    <td><span className="tier-label">{pretty(order.eligible_tier)}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!orders.length && <div className="empty-table">No matching orders in this slice.</div>}
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
        <button className="primary-button" onClick={openPackage} type="button">Open dispute package <span>→</span></button>
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
