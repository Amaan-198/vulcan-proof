import playwright from "../vulcan_proof/ui/node_modules/playwright/index.js";

const { chromium } = playwright;

const baseURL = process.env.PHASE5_BASE_URL || "http://127.0.0.1:8765";
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const consoleErrors = [];
const pageErrors = [];
const requestFailures = [];
const httpErrors = [];
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => pageErrors.push(error.message));
page.on("requestfailed", (request) => requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText || "failed"}`));
page.on("response", (response) => {
  if (response.status() >= 400) httpErrors.push(`${response.status()} ${response.url()}`);
});

const checks = [];
function check(name, pass, detail = "") {
  checks.push({ name, pass, detail });
}
async function layout(name) {
  const data = await page.evaluate(() => ({
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight,
    bodyWidth: document.body.scrollWidth,
    active: document.querySelector(".rail-item.active")?.textContent?.trim() || "",
    clipped: Array.from(document.querySelectorAll("body *")).filter((element) => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0
        && (rect.left < -1 || rect.right > window.innerWidth + 1);
    }).slice(0, 8).map((element) => ({ tag: element.tagName, className: element.className, text: element.textContent?.trim().slice(0, 40) })),
  }));
  check(`${name} has no horizontal overflow`, data.scrollWidth <= data.viewportWidth && data.bodyWidth <= data.viewportWidth, JSON.stringify(data));
  check(`${name} has no visibly clipped layout elements`, data.clipped.length === 0, JSON.stringify(data.clipped));
  return data;
}

async function footerAtBottom(name) {
  await page.evaluate(() => window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "instant" }));
  const visible = await page.locator(".app-footer").isVisible();
  const inViewport = await page.locator(".app-footer").evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return rect.bottom <= window.innerHeight + 1 && rect.top >= -1;
  });
  check(`${name} footer is reachable at scroll end`, visible && inViewport);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
}

await page.goto(baseURL, { waitUntil: "networkidle" });
await page.screenshot({ path: "outputs/phase5/playwright-order.png", fullPage: true });
check("order screen renders", await page.getByRole("heading", { name: "Decision desk" }).isVisible());
check("report option is absent", await page.getByRole("button", { name: "Report", exact: true }).count() === 0);
const reportRequested = await page.evaluate(() => performance.getEntriesByType("resource").some((entry) => entry.name.includes("/report/")));
check("report API is not requested", !reportRequested);
await layout("order screen");
await footerAtBottom("order screen");

await page.getByRole("button", { name: /Review evidence plan/ }).click();
await page.getByRole("heading", { name: "Evidence plan" }).waitFor({ state: "visible", timeout: 60000 });
await page.screenshot({ path: "outputs/phase5/playwright-plan.png", fullPage: true });
check("plan screen renders", true);
await layout("plan screen");
check("plan rows render", (await page.locator(".evidence-row").count()) === 9);
check("plan recommendation is populated", (await page.locator(".recommendation-title").textContent())?.trim().length > 0);
await footerAtBottom("plan screen");

await page.getByRole("button", { name: /Open dispute package/ }).click();
await page.getByRole("heading", { name: "Dispute package" }).waitFor({ state: "visible", timeout: 30000 });
await page.screenshot({ path: "outputs/phase5/playwright-package.png", fullPage: true });
await layout("package screen");
check("package screen renders", true);
check("package has provenance", await page.getByRole("heading", { name: "Bound to this order" }).isVisible());
await footerAtBottom("package screen");

await page.getByRole("button", { name: /Order/ }).click();
await page.getByRole("heading", { name: "Decision desk" }).waitFor({ state: "visible" });
await page.getByLabel("Filter category").selectOption("Apparel");
await page.getByLabel("Search orders").fill("merchant_");
await page.waitForFunction(() => Array.from(document.querySelectorAll(".selected-card .summary-field")).some((field) => field.textContent.includes("Category") && field.querySelector("strong")?.textContent.trim() === "Apparel"), { timeout: 30000 });
check("category filter leaves visible rows", await page.locator(".order-table tbody tr").count() > 0);
const selectedCategory = await page.locator(".selected-card .summary-field").filter({ hasText: "Category" }).locator("strong").textContent();
check("selection follows category filter", selectedCategory?.trim() === "Apparel", selectedCategory?.trim() || "");
await page.screenshot({ path: "outputs/phase5/playwright-filter.png", fullPage: true });
await layout("filtered order screen");
await footerAtBottom("filtered order screen");

for (const viewport of [
  { width: 1366, height: 768 },
  { width: 1280, height: 800 },
  { width: 1024, height: 768 },
  { width: 960, height: 800 },
]) {
  await page.setViewportSize(viewport);
  await layout(`${viewport.width}x${viewport.height} laptop viewport`);
}

check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | "));
check("no page errors", pageErrors.length === 0, pageErrors.join(" | "));
check("no failed requests", requestFailures.length === 0, requestFailures.join(" | "));
check("no HTTP error responses", httpErrors.length === 0, httpErrors.join(" | "));
console.log(JSON.stringify({ checks, consoleErrors, pageErrors, requestFailures, httpErrors }, null, 2));
await browser.close();
if (checks.some((item) => !item.pass) || consoleErrors.length || pageErrors.length || requestFailures.length || httpErrors.length) process.exitCode = 1;
