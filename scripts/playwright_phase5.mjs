import playwright from "../vulcan_proof/ui/node_modules/playwright/index.js";

const { chromium } = playwright;

const baseURL = process.env.PHASE5_BASE_URL || "http://127.0.0.1:8765";
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const consoleErrors = [];
const pageErrors = [];
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => pageErrors.push(error.message));

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
  }));
  check(`${name} has no horizontal overflow`, data.scrollWidth <= data.viewportWidth && data.bodyWidth <= data.viewportWidth, JSON.stringify(data));
  return data;
}

await page.goto(baseURL, { waitUntil: "networkidle" });
await page.screenshot({ path: "outputs/phase5/playwright-order.png", fullPage: true });
check("order screen renders", await page.getByRole("heading", { name: "Decision desk" }).isVisible());
await layout("order screen");

await page.getByRole("button", { name: /Review evidence plan/ }).click();
await page.getByRole("heading", { name: "Evidence plan" }).waitFor({ state: "visible", timeout: 60000 });
await page.screenshot({ path: "outputs/phase5/playwright-plan.png", fullPage: true });
check("plan screen renders", true);
await layout("plan screen");
check("plan rows render", (await page.locator(".evidence-row").count()) === 9);

await page.getByRole("button", { name: /Open dispute package/ }).click();
await page.getByRole("heading", { name: "Dispute package" }).waitFor({ state: "visible", timeout: 30000 });
await page.screenshot({ path: "outputs/phase5/playwright-package.png", fullPage: true });
await layout("package screen");
check("package screen renders", true);

await page.getByRole("button", { name: /Report/ }).click();
await page.getByRole("heading", { name: "Validation report" }).waitFor({ state: "visible" });
await page.screenshot({ path: "outputs/phase5/playwright-report.png", fullPage: true });
await layout("report screen");
check("deferred status is visible", await page.getByRole("heading", { name: /Production-scale robustness validation is deferred/ }).isVisible());

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
await page.setViewportSize({ width: 1280, height: 800 });
await layout("1280px laptop viewport");

check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | "));
check("no page errors", pageErrors.length === 0, pageErrors.join(" | "));
console.log(JSON.stringify({ checks, consoleErrors, pageErrors }, null, 2));
await browser.close();
if (checks.some((item) => !item.pass) || consoleErrors.length || pageErrors.length) process.exitCode = 1;
