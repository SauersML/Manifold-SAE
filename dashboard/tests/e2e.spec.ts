// Playwright end-to-end test.
//
// Setup:
//   cd dashboard/tests
//   npm init -y && npm i -D @playwright/test && npx playwright install chromium
//   # In one terminal:  bash dashboard/scripts/dev.sh
//   # In another:       npx playwright test e2e.spec.ts
import { test, expect } from "@playwright/test";

const URL = process.env.DASHBOARD_URL || "http://localhost:3000";

test("dashboard boots and shows manifold + steering chat", async ({ page }) => {
  await page.goto(URL);

  // Header
  await expect(page.locator("header h1")).toContainText("Manifold-SAE");

  // Sidebar tabs
  await expect(page.locator(".tab", { hasText: "Manifold" })).toBeVisible();
  await expect(page.locator(".tab", { hasText: "Diagnostics" })).toBeVisible();
  await expect(page.locator(".tab", { hasText: "History" })).toBeVisible();

  // Three.js canvas mounted
  await expect(page.locator("canvas")).toBeVisible();

  // Click the "Steer with current concept" button and expect a completion to appear
  await page.locator("button.primary").click();
  await expect(page.locator(".chat-msg")).toBeVisible({ timeout: 5_000 });

  // Switch to Diagnostics
  await page.locator(".tab", { hasText: "Diagnostics" }).click();
  await expect(page.locator("text=Variance per axis")).toBeVisible({ timeout: 5_000 });

  // Switch to History
  await page.locator(".tab", { hasText: "History" }).click();
  await expect(page.locator("text=Steering history")).toBeVisible();
});

test("color-mode legend toggles", async ({ page }) => {
  await page.goto(URL);
  await page.locator(".legend button", { hasText: "hue" }).click();
  await expect(page.locator(".legend button.active", { hasText: "hue" })).toBeVisible();
  await page.locator(".legend button", { hasText: "modifier" }).click();
  await expect(page.locator(".legend button.active", { hasText: "modifier" })).toBeVisible();
});
