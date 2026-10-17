const { test, expect } = require('@playwright/test');

const API = 'http://127.0.0.1:8000';
const DASHBOARD = 'http://127.0.0.1:5500/?mode=live';

test('live dashboard matches the read-only API state', async ({ page, request }) => {
  // Read-only baseline: GET requests only; no replay or twin writes.
  const apiResponse = await request.get(`${API}/api/twin-state/P001`);
  expect(apiResponse.ok()).toBeTruthy();

  const twin = await apiResponse.json();

  expect(twin.success).toBe(true);
  expect(twin.twin_id).toBe('P001');
  expect(twin.current_glucose).toBeCloseTo(88.2, 1);
  expect(twin.risk_level).toBe('LOW');
  expect(twin.risk_score).toBeCloseTo(0.1811013017677028, 6);
  expect(twin.data_provenance).toContain('Simulated replay of historical CGM data');

  // Open the real dashboard in Chromium.
  const response = await page.goto(DASHBOARD, { waitUntil: 'domcontentloaded' });
  expect(response).not.toBeNull();
  expect(response.status()).toBe(200);

  // Wait for asynchronous dashboard rendering.
  const body = page.locator('body');
  await expect(body).toContainText(/88\.2/, { timeout: 15000 });
  await expect(body).toContainText(/LOW/i, { timeout: 15000 });

  const visibleText = await body.innerText();

  // Supports dashboards that display the score as either a percentage
  // (18.11%) or a decimal (0.1811).
  expect(visibleText).toMatch(/18\.11|0\.1811/);

  // Confirm that the displayed state has the correct data provenance.
  expect(visibleText).toMatch(/SIMULATED HISTORICAL CGM REPLAY/i);

  console.log('Verified twin:', twin.twin_id);
  console.log('API glucose:', twin.current_glucose);
  console.log('API risk:', twin.risk_level);
  console.log('API risk score:', twin.risk_score);
});

