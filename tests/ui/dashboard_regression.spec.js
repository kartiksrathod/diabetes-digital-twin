const { test, expect } = require('@playwright/test');
const path = require('path');

const DASHBOARD_HTML_PATH = path.resolve(__dirname, '../../dashboard/index.html');
const DASHBOARD_FILE_URL = 'file:///' + DASHBOARD_HTML_PATH.replace(/\\/g, '/');

// Deterministic API fixture mocks
const mockReplayHistory = {
  success: true,
  patient_id: "2001",
  session_id: "2001_0_20201102",
  data_provenance: "Historical ShanghaiT2DM rows replayed as simulated inputs",
  threshold: 0.57,
  prediction_window: "Next 2 hours",
  count: 3,
  rows: [
    {
      time: "11:40",
      timestamp: "2020-11-02T11:40:00",
      glucose: 88.2,
      delta30m: 1.8,
      risk: 0.1811,
      trend: "Stable",
      level: "LOW",
      predicted_event: "No glucose spike"
    },
    {
      time: "11:55",
      timestamp: "2020-11-02T11:55:00",
      glucose: 90.0,
      delta30m: 2.5,
      risk: 0.2015,
      trend: "Stable",
      level: "LOW",
      predicted_event: "No glucose spike"
    },
    {
      time: "12:10",
      timestamp: "2020-11-02T12:10:00",
      glucose: 95.0,
      delta30m: 5.0,
      risk: 0.2500,
      trend: "Stable",
      level: "LOW",
      predicted_event: "No glucose spike"
    }
  ]
};

const mockLowRiskTwin = {
  success: true,
  twin_id: "P001",
  patient_id: "2001",
  current_glucose: 88.2,
  delta_30m: 1.8,
  glucose_trend: "Stable",
  risk_score: 0.1811,
  risk_level: "LOW",
  predicted_event: "No glucose spike",
  prediction_window: "Next 2 hours",
  last_updated: "2026-10-17T12:00:00+00:00",
  source_timestamp: "2020-11-02 11:40:00",
  data_provenance: "Simulated replay of historical CGM data"
};

const mockHighRiskTwin = {
  success: true,
  twin_id: "P001",
  patient_id: "2001",
  current_glucose: 163.8,
  delta_30m: 21.6,
  glucose_trend: "Rising",
  risk_score: 0.6775,
  risk_level: "HIGH",
  predicted_event: "Glucose spike",
  prediction_window: "Next 2 hours",
  last_updated: "2026-10-17T12:05:00+00:00",
  source_timestamp: "2020-11-02 13:55:00",
  data_provenance: "Simulated replay of historical CGM data"
};

const mockSystemStatus = {
  success: true,
  azure_twin_reachable: true,
  azure_error: null,
  feature_csv_exists: true,
  model_file_exists: true,
  mcp_server: "Separate stdio server; not called directly by browser",
  antigravity_agent: "Runs in Antigravity; browser does not directly query the agent",
  replay_source: "Historical fixture; chart endpoint is read-only"
};

test.describe('Dashboard Browser Regression Suite', () => {

  test('deterministic live API mode renders telemetry and provenance correctly', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    // Mock API routes
    await page.route('**/api/replay-history*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayHistory),
    }));

    await page.route('**/api/twin-state/P001', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockLowRiskTwin),
    }));

    await page.route('**/api/system-status', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockSystemStatus),
    }));

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    // Verify mode badge reflects live mode
    const modeBadge = page.locator('#mode-badge');
    await expect(modeBadge).toHaveText('LIVE TWIN API');

    // Telemetry cards verification
    const cardGlucose = page.locator('#card-glucose');
    await expect(cardGlucose).toHaveText('88.2');

    const cardDelta = page.locator('#card-delta');
    await expect(cardDelta).toHaveText('+1.8 mg/dL');

    const cardTrend = page.locator('#card-trend');
    await expect(cardTrend).toHaveText('Stable');

    const cardRiskScore = page.locator('#card-risk-score');
    await expect(cardRiskScore).toHaveText('18.11');

    const badgeClassification = page.locator('#badge-classification');
    await expect(badgeClassification).toContainText(/LOW/i);

    // Decision math logic (0.57 threshold)
    const decisionMathBadge = page.locator('#decision-math-badge');
    await expect(decisionMathBadge).toHaveText('SCORE < 0.57 (BELOW THRESHOLD)');

    const mathScore = page.locator('#math-score');
    await expect(mathScore).toHaveText('0.1811');

    // Provenance verification
    const body = page.locator('body');
    await expect(body).toContainText(/SIMULATED HISTORICAL CGM REPLAY/i);

    // Header timestamp has source row
    const headerTimestamp = page.locator('#header-timestamp');
    await expect(headerTimestamp).toContainText('2020-11-02 11:40:00');

    // System status indicator
    const statusAzure = page.locator('#status-azure');
    await expect(statusAzure).toHaveText('Reachable');

    // No uncaught JavaScript errors
    expect(consoleErrors).toEqual([]);
  });

  test('high risk twin renders HIGH alerts, margin, and event prediction', async ({ page }) => {
    await page.route('**/api/replay-history*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayHistory),
    }));

    await page.route('**/api/twin-state/P001', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockHighRiskTwin),
    }));

    await page.route('**/api/system-status', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockSystemStatus),
    }));

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    await expect(page.locator('#card-glucose')).toHaveText('163.8');
    await expect(page.locator('#card-delta')).toHaveText('+21.6 mg/dL');
    await expect(page.locator('#card-trend')).toHaveText('Rising');
    await expect(page.locator('#card-risk-score')).toHaveText('67.75');
    await expect(page.locator('#card-class-text')).toHaveText('HIGH RISK');
    await expect(page.locator('#card-predicted-event')).toHaveText('Glucose spike');

    // Decision math at or above threshold
    await expect(page.locator('#decision-math-badge')).toHaveText('SCORE ≥ 0.57 (SPIKE PREDICTED)');
  });

  test('null timestamps and deltas do not crash the page', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    const sparseTwin = {
      success: true,
      twin_id: "P001",
      patient_id: "2001",
      current_glucose: 110.0,
      delta_30m: null,
      glucose_trend: "Stable",
      risk_score: 0.25,
      risk_level: "LOW",
      predicted_event: "No glucose spike",
      prediction_window: "Next 2 hours",
      last_updated: null,
      source_timestamp: null,
      data_provenance: "Simulated replay of historical CGM data"
    };

    await page.route('**/api/replay-history*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayHistory),
    }));

    await page.route('**/api/twin-state/P001', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(sparseTwin),
    }));

    await page.route('**/api/system-status', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockSystemStatus),
    }));

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    // Delta card must gracefully show 'Not stored in twin'
    const cardDelta = page.locator('#card-delta');
    await expect(cardDelta).toHaveText('Not stored in twin');

    // Header timestamp must not show undefined
    const headerTimestamp = page.locator('#header-timestamp');
    await expect(headerTimestamp).not.toContainText('undefined');

    expect(consoleErrors).toEqual([]);
  });

  test('api failure falls back gracefully with clear error notification', async ({ page }) => {
    // Handle alert dialog automatically
    page.on('dialog', async dialog => {
      expect(dialog.message()).toContain('Live API could not be reached');
      await dialog.dismiss();
    });

    // Mock API 502 Bad Gateway
    await page.route('**/api/**', route => route.fulfill({
      status: 502,
      contentType: 'application/json',
      body: JSON.stringify({ detail: "Azure twin read failed" }),
    }));

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    // Mode badge must clearly indicate API UNAVAILABLE
    const modeBadge = page.locator('#mode-badge');
    await expect(modeBadge).toContainText('API UNAVAILABLE');
  });

  test('local fixture mode supports scrubbing and controls', async ({ page }) => {
    // Navigate without ?mode=live
    await page.goto(DASHBOARD_FILE_URL, { waitUntil: 'domcontentloaded' });

    const modeBadge = page.locator('#mode-badge');
    await expect(modeBadge).toHaveText('LOCAL FIXTURE REPLAY');

    // Controls must be enabled in fixture mode
    const btnPlay = page.locator('#btn-play');
    await expect(btnPlay).toBeEnabled();

    const slider = page.locator('#step-slider');
    await expect(slider).toBeEnabled();

    // Reset button resets to initial index
    const btnReset = page.locator('#btn-reset');
    await btnReset.click();

    const stepCounter = page.locator('#step-counter');
    await expect(stepCounter).toContainText('1 / 10');
  });

  test('window resize event does not break responsive SVG charts', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    await page.goto(DASHBOARD_FILE_URL, { waitUntil: 'domcontentloaded' });

    // Resize viewport
    await page.setViewportSize({ width: 800, height: 600 });
    await page.waitForTimeout(100);

    const glucoseSvg = page.locator('#glucose-svg');
    await expect(glucoseSvg).toBeVisible();

    expect(consoleErrors).toEqual([]);
  });

});
