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

const mockReplayStatus = {
  success: true,
  running: false,
  busy: false,
  paused: false,
  completed: false,
  patient_id: "2001",
  session_id: "2001_0_20201102",
  twin_id: "P001",
  current_index: 0,
  total_rows: 3,
  processed_rows: 0,
  delay_seconds: 2.0,
  threshold: 0.57,
  prediction_window: "Next 2 hours",
  data_provenance: "Simulated replay of historical CGM data",
  last_error: null,
  last_record: null
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

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayStatus),
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

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayStatus),
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

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayStatus),
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

  test('live-mode play and pause calls /api/replay/start and /api/replay/pause', async ({ page }) => {
    let startCalled = false;
    let pauseCalled = false;
    let isRunning = false;

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

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ...mockReplayStatus, running: isRunning, paused: !isRunning && startCalled }),
    }));

    await page.route('**/api/replay/start', route => {
      startCalled = true;
      isRunning = true;
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ...mockReplayStatus, running: true, total_rows: 20 }),
      });
    });

    await page.route('**/api/replay/pause', route => {
      pauseCalled = true;
      isRunning = false;
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ...mockReplayStatus, running: false, paused: true, total_rows: 20 }),
      });
    });

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    const btnPlay = page.locator('#btn-play');
    const playText = page.locator('#play-text');

    await expect(btnPlay).toBeEnabled();
    await expect(playText).toHaveText('Start Replay');

    // Click play (start)
    await btnPlay.click();
    expect(startCalled).toBe(true);
    await expect(playText).toHaveText('Pause');

    // Click again (pause)
    await btnPlay.click();
    expect(pauseCalled).toBe(true);
    await expect(playText).toHaveText(/Resume Replay|Start Replay/);
  });

  test('live-mode step forward calls /api/replay/step and updates telemetry', async ({ page }) => {
    let stepCalled = false;
    let currentTwinState = { ...mockLowRiskTwin };

    await page.route('**/api/replay-history*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayHistory),
    }));

    await page.route('**/api/twin-state/P001', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(currentTwinState),
    }));

    await page.route('**/api/system-status', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockSystemStatus),
    }));

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayStatus),
    }));

    await page.route('**/api/replay/step', route => {
      stepCalled = true;
      currentTwinState = {
        ...mockLowRiskTwin,
        current_glucose: 90.0,
        delta_30m: 2.5,
        risk_score: 0.2015,
        source_timestamp: "2020-11-02 11:55:00"
      };
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...mockReplayStatus,
          current_index: 1,
          total_rows: 3,
          last_record: {
            current_glucose: 90.0,
            delta_30m: 2.5,
            risk_score: 0.2015,
            risk_level: "LOW"
          }
        }),
      });
    });

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    const btnNext = page.locator('#btn-next');
    await expect(btnNext).toBeEnabled();

    await btnNext.click();
    expect(stepCalled).toBe(true);

    const cardGlucose = page.locator('#card-glucose');
    await expect(cardGlucose).toHaveText('90.0');
  });

  test('live-mode reset calls /api/replay/reset and rewinds cursor', async ({ page }) => {
    let resetCalled = false;

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

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ...mockReplayStatus, current_index: 2 }),
    }));

    await page.route('**/api/replay/reset', route => {
      resetCalled = true;
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ...mockReplayStatus, current_index: 0, total_rows: 3 }),
      });
    });

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    const btnReset = page.locator('#btn-reset');
    await expect(btnReset).toBeEnabled();

    await btnReset.click();
    expect(resetCalled).toBe(true);
  });

  test('unsupported live-mode controls remain disabled', async ({ page }) => {
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

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayStatus),
    }));

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    const btnPrev = page.locator('#btn-prev');
    await expect(btnPrev).toBeDisabled();

    const slider = page.locator('#step-slider');
    await expect(slider).toBeDisabled();
  });

  test('automatic polling updates displayed state when new record arrives', async ({ page }) => {
    let pollCount = 0;
    let twinState = { ...mockLowRiskTwin };

    await page.route('**/api/replay-history*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayHistory),
    }));

    await page.route('**/api/twin-state/P001', route => {
      pollCount++;
      if (pollCount > 1) {
        twinState = {
          ...mockLowRiskTwin,
          current_glucose: 95.0,
          source_timestamp: "2020-11-02 12:10:00"
        };
      }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(twinState),
      });
    });

    await page.route('**/api/system-status', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockSystemStatus),
    }));

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...mockReplayStatus,
        current_index: pollCount > 1 ? 2 : 0
      }),
    }));

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    const cardGlucose = page.locator('#card-glucose');
    await expect(cardGlucose).toHaveText('88.2');

    // Wait for automatic poll cycle (2.5s) to pick up new record
    await expect(cardGlucose).toHaveText('95.0', { timeout: 8000 });
  });

  test('polling error does not erase last valid telemetry', async ({ page }) => {
    let twinPollCount = 0;

    await page.route('**/api/replay-history*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayHistory),
    }));

    await page.route('**/api/twin-state/P001', route => {
      twinPollCount++;
      if (twinPollCount > 1) {
        return route.fulfill({ status: 502, contentType: 'application/json', body: JSON.stringify({ detail: "Azure timeout" }) });
      }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockLowRiskTwin),
      });
    });

    await page.route('**/api/system-status', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockSystemStatus),
    }));

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockReplayStatus),
    }));

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    const cardGlucose = page.locator('#card-glucose');
    await expect(cardGlucose).toHaveText('88.2');

    // Wait past the polling interval to ensure 502 occurs
    await page.waitForTimeout(3000);

    // Telemetry must still show 88.2, not Unavailable or blank
    await expect(cardGlucose).toHaveText('88.2');
  });

  test('twenty available replay records reach chart and render all 20 data points', async ({ page }) => {
    const mock20Rows = Array.from({ length: 20 }, (_, i) => ({
      time: `${11 + Math.floor(i / 4)}:${String((i % 4) * 15).padStart(2, '0')}`,
      timestamp: `2020-11-02T${String(11 + Math.floor(i / 4)).padStart(2, '0')}:${String((i % 4) * 15).padStart(2, '0')}:00`,
      glucose: 85.0 + i * 4.0,
      delta30m: 1.5 + i * 0.5,
      risk: 0.15 + i * 0.03,
      trend: i > 5 ? "Rising" : "Stable",
      level: (0.15 + i * 0.03 >= 0.57) ? "HIGH" : "LOW",
      predicted_event: (0.15 + i * 0.03 >= 0.57) ? "Glucose spike" : "No glucose spike"
    }));

    let requestedLimit = null;
    await page.route('**/api/replay-history*', route => {
      const url = new URL(route.request().url());
      requestedLimit = url.searchParams.get('limit');
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          patient_id: "2001",
          session_id: "2001_0_20201102",
          count: 20,
          rows: mock20Rows
        })
      });
    });

    await page.route('**/api/twin-state/P001', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...mockLowRiskTwin,
        source_timestamp: mock20Rows[0].timestamp
      })
    }));

    await page.route('**/api/system-status', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockSystemStatus)
    }));

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...mockReplayStatus,
        total_rows: 20,
        current_index: 1
      })
    }));

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    // Both charts must render all 20 data points
    const glucosePoints = page.locator('#glucose-svg .chart-point');
    await expect(glucosePoints).toHaveCount(20);

    // Verify limit query parameter was requested as 20
    expect(requestedLimit).toBe('20');

    const riskPoints = page.locator('#risk-svg .chart-point');
    await expect(riskPoints).toHaveCount(20);

    // Verify slider and counter represent 20 records
    const slider = page.locator('#step-slider');
    await expect(slider).toHaveAttribute('max', '19');

    const stepCounter = page.locator('#step-counter');
    await expect(stepCounter).toHaveText('1 / 20');
  });

  test('chart gracefully handles and renders when only 10 replay records are available', async ({ page }) => {
    const mock10Rows = Array.from({ length: 10 }, (_, i) => ({
      time: `${11 + Math.floor(i / 4)}:${String((i % 4) * 15).padStart(2, '0')}`,
      timestamp: `2020-11-02T${String(11 + Math.floor(i / 4)).padStart(2, '0')}:${String((i % 4) * 15).padStart(2, '0')}:00`,
      glucose: 85.0 + i * 3.0,
      delta30m: 1.0,
      risk: 0.18,
      trend: "Stable",
      level: "LOW",
      predicted_event: "No glucose spike"
    }));

    await page.route('**/api/replay-history*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        patient_id: "2001",
        session_id: "2001_0_20201102",
        count: 10,
        rows: mock10Rows
      })
    }));

    await page.route('**/api/twin-state/P001', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...mockLowRiskTwin,
        source_timestamp: mock10Rows[0].timestamp
      })
    }));

    await page.route('**/api/system-status', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockSystemStatus)
    }));

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...mockReplayStatus,
        total_rows: 10,
        current_index: 1
      })
    }));

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    // Both charts must render all 10 data points
    const glucosePoints = page.locator('#glucose-svg .chart-point');
    await expect(glucosePoints).toHaveCount(10);

    const riskPoints = page.locator('#risk-svg .chart-point');
    await expect(riskPoints).toHaveCount(10);

    // Verify slider and counter represent 10 records
    const slider = page.locator('#step-slider');
    await expect(slider).toHaveAttribute('max', '9');

    const stepCounter = page.locator('#step-counter');
    await expect(stepCounter).toHaveText('1 / 10');
  });

  test('graph updates cursor and points beyond record 10 after replay step', async ({ page }) => {
    const mock20Rows = Array.from({ length: 20 }, (_, i) => ({
      time: `${11 + Math.floor(i / 4)}:${String((i % 4) * 15).padStart(2, '0')}`,
      timestamp: `2020-11-02T${String(11 + Math.floor(i / 4)).padStart(2, '0')}:${String((i % 4) * 15).padStart(2, '0')}:00`,
      glucose: 90.0 + i * 3.5,
      delta30m: 2.0,
      risk: 0.15 + i * 0.02,
      trend: "Rising",
      level: (0.15 + i * 0.02 >= 0.57) ? "HIGH" : "LOW",
      predicted_event: (0.15 + i * 0.02 >= 0.57) ? "Glucose spike" : "No glucose spike"
    }));

    let currentStepIndex = 1;
    await page.route('**/api/replay-history*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        patient_id: "2001",
        session_id: "2001_0_20201102",
        count: 20,
        rows: mock20Rows
      })
    }));

    await page.route('**/api/twin-state/P001', route => {
      const row = mock20Rows[currentStepIndex - 1];
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          twin_id: "P001",
          patient_id: "2001",
          current_glucose: row.glucose,
          delta_30m: row.delta30m,
          glucose_trend: row.trend,
          risk_score: row.risk,
          risk_level: row.level,
          predicted_event: row.predicted_event,
          prediction_window: "Next 2 hours",
          last_updated: "2026-10-18T12:00:00+00:00",
          source_timestamp: row.timestamp,
          data_provenance: "Simulated replay of historical CGM data"
        })
      });
    });

    await page.route('**/api/system-status', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockSystemStatus)
    }));

    await page.route('**/api/replay/status*', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...mockReplayStatus,
        total_rows: 20,
        current_index: currentStepIndex
      })
    }));

    await page.route('**/api/replay/step', route => {
      currentStepIndex = 15; // advance past record 10
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...mockReplayStatus,
          total_rows: 20,
          current_index: currentStepIndex
        })
      });
    });

    await page.goto(`${DASHBOARD_FILE_URL}?mode=live`, { waitUntil: 'domcontentloaded' });

    const stepCounter = page.locator('#step-counter');
    await expect(stepCounter).toHaveText('1 / 20');

    // Click step button to jump to record 15
    const btnNext = page.locator('#btn-next');
    await btnNext.click();

    // Verify step counter advanced past 10 to 15 / 20
    await expect(stepCounter).toHaveText('15 / 20');

    // Slider must reflect index 14
    const slider = page.locator('#step-slider');
    await expect(slider).toHaveValue('14');

    // Active chart halo must be visible
    const activeHalo = page.locator('#glucose-svg .halo');
    await expect(activeHalo).toBeVisible();
  });

});
