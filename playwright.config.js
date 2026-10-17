const { defineConfig } = require('@playwright/test');

/**
 * Playwright Configuration for GlycoTwin browser test suite.
 */
module.exports = defineConfig({
  testDir: './tests/ui',
  timeout: 30000,
  expect: {
    timeout: 10000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    headless: true,
    viewport: { width: 1280, height: 720 },
    actionTimeout: 10000,
    navigationTimeout: 15000,
  },
});
