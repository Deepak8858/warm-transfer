import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: { timeout: 10_000 },
  fullyParallel: true,
  retries: 0,
  workers: 2,
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: [
    {
      command: 'ENABLE_AGENT_MOCK=0 FORCE_MOCK_GROQ=1 python3 -m uvicorn main:app --host 127.0.0.1 --port 8000',
      port: 8000,
      reuseExistingServer: true,
      timeout: 60_000,
      cwd: '../backend',
    },
    {
      command: 'npm run dev',
      port: 3000,
      reuseExistingServer: true,
      timeout: 60_000,
      cwd: __dirname,
    },
  ],
  projects: [
    {
      name: 'Chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
