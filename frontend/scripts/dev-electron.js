#!/usr/bin/env node
/**
 * Flexible dev launcher for Electron without hard-coded port.
 * Usage:
 *   node scripts/dev-electron.js            # defaults (3000)
 *   FRONTEND_PORT=4000 node scripts/dev-electron.js
 *   PORT=5000 node scripts/dev-electron.js  # falls back if FRONTEND_PORT unset
 */
const { spawn } = require('child_process');
const http = require('http');

const desiredPort = parseInt(process.env.FRONTEND_PORT || process.env.PORT || '3000', 10);
const waitTimeoutMs = 5 * 60 * 1000; // 5 minutes max
const pollIntervalMs = 1500;

function checkServer(port) {
  return new Promise((resolve, reject) => {
    const req = http.get({ host: '127.0.0.1', port, path: '/', timeout: 2000 }, (res) => {
      res.resume();
      resolve(true);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

async function waitForServer(port) {
  const start = Date.now();
  process.stdout.write(`Waiting for frontend dev server on port ${port}`);
  while (Date.now() - start < waitTimeoutMs) {
    const up = await checkServer(port);
    if (up) {
      console.log(`\n✅ Frontend dev server detected at http://localhost:${port}`);
      return;
    }
    process.stdout.write('.')
    await new Promise(r => setTimeout(r, pollIntervalMs));
  }
  console.error(`\n❌ Timeout: frontend dev server not reachable on port ${port}`);
  process.exit(1);
}

(async () => {
  await waitForServer(desiredPort);
  const env = { ...process.env, FRONTEND_PORT: String(desiredPort) };
  const child = spawn('electron', ['.'], { stdio: 'inherit', env });
  child.on('exit', (code) => process.exit(code || 0));
})();
