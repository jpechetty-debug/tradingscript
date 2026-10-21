import { after, test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';
import { emptyScanState } from '../src/evidence.js';

// Compile and render the production JSX components using the existing Vite runtime.
const vite = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)), configFile: false,
  server: { middlewareMode: true, watch: null }, appType: 'custom', logLevel: 'silent',
});
after(() => vite.close());
const load = async (name) => (await vite.ssrLoadModule(`/src/components/${name}.jsx`)).default;
const [Portfolio, Candidates, Modal, Evidence, Header] = await Promise.all(
  ['PortfolioTable', 'CandidatesTable', 'FactorAttributionModal', 'EvidencePanel', 'Header'].map(load));
const render = (Component, props) => renderToStaticMarkup(React.createElement(Component, props));
const now = Date.parse('2026-10-20T10:00:00Z');
const row = { ticker: 'EXAMPLE', direction: 'LONG', prob_win: .63,
  probability_status: 'HEURISTIC_UNVALIDATED', strategy_id: 'leader_pullback', strategy_version: 'v1',
  entry: 100, stop: 95, t1: 110, expectancy_r: .5, rr_t1: 2 };
const views = [[Portfolio, { portfolio: [row] }], [Candidates, { candidates: [row] }],
  [Modal, { isOpen: true, tickerData: row }]];

test('both tables and the factor modal label heuristic scores and strategy identity', () => {
  for (const [Component, props] of views) {
    const html = render(Component, props);
    assert.match(html, /Heuristic · unvalidated/);
    assert.match(html, /0\.63/);
    assert.match(html, /Strategy: leader_pullback/);
    assert.match(html, /v1/);
    assert.doesNotMatch(html, /63%|P\(WIN\)|P\(Win\)|calibrated Kelly/);
  }
});

test('legacy metadata stays unverified and absent values never become 0%', () => {
  for (const Component of [Portfolio, Candidates]) {
    const key = Component === Portfolio ? 'portfolio' : 'candidates';
    const legacy = render(Component, { [key]: [{ ...row, strategy_id: null, strategy_version: null, probability_status: undefined }] });
    assert.match(legacy, /Unverified model output/);
    assert.match(legacy, /Strategy: Not reported/);
    const missing = render(Component, { [key]: [{ ...row, prob_win: null }] });
    assert.match(missing, /Probability unavailable/);
    assert.doesNotMatch(missing, />0%<|>63%</);
    const calibrated = render(Component, { [key]: [{ ...row, probability_status: 'CALIBRATED' }] });
    assert.match(calibrated, /63%/);
    assert.match(calibrated, /Calibrated estimate/);
  }
});

test('the evidence panel exposes missing feeds and the historical scope', () => {
  const html = render(Evidence, { now, audit: { start: '2021-10-04', end: '2026-10-01',
    feeds: { earnings: 'MISSING', fundamentals: 'INVALID', indices: 'VALID_RECORDED_TIMESTAMPS_NOT_PROVIDER_VERIFIED' },
    membership_status_sessions: { UNAVAILABLE_STALE_MEMBERSHIP: 1117 },
    membership_integrity_sessions: { OVERLAPPING_INTERVALS: 1222 },
    price_asof_status: 'UNAVAILABLE_PRICE_OBSERVATION_TIMESTAMPS', missing_benchmark_sessions: ['2026-05-01'] } });
  assert.match(html, /Research inputs incomplete/);
  assert.match(html, /Missing or invalid feeds: Earnings, Fundamentals/);
  assert.match(html, /not the current scan&#x27;s required feeds/);
  assert.match(html, /Membership stale in 1,117/);
  assert.match(html, /Overlapping membership/);
  assert.match(html, /price-ingestion timing is unavailable/);
  assert.match(html, /Benchmark bars missing in 1/);
});

test('a successful refresh cannot make an old snapshot fresh', () => {
  const html = render(Evidence, { now, scan: { last_scan_time: '2026-10-20T09:00:00Z' },
    requests: { scan: { lastSuccess: now } } });
  assert.match(html, /Stale snapshot · 60 min/);
  assert.match(html, /not market-bar freshness/);
  assert.match(html, /Research coverage not reported/);
});

test('unknown, invalid and future scan timestamps remain explicit', () => {
  for (const timestamp of [null, 'badZ', '2026-10-20T09:59:00']) {
    assert.match(render(Evidence, { now, scan: { last_scan_time: timestamp } }), /Snapshot time unavailable/);
  }
  assert.match(render(Evidence, { now, scan: { last_scan_time: '2026-10-20T10:02:00Z' } }), /ahead of this device/);
  assert.match(render(Evidence, { now, audit: { status: 'UNAVAILABLE_INVALID_AUDIT' } }), /saved audit could not be validated/);
});

test('failed refreshes remain visible alongside retained results', () => {
  const html = render(Evidence, { now, requests: {
    scan: { error: 'Authorization failed' }, audit: { error: 'Network offline' } } });
  assert.match(html, /role="alert"/);
  assert.match(html, /retained data may be old/);
  assert.match(html, /scan: Authorization failed/);
  assert.match(html, /audit: Network offline/);
  assert.match(render(Evidence, { now, scan: { scan_error: 'Required price bars missing' } }), /Scan failed: Required price bars missing/);
});

test('empty table copy distinguishes loading, failed refreshes, stale scans and true empty selections', () => {
  for (const Component of [Portfolio, Candidates]) {
    const kind = Component === Portfolio ? 'portfolio' : 'candidates';
    for (const [scan, request] of [[null, {}], [null, { error: 'Unauthorized' }],
      [{ last_scan_time: '2026-10-20T09:00:00Z' }, {}], [{ last_scan_time: '2026-10-20T09:59:00Z' }, {}]]) {
      const state = emptyScanState(kind, scan, request, now);
      const html = render(Component, { [kind]: [], emptyState: state });
      assert.ok(html.includes(state.title));
      assert.doesNotMatch(html, /cleared probability thresholds|regime restricts exposure/);
    }
  }
});

test('connection status makes no claim about data freshness or unknown regime state', () => {
  const html = render(Header, { status: null, isSseConnected: true, apiKey: '' });
  assert.match(html, /EVENT STREAM CONNECTED/);
  assert.match(html, /REGIME STATUS UNKNOWN/);
  assert.doesNotMatch(html, /LIVE ADAPTIVE|REAL-TIME SSE/);
});
