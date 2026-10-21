import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyScanState, fetchResource, probabilityEvidence, snapshotFreshness } from '../src/evidence.js';

const now = Date.parse('2026-10-20T10:00:00Z');

test('unvalidated and legacy outputs do not claim calibrated probabilities', () => {
  assert.equal(probabilityEvidence({ prob_win: .63, probability_status: 'HEURISTIC_UNVALIDATED' }).text, '0.63');
  assert.equal(probabilityEvidence({ prob_win: .63 }).label, 'Unverified model output');
  assert.equal(probabilityEvidence({ prob_win: .63, probability_status: 'NEW_UNKNOWN_STATUS' }).text, '0.63');
  assert.equal(probabilityEvidence({ prob_win: .63, probability_status: 'CALIBRATED' }).text, '63%');
  assert.equal(probabilityEvidence({ prob_win: 0 }).text, '0.00');
});

test('missing, non-numeric and out-of-range probabilities stay unavailable', () => {
  for (const value of [undefined, null, NaN, Infinity, -1, 1.01, '0.63']) {
    assert.equal(probabilityEvidence({ prob_win: value }).value, null);
  }
  for (const status of ['UNUSED', 'UNAVAILABLE', 'NOT_APPLICABLE']) {
    assert.equal(probabilityEvidence({ prob_win: .8, probability_status: status }).text, 'Unavailable');
  }
});

test('snapshot freshness depends on the scan time, with explicit timezone and future checks', () => {
  assert.equal(snapshotFreshness('2026-10-20T09:59:00Z', now).state, 'recent');
  assert.equal(snapshotFreshness('2026-10-20T09:45:00Z', now).state, 'recent');
  assert.equal(snapshotFreshness('2026-10-20T09:44:59Z', now).state, 'stale');
  assert.equal(snapshotFreshness('2026-10-20T15:29:00+05:30', now).state, 'recent');
  assert.equal(snapshotFreshness('2026-10-20T10:02:00Z', now).state, 'future');
  for (const value of [null, '', 'invalidZ', '2026-10-20T10:00:00']) {
    assert.equal(snapshotFreshness(value, now).state, 'unknown');
  }
});

test('empty results distinguish initial load, failed refresh, scan failure, stale data and completed scans', () => {
  const scan = { last_scan_time: '2026-10-20T09:59:00Z' };
  assert.equal(emptyScanState('portfolio', null).title, 'Waiting for scan data');
  assert.equal(emptyScanState('portfolio', scan, { error: 'HTTP 401' }, now).title, 'Scan data unavailable');
  assert.equal(emptyScanState('portfolio', { ...scan, scan_error: 'Missing bars' }, {}, now).message, 'Missing bars');
  assert.equal(emptyScanState('portfolio', scan, {}, now + 3600000).title, 'No current scan results');
  assert.equal(emptyScanState('portfolio', scan, {}, now).title, 'No picks returned by this scan');
  assert.equal(emptyScanState('candidates', scan, {}, now).title, 'No candidates returned by this scan');
});

test('fetch failures stay failures, including auth, network and invalid JSON bodies', async () => {
  const controller = new AbortController();
  const headers = { 'X-API-Key': 'test-key' };
  const fetcher = async (url, options) => {
    assert.equal(url, '/api/evidence');
    assert.equal(options.headers, headers);
    assert.equal(options.signal, controller.signal);
    return { ok: true, json: async () => ({ status: 'MISSING' }) };
  };
  assert.deepEqual(await fetchResource('/api/evidence', headers, controller.signal, fetcher), { status: 'MISSING' });
  for (const status of [401, 403, 500]) {
    await assert.rejects(fetchResource('/api/scan', {}, undefined, async () => ({ ok: false, status })),
      status === 401 ? /Authorization failed/ : new RegExp(`HTTP ${status}`));
  }
  await assert.rejects(fetchResource('/api/scan', {}, undefined, async () => { throw new Error('Network offline'); }), /Network offline/);
  for (const data of [null, [], 'not data']) {
    await assert.rejects(fetchResource('/api/scan', {}, undefined, async () => ({ ok: true, json: async () => data })), /Invalid response/);
  }
});
