export const SNAPSHOT_STALE_MS = 15 * 60 * 1000;

export function probabilityEvidence(item = {}) {
  const status = item.probability_status || 'UNVERIFIED';
  const value = typeof item.prob_win === 'number' && Number.isFinite(item.prob_win)
    && item.prob_win >= 0 && item.prob_win <= 1 ? item.prob_win : null;
  if (value === null || ['UNAVAILABLE', 'UNUSED', 'NOT_APPLICABLE'].includes(status)) {
    return { value: null, text: 'Unavailable', label: 'Probability unavailable', tone: 'muted' };
  }
  if (['CALIBRATED', 'CALIBRATED_OUT_OF_SAMPLE'].includes(status)) {
    return { value, text: `${(value * 100).toFixed(0)}%`, label: 'Calibrated estimate', tone: 'cyan' };
  }
  if (status === 'HEURISTIC_UNVALIDATED') {
    return { value, text: value.toFixed(2), label: 'Heuristic \u00b7 unvalidated', tone: 'amber' };
  }
  return { value, text: value.toFixed(2), label: 'Unverified model output', tone: 'amber' };
}

export function snapshotFreshness(timestamp, now = Date.now()) {
  if (typeof timestamp !== 'string' || !/(Z|[+-]\d{2}:\d{2})$/i.test(timestamp)) {
    return { state: 'unknown', label: 'Snapshot time unavailable', timestamp: null };
  }
  const parsed = Date.parse(timestamp);
  if (!Number.isFinite(parsed)) return { state: 'unknown', label: 'Snapshot time unavailable', timestamp: null };
  if (parsed > now + 60_000) return { state: 'future', label: 'Snapshot time is ahead of this device', timestamp: parsed };
  const minutes = Math.floor(Math.max(0, now - parsed) / 60_000);
  const stale = now - parsed > SNAPSHOT_STALE_MS;
  return { state: stale ? 'stale' : 'recent', label: `${stale ? 'Stale snapshot' : 'Snapshot age'} \u00b7 ${minutes} min`, timestamp: parsed };
}

export function emptyScanState(kind, scan, request = {}, now = Date.now()) {
  if (request.error) return { title: 'Scan data unavailable', message: 'The latest refresh failed. Any retained results belong to an earlier snapshot.' };
  if (scan?.scan_error) return { title: 'Scan failed', message: scan.scan_error };
  if (!scan || !scan.last_scan_time) return { title: 'Waiting for scan data', message: 'No completed snapshot has been received.' };
  const freshness = snapshotFreshness(scan.last_scan_time, now);
  if (freshness.state !== 'recent') return { title: 'No current scan results', message: freshness.label };
  return { title: kind === 'portfolio' ? 'No picks returned by this scan' : 'No candidates returned by this scan',
    message: 'No selection was returned. Research input coverage is reported in the evidence panel.' };
}

export const FEED_LABELS = {
  indices: 'Index history', sectors: 'Sector mappings', surveillance: 'Surveillance',
  earnings: 'Earnings', delivery: 'Delivery', fundamentals: 'Fundamentals', flows: 'Instrument flows',
};

export async function fetchResource(url, headers, signal, fetcher = fetch) {
  const response = await fetcher(url, { headers, signal });
  if (!response.ok) {
    throw new Error(response.status === 401 ? 'Authorization failed' : `Refresh failed (HTTP ${response.status})`);
  }
  const data = await response.json();
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Invalid response');
  return data;
}
