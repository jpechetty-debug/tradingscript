import React from 'react';
import { FEED_LABELS, snapshotFreshness } from '../evidence';

export default function EvidencePanel({ scan, audit, requests = {}, now = Date.now() }) {
  const freshness = snapshotFreshness(scan?.last_scan_time, now);
  const errors = Object.entries(requests).filter(([, request]) => request.error);
  const feedEntries = Object.entries(audit?.feeds || {});
  const missing = feedEntries.filter(([, status]) => !String(status).startsWith('VALID_'));
  const membership = audit?.membership_status_sessions || {};
  const overlaps = audit?.membership_integrity_sessions?.OVERLAPPING_INTERVALS || 0;
  const hasAudit = audit?.start && audit?.end;
  return (
    <section className="evidence-panel card" aria-label="Strategy and evidence status">
      <div className="evidence-grid">
        <div>
          <h2>Scan snapshot</h2>
          <p className={`evidence-status ${freshness.state === 'recent' ? 'cyan' : 'amber'}`}>{freshness.label}</p>
          <p className="evidence-detail">{freshness.timestamp !== null
            ? `${new Date(freshness.timestamp).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })} IST`
            : 'A successful refresh does not establish a current scan.'}</p>
          <p className="evidence-detail">Snapshots older than 15 minutes are flagged. This measures scan age, not market-bar freshness.</p>
          {scan?.scan_error && <p className="evidence-status amber" role="alert">Scan failed: {scan.scan_error}</p>}
        </div>
        <div>
          <h2>Model estimates</h2>
          <p className="evidence-status amber">Recommendations require evidence</p>
          <p className="evidence-detail">Heuristic and unverified outputs are shown as scores. Only explicitly calibrated estimates use percentages.</p>
          <p className="evidence-detail">Picks are recommendations. Fill history is recorded separately.</p>
        </div>
        <div>
          <h2>Historical research inputs</h2>
          {hasAudit ? <>
            <p className="evidence-status amber">{missing.length ? 'Research inputs incomplete' : 'Local evidence · verification pending'}</p>
            <p className="evidence-detail">Audit period: {audit.start} – {audit.end}. These findings describe research snapshots, not the current scan's required feeds.</p>
            {missing.length > 0 && <p className="evidence-detail">Missing or invalid feeds: {missing.map(([name]) => FEED_LABELS[name] || name).join(', ')}.</p>}
            {membership.UNAVAILABLE_STALE_MEMBERSHIP > 0 && <p className="evidence-detail">Membership stale in {membership.UNAVAILABLE_STALE_MEMBERSHIP.toLocaleString('en-IN')} audited sessions.</p>}
            {overlaps > 0 && <p className="evidence-detail">Overlapping membership intervals reported.</p>}
            {audit.missing_benchmark_sessions?.length > 0 && <p className="evidence-detail">Benchmark bars missing in {audit.missing_benchmark_sessions.length} audited sessions.</p>}
            {audit.price_asof_status?.startsWith('UNAVAILABLE') && <p className="evidence-detail">Historical price-ingestion timing is unavailable.</p>}
            {audit.unreadable_reports > 0 && <p className="evidence-detail">Some audit reports could not be read.</p>}
          </> : <>
            <p className="evidence-status amber">Research coverage not reported</p>
            <p className="evidence-detail">{audit?.status === 'UNAVAILABLE_INVALID_AUDIT'
              ? 'The saved audit could not be validated.' : 'No usable research coverage audit has been received.'}</p>
          </>}
        </div>
      </div>
      {errors.length > 0 && <div className="refresh-errors" role="alert">
        <strong>Refresh incomplete · retained data may be old</strong>
        <ul>{errors.map(([name, request]) => <li key={name}>{name}: {request.error}</li>)}</ul>
      </div>}
    </section>
  );
}
