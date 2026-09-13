import React from 'react';
import { PieChart, TrendingUp, TrendingDown } from 'lucide-react';

export default function SectorRSGrid({ sectorsData }) {
  if (!sectorsData || !sectorsData.sectors) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
        <PieChart size={32} style={{ color: 'var(--text-faint)', marginBottom: '12px' }} />
        <h3 style={{ fontSize: '16px', color: 'var(--text-muted)' }}>Loading Sector Allocations...</h3>
      </div>
    );
  }

  const { sectors, last_sector_rs = {} } = sectorsData;
  const sectorEntries = Object.keys(sectors).map(sec => ({
    name: sec,
    tickers: sectors[sec],
    rs: last_sector_rs[sec] !== undefined ? last_sector_rs[sec] : 0,
  })).sort((a, b) => b.rs - a.rs);

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
            Sector Relative Strength (RS)
          </h2>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
            Calculated against Nifty 50 benchmark to isolate alpha dispersion
          </p>
        </div>
        <span className="metric-pill cyan">
          {sectorEntries.length} SECTORS TRACKED
        </span>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
        gap: '16px'
      }}>
        {sectorEntries.map((sec, idx) => {
          const isPositive = sec.rs >= 0;
          return (
            <div
              key={sec.name}
              style={{
                background: 'var(--surface2)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>#{idx + 1}</span>
                <span className={`badge ${isPositive ? 'badge-long' : 'badge-short'}`}>
                  {isPositive ? <TrendingUp size={12} style={{ verticalAlign: 'middle' }} /> : <TrendingDown size={12} style={{ verticalAlign: 'middle' }} />}
                  {' '}{sec.rs ? `${sec.rs > 0 ? '+' : ''}${sec.rs.toFixed(2)}%` : '0.00%'}
                </span>
              </div>

              <div>
                <h4 style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-main)', marginBottom: '4px' }}>
                  {sec.name}
                </h4>
                <div className="mono" style={{ fontSize: '11px', color: 'var(--text-faint)' }}>
                  {sec.tickers.length} tickers in coverage
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
