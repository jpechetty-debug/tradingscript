import React, { useState } from 'react';
import { ShieldCheck, Crosshair, BarChart2, TrendingUp, TrendingDown, Clock, Zap } from 'lucide-react';

export function getTradeAction(item) {
  if (item.action) return item.action;
  return item.direction === 'LONG' ? 'BUY' : 'SELL';
}

export function getTradeHorizon(item) {
  if (item.trade_horizon) return item.trade_horizon;
  if (item.direction === 'SHORT') return 'INTRADAY';
  const entry = item.entry || 0;
  const stop = item.stop || 0;
  const slPct = entry > 0 ? Math.abs(entry - stop) / entry : 0.05;
  const isMeanRev = Boolean(item.reasons && item.reasons.some(r => r.includes('MeanRev')));
  if (slPct < 0.025 || (isMeanRev && slPct < 0.03)) {
    return 'INTRADAY';
  }
  return 'SWING';
}

export default function PortfolioTable({ portfolio, onInspectFactors }) {
  const [filterAction, setFilterAction] = useState('ALL'); // 'ALL' | 'BUY' | 'SELL'
  const [filterHorizon, setFilterHorizon] = useState('ALL'); // 'ALL' | 'SWING' | 'INTRADAY'

  if (!portfolio || portfolio.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
        <Crosshair size={32} style={{ color: 'var(--text-faint)', marginBottom: '12px' }} />
        <h3 style={{ fontSize: '16px', color: 'var(--text-muted)' }}>No Portfolio Positions Selected</h3>
        <p style={{ fontSize: '12px', color: 'var(--text-faint)', marginTop: '6px' }}>
          Either no candidates cleared probability thresholds or current market regime restricts exposure.
        </p>
      </div>
    );
  }

  const buyCount = portfolio.filter(p => getTradeAction(p) === 'BUY').length;
  const sellCount = portfolio.filter(p => getTradeAction(p) === 'SELL').length;
  const swingCount = portfolio.filter(p => getTradeHorizon(p) === 'SWING').length;
  const intradayCount = portfolio.filter(p => getTradeHorizon(p) === 'INTRADAY').length;

  const filtered = portfolio.filter(item => {
    const action = getTradeAction(item);
    const horizon = getTradeHorizon(item);

    const matchesAction = filterAction === 'ALL' || action === filterAction;
    const matchesHorizon = filterHorizon === 'ALL' || horizon === filterHorizon;
    return matchesAction && matchesHorizon;
  });

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', marginBottom: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
              Optimized Portfolio Picks ({filtered.length} / {portfolio.length})
            </h2>
            <div style={{ display: 'flex', gap: '6px' }}>
              <span className="badge badge-action-buy" style={{ fontSize: '11px', padding: '2px 8px' }}>
                {buyCount} BUY
              </span>
              {sellCount > 0 && (
                <span className="badge badge-action-sell" style={{ fontSize: '11px', padding: '2px 8px' }}>
                  {sellCount} SELL
                </span>
              )}
              <span className="badge badge-horizon-swing">
                {swingCount} SWING
              </span>
              {intradayCount > 0 && (
                <span className="badge badge-horizon-intraday">
                  {intradayCount} INTRADAY
                </span>
              )}
            </div>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            High-conviction allocations with calibrated Kelly sizing, stop-loss distance, and trade horizon
          </p>
        </div>

        {/* Action & Horizon Quick Filters */}
        <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
          <div className="filter-btn-group">
            <button
              className={`filter-pill-btn ${filterAction === 'ALL' ? 'active' : ''}`}
              onClick={() => setFilterAction('ALL')}
            >
              ALL
            </button>
            <button
              className={`filter-pill-btn ${filterAction === 'BUY' ? 'active-buy' : ''}`}
              onClick={() => setFilterAction('BUY')}
            >
              BUY ({buyCount})
            </button>
            {sellCount > 0 && (
              <button
                className={`filter-pill-btn ${filterAction === 'SELL' ? 'active-sell' : ''}`}
                onClick={() => setFilterAction('SELL')}
              >
                SELL ({sellCount})
              </button>
            )}
          </div>

          <div className="filter-btn-group">
            <button
              className={`filter-pill-btn ${filterHorizon === 'ALL' ? 'active' : ''}`}
              onClick={() => setFilterHorizon('ALL')}
            >
              ALL
            </button>
            <button
              className={`filter-pill-btn ${filterHorizon === 'SWING' ? 'active' : ''}`}
              onClick={() => setFilterHorizon('SWING')}
              title="Positional Swing trades (CNC) - 2 to 5 days hold"
            >
              SWING ({swingCount})
            </button>
            <button
              className={`filter-pill-btn ${filterHorizon === 'INTRADAY' ? 'active' : ''}`}
              onClick={() => setFilterHorizon('INTRADAY')}
              title="Intraday trades (MIS) - must square off by 15:15 IST"
            >
              INTRADAY ({intradayCount})
            </button>
          </div>
        </div>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Action</th>
              <th>Horizon</th>
              <th>Factors</th>
              <th>P(Win)</th>
              <th>E(R)</th>
              <th>RR (T1)</th>
              <th>Entry Price</th>
              <th>Stop Loss</th>
              <th>Target 1</th>
              <th>Shares</th>
              <th>Allocated Risk</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((item, idx) => {
              const action = getTradeAction(item);
              const isBuy = action === 'BUY';
              const horizon = getTradeHorizon(item);
              const probPct = ((item.prob_win || 0) * 100).toFixed(0);

              const entry = item.entry || 0;
              const stop = item.stop || 0;
              const t1 = item.t1 || 0;

              const slPct = entry > 0 ? (Math.abs(entry - stop) / entry * 100).toFixed(1) : '0.0';
              const t1Pct = entry > 0 ? (Math.abs(t1 - entry) / entry * 100).toFixed(1) : '0.0';

              return (
                <tr key={item.ticker || idx}>
                  <td>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                      <span className="mono" style={{ fontWeight: '800', fontSize: '14px', color: 'var(--text-main)' }}>
                        {item.ticker}
                      </span>
                      <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                        {item.sector || 'NSE'}
                      </span>
                    </div>
                  </td>
                  <td>
                    {isBuy ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                        <span className="badge badge-action-buy">
                          <TrendingUp size={13} style={{ strokeWidth: 3 }} />
                          BUY
                        </span>
                        <span className="mono" style={{ fontSize: '10px', color: 'var(--green)', opacity: 0.85, paddingLeft: '4px' }}>
                          LONG
                        </span>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                        <span className="badge badge-action-sell">
                          <TrendingDown size={13} style={{ strokeWidth: 3 }} />
                          SELL
                        </span>
                        <span className="mono" style={{ fontSize: '10px', color: 'var(--red)', opacity: 0.85, paddingLeft: '4px' }}>
                          SHORT
                        </span>
                      </div>
                    )}
                  </td>
                  <td>
                    {horizon === 'SWING' ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span className="badge badge-horizon-swing" title="Multi-day positional swing hold (CNC product)">
                          <Clock size={11} />
                          SWING
                        </span>
                        <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                          CNC • 2–5D
                        </span>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span className="badge badge-horizon-intraday" title="Intraday trade (MIS) - must square off by 15:15 IST">
                          <Zap size={11} />
                          INTRADAY
                        </span>
                        <span className="mono" style={{ fontSize: '10px', color: 'var(--amber)', opacity: 0.9 }}>
                          MIS • 15:15
                        </span>
                      </div>
                    )}
                  </td>
                  <td>
                    <button
                      className="btn"
                      onClick={() => onInspectFactors && onInspectFactors(item)}
                      style={{ padding: '4px 8px', fontSize: '11px' }}
                      title="Inspect 7-Factor attribution breakdown"
                    >
                      <BarChart2 size={12} />
                      <span>Factors</span>
                    </button>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="mono" style={{ fontWeight: '700', color: Number(probPct) >= 55 ? 'var(--green)' : 'var(--text-main)' }}>
                        {probPct}%
                      </span>
                      <div style={{
                        width: '38px',
                        height: '4px',
                        background: 'rgba(255,255,255,0.08)',
                        borderRadius: '2px',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          width: `${probPct}%`,
                          height: '100%',
                          background: item.prob_win >= 0.55 ? 'var(--green)' : 'var(--amber)'
                        }} />
                      </div>
                    </div>
                  </td>
                  <td className="mono" style={{ fontWeight: '600', color: item.expectancy_r >= 0.5 ? 'var(--green)' : 'var(--text-main)' }}>
                    {item.expectancy_r?.toFixed(2)}R
                  </td>
                  <td className="mono" style={{ color: 'var(--cyan)', fontWeight: '600' }}>
                    {item.rr_t1?.toFixed(1)}x
                  </td>
                  <td className="mono" style={{ fontWeight: '700', color: 'var(--text-main)' }}>
                    ₹{item.entry?.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                  </td>
                  <td className="mono">
                    <div style={{ color: 'var(--red)', fontWeight: '600' }}>
                      ₹{item.stop?.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </div>
                    <span className="pct-pill-red">-{slPct}%</span>
                  </td>
                  <td className="mono">
                    <div style={{ color: 'var(--green)', fontWeight: '600' }}>
                      ₹{item.t1?.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </div>
                    <span className="pct-pill-green">+{t1Pct}%</span>
                  </td>
                  <td className="mono" style={{ fontWeight: '700' }}>
                    {item.shares?.toLocaleString('en-IN')}
                  </td>
                  <td className="mono" style={{ color: 'var(--text-main)', fontWeight: '600' }}>
                    ₹{item.risk_inr?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
