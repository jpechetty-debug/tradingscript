import React from 'react';
import { History, DollarSign, TrendingUp, TrendingDown } from 'lucide-react';

export default function TradeLogHistory({ tradesData }) {
  const trades = tradesData?.trades || [];

  if (trades.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
        <History size={32} style={{ color: 'var(--text-faint)', marginBottom: '12px' }} />
        <h3 style={{ fontSize: '16px', color: 'var(--text-muted)' }}>No Trade Logs Recorded</h3>
        <p style={{ fontSize: '12px', color: 'var(--text-faint)', marginTop: '6px' }}>
          Historical trade logs from SQLite persistence will show up here as trades close or simulate.
        </p>
      </div>
    );
  }

  // Calculate cumulative stats
  const totalTrades = trades.length;
  const wins = trades.filter(t => (t.pnl || 0) > 0).length;
  const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(0) : '0';
  const totalPnL = trades.reduce((acc, t) => acc + (t.pnl || 0), 0);

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', marginBottom: '20px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
            Trade Execution Log (SQLite Persistence)
          </h2>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
            ACID transaction logs with factor attribution for Platt & Calibrator loop
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="metric-pill cyan">
            <span>TOTAL: {totalTrades}</span>
          </div>
          <div className={`metric-pill ${Number(winRate) >= 50 ? 'green' : 'amber'}`}>
            <span>WIN RATE: {winRate}%</span>
          </div>
          <div className={`metric-pill ${totalPnL >= 0 ? 'green' : 'red'}`}>
            <span>PNL: ₹{totalPnL.toLocaleString('en-IN', { minimumFractionDigits: 2 })}</span>
          </div>
        </div>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Ticker</th>
              <th>Direction</th>
              <th>Realized PnL</th>
              <th>Composite Score</th>
            </tr>
          </thead>
          <tbody>
            {trades.slice(-50).reverse().map((trade, idx) => {
              const isProfit = (trade.pnl || 0) >= 0;
              return (
                <tr key={idx}>
                  <td className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    {trade.ts ? new Date(trade.ts).toLocaleString() : 'N/A'}
                  </td>
                  <td className="mono" style={{ fontWeight: '700' }}>{trade.ticker || 'UNKNOWN'}</td>
                  <td>
                    <span className={`badge ${trade.direction === 'SHORT' ? 'badge-short' : 'badge-long'}`}>
                      {trade.direction || 'LONG'}
                    </span>
                  </td>
                  <td className="mono" style={{ fontWeight: '700', color: isProfit ? 'var(--green)' : 'var(--red)' }}>
                    {isProfit ? '+' : ''}₹{(trade.pnl || 0).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                  </td>
                  <td className="mono" style={{ color: 'var(--text-muted)' }}>
                    {trade.composite !== undefined ? trade.composite.toFixed(3) : 'N/A'}
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
