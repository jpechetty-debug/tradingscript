import React, { useState } from 'react';
import { Search, Layers, BarChart2, TrendingUp, TrendingDown, Clock, Zap } from 'lucide-react';
import { getTradeAction, getTradeHorizon } from './PortfolioTable';

export default function CandidatesTable({ candidates, onInspectFactors }) {
  const [search, setSearch] = useState('');
  const [minProb, setMinProb] = useState(0);
  const [tierFilter, setTierFilter] = useState('ALL'); // 'ALL' | 'PRIMARY' | 'WATCHLIST' | 'MEANREV'
  const [actionFilter, setActionFilter] = useState('ALL'); // 'ALL' | 'BUY' | 'SELL'
  const [horizonFilter, setHorizonFilter] = useState('ALL'); // 'ALL' | 'SWING' | 'INTRADAY'

  if (!candidates || candidates.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
        <Layers size={32} style={{ color: 'var(--text-faint)', marginBottom: '12px' }} />
        <h3 style={{ fontSize: '16px', color: 'var(--text-muted)' }}>No Screened Candidates</h3>
        <p style={{ fontSize: '12px', color: 'var(--text-faint)', marginTop: '6px' }}>
          Trigger a scan to populate universe signals and factor evaluations.
        </p>
      </div>
    );
  }

  const primaryCount = candidates.filter(c => !c.is_watchlist).length;
  const watchlistCount = candidates.filter(c => c.is_watchlist).length;
  const meanRevCount = candidates.filter(c => c.reasons && c.reasons.some(r => r.includes('MeanRev'))).length;

  const buyCount = candidates.filter(c => getTradeAction(c) === 'BUY').length;
  const sellCount = candidates.filter(c => getTradeAction(c) === 'SELL').length;

  const swingCount = candidates.filter(c => getTradeHorizon(c) === 'SWING').length;
  const intradayCount = candidates.filter(c => getTradeHorizon(c) === 'INTRADAY').length;

  const filtered = candidates.filter(c => {
    const matchesSearch = c.ticker.toLowerCase().includes(search.toLowerCase()) ||
                          (c.sector && c.sector.toLowerCase().includes(search.toLowerCase()));
    const matchesProb = (c.prob_win || 0) >= minProb / 100;
    const isWatchlist = Boolean(c.is_watchlist);
    const isMeanRev = Boolean(c.reasons && c.reasons.some(r => r.includes('MeanRev')));

    let matchesTier = true;
    if (tierFilter === 'PRIMARY') matchesTier = !isWatchlist;
    if (tierFilter === 'WATCHLIST') matchesTier = isWatchlist;
    if (tierFilter === 'MEANREV') matchesTier = isMeanRev;

    const action = getTradeAction(c);
    const matchesAction = actionFilter === 'ALL' || action === actionFilter;

    const horizon = getTradeHorizon(c);
    const matchesHorizon = horizonFilter === 'ALL' || horizon === horizonFilter;

    return matchesSearch && matchesProb && matchesTier && matchesAction && matchesHorizon;
  });

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', marginBottom: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
              Universe Candidates ({filtered.length} / {candidates.length})
            </h2>
            <div style={{ display: 'flex', gap: '6px' }}>
              <span className="badge badge-primary" style={{ cursor: 'pointer' }} onClick={() => setTierFilter('PRIMARY')}>
                {primaryCount} PRIMARY
              </span>
              <span className="badge badge-watchlist" style={{ cursor: 'pointer' }} onClick={() => setTierFilter('WATCHLIST')}>
                {watchlistCount} WATCHLIST
              </span>
              {meanRevCount > 0 && (
                <span className="badge badge-meanrev" style={{ cursor: 'pointer' }} onClick={() => setTierFilter('MEANREV')}>
                  {meanRevCount} MEAN-REV
                </span>
              )}
            </div>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Screened opportunities classified into BUY / SELL, SWING vs INTRADAY, and conviction tiers
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          {/* Action Filter Pills */}
          <div className="filter-btn-group">
            <button
              className={`filter-pill-btn ${actionFilter === 'ALL' ? 'active' : ''}`}
              onClick={() => setActionFilter('ALL')}
            >
              ALL
            </button>
            <button
              className={`filter-pill-btn ${actionFilter === 'BUY' ? 'active-buy' : ''}`}
              onClick={() => setActionFilter('BUY')}
            >
              BUY ({buyCount})
            </button>
            {sellCount > 0 && (
              <button
                className={`filter-pill-btn ${actionFilter === 'SELL' ? 'active-sell' : ''}`}
                onClick={() => setActionFilter('SELL')}
              >
                SELL ({sellCount})
              </button>
            )}
          </div>

          {/* Horizon Filter Pills */}
          <div className="filter-btn-group">
            <button
              className={`filter-pill-btn ${horizonFilter === 'ALL' ? 'active' : ''}`}
              onClick={() => setHorizonFilter('ALL')}
            >
              ALL
            </button>
            <button
              className={`filter-pill-btn ${horizonFilter === 'SWING' ? 'active' : ''}`}
              onClick={() => setHorizonFilter('SWING')}
              title="Positional Swing trades (CNC) - multi-day hold"
            >
              SWING ({swingCount})
            </button>
            <button
              className={`filter-pill-btn ${horizonFilter === 'INTRADAY' ? 'active' : ''}`}
              onClick={() => setHorizonFilter('INTRADAY')}
              title="Intraday trades (MIS) - must square off today"
            >
              INTRADAY ({intradayCount})
            </button>
          </div>

          {/* Search Box */}
          <div style={{ position: 'relative', width: '150px' }}>
            <Search size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Search ticker..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ paddingLeft: '32px', margin: 0 }}
            />
          </div>

          {/* Tier Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Tier:</span>
            <select
              value={tierFilter}
              onChange={(e) => setTierFilter(e.target.value)}
              style={{ width: '115px', margin: 0, padding: '7px 8px' }}
            >
              <option value="ALL">All Tiers ({candidates.length})</option>
              <option value="PRIMARY">Primary ({primaryCount})</option>
              <option value="WATCHLIST">Watchlist ({watchlistCount})</option>
              {meanRevCount > 0 && <option value="MEANREV">Mean-Rev ({meanRevCount})</option>}
            </select>
          </div>

          {/* Min Win Prob Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Min P(Win):</span>
            <select
              value={minProb}
              onChange={(e) => setMinProb(Number(e.target.value))}
              style={{ width: '75px', margin: 0, padding: '7px 8px' }}
            >
              <option value="0">0%</option>
              <option value="45">45%</option>
              <option value="50">50%</option>
              <option value="52">52%</option>
              <option value="55">55%</option>
              <option value="60">60%</option>
            </select>
          </div>
        </div>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Ticker & Tier</th>
              <th>Action</th>
              <th>Horizon</th>
              <th>Sector</th>
              <th>Signals</th>
              <th>Factors</th>
              <th>Composite</th>
              <th>P(Win)</th>
              <th>E(R)</th>
              <th>Entry</th>
              <th>Stop Loss</th>
              <th>Target 1</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((item, idx) => {
              const action = getTradeAction(item);
              const isBuy = action === 'BUY';
              const horizon = getTradeHorizon(item);
              const probPct = ((item.prob_win || 0) * 100).toFixed(0);
              const isWatchlist = Boolean(item.is_watchlist);
              const isMeanRev = Boolean(item.reasons && item.reasons.some(r => r.includes('MeanRev')));
              const displayReasons = (item.reasons || []).filter(r => !r.startsWith('Regime:') && !r.startsWith('RR:') && !r.startsWith('Kurt:'));

              const entry = item.entry || 0;
              const stop = item.stop || 0;
              const t1 = item.t1 || 0;
              const slPct = entry > 0 ? (Math.abs(entry - stop) / entry * 100).toFixed(1) : '0.0';
              const t1Pct = entry > 0 ? (Math.abs(t1 - entry) / entry * 100).toFixed(1) : '0.0';

              return (
                <tr key={item.ticker || idx} style={isWatchlist ? { opacity: 0.92 } : {}}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                      <span className="mono" style={{ fontWeight: '800', fontSize: '13px' }}>{item.ticker}</span>
                      {isWatchlist ? (
                        <span className="badge badge-watchlist" title="P(win) in [0.45, 0.52) - Candidate for monitoring">
                          WATCHLIST
                        </span>
                      ) : (
                        <span className="badge badge-primary" title="P(win) >= 0.52 - High-conviction execution candidate">
                          PRIMARY
                        </span>
                      )}
                      {isMeanRev && (
                        <span className="badge badge-meanrev" title="Mean-reversion trade setup in Range regime">
                          MEAN-REV
                        </span>
                      )}
                    </div>
                  </td>
                  <td>
                    {isBuy ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span className="badge badge-action-buy">
                          <TrendingUp size={12} style={{ strokeWidth: 3 }} />
                          BUY
                        </span>
                        <span className="mono" style={{ fontSize: '9px', color: 'var(--green)', opacity: 0.8, paddingLeft: '3px' }}>
                          LONG
                        </span>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span className="badge badge-action-sell">
                          <TrendingDown size={12} style={{ strokeWidth: 3 }} />
                          SELL
                        </span>
                        <span className="mono" style={{ fontSize: '9px', color: 'var(--red)', opacity: 0.8, paddingLeft: '3px' }}>
                          SHORT
                        </span>
                      </div>
                    )}
                  </td>
                  <td>
                    {horizon === 'SWING' ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span className="badge badge-horizon-swing" title="Multi-day positional swing hold (CNC product)">
                          <Clock size={10} />
                          SWING
                        </span>
                        <span className="mono" style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
                          CNC • 2–5D
                        </span>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span className="badge badge-horizon-intraday" title="Intraday trade (MIS) - square off by 15:15 IST">
                          <Zap size={10} />
                          INTRADAY
                        </span>
                        <span className="mono" style={{ fontSize: '9px', color: 'var(--amber)', opacity: 0.85 }}>
                          MIS • 15:15
                        </span>
                      </div>
                    )}
                  </td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '12px' }}>{item.sector || 'N/A'}</td>
                  <td>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', maxWidth: '180px' }}>
                      {displayReasons.slice(0, 3).map((r, i) => (
                        <span key={i} className="badge badge-tag">{r}</span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <button
                      className="btn"
                      onClick={() => onInspectFactors && onInspectFactors(item)}
                      style={{ padding: '3px 8px', fontSize: '11px' }}
                      title="Inspect 7-Factor attribution breakdown"
                    >
                      <BarChart2 size={12} />
                      <span>Factors</span>
                    </button>
                  </td>
                  <td className="mono" style={{ color: (item.composite_score || 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {(item.composite_score || 0).toFixed(2)}
                  </td>
                  <td className="mono" style={{ fontWeight: '700', color: Number(probPct) >= 52 ? 'var(--green)' : 'var(--amber)' }}>
                    {probPct}%
                  </td>
                  <td className="mono">{(item.expectancy_r || 0).toFixed(2)}R</td>
                  <td className="mono" style={{ fontWeight: '600' }}>
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
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
