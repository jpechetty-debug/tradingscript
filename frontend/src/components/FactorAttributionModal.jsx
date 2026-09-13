import React from 'react';
import { X, TrendingUp, BarChart2, Zap, Activity, Award, ArrowUpRight, ArrowDownRight, Layers, ShieldCheck } from 'lucide-react';

const FACTOR_METRICS = [
  { key: 'trend', label: 'Trend Alignment', icon: TrendingUp, desc: 'Multi-timeframe EMA Stack & Supertrend regime' },
  { key: 'rs', label: 'Relative Strength (RS)', icon: Award, desc: 'Alpha velocity vs Benchmark (Nifty 50)' },
  { key: 'momentum', label: 'Momentum Velocity', icon: Zap, desc: 'MACD histogram & RSI momentum slope' },
  { key: 'volume', label: 'Volume Flow', icon: BarChart2, desc: 'Relative Volume (RVol) & True Volume Profile' },
  { key: 'volatility', label: 'Volatility State', icon: Activity, desc: 'ATR contraction & Bollinger Band squeeze' },
  { key: 'quality', label: 'Price Action Quality', icon: ShieldCheck, desc: 'Low noise drift & directional efficiency' },
  { key: 'breakout', label: 'Breakout Expansion', icon: Layers, desc: 'Donchian channel & range expansion pressure' },
];

export default function FactorAttributionModal({ isOpen, onClose, tickerData }) {
  if (!isOpen || !tickerData) return null;

  const factors = tickerData.factors || {};
  const isLong = tickerData.direction === 'LONG';
  const probPct = ((tickerData.prob_win || 0) * 100).toFixed(0);
  const compositeScore = (tickerData.composite ?? tickerData.composite_score ?? 0).toFixed(2);

  const getScoreColor = (val) => {
    if (val >= 0.7) return 'var(--green)';
    if (val >= 0.45) return 'var(--cyan)';
    if (val >= 0.3) return 'var(--amber)';
    return 'var(--red)';
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div 
        className="modal-content" 
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '640px', maxHeight: '90vh', overflowY: 'auto' }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
              <span className="mono" style={{ fontSize: '20px', fontWeight: '800', color: 'var(--text-main)' }}>
                {tickerData.ticker}
              </span>
              <span className={`badge ${isLong ? 'badge-long' : 'badge-short'}`}>
                {isLong ? <ArrowUpRight size={12} style={{ verticalAlign: 'middle' }} /> : <ArrowDownRight size={12} style={{ verticalAlign: 'middle' }} />}
                {' '}{tickerData.direction}
              </span>
              <span className="metric-pill green" style={{ fontSize: '10px' }}>
                P(WIN): {probPct}%
              </span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              {tickerData.sector ? `${tickerData.sector} • ` : ''}7-Factor Quantitative Signal Model Attribution
            </p>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '4px' }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Top Summary Banner */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '12px',
          background: 'var(--surface2)',
          padding: '14px',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border)',
          marginBottom: '20px',
        }}>
          <div>
            <div className="mono" style={{ fontSize: '10px', color: 'var(--text-faint)', textTransform: 'uppercase' }}>Composite Score</div>
            <div className="mono" style={{ fontSize: '18px', fontWeight: '800', color: Number(compositeScore) >= 0 ? 'var(--green)' : 'var(--red)', marginTop: '2px' }}>
              {compositeScore}
            </div>
          </div>
          <div>
            <div className="mono" style={{ fontSize: '10px', color: 'var(--text-faint)', textTransform: 'uppercase' }}>Expectancy E(R)</div>
            <div className="mono" style={{ fontSize: '18px', fontWeight: '800', color: 'var(--cyan)', marginTop: '2px' }}>
              {(tickerData.expectancy_r || 0).toFixed(2)}R
            </div>
          </div>
          <div>
            <div className="mono" style={{ fontSize: '10px', color: 'var(--text-faint)', textTransform: 'uppercase' }}>Risk/Reward T1</div>
            <div className="mono" style={{ fontSize: '18px', fontWeight: '800', color: 'var(--text-main)', marginTop: '2px' }}>
              {(tickerData.rr_t1 || 0).toFixed(1)}x
            </div>
          </div>
        </div>

        {/* 7 Factor List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '24px' }}>
          <div className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
            FACTOR SCORE BREAKDOWN (IC-WEIGHTED)
          </div>

          {FACTOR_METRICS.map(({ key, label, icon: Icon, desc }) => {
            const rawScore = typeof factors[key] === 'number' ? factors[key] : 0;
            const scorePct = Math.round(rawScore * 100);
            const scoreColor = getScoreColor(rawScore);

            return (
              <div 
                key={key}
                style={{
                  background: 'var(--surface2)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '12px 14px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Icon size={14} style={{ color: scoreColor }} />
                    <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-main)' }}>{label}</span>
                  </div>
                  <div className="mono" style={{ fontSize: '13px', fontWeight: '700', color: scoreColor }}>
                    {scorePct}%
                  </div>
                </div>

                <div style={{
                  width: '100%',
                  height: '6px',
                  background: 'rgba(255, 255, 255, 0.06)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                  marginBottom: '6px',
                }}>
                  <div style={{
                    width: `${scorePct}%`,
                    height: '100%',
                    background: scoreColor,
                    borderRadius: '3px',
                    transition: 'width 0.4s ease',
                  }} />
                </div>

                <div style={{ fontSize: '11px', color: 'var(--text-faint)', lineHeight: '1.4' }}>
                  {desc}
                </div>
              </div>
            );
          })}
        </div>

        {/* Trade Levels Summary */}
        {(tickerData.entry || tickerData.stop || tickerData.t1) && (
          <div style={{
            background: 'var(--surface3)',
            padding: '12px 16px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '12px',
          }}>
            <div>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Entry: </span>
              <span className="mono" style={{ fontWeight: '700' }}>₹{tickerData.entry?.toFixed(2)}</span>
            </div>
            <div>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Stop: </span>
              <span className="mono" style={{ fontWeight: '700', color: 'var(--red)' }}>₹{tickerData.stop?.toFixed(2)}</span>
            </div>
            <div>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Target 1: </span>
              <span className="mono" style={{ fontWeight: '700', color: 'var(--green)' }}>₹{tickerData.t1?.toFixed(2)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
