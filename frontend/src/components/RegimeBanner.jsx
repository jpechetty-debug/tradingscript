import React from 'react';
import { TrendingUp, TrendingDown, Activity, AlertTriangle, ShieldCheck, Gauge } from 'lucide-react';

export default function RegimeBanner({ regime }) {
  if (!regime) {
    return (
      <div className="card" style={{ marginBottom: '24px', textAlign: 'center', padding: '30px' }}>
        <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>Awaiting initial scan regime data...</p>
      </div>
    );
  }

  const getRegimeTheme = (type) => {
    switch (type) {
      case 'TREND_UP':
        return { color: 'var(--green)', icon: <TrendingUp size={20} />, label: 'BULLISH TREND' };
      case 'TREND_DOWN':
      case 'PANIC':
        return { color: 'var(--red)', icon: <AlertTriangle size={20} />, label: 'BEARISH / PANIC' };
      case 'EXPANSION':
        return { color: 'var(--amber)', icon: <Activity size={20} />, label: 'VOLATILITY EXPANSION' };
      case 'RANGE':
      default:
        return { color: 'var(--cyan)', icon: <Gauge size={20} />, label: 'RANGE-BOUND' };
    }
  };

  const theme = getRegimeTheme(regime.regime);
  const breadthPct = ((regime.breadth || 0) * 100).toFixed(0);
  const sectorConcPct = ((regime.sector_concentration || 0) * 100).toFixed(0);
  const confidencePct = ((regime.confidence || 0) * 100).toFixed(0);

  return (
    <div
      className="card"
      style={{
        marginBottom: '24px',
        borderLeft: `4px solid ${theme.color}`,
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '24px',
        alignItems: 'center'
      }}
    >
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <span style={{ color: theme.color }}>{theme.icon}</span>
          <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Market Regime Classification
          </span>
        </div>
        <div style={{ fontSize: '22px', fontWeight: '800', color: theme.color, letterSpacing: '-0.01em' }}>
          {regime.label || regime.regime}
        </div>
        <div style={{ marginTop: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
          {regime.confirmed ? (
            <span style={{ color: 'var(--green)', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
              <ShieldCheck size={14} /> Confirmed Regime
            </span>
          ) : (
            <span style={{ color: 'var(--amber)' }}>Confirmation Pending (Bar Threshold)</span>
          )}
        </div>
      </div>

      {/* Regime Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '14px' }}>
        <div>
          <div className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Market Breadth
          </div>
          <div className="mono" style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
            {breadthPct}%{' '}
            <span style={{ fontSize: '11px', color: regime.breadth_delta >= 0 ? 'var(--green)' : 'var(--red)' }}>
              ({regime.breadth_delta >= 0 ? '+' : ''}{(regime.breadth_delta || 0).toFixed(2)})
            </span>
          </div>
        </div>

        <div>
          <div className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Regime Confidence
          </div>
          <div className="mono" style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
            {confidencePct}%
          </div>
        </div>

        <div>
          <div className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            ADX Median
          </div>
          <div className="mono" style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
            {(regime.adx_median || 0).toFixed(1)}
          </div>
        </div>

        <div>
          <div className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            ATR Vol Ratio
          </div>
          <div className="mono" style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
            {(regime.atr_ratio || 0).toFixed(2)}x
          </div>
        </div>
      </div>

      {/* Strategy Guidance */}
      <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: '20px' }}>
        <div className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
          Execution Strategy Directive
        </div>
        <p style={{ fontSize: '13px', lineHeight: '1.4', color: 'var(--text-main)', marginBottom: '8px' }}>
          {regime.strategy_hint || 'Execute normal trend-following factor models.'}
        </p>
        <span className={`badge ${regime.is_tradeable ? 'badge-long' : 'badge-short'}`}>
          {regime.is_tradeable ? 'ACTIVE TRADING PERMITTED' : 'PANIC LOCK — NO NEW POSITIONS'}
        </span>
      </div>
    </div>
  );
}
