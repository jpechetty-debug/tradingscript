import React from 'react';
import { Activity, Play, RefreshCw, ShieldAlert, AlertOctagon, Zap, Radio } from 'lucide-react';

export default function Header({
  status,
  isScanning,
  isKilled,
  isSseConnected,
  onTriggerScan,
  onOpenOverrideModal,
  onOpenKillswitchModal,
  apiKey,
  setApiKey,
}) {
  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '20px',
      paddingBottom: '24px',
      marginBottom: '28px',
      borderBottom: '1px solid var(--border)'
    }}>
      <div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontFamily: 'var(--mono)',
          fontSize: '11px',
          letterSpacing: '0.15em',
          color: isKilled ? 'var(--red)' : 'var(--green)',
          textTransform: 'uppercase',
          marginBottom: '6px'
        }}>
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: isKilled ? 'var(--red)' : isScanning ? 'var(--amber)' : 'var(--green)',
            boxShadow: isKilled ? '0 0 10px var(--red)' : isScanning ? '0 0 8px var(--amber)' : '0 0 8px var(--green)',
            display: 'inline-block'
          }} />
          Sovereign Engine v{status?.version || '14.6'} {isKilled && '• [EMERGENCY HALT]'}
        </div>
        <h1 style={{
          fontSize: '26px',
          fontWeight: '800',
          letterSpacing: '-0.02em',
          color: 'var(--text-main)'
        }}>
          Institutional <span style={{ color: isKilled ? 'var(--red)' : 'var(--green)' }}>Market Scanner</span>
        </h1>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
        {/* SSE Stream Status Pill */}
        <div className={`metric-pill ${isSseConnected ? 'green' : 'amber'}`}>
          <Radio size={12} style={{ animation: isSseConnected ? 'pulse 2s infinite' : 'none' }} />
          <span>{isSseConnected ? 'REAL-TIME SSE' : 'POLLING'}</span>
        </div>

        {/* Session Pill */}
        <div className="metric-pill cyan">
          <Activity size={13} />
          <span>SESSION: {status?.session || 'UNKNOWN'}</span>
        </div>

        {/* Regime Lock Pill */}
        <div className={`metric-pill ${status?.regime_locked ? 'amber' : 'green'}`}>
          <Zap size={13} />
          <span>{status?.regime_locked ? 'REGIME LOCKED (NOISE WINDOW)' : 'LIVE ADAPTIVE'}</span>
        </div>

        {/* Regime Override Pill if Active */}
        {status?.regime_override && (
          <div className="metric-pill red">
            <ShieldAlert size={13} />
            <span>OVERRIDE: {status.regime_override}</span>
          </div>
        )}

        {/* API Key Input */}
        <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
          <input
            type="password"
            placeholder="X-API-Key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            style={{
              width: '130px',
              padding: '6px 10px',
              fontSize: '11px',
              marginTop: 0,
              background: 'var(--surface2)',
              borderRadius: 'var(--radius-sm)'
            }}
            title="Enter API Key for triggering scans, overrides, and killswitch"
          />
        </div>

        {/* Manual Trigger Scan Button */}
        <button
          className="btn primary"
          onClick={onTriggerScan}
          disabled={isScanning || isKilled}
          style={{ opacity: (isScanning || isKilled) ? 0.6 : 1 }}
        >
          {isScanning ? (
            <>
              <RefreshCw size={14} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
              <span>Scanning...</span>
            </>
          ) : (
            <>
              <Play size={14} />
              <span>Trigger Scan</span>
            </>
          )}
        </button>

        {/* Override Modal Trigger */}
        <button className="btn" onClick={onOpenOverrideModal} disabled={isKilled}>
          <ShieldAlert size={14} />
          <span>Override Regime</span>
        </button>

        {/* Emergency Kill-Switch Button */}
        <button
          className="btn"
          onClick={onOpenKillswitchModal}
          style={{
            background: isKilled ? 'var(--red)' : 'var(--red-dim)',
            borderColor: 'var(--red-border)',
            color: isKilled ? '#fff' : 'var(--red)',
            fontWeight: '700',
          }}
          title={isKilled ? 'Click to reset kill-switch' : 'Click to trigger emergency circuit breaker'}
        >
          <AlertOctagon size={14} />
          <span>{isKilled ? 'KILLSWITCH ENGAGED' : 'Kill-Switch'}</span>
        </button>
      </div>
    </header>
  );
}
