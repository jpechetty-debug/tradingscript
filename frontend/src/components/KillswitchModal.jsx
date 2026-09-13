import React, { useState } from 'react';
import { X, AlertOctagon, RefreshCw, Check } from 'lucide-react';

export default function KillswitchModal({
  isOpen,
  onClose,
  isKilled,
  onToggleKillswitch,
  apiKey,
}) {
  const [localApiKey, setLocalApiKey] = useState(apiKey || '');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await onToggleKillswitch(!isKilled, localApiKey);
      onClose();
    } catch (err) {
      setError(err.message || 'Failed to update killswitch state');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '480px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertOctagon size={20} style={{ color: isKilled ? 'var(--green)' : 'var(--red)' }} />
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
              {isKilled ? 'Reset Emergency Circuit Breaker' : 'Engage Emergency Kill-Switch'}
            </h3>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
          >
            <X size={18} />
          </button>
        </div>

        <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: '1.5', marginBottom: '16px' }}>
          {isKilled
            ? 'Resetting the killswitch will restore automated background scanning, schedule tasks, and allow new trade signals to be generated.'
            : 'CRITICAL: Engaging the emergency kill-switch will immediately abort any running market scan, halt all algorithmic execution, and lock the system into defensive safemode.'}
        </p>

        {error && (
          <div style={{
            background: 'var(--red-dim)',
            border: '1px solid var(--red-border)',
            padding: '10px 14px',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--red)',
            fontSize: '12px',
            marginBottom: '16px'
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <label className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            CONFIRM WITH X-API-KEY:
            <input
              type="password"
              placeholder="Enter server API key..."
              value={localApiKey}
              onChange={(e) => setLocalApiKey(e.target.value)}
              required
              style={{ marginBottom: '24px' }}
            />
          </label>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
            <button type="button" className="btn" onClick={onClose} disabled={submitting}>
              Cancel
            </button>
            <button
              type="submit"
              className="btn"
              style={{
                background: isKilled ? 'var(--green)' : 'var(--red)',
                color: '#000',
                fontWeight: '700',
                border: 'none',
              }}
              disabled={submitting}
            >
              {submitting ? (
                <RefreshCw size={14} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
              ) : isKilled ? (
                <Check size={14} />
              ) : (
                <AlertOctagon size={14} />
              )}
              <span>{submitting ? 'Processing...' : isKilled ? 'Confirm Reset' : 'CONFIRM EMERGENCY HALT'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
