import React, { useState } from 'react';
import { X, ShieldAlert, Check } from 'lucide-react';

export default function OverrideModal({ isOpen, onClose, currentOverride, onSaveOverride, apiKey }) {
  const [selectedRegime, setSelectedRegime] = useState(currentOverride || 'CLEAR');
  const [localApiKey, setLocalApiKey] = useState(apiKey || '');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await onSaveOverride(selectedRegime, localApiKey);
      onClose();
    } catch (err) {
      setError(err.message || 'Failed to update regime override');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldAlert size={18} style={{ color: 'var(--amber)' }} />
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)' }}>
              Market Regime Manual Override
            </h3>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
          >
            <X size={18} />
          </button>
        </div>

        <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5', marginBottom: '20px' }}>
          Overriding bypasses automated market breadth and ADX detection. Selecting <strong>CLEAR / AUTO</strong> restores engine self-classification.
        </p>

        {error && (
          <div style={{ background: 'var(--red-dim)', border: '1px solid var(--red-border)', padding: '10px 14px', borderRadius: 'var(--radius-sm)', color: 'var(--red)', fontSize: '12px', marginBottom: '16px' }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <label className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            TARGET REGIME:
            <select
              value={selectedRegime}
              onChange={(e) => setSelectedRegime(e.target.value)}
              style={{ marginBottom: '16px' }}
            >
              <option value="CLEAR">AUTOMATIC (Engine Managed)</option>
              <option value="TREND_UP">TREND_UP (Aggressive Momentum)</option>
              <option value="TREND_DOWN">TREND_DOWN (Defensive / Short Only)</option>
              <option value="RANGE">RANGE (Mean Reversion)</option>
              <option value="EXPANSION">EXPANSION (Breakout Expansion)</option>
              <option value="PANIC">PANIC (Strict Trading Veto)</option>
            </select>
          </label>

          <label className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            X-API-KEY (ADMIN AUTH):
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
            <button type="submit" className="btn primary" disabled={submitting}>
              <Check size={14} />
              <span>{submitting ? 'Applying...' : 'Confirm Override'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
