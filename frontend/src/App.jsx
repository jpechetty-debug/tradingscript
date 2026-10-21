import React, { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header';
import RegimeBanner from './components/RegimeBanner';
import PortfolioTable from './components/PortfolioTable';
import CandidatesTable from './components/CandidatesTable';
import SectorRSGrid from './components/SectorRSGrid';
import TradeLogHistory from './components/TradeLogHistory';
import OverrideModal from './components/OverrideModal';
import KillswitchModal from './components/KillswitchModal';
import FactorAttributionModal from './components/FactorAttributionModal';
import { Briefcase, Layers, PieChart, History, AlertCircle } from 'lucide-react';
import { streamSse } from './sse';
import EvidencePanel from './components/EvidencePanel';
import { emptyScanState, fetchResource } from './evidence';

const RESOURCE_URLS = { status: '/api/status', scan: '/api/scan', sectors: '/api/sectors', trades: '/api/trades?limit=100', audit: '/api/evidence' };

export default function App() {
  const [status, setStatus] = useState(null);
  const [scanData, setScanData] = useState(null);
  const [sectorsData, setSectorsData] = useState(null);
  const [tradesData, setTradesData] = useState(null);
  const [auditData, setAuditData] = useState(null);
  const [requests, setRequests] = useState({});
  const [now, setNow] = useState(Date.now);
  const controllers = useRef({});
  const requestEpoch = useRef(0);
  const batchRef = useRef(null);
  const [activeTab, setActiveTab] = useState('portfolio');
  const [isOverrideModalOpen, setIsOverrideModalOpen] = useState(false);
  const [isKillswitchModalOpen, setIsKillswitchModalOpen] = useState(false);
  const [selectedTickerFactors, setSelectedTickerFactors] = useState(null);
  const [isSseConnected, setIsSseConnected] = useState(false);
  const [errorToast, setErrorToast] = useState(null);

  // Store API Key in sessionStorage (defaults to local dev key on localhost/127.0.0.1)
  const [apiKey, setApiKey] = useState(() => {
    const saved = sessionStorage.getItem('sovereign_api_key');
    if (saved) return saved;
    if (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
      return 'sovereign-dev-secret-key';
    }
    return '';
  });

  const handleApiKeyChange = (val) => {
    if (val !== apiKey) {
      requestEpoch.current += 1;
      Object.values(controllers.current).forEach(controller => controller.abort());
      controllers.current = {};
      batchRef.current = null;
      setStatus(null);
      setScanData(null);
      setSectorsData(null);
      setTradesData(null);
      setAuditData(null);
      setRequests({});
      setSelectedTickerFactors(null);
      setIsSseConnected(false);
    }
    setApiKey(val);
    sessionStorage.setItem('sovereign_api_key', val);
  };

  const showToast = (msg) => {
    setErrorToast(msg);
    setTimeout(() => setErrorToast(null), 5000);
  };

  const getAuthHeaders = useCallback(() => {
    const keyToSend = apiKey || (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'sovereign-dev-secret-key' : '');
    return keyToSend ? { 'X-API-Key': keyToSend } : {};
  }, [apiKey]);

  const refreshResource = useCallback(async (name) => {
    const epoch = requestEpoch.current;
    controllers.current[name]?.abort();
    const controller = new AbortController();
    controllers.current[name] = controller;
    const setters = { status: setStatus, scan: setScanData, sectors: setSectorsData, trades: setTradesData, audit: setAuditData };
    try {
      const data = await fetchResource(RESOURCE_URLS[name], getAuthHeaders(), controller.signal);
      if (controller.signal.aborted || epoch !== requestEpoch.current) return;
      setters[name](data);
      setRequests(prev => ({ ...prev, [name]: { error: null, lastSuccess: Date.now() } }));
    } catch (error) {
      if (controller.signal.aborted || epoch !== requestEpoch.current) return;
      setRequests(prev => ({ ...prev, [name]: { ...prev[name], error: error.message || 'Refresh failed' } }));
    } finally {
      if (controllers.current[name] === controller) delete controllers.current[name];
    }
  }, [getAuthHeaders]);

  const fetchStatus = useCallback(() => refreshResource('status'), [refreshResource]);
  const fetchScan = useCallback(() => refreshResource('scan'), [refreshResource]);
  const refreshAll = useCallback(async () => {
    if (batchRef.current) return;
    const batch = {};
    batchRef.current = batch;
    try {
      await Promise.all(Object.keys(RESOURCE_URLS).map(refreshResource));
    } finally {
      if (batchRef.current === batch) batchRef.current = null;
    }
  }, [refreshResource]);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 30000);
    return () => {
      clearInterval(timer);
      Object.values(controllers.current).forEach(controller => controller.abort());
      batchRef.current = null;
    };
  }, []);

  // Initial load + interval polling fallback
  useEffect(() => {
    refreshAll();
    const interval = setInterval(refreshAll, 15000);
    return () => clearInterval(interval);
  }, [refreshAll]);

  // Real-Time Server-Sent Events (SSE) Stream
  useEffect(() => {
    let reconnectTimer = null;
    let controller = null;
    let stopped = false;

    const handleMessage = (rawMessage) => {
      if (stopped) return;
      try {
        const payload = JSON.parse(rawMessage);
        const { event, data } = payload;
        if (event === 'connected') {
          setIsSseConnected(true);
          if (data && typeof data.killswitch_active === 'boolean') {
            setStatus(prev => prev ? { ...prev, killswitch_active: data.killswitch_active } : prev);
          }
        } else if (event === 'scan_started') {
          setStatus(prev => prev ? { ...prev, is_scanning: true } : { is_scanning: true });
        } else if (event === 'scan_completed') {
          setStatus(prev => prev ? { ...prev, is_scanning: false } : { is_scanning: false });
          refreshAll();
          showToast(`Scan complete: ${data?.portfolio_count || 0} picks, ${data?.candidates_count || 0} candidates.`);
        } else if (event === 'scan_failed' || event === 'scan_cancelled') {
          setStatus(prev => prev ? { ...prev, is_scanning: false } : { is_scanning: false });
          showToast(`Scan stopped: ${data?.error || 'Unknown error'}`);
        } else if (event === 'killswitch_engaged') {
          setStatus(prev => prev ? { ...prev, killswitch_active: true } : { killswitch_active: true });
        } else if (event === 'killswitch_reset') {
          setStatus(prev => prev ? { ...prev, killswitch_active: false } : { killswitch_active: false });
        }
      } catch (err) {
        console.warn('SSE message parse error:', err);
      }
    };

    const connectSSE = async () => {
      controller = new AbortController();
      try {
        await streamSse('/api/events', {
          headers: getAuthHeaders(),
          signal: controller.signal,
          onMessage: handleMessage,
        });
      } catch (err) {
        if (err.name !== 'AbortError') console.warn('SSE connection error:', err);
      } finally {
        setIsSseConnected(false);
        if (!stopped) reconnectTimer = setTimeout(connectSSE, 5000);
      }
    };

    connectSSE();

    return () => {
      stopped = true;
      if (controller) controller.abort();
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, [getAuthHeaders, refreshAll]);

  // Trigger scan handler
  const handleTriggerScan = async () => {
    const keyToSend = apiKey || (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'sovereign-dev-secret-key' : '');
    try {
      const res = await fetch('/api/scan/trigger', {
        method: 'POST',
        headers: {
          'X-API-Key': keyToSend,
        },
      });
      if (res.status === 401) {
        showToast('Unauthorized: Invalid X-API-Key. Please verify the API_KEY set in your .env or modal.');
        return;
      }
      if (res.status === 403) {
        showToast('Forbidden: Emergency Killswitch is active.');
        return;
      }
      if (res.status === 429) {
        showToast('Rate Limit Exceeded: Please wait before re-triggering.');
        return;
      }
      const data = await res.json();
      if (data.status === 'already_running') {
        showToast('Scan is already in progress.');
      }
      fetchStatus();
    } catch (e) {
      showToast(`Scan Trigger Error: ${e.message}`);
    }
  };

  // Regime override handler
  const handleSaveOverride = async (regimeVal, keyVal) => {
    const res = await fetch('/api/regime/override', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': keyVal,
      },
      body: JSON.stringify({ regime: regimeVal }),
    });

    if (res.status === 401) {
      throw new Error('Unauthorized: Invalid X-API-Key.');
    }
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Failed to apply regime override');
    }
    handleApiKeyChange(keyVal);
    if (keyVal === apiKey) {
      fetchStatus();
      fetchScan();
    }
  };

  // Emergency Killswitch handler
  const handleToggleKillswitch = async (shouldKill, keyVal) => {
    const endpoint = shouldKill ? '/api/killswitch' : '/api/killswitch/reset';
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'X-API-Key': keyVal,
      },
    });

    if (res.status === 401) {
      throw new Error('Unauthorized: Invalid X-API-Key.');
    }
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Failed to update killswitch state');
    }
    handleApiKeyChange(keyVal);
    if (keyVal === apiKey) fetchStatus();
  };

  const isScanning = status?.is_scanning || false;
  const isKilled = Boolean(status?.killswitch_active);

  return (
    <div className="dashboard-container">
      {/* Toast Alert */}
      {errorToast && (
        <div style={{
          position: 'fixed',
          top: '24px',
          right: '24px',
          background: 'var(--surface)',
          border: '1px solid var(--red-border)',
          color: 'var(--red)',
          padding: '12px 18px',
          borderRadius: 'var(--radius-sm)',
          boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          zIndex: 11000,
          fontFamily: 'var(--mono)',
          fontSize: '12px'
        }}>
          <AlertCircle size={16} />
          <span>{errorToast}</span>
        </div>
      )}

      {/* Main Header */}
      <Header
        status={status}
        isScanning={isScanning}
        isKilled={isKilled}
        isSseConnected={isSseConnected}
        onTriggerScan={handleTriggerScan}
        onOpenOverrideModal={() => setIsOverrideModalOpen(true)}
        onOpenKillswitchModal={() => setIsKillswitchModalOpen(true)}
        apiKey={apiKey}
        setApiKey={handleApiKeyChange}
      />

      <EvidencePanel scan={scanData} audit={auditData} requests={requests} now={now} />

      {/* Market Regime Overview */}
      <RegimeBanner regime={scanData?.regime || scanData?.regime_info || status?.last_known_regime || status?.regime_tracker} />

      {/* Navigation Tabs */}
      <div style={{
        display: 'flex',
        gap: '10px',
        borderBottom: '1px solid var(--border)',
        marginBottom: '24px',
        overflowX: 'auto',
        paddingBottom: '2px'
      }}>
        <button
          onClick={() => setActiveTab('portfolio')}
          className={`btn ${activeTab === 'portfolio' ? 'primary' : ''}`}
          style={{ borderBottomLeftRadius: 0, borderBottomRightRadius: 0 }}
        >
          <Briefcase size={14} />
          <span>PORTFOLIO PICKS ({scanData?.portfolio?.length || 0})</span>
        </button>

        <button
          onClick={() => setActiveTab('candidates')}
          className={`btn ${activeTab === 'candidates' ? 'primary' : ''}`}
          style={{ borderBottomLeftRadius: 0, borderBottomRightRadius: 0 }}
        >
          <Layers size={14} />
          <span>SCREENED CANDIDATES ({scanData?.candidates?.length || 0})</span>
        </button>

        <button
          onClick={() => setActiveTab('sectors')}
          className={`btn ${activeTab === 'sectors' ? 'primary' : ''}`}
          style={{ borderBottomLeftRadius: 0, borderBottomRightRadius: 0 }}
        >
          <PieChart size={14} />
          <span>SECTOR MOMENTUM</span>
        </button>

        <button
          onClick={() => setActiveTab('trades')}
          className={`btn ${activeTab === 'trades' ? 'primary' : ''}`}
          style={{ borderBottomLeftRadius: 0, borderBottomRightRadius: 0 }}
        >
          <History size={14} />
          <span>TRADE EXECUTION LOGS ({tradesData?.count || 0})</span>
        </button>
      </div>

      {/* Active Tab View */}
      <main>
        {activeTab === 'portfolio' && (
          <PortfolioTable 
            emptyState={emptyScanState('portfolio', scanData, requests.scan, now)}
            portfolio={scanData?.portfolio} 
            onInspectFactors={(item) => setSelectedTickerFactors(item)}
          />
        )}
        {activeTab === 'candidates' && (
          <CandidatesTable 
            emptyState={emptyScanState('candidates', scanData, requests.scan, now)}
            candidates={scanData?.candidates} 
            onInspectFactors={(item) => setSelectedTickerFactors(item)}
          />
        )}
        {activeTab === 'sectors' && (
          <SectorRSGrid sectorsData={sectorsData} />
        )}
        {activeTab === 'trades' && (
          <TradeLogHistory tradesData={tradesData} />
        )}
      </main>

      {/* Regime Override Modal */}
      <OverrideModal
        isOpen={isOverrideModalOpen}
        onClose={() => setIsOverrideModalOpen(false)}
        currentOverride={status?.regime_override}
        onSaveOverride={handleSaveOverride}
        apiKey={apiKey}
      />

      {/* Emergency Circuit Breaker (Killswitch) Modal */}
      <KillswitchModal
        isOpen={isKillswitchModalOpen}
        onClose={() => setIsKillswitchModalOpen(false)}
        isKilled={isKilled}
        onToggleKillswitch={handleToggleKillswitch}
        apiKey={apiKey}
      />

      {/* 7-Factor Signal Attribution Modal */}
      <FactorAttributionModal
        isOpen={Boolean(selectedTickerFactors)}
        onClose={() => setSelectedTickerFactors(null)}
        tickerData={selectedTickerFactors}
      />
    </div>
  );
}
