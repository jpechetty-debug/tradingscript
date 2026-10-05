import numpy as np
import pandas as pd

from core.database import SqliteDatabase
from core.leader_pullback import scan_leader_pullback
from core.paper_ledger import PaperLedger

IDX = pd.bdate_range("2024-01-01", periods=300)


def frame(close, volume=1e6):
    c = pd.Series(close, index=IDX, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": volume})


def universe():
    trend = np.linspace(100, 200, 300)
    leader = trend.copy(); leader[-3:] = [200, 197, 195]          # strong stock, sharp 2-day dip
    laggards = {f"L{i}.NS": frame(np.linspace(100, 100 + i, 300)) for i in range(9)}
    return {"LEAD.NS": frame(leader), **laggards}


def test_signals_leader_dip_in_bull_market():
    sig = scan_leader_pullback(universe(), pd.Series(np.linspace(100, 150, 300), index=IDX), IDX[-1])
    assert [s.ticker for s in sig] == ["LEAD"]
    assert sig[0].rsi2 < 10 and sig[0].stop_offset > 0


def test_no_signals_when_market_below_sma200():
    bench = pd.Series(np.linspace(150, 100, 300), index=IDX)
    assert scan_leader_pullback(universe(), bench, IDX[-1]) == []


def test_illiquid_leader_skipped():
    u = universe(); u["LEAD.NS"]["Volume"] = 10.0
    assert scan_leader_pullback(u, pd.Series(np.linspace(100, 150, 300), index=IDX), IDX[-1]) == []


def test_ledger_records_pullback_signals_idempotently(tmp_path):
    ledger = PaperLedger(SqliteDatabase(tmp_path / "state.db"))
    sig = scan_leader_pullback(universe(), pd.Series(np.linspace(100, 150, 300), index=IDX), IDX[-1])
    ledger.record_pullback_signals(sig); ledger.record_pullback_signals(sig)
    events = ledger.db.fetch_paper_events()
    assert len(events) == 1


def test_position_size_equal_risk_and_capped():
    from core.leader_pullback import position_size
    assert position_size(1_000_000, 100.0, 5.0) == 1000          # 0.5% risk / 5 = 1000, cap 10% = 1000
    assert position_size(1_000_000, 100.0, 1.0) == 1000          # risk says 5000, cap wins
    assert position_size(0, 100.0, 5.0) == 0


def test_members_filter_excludes_non_constituents(tmp_path):
    from core.leader_pullback import load_members, members_on
    p = tmp_path / "m.csv"
    p.write_text("ticker,start,end\nLEAD,2023-01-01,2024-06-01\nL0,2023-01-01,\n")
    m = members_on(load_members(p), IDX[-1])
    assert m == {"L0.NS"}
    assert scan_leader_pullback(universe(), pd.Series(np.linspace(100, 150, 300), index=IDX), IDX[-1], members=m) == []
