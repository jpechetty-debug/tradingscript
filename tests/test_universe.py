import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.universe import (
    ALL_TICKERS, N_TICKERS, TICKER_TO_SECTOR, N_SECTORS, SECTOR_LIST,
    TIER_1_GENERALS, TIER_2_ROTATION, TIER_3A_DEFENSIVE, TIER_3B_EVENT,
    TIER_4_MACRO, TIER_5_BROAD_500, CORE_15_PULSE, TIERS,
    get_fyers_symbol, fyers_to_yfi, refresh_universe
)
def test_ticker_to_sector_is_dict():
    assert isinstance(TICKER_TO_SECTOR, dict)
    assert len(TICKER_TO_SECTOR) == 503

def test_universe_sizes_and_tiers():
    assert N_TICKERS == 503
    assert len(ALL_TICKERS) == 503
    assert len(set(ALL_TICKERS)) == 503  # No duplicates
    assert len(TIER_1_GENERALS) == 20
    assert len(TIER_2_ROTATION) == 32
    assert len(TIER_3A_DEFENSIVE) == 15
    assert len(TIER_3B_EVENT) == 15
    assert len(TIER_4_MACRO) == 18
    assert len(TIER_5_BROAD_500) == 403
    assert len(CORE_15_PULSE) == 15

def test_tiers_mapping():
    assert "tier1" in TIERS
    assert "tier2" in TIERS
    assert "tier3a" in TIERS
    assert "tier3b" in TIERS
    assert "tier4" in TIERS
    assert "tier5" in TIERS
    assert "core15" in TIERS
    assert "core100" in TIERS
    assert "all" in TIERS
    assert len(TIERS["tier5"]) == 403
    assert len(TIERS["all"]) == 503

def test_n_sectors_matches_sector_list():
    assert N_SECTORS == len(SECTOR_LIST)
    assert N_SECTORS == 12

def test_all_tickers_map_to_known_sector():
    for ticker, sector in TICKER_TO_SECTOR.items():
        assert sector in SECTOR_LIST, f"{ticker} maps to unknown sector {sector!r}"

def test_benchmark_not_in_ticker_map():
    assert "^NSEI" not in TICKER_TO_SECTOR
    assert "^NSEI.NS" not in TICKER_TO_SECTOR

def test_tracked_fundamental_stocks_present():
    expected_stocks = {
        "RELIANCE.NS": "ENERGY",
        "HDFCBANK.NS": "FINANCIALS",
        "TECHM.NS": "IT",
        "TVSMOTOR.NS": "AUTO",
        "TATASTEEL.NS": "METALS",
        "HAL.NS": "DEFENCE",
        "LT.NS": "INDUSTRIALS",
        "DLF.NS": "REALTY",
        "SUNPHARMA.NS": "PHARMA",
        "ITC.NS": "FMCG",
        # Sample newly added stocks from TIER_5
        "CUPID.NS": "FMCG",
        "STLTECH.NS": "TELECOM",
        "MTARTECH.NS": "DEFENCE",
        "ATHERENERG.NS": "AUTO",
        "FRACTAL.NS": "IT",
        "CANHLIFE.NS": "FINANCIALS",
        "GVT&D.NS": "INDUSTRIALS",
        "ARE&M.NS": "AUTO",
    }
    for ticker, expected_sector in expected_stocks.items():
        assert ticker in TICKER_TO_SECTOR, f"Expected {ticker} in TICKER_TO_SECTOR"
        assert TICKER_TO_SECTOR[ticker] == expected_sector

def test_get_fyers_symbol():
    assert get_fyers_symbol("RELIANCE.NS") == "NSE:RELIANCE-EQ"
    assert get_fyers_symbol("NSE:RELIANCE-EQ") == "NSE:RELIANCE-EQ"
    assert get_fyers_symbol("GVT&D.NS") == "NSE:GVT&D-EQ"
    assert get_fyers_symbol("ARE&M.NS") == "NSE:ARE&M-EQ"

def test_fyers_to_yfi():
    assert fyers_to_yfi("NSE:RELIANCE-EQ") == "RELIANCE.NS"
    assert fyers_to_yfi("RELIANCE.NS") == "RELIANCE.NS"
    assert fyers_to_yfi("NSE:GVT&D-EQ") == "GVT&D.NS"
    assert fyers_to_yfi("NSE:ARE&M-EQ") == "ARE&M.NS"

def test_refresh_universe(caplog):
    refresh_universe()
    assert "Dynamic universe refresh not fully implemented" in caplog.text

