import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.universe import TICKER_TO_SECTOR, N_SECTORS, SECTOR_LIST

def test_ticker_to_sector_is_dict():
    assert isinstance(TICKER_TO_SECTOR, dict)
    assert len(TICKER_TO_SECTOR) > 0

def test_n_sectors_matches_sector_list():
    assert N_SECTORS == len(SECTOR_LIST)

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
    }
    for ticker, expected_sector in expected_stocks.items():
        assert ticker in TICKER_TO_SECTOR, f"Expected {ticker} in TICKER_TO_SECTOR"
        assert TICKER_TO_SECTOR[ticker] == expected_sector, (
            f"Expected {ticker} in sector {expected_sector}, got {TICKER_TO_SECTOR[ticker]}"
        )
