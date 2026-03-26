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
