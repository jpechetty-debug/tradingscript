"""
core/universe.py
================
Single source of truth for the curated NSE ticker universe and sector mapping.
Previously embedded in screener.py; extracted so core/ modules can import
it without pulling in the monolith.
"""

from __future__ import annotations

# ==========================================
# TIER DEFINITIONS (Sovereign Scanning Tiers)
# ==========================================

TIER_1_GENERALS = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS",
    "KOTAKBANK.NS","BAJFINANCE.NS","INFY.NS","TCS.NS","HCLTECH.NS",
    "BHARTIARTL.NS","LT.NS","M&M.NS","MARUTI.NS","HAL.NS",
    "BEL.NS","TATASTEEL.NS","JSWSTEEL.NS","ADANIPORTS.NS","SIEMENS.NS"
]

TIER_2_ROTATION = [
    "INDUSINDBK.NS","BANKBARODA.NS","CANBK.NS","BAJAJFINSV.NS",
    "SHRIRAMFIN.NS","CHOLAFIN.NS","PFC.NS","RECLTD.NS",
    "TECHM.NS","COFORGE.NS","PERSISTENT.NS","KPITTECH.NS",
    "TVSMOTOR.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","HEROMOTOCO.NS",
    "ABB.NS","CGPOWER.NS","POLYCAB.NS","HAVELLS.NS","CUMMINSIND.NS",
    "NTPC.NS","POWERGRID.NS","TATAPOWER.NS",
    "DLF.NS","LODHA.NS","GODREJPROP.NS",
    "RVNL.NS","IRFC.NS","RAILTEL.NS","IRCON.NS","NBCC.NS"
]

TIER_3A_DEFENSIVE = [
    "SUNPHARMA.NS","CIPLA.NS","DRREDDY.NS","DIVISLAB.NS","LUPIN.NS",
    "ITC.NS","HINDUNILVR.NS","NESTLEIND.NS","VBL.NS","BRITANNIA.NS",
    "ULTRACEMCO.NS","AMBUJACEM.NS","SRF.NS","PIIND.NS","NAVINFLUOR.NS"
]

TIER_3B_EVENT = [
    "ADANIENT.NS","ADANIGREEN.NS","ADANIPOWER.NS",
    "BSE.NS","MCX.NS",
    "PAYTM.NS","DIXON.NS","INDIGO.NS",
    "SOLARINDS.NS","WAAREEENER.NS",
    "MAZDOCK.NS","GRSE.NS","BDL.NS","COCHINSHIP.NS","DATAPATTNS.NS"
]

TIER_4_MACRO = [
    "ONGC.NS","BPCL.NS","IOC.NS","GAIL.NS","COALINDIA.NS",
    "HINDALCO.NS","JINDALSTEL.NS","NMDC.NS","VEDL.NS",
    "PNB.NS","MUTHOOTFIN.NS","M&MFIN.NS",
    "WIPRO.NS","MPHASIS.NS","LTIM.NS",
    "MOTHERSON.NS","BHARATFORG.NS","ASHOKLEY.NS"
]

CORE_15_PULSE = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS",
    "BAJFINANCE.NS","INFY.NS","TCS.NS","BHARTIARTL.NS","LT.NS",
    "M&M.NS","HAL.NS","BEL.NS","TATASTEEL.NS","ADANIPORTS.NS"
]

# ==========================================
# SECTOR MAPPINGS
# (Used by Scoring/Gate-Check layer for weights)
# ==========================================

SECTORS: dict[str, list[str]] = {
    "FINANCIALS": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
        "INDUSINDBK.NS", "BANKBARODA.NS", "CANBK.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS", "CHOLAFIN.NS", "PFC.NS", "RECLTD.NS",
        "BSE.NS", "MCX.NS", "PAYTM.NS",
        "PNB.NS", "MUTHOOTFIN.NS", "M&MFIN.NS"
    ],
    "IT": [
        "INFY.NS", "TCS.NS", "HCLTECH.NS",
        "TECHM.NS", "COFORGE.NS", "PERSISTENT.NS", "KPITTECH.NS",
        "WIPRO.NS", "MPHASIS.NS", "LTIM.NS"
    ],
    "ENERGY": [
        "RELIANCE.NS",
        "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS",
        "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "WAAREEENER.NS",
        "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "COALINDIA.NS"
    ],
    "AUTO": [
        "M&M.NS", "MARUTI.NS",
        "TVSMOTOR.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
        "MOTHERSON.NS", "BHARATFORG.NS", "ASHOKLEY.NS"
    ],
    "METALS": [
        "TATASTEEL.NS", "JSWSTEEL.NS",
        "HINDALCO.NS", "JINDALSTEL.NS", "NMDC.NS", "VEDL.NS"
    ],
    "DEFENCE": [
        "HAL.NS", "BEL.NS",
        "SOLARINDS.NS",
        "MAZDOCK.NS", "GRSE.NS", "BDL.NS", "COCHINSHIP.NS", "DATAPATTNS.NS"
    ],
    "INDUSTRIALS": [
        "LT.NS", "ADANIPORTS.NS", "SIEMENS.NS",
        "ABB.NS", "CGPOWER.NS", "POLYCAB.NS", "HAVELLS.NS", "CUMMINSIND.NS",
        "RVNL.NS", "IRFC.NS", "RAILTEL.NS", "IRCON.NS", "NBCC.NS",
        "ULTRACEMCO.NS", "AMBUJACEM.NS",
        "DIXON.NS", "INDIGO.NS"
    ],
    "REALTY": [
        "DLF.NS", "LODHA.NS", "GODREJPROP.NS"
    ],
    "PHARMA": [
        "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS"
    ],
    "FMCG": [
        "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "VBL.NS", "BRITANNIA.NS"
    ],
    "CHEMICALS": [
        "SRF.NS", "PIIND.NS", "NAVINFLUOR.NS"
    ],
    "TELECOM": [
        "BHARTIARTL.NS"
    ]
}

ALL_TICKERS: list[str] = list(
    dict.fromkeys(t for tickers in SECTORS.values() for t in tickers)
)
TICKER_TO_SECTOR: dict[str, str] = {
    t: s for s, tickers in SECTORS.items() for t in tickers
}

N_SECTORS = len(SECTORS)
N_TICKERS = len(ALL_TICKERS)
SECTOR_LIST = list(SECTORS.keys())


def get_fyers_symbol(ticker: str) -> str:
    """Convert yfinance ticker (RELIANCE.NS) → Fyers symbol (NSE:RELIANCE-EQ)."""
    if ":" in ticker:
        return ticker
    return f"NSE:{ticker.replace('.NS', '')}-EQ"


def fyers_to_yfi(symbol: str) -> str:
    """Convert Fyers symbol (NSE:RELIANCE-EQ) → yfinance ticker (RELIANCE.NS)."""
    if ":" not in symbol:
        return symbol
    return f"{symbol.split(':')[1].replace('-EQ', '')}.NS"
