"""
core/universe.py
================
Single source of truth for the NSE 178-ticker universe and sector mapping.
Previously embedded in screener.py; extracted so core/ modules can import
it without pulling in the monolith.
"""

from __future__ import annotations

SECTORS: dict[str, list[str]] = {
    "BANKING": [
        "SBIN.NS", "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS",
        "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS",
        "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "AUBANK.NS", "RBLBANK.NS",
    ],
    "IT": [
        "INFY.NS", "TCS.NS", "HCLTECH.NS", "TECHM.NS", "WIPRO.NS",
        "LTIM.NS", "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS", "OFSS.NS",
        "KPITTECH.NS", "TATAELXSI.NS", "LTTS.NS", "CYIENT.NS",
    ],
    "AUTO": [
        "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "MOTHERSON.NS", "BOSCHLTD.NS",
        "BHARATFORG.NS", "APOLLOTYRE.NS", "MRF.NS", "BALKRISIND.NS", "EXIDEIND.NS",
    ],
    "METALS": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "JINDALSTEL.NS",
        "NMDC.NS", "SAIL.NS", "NATIONALUM.NS", "APLAPOLLO.NS", "RATNAMANI.NS",
        "WELCORP.NS", "HINDCOPPER.NS",
    ],
    "PHARMA": [
        "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "LUPIN.NS", "TORNTPHARM.NS", "BIOCON.NS", "ALKEM.NS", "IPCALAB.NS",
        "AUROPHARMA.NS", "GLENMARK.NS", "ABBOTINDIA.NS", "PFIZER.NS", "GLAXO.NS",
    ],
    "FMCG": [
        "ITC.NS", "HINDUNILVR.NS", "BRITANNIA.NS", "TATACONSUM.NS", "NESTLEIND.NS",
        "MARICO.NS", "DABUR.NS", "GODREJCP.NS", "COLPAL.NS", "EMAMILTD.NS",
        "RADICO.NS", "UBL.NS",
    ],
    "ENERGY": [
        "RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ADANIENT.NS", "ONGC.NS",
        "COALINDIA.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "HINDPETRO.NS",
        "TATAPOWER.NS", "ADANIGREEN.NS", "TORNTPOWER.NS", "CESC.NS", "NLCINDIA.NS",
    ],
    "REALTY": [
        "DLF.NS", "GODREJPROP.NS", "PRESTIGE.NS", "OBEROIRLTY.NS", "LODHA.NS",
        "PHOENIXLTD.NS", "BRIGADE.NS", "SOBHA.NS", "KOLTEPATIL.NS",
    ],
    "FINANCE": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "SBILIFE.NS",
        "HDFCLIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS", "SBICARD.NS",
        "M&MFIN.NS", "MANAPPURAM.NS", "LICHSGFIN.NS", "SHRIRAMFIN.NS", "POONAWALLA.NS",
    ],
    "CAPITAL_GOODS": [
        "LT.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "BHEL.NS",
        "CUMMINSIND.NS", "THERMAX.NS", "VOLTAS.NS", "AIAENG.NS", "BEL.NS",
        "HAL.NS", "GRINDWELL.NS", "TIINDIA.NS",
    ],
    "CONSUMER": [
        "TITAN.NS", "ASIANPAINT.NS", "PIDILITIND.NS", "WHIRLPOOL.NS",
        "CROMPTON.NS", "VGUARD.NS", "KAJARIACER.NS", "BATAINDIA.NS", "PAGEIND.NS",
    ],
    "TELECOM": [
        "BHARTIARTL.NS", "INDUSTOWER.NS",
    ],
    "CEMENT": [
        "ULTRACEMCO.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS",
        "RAMCOCEM.NS", "JKCEMENT.NS", "HEIDELBERG.NS",
    ],
    "CHEMICALS": [
        "SRF.NS", "ATUL.NS", "NAVINFLUOR.NS", "TATACHEM.NS",
        "GNFC.NS", "AARTIIND.NS", "CLEAN.NS",
    ],
    "INFRASTRUCTURE": [
        "ADANIPORTS.NS", "IRB.NS", "KNRCON.NS", "NCC.NS",
        "NBCC.NS", "RVNL.NS", "IRCON.NS", "HFCL.NS",
    ],
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
