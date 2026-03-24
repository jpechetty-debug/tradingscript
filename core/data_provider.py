import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from .config import IST

log = logging.getLogger("sovereign.data")

class FyersSessionManager:
    _instance = None
    @classmethod
    def get_client(cls, config):
        if cls._instance: return cls._instance
        try:
            from fyers_apiv3 import fyersModel
            cls._instance = fyersModel.FyersModel(
                client_id=config.FYERS_CLIENT_ID,
                token=config.FYERS_ACCESS_TOKEN,
                log_path=os.getcwd()
            )
            return cls._instance
        except Exception as e:
            log.error(f"Fyers Init Error: {e}")
            return None

def _fyers_to_df(data: dict) -> pd.DataFrame:
    if data.get("s") != "ok" or "candles" not in data: return pd.DataFrame()
    df = pd.DataFrame(data["candles"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST)
    df.set_index("Timestamp", inplace=True)
    return df

def fetch_single_ticker(ticker: str, fsym: str, config):
    fyers = FyersSessionManager.get_client(config)
    if fyers:
        data = {
            "symbol": fsym,
            "resolution": "D",
            "date_format": "1",
            "range_from": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "range_to": datetime.now().strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        try:
            res = fyers.history(data=data)
            df = _fyers_to_df(res)
            if not df.empty: return ticker, df
        except Exception as e:
            log.debug(f"Fyers error for {ticker}: {e}")
    return ticker, None

def fetch_daily_batch(tickers: list, config) -> dict[str, pd.DataFrame]:
    out = {}
    fyers_map = {t: f"NSE:{t.replace('.NS','')}-EQ" for t in tickers}
    fyers_map[config.BENCHMARK] = "NSE:NIFTY50-INDEX"

    if config.USE_FYERS:
        log.info("📡 Downloading daily data via Fyers (Parallel)...")
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = [executor.submit(fetch_single_ticker, t, fyers_map[t], config) for t in tickers + [config.BENCHMARK]]
            for future in futures:
                ticker, df = future.result()
                if df is not None: out[ticker] = df
        if out: return out

    log.warning("Falling back to yfinance (chunked)")
    full_list = tickers + [config.BENCHMARK]
    chunk_size = 40
    for i in range(0, len(full_list), chunk_size):
        chunk = full_list[i : i + chunk_size]
        try:
            raw = yf.download(chunk, period=config.DAILY_PERIOD, interval="1d", group_by="ticker", progress=False)
            log.info("Chunk: %s | Raw shape: %s | Columns: %s", chunk, raw.shape, list(raw.columns))
            if raw.empty: continue
            
            for t in chunk:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if t not in raw.columns.levels[0]: continue
                        df = raw.xs(t, axis=1, level=0).copy()
                    else:
                        if len(chunk) == 1: df = raw.copy()
                        else: continue # should not happen with group_by=ticker
                    
                    df.dropna(how="all", inplace=True)
                    df.columns = [c.title() for c in df.columns]
                    if not df.empty: out[t] = df
                except Exception as e:
                    log.debug(f"yfinance extracting error for {t}: {e}")
        except Exception as e:
            log.error(f"yfinance batch error for chunk starting {chunk[0]}: {e}")
            
    return out
