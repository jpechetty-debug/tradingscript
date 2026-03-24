import pandas as pd
import numpy as np

def _true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"] - df["Close"].shift(1)).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

def _wilder(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()

def _supertrend_vectorised(df: pd.DataFrame, period: int, mult: float) -> tuple[np.ndarray, np.ndarray]:
    tr = _true_range(df)
    atr = _wilder(tr, period)
    hl2 = (df["High"] + df["Low"]) / 2
    close = df["Close"].values
    n = len(close)

    final_upper = (hl2 + mult * atr).values
    final_lower = (hl2 - mult * atr).values

    for i in range(1, n):
        final_upper[i] = (final_upper[i] if close[i - 1] > final_upper[i - 1]
                          else min(final_upper[i], final_upper[i - 1]))
        final_lower[i] = (final_lower[i] if close[i - 1] < final_lower[i - 1]
                          else max(final_lower[i], final_lower[i - 1]))

    trend_up = np.ones(n, dtype=bool)
    st = np.where(trend_up, final_lower, final_upper)
    st[0] = final_upper[0]
    trend_up[0] = True

    for i in range(1, n):
        if st[i - 1] == final_lower[i - 1]:
            trend_up[i] = close[i] >= final_lower[i]
        else:
            trend_up[i] = close[i] > final_upper[i]
        st[i] = final_lower[i] if trend_up[i] else final_upper[i]

    return st, trend_up

def add_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]

    df["EMA_20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = c.ewm(span=50, adjust=False).mean()
    df["EMA_200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    avg_gain = _wilder(delta.clip(lower=0), 14)
    avg_loss = _wilder((-delta.clip(upper=0)), 14)
    rs_s = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    df["RSI"] = 100 - (100 / (1 + rs_s))

    tr = _true_range(df)
    df["ATR"] = _wilder(tr, 14)
    df["ATR_50_mean"] = df["ATR"].rolling(50).mean()
    df["ATR_Pctile"] = df["ATR"].rolling(252, min_periods=50).rank(pct=True) * 100

    st_vals, trend_up = _supertrend_vectorised(df, config.SUPER_PERIOD, config.SUPER_MULT)
    df["Supertrend"] = st_vals
    df["Super_Up"] = trend_up

    adx_p = config.ADX_PERIOD
    h, l_ = df["High"].values, df["Low"].values
    pdm = np.where((h[1:]-h[:-1]>l_[:-1]-l_[1:])&(h[1:]-h[:-1]>0), h[1:]-h[:-1], 0.0)
    ndm = np.where((l_[:-1]-l_[1:]>h[1:]-h[:-1])&(l_[:-1]-l_[1:]>0), l_[:-1]-l_[1:], 0.0)
    pdm = pd.Series(np.insert(pdm, 0, 0.0), index=df.index)
    ndm = pd.Series(np.insert(ndm, 0, 0.0), index=df.index)
    tr_s = pd.Series(tr.values, index=df.index)
    pdm_s = _wilder(pdm, adx_p); ndm_s = _wilder(ndm, adx_p)
    tr_sm = _wilder(tr_s, adx_p).replace(0, np.nan)
    pdi = 100*pdm_s/tr_sm; ndi = 100*ndm_s/tr_sm
    dx = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0, np.nan)
    df["ADX"] = _wilder(dx, adx_p)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    ml = ema12 - ema26
    df["MACD_Hist"] = ml - ml.ewm(span=9, adjust=False).mean()

    df["Vol_Avg_20"] = df["Volume"].rolling(20).mean()
    df["Turnover_Avg_20"] = (c * df["Volume"]).rolling(20).mean()
    df["Up_Day"] = (c > df["Open"]).astype(int)
    df["Dn_Day"] = (c < df["Open"]).astype(int)
    
    df.dropna(subset=["EMA_20","EMA_50","EMA_200","RSI","ATR","ADX","Vol_Avg_20","MACD_Hist"], inplace=True)
    return df
