"""Technical indicators, trend scoring and market-regime logic."""
import numpy as np
import pandas as pd

SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLY": "Cons. Discretionary", "XLP": "Cons. Staples", "XLI": "Industrials",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Comm. Services",
}
INDICES = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "DIA": "Dow"}


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def macd(close, fast=12, slow=26, sig=9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df):
    df = df.copy()
    c = df["Close"]
    for n in (20, 50, 200):
        df[f"SMA{n}"] = c.rolling(n).mean()
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["RSI"] = rsi(c)
    df["MACD"], df["MACDsig"], df["MACDhist"] = macd(c)
    df["ATR"] = atr(df)
    df["VolRatio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return df


def _ret(c, n):
    return c.iloc[-1] / c.iloc[-n - 1] - 1 if len(c) > n else np.nan


def snapshot(df, bench=None):
    """Latest metrics + swing / long-term scores (each -100..+100)."""
    d = add_indicators(df).dropna(subset=["SMA50"])
    if d.empty:
        return None
    last, c = d.iloc[-1], d["Close"]
    px = last["Close"]
    hi52 = df["Close"].tail(252).max()
    m = {
        "Price": px,
        "1D %": _ret(df["Close"], 1) * 100,
        "1W %": _ret(df["Close"], 5) * 100,
        "1M %": _ret(df["Close"], 21) * 100,
        "3M %": _ret(df["Close"], 63) * 100,
        "6M %": _ret(df["Close"], 126) * 100,
        "12M %": _ret(df["Close"], 252) * 100,
        "RSI": last["RSI"],
        "vs SMA20 %": (px / last["SMA20"] - 1) * 100,
        "vs SMA50 %": (px / last["SMA50"] - 1) * 100,
        "vs SMA200 %": (px / last["SMA200"] - 1) * 100 if pd.notna(last["SMA200"]) else np.nan,
        "From 52w high %": (px / hi52 - 1) * 100,
        "Vol ratio": last["VolRatio"],
        "ATR %": last["ATR"] / px * 100,
    }
    if bench is not None and len(bench) > 64 and len(df) > 64:
        m["RS 3M %"] = m["3M %"] - _ret(bench["Close"], 63) * 100
    else:
        m["RS 3M %"] = np.nan

    # swing score: short-term trend, momentum, volume confirmation
    s = 0
    s += 20 if px > last["SMA20"] else -20
    s += 15 if last["SMA20"] > last["SMA50"] else -15
    s += 15 if last["MACDhist"] > 0 else -15
    s += 10 if d["MACDhist"].iloc[-1] > d["MACDhist"].iloc[-4] else -10
    r = last["RSI"]
    s += 15 if 50 <= r <= 70 else (-15 if r < 40 else (-5 if r > 75 else 0))
    s += 10 if (last["VolRatio"] > 1.2 and m["1D %"] > 0) else (-10 if (last["VolRatio"] > 1.2 and m["1D %"] < 0) else 0)
    s += 15 if m["1M %"] > 0 else -15
    m["Swing score"] = int(np.clip(s, -100, 100))

    # long-term score: stage, 6-12M momentum, relative strength, proximity to highs
    t = 0
    if pd.notna(last["SMA200"]):
        t += 25 if px > last["SMA200"] else -25
        t += 15 if last["SMA50"] > last["SMA200"] else -15
        t += 10 if d["SMA200"].iloc[-1] > d["SMA200"].iloc[-21] else -10
    t += 20 if m["12M %"] > 0 else -20
    t += 10 if m["6M %"] > 0 else -10
    if pd.notna(m["RS 3M %"]):
        t += 10 if m["RS 3M %"] > 0 else -10
    t += 10 if m["From 52w high %"] > -10 else (-10 if m["From 52w high %"] < -30 else 0)
    m["Long score"] = int(np.clip(t, -100, 100))
    m["Trend"] = label(m["Swing score"], m["Long score"])
    return m


def label(swing, long):
    if swing >= 40 and long >= 40:
        return "Strong uptrend"
    if long >= 40 and swing < 0:
        return "Pullback in uptrend"
    if swing >= 40 and long < 0:
        return "Bounce in downtrend"
    if swing <= -40 and long <= -40:
        return "Strong downtrend"
    return "Mixed / range"


def regime(spy, vix, sector_snaps):
    """Overall market read from SPY trend, VIX and sector breadth."""
    d = add_indicators(spy)
    last = d.iloc[-1]
    above50 = sum(1 for s in sector_snaps.values() if s and s["vs SMA50 %"] > 0)
    breadth = above50 / max(len(sector_snaps), 1) * 100
    vx = float(vix["Close"].iloc[-1]) if vix is not None and len(vix) else np.nan
    pts = 0
    pts += 1 if last["Close"] > last["SMA200"] else -1
    pts += 1 if last["Close"] > last["SMA50"] else -1
    pts += 1 if last["SMA50"] > last["SMA200"] else -1
    pts += 1 if breadth >= 60 else (-1 if breadth <= 40 else 0)
    pts += 1 if vx < 18 else (-1 if vx > 25 else 0)
    name = "Risk-on" if pts >= 3 else ("Risk-off" if pts <= -3 else "Neutral / transitioning")
    return {"regime": name, "points": pts, "breadth_pct": breadth, "vix": vx,
            "spy_vs_sma200_pct": (last["Close"] / last["SMA200"] - 1) * 100}
