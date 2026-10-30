"""Stock trend dashboard. Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import llm
from indicators import INDICES, SECTORS, add_indicators, regime, snapshot

st.set_page_config(page_title="Stock Trends", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def fetch(tickers: tuple, period="2y"):
    raw = yf.download(list(tickers), period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t] if len(tickers) > 1 else raw
            df = df.dropna(subset=["Close"])
            if len(df) > 60:
                out[t] = df
        except Exception:
            pass
    return out


def color_score(v):
    if pd.isna(v):
        return ""
    return f"color: {'#1a9850' if v > 0 else '#d73027'}"


# ---------------- sidebar ----------------
CLOUD = llm.is_cloud()
# One config per visitor session (never shared between visitors, never written to disk online)
cfg = st.session_state.setdefault("llm_cfg", llm.load_config())
with st.sidebar:
    st.header("Watchlist")
    wl = st.text_area("Tickers (comma separated)", "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, XOM, LLY",
                      height=90)
    tickers = tuple(dict.fromkeys(t.strip().upper() for t in wl.replace("\n", ",").split(",") if t.strip()))
    horizon = st.radio("Rank by", ["Swing (days–weeks)", "Long-term (months+)"])
    if st.button("Refresh data"):
        st.cache_data.clear()
    st.divider()
    st.header("AI settings (optional)" if CLOUD else "Gemma (Bionic API)")
    if CLOUD:
        st.caption("Bring your own OpenAI-compatible API (https only). Your key is kept only for this "
                   "browser session and is not saved, but it is sent to the address you enter, "
                   "so use a key you can revoke.")
    cfg["base_url"] = st.text_input("Base URL", cfg["base_url"],
                                    placeholder="https://.../v1" if CLOUD else "")
    cfg["api_key"] = st.text_input("API key", cfg["api_key"], type="password")
    cfg["model"] = st.text_input("Model name", cfg["model"])
    if CLOUD:
        if st.button("Test"):
            try:
                st.success(", ".join(llm.list_models(cfg)) or "Connected (no model list)")
            except Exception as e:
                st.error(str(e)[:200])
    else:
        c1, c2 = st.columns(2)
        if c1.button("Save"):
            llm.save_config(cfg)
            st.success("Saved")
        if c2.button("Test"):
            try:
                st.success(", ".join(llm.list_models(cfg)) or "Connected (no model list)")
            except Exception as e:
                st.error(str(e)[:200])

st.title("Stock Trend Dashboard")
st.caption("Daily data via Yahoo Finance (delayed). Scores are rule-based technical reads, not advice.")

# ---------------- data ----------------
bench_syms = tuple(INDICES) + tuple(SECTORS) + ("^VIX",)
with st.spinner("Downloading market data..."):
    mkt = fetch(bench_syms)
    stk = fetch(tickers) if tickers else {}

spy = mkt.get("SPY")
if spy is None:
    st.error("Could not download SPY. Check internet connection.")
    st.stop()

sec_snaps = {t: snapshot(mkt[t], spy) for t in SECTORS if t in mkt}
reg = regime(spy, mkt.get("^VIX"), sec_snaps)

tab1, tab2, tab3, tab4 = st.tabs(["Market & sectors", "Watchlist scanner", "Chart", "AI commentary"])

# ---------------- tab 1 ----------------
with tab1:
    a, b, c, d = st.columns(4)
    a.metric("Regime", reg["regime"], f"score {reg['points']:+d}/5")
    b.metric("VIX", f"{reg['vix']:.1f}")
    c.metric("Sectors > 50d SMA", f"{reg['breadth_pct']:.0f}%")
    d.metric("SPY vs 200d SMA", f"{reg['spy_vs_sma200_pct']:+.1f}%")

    idx = pd.DataFrame({INDICES[t]: snapshot(mkt[t], spy) for t in INDICES if t in mkt}).T
    st.subheader("Indices")
    st.dataframe(idx[["Price", "1D %", "1W %", "1M %", "3M %", "12M %", "RSI", "Swing score", "Long score", "Trend"]]
                 .style.format(precision=1).map(color_score, subset=["1D %", "1W %", "1M %", "3M %", "12M %"]),
                 use_container_width=True)

    sec = pd.DataFrame({f"{SECTORS[t]} ({t})": s for t, s in sec_snaps.items() if s}).T
    sec = sec.sort_values("3M %", ascending=False)
    st.subheader("Sector rotation")
    fig = go.Figure(go.Bar(x=sec["3M %"], y=sec.index, orientation="h",
                           marker_color=["#1a9850" if v > 0 else "#d73027" for v in sec["3M %"]]))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"),
                      xaxis_title="3-month return %")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sec[["1W %", "1M %", "3M %", "6M %", "RS 3M %", "RSI", "vs SMA50 %", "Swing score", "Long score", "Trend"]]
                 .style.format(precision=1).map(color_score, subset=["1W %", "1M %", "3M %", "6M %", "RS 3M %"]),
                 use_container_width=True)

# ---------------- tab 2 ----------------
rows = {t: snapshot(df, spy) for t, df in stk.items()}
tbl = pd.DataFrame({t: s for t, s in rows.items() if s}).T
with tab2:
    missing = [t for t in tickers if t not in stk]
    if missing:
        st.warning("No data for: " + ", ".join(missing))
    if tbl.empty:
        st.info("Add tickers in the sidebar.")
    else:
        key = "Swing score" if horizon.startswith("Swing") else "Long score"
        tbl = tbl.sort_values(key, ascending=False)
        cols = ["Price", "1D %", "1M %", "3M %", "12M %", "RS 3M %", "RSI", "vs SMA50 %", "vs SMA200 %",
                "From 52w high %", "Vol ratio", "ATR %", "Swing score", "Long score", "Trend"]
        st.dataframe(tbl[cols].style.format(precision=1)
                     .map(color_score, subset=["1D %", "1M %", "3M %", "12M %", "RS 3M %"])
                     .background_gradient(subset=["Swing score", "Long score"], cmap="RdYlGn", vmin=-100, vmax=100),
                     use_container_width=True, height=min(80 + 36 * len(tbl), 600))
        st.download_button("Download CSV", tbl[cols].round(2).to_csv().encode(), "watchlist_trends.csv", "text/csv")

# ---------------- tab 3 ----------------
with tab3:
    pool = list(tbl.index) if not tbl.empty else []
    pool += list(INDICES) + list(SECTORS)
    sym = st.selectbox("Symbol", pool)
    src = stk.get(sym) if sym in stk else mkt.get(sym)
    if src is not None:
        d = add_indicators(src).tail(250)
        f = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
        f.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name=sym), 1, 1)
        for n, col in ((20, "#f1a340"), (50, "#2166ac"), (200, "#762a83")):
            f.add_trace(go.Scatter(x=d.index, y=d[f"SMA{n}"], name=f"SMA{n}", line=dict(width=1.3, color=col)), 1, 1)
        f.add_trace(go.Scatter(x=d.index, y=d.RSI, name="RSI", line=dict(color="#555")), 2, 1)
        f.add_hline(y=70, line_dash="dot", row=2, col=1)
        f.add_hline(y=30, line_dash="dot", row=2, col=1)
        f.add_trace(go.Bar(x=d.index, y=d.MACDhist, name="MACD hist",
                           marker_color=np.where(d.MACDhist > 0, "#1a9850", "#d73027")), 3, 1)
        f.update_layout(height=720, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0),
                        legend=dict(orientation="h"))
        st.plotly_chart(f, use_container_width=True)

# ---------------- tab 4 ----------------
with tab4:
    if CLOUD:
        st.write("Optional. Sends the computed numbers (not raw data) to the AI service you set in the sidebar "
                 "for a short written read. You need your own API key.")
    else:
        st.write("Sends the computed numbers (not raw data) to your local Gemma model for a short written read.")
    if CLOUD and not (cfg["base_url"] and cfg["model"]):
        st.info("Enter a Base URL, API key and model name in the sidebar to use this tab.")
    elif st.button("Generate commentary"):
        top = sec.head(3).index.tolist()
        bot = sec.tail(3).index.tolist()
        lines = [f"Market regime: {reg['regime']} (score {reg['points']}/5), VIX {reg['vix']:.1f}, "
                 f"sector breadth {reg['breadth_pct']:.0f}% above 50d SMA, SPY {reg['spy_vs_sma200_pct']:+.1f}% vs 200d SMA.",
                 f"Leading sectors (3M): {', '.join(top)}. Lagging: {', '.join(bot)}.", "Watchlist:"]
        for t, r in tbl.head(15).iterrows():
            lines.append(f"{t}: px {r['Price']:.2f}, 1M {r['1M %']:+.1f}%, 3M {r['3M %']:+.1f}%, RS3M {r['RS 3M %']:+.1f}%, "
                         f"RSI {r['RSI']:.0f}, vs50d {r['vs SMA50 %']:+.1f}%, vs200d {r['vs SMA200 %']:+.1f}%, "
                         f"{r['From 52w high %']:+.1f}% from 52w high, swing {r['Swing score']}, long {r['Long score']}, {r['Trend']}")
        prompt = "\n".join(lines)
        with st.expander("Prompt sent"):
            st.code(prompt)
        try:
            with st.spinner("AI is thinking..."):
                st.markdown(llm.ask(cfg, llm.SYSTEM, prompt))
        except Exception as e:
            st.error(f"LLM call failed: {str(e)[:300]}. Check Base URL / model in the sidebar and press Test.")
