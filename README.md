# Stock Trends

Streamlit dashboard: market regime, sector rotation, watchlist scanner (swing + long-term scores), charts, and Gemma commentary via Bionic's OpenAI-compatible API.

## Move to personal laptop
1. Copy this whole folder (skip `.venv`).
2. Double-click `run.bat` (creates a venv, installs requirements, launches the app at http://localhost:8501).
3. In the sidebar set **Base URL** (e.g. `http://localhost:3000/v1`), **API key**, **Model name** (exactly as Bionic lists it), press **Test**, then **Save**.

## Scores
- **Swing** (-100..+100): price vs SMA20, SMA20 vs SMA50, MACD histogram + slope, RSI zone, volume confirmation, 1M return.
- **Long** (-100..+100): price vs SMA200, SMA50 vs SMA200, SMA200 slope, 6M/12M return, 3M relative strength vs SPY, distance from 52w high.
- **Regime**: SPY vs SMA50/200, SMA50 vs SMA200, sector breadth, VIX level.

Educational use only; not financial advice.
