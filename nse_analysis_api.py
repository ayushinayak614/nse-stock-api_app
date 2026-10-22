"""
NSE Stock Analysis API (v2: signals, market overview, news, compare, backtesting)
--------------------------------------------------------------------------------
Original description: NSE Stock Analysis API — for n8n integration
================================================================================
A single FastAPI service that does ALL the real work (data pull + every
indicator built in our Jupyter session) and exposes it as one clean HTTP
endpoint. n8n calls this URL and gets back a complete JSON analysis for
ANY NSE-listed company — no coding required inside n8n itself.

REQUIREMENTS
------------
    pip install fastapi uvicorn yfinance pandas numpy

RUN LOCALLY (for testing in Jupyter/terminal)
-----------------------------------------------
    uvicorn nse_analysis_api:app --reload --port 8000

Then test in your browser or Jupyter:
    http://127.0.0.1:8000/analyze/RELIANCE

EXPOSING THIS TO n8n
---------------------
- If n8n runs on the SAME machine: use http://127.0.0.1:8000 directly (self-hosted n8n)
- If n8n is in the cloud (n8n.cloud) or on a different machine: you need a
  public URL. Easiest free options:
    1. ngrok (quick, good for testing):  ngrok http 8000
       -> gives you a temporary public URL like https://abc123.ngrok.io
    2. Deploy this file properly on Render.com or Railway.app (free tier)
       for a permanent URL — recommended once you move past testing

USAGE FROM n8n
---------------
In n8n, add an "HTTP Request" node:
    Method: GET
    URL: http://YOUR_URL/analyze/{{ $json.symbol }}
No code needed on the n8n side at all — this endpoint does everything.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="NSE Stock Analysis API")

# Allow any frontend (v0.dev, Lovable, or anywhere else) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── INDICATOR FUNCTIONS (same ones built in our Jupyter session today) ────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_moving_averages(prices, short_window=50, long_window=200):
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    return ma_short, ma_long

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100

def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth — the corrected version, fixing the bug
    we caught earlier today where fixed-position matching broke on
    stocks with an irregularly reported quarter."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def safe_round(val, decimals=2):
    if val is None or pd.isna(val):
        return None
    return round(float(val), decimals)


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    """Same date-matched logic as earnings growth, applied to revenue."""
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Extended fundamentals: earnings growth, revenue growth, margins, ROE, debt-to-equity.

    IMPORTANT: this deliberately avoids stock.info — Yahoo Finance blocks/rate-limits
    that specific endpoint for many cloud hosting providers (Render, Railway, Heroku),
    returning an almost-empty dict even though the price-history endpoint works fine.
    Instead, everything here is calculated directly from the income statement and
    balance sheet, which use the endpoint that's actually working for us."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        # margins calculated directly — same math a real analyst would do by hand
        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                # ROE using latest quarter's net income annualized (x4) against latest equity —
                # a simplified TTM-style approximation, standard when only one quarter is available
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    # market cap: try the lighter fast_info endpoint, which is less likely to be blocked
    # than the full .info dict — separate try block so a failure here doesn't affect anything above
    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass  # market cap is a nice-to-have; not worth failing the request over

    return fundamentals


def get_technical_snapshot(hist):
    """The core technical calculation, factored out so both /analyze and /watchlist can reuse it."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


# ── SIGNAL LABELS (used for colour-coded badges + heatmap) ──────────────────

def build_signals(technical):
    """Turn raw indicator values into simple labels the frontend can colour-code."""
    rsi = technical.get("rsi_14")
    hist_val = technical.get("macd_histogram")
    ma_trend = technical.get("ma_trend")
    pct52 = technical.get("pct_of_52w_high")

    if rsi is None:
        rsi_signal = None
    elif rsi > 70:
        rsi_signal = "Overbought"
    elif rsi < 30:
        rsi_signal = "Oversold"
    else:
        rsi_signal = "Neutral"

    if hist_val is None:
        macd_tag = None
    else:
        macd_tag = "Bullish" if hist_val > 0 else "Bearish"

    # Heatmap score: each of 4 components contributes -1, 0 or +1  ->  scaled to -100..+100
    score = 0
    if ma_trend == "bullish": score += 1
    elif ma_trend == "bearish": score -= 1
    if macd_tag == "Bullish": score += 1
    elif macd_tag == "Bearish": score -= 1
    if rsi is not None:
        if rsi >= 55: score += 1
        elif rsi <= 45: score -= 1
    if pct52 is not None:
        if pct52 >= 90: score += 1
        elif pct52 < 70: score -= 1
    score_pct = score * 25

    if score_pct >= 50: label = "Strong Bullish"
    elif score_pct > 0: label = "Bullish"
    elif score_pct == 0: label = "Neutral"
    elif score_pct > -50: label = "Bearish"
    else: label = "Strong Bearish"

    return {
        "rsi_signal": rsi_signal,        # Overbought (red) / Oversold (green) / Neutral
        "macd_signal": macd_tag,         # Bullish / Bearish
        "ma_signal": ma_trend.capitalize() if ma_trend else None,
        "signal_score": score_pct,       # -100 (very bearish) .. +100 (very bullish)
        "signal_label": label,
    }


@app.get("/")
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API is running"}


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_fundamentals(stock)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "signals": build_signals(technical),
            "fundamentals": fundamentals,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """
    OHLC candlestick data + overlay indicator series for charting.
    Returns one row per trading day: date, OHLCV, RSI, MACD line/signal, MA50, MA200.
    `months` controls how much history to return in the chart (default 6, max 24).
    """
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    months = min(max(months, 1), 24)

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)  # need 2y so MA200 has enough lookback, even though we only return the recent slice
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        trading_days = months * 21  # approx trading days per month
        recent = hist.tail(trading_days)

        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]) if date in rsi.index else None,
                "macd_line": safe_round(macd_line.loc[date]) if date in macd_line.index else None,
                "macd_signal": safe_round(signal_line.loc[date]) if date in signal_line.index else None,
                "ma_50": safe_round(ma50.loc[date]) if date in ma50.index else None,
                "ma_200": safe_round(ma200.loc[date]) if date in ma200.index else None,
            })

        return {"symbol": ticker, "months": months, "data": rows}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """
    Lightweight multi-stock snapshot for a watchlist view.
    Pass symbols comma-separated: /watchlist?symbols=RELIANCE,TCS,INFY
    Capped at 10 symbols per request to keep response time reasonable on free hosting.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", auto_adjust=True)
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue

            technical, corporate_action_flag = get_technical_snapshot(hist)
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
                **build_signals(technical),
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return {"count": len(results), "stocks": results}


# ═══════════════════════════════════════════════════════════════════════════
#  NEW IN v2:  /market-overview   /news   /compare   /backtest   /backtest-all
# ═══════════════════════════════════════════════════════════════════════════

def to_ticker(symbol: str) -> str:
    symbol = symbol.upper().strip()
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


# ── 1. MARKET OVERVIEW (NIFTY 50 / SENSEX / BANK NIFTY) ─────────────────────

INDEX_MAP = [
    ("NIFTY 50", "^NSEI"),
    ("SENSEX", "^BSESN"),
    ("BANK NIFTY", "^NSEBANK"),
]

@app.get("/market-overview")
def market_overview():
    """Index level, daily change and a 1-month sparkline for the top-of-page ticker bar."""
    indices = []
    for name, ticker in INDEX_MAP:
        try:
            h = yf.Ticker(ticker).history(period="1mo", auto_adjust=False)
            closes = h["Close"].dropna()
            if len(closes) < 2:
                indices.append({"name": name, "symbol": ticker, "error": "No data"})
                continue
            last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
            indices.append({
                "name": name,
                "symbol": ticker,
                "level": safe_round(last),
                "change": safe_round(last - prev),
                "change_pct": safe_round((last / prev - 1) * 100),
                "as_of": closes.index[-1].strftime("%Y-%m-%d"),
                "sparkline": [safe_round(v) for v in closes.tolist()],
            })
        except Exception as e:
            indices.append({"name": name, "symbol": ticker, "error": str(e)})
    return {"timestamp": datetime.now().isoformat(), "indices": indices}


# ── 2. NEWS HEADLINES ───────────────────────────────────────────────────────

@app.get("/news/{symbol}")
def stock_news(symbol: str, limit: int = 10):
    """Latest headlines via yfinance. Handles both old and new yfinance news formats."""
    ticker = to_ticker(symbol)
    limit = min(max(limit, 1), 20)
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as e:
        return {"symbol": ticker, "count": 0, "news": [], "note": f"News unavailable: {e}"}

    items = []
    for n in raw[:limit]:
        try:
            if isinstance(n, dict) and "content" in n and isinstance(n["content"], dict):   # newer yfinance
                c = n["content"]
                url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
                items.append({
                    "title": c.get("title"),
                    "summary": c.get("summary"),
                    "publisher": (c.get("provider") or {}).get("displayName"),
                    "published": c.get("pubDate"),
                    "url": url,
                })
            else:                                                                              # older yfinance
                ts = n.get("providerPublishTime")
                items.append({
                    "title": n.get("title"),
                    "summary": None,
                    "publisher": n.get("publisher"),
                    "published": datetime.fromtimestamp(ts).isoformat() if ts else None,
                    "url": n.get("link"),
                })
        except Exception:
            continue
    items = [i for i in items if i.get("title")]
    return {"symbol": ticker, "count": len(items), "news": items}


# ── 3. COMPARE 2-3 STOCKS (normalised to 100) ───────────────────────────────

@app.get("/compare")
def compare(symbols: str, months: int = 6):
    """
    Rebased performance: every stock starts at 100 on the first day of the window.
    Example: /compare?symbols=RELIANCE,TCS,INFY&months=6
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:3]
    if len(symbol_list) < 2:
        raise HTTPException(status_code=400, detail="Provide 2 or 3 symbols, e.g. ?symbols=RELIANCE,TCS")
    months = min(max(months, 1), 24)

    series, summary, errors = {}, [], []
    for sym in symbol_list:
        ticker = to_ticker(sym)
        try:
            hist = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
            if hist.empty:
                errors.append({"symbol": sym, "error": "No data found"})
                continue
            close = hist["Close"].dropna()
            close.index = close.index.tz_localize(None).normalize()
            cutoff = close.index[-1] - pd.DateOffset(months=months)
            close = close[close.index >= cutoff]
            series[sym.replace(".NS", "")] = close / close.iloc[0] * 100
            rets = close.pct_change().dropna()
            summary.append({
                "symbol": sym.replace(".NS", ""),
                "start_price": safe_round(close.iloc[0]),
                "end_price": safe_round(close.iloc[-1]),
                "total_return_pct": safe_round((close.iloc[-1] / close.iloc[0] - 1) * 100),
                "annualised_volatility_pct": safe_round(rets.std() * np.sqrt(252) * 100),
            })
        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)})

    if len(series) < 2:
        raise HTTPException(status_code=404, detail=f"Could not load at least 2 stocks. Details: {errors}")

    df = pd.DataFrame(series).sort_index().ffill().dropna()
    rows = []
    for date, r in df.iterrows():
        row = {"date": date.strftime("%Y-%m-%d")}
        for col in df.columns:
            row[col] = safe_round(r[col])
        rows.append(row)
    return {"symbols": list(df.columns), "months": months, "data": rows, "summary": summary, "errors": errors}


# ── 4. BACKTESTING ENGINE ───────────────────────────────────────────────────

STRATEGIES = {
    "buy_and_hold": "Buy on day 1 and hold. The benchmark every strategy must beat.",
    "ma_crossover": "Long when the fast MA is above the slow MA (default 50/200 'golden cross'); cash otherwise.",
    "rsi_mean_reversion": "Buy when RSI falls below the buy level (default 30); sell when it rises above the sell level (default 70).",
    "macd_crossover": "Long while the MACD line is above its signal line; cash otherwise.",
    "bollinger_reversion": "Buy when price closes below the lower Bollinger Band; sell when it recovers to the middle band.",
}

@app.get("/backtest-strategies")
def backtest_strategies():
    return {"strategies": [{"id": k, "description": v} for k, v in STRATEGIES.items()]}


def _build_positions(strategy, close, fast, slow, rsi_period, rsi_buy, rsi_sell):
    """Returns a 0/1 series: 1 = holding the stock at the close of that day, 0 = in cash."""
    if strategy == "buy_and_hold":
        return pd.Series(1, index=close.index)

    if strategy == "ma_crossover":
        return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(int)

    if strategy == "macd_crossover":
        line, sig, _ = calculate_macd(close)
        return (line > sig).astype(int)

    if strategy == "rsi_mean_reversion":
        rsi = calculate_rsi(close, rsi_period).values
        out, state = [], 0
        for v in rsi:
            if not np.isnan(v):
                if state == 0 and v < rsi_buy:
                    state = 1
                elif state == 1 and v > rsi_sell:
                    state = 0
            out.append(state)
        return pd.Series(out, index=close.index)

    if strategy == "bollinger_reversion":
        mid = close.rolling(20).mean()
        lower = mid - 2 * close.rolling(20).std()
        out, state = [], 0
        for c, m, l in zip(close.values, mid.values, lower.values):
            if not (np.isnan(m) or np.isnan(l)):
                if state == 0 and c < l:
                    state = 1
                elif state == 1 and c > m:
                    state = 0
            out.append(state)
        return pd.Series(out, index=close.index)

    raise HTTPException(status_code=400, detail=f"Unknown strategy. Choose from: {list(STRATEGIES)}")


def _metrics_from_returns(rets, equity, capital, dates):
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    final = float(equity.iloc[-1])
    vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0
    cagr = (final / capital) ** (1 / years) - 1 if final > 0 else -1.0
    sharpe = ((rets.mean() * 252 - 0.06) / vol) if vol > 0 else None      # 6% approx. Indian risk-free rate
    max_dd = float((equity / equity.cummax() - 1).min())
    return {
        "final_value": safe_round(final),
        "total_return_pct": safe_round((final / capital - 1) * 100),
        "cagr_pct": safe_round(cagr * 100),
        "annualised_volatility_pct": safe_round(vol * 100),
        "sharpe_ratio": safe_round(sharpe) if sharpe is not None else None,
        "max_drawdown_pct": safe_round(max_dd * 100),
    }


def _run_backtest(hist, strategy, months, fast, slow, rsi_period, rsi_buy, rsi_sell, cost_pct, capital):
    close_full = hist["Close"].dropna().copy()
    close_full.index = close_full.index.tz_localize(None).normalize()

    positions_full = _build_positions(strategy, close_full, fast, slow, rsi_period, rsi_buy, rsi_sell)

    cutoff = close_full.index[-1] - pd.DateOffset(months=months)
    mask = close_full.index >= cutoff
    close = close_full[mask]
    pos = positions_full[mask]
    if len(close) < 30:
        raise HTTPException(status_code=400, detail="Not enough data in the chosen window to backtest.")

    ret = close.pct_change().fillna(0)
    # signal known at today's close -> position is held from tomorrow's return onward (no look-ahead)
    held = pos.shift(1).fillna(0)
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    strat_ret = held * ret - turnover * (cost_pct / 100)

    equity = capital * (1 + strat_ret).cumprod()
    bh_equity = capital * close / close.iloc[0]

    # Trade list (entries/exits at the closing price of the signal day)
    trades, entry_date, entry_price = [], None, None
    prev = 0
    for date, p, c in zip(close.index, pos.values, close.values):
        if p == 1 and prev == 0:
            entry_date, entry_price = date, float(c)
        elif p == 0 and prev == 1 and entry_date is not None:
            trades.append({
                "entry_date": entry_date.strftime("%Y-%m-%d"), "entry_price": safe_round(entry_price),
                "exit_date": date.strftime("%Y-%m-%d"), "exit_price": safe_round(c),
                "return_pct": safe_round((float(c) / entry_price - 1) * 100 - 2 * cost_pct),
                "open": False,
            })
            entry_date = None
        prev = int(p)
    if entry_date is not None:   # still holding at the end
        c = float(close.iloc[-1])
        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"), "entry_price": safe_round(entry_price),
            "exit_date": close.index[-1].strftime("%Y-%m-%d"), "exit_price": safe_round(c),
            "return_pct": safe_round((c / entry_price - 1) * 100 - cost_pct),
            "open": True,
        })

    closed = [t for t in trades if not t["open"]]
    wins = [t for t in closed if t["return_pct"] is not None and t["return_pct"] > 0]
    metrics = _metrics_from_returns(strat_ret, equity, capital, close.index)
    metrics.update({
        "num_trades": len(trades),
        "win_rate_pct": safe_round(len(wins) / len(closed) * 100) if closed else None,
        "time_in_market_pct": safe_round(float(held.mean()) * 100),
    })
    bh_metrics = _metrics_from_returns(close.pct_change().fillna(0), bh_equity, capital, close.index)

    curve = [{
        "date": d.strftime("%Y-%m-%d"),
        "strategy": safe_round(e),
        "buy_and_hold": safe_round(b),
        "close": safe_round(c),
        "in_position": int(p),
    } for d, e, b, c, p in zip(close.index, equity.values, bh_equity.values, close.values, held.values)]

    return {"metrics": metrics, "buy_and_hold_metrics": bh_metrics, "trades": trades, "equity_curve": curve}


def _load_backtest_history(ticker):
    hist = yf.Ticker(ticker).history(period="8y", auto_adjust=True)   # extra years so slow indicators have warm-up data
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")
    return hist


@app.get("/backtest/{symbol}")
def backtest(
    symbol: str,
    strategy: str = "ma_crossover",
    months: int = 36,          # window to test over: 6 .. 60 months
    fast: int = 50,            # MA crossover: fast window
    slow: int = 200,           # MA crossover: slow window
    rsi_period: int = 14,
    rsi_buy: float = 30,
    rsi_sell: float = 70,
    cost_pct: float = 0.1,     # transaction cost per trade side, in %
    capital: float = 100000,   # starting capital in INR
):
    """
    Long-only backtest of ONE strategy vs buy-and-hold.
    Strategies: buy_and_hold, ma_crossover, rsi_mean_reversion, macd_crossover, bollinger_reversion
    Example: /backtest/RELIANCE?strategy=rsi_mean_reversion&months=36
    """
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy. Choose from: {list(STRATEGIES)}")
    months = min(max(months, 6), 60)
    if fast >= slow:
        raise HTTPException(status_code=400, detail="'fast' must be smaller than 'slow'.")
    ticker = to_ticker(symbol)
    try:
        hist = _load_backtest_history(ticker)
        result = _run_backtest(hist, strategy, months, fast, slow, rsi_period, rsi_buy, rsi_sell, cost_pct, capital)
        return {
            "symbol": ticker, "strategy": strategy, "description": STRATEGIES[strategy],
            "months": months, "starting_capital": capital, "cost_pct_per_side": cost_pct,
            "assumptions": "Long-only, one position at a time, trades at the closing price of the signal day, "
                           "no look-ahead bias. Past performance does not predict future results. Educational use only.",
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error for {ticker}: {str(e)}")


@app.get("/backtest-all/{symbol}")
def backtest_all(symbol: str, months: int = 36, cost_pct: float = 0.1, capital: float = 100000):
    """Runs every strategy with default settings: leaderboard table + one combined equity-curve chart."""
    months = min(max(months, 6), 60)
    ticker = to_ticker(symbol)
    try:
        hist = _load_backtest_history(ticker)
        leaderboard, curves = [], {}
        for strat in STRATEGIES:
            r = _run_backtest(hist, strat, months, 50, 200, 14, 30, 70, cost_pct, capital)
            leaderboard.append({"strategy": strat, **r["metrics"]})
            curves[strat] = {row["date"]: row["strategy"] for row in r["equity_curve"]}
        leaderboard.sort(key=lambda x: (x["total_return_pct"] is None, -(x["total_return_pct"] or 0)))
        all_dates = sorted(set().union(*[set(c) for c in curves.values()]))
        rows = [{"date": d, **{k: curves[k].get(d) for k in curves}} for d in all_dates]
        return {"symbol": ticker, "months": months, "starting_capital": capital,
                "leaderboard": leaderboard, "equity_curves": rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error for {ticker}: {str(e)}")
