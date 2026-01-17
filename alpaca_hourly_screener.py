#!/usr/bin/env python3
"""
Alpaca Hourly Stock Screener
-------------------------------------------------
• Data source: Alpaca (FREE)
• Timeframe: 1-Hour candles
• Universe: S&P 500 + Nasdaq-100
• ETFs excluded (best-effort)
• Alerts:
    1) Breakout
    2) Trend Pullback
    3) Mean Reversion (bullish)
• Target: +10%
• Output: CSV + Email (+ optional SMS)
• Cloud-ready (GitHub Actions)

Informational alerts only. Not financial advice.
"""

import os
import time
import argparse
import datetime as dt
import pandas as pd
import numpy as np
import requests
import schedule
import smtplib
from email.mime.text import MIMEText

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass


# =========================
# CONFIG
# =========================

CONFIG = {
    "scan_interval_minutes": 60,
    "bars_limit": 600,

    "min_price": 10,
    "min_avg_dollar_vol_20h": 5_000_000,

    "ema_fast": 20,
    "ema_slow": 50,

    "rsi_period": 14,
    "rsi_min": 45,
    "rsi_max": 75,

    "breakout_lookback": 40,
    "breakout_near_pct": 0.003,
    "volume_ratio_min": 1.3,

    "pullback_ema_pct": 0.005,

    "vwap_lookback": 30,
    "vwap_far_pct": 0.015,

    "target_pct": 0.10,
    "atr_period": 14,
    "atr_mult": 1.5,

    "alerts_csv": "alerts_log.csv",
}


# =========================
# INDICATORS
# =========================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = -d.clip(upper=0).rolling(n).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))

def macd_hist(close):
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd - signal

def atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rolling_vwap(df, n):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    return pv.rolling(n).sum() / df["volume"].rolling(n).sum()


# =========================
# EMAIL / SMS
# =========================

def send_email(subject, body):
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    to = os.getenv("ALERT_TO_EMAIL")
    port = int(os.getenv("SMTP_PORT", "587"))
    if not all([host, user, pwd, to]):
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pwd)
        s.sendmail(user, [to], msg.as_string())

def send_sms(body):
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    frm = os.getenv("TWILIO_FROM")
    to = os.getenv("TWILIO_TO")
    if not all([sid, token, frm, to]):
        return
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    requests.post(url, data={"From": frm, "To": to, "Body": body}, auth=(sid, token))


# =========================
# UNIVERSE (SP500 + NASDAQ-100)
# =========================

def fetch_sp500():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return [s.replace(".", "-") for s in df["Symbol"]]

def fetch_nasdaq100():
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    for t in tables:
        if "Ticker" in t.columns:
            return [s.replace(".", "-") for s in t["Ticker"]]
    return []


# =========================
# CORE SCAN
# =========================

def analyze_symbol(data, symbol):
    req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Hour, limit=CONFIG["bars_limit"])
    bars = data.get_stock_bars(req).df
    if bars is None or bars.empty:
        return []

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)

    bars = bars.dropna()
    if len(bars) < 100:
        return []

    bars["ema20"] = ema(bars["close"], 20)
    bars["ema50"] = ema(bars["close"], 50)
    bars["rsi"] = rsi(bars["close"])
    bars["macd_h"] = macd_hist(bars["close"])
    bars["atr"] = atr(bars)
    bars["vwap"] = rolling_vwap(bars, CONFIG["vwap_lookback"])
    bars["vol_ma"] = bars["volume"].rolling(20).mean()

    last = bars.iloc[-1]
    price = last["close"]

    if price < CONFIG["min_price"]:
        return []

    dollar_vol = (bars["volume"].tail(20) * bars["close"].tail(20)).mean()
    if dollar_vol < CONFIG["min_avg_dollar_vol_20h"]:
        return []

    alerts = []
    uptrend = price > last["ema50"] and last["ema20"] > last["ema50"]
    momentum = CONFIG["rsi_min"] <= last["rsi"] <= CONFIG["rsi_max"] and last["macd_h"] > 0
    vol_ratio = last["volume"] / last["vol_ma"]

    # BREAKOUT
    high = bars["high"].iloc[-CONFIG["breakout_lookback"]:-1].max()
    if uptrend and momentum and vol_ratio >= CONFIG["volume_ratio_min"]:
        if price >= high * (1 - CONFIG["breakout_near_pct"]):
            entry = high
            alerts.append(("BREAKOUT", entry))

    # PULLBACK
    if uptrend and momentum and abs(price - last["ema20"]) / last["ema20"] <= CONFIG["pullback_ema_pct"]:
        alerts.append(("PULLBACK", price))

    # MEAN REVERSION
    if price < last["vwap"] * (1 - CONFIG["vwap_far_pct"]):
        alerts.append(("MEAN_REVERSION", price))

    results = []
    for typ, entry in alerts:
        target = entry * (1 + CONFIG["target_pct"])
        stop = entry - last["atr"] * CONFIG["atr_mult"]
        results.append({
            "timestamp": str(bars.index[-1]),
            "symbol": symbol,
            "type": typ,
            "price": round(price, 2),
            "entry": round(entry, 2),
            "target_10pct": round(target, 2),
            "stop": round(stop, 2)
        })

    return results


# =========================
# RUNNER
# =========================

def run_once():
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing Alpaca API keys")

    data = StockHistoricalDataClient(key, secret)
    trading = TradingClient(key, secret, paper=True)

    universe = set(fetch_sp500() + fetch_nasdaq100())
    assets = trading.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    tradable = {a.symbol for a in assets if a.tradable}

    universe = sorted(universe & tradable)
    all_alerts = []

    for sym in universe:
        try:
            alerts = analyze_symbol(data, sym)
            all_alerts.extend(alerts)
        except:
            pass

    if not all_alerts:
        print("No alerts this run.")
        return

    df = pd.DataFrame(all_alerts)
    df.to_csv(CONFIG["alerts_csv"], mode="a", index=False, header=not os.path.exists(CONFIG["alerts_csv"]))

    body = "\n".join(
        f"{a['type']} {a['symbol']} entry={a['entry']} target={a['target_10pct']}"
        for a in all_alerts
    )
    send_email(f"Screener Alerts ({len(all_alerts)})", body)
    send_sms(f"{len(all_alerts)} screener alerts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        schedule.every(CONFIG["scan_interval_minutes"]).minutes.do(run_once)
        run_once()
        while True:
            schedule.run_pending()
            time.sleep(5)


if __name__ == "__main__":
    main()

