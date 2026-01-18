#!/usr/bin/env python3
"""
alpaca_hourly_screener.py (ALPACA universe + IEX feed + ALWAYS emails a list + EMAIL FAILURES NON-FATAL)

Key fixes included:
✅ NO Wikipedia scraping (no 403)
✅ Universe from Alpaca assets (best-effort ETF exclusion)
✅ Alpaca Free compatible (forces DataFeed.IEX)
✅ ALWAYS builds an email summary with Top Candidates (even if 0 alerts)
✅ Email sending is NON-FATAL (SMTP timeouts won’t fail the GitHub Action)
✅ Clear log lines so you can confirm email attempt/errors in Actions logs

Required env vars (GitHub secrets):
  APCA_API_KEY_ID
  APCA_API_SECRET_KEY

Email env vars (optional, GitHub secrets):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_TO_EMAIL

Optional Twilio env vars:
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO

DISCLAIMER: Informational only. Not financial advice.
"""

from __future__ import annotations

import os
import time
import argparse
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import schedule
import smtplib
from email.mime.text import MIMEText

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = {
    # Scan settings
    "scan_interval_minutes": 60,
    "bars_timeframe": TimeFrame.Hour,
    "bars_limit": 600,              # ~25 days of 1H bars
    "min_bars_required": 140,       # enough for indicators + lookbacks

    # Universe/runtime safety
    "max_symbols_per_run": 1200,    # reduce if Actions runtime is too long
    "throttle_every": 200,
    "throttle_sleep": 0.5,

    # Filters
    "min_price": 10.0,
    "min_avg_dollar_vol_20h": 5_000_000,  # avg(volume*close) last 20 hourly bars

    # Trend/Momentum
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_min": 45,
    "rsi_max": 75,
    "require_macd_hist_positive": True,

    # Breakout logic
    "breakout_lookback_hours": 40,
    "breakout_near_pct": 0.003,     # within 0.3% of prior high = "near"
    "breakout_buffer_pct": 0.001,   # trigger just above prior high
    "volume_ratio_min": 1.3,        # last vol / vol_ma20

    # Pullback logic
    "pullback_near_ema_pct": 0.005, # within 0.5% of EMA20
    "require_green_candle": True,

    # Mean reversion logic
    "vwap_lookback_hours": 30,
    "vwap_far_pct": 0.015,          # 1.5% below VWAP
    "reversal_wick_ratio_min": 1.2, # lower wick/body

    # Target / Risk (informational)
    "target_pct": 0.10,             # +10% target
    "atr_period": 14,
    "stop_atr_mult": 1.5,

    # Output
    "alerts_csv": "alerts_log.csv",
    "top_candidates_n": 25,

    # Email network timeout (seconds)
    "smtp_timeout": 20,
}


# ---------------------------
# Indicators
# ---------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - sig

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def vwap_rolling(df: pd.DataFrame, lookback: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.rolling(lookback).sum() / df["volume"].rolling(lookback).sum()

def wick_body_ratios(row: pd.Series) -> Tuple[float, float]:
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    body = body if body > 1e-9 else 1e-9
    return upper / body, lower / body


# ---------------------------
# Email / SMS
# ---------------------------

def send_email(subject: str, body: str) -> None:
    host = os.getenv("SMTP_HOST")
    port_raw = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    to_email = os.getenv("ALERT_TO_EMAIL")

    if not all([host, port_raw, user, pwd, to_email]):
        print("Email skipped (SMTP secrets missing).")
        return

    port = int(port_raw)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email

    # IMPORTANT: add timeout so it doesn't hang forever
    with smtplib.SMTP(host, port, timeout=int(CONFIG["smtp_timeout"])) as s:
        s.starttls()
        s.login(user, pwd)
        s.sendmail(user, [to_email], msg.as_string())

def send_sms_twilio(body: str) -> None:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_ = os.getenv("TWILIO_FROM")
    to_ = os.getenv("TWILIO_TO")
    if not all([sid, token, from_, to_]):
        return
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    r = requests.post(url, data={"From": from_, "To": to_, "Body": body}, auth=(sid, token), timeout=20)
    if r.status_code >= 300:
        print("Twilio SMS error:", r.status_code, r.text[:250])


# ---------------------------
# CSV logging
# ---------------------------

def append_alerts_csv(alerts: List[Dict], path: str) -> None:
    if not alerts:
        return
    df = pd.DataFrame(alerts)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)


# ---------------------------
# Alpaca clients
# ---------------------------

def make_clients() -> Tuple[StockHistoricalDataClient, TradingClient]:
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY secrets.")
    data_client = StockHistoricalDataClient(key, secret)
    trading_client = TradingClient(key, secret, paper=True)  # uses PAPER keys
    return data_client, trading_client


# ---------------------------
# Universe: Alpaca assets (best-effort ETF exclusion)
# ---------------------------

ETF_NAME_HINTS = [
    " ETF", "TRUST", "FUND", "INDEX", "SPDR", "ISHARES", "VANGUARD", "INVESCO",
    "PROSHARES", "WISDOMTREE", "VANECK", "DIREXION", "GLOBAL X", "ARK",
]

def looks_like_etf(name: str) -> bool:
    if not name:
        return False
    u = name.upper()
    if " ETF" in u or u.endswith("ETF"):
        return True
    return any(h in u for h in ETF_NAME_HINTS)

def build_universe_from_alpaca(trading: TradingClient) -> List[str]:
    assets = trading.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    universe: List[str] = []

    for a in assets:
        if not getattr(a, "tradable", False):
            continue

        sym = getattr(a, "symbol", None)
        if not sym:
            continue

        if any(ch in sym for ch in ["^", "/", " "]):
            continue

        # exclude warrants/rights/units best-effort by suffix
        if sym.endswith(("W", "WS", "R", "U")):
            continue

        name = getattr(a, "name", "") or ""
        if looks_like_etf(name):
            continue

        universe.append(sym)

    universe = sorted(set(universe))
    cap = int(CONFIG["max_symbols_per_run"] or 0)
    if cap > 0 and len(universe) > cap:
        universe = universe[:cap]
    return universe


# ---------------------------
# Data fetch
# ---------------------------

def get_hourly_bars(data_client: StockHistoricalDataClient, symbol: str) -> Optional[pd.DataFrame]:
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=CONFIG["bars_timeframe"],
        limit=CONFIG["bars_limit"],
        feed=DataFeed.IEX,  # Alpaca Free compatible
    )
    bars = data_client.get_stock_bars(req)
    if bars is None:
        return None

    df = bars.df
    if df is None or df.empty:
        return None

    if isinstance(df.index, pd.MultiIndex):
        if symbol not in df.index.get_level_values(0):
            return None
        df = df.xs(symbol, level=0)

    df = df.sort_index().dropna()
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(set(df.columns)):
        return None

    return df


# ---------------------------
# Candidate scoring (for always-on email list)
# ---------------------------

def score_candidate(df: pd.DataFrame) -> Optional[Dict]:
    if df is None or df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])

    ema20_v = float(last["ema20"])
    ema50_v = float(last["ema50"])
    rsi_v = float(last["rsi14"])
    macd_h_v = float(last["macd_h"])
    vwap_v = float(last["vwap"]) if pd.notna(last["vwap"]) else np.nan

    vol = float(last["volume"])
    vol_ma20 = float(last["vol_ma20"]) if pd.notna(last["vol_ma20"]) else 0.0
    vol_ratio = (vol / vol_ma20) if vol_ma20 > 0 else 0.0

    uptrend = (price > ema50_v) and (ema20_v > ema50_v)

    lb = CONFIG["breakout_lookback_hours"]
    dist_to_high = np.nan
    if len(df) > lb + 2:
        prior_high = float(df["high"].iloc[-(lb + 1):-1].max())
        dist_to_high = (prior_high - price) / prior_high

    dist_vwap = (price - vwap_v) / vwap_v if (not np.isnan(vwap_v) and vwap_v > 0) else np.nan

    score = 0.0
    score += 2.0 if uptrend else 0.0
    score += min(max(vol_ratio, 0.0), 5.0) * 0.6
    score += 1.0 if macd_h_v > 0 else 0.0

    if 40 <= rsi_v <= 80:
        score += (1.0 - abs(rsi_v - 65) / 25.0)

    if not np.isnan(dist_to_high):
        score += max(0.0, 1.2 - (dist_to_high * 40.0))

    return {
        "price": round(price, 2),
        "uptrend": bool(uptrend),
        "rsi": round(rsi_v, 1),
        "macd_h": round(macd_h_v, 4),
        "vol_ratio": round(vol_ratio, 2),
        "dist_to_high_pct": round(dist_to_high * 100, 2) if not np.isnan(dist_to_high) else None,
        "dist_to_vwap_pct": round(dist_vwap * 100, 2) if not np.isnan(dist_vwap) else None,
        "score": round(score, 3),
    }


# ---------------------------
# Signal detection
# ---------------------------

def analyze_symbol(data_client: StockHistoricalDataClient, symbol: str) -> Tuple[List[Dict], Optional[Dict]]:
    df = get_hourly_bars(data_client, symbol)
    if df is None or len(df) < CONFIG["min_bars_required"]:
        return [], None

    df["ema20"] = ema(df["close"], CONFIG["ema_fast"])
    df["ema50"] = ema(df["close"], CONFIG["ema_slow"])
    df["rsi14"] = rsi(df["close"], CONFIG["rsi_period"])
    df["macd_h"] = macd_hist(df["close"])
    df["atr14"] = atr(df, CONFIG["atr_period"])
    df["vwap"] = vwap_rolling(df, CONFIG["vwap_lookback_hours"])
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    last = df.iloc[-1]
    price = float(last["close"])
    if price < CONFIG["min_price"]:
        return [], None

    avg_dollar_vol_20h = float((df["volume"].tail(20) * df["close"].tail(20)).mean())
    if avg_dollar_vol_20h < CONFIG["min_avg_dollar_vol_20h"]:
        return [], None

    ema20_v = float(last["ema20"])
    ema50_v = float(last["ema50"])
    rsi_v = float(last["rsi14"])
    macd_h_v = float(last["macd_h"])
    atr_v = float(last["atr14"]) if pd.notna(last["atr14"]) else np.nan
    vwap_v = float(last["vwap"]) if pd.notna(last["vwap"]) else np.nan

    uptrend = (price > ema50_v) and (ema20_v > ema50_v)
    momentum_ok = (CONFIG["rsi_min"] <= rsi_v <= CONFIG["rsi_max"])
    if CONFIG["require_macd_hist_positive"]:
        momentum_ok = momentum_ok and (macd_h_v > 0)

    vol = float(last["volume"])
    vol_ma20 = float(last["vol_ma20"]) if pd.notna(last["vol_ma20"]) else 0.0
    vol_ratio = (vol / vol_ma20) if vol_ma20 > 0 else 0.0

    now_ts = str(df.index[-1])
    alerts: List[Dict] = []

    cand = score_candidate(df)

    # BREAKOUT
    lb = CONFIG["breakout_lookback_hours"]
    if len(df) > lb + 2:
        prior_high = float(df["high"].iloc[-(lb + 1):-1].max())
        near = ((prior_high - price) / prior_high) <= CONFIG["breakout_near_pct"]
        broke = price > prior_high
        if uptrend and momentum_ok and (near or broke) and (vol_ratio >= CONFIG["volume_ratio_min"]):
            trigger = prior_high * (1 + CONFIG["breakout_buffer_pct"])
            entry = trigger
            target = entry * (1 + CONFIG["target_pct"])
            stop = (entry - CONFIG["stop_atr_mult"] * atr_v) if not np.isnan(atr_v) else None
            alerts.append({
                "timestamp": now_ts,
                "symbol": symbol,
                "alert_type": "BREAKOUT",
                "price": round(price, 4),
                "entry_trigger": round(entry, 4),
                "target_10pct": round(target, 4),
                "stop_atr": round(stop, 4) if stop else None,
                "vol_ratio": round(vol_ratio, 2),
                "notes": f"prior {lb}h high={prior_high:.2f}"
            })

    # TREND PULLBACK
    dist_ema20 = abs(price - ema20_v) / ema20_v if ema20_v else 1.0
    green = float(last["close"]) > float(last["open"])
    if uptrend and momentum_ok and (dist_ema20 <= CONFIG["pullback_near_ema_pct"]) and (not CONFIG["require_green_candle"] or green):
        entry = price
        target = entry * (1 + CONFIG["target_pct"])
        stop = (entry - CONFIG["stop_atr_mult"] * atr_v) if not np.isnan(atr_v) else None
        alerts.append({
            "timestamp": now_ts,
            "symbol": symbol,
            "alert_type": "TREND_PULLBACK",
            "price": round(price, 4),
            "entry_trigger": round(entry, 4),
            "target_10pct": round(target, 4),
            "stop_atr": round(stop, 4) if stop else None,
            "vol_ratio": round(vol_ratio, 2),
            "notes": f"near EMA20 ({dist_ema20*100:.2f}%)"
        })

    # MEAN REVERSION UP
    if not np.isnan(vwap_v) and vwap_v > 0:
        dist_vwap = (price - vwap_v) / vwap_v
        _, lower_w = wick_body_ratios(last)
        if (dist_vwap <= -CONFIG["vwap_far_pct"]) and (lower_w >= CONFIG["reversal_wick_ratio_min"]):
            entry = price
            target = entry * (1 + CONFIG["target_pct"])
            stop = (entry - CONFIG["stop_atr_mult"] * atr_v) if not np.isnan(atr_v) else None
            alerts.append({
                "timestamp": now_ts,
                "symbol": symbol,
                "alert_type": "MEAN_REVERSION_UP",
                "price": round(price, 4),
                "entry_trigger": round(entry, 4),
                "target_10pct": round(target, 4),
                "stop_atr": round(stop, 4) if stop else None,
                "vol_ratio": round(vol_ratio, 2),
                "notes": f"{dist_vwap*100:.2f}% below VWAP; lower/body={lower_w:.2f}"
            })

    return alerts, cand


# ---------------------------
# Scan runner (ALWAYS emails; EMAIL FAILURES NON-FATAL)
# ---------------------------

def run_scan(data_client: StockHistoricalDataClient, trading_client: TradingClient, universe: List[str]) -> None:
    start = time.time()
    ts = dt.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{ts}] Scanning {len(universe)} symbols (1H)...")

    all_alerts: List[Dict] = []
    candidates: List[Dict] = []
    errors = 0

    for i, sym in enumerate(universe, 1):
        try:
            alerts, cand = analyze_symbol(data_client, sym)

            if cand is not None:
                candidates.append({"symbol": sym, **cand})

            if alerts:
                all_alerts.extend(alerts)
                for a in alerts:
                    print(f"ALERT {a['alert_type']}: {a['symbol']} price={a['price']} entry={a['entry_trigger']} target={a['target_10pct']}")

        except Exception as e:
            errors += 1
            print(f"ERROR {sym}: {e}")

        if CONFIG["throttle_every"] and i % int(CONFIG["throttle_every"]) == 0:
            time.sleep(float(CONFIG["throttle_sleep"]))

    runtime_sec = round(time.time() - start, 2)

    # Log alerts
    append_alerts_csv(all_alerts, CONFIG["alerts_csv"])

    # Rank candidates
    top = []
    if candidates:
        cdf = pd.DataFrame(candidates).sort_values("score", ascending=False).head(int(CONFIG["top_candidates_n"]))
        top = cdf.to_dict(orient="records")

    # Build summary
    lines: List[str] = []
    lines.append(f"Scan time: {ts}")
    lines.append(f"Universe scanned: {len(universe)}")
    lines.append(f"Runtime: {runtime_sec}s")
    lines.append(f"Errors: {errors}")
    lines.append(f"Alerts found: {len(all_alerts)}")
    lines.append("")

    if all_alerts:
        lines.append("=== ALERTS (triggered) ===")
        for a in all_alerts[:60]:
            lines.append(
                f"{a['alert_type']} {a['symbol']} | price={a['price']} | entry={a['entry_trigger']} | "
                f"target10={a['target_10pct']} | stop={a.get('stop_atr')} | volR={a.get('vol_ratio')} | {a.get('notes','')}"
            )
        lines.append("")

    lines.append(f"=== TOP {CONFIG['top_candidates_n']} CANDIDATES (ranked watchlist) ===")
    if top:
        for c in top:
            lines.append(
                f"{c['symbol']} | score={c['score']} | price={c['price']} | uptrend={c['uptrend']} | "
                f"RSI={c['rsi']} | MACDh={c['macd_h']} | volR={c['vol_ratio']} | "
                f"distHigh%={c.get('dist_to_high_pct')} | distVWAP%={c.get('dist_to_vwap_pct')}"
            )
    else:
        lines.append("No candidates captured (filters too strict or insufficient bars/data).")

    body = "\n".join(lines)

    # ✅ NON-FATAL EMAIL SEND
    try:
        send_email(subject=f"Hourly Screener Summary (alerts={len(all_alerts)})", body=body)
        print("Email summary attempted.")
    except Exception as e:
        print(f"EMAIL ERROR (non-fatal): {e}")

    # Optional SMS only if alerts exist
    try:
        if all_alerts:
            send_sms_twilio(body=f"Alpaca screener: {len(all_alerts)} alerts. Check email.")
    except Exception as e:
        print(f"SMS ERROR (non-fatal): {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Run one scan and exit (best for cloud schedulers)")
    args = ap.parse_args()

    data_client, trading_client = make_clients()
    universe = build_universe_from_alpaca(trading_client)
    print(f"Universe ready (Alpaca assets): {len(universe)} symbols (ETFs excluded best-effort).")

    if args.once:
        run_scan(data_client, trading_client, universe)
        return

    schedule.every(int(CONFIG["scan_interval_minutes"])).minutes.do(
        run_scan, data_client=data_client, trading_client=trading_client, universe=universe
    )

    run_scan(data_client, trading_client, universe)

    while True:
        schedule.run_pending()
        time.sleep(5)


if __name__ == "__main__":
    main()
