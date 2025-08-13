# ===== Patch for Python 3.13 imghdr removal (needed by python-telegram-bot==13.15) =====
import sys, types
if "imghdr" not in sys.modules:
    imghdr = types.ModuleType("imghdr")
    imghdr.what = lambda file, h=None: None
    sys.modules["imghdr"] = imghdr
# ========================================================================================

import os
import logging
import threading
import time
import sqlite3
import math
import random
from datetime import datetime, timezone
from flask import Flask, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import ccxt
from telegram import Bot

# ========= Config (env overrides) =========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8384498061:AAElt7HeM88jfune948IcKkysHpw1tmXrlc")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "1040990874")

# core cadence
TIMEFRAME = os.environ.get("TIMEFRAME", "1h")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "120"))  # seconds

# broad scanning controls
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))                 # pairs per cycle
SLEEP_BETWEEN_CALLS = float(os.environ.get("SLEEP_BETWEEN_CALLS", "0.5"))
JITTER_MAX = float(os.environ.get("JITTER_MAX", "0.25"))             # random 0..JITTER_MAX sec extra between calls
MAX_PAIRS_PER_EXCHANGE = int(os.environ.get("MAX_PAIRS_PER_EXCHANGE", "220"))
OHLCV_LIMIT = int(os.environ.get("OHLCV_LIMIT", "60"))               # 60 candles (enough for EMAs/ATR/MFI/etc.)

# streak bounds
MIN_GREEN_CANDLES = int(os.environ.get("MIN_GREEN_CANDLES", "4"))
MAX_GREEN_CANDLES = int(os.environ.get("MAX_GREEN_CANDLES", "9"))

# liquidity filter via cached tickers
ENABLE_LIQ_FILTER = os.environ.get("ENABLE_LIQ_FILTER", "1") == "1"
LIQ_MIN_USD = float(os.environ.get("LIQ_MIN_USD", "2000000"))  # $2M
TICKERS_TTL = int(os.environ.get("TICKERS_TTL", "900"))        # seconds; 15 min

# probability threshold for alerts (tighter default)
PROB_THRESHOLD = float(os.environ.get("PROB_THRESHOLD", "0.70"))

# anti-ban engine
BAN_COOLDOWN_SECONDS = int(os.environ.get("BAN_COOLDOWN_SECONDS", "2700"))  # base 45m
ROTATE_EXCHANGE_EACH_CYCLE = os.environ.get("ROTATE_EXCHANGE_EACH_CYCLE", "1") == "1"

# per-exchange budgets (requests per minute)
RPM_BINANCE = int(os.environ.get("RPM_BINANCE", "60"))
RPM_BYBIT   = int(os.environ.get("RPM_BYBIT",   "60"))
RPM_OKX     = int(os.environ.get("RPM_OKX",     "60"))

# symbol alert cooldown
SYMBOL_ALERT_COOLDOWN = int(os.environ.get("SYMBOL_ALERT_COOLDOWN", "3600"))  # 1 hour

# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crypto-bot")

# ========= SQLite (no ORM) =========
conn = sqlite3.connect("alerts.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    price REAL,
    green_candles INTEGER,
    breakout_probability REAL
)
""")
conn.commit()

def save_alert(message, price, green_candles, probability):
    cur.execute(
        "INSERT INTO alerts (message, timestamp, price, green_candles, breakout_probability) VALUES (?, ?, ?, ?, ?)",
        (message, datetime.now(timezone.utc).isoformat(), price, green_candles, probability),
    )
    conn.commit()

def get_recent_alerts(limit=10):
    cur.execute(
        "SELECT id, message, timestamp, price, green_candles, breakout_probability FROM alerts ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return cur.fetchall()

# ========= Notifier (anti-spam cooldowns) =========
class Notifier:
    def __init__(self, token, chat_id):
        self.enabled = bool(token and chat_id)
        self.bot = Bot(token=token) if self.enabled else None
        self.chat_id = chat_id
        self.last = {}  # key -> last epoch sent

    def send(self, key, text, cooldown=900):
        if not self.enabled:
            return
        now = time.time()
        last = self.last.get(key, 0)
        if now - last >= cooldown:
            try:
                self.bot.send_message(chat_id=self.chat_id, text=text)
                self.last[key] = now
            except Exception as e:
                logger.error(f"Notifier send error: {e}")

notifier = Notifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

# ========= Indicator helpers (pure Python) =========
def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        hi = candles[-i]["high"]
        lo = candles[-i]["low"]
        pc = candles[-i-1]["close"]
        tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
        trs.append(tr)
    return sum(trs) / period

def calculate_rsi(candles, period=14):
    if len(candles) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        change = candles[i]["close"] - candles[i-1]["close"]
        if change > 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def calculate_volume_surge(candles, lookback=20, surge_factor=1.6):
    if len(candles) < lookback + 1:
        return False
    avg_vol = sum(c["volume"] for c in candles[-lookback-1:-1]) / lookback
    return candles[-1]["volume"] > avg_vol * surge_factor

def near_resistance(candles, lookback=30, tolerance=0.004):
    if len(candles) < lookback + 1:
        return False
    resistance = max(c["close"] for c in candles[-lookback-1:-1])
    return candles[-1]["close"] >= resistance * (1 - tolerance)

def sma(vals, period):
    if len(vals) < period: return None
    return sum(vals[-period:]) / period

def ema(vals, period):
    if len(vals) < period: return None
    k = 2 / (period + 1)
    e = vals[-period]
    for v in vals[-period+1:]:
        e = v * k + e * (1 - k)
    return e

def stddev(vals, period):
    if len(vals) < period: return None
    subset = vals[-period:]
    m = sum(subset) / period
    var = sum((x - m) ** 2 for x in subset) / period
    return var ** 0.5

def bollinger_bands(closes, period=20, mult=2.0):
    m = sma(closes, period); s = stddev(closes, period)
    if m is None or s is None: return None
    upper, lower = m + mult * s, m - mult * s
    width = (upper - lower) / (m if m else 1)
    return {"mid": m, "upper": upper, "lower": lower, "width": width}

def keltner_channels(candles, period=20, mult=1.5):
    closes = [c["close"] for c in candles]
    mid = ema(closes, period)
    atr = calculate_atr(candles, period)
    if mid is None or atr == 0: return None
    upper, lower = mid + mult * atr, mid - mult * atr
    width = (upper - lower) / (mid if mid else 1)
    return {"mid": mid, "upper": upper, "lower": lower, "width": width}

def is_squeeze_on(candles, bb_period=20, kc_period=20, bb_mult=2.0, kc_mult=1.5):
    closes = [c["close"] for c in candles]
    bb = bollinger_bands(closes, bb_period, bb_mult)
    kc = keltner_channels(candles, kc_period, kc_mult)
    if not bb or not kc: return False
    return (bb["upper"] < kc["upper"]) and (bb["lower"] > kc["lower"])

def nr7(candles, lookback=7):
    if len(candles) < lookback: return False
    ranges = [(c["high"] - c["low"]) for c in candles[-lookback:]]
    return ranges[-1] == min(ranges)

def donchian_breakout(candles, lookback=20):
    if len(candles) < lookback + 1: return False
    prior_high = max(c["high"] for c in candles[-lookback-1:-1])
    return candles[-1]["close"] > prior_high

def candle_close_position(c):
    rng = (c["high"] - c["low"])
    if rng <= 0: return 0.5
    return (c["close"] - c["low"]) / rng  # 0 bottom, 1 top

def obv_slope(candles, lookback=20):
    if len(candles) < lookback + 1: return 0.0
    obv = [0.0]
    for i in range(1, len(candles)):
        if candles[i]["close"] > candles[i-1]["close"]:
            obv.append(obv[-1] + candles[i]["volume"])
        elif candles[i]["close"] < candles[i-1]["close"]:
            obv.append(obv[-1] - candles[i]["volume"])
        else:
            obv.append(obv[-1])
    num = obv[-1] - obv[-lookback-1]
    denom = max(1.0, sum(c["volume"] for c in candles[-lookback:]))
    return num / denom

def mfi(candles, period=14):
    if len(candles) < period + 1: return 50.0
    pos_flow = neg_flow = 0.0
    for i in range(-period, 0):
        tp_now = (candles[i]["high"] + candles[i]["low"] + candles[i]["close"]) / 3
        tp_prev = (candles[i-1]["high"] + candles[i-1]["low"] + candles[i-1]["close"]) / 3
        flow = tp_now * candles[i]["volume"]
        if tp_now > tp_prev: pos_flow += flow
        else: neg_flow += flow
    if neg_flow == 0: return 100.0
    mr = pos_flow / neg_flow
    return 100 - (100 / (1 + mr))

def anchored_vwap(candles, lookback=30):
    if len(candles) < lookback:
        return None
    num = 0.0
    den = 0.0
    for c in candles[-lookback:]:
        tp = (c["high"] + c["low"] + c["close"]) / 3
        v = c["volume"]
        num += tp * v
        den += v
    if den <= 0:
        return None
    return num / den

# ========= Liquidity from tickers =========
def est_quote_usd_volume(ticker: dict) -> float:
    try:
        qv = ticker.get("quoteVolume")
        if qv is not None:
            return float(qv)
    except Exception:
        pass
    try:
        base = ticker.get("baseVolume")
        last = ticker.get("last")
        if base is not None and last is not None:
            return float(base) * float(last)
    except Exception:
        pass
    info = ticker.get("info") or {}
    for k in ("qv", "quoteVolume", "quote_volume", "volValue", "turnover"):
        if k in info:
            try: return float(info[k])
            except Exception: pass
    return 0.0

# ========= Token Bucket & Adaptive Budget =========
class TokenBucket:
    def __init__(self, capacity, refill_per_sec):
        self.capacity = max(1, capacity)
        self.tokens = self.capacity
        self.refill_per_sec = max(0.1, refill_per_sec)
        self.last = time.time()
    def refill(self):
        now = time.time()
        delta = now - self.last
        if delta > 0:
            self.tokens = min(self.capacity, self.tokens + delta * self.refill_per_sec)
            self.last = now
    def available(self):
        self.refill()
        return math.floor(self.tokens)
    def consume(self, n=1):
        self.refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

class AdaptiveBudget:
    def __init__(self, base_rpm, min_factor=0.3, max_factor=1.4):
        self.base_rpm = base_rpm
        self.cur_rpm = base_rpm
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.ban_count = 0
        self.stable_cycles = 0
    def on_ban(self):
        self.ban_count += 1
        self.stable_cycles = 0
        self.cur_rpm = max(1, int(self.cur_rpm * 0.7))
    def on_stable(self):
        self.stable_cycles += 1
        if self.stable_cycles >= 3:
            self.cur_rpm = min(int(self.cur_rpm * 1.1), int(self.base_rpm * self.max_factor))
            self.stable_cycles = 0
    def per_min(self):
        return int(max(1, self.cur_rpm))

# ========= Bot status =========
bot_status = {
    "is_running": False,
    "last_check": None,
    "alerts_sent": 0,
    "last_alert": None,
    "errors": [],
}

# ========= Core Monitor =========
class CryptoMonitor:
    def __init__(self, status_callback, notifier: Notifier):
        self.notifier = notifier
        self.bot = notifier.bot if notifier.enabled else None
        self.chat_id = notifier.chat_id if notifier.enabled else None
        self.status_callback = status_callback

        self.exchanges = [
            {"name": "binance", "params": {"timeout": 15000}, "banned_until": 0, "budget": AdaptiveBudget(RPM_BINANCE)},
            {"name": "bybit",   "params": {"timeout": 15000}, "banned_until": 0, "budget": AdaptiveBudget(RPM_BYBIT)},
            {"name": "okx",     "params": {"timeout": 15000}, "banned_until": 0, "budget": AdaptiveBudget(RPM_OKX)},
        ]
        self.exchange_index = 0

        self.symbols_by_exchange = {}         # ex_id -> [USDT symbols]
        self.batch_idx_by_exchange = {}       # ex_id -> int
        self.tickers_cache = {}               # ex_id -> {"ts": epoch, "data": dict}

        self.buckets = {
            "binance": TokenBucket(capacity=max(1, RPM_BINANCE), refill_per_sec=max(1, RPM_BINANCE)/60.0),
            "bybit":   TokenBucket(capacity=max(1, RPM_BYBIT),   refill_per_sec=max(1, RPM_BYBIT)/60.0),
            "okx":     TokenBucket(capacity=max(1, RPM_OKX),     refill_per_sec=max(1, RPM_OKX)/60.0),
        }

        self.last_candles = []
        self.had_all_failed = False
        self.last_alert_time = {}             # symbol -> last epoch

    def _now(self): return time.time()

    def get_exchange(self, name, params):
        ex = getattr(ccxt, name)(params)
        ex.enableRateLimit = True
        ex.load_markets()
        return ex

    def _is_banned(self, exmeta): return self._now() < exmeta["banned_until"]

    def _ban_exchange(self, exmeta, reason):
        budget: AdaptiveBudget = exmeta["budget"]
        budget.on_ban()
        factor = 2 ** min(budget.ban_count, 3)  # up to 8x
        backoff = BAN_COOLDOWN_SECONDS * factor
        exmeta["banned_until"] = self._now() + backoff
        until_dt = datetime.fromtimestamp(exmeta["banned_until"], tz=timezone.utc).isoformat()
        msg = f"⚠️ {exmeta['name']} rate-limited/banned ({reason}). Cooldown {int(backoff)}s until {until_dt}. Rotating."
        logger.warning(msg)
        notifier.send(f"ban_{exmeta['name']}", msg, cooldown=1200)

    def _err_is_ban(self, s: str) -> bool:
        s = (s or "").lower()
        return any(x in s for x in ["banned", "too many requests", "418", "-1003", "429", "403", "forbidden", "cloudflare"])

    def _get_tickers_cached(self, exchange):
        if not ENABLE_LIQ_FILTER or LIQ_MIN_USD <= 0:
            return {}
        cache = self.tickers_cache.get(exchange.id)
        now = self._now()
        if cache and (now - cache["ts"] < TICKERS_TTL):
            return cache["data"]
        try:
            data = exchange.fetch_tickers()
            self.tickers_cache[exchange.id] = {"ts": now, "data": data}
            return data
        except ccxt.BaseError as e:
            msg = str(e)
            logger.error(f"{exchange.id} fetch_tickers failed: {e}")
            if self._err_is_ban(msg):
                for exmeta in self.exchanges:
                    if exmeta["name"] == exchange.id:
                        self._ban_exchange(exmeta, "tickers")
                        break
            return {}
        except Exception as e:
            logger.error(f"{exchange.id} fetch_tickers unexpected error: {e}")
            return {}

    def _build_usdt_pairs_for_exchange(self, exchange):
        pairs = []
        for sym, m in exchange.markets.items():
            if sym.endswith("/USDT") and m.get("spot", True) and m.get("active", True):
                pairs.append(sym)
        pairs = sorted(set(pairs))
        if ENABLE_LIQ_FILTER:
            tickers = self._get_tickers_cached(exchange)
            if LIQ_MIN_USD > 0 and tickers:
                filtered = []
                for sym in pairs:
                    t = tickers.get(sym, {})
                    if est_quote_usd_volume(t) >= LIQ_MIN_USD:
                        filtered.append(sym)
                pairs = filtered or pairs
        if MAX_PAIRS_PER_EXCHANGE > 0:
            pairs = pairs[:MAX_PAIRS_PER_EXCHANGE]
        self.symbols_by_exchange[exchange.id] = pairs
        if exchange.id not in self.batch_idx_by_exchange:
            self.batch_idx_by_exchange[exchange.id] = 0
        logger.info(f"{exchange.id}: prepared {len(pairs)} USDT spot pairs (cap={MAX_PAIRS_PER_EXCHANGE}, liq_filter={'on' if ENABLE_LIQ_FILTER else 'off'})")

    def fetch_exchange_with_rotation(self):
        start_idx = self.exchange_index
        for i in range(len(self.exchanges)):
            idx = (start_idx + i) % len(self.exchanges)
            exmeta = self.exchanges[idx]
            if self._is_banned(exmeta):
                continue
            try:
                exchange = self.get_exchange(exmeta["name"], exmeta["params"])
                try: exchange.fetch_time()
                except Exception: pass
                self.exchange_index = idx
                if self.had_all_failed:
                    notifier.send("recovered", f"✅ Recovered: using {exmeta['name']} again.", cooldown=900)
                    self.had_all_failed = False
                if exchange.id not in self.symbols_by_exchange:
                    self._build_usdt_pairs_for_exchange(exchange)
                return exchange, exmeta
            except ccxt.BaseError as e:
                msg = str(e)
                logger.error(f"{exmeta['name']} init failed: {e}")
                if self._err_is_ban(msg):
                    self._ban_exchange(exmeta, "init")
                continue
            except Exception as e:
                logger.error(f"{exmeta['name']} unexpected init error: {e}")
                self._ban_exchange(exmeta, "unexpected-init")
                continue
        self.had_all_failed = True
        notifier.send("all_failed", "❌ All exchanges unavailable this cycle. Will retry.", cooldown=600)
        return None, None

    def fetch_candles(self, exchange, symbol, limit=OHLCV_LIMIT):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
            return [
                {
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                    "open": op, "high": hi, "low": lo, "close": cl, "volume": vol
                }
                for ts, op, hi, lo, cl, vol in ohlcv
            ]
        except ccxt.BaseError as e:
            msg = str(e)
            logger.error(f"OHLCV error {symbol} on {exchange.id}: {e}")
            if self._err_is_ban(msg):
                for exmeta in self.exchanges:
                    if exmeta["name"] == exchange.id:
                        self._ban_exchange(exmeta, "ohlcv")
                        break
            return []
        except Exception as e:
            logger.error(f"OHLCV unexpected error {symbol}: {e}")
            return []

    def _detect_green_streak(self, candles):
        streak = 0
        for c in reversed(candles):
            if c["close"] > c["open"]:
                streak += 1
            else:
                break
        return streak

    def probability_score(self, candles):
        streak = self._detect_green_streak(candles)
        closes = [c["close"] for c in candles]
        vol_surge = calculate_volume_surge(candles)
        atr_now  = calculate_atr(candles)
        atr_prev = calculate_atr(candles[:-1]) if len(candles) > 1 else 0.0
        rsi_now  = calculate_rsi(candles)
        res_break = near_resistance(candles)
        squeeze  = is_squeeze_on(candles)
        narrow7  = nr7(candles)
        donch    = donchian_breakout(candles)
        pos_last = candle_close_position(candles[-1])
        obv_up   = obv_slope(candles) > 0
        mfi_now  = mfi(candles)
        ema20 = ema(closes, 20) or 0
        ema50 = ema(closes, 50) or 0
        ema200 = ema(closes, 200) or 0
        vwap = anchored_vwap(candles, lookback=30) or 0

        # regime filters
        regime_ok = (ema50 and ema200 and ema50 > ema200)
        vwap_ok   = (vwap and closes[-1] >= vwap)

        score = 0.0; max_score = 14.0

        if MIN_GREEN_CANDLES <= streak <= MAX_GREEN_CANDLES:
            span = max(1, (MAX_GREEN_CANDLES - MIN_GREEN_CANDLES + 1))
            score += 2.0 * min(1.0, (streak - MIN_GREEN_CANDLES + 1) / span)
        if vol_surge: score += 1.6
        if atr_prev > 0 and atr_now > atr_prev:
            growth = (atr_now - atr_prev) / atr_prev
            score += min(1.6, max(0.0, growth * 3))
        if 55 < rsi_now < 75: score += 1.0
        if obv_up: score += 0.5
        if 50 < mfi_now < 80: score += 0.5
        if res_break: score += 1.0
        if donch: score += 1.4
        if squeeze and narrow7: score += 1.0
        score += 1.5 * max(0.0, min(1.0, pos_last))

        # regime bonuses
        if regime_ok: score += 1.0
        if vwap_ok:   score += 0.5

        if donch and pos_last > 0.7 and vol_surge:
            score += 1.0

        probability = round(min(1.0, score / max_score), 2)
        detail = {
            "streak": streak, "vol_surge": vol_surge, "atr_now": atr_now, "atr_prev": atr_prev,
            "rsi": rsi_now, "res_break": res_break, "squeeze": squeeze, "nr7": narrow7,
            "donchian": donch, "close_pos": round(pos_last, 2), "obv_up": obv_up, "mfi": round(mfi_now, 1),
            "ema20": round(ema20, 6), "ema50": round(ema50, 6), "ema200": round(ema200, 6),
            "vwap": round(vwap, 6) if vwap else 0.0, "regime_ok": regime_ok, "vwap_ok": vwap_ok,
            "prob": probability
        }
        return probability, detail

    def _cycle_exchange_index(self):
        if ROTATE_EXCHANGE_EACH_CYCLE:
            self.exchange_index = (self.exchange_index + 1) % len(self.exchanges)

    def start_monitoring(self):
        notifier.send("startup", f"🚀 Bot started. TF={TIMEFRAME}, BATCH={BATCH_SIZE}, INTERVAL={CHECK_INTERVAL}s", cooldown=60)
        logger.info("CryptoMonitor started")
        bot_status["is_running"] = True

        while True:
            try:
                exchange, exmeta = self.fetch_exchange_with_rotation()
                if exchange is None:
                    bot_status["errors"] = (bot_status["errors"][-9:] if len(bot_status["errors"]) > 9 else bot_status["errors"]) + ["All exchanges failed"]
                    bot_status["last_check"] = datetime.now(timezone.utc).isoformat()
                    self.status_callback(bot_status)
                    time.sleep(CHECK_INTERVAL)
                    continue

                # adapt bucket to budget
                budget: AdaptiveBudget = exmeta["budget"]
                per_min = budget.per_min()
                bucket = self.buckets.get(exchange.id)
                if bucket:
                    bucket.capacity = per_min
                    bucket.refill_per_sec = per_min / 60.0

                # ensure symbols
                symbols = self.symbols_by_exchange.get(exchange.id, [])
                if not symbols:
                    self._build_usdt_pairs_for_exchange(exchange)
                    symbols = self.symbols_by_exchange.get(exchange.id, [])

                if not symbols:
                    logger.warning(f"{exchange.id}: no USDT spot symbols found")
                    bot_status["last_check"] = datetime.now(timezone.utc).isoformat()
                    self.status_callback(bot_status)
                    time.sleep(CHECK_INTERVAL)
                    self._cycle_exchange_index()
                    continue

                # batch
                idx = self.batch_idx_by_exchange.get(exchange.id, 0)
                total = len(symbols)
                start = idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, total)
                batch = symbols[start:end]
                next_idx = (idx + 1) % ((total + BATCH_SIZE - 1) // BATCH_SIZE or 1)
                self.batch_idx_by_exchange[exchange.id] = next_idx

                allowed = bucket.available() if bucket else len(batch)
                if allowed <= 0:
                    logger.info(f"{exchange.id}: no tokens available this cycle; skipping")
                    budget.on_stable()
                    bot_status["last_check"] = datetime.now(timezone.utc).isoformat()
                    self.status_callback(bot_status)
                    self._cycle_exchange_index()
                    time.sleep(CHECK_INTERVAL)
                    continue

                to_process = batch[:allowed]
                logger.info(f"Checking batch {next_idx} on {exchange.id}: {len(to_process)}/{len(batch)} pairs (allowed={allowed}, rpm={per_min})")

                calls_made = 0
                any_error = False

                for symbol in to_process:
                    # de-dupe
                    last_t = self.last_alert_time.get(symbol, 0)
                    if time.time() - last_t < SYMBOL_ALERT_COOLDOWN:
                        continue

                    if bucket and not bucket.consume(1):
                        break

                    candles = self.fetch_candles(exchange, symbol, limit=OHLCV_LIMIT)
                    time.sleep(SLEEP_BETWEEN_CALLS + random.uniform(0, JITTER_MAX))
                    if not candles:
                        any_error = True
                        continue

                    probability, detail = self.probability_score(candles)
                    if probability >= PROB_THRESHOLD and detail["regime_ok"] and detail["vwap_ok"]:
                        msg = (
                            f"{symbol}: {detail['streak']} green candles ({TIMEFRAME}) on {exchange.id}\n"
                            f"Current: ${candles[-1]['close']:,.2f}\n"
                            f"ATR: {detail['atr_now']:.4f} (prev {detail['atr_prev']:.4f}), RSI: {detail['rsi']:.2f}\n"
                            f"VolSurge: {detail['vol_surge']} | ResBreak: {detail['res_break']} | Donch: {detail['donchian']}\n"
                            f"Squeeze: {detail['squeeze']} | NR7: {detail['nr7']} | OBV_up: {detail['obv_up']} | MFI: {detail['mfi']}\n"
                            f"EMA20/50/200: {detail['ema20']}/{detail['ema50']}/{detail['ema200']} | VWAP: {detail['vwap']}\n"
                            f"ClosePos: {detail['close_pos']} | Probability: {detail['prob']}"
                        )
                        try:
                            if self.bot:
                                self.bot.send_message(chat_id=self.chat_id, text=msg)
                        except Exception as e:
                            logger.error(f"Telegram send error: {e}")
                            notifier.send("tg_error", f"❗ Telegram send failed: {e}", cooldown=600)

                        save_alert(msg, candles[-1]['close'], detail['streak'], detail['prob'])
                        bot_status["alerts_sent"] += 1
                        bot_status["last_alert"] = msg
                        self.last_alert_time[symbol] = time.time()

                    self.last_candles = candles
                    calls_made += 1

                if any_error is False and calls_made > 0:
                    budget.on_stable()

                bot_status["last_check"] = datetime.now(timezone.utc).isoformat()
                self.status_callback(bot_status)
                self._cycle_exchange_index()
                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                bot_status["errors"] = (bot_status["errors"][-9:] if len(bot_status["errors"]) > 9 else bot_status["errors"]) + [str(e)]
                bot_status["last_check"] = datetime.now(timezone.utc).isoformat()
                self.status_callback(bot_status)
                notifier.send("loop_error", f"❗ Monitor loop error: {e}", cooldown=600)
                time.sleep(60)

# ========= Flask app & routes =========
app = Flask(__name__)
app.secret_key = "dev-secret"
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

@app.route("/")
def index():
    return jsonify(bot_status)

@app.route("/api/status")
def api_status():
    return jsonify(bot_status)

@app.route("/api/alerts")
def api_alerts():
    rows = get_recent_alerts()
    return jsonify([
        {
            "id": r[0],
            "message": r[1],
            "timestamp": r[2],
            "price": r[3],
            "green_candles": r[4],
            "breakout_probability": r[5]
        } for r in rows
    ])

@app.route("/api/price_data")
def api_price_data():
    monitor = globals().get("crypto_monitor")
    if monitor and monitor.last_candles:
        return jsonify([
            {
                "timestamp": row["timestamp"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"])
            } for row in monitor.last_candles
        ])
    return jsonify([])

@app.route("/ping")
def ping():
    return "pong"

@app.route("/settings")
def settings():
    cfg = {
        "TIMEFRAME": TIMEFRAME,
        "CHECK_INTERVAL": CHECK_INTERVAL,
        "BATCH_SIZE": BATCH_SIZE,
        "SLEEP_BETWEEN_CALLS": SLEEP_BETWEEN_CALLS,
        "JITTER_MAX": JITTER_MAX,
        "MAX_PAIRS_PER_EXCHANGE": MAX_PAIRS_PER_EXCHANGE,
        "OHLCV_LIMIT": OHLCV_LIMIT,
        "MIN_GREEN_CANDLES": MIN_GREEN_CANDLES,
        "MAX_GREEN_CANDLES": MAX_GREEN_CANDLES,
        "ENABLE_LIQ_FILTER": ENABLE_LIQ_FILTER,
        "LIQ_MIN_USD": LIQ_MIN_USD,
        "TICKERS_TTL": TICKERS_TTL,
        "PROB_THRESHOLD": PROB_THRESHOLD,
        "BAN_COOLDOWN_SECONDS": BAN_COOLDOWN_SECONDS,
        "ROTATE_EXCHANGE_EACH_CYCLE": ROTATE_EXCHANGE_EACH_CYCLE,
        "RPM_BINANCE": RPM_BINANCE,
        "RPM_BYBIT": RPM_BYBIT,
        "RPM_OKX": RPM_OKX,
        "SYMBOL_ALERT_COOLDOWN": SYMBOL_ALERT_COOLDOWN,
    }
    return jsonify(cfg)

@app.route("/favicon.ico")
def favicon():
    # avoid 404 spam
    return ("", 204)

# ========= Start + Watchdog =========
def update_status(status_update):
    bot_status.update(status_update)

crypto_monitor = CryptoMonitor(status_callback=update_status, notifier=notifier)
monitor_thread = threading.Thread(target=crypto_monitor.start_monitoring, daemon=True)
monitor_thread.start()

def watchdog():
    global monitor_thread, crypto_monitor
    while True:
        if not monitor_thread.is_alive():
            notifier.send("watchdog_restart", "⚠️ Monitor thread stopped. Restarting now.", cooldown=300)
            crypto_monitor = CryptoMonitor(status_callback=update_status, notifier=notifier)
            monitor_thread = threading.Thread(target=crypto_monitor.start_monitoring, daemon=True)
            monitor_thread.start()
        time.sleep(30)

threading.Thread(target=watchdog, daemon=True).start()

PORT = int(os.environ.get("PORT", "10000"))
logger.info("App starting...")
app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)