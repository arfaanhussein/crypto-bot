# ===== Patch for Python 3.13 imghdr removal (needed by python-telegram-bot==13.15) =====
import sys, types
if "imghdr" not in sys.modules:
    imghdr = types.ModuleType("imghdr")
    imghdr.what = lambda file, h=None: None
    sys.modules["imghdr"] = imghdr
# ========================================================================================

import os, time, random, logging, threading
from datetime import datetime, timezone
from flask import Flask, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import ccxt
from telegram import Bot

# ==================== SIMPLE SETTINGS (hardcoded once) ====================
TELEGRAM_TOKEN = "8384498061:AAElt7HeM88jfune948IcKkysHpw1tmXrlc"
TELEGRAM_CHAT_ID = "1040990874"

EXCHANGES = ["bybit", "binance", "okx"]  # scan all three, sequentially
TIMEFRAME = "1h"
CHECK_INTERVAL = 120         # seconds between cycles
BATCH_PER_EX = 25           # symbols per exchange per cycle (keep modest to avoid bans)
SLEEP_BETWEEN_CALLS = 0.7    # seconds between OHLCV calls
JITTER_MAX = 0.25            # add 0..JITTER_MAX random seconds to each sleep
OHLCV_LIMIT = 60             # candles per request
BAN_COOLDOWN = 45 * 60       # 45 minutes if exchange rate-limits (418/-1003/403 etc.)

# tolerant uptrend (allows one small red inside)
MIN_GREEN_CANDLES = 4
MAX_GREEN_CANDLES = 9
TOL_MAX_RED = 1
TOL_WINDOW = 8
TOL_MIN_GREEN_RATIO = 0.7
TOL_MIN_NET_GAIN = 0.008     # 0.8%
TOL_RED_MAX_ATR_FACTOR = 0.6 # red body <= 0.6 * ATR

PROB_THRESHOLD = 0.70        # tighten if too noisy
SYMBOL_ALERT_COOLDOWN = 3600 # 1 hour per-symbol cooldown

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("multi-simple")

# ==================== TELEGRAM (simple wrapper) ====================
class Notifier:
    def __init__(self, token, chat_id):
        self.enabled = bool(token and chat_id)
        self.chat_id = chat_id
        self.bot = Bot(token=token) if self.enabled else None
        self.last = {}
    def send(self, key, text, cooldown=600):
        if not self.enabled: return
        now = time.time()
        if now - self.last.get(key, 0) >= cooldown:
            try:
                self.bot.send_message(chat_id=self.chat_id, text=text)
                self.last[key] = now
            except Exception as e:
                logger.error(f"Telegram send error: {e}")

notifier = Notifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

# ==================== INDICATORS (pure Python) ====================
def calculate_atr(candles, period=14):
    if len(candles) < period + 1: return 0.0
    trs = []
    for i in range(1, period + 1):
        hi = candles[-i]["high"]; lo = candles[-i]["low"]; pc = candles[-i-1]["close"]
        trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
    return sum(trs) / period

def calculate_rsi(candles, period=14):
    if len(candles) < period + 1: return 50.0
    gains = losses = 0.0
    for i in range(-period, 0):
        ch = candles[i]["close"] - candles[i-1]["close"]
        gains += max(0.0, ch); losses += max(0.0, -ch)
    if losses == 0: return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def calculate_volume_surge(candles, lookback=20, surge_factor=1.6):
    if len(candles) < lookback + 1: return False
    avg_vol = sum(c["volume"] for c in candles[-lookback-1:-1]) / lookback
    return candles[-1]["volume"] > avg_vol * surge_factor

def near_resistance(candles, lookback=30, tolerance=0.004):
    if len(candles) < lookback + 1: return False
    resistance = max(c["close"] for c in candles[-lookback-1:-1])
    return candles[-1]["close"] >= resistance * (1 - tolerance)

def sma(vals, period):
    if len(vals) < period: return None
    return sum(vals[-period:]) / period

def ema(vals, period):
    if len(vals) < period: return None
    k = 2 / (period + 1); e = vals[-period]
    for v in vals[-period+1:]: e = v * k + e * (1 - k)
    return e

def stddev(vals, period):
    if len(vals) < period: return None
    subset = vals[-period:]; m = sum(subset) / period
    return (sum((x - m) ** 2 for x in subset) / period) ** 0.5

def bollinger_bands(closes, period=20, mult=2.0):
    m = sma(closes, period); s = stddev(closes, period)
    if m is None or s is None: return None
    upper, lower = m + mult * s, m - mult * s
    width = (upper - lower) / (m if m else 1)
    return {"mid": m, "upper": upper, "lower": lower, "width": width}

def keltner_channels(candles, period=20, mult=1.5):
    closes = [c["close"] for c in candles]
    mid = ema(closes, period); atr = calculate_atr(candles, period)
    if mid is None or atr == 0: return None
    upper, lower = mid + mult * atr, mid - mult * atr
    width = (upper - lower) / (mid if mid else 1)
    return {"mid": mid, "upper": upper, "lower": lower, "width": width}

def is_squeeze_on(candles, bb_period=20, kc_period=20, bb_mult=2.0, kc_mult=1.5):
    closes = [c["close"] for c in candles]
    bb = bollinger_bands(closes, bb_period, bb_mult); kc = keltner_channels(candles, kc_period, kc_mult)
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
    return 0.5 if rng <= 0 else (c["close"] - c["low"]) / rng

def obv_slope(candles, lookback=20):
    if len(candles) < lookback + 1: return 0.0
    obv = [0.0]
    for i in range(1, len(candles)):
        if candles[i]["close"] > candles[i-1]["close"]: obv.append(obv[-1] + candles[i]["volume"])
        elif candles[i]["close"] < candles[i-1]["close"]: obv.append(obv[-1] - candles[i]["volume"])
        else: obv.append(obv[-1])
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
    if len(candles) < lookback: return None
    num = den = 0.0
    for c in candles[-lookback:]:
        tp = (c["high"] + c["low"] + c["close"]) / 3; v = c["volume"]
        num += tp * v; den += v
    return None if den <= 0 else num / den

def body_size(c): return abs(c["close"] - c["open"])
def is_green(c): return c["close"] > c["open"]

def tolerant_uptrend_streak(candles, max_red=1, red_max_atr_factor=0.6, atr_period=14):
    if len(candles) < atr_period + 2: return 0
    atr = calculate_atr(candles, atr_period) or 0.0
    if atr <= 0: return 0
    streak = 0; red_used = 0
    for i in range(len(candles) - 1, -1, -1):
        c = candles[i]
        if is_green(c): streak += 1; continue
        if red_used < max_red:
            b = body_size(c)
            prev_close = candles[i-1]["close"] if i > 0 else c["close"]
            if b <= red_max_atr_factor * atr and c["close"] >= prev_close * 0.995:
                red_used += 1; streak += 1; continue
        break
    return streak

def uptrend_cluster_ok(candles, window=8, max_red=1, min_green_ratio=0.7, min_net_gain=0.008, red_max_atr_factor=0.6):
    n = len(candles); 
    if n < window + 2: return False
    w = candles[-window:]
    greens = sum(1 for c in w if is_green(c)); reds = window - greens
    if reds > max_red: return False
    start = w[0]["close"]; end = w[-1]["close"]; net = (end - start) / (start if start else 1)
    if net < min_net_gain: return False
    if reds > 0:
        atr = calculate_atr(candles, 14) or 0.0
        if atr <= 0: return False
        for c in w:
            if not is_green(c) and body_size(c) > red_max_atr_factor * atr:
                return False
    return (greens / window) >= min_green_ratio

def probability(candles):
    tol_streak = tolerant_uptrend_streak(candles, TOL_MAX_RED, TOL_RED_MAX_ATR_FACTOR, 14)
    cluster_ok = uptrend_cluster_ok(candles, TOL_WINDOW, TOL_MAX_RED, TOL_MIN_GREEN_RATIO, TOL_MIN_NET_GAIN, TOL_RED_MAX_ATR_FACTOR)

    closes = [c["close"] for c in candles]
    vol_surge = calculate_volume_surge(candles)
    atr_now  = calculate_atr(candles); atr_prev = calculate_atr(candles[:-1]) if len(candles) > 1 else 0.0
    rsi_now  = calculate_rsi(candles); res_break = near_resistance(candles)
    squeeze  = is_squeeze_on(candles); narrow7 = nr7(candles); donch = donchian_breakout(candles)
    pos_last = candle_close_position(candles[-1]); obv_up = obv_slope(candles) > 0; mfi_now = mfi(candles)
    ema20 = ema(closes, 20) or 0; ema50 = ema(closes, 50) or 0; ema200 = ema(closes, 200) or 0
    vwap = anchored_vwap(candles, 30) or 0
    regime_ok = (ema50 and ema200 and ema50 > ema200); vwap_ok = (vwap and closes[-1] >= vwap)

    score = 0.0; max_score = 14.0
    if tol_streak >= MIN_GREEN_CANDLES:
        span = max(1, (MAX_GREEN_CANDLES - MIN_GREEN_CANDLES + 1))
        clipped = min(MAX_GREEN_CANDLES, tol_streak)
        score += 2.0 * min(1.0, (clipped - MIN_GREEN_CANDLES + 1) / span)
    if cluster_ok: score += 1.0
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
    if regime_ok: score += 1.0
    if vwap_ok:   score += 0.5
    if donch and pos_last > 0.7 and vol_surge: score += 1.0

    prob = round(min(1.0, score / max_score), 2)
    return prob, {
        "tol_streak": tol_streak, "cluster_ok": cluster_ok, "vol_surge": vol_surge,
        "atr_now": atr_now, "atr_prev": atr_prev, "rsi": rsi_now, "res_break": res_break,
        "squeeze": squeeze, "nr7": narrow7, "donchian": donch, "close_pos": round(pos_last, 2),
        "obv_up": obv_up, "mfi": round(mfi_now, 1), "ema50_gt_200": regime_ok, "vwap_ok": vwap_ok,
        "prob": prob
    }

# ==================== MULTI-EX MONITOR (simple) ====================
class MultiExchangeMonitor:
    def __init__(self):
        self.notifier = notifier
        self.bot = notifier.bot if notifier.enabled else None
        self.chat_id = notifier.chat_id if notifier.enabled else None

        self.clients = {}
        self.symbols = {}
        self.batch_idx = {}
        self.banned_until = {}  # ex_id -> epoch
        self.last_alert_time = {}  # symbol -> epoch
        self.last_check = None

        for name in EXCHANGES:
            try:
                ex = getattr(ccxt, name)({'enableRateLimit': True, 'timeout': 15000})
                ex.load_markets()
                usdt = sorted([s for s, m in ex.markets.items() if s.endswith("/USDT") and m.get("spot", True) and m.get("active", True)])
                self.clients[name] = ex
                self.symbols[name] = usdt
                self.batch_idx[name] = 0
                self.banned_until[name] = 0
                logger.info(f"{name}: prepared {len(usdt)} USDT spot pairs")
            except Exception as e:
                logger.error(f"{name} init failed: {e}")
                self.banned_until[name] = time.time() + BAN_COOLDOWN  # cool off and retry later

    def banned(self, ex_id):
        return time.time() < self.banned_until.get(ex_id, 0)

    def mark_banned(self, ex_id, reason="rate-limit"):
        self.banned_until[ex_id] = time.time() + BAN_COOLDOWN
        until = datetime.fromtimestamp(self.banned_until[ex_id], tz=timezone.utc).isoformat()
        msg = f"⚠️ {ex_id} {reason}. Cooling down until {until}."
        logger.warning(msg)
        self.notifier.send(f"ban_{ex_id}", msg, cooldown=1200)

    def fetch_candles(self, ex, symbol):
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
            return [{"timestamp": datetime.fromtimestamp(ts/1000, tz=timezone.utc),
                     "open": op, "high": hi, "low": lo, "close": cl, "volume": vol}
                    for ts, op, hi, lo, cl, vol in ohlcv]
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
            self.mark_banned(ex.id, "rate-limit")
            return []
        except ccxt.ExchangeError as e:
            s = str(e).lower()
            if any(k in s for k in ["418", "-1003", "too many requests", "403", "cloudflare", "forbidden"]):
                self.mark_banned(ex.id, "rate-limit")
                return []
            logger.error(f"{ex.id} {symbol} OHLCV exchange error: {e}")
            return []
        except Exception as e:
            logger.error(f"{ex.id} {symbol} OHLCV error: {e}")
            return []

    def loop(self):
        self.notifier.send("startup", f"🚀 Bot started. TF={TIMEFRAME}, BATCH={BATCH_PER_EX}, INTERVAL={CHECK_INTERVAL}s", cooldown=60)
        while True:
            try:
                for ex_id in EXCHANGES:
                    if self.banned(ex_id):
                        continue
                    ex = self.clients.get(ex_id)
                    syms = self.symbols.get(ex_id, [])
                    if not ex or not syms:
                        continue

                    # batch slice
                    total = len(syms)
                    start = self.batch_idx[ex_id] * BATCH_PER_EX
                    end = min(start + BATCH_PER_EX, total)
                    batch = syms[start:end]
                    self.batch_idx[ex_id] = (self.batch_idx[ex_id] + 1) % ((total + BATCH_PER_EX - 1)//BATCH_PER_EX or 1)

                    logger.info(f"[{ex_id}] Checking {len(batch)} symbols")

                    for sym in batch:
                        # per-symbol alert cooldown
                        if time.time() - self.last_alert_time.get(sym, 0) < SYMBOL_ALERT_COOLDOWN:
                            continue

                        candles = self.fetch_candles(ex, sym)
                        time.sleep(SLEEP_BETWEEN_CALLS + random.uniform(0, JITTER_MAX))
                        if not candles:
                            continue

                        prob, detail = probability(candles)
                        if prob >= PROB_THRESHOLD and detail["ema50_gt_200"] and detail["vwap_ok"]:
                            msg = (
                                f"{sym} [{ex_id}]: tolStreak {detail['tol_streak']} ({TIMEFRAME})\n"
                                f"Prob: {prob} | RSI: {detail['rsi']:.2f} | ATR: {detail['atr_now']:.4f} (prev {detail['atr_prev']:.4f})\n"
                                f"VolSurge:{detail['vol_surge']} Donch:{detail['donchian']} Squeeze:{detail['squeeze']} NR7:{detail['nr7']}\n"
                                f"ClosePos:{detail['close_pos']} OBV_up:{detail['obv_up']} MFI:{detail['mfi']}"
                            )
                            try:
                                if notifier.enabled:
                                    notifier.bot.send_message(chat_id=notifier.chat_id, text=msg)
                            except Exception as e:
                                logger.error(f"Telegram send error: {e}")
                            self.last_alert_time[sym] = time.time()

                self.last_check = datetime.now(timezone.utc).isoformat()
                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(60)

monitor = MultiExchangeMonitor()

# ==================== FLASK (keep-alive + status) ====================
app = Flask(__name__)
app.secret_key = "dev-secret"
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

@app.route("/ping")
def ping(): return "pong"

@app.route("/status")
def status():
    return jsonify({
        "exchanges": EXCHANGES,
        "timeframe": TIMEFRAME,
        "is_running": True,
        "last_check": monitor.last_check
    })

# start monitor thread and web app
threading.Thread(target=monitor.loop, daemon=True).start()
PORT = int(os.environ.get("PORT", "10000"))
logger.info("App starting...")
app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
