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
from datetime import datetime, timezone
from flask import Flask, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import ccxt
from telegram import Bot

# ========= Config (env overrides) =========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8384498061:AAElt7HeM88jfune948IcKkysHpw1tmXrlc")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "1040990874")

TIMEFRAME = os.environ.get("TIMEFRAME", "1h")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "120"))  # seconds between cycles
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))          # pairs per cycle
SLEEP_BETWEEN_CALLS = float(os.environ.get("SLEEP_BETWEEN_CALLS", "0.5"))  # sec between OHLCV calls

MIN_GREEN_CANDLES = int(os.environ.get("MIN_GREEN_CANDLES", "4"))
MAX_GREEN_CANDLES = int(os.environ.get("MAX_GREEN_CANDLES", "9"))
BREAKOUT_THRESHOLD = float(os.environ.get("BREAKOUT_THRESHOLD", "0.01"))

BAN_COOLDOWN_SECONDS = int(os.environ.get("BAN_COOLDOWN_SECONDS", "2700"))  # 45 min ban cooldown
ROTATE_EXCHANGE_EACH_CYCLE = os.environ.get("ROTATE_EXCHANGE_EACH_CYCLE", "1") == "1"

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
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self.last = {}  # key -> last epoch sent

    def send(self, key, text, cooldown=900):
        now = time.time()
        last = self.last.get(key, 0)
        if now - last >= cooldown:
            try:
                self.bot.send_message(chat_id=self.chat_id, text=text)
                self.last[key] = now
            except Exception as e:
                logger.error(f"Notifier send error: {e}")

notifier = Notifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

# ========= Indicators (pure Python) =========
def calculate_atr(candles, period=5):
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

def calculate_volume_surge(candles, lookback=5, surge_factor=1.5):
    if len(candles) < lookback + 1:
        return False
    avg_vol = sum(c["volume"] for c in candles[-lookback-1:-1]) / lookback
    return candles[-1]["volume"] > avg_vol * surge_factor

def near_resistance(candles, lookback=20, tolerance=0.003):
    if len(candles) < lookback + 1:
        return False
    resistance = max(c["close"] for c in candles[-lookback-1:-1])
    return candles[-1]["close"] >= resistance * (1 - tolerance)

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
        self.bot = notifier.bot
        self.chat_id = notifier.chat_id
        self.status_callback = status_callback

        self.exchanges = [
            {"name": "binance", "params": {}, "banned_until": 0},
            {"name": "bybit",   "params": {}, "banned_until": 0},
            {"name": "okx",     "params": {}, "banned_until": 0},
        ]
        self.exchange_index = 0
        self.usdt_pairs = []
        self.batch_index = 0
        self.last_candles = []
        self.had_all_failed = False  # for recovery notification

    def _now(self):
        return time.time()

    def get_exchange(self, name, params):
        ex = getattr(ccxt, name)(params)
        ex.enableRateLimit = True
        ex.load_markets()
        return ex

    def _is_banned(self, ex):
        return self._now() < ex["banned_until"]

    def _ban_exchange(self, ex, reason):
        ex["banned_until"] = self._now() + BAN_COOLDOWN_SECONDS
        until_dt = datetime.fromtimestamp(ex["banned_until"], tz=timezone.utc).isoformat()
        msg = f"⚠️ {ex['name']} rate-limited/banned ({reason}). Cooling down until {until_dt}. Rotating exchange."
        logger.warning(msg)
        self.notifier.send(f"ban_{ex['name']}", msg, cooldown=1200)

    def fetch_tickers_with_rotation(self):
        start_idx = self.exchange_index
        for i in range(len(self.exchanges)):
            idx = (start_idx + i) % len(self.exchanges)
            exmeta = self.exchanges[idx]
            if self._is_banned(exmeta):
                continue
            try:
                exchange = self.get_exchange(exmeta["name"], exmeta["params"])
                tickers = exchange.fetch_tickers()
                self.exchange_index = idx
                if self.had_all_failed:
                    self.notifier.send("recovered", f"✅ Recovered: using {exmeta['name']} tickers again.", cooldown=900)
                    self.had_all_failed = False
                return exchange, tickers, exmeta
            except ccxt.BaseError as e:
                msg = str(e).lower()
                logger.error(f"{exmeta['name']} tickers failed: {e}")
                if "banned" in msg or "too many requests" in msg or "418" in msg or "-1003" in msg:
                    self._ban_exchange(exmeta, "tickers")
                continue
            except Exception as e:
                logger.error(f"{exmeta['name']} unexpected error: {e}")
                self._ban_exchange(exmeta, "unexpected")
                continue
        self.had_all_failed = True
        self.notifier.send("all_failed", "❌ All exchanges unavailable this cycle. Will retry.", cooldown=600)
        return None, {}, None

    def _refresh_usdt_pairs_if_needed(self, tickers):
        if not self.usdt_pairs and tickers:
            self.usdt_pairs = [sym for sym in tickers.keys() if sym.endswith("/USDT")]
            if self.usdt_pairs:
                logger.info(f"Total USDT pairs discovered: {len(self.usdt_pairs)}")
                self.batch_index = 0
            else:
                logger.warning("No USDT pairs found in tickers")

    def fetch_candles(self, exchange, symbol, limit=30):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
            candles = [
                {
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                    "open": op, "high": hi, "low": lo, "close": cl, "volume": vol
                }
                for ts, op, hi, lo, cl, vol in ohlcv
            ]
            return candles
        except ccxt.BaseError as e:
            msg = str(e).lower()
            logger.error(f"OHLCV error {symbol} on {exchange.id}: {e}")
            if "banned" in msg or "too many requests" in msg or "418" in msg or "-1003" in msg:
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

    def _probability_score(self, streak, vol_surge, atr_now, atr_prev, rsi_now, res_break):
        score = 0
        if MIN_GREEN_CANDLES <= streak <= MAX_GREEN_CANDLES: score += 2
        if vol_surge: score += 2
        if atr_now > atr_prev: score += 1
        if 55 < rsi_now < 75: score += 1
        if res_break: score += 2
        return round(score / 8, 2)

    def _cycle_exchange_index(self):
        if ROTATE_EXCHANGE_EACH_CYCLE:
            self.exchange_index = (self.exchange_index + 1) % len(self.exchanges)

    def start_monitoring(self):
        # Startup notification
        self.notifier.send("startup", f"🚀 Bot started. TF={TIMEFRAME}, BATCH={BATCH_SIZE}, INTERVAL={CHECK_INTERVAL}s", cooldown=60)
        logger.info("CryptoMonitor started")
        bot_status["is_running"] = True

        while True:
            try:
                exchange, tickers, exmeta = self.fetch_tickers_with_rotation()
                if exchange is None:
                    logger.error("All exchanges unavailable; sleeping...")
                    bot_status["errors"].append("All exchanges failed this cycle")
                    bot_status["last_check"] = datetime.now(timezone.utc).isoformat()
                    self.status_callback(bot_status)
                    time.sleep(CHECK_INTERVAL)
                    continue

                self._refresh_usdt_pairs_if_needed(tickers)
                if not self.usdt_pairs:
                    bot_status["last_check"] = datetime.now(timezone.utc).isoformat()
                    self.status_callback(bot_status)
                    time.sleep(CHECK_INTERVAL)
                    self._cycle_exchange_index()
                    continue

                total = len(self.usdt_pairs)
                start = self.batch_index * BATCH_SIZE
                end = min(start + BATCH_SIZE, total)
                batch_pairs = self.usdt_pairs[start:end]
                self.batch_index = (self.batch_index + 1) % ((total + BATCH_SIZE - 1) // BATCH_SIZE or 1)

                logger.info(f"Checking batch {self.batch_index}: {len(batch_pairs)} USDT pairs on {exchange.id}")

                for symbol in batch_pairs:
                    candles = self.fetch_candles(exchange, symbol, limit=30)
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    if not candles:
                        continue

                    streak = self._detect_green_streak(candles)
                    if MIN_GREEN_CANDLES <= streak <= MAX_GREEN_CANDLES:
                        vol_surge = calculate_volume_surge(candles)
                        atr_now = calculate_atr(candles)
                        atr_prev = calculate_atr(candles[:-1]) if len(candles) > 1 else 0.0
                        rsi_now = calculate_rsi(candles)
                        res_break = near_resistance(candles)

                        probability = self._probability_score(streak, vol_surge, atr_now, atr_prev, rsi_now, res_break)
                        if probability >= 0.5:
                            msg = (
                                f"{symbol}: {streak} green candles ({TIMEFRAME}) on {exchange.id}\n"
                                f"Current: ${candles[-1]['close']:,.2f}\n"
                                f"ATR: {atr_now:.4f}, RSI: {rsi_now:.2f}\n"
                                f"VolSurge: {vol_surge}, NearRes: {res_break}\n"
                                f"Breakout Probability: {probability}"
                            )
                            try:
                                self.bot.send_message(chat_id=self.chat_id, text=msg)
                            except Exception as e:
                                logger.error(f"Telegram send error: {e}")
                                self.notifier.send("tg_error", f"❗ Telegram send failed: {e}", cooldown=600)
                            save_alert(msg, candles[-1]['close'], streak, probability)
                            bot_status["alerts_sent"] += 1
                            bot_status["last_alert"] = msg

                    self.last_candles = candles

                bot_status["last_check"] = datetime.now(timezone.utc).isoformat()
                self.status_callback(bot_status)
                self._cycle_exchange_index()
                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                # Never exit; notify with cooldown to avoid spam
                logger.error(f"Monitor loop error: {e}")
                bot_status["errors"].append(str(e))
                bot_status["last_check"] = datetime.now(timezone.utc).isoformat()
                self.status_callback(bot_status)
                self.notifier.send("loop_error", f"❗ Monitor loop error: {e}", cooldown=600)
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
        }
        for r in rows
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