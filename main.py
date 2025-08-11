# ===== Patch for Python 3.13 imghdr removal =====
import sys, types
if "imghdr" not in sys.modules:
    imghdr = types.ModuleType("imghdr")
    imghdr.what = lambda file, h=None: None
    sys.modules["imghdr"] = imghdr
# ================================================

import os
import logging
import threading
import time
from datetime import datetime
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
import ccxt
from telegram import Bot

# Config
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8384498061:AAElt7HeM88jfune948IcKkysHpw1tmXrlc")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "1040990874")

TIMEFRAME = "1h"
CHECK_INTERVAL = 300  # seconds
MIN_GREEN_CANDLES = 4
MAX_GREEN_CANDLES = 9
BREAKOUT_THRESHOLD = 0.01
TOP_N = 15  # limit pairs checked in-depth to avoid API bans

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Database ---
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = "dev-secret"
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///crypto_bot.db"
db.init_app(app)

from sqlalchemy import Column, Integer, Float, String, DateTime
class Alert(db.Model):
    id = Column(Integer, primary_key=True)
    message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    price = Column(Float)
    green_candles = Column(Integer)

with app.app_context():
    db.create_all()

bot_status = {
    'is_running': False,
    'last_check': None,
    'alerts_sent': 0,
    'last_alert': None,
    'errors': []
}

# === Indicator Functions (No pandas) ===
def calculate_atr(candles, period=5):
    if len(candles) < period + 1:
        return 0
    trs = []
    for i in range(1, period + 1):
        high = candles[-i]["high"]
        low = candles[-i]["low"]
        prev_close = candles[-i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / period

def calculate_rsi(candles, period=14):
    if len(candles) < period + 1:
        return 50
    gains, losses = 0, 0
    for i in range(-period, 0):
        change = candles[i]["close"] - candles[i-1]["close"]
        if change > 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def calculate_volume_surge(candles, lookback=5, surge_factor=1.5):
    if len(candles) < lookback:
        return False
    avg_vol = sum(c["volume"] for c in candles[-lookback-1:-1]) / lookback
    return candles[-1]["volume"] > avg_vol * surge_factor

def near_resistance(candles, lookback=20, tolerance=0.003):
    if len(candles) < lookback:
        return False
    resistance = max(c["close"] for c in candles[-lookback-1:-1])
    return candles[-1]["close"] >= resistance * (1 - tolerance)

# === Bot Class ===
class CryptoMonitor:
    def __init__(self, telegram_token, telegram_chat_id, status_callback):
        self.bot = Bot(token=telegram_token)
        self.chat_id = telegram_chat_id
        self.status_callback = status_callback
        self.exchanges = [("binance", {}), ("bybit", {}), ("okx", {})]
        self.last_candles = []

    def get_exchange(self, name, params):
        ex = getattr(ccxt, name)(params)
        ex.load_markets()
        return ex

    def fetch_tickers_with_fallback(self):
        for name, params in self.exchanges:
            try:
                exchange = self.get_exchange(name, params)
                tickers = exchange.fetch_tickers()
                return exchange, tickers
            except Exception as e:
                logger.error(f"{name} tickers failed: {e}")
        raise RuntimeError("All exchanges failed")

    def fetch_candles(self, exchange, symbol):
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=30)
            return [
                {
                    "timestamp": datetime.fromtimestamp(ts / 1000),
                    "open": op,
                    "high": hi,
                    "low": lo,
                    "close": cl,
                    "volume": vol
                }
                for ts, op, hi, lo, cl, vol in candles
            ]
        except Exception as e:
            logger.error(f"OHLCV fetch failed for {symbol}: {e}")
            return []

    def detect_green_streak(self, candles):
        green_streak = 0
        for candle in reversed(candles):
            if candle["close"] > candle["open"]:
                green_streak += 1
            else:
                break
        return green_streak

    def send_alert(self, message):
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def start_monitoring(self):
        logger.info("CryptoMonitor started")
        bot_status['is_running'] = True

        while True:
            try:
                exchange, tickers = self.fetch_tickers_with_fallback()
                # Top-N movers filter
                candidates = [
                    (sym, data.get('percentage', 0))
                    for sym, data in tickers.items()
                    if sym.endswith('/USDT')
                ]
                candidates.sort(key=lambda x: abs(x[1] or 0), reverse=True)
                hot_pairs = [sym for sym, _ in candidates[:TOP_N]]

                logger.info(f"Checking {len(hot_pairs)} top USDT pairs")

                for symbol in hot_pairs:
                    candles = self.fetch_candles(exchange, symbol)
                    if not candles:
                        continue
                    green_streak = self.detect_green_streak(candles)
                    if MIN_GREEN_CANDLES <= green_streak <= MAX_GREEN_CANDLES:
                        vol_surge = calculate_volume_surge(candles)
                        atr_now = calculate_atr(candles)
                        atr_prev = calculate_atr(candles[:-1])
                        rsi_now = calculate_rsi(candles)
                        res_break = near_resistance(candles)

                        if vol_surge and atr_now > atr_prev and 55 < rsi_now < 75 and res_break:
                            prob = "HIGH"
                        else:
                            prob = "LOW"

                        msg = (f"{symbol}: {green_streak} green candles ({TIMEFRAME})\n"
                               f"Current: ${candles[-1]['close']:,.2f}\n"
                               f"ATR: {atr_now:.4f}, RSI: {rsi_now:.2f}\n"
                               f"Volume Surge: {vol_surge}, Near Resistance: {res_break}\n"
                               f"Breakout Probability: {prob}")

                        self.send_alert(msg)
                        with app.app_context():
                            alert = Alert(
                                message=msg,
                                price=candles[-1]['close'],
                                green_candles=green_streak
                            )
                            db.session.add(alert)
                            db.session.commit()
                        bot_status['alerts_sent'] += 1
                        bot_status['last_alert'] = msg

                    self.last_candles = candles

                bot_status['last_check'] = datetime.utcnow()
                self.status_callback(bot_status)
                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                bot_status['errors'].append(str(e))
                time.sleep(CHECK_INTERVAL)

# === Status Callback ===
def update_status(status_update):
    bot_status.update(status_update)

# === Flask Routes ===
@app.route('/')
def index():
    return jsonify(bot_status)

@app.route('/api/status')
def api_status():
    return jsonify(bot_status)

@app.route('/api/alerts')
def api_alerts():
    alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(10).all()
    return jsonify([{
        'id': a.id,
        'message': a.message,
        'timestamp': a.timestamp.isoformat(),
        'price': a.price,
        'green_candles': a.green_candles
    } for a in alerts])

@app.route('/api/price_data')
def api_price_data():
    return jsonify([
        {
            'timestamp': row["timestamp"].isoformat(),
            'open': float(row["open"]),
            'high': float(row["high"]),
            'low': float(row["low"]),
            'close': float(row["close"]),
            'volume': float(row["volume"])
        } for row in crypto_monitor.last_candles
    ]) if crypto_monitor.last_candles else jsonify([])

@app.route('/ping')
def ping():
    return "pong"

# === Start Bot ===
crypto_monitor = CryptoMonitor(
    telegram_token=TELEGRAM_TOKEN,
    telegram_chat_id=TELEGRAM_CHAT_ID,
    status_callback=update_status
)
monitor_thread = threading.Thread(target=crypto_monitor.start_monitoring, daemon=True)
monitor_thread.start()

PORT = int(os.environ.get("PORT", 8080))
logger.info("App starting...")
app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)