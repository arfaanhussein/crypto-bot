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
CHECK_INTERVAL = 300
MIN_GREEN_CANDLES = 4
MAX_GREEN_CANDLES = 9
BREAKOUT_THRESHOLD = 0.01

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- DB Setup ---
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

# --- Bot ---
class CryptoMonitor:
    def __init__(self, telegram_token, telegram_chat_id, status_callback):
        self.bot = Bot(token=telegram_token)
        self.chat_id = telegram_chat_id
        self.status_callback = status_callback
        self.exchanges = [("binance", {}), ("bybit", {}), ("okx", {})]
        self.last_candles = None

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
            import pandas as pd  # lazy import
            candles = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=5)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"OHLCV fetch failed for {symbol}: {e}")
            return None

    def detect_green_candles(self, df):
        green_streak = 0
        for i in range(len(df) - 1, -1, -1):
            if df['close'][i] > df['open'][i]:
                green_streak += 1
            else:
                break
        return green_streak if MIN_GREEN_CANDLES <= green_streak <= MAX_GREEN_CANDLES else 0

    def light_breakout_check(self, df):
        current_price = df['close'].iloc[-1]
        avg_future = df['close'].tail(3).mean()
        increase_pct = (avg_future - current_price) / current_price
        return increase_pct > BREAKOUT_THRESHOLD, avg_future

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
                hot_pairs = [sym for sym, data in tickers.items()
                             if sym.endswith('/USDT') and data.get('percentage') is not None and abs(data['percentage']) > 1]

                logger.info(f"Hot pairs found: {len(hot_pairs)}")
                for symbol in hot_pairs:
                    df = self.fetch_candles(exchange, symbol)
                    if df is None:
                        continue

                    green_count = self.detect_green_candles(df)
                    if green_count > 0:
                        is_breakout, predicted = self.light_breakout_check(df)
                        prob = "High" if is_breakout else "Low"
                        msg = f"{symbol}: {green_count} green candles\n" \
                              f"Current: ${df['close'].iloc[-1]:,.2f} | Predicted: ${predicted:,.2f}\n" \
                              f"Breakout Probability: {prob}"
                        self.send_alert(msg)
                        with app.app_context():
                            alert = Alert(message=msg, price=df['close'].iloc[-1], green_candles=green_count)
                            db.session.add(alert)
                            db.session.commit()
                        bot_status['alerts_sent'] += 1
                        bot_status['last_alert'] = msg
                    self.last_candles = df

                bot_status['last_check'] = datetime.utcnow()
                self.status_callback(bot_status)
                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                bot_status['errors'].append(str(e))
                time.sleep(CHECK_INTERVAL)

# --- Status callback ---
def update_status(status_update):
    bot_status.update(status_update)

# --- Flask routes ---
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
    if crypto_monitor and crypto_monitor.last_candles is not None:
        df = crypto_monitor.last_candles
        return jsonify([{
            'timestamp': row['timestamp'].isoformat(),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        } for idx, row in df.iterrows()])
    return jsonify([])

@app.route('/ping')
def ping():
    return "pong"

# --- Start bot ---
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