# config.py
import os

# Struktur for lagring
DATA_DIR = "data_storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DATA_FILE = os.path.join(DATA_DIR, "oslo_market_10y.parquet")
MARKET_DATA_FILE = os.path.join(DATA_DIR, "oslo_benchmark.parquet")
TICKER_LIST_FILE = os.path.join(DATA_DIR, "tickers.txt")

# Dataparametere
HISTORICAL_YEARS = 10

# Risiko og kapital
DEFAULT_KAPITAL = 100000
DEFAULT_RISIKO_PROSENT = 1.0

# Visuelle temaer
CHART_THEME = "plotly_white"