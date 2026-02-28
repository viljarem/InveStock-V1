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

# Transaksjonskostnader (brukes i backtest)
KURTASJE_PCT = 0.05      # Kurtasje per vei i prosent (0.05% = typisk Nordnet/DNB)
SPREAD_SLIPPAGE_PCT = 0.10  # Spread + slippage i prosent (0.10% = realistisk for likvide OBX-aksjer)

# Ny modul: Signalmodul-side
ENABLE_SIGNALMODUL_PAGE = True
SIGNALMODUL_DEFAULT_MIN_KVALITET = "C"
SIGNALMODUL_DEFAULT_MAX_DAGER = 30
SIGNALMODUL_MIN_VOL_RATIO = 0.8

# Exit-signal innstillinger
EXIT_TRAILING_STOP_PCT = 8.0      # Trailing stop på 8%
EXIT_MAX_DRAWDOWN_PCT = 15.0      # Maks drawdown før tvungen salg
EXIT_TIME_STOP_DAYS = 60          # Tidsstopp etter X dager uten gevinst
EXIT_PROFIT_TARGET_PCT = 20.0     # Automatisk profittmål

# E-post varsling (for fremtidig bruk)
EMAIL_ENABLED = False
EMAIL_RECIPIENT = ""
EMAIL_MIN_SIGNAL_QUALITY = "A"
EMAIL_MIN_VOLUME_NOK = 3000000    # Minimum 3M NOK dagsomsetning

# Visuelle temaer
CHART_THEME = "plotly_white"

# Makro-data (oljepris, valuta)
BRENT_DATA_FILE = os.path.join(DATA_DIR, "brent_crude.parquet")
USDNOK_DATA_FILE = os.path.join(DATA_DIR, "usdnok.parquet")

# Sektor-mapping cache
SEKTOR_MAPPING_FILE = os.path.join(DATA_DIR, "sektor_mapping.json")