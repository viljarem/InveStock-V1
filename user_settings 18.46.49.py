"""
Brukerinnstillinger — Lagrer og laster brukerpreferanser.

Innstillinger lagres i data_storage/user_settings.json.
Brukes av scanner, teknisk analyse, portefølje, og chart for å
huske brukerens valg mellom sesjoner.
"""
import os
import json
from log_config import get_logger

logger = get_logger(__name__)

SETTINGS_FILE = os.path.join("data_storage", "user_settings.json")

# Standard innstillinger
DEFAULT_SETTINGS = {
    # Scanner-filtre
    "scanner": {
        "min_kvalitet": "C",
        "max_dager": 30,
        "min_volum_ratio": 0.8,
        "default_strategi": "Alle strategier",
        "vis_exit": True,
        "vis_sektor": True,
        "vis_stotte": False,
        "vis_motstand": False,
        "vis_konvergens": False,
        "sorter_etter": "Signal (dato)",
        "sorter_retning": "Nyeste først",
        "filtrer_false_breakout": False,
        "min_rr": 0.0,
    },
    # Chart-innstillinger
    "chart": {
        "chart_type": "candlestick",         # candlestick | linje
        "tidsperiode": "1Y",                  # 3M, 6M, 1Y, 2Y, 5Y, Max
        "indikatorer": {
            "sma_50": True,
            "sma_200": True,
            "sma_20": False,
            "bollinger": False,
            "rsi": False,
            "macd": False,
            "volum": True,
            "ichimoku": False,
            "atr": False,
        },
        "vis_signaler": True,
        "vis_insider": True,
        "tema": "dark",
    },
    # Teknisk analyse
    "teknisk_analyse": {
        "default_ticker": None,
    },
    # Portefølje
    "portefolje": {
        "valuta": "NOK",
        "kurtasje_pct": 0.05,
        "spread_pct": 0.10,
    },
    # Generelt
    "generelt": {
        "min_dagsomsetning": 500000,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merger override inn i base, rekursivt for nestede dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings() -> dict:
    """Laster brukerinnstillinger fra fil. Returnerer defaults ved feil."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            # Merge med defaults (nye innstillinger får default-verdier)
            return _deep_merge(DEFAULT_SETTINGS, saved)
    except Exception as e:
        logger.warning(f"Kunne ikke laste innstillinger: {e}")
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> bool:
    """Lagrer brukerinnstillinger til fil."""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Kunne ikke lagre innstillinger: {e}")
        return False


def get_setting(path: str, default=None):
    """
    Henter én innstilling via punktnotasjon.
    Eksempel: get_setting('scanner.min_kvalitet') → 'C'
    """
    settings = load_settings()
    keys = path.split('.')
    current = settings
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def set_setting(path: str, value) -> bool:
    """
    Setter én innstilling via punktnotasjon.
    Eksempel: set_setting('scanner.min_kvalitet', 'B')
    """
    settings = load_settings()
    keys = path.split('.')
    current = settings
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return save_settings(settings)


def reset_settings() -> bool:
    """Tilbakestiller alle innstillinger til default."""
    return save_settings(DEFAULT_SETTINGS)
