# utils.py
"""
Felles hjelpefunksjoner for InveStock Pro.
Inneholder delt funksjonalitet brukt av flere moduler.
"""

import os
import json
from datetime import datetime

# =============================================================================
# WATCHLIST SYSTEM - Delt mellom Scanner og AI Scanner
# Lagrer metadata: {ticker: {added_date, reason, price_at_add, target_price, notes}}
# =============================================================================

WATCHLIST_FILE = "data_storage/ai_watchlist.json"


def _migrate_watchlist(data) -> dict:
    """
    Migrerer fra gammel liste-format til nytt dict-format.
    
    Gammel: ["EQNR.OL", "DNB.OL"]
    Ny:     {"EQNR.OL": {"added_date": "...", ...}, "DNB.OL": {...}}
    """
    if isinstance(data, list):
        migrated = {}
        for ticker in data:
            if isinstance(ticker, str):
                migrated[ticker] = {
                    "added_date": datetime.now().strftime("%Y-%m-%d"),
                    "reason": "",
                    "price_at_add": None,
                    "target_price": None,
                    "notes": "Migrert fra gammel liste"
                }
        return migrated
    elif isinstance(data, dict):
        # Sjekk om det er gammel {tickers: [], added_dates: {}} format
        if "tickers" in data and isinstance(data.get("tickers"), list):
            migrated = {}
            for ticker in data["tickers"]:
                migrated[ticker] = {
                    "added_date": data.get("added_dates", {}).get(ticker, datetime.now().strftime("%Y-%m-%d")),
                    "reason": "",
                    "price_at_add": None,
                    "target_price": None,
                    "notes": "Migrert fra gammel liste"
                }
            return migrated
        return data
    return {}


def _load_watchlist_raw() -> dict:
    """
    Laster watchlist fra fil som dict med metadata.
    Migrerer automatisk fra gammel liste-format.
    
    Returns:
        Dict: {ticker: {added_date, reason, price_at_add, target_price, notes}}
    """
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Migrer om nødvendig
            if isinstance(data, list) or (isinstance(data, dict) and "tickers" in data):
                migrated = _migrate_watchlist(data)
                _save_watchlist_raw(migrated)
                return migrated
            
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_watchlist_raw(watchlist_data: dict) -> bool:
    """
    Lagrer watchlist (dict-format) til fil.
    
    Args:
        watchlist_data: Dict med ticker → metadata
    """
    try:
        os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)
        with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(watchlist_data, f, indent=2, ensure_ascii=False)
        return True
    except IOError:
        return False


def load_watchlist() -> list:
    """
    Laster watchlist fra fil.
    Returnerer LISTE med ticker-symboler for bakoverkompatibilitet.
    
    Returns:
        Liste med ticker-symboler i watchlisten
    """
    return list(_load_watchlist_raw().keys())


def load_watchlist_metadata() -> dict:
    """
    Laster watchlist med full metadata.
    
    Returns:
        Dict: {ticker: {added_date, reason, price_at_add, target_price, notes}}
    """
    return _load_watchlist_raw()


def save_watchlist(watchlist: list) -> bool:
    """
    Lagrer watchlist til fil (bakoverkompatibelt — konverterer liste til dict).
    
    Args:
        watchlist: Liste med ticker-symboler
        
    Returns:
        True hvis lagring var vellykket
    """
    existing = _load_watchlist_raw()
    
    # Behold metadata for tickers som fortsatt er i listen
    new_data = {}
    for ticker in watchlist:
        if ticker in existing:
            new_data[ticker] = existing[ticker]
        else:
            new_data[ticker] = {
                "added_date": datetime.now().strftime("%Y-%m-%d"),
                "reason": "",
                "price_at_add": None,
                "target_price": None,
                "notes": ""
            }
    
    return _save_watchlist_raw(new_data)


def add_to_watchlist(ticker: str, reason: str = "", price_at_add: float = None,
                     target_price: float = None, notes: str = "") -> bool:
    """
    Legger til ticker i watchlist med metadata.
    
    Args:
        ticker: Ticker-symbol å legge til
        reason: Grunn for tillegging (f.eks. 'VCP-signal', 'AI Scanner')
        price_at_add: Kurs ved tillegging
        target_price: Kursmål
        notes: Egne notater
        
    Returns:
        True hvis ticker ble lagt til (ikke allerede i listen)
    """
    data = _load_watchlist_raw()
    if ticker not in data:
        data[ticker] = {
            "added_date": datetime.now().strftime("%Y-%m-%d"),
            "reason": reason,
            "price_at_add": price_at_add,
            "target_price": target_price,
            "notes": notes
        }
        _save_watchlist_raw(data)
        return True
    return False


def remove_from_watchlist(ticker: str) -> bool:
    """
    Fjerner ticker fra watchlist.
    
    Args:
        ticker: Ticker-symbol å fjerne
        
    Returns:
        True hvis ticker ble fjernet (var i listen)
    """
    data = _load_watchlist_raw()
    if ticker in data:
        del data[ticker]
        _save_watchlist_raw(data)
        return True
    return False


def is_in_watchlist(ticker: str) -> bool:
    """
    Sjekker om ticker er i watchlist.
    
    Args:
        ticker: Ticker-symbol å sjekke
        
    Returns:
        True hvis ticker er i watchlisten
    """
    return ticker in _load_watchlist_raw()


def update_watchlist_metadata(ticker: str, **kwargs) -> bool:
    """
    Oppdaterer metadata for en ticker i watchlisten.
    
    Args:
        ticker: Ticker-symbol
        **kwargs: Felter å oppdatere (reason, price_at_add, target_price, notes)
        
    Returns:
        True hvis oppdatering var vellykket
    """
    data = _load_watchlist_raw()
    if ticker not in data:
        return False
    
    for key, value in kwargs.items():
        if key in ('reason', 'price_at_add', 'target_price', 'notes', 'added_date'):
            data[ticker][key] = value
    
    return _save_watchlist_raw(data)
