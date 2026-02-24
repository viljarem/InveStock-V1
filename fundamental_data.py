# fundamental_data.py
"""
Fundamental Data Modul for InveStock.

Henter og analyserer fundamental data fra yfinance:
- P/E ratio
- P/B ratio  
- Dividend Yield
- Debt/Equity
- Profit Margin
- Revenue Growth
- ROE (Return on Equity)

Beregner en "Fundamental Score" (0-100) som kombinerer disse.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
import os

# Cache-konfigurasjon
CACHE_DIR = "data_storage/fundamental_cache"
CACHE_MAX_AGE_HOURS = 24  # Fundamental data endres ikke ofte

def _get_cache_path(ticker: str) -> str:
    """Returnerer cache-filsti for ticker."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker.replace('.', '_')}_fundamental.json")

def _is_cache_valid(cache_path: str) -> bool:
    """Sjekker om cache er gyldig."""
    if not os.path.exists(cache_path):
        return False
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - file_time
    return age < timedelta(hours=CACHE_MAX_AGE_HOURS)

def _load_from_cache(cache_path: str) -> Optional[dict]:
    """Laster data fra cache."""
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except:
        return None

def _save_to_cache(data: dict, cache_path: str):
    """Lagrer data til cache."""
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except:
        pass


def hent_fundamental_data(ticker: str, use_cache: bool = True) -> Optional[Dict]:
    """
    Henter fundamental data for en ticker.
    
    Returns:
        Dict med fundamental metrics eller None hvis feil.
    """
    # Sjekk cache først
    cache_path = _get_cache_path(ticker)
    if use_cache and _is_cache_valid(cache_path):
        cached = _load_from_cache(cache_path)
        if cached:
            return cached
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or info.get('regularMarketPrice') is None:
            return None
        
        # Hent nøkkelmetrikker (med fallbacks)
        data = {
            'ticker': ticker,
            'navn': info.get('shortName', info.get('longName', ticker)),
            'sektor': info.get('sector', 'Ukjent'),
            'bransje': info.get('industry', 'Ukjent'),
            
            # Verdsettelse
            'pe_trailing': info.get('trailingPE'),
            'pe_forward': info.get('forwardPE'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'peg_ratio': info.get('pegRatio'),
            
            # Lønnsomhet
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            
            # Vekst
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            
            # Finansiell helse
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            
            # Utbytte
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),
            
            # Størrelse
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            
            # Meta
            'hentet': datetime.now().isoformat()
        }
        
        # Lagre til cache
        if use_cache:
            _save_to_cache(data, cache_path)
        
        return data
        
    except Exception as e:
        print(f"Feil ved henting av fundamental data for {ticker}: {e}")
        return None


def beregn_fundamental_score(data: Dict) -> Tuple[float, Dict[str, float]]:
    """
    Beregner en fundamental score (0-100) basert på flere metrikker.
    
    Høyere score = bedre fundamentale forhold.
    
    Returns:
        (total_score, delscorer_dict)
    """
    if not data:
        return 50.0, {}
    
    scores = {}
    weights = {}
    
    # === VERDSETTELSE (25%) ===
    # Lavere P/E er bedre (men ikke negativ)
    pe = data.get('pe_trailing') or data.get('pe_forward')
    if pe and pe > 0:
        if pe < 10:
            scores['pe'] = 100
        elif pe < 15:
            scores['pe'] = 80
        elif pe < 20:
            scores['pe'] = 60
        elif pe < 30:
            scores['pe'] = 40
        elif pe < 50:
            scores['pe'] = 20
        else:
            scores['pe'] = 10
        weights['pe'] = 0.15
    
    # Lavere P/B er bedre
    pb = data.get('pb_ratio')
    if pb and pb > 0:
        if pb < 1:
            scores['pb'] = 100
        elif pb < 2:
            scores['pb'] = 75
        elif pb < 3:
            scores['pb'] = 50
        elif pb < 5:
            scores['pb'] = 30
        else:
            scores['pb'] = 15
        weights['pb'] = 0.10
    
    # === LØNNSOMHET (30%) ===
    # Høyere profit margin er bedre
    pm = data.get('profit_margin')
    if pm is not None:
        pm_pct = pm * 100 if pm < 1 else pm
        if pm_pct > 20:
            scores['profit_margin'] = 100
        elif pm_pct > 15:
            scores['profit_margin'] = 80
        elif pm_pct > 10:
            scores['profit_margin'] = 60
        elif pm_pct > 5:
            scores['profit_margin'] = 40
        elif pm_pct > 0:
            scores['profit_margin'] = 25
        else:
            scores['profit_margin'] = 10
        weights['profit_margin'] = 0.15
    
    # ROE (Return on Equity) - høyere er bedre
    roe = data.get('roe')
    if roe is not None:
        roe_pct = roe * 100 if roe < 1 else roe
        if roe_pct > 20:
            scores['roe'] = 100
        elif roe_pct > 15:
            scores['roe'] = 80
        elif roe_pct > 10:
            scores['roe'] = 60
        elif roe_pct > 5:
            scores['roe'] = 40
        elif roe_pct > 0:
            scores['roe'] = 25
        else:
            scores['roe'] = 10
        weights['roe'] = 0.15
    
    # === VEKST (20%) ===
    # Revenue growth
    rev_growth = data.get('revenue_growth')
    if rev_growth is not None:
        rg_pct = rev_growth * 100 if abs(rev_growth) < 2 else rev_growth
        if rg_pct > 20:
            scores['revenue_growth'] = 100
        elif rg_pct > 10:
            scores['revenue_growth'] = 80
        elif rg_pct > 5:
            scores['revenue_growth'] = 60
        elif rg_pct > 0:
            scores['revenue_growth'] = 40
        elif rg_pct > -10:
            scores['revenue_growth'] = 25
        else:
            scores['revenue_growth'] = 10
        weights['revenue_growth'] = 0.20
    
    # === FINANSIELL HELSE (15%) ===
    # Debt/Equity - lavere er bedre
    de = data.get('debt_to_equity')
    if de is not None:
        if de < 0.3:
            scores['debt_equity'] = 100
        elif de < 0.5:
            scores['debt_equity'] = 80
        elif de < 1.0:
            scores['debt_equity'] = 60
        elif de < 2.0:
            scores['debt_equity'] = 40
        else:
            scores['debt_equity'] = 20
        weights['debt_equity'] = 0.15
    
    # === UTBYTTE (10%) ===
    div_yield = data.get('dividend_yield')
    if div_yield is not None:
        dy_pct = div_yield * 100 if div_yield < 1 else div_yield
        if dy_pct > 5:
            scores['dividend'] = 100
        elif dy_pct > 3:
            scores['dividend'] = 80
        elif dy_pct > 2:
            scores['dividend'] = 60
        elif dy_pct > 1:
            scores['dividend'] = 40
        elif dy_pct > 0:
            scores['dividend'] = 25
        else:
            scores['dividend'] = 10  # Ingen utbytte er ikke nødvendigvis dårlig
        weights['dividend'] = 0.10
    
    # Beregn vektet gjennomsnitt
    if not scores:
        return 50.0, {}
    
    # Normaliser vekter
    total_weight = sum(weights.values())
    if total_weight > 0:
        weighted_sum = sum(scores[k] * weights[k] for k in scores)
        total_score = weighted_sum / total_weight
    else:
        total_score = np.mean(list(scores.values()))
    
    return round(total_score, 1), scores


def get_fundamental_verdict(score: float) -> Tuple[str, str]:
    """
    Returnerer en vurdering basert på fundamental score.
    
    Returns:
        (emoji, beskrivelse)
    """
    if score >= 80:
        return "🟢", "Sterk"
    elif score >= 65:
        return "🟡", "God"
    elif score >= 50:
        return "🟠", "Middels"
    elif score >= 35:
        return "🔴", "Svak"
    else:
        return "⚫", "Dårlig"


def formater_fundamental_visning(data: Dict) -> str:
    """
    Formaterer fundamental data for visning i UI.
    """
    if not data:
        return "Ingen data tilgjengelig"
    
    lines = []
    
    # Verdsettelse
    pe = data.get('pe_trailing') or data.get('pe_forward')
    if pe:
        lines.append(f"P/E: {pe:.1f}")
    
    pb = data.get('pb_ratio')
    if pb:
        lines.append(f"P/B: {pb:.2f}")
    
    # Lønnsomhet
    pm = data.get('profit_margin')
    if pm is not None:
        pm_pct = pm * 100 if pm < 1 else pm
        lines.append(f"Profit Margin: {pm_pct:.1f}%")
    
    roe = data.get('roe')
    if roe is not None:
        roe_pct = roe * 100 if roe < 1 else roe
        lines.append(f"ROE: {roe_pct:.1f}%")
    
    # Vekst
    rev_growth = data.get('revenue_growth')
    if rev_growth is not None:
        rg_pct = rev_growth * 100 if abs(rev_growth) < 2 else rev_growth
        lines.append(f"Omsetningsvekst: {rg_pct:+.1f}%")
    
    # Gjeld
    de = data.get('debt_to_equity')
    if de is not None:
        lines.append(f"Gjeld/EK: {de:.2f}")
    
    # Utbytte
    div_yield = data.get('dividend_yield')
    if div_yield is not None:
        dy_pct = div_yield * 100 if div_yield < 1 else div_yield
        lines.append(f"Utbytte: {dy_pct:.1f}%")
    
    return " | ".join(lines) if lines else "Begrenset data"


def hent_fundamental_for_liste(tickers: list, use_cache: bool = True) -> Dict[str, Dict]:
    """
    Henter fundamental data for en liste med tickers.
    
    Returns:
        Dict med ticker -> fundamental_data
    """
    resultater = {}
    
    for ticker in tickers:
        data = hent_fundamental_data(ticker, use_cache=use_cache)
        if data:
            score, delscorer = beregn_fundamental_score(data)
            data['fundamental_score'] = score
            data['delscorer'] = delscorer
            emoji, verdict = get_fundamental_verdict(score)
            data['verdict'] = verdict
            data['verdict_emoji'] = emoji
            resultater[ticker] = data
    
    return resultater


def clear_fundamental_cache():
    """Sletter all fundamental-cache."""
    if os.path.exists(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            if f.endswith('_fundamental.json'):
                try:
                    os.remove(os.path.join(CACHE_DIR, f))
                except:
                    pass


# === TEST ===
if __name__ == "__main__":
    # Test med noen Oslo Børs-aksjer
    test_tickers = ['EQNR.OL', 'MOWI.OL', 'DNB.OL']
    
    for ticker in test_tickers:
        print(f"\n{'='*50}")
        print(f"Testing {ticker}")
        print('='*50)
        
        data = hent_fundamental_data(ticker, use_cache=False)
        if data:
            score, delscorer = beregn_fundamental_score(data)
            emoji, verdict = get_fundamental_verdict(score)
            
            print(f"Sektor: {data.get('sektor')}")
            print(f"Fundamental Score: {score} {emoji} ({verdict})")
            print(f"Delscorer: {delscorer}")
            print(f"Visning: {formater_fundamental_visning(data)}")
        else:
            print("Kunne ikke hente data")
