# shared_cache.py
"""
Sentraliserte cache-funksjoner for InveStock Pro.
Brukes av alle sider for å hente data og beregninger effektivt.
"""

import streamlit as st
import pandas as pd
import yfinance as yf

import data
import logic


# =============================================================================
# DATA CACHING
# =============================================================================

@st.cache_data(show_spinner=False)
def cached_hent_data() -> pd.DataFrame:
    """Henter hoveddata med caching."""
    return data.hent_data()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_hent_markedsdata_df() -> pd.DataFrame:
    """Henter markedsdata med 1-times cache."""
    return data.hent_markedsdata_df()


@st.cache_data(show_spinner=False)
def cached_beregn_tekniske_indikatorer(df: pd.DataFrame) -> pd.DataFrame:
    """Beregner tekniske indikatorer med caching."""
    return logic.beregn_tekniske_indikatorer(df)


# =============================================================================
# SIGNAL CACHING (session state basert)
# =============================================================================

def hent_signaler_cached(ticker: str, df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Cache signalberegning per ticker.
    Bruker session_state for å holde data mellom reruns.
    """
    if df_t.empty:
        return pd.DataFrame(index=df_t.index)
    
    cache = st.session_state.get('signal_cache', {})
    last_idx = df_t.index[-1]
    sig = cache.get(ticker)
    
    if (sig is None) or (sig.get('len') != len(df_t)) or (sig.get('last_idx') != last_idx):
        cache[ticker] = {
            'len': len(df_t),
            'last_idx': last_idx,
            'signaler': logic.sjekk_strategier(df_t)
        }
        st.session_state['signal_cache'] = cache
    
    return cache[ticker]['signaler']


def hent_backtest_cached(ticker: str, df_t: pd.DataFrame, signaler: pd.DataFrame, 
                         strat_key: str, holdingperiode: int = 20) -> dict:
    """
    Cache backtest per ticker/strategi/datasignatur.
    """
    if df_t.empty:
        return None
    
    # Strategi-specifika backtest-parametere
    if strat_key == 'Strength_Pullback':
        # Pullback-strategier trenger lösare trailing stop för høyere volatilitet
        trailing_stop_atr = 1.5
        profit_target_atr = 2.5
    else:
        # Default for andre strategier
        trailing_stop_atr = 1.0
        profit_target_atr = 2.0
    
    # Likviditetsfilter: Min dagsomsetning 5 mill NOK
    min_dagsomsetning = 5_000_000
    
    cache_key = (ticker, strat_key, holdingperiode, trailing_stop_atr, 'realistic_entry_v3_liquidity', len(df_t), df_t.index[-1])
    bt_cache = st.session_state.get('backtest_cache', {})
    
    if cache_key not in bt_cache:
        bt_cache[cache_key] = logic.backtest_strategi(
            df_t, signaler, strat_key, 
            holdingperiode=holdingperiode,
            trailing_stop_atr=trailing_stop_atr,
            profit_target_atr=profit_target_atr,
            min_dagsomsetning=min_dagsomsetning
        )
        st.session_state['backtest_cache'] = bt_cache
    
    return bt_cache[cache_key]


# =============================================================================
# HJELPEFUNKSJONER FOR TEKNISK ANALYSE
# =============================================================================

def get_ticker_data_with_indicators(ticker: str, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Henter data for én ticker med tekniske indikatorer.
    Bruker caching for å unngå gjentatte beregninger.
    """
    df_t = df_clean[df_clean['Ticker'] == ticker].copy()
    
    if df_t.empty or len(df_t) < 50:
        return df_t
    
    teknisk_cache = st.session_state.get('teknisk_cache', {})
    
    if ticker not in teknisk_cache:
        teknisk_cache[ticker] = cached_beregn_tekniske_indikatorer(df_t)
        st.session_state['teknisk_cache'] = teknisk_cache
    
    return teknisk_cache[ticker]
