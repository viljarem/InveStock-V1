# shared_state.py
"""
Sentralisert håndtering av Streamlit session state.
Brukes av alle sider for å dele data og unngå duplisert initialisering.
"""

import streamlit as st
import pandas as pd


def init_session_state():
    """
    Initialiserer all nødvendig session state.
    Kall denne funksjonen først i hver side.
    """
    # Lazy import for å unngå sirkulære avhengigheter
    from shared_cache import cached_hent_data, cached_hent_markedsdata_df
    
    # Hoveddata
    if 'df' not in st.session_state:
        st.session_state['df'] = cached_hent_data()
    
    if 'df_market' not in st.session_state:
        st.session_state['df_market'] = cached_hent_markedsdata_df()
    
    # Cacher for beregninger
    if 'teknisk_cache' not in st.session_state:
        st.session_state['teknisk_cache'] = {}
    
    if 'signal_cache' not in st.session_state:
        st.session_state['signal_cache'] = {}
    
    if 'backtest_cache' not in st.session_state:
        st.session_state['backtest_cache'] = {}
    
    # Navigasjon
    if 'side_valg_radio' not in st.session_state:
        st.session_state['side_valg_radio'] = "Hjem"
    
    # Valgt ticker (for navigasjon mellom sider)
    if 'valgt_ticker' not in st.session_state:
        st.session_state['valgt_ticker'] = None
    
    # Sist oppdatert timestamp
    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = None


def clear_all_caches():
    """Tømmer alle cacher. Brukes ved dataoppdatering."""
    st.session_state['teknisk_cache'] = {}
    st.session_state['signal_cache'] = {}
    st.session_state['backtest_cache'] = {}
    st.cache_data.clear()


def get_df() -> pd.DataFrame:
    """Returnerer hoveddata DataFrame."""
    return st.session_state.get('df', pd.DataFrame())


def get_df_market() -> pd.DataFrame:
    """Returnerer markedsdata DataFrame."""
    return st.session_state.get('df_market', pd.DataFrame())


def get_teknisk_cache() -> dict:
    """Returnerer teknisk-indikator cache."""
    return st.session_state.get('teknisk_cache', {})


def set_valgt_ticker(ticker: str):
    """Setter valgt ticker for navigasjon."""
    st.session_state['valgt_ticker'] = ticker


def get_valgt_ticker() -> str:
    """Henter valgt ticker."""
    return st.session_state.get('valgt_ticker')


def navigate_to(page: str):
    """Navigerer til en annen side."""
    st.session_state['navigate_to'] = page
    st.rerun()
