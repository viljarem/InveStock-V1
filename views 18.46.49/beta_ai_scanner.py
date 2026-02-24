# pages/beta_ai_scanner.py
"""Beta: AI Scanner side — delegerer til beta_ml-modulen."""

import streamlit as st


def render():
    """Renderer Beta AI Scanner-siden."""
    try:
        import beta_ml
    except ImportError:
        st.error("Beta ML-modulen er ikke tilgjengelig.")
        return

    df_clean = st.session_state.get('df_clean')
    unike_tickers = st.session_state.get('unike_tickers', [])

    if df_clean is None or df_clean.empty:
        st.warning("Ingen data tilgjengelig.")
        return

    beta_ml.vis_beta_side(df_clean, unike_tickers)
