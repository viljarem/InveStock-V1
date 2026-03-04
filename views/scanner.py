import streamlit as st
import pandas as pd
import time
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import logic
import data
import config
import fundamental_data
from shared_cache import cached_beregn_tekniske_indikatorer, hent_signaler_cached, hent_backtest_cached
from components import highlight_kvalitet, highlight_rs, highlight_exit, highlight_utvikling, highlight_false_breakout, highlight_rr
import user_settings
from log_config import get_logger
import pine_signal as ps

_scanner_logger = get_logger(__name__)

try:
    import smart_money
    smart_money_tilgjengelig = True
except ImportError:
    smart_money_tilgjengelig = False

try:
    import regime_model
    regime_tilgjengelig = True
except ImportError:
    regime_tilgjengelig = False

try:
    import insider_monitor
    insider_tilgjengelig = True
except ImportError:
    insider_tilgjengelig = False

try:
    import pattern_logic
    pattern_tilgjengelig = True
except ImportError:
    pattern_tilgjengelig = False

try:
    from chart_utils import render_modern_chart, LWC_INSTALLED
    CHART_OK = LWC_INSTALLED
except ImportError:
    CHART_OK = False


# === LIVE PRIS FRA YAHOO FINANCE ===
@st.cache_data(ttl=300, show_spinner=False)
def _hent_live_priser(tickers_tuple):
    """
    Batch-henter siste kurs fra Yahoo Finance for en liste tickers.
    Cachet i 5 minutter. Tar tuple for hashbarhet.
    Returnerer dict: {ticker: siste_pris} (float eller None).
    """
    tickers = list(tickers_tuple)
    resultat = {}
    if not tickers:
        return resultat
    
    try:
        # Batch-download: period='1d' gir dagens data (eller siste handelsdag)
        df = yf.download(tickers, period='2d', progress=False, threads=True)
        
        if df.empty:
            return resultat
        
        if len(tickers) == 1:
            # Enkel ticker: ingen MultiIndex
            if 'Close' in df.columns and len(df) > 0:
                resultat[tickers[0]] = float(df['Close'].iloc[-1])
        else:
            # Multi-ticker: MultiIndex kolonner
            if 'Close' in df.columns.get_level_values(0):
                close_df = df['Close']
                for t in tickers:
                    if t in close_df.columns:
                        vals = close_df[t].dropna()
                        if len(vals) > 0:
                            resultat[t] = float(vals.iloc[-1])
    except Exception as e:
        _scanner_logger.warning(f"Live pris-henting feilet: {e}")
    
    return resultat


def _highlight_dagendring(val):
    """Fargekoder daglig endring (Dag%)."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ''
    if v > 1.5:
        return 'color: #00c853; font-weight: 600'
    elif v > 0:
        return 'color: #66bb6a'
    elif v > -1.5:
        return 'color: #ffa726'
    else:
        return 'color: #ef5350; font-weight: 600'


@st.dialog("Scanner — Quick View", width="large")
def _vis_scanner_popup():
    """Popup med kursdiagram, signalmarkører og info for valgt scanner-rad."""
    info = st.session_state.get('_scanner_popup')
    if not info or info is None:
        return

    ticker = info['ticker']
    row_data = info.get('row', {})
    strat_navn = str(row_data.get('Strategi', ''))
    pris = row_data.get('Pris', '')
    kvalitet = row_data.get('Kvalitet', '')
    signal_dato = row_data.get('Signal', '')
    dager_siden = row_data.get('Dager', '')
    utv_pct = row_data.get('Utv%', '')

    selskap = data.ticker_til_navn(ticker) if hasattr(data, 'ticker_til_navn') else ticker.replace('.OL', '')

    # Strategi-navn → nøkkel mapping
    _navn_til_nøkkel = {
        "Kort Sikt (RSI Dip)": "Kort_Sikt_RSI",
        "Momentum Burst": "Momentum_Burst",
        "Golden Cross": "Golden_Cross",
        "Ichimoku Breakout": "Ichimoku_Breakout",
        "Wyckoff Spring": "Wyckoff_Spring",
        "Bull Race Prep": "Bull_Race_Prep",
        "VCP (Minervini)": "VCP_Pattern",
        "Pocket Pivot": "Pocket_Pivot",
        "Strength Pullback": "Strength_Pullback",
    }
    strat_key = _navn_til_nøkkel.get(strat_navn)

    # Header med strategi-badge
    badge_html = ""
    if strat_navn and strat_navn != '-':
        badge_html = (f'<span style="background: #667eea; color: #fff; padding: 3px 12px; '
                      f'border-radius: 6px; font-size: 13px; font-weight: 600;">{strat_navn}</span>')
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
        <span style="font-size: 22px; font-weight: 700; color: #fff;">{selskap}</span>
        <span style="color: #888;">{ticker}</span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)

    # Nøkkeltall fra scanner-raden
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Signal", signal_dato if signal_dato else '—')
    m2.metric("Dager siden", dager_siden if dager_siden != '' else '—')
    m3.metric("Kvalitet", kvalitet if kvalitet else '—')
    utv_str = f"{utv_pct:+.1f}%" if isinstance(utv_pct, (int, float)) else str(utv_pct)
    m4.metric("Utvikling", utv_str)

    # Ekstra nøkkeltall fra scanner-raden
    score_val = row_data.get('Score', '')
    rs_val = row_data.get('RS', '')
    sektor_val = row_data.get('Sektor', '')
    exit_val = row_data.get('Exit⚠️', '')
    peak_val = row_data.get('Peak%', '')
    fb_val = row_data.get('FB', '')
    rr_val = row_data.get('R:R', '')
    siste_val = row_data.get('Siste', '')
    dag_val = row_data.get('Dag%', '')
    
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Score", score_val if score_val != '' else '—')
    m6.metric("RS", rs_val if rs_val != '' else '—')
    if siste_val and dag_val != '':
        dag_delta = f"{dag_val:+.2f}%" if isinstance(dag_val, (int, float)) else str(dag_val)
        m7.metric("Live kurs", siste_val, delta=dag_delta)
    elif rr_val != '' and isinstance(rr_val, (int, float)):
        m7.metric("R:R", f"{rr_val:.1f}")
    elif peak_val != '' and isinstance(peak_val, (int, float)):
        m7.metric("Peak%", f"{peak_val:+.1f}%")
    else:
        m7.metric("R:R", '—')
    exit_display = "⚠️ Ja" if exit_val == '⚠️' else ("❌ FB" if fb_val == '❌' else "✅ Nei")
    m8.metric("Exit-signal", exit_display)

    # === CHART MED SIGNALMARKØRER ===
    # Hent brukerens chart-preferanser
    _chart_prefs = user_settings.load_settings().get('chart', {})
    _ind_prefs = _chart_prefs.get('indikatorer', {})
    
    df_clean = st.session_state.get('df_clean', pd.DataFrame())
    df_t = df_clean[df_clean['Ticker'] == ticker].copy()

    if not df_t.empty and len(df_t) >= 50 and CHART_OK:
        # Beregn tekniske indikatorer (SMA etc.)
        df_ind = cached_beregn_tekniske_indikatorer(df_t)
        df_chart = df_ind.iloc[-260:].copy()  # ~1 år data

        # Signaler — beregn og vis på chartet
        signaler = hent_signaler_cached(ticker, df_ind)
        signal_data = {}

        if strat_key:
            # Vis kun den aktive strategien
            signal_keys_to_show = [strat_key]
        else:
            # "Alle strategier" — vis alle
            signal_keys_to_show = list(_navn_til_nøkkel.values())

        chart_start = df_chart.index[0]
        signal_data = logic.hent_signaler_for_chart(
            df_ind, signaler, signal_keys_to_show, start_date=chart_start
        )

        # Insider-markeringer
        insider_trades = None
        if insider_tilgjengelig:
            try:
                handler = insider_monitor.hent_innsidehandler(dager=90)
                insider_trades = [h for h in handler if h.get('ticker') == ticker]
            except Exception:
                pass

        render_modern_chart(
            df_chart,
            indicators={
                'sma_50': _ind_prefs.get('sma_50', True),
                'sma_200': _ind_prefs.get('sma_200', True),
                'sma_20': _ind_prefs.get('sma_20', False),
                'bollinger': _ind_prefs.get('bollinger', False),
                'rsi': _ind_prefs.get('rsi', False),
                'macd': _ind_prefs.get('macd', False),
                'volume': _ind_prefs.get('volum', True),
            },
            signals=signal_data,
            signal_keys=signal_keys_to_show,
            chart_height=400,
            insider_trades=insider_trades if insider_trades else None,
        )

        # Vis signalhistorikk under chartet
        total_signals = sum(len(v) for v in signal_data.values() if not v.empty) if signal_data else 0
        if total_signals > 0:
            with st.expander(f"📊 Signalhistorikk ({total_signals} signaler i perioden)", expanded=False):
                # Samle alle signaler og sorter etter dato (nyeste først)
                all_signal_items = []
                for sk in signal_keys_to_show:
                    sd = signal_data.get(sk, pd.DataFrame())
                    if sd.empty:
                        continue
                    sn = next((n for n, k in _navn_til_nøkkel.items() if k == sk), sk)
                    for idx, row in sd.iterrows():
                        all_signal_items.append({
                            'date': idx,
                            'strategi': sn,
                            'kvalitet': row.get('kvalitet_klasse', '?'),
                            'utvikling': row.get('utvikling_pst', 0),
                            'dager': row.get('dager_siden', 0)
                        })
                
                # Sorter etter dato, nyeste først
                all_signal_items.sort(key=lambda x: x['date'], reverse=True)
                
                for item in all_signal_items:
                    farge = '#00c853' if item['utvikling'] > 0 else '#ff5252'
                    st.markdown(
                        f"- **{item['strategi']}** {item['date'].strftime('%d.%m.%Y')} — "
                        f"Kvalitet **{item['kvalitet']}** — "
                        f"<span style='color:{farge}'>{item['utvikling']:+.1f}%</span> "
                        f"({item['dager']}d siden)",
                        unsafe_allow_html=True,
                    )
    elif not CHART_OK:
        st.caption("Chart ikke tilgjengelig (mangler streamlit-lightweight-charts)")
    elif df_t.empty:
        st.caption("Ingen prisdata tilgjengelig for denne tickeren.")
    else:
        st.caption("For lite data for chart (trenger minst 50 datapunkter).")

    # Knapperrad
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Åpne i Teknisk Analyse", type="primary", key="popup_ta", use_container_width=True):
            st.session_state['valgt_ticker'] = ticker
            st.session_state['navigate_to'] = "Teknisk Analyse"
            st.rerun()
    with col2:
        if st.button("Lukk", key="popup_close", use_container_width=True):
            # Marker at popup er lukket manuelt og rens popup-state
            st.session_state['_popup_manually_closed'] = True
            st.session_state['_scanner_popup'] = None
            st.stop()


# ── Pine Signal highlight-hjelpere ──────────────────────────────────────────

def _highlight_pine_signal(val: str) -> str:
    """Fargekoder Pine Signal-kolonnen."""
    farge = ps.SIGNAL_FARGE.get(val, '')
    if not farge:
        return ''
    return f'color: {farge}; font-weight: 600'


def _highlight_pine_score(val) -> str:
    """Fargekoder Pine Score [-1, +1]."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ''
    if v >= 0.60:
        return 'color: #00c853; font-weight: 700'
    if v >= 0.20:
        return 'color: #66bb6a; font-weight: 600'
    if v <= -0.60:
        return 'color: #b71c1c; font-weight: 700'
    if v <= -0.20:
        return 'color: #ef5350; font-weight: 600'
    return 'color: #ffa726'


# ── Pine Signal render-funksjon ──────────────────────────────────────────────

def _render_pine_signal_modus(df_clean: pd.DataFrame, unike_tickers: list) -> None:
    """
    Rendrer Pine Signal-modus: TradingView-stil sammensatt kjøpssignal-scanner
    for alle aksjer på Oslo Børs.

    Kombiner 7 oscillatorer + 13 glidende gjennomsnitt til ett signal
    (Sterkt Kjøp / Kjøp / Nøytral / Selg / Sterkt Selg) per aksje.
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d2137 0%, #0a1628 100%);
                border-radius: 14px; padding: 18px 22px; margin-bottom: 18px;
                border: 1px solid rgba(102,126,234,0.2);">
        <div style="font-size: 15px; font-weight: 700; color: #a8b4f0; margin-bottom: 4px;">
            🌲 Pine Signal — TradingView-stil sammensatt signal
        </div>
        <div style="font-size: 12px; color: rgba(255,255,255,0.55); line-height: 1.5;">
            Kombinerer <b>7 oscillatorer</b> (RSI, MACD, Stochastic, CCI, Momentum, AO, Bull/Bear Power)
            og <b>opptil 13 glidende gjennomsnitt</b> (EMA/SMA 10–200) til ett samlet signal per aksje.
            Samme metodikk som TradingView sin «Technical Analysis»-sammendrag.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Filtre ──────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        filter_signal = st.multiselect(
            "Vis signal",
            options=[ps.STERKT_KJOP, ps.KJOP, ps.NØYTRAL, ps.SELG, ps.STERKT_SELG],
            default=[ps.STERKT_KJOP, ps.KJOP],
            help="Filtrer på ønsket signal-type"
        )
    with fc2:
        min_kjop = st.slider("Min Kjøp-signaler", 0, 20, 0,
                             help="Minimum antall kjøpssignaler (av ~20 totalt)")
    with fc3:
        sorter_pine = st.selectbox(
            "Sorter etter",
            ["Score (høy→lav)", "Kjøp (flest)", "Selg (flest)"],
            index=0,
        )

    # ── Skann-knapp og cache ─────────────────────────────────────────────
    cache_key = 'pine_signal_cache'
    pine_df_cached = st.session_state.get(cache_key)

    col_knapp, col_info = st.columns([1, 3])
    with col_knapp:
        skann_nå = st.button("🔍 Kjør Pine Scan", type="primary", use_container_width=True)
    with col_info:
        if pine_df_cached is not None:
            st.caption(f"Viser cachet scan ({len(pine_df_cached)} aksjer). Klikk «Kjør Pine Scan» for å oppdatere.")
        else:
            st.caption("Klikk «Kjør Pine Scan» for å analysere alle aksjer med Pine-metoden.")

    if skann_nå:
        with st.spinner(f"Analyserer {len(unike_tickers)} aksjer med Pine Signal-metoden…"):
            df_dict: dict[str, pd.DataFrame] = {}
            for ticker in unike_tickers:
                df_t = df_clean[df_clean['Ticker'] == ticker]
                if len(df_t) < 50:
                    continue
                try:
                    df_ind = cached_beregn_tekniske_indikatorer(df_t)
                    df_dict[ticker] = df_ind
                except Exception:
                    pass
            pine_df_cached = ps.skann_pine_signaler(df_dict)
        st.session_state[cache_key] = pine_df_cached

    if pine_df_cached is None or pine_df_cached.empty:
        st.info("Trykk «Kjør Pine Scan» for å starte analysen.")
        _render_pine_forklaring()
        return

    # ── Filtrer og sorter ────────────────────────────────────────────────
    vis_df = pine_df_cached.copy()
    if filter_signal:
        vis_df = vis_df[vis_df['Signal'].isin(filter_signal)]
    if min_kjop > 0:
        vis_df = vis_df[vis_df['Kjøp'] >= min_kjop]

    if sorter_pine == "Score (høy→lav)":
        vis_df = vis_df.sort_values('Score', ascending=False)
    elif sorter_pine == "Kjøp (flest)":
        vis_df = vis_df.sort_values('Kjøp', ascending=False)
    else:
        vis_df = vis_df.sort_values('Selg', ascending=False)

    vis_df = vis_df.reset_index(drop=True)

    # ── Sammendrag-metrics ───────────────────────────────────────────────
    total = len(pine_df_cached)
    n_sk  = (pine_df_cached['Signal'] == ps.STERKT_KJOP).sum()
    n_k   = (pine_df_cached['Signal'] == ps.KJOP).sum()
    n_n   = (pine_df_cached['Signal'] == ps.NØYTRAL).sum()
    n_s   = (pine_df_cached['Signal'] == ps.SELG).sum()
    n_ss  = (pine_df_cached['Signal'] == ps.STERKT_SELG).sum()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🟢 Sterkt Kjøp", n_sk)
    m2.metric("🟩 Kjøp",        n_k)
    m3.metric("🟡 Nøytral",     n_n)
    m4.metric("🔴 Selg",        n_s)
    m5.metric("⛔ Sterkt Selg", n_ss)

    st.markdown(f"**{len(vis_df)} av {total} aksjer** vises (filtrert)")

    # ── Legg til selskapsnavn for bedre lesbarhet ────────────────────────
    if hasattr(data, 'ticker_til_navn'):
        vis_df.insert(1, 'Selskap', vis_df['Ticker'].apply(
            lambda t: data.ticker_til_navn(t) or t.replace('.OL', '')))
    
    # Legg til emoji foran signal
    vis_df['Signal'] = vis_df['Signal'].apply(
        lambda s: f"{ps.SIGNAL_EMOJI.get(s, '')} {s}"
    )

    # ── Tabell ───────────────────────────────────────────────────────────
    styled = vis_df.style
    # Fargekod Signal-kolonnen (fjern emoji for oppslag)
    def _hl_sig(val: str) -> str:
        signal_text = val.split(' ', 1)[-1] if ' ' in val else val
        return _highlight_pine_signal(signal_text)

    styled = styled.map(_hl_sig, subset=['Signal'])
    styled = styled.map(_highlight_pine_score, subset=['Score'])
    styled = styled.format({'Score': '{:+.3f}', 'Osc Score': '{:+.3f}', 'MA Score': '{:+.3f}'})

    st.dataframe(styled, hide_index=True, width="stretch",
                 use_container_width=True)

    # ── Eksport ──────────────────────────────────────────────────────────
    csv = vis_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Last ned CSV",
        data=csv,
        file_name="pine_signal_oslo_bors.csv",
        mime="text/csv",
    )

    _render_pine_forklaring()


def _render_pine_forklaring() -> None:
    """Viser kort metodikk-forklaring for Pine Signal-modus."""
    with st.expander("ℹ️ Metodikk — Pine Signal", expanded=False):
        st.markdown(f"""
### Hvordan Pine Signal beregnes

Hvert subsignal gir **+1 (Kjøp)**, **0 (Nøytral)** eller **−1 (Selg)**.
Samlet score = gjennomsnitt av alle subsignaler → normalisert **[-1, +1]**.

| Terskel | Signal |
|---------|--------|
| score ≥ +0.60 | {ps.SIGNAL_EMOJI[ps.STERKT_KJOP]} **Sterkt Kjøp** |
| score ≥ +0.20 | {ps.SIGNAL_EMOJI[ps.KJOP]} **Kjøp** |
| −0.19 til +0.19 | {ps.SIGNAL_EMOJI[ps.NØYTRAL]} **Nøytral** |
| score ≤ −0.20 | {ps.SIGNAL_EMOJI[ps.SELG]} **Selg** |
| score ≤ −0.60 | {ps.SIGNAL_EMOJI[ps.STERKT_SELG]} **Sterkt Selg** |

#### Oscillatorer (7 subsignaler)
| Indikator | Kjøp-betingelse | Selg-betingelse |
|-----------|-----------------|-----------------|
| RSI(14) | RSI < 30 (oversold) | RSI > 70 (overkjøpt) |
| MACD(12,26,9) | MACD > Signal-linje | MACD < Signal-linje |
| Stochastic %K/%D(14,3) | Begge < 20 eller K > D | Begge > 80 eller K < D |
| CCI(20) | CCI < −100 | CCI > +100 |
| Momentum(10) | Momentum > 0 | Momentum < 0 |
| Awesome Oscillator | AO > 0 og stigende | AO < 0 og fallende |
| Bull/Bear Power | High og Low begge > EMA13 | High og Low begge < EMA13 |

#### Glidende gjennomsnitt (opptil 13 subsignaler)
Pris **over** EMA/SMA = Kjøp, pris **under** = Selg.
Perioder: EMA 10/20/30/50/100/200 og SMA 10/20/30/50/100/150/200.

> **Merk:** Pine Signal er et *teknisk* hjelpemiddel.
> Bruk alltid fundamentalanalyse og risikostyring i tillegg.
        """)


def render():
    df_clean = st.session_state['df_clean']
    unike_tickers = st.session_state['unike_tickers']

    # === STRATEGI-KONFIGURASJON ===
    navn_til_nøkkel = {
        "Kort Sikt (RSI Dip)": "Kort_Sikt_RSI", 
        "Momentum Burst": "Momentum_Burst",
        "Golden Cross": "Golden_Cross", 
        "Ichimoku Breakout": "Ichimoku_Breakout",
        "Wyckoff Spring": "Wyckoff_Spring", 
        "Bull Race Prep": "Bull_Race_Prep",
        "VCP (Minervini)": "VCP_Pattern",
        "Pocket Pivot": "Pocket_Pivot",
        "Strength Pullback": "Strength_Pullback"
    }
    
    # Strategi-metadata for visuell sammenligning
    strategi_metadata = {
        "Kort Sikt (RSI Dip)": {"horisont": "1-5 dager", "stil": "Reversal", "risiko": "Høy", "ikon": "⚡"},
        "Momentum Burst": {"horisont": "3-10 dager", "stil": "Momentum", "risiko": "Høy", "ikon": "🚀"},
        "Golden Cross": {"horisont": "Uker-måneder", "stil": "Trend", "risiko": "Lav", "ikon": "✨"},
        "Ichimoku Breakout": {"horisont": "1-4 uker", "stil": "Trend", "risiko": "Medium", "ikon": "☁️"},
        "Wyckoff Spring": {"horisont": "1-3 uker", "stil": "Reversal", "risiko": "Medium", "ikon": "🌱"},
        "Bull Race Prep": {"horisont": "1-4 uker", "stil": "Momentum", "risiko": "Medium", "ikon": "🐂"},
        "VCP (Minervini)": {"horisont": "2-8 uker", "stil": "Pattern", "risiko": "Lav", "ikon": "📐"},
        "Pocket Pivot": {"horisont": "3-10 dager", "stil": "Momentum", "risiko": "Medium", "ikon": "🎯"},
        "Strength Pullback": {"horisont": "5-20 dager", "stil": "Pullback", "risiko": "Lav", "ikon": "💪"}
    }
    
    # === KOMPAKT HEADER ===
    antall_aksjer = len(unike_tickers)
    st.markdown(f"""
    <style>
    .scanner-header {{
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }}
    .scanner-title {{
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }}
    .scanner-subtitle {{
        font-size: 14px;
        color: rgba(255,255,255,0.6);
    }}
    .filter-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }}
    .filter-card-title {{
        font-size: 11px;
        font-weight: 600;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }}
    .strat-badge {{
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        color: #a8b4f0;
        margin-right: 6px;
    }}
    </style>
    
    <div class="scanner-header">
        <div class="scanner-title">📡 Scanner</div>
        <div class="scanner-subtitle">Skanner {antall_aksjer} aksjer med tekniske strategier og mønstergjenkjenning</div>
    </div>
    """, unsafe_allow_html=True)
    
    # === MODUS-VELGER ===
    if pattern_tilgjengelig:
        modus_col1, modus_col2 = st.columns([1, 3])
        with modus_col1:
            scanner_modus = st.radio(
                "Modus",
                ["📊 Signaler", "📐 Mønstre", "🌲 Pine Signal"],
                horizontal=True,
                label_visibility="collapsed"
            )
        bruk_mønster_modus = (scanner_modus == "📐 Mønstre")
        bruk_pine_modus = (scanner_modus == "🌲 Pine Signal")
    else:
        bruk_mønster_modus = False
        bruk_pine_modus = False
    
    # Last brukerpreferanser
    _prefs = user_settings.load_settings().get('scanner', {})
    
    # === PINE SIGNAL MODUS — komplett TradingView-stil skanner ===
    if bruk_pine_modus:
        _render_pine_signal_modus(df_clean, unike_tickers)
        return

    # === HOVEDFILTER-PANEL ===
    if not bruk_mønster_modus:
        # SIGNAL-MODUS
        with st.expander("⚙️ Filterinnstillinger", expanded=True):
            # Rad 1: Strategi + Kvalitet + Dager
            r1c1, r1c2, r1c3 = st.columns([2, 1, 1])
            
            with r1c1:
                strategi_valg = ["Alle strategier"] + list(navn_til_nøkkel.keys())
                valgt_navn = st.selectbox(
                    "Strategi",
                    strategi_valg,
                    index=0,
                    help="Velg én strategi eller skann med alle"
                )
            
            with r1c2:
                min_kvalitet = st.select_slider(
                    "Min. kvalitet",
                    options=['D', 'C', 'B', 'A'],
                    value=_prefs.get('min_kvalitet', 'C'),
                    help="A = Beste signaler, D = Inkluder alle"
                )
            
            with r1c3:
                max_dager = st.slider(
                    "Maks dager",
                    1, 60, _prefs.get('max_dager', 14),
                    help="Vis signaler fra siste X dager"
                )
            
            # Rad 2: RS + Volum + R:R
            r2c1, r2c2, r2c3 = st.columns(3)
            
            with r2c1:
                min_rs_rating = st.slider("Min RS-rating", 1, 99, 50, help="Relativ styrke vs. markedet (IBD-stil)")
            
            with r2c2:
                min_volum_ratio_filter = st.slider("Min volum-ratio", 0.5, 2.0, _prefs.get('min_volum_ratio', 1.0), 0.1, help="Volum relativt til 20-dagers snitt")
            
            with r2c3:
                min_rr_ratio = st.slider("Min R:R", 0.0, 4.0, _prefs.get('min_rr', 0.0), 0.5, help="Risk/Reward basert på støtte/motstand")
            
            # Rad 3: Toggles i kompakt layout
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            t1, t2, t3, t4, t5 = st.columns(5)
            
            with t1:
                vis_exit_signaler = st.checkbox("Exit-varsler", value=_prefs.get('vis_exit', True), help="Vis salgssignaler")
            with t2:
                filtrer_false_breakout = st.checkbox("FB-filter", value=_prefs.get('filtrer_false_breakout', False), help="Skjul feilet breakout")
            with t3:
                sorter_rr = st.checkbox("Sorter R:R", help="Prioriter høy R:R")
            with t4:
                bruk_regime_filter = False
                if regime_tilgjengelig:
                    bruk_regime_filter = st.checkbox("Regime", help="Tilpass til markedsregime")
            with t5:
                vis_fundamental = st.checkbox("Fundamental", help="Vis P/E, ROE etc.")
            
            # Ekstra features (skjult med mindre de er tilgjengelige)
            if smart_money_tilgjengelig or insider_tilgjengelig:
                st.markdown("---")
                e1, e2, e3 = st.columns([1, 1, 2])
                with e1:
                    bruk_smart_money = st.checkbox("Smart Money", help="Institusjonell akkumulering") if smart_money_tilgjengelig else False
                with e2:
                    bruk_insider = st.checkbox("Insider", help="Meldepliktige handler") if insider_tilgjengelig else False
            else:
                bruk_smart_money = False
                bruk_insider = False
            
            # Mønsterfilter (for signal-modus)
            if pattern_tilgjengelig:
                mønster_valg = ["Ingen mønsterfilter", "Kun bullish mønstre", "Kun bearish mønstre", "Alle mønstre"]
                mønster_filter = st.selectbox("Mønsterfilter", mønster_valg, index=0, help="Kombiner med prisformasjoner")
            else:
                mønster_filter = "Ingen mønsterfilter"
        
        mønster_type_filter = "Alle mønstre"  # Default for signal-modus
        
        # Vis strategi-info hvis valgt
        if valgt_navn != "Alle strategier":
            meta = strategi_metadata.get(valgt_navn, {})
            detaljer = logic.hent_strategi_detaljer(valgt_navn)
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.08); border-radius: 10px; padding: 14px 18px; margin: 8px 0 16px 0; border-left: 3px solid #667eea;">
                <div style="font-size: 15px; font-weight: 600; margin-bottom: 6px;">
                    {meta.get('ikon', '📊')} {valgt_navn}
                </div>
                <div style="color: rgba(255,255,255,0.7); font-size: 13px; margin-bottom: 8px;">
                    {detaljer['beskrivelse']}
                </div>
                <span class="strat-badge">{meta.get('horisont', 'Medium')}</span>
                <span class="strat-badge">{meta.get('stil', 'Trend')}</span>
                <span class="strat-badge">{meta.get('risiko', 'Medium')} risiko</span>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # MØNSTER-MODUS
        valgt_navn = "Alle strategier"
        mønster_filter = "Ingen mønsterfilter"
        
        with st.expander("⚙️ Filterinnstillinger", expanded=True):
            m1, m2, m3 = st.columns(3)
            with m1:
                mønster_type_valg = ["Alle mønstre", "Kun bullish", "Kun bearish"]
                mønster_type_filter = st.selectbox("Mønstertype", mønster_type_valg)
            with m2:
                max_dager = st.slider("Maks dager siden", 1, 30, _prefs.get('max_dager', 14), help="Mønstre fullført siste X dager")
            with m3:
                min_kvalitet = st.select_slider("Min. styrke", options=['D', 'C', 'B', 'A'], value='C', help="Mønsterkvalitet")
        
        # Defaults for mønster-modus
        min_rs_rating = 1
        min_volum_ratio_filter = 0.5
        min_rr_ratio = 0.0
        vis_exit_signaler = False
        filtrer_false_breakout = False
        sorter_rr = False
        bruk_regime_filter = False
        vis_fundamental = False
        bruk_smart_money = False
        bruk_insider = False
    
    # Sett default for maks_per_sektor
    maks_per_sektor = 999
    
    # === START SKANNING ===
    scan_col1, scan_col2 = st.columns([1, 3])
    with scan_col1:
        start_skanning = st.button("🔍 Start skanning", type="primary", use_container_width=True)
    with scan_col2:
        st.caption("Skanner alle aksjer med valgte filtre")
    
    if start_skanning:
        # Reset popup-state ved ny skanning
        st.session_state['_popup_manually_closed'] = False
        if '_scanner_popup' in st.session_state:
            st.session_state['_scanner_popup'] = None
            
        resultater = []
        filtrert_bort = 0
        scan_ticker_cache = {}  # lokal gjenbruk i samme skanning
        
        # Bestem hvilke strategier som skal skannes
        if valgt_navn == "Alle strategier":
            strategier_å_skanne = list(navn_til_nøkkel.items())
        else:
            strategier_å_skanne = [(valgt_navn, navn_til_nøkkel[valgt_navn])]
        
        # Hent markedsregime hvis aktivert
        current_regime = None
        regime_krav = None
        if bruk_regime_filter and regime_tilgjengelig:
            df_market = st.session_state.get('df_market', pd.DataFrame())
            if not df_market.empty:
                regime_data = regime_model.full_regime_analyse(df_market, n_regimes=5)
                if regime_data:
                    current_regime = regime_data['current_info']['name']
                    regime_krav = logic.regime_signal_krav(current_regime)
                    conf = regime_data['current_info']
                    conf_emoji = conf.get('confidence_emoji', '')
                    conf_level = conf.get('confidence', 'medium')
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
                                border-radius: 12px; padding: 16px; margin-bottom: 12px;
                                border-left: 4px solid {conf.get('color', '#667eea')};">
                        <div style="font-size: 16px; font-weight: 600; margin-bottom: 6px;">
                            Regime: <b>{current_regime}</b>
                            <span style="font-size: 13px; opacity: 0.8;"> — {conf_emoji} Confidence: {conf_level}</span>
                        </div>
                        <div style="font-size: 14px; color: rgba(255,255,255,0.85);">
                            Filterkrav: <b>{regime_krav['beskrivelse']}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        pb = st.progress(0, text="Analyserer tickers...")
        total_kombinasjoner = len(unike_tickers) * len(strategier_å_skanne)
        current_idx = 0
        
        # Pre-beregn sektor-momentum (20-dagers avkastning per sektor)
        sektor_momentum = logic.beregn_sektor_momentum(df_clean, perioder=20)
        
        # Hent makro-data (oljepris + valuta) for score-justering
        try:
            df_brent = data.hent_brent_data()
            olje_signal = logic.analyser_oljepris(df_brent)
        except Exception:
            olje_signal = {'signal': 'neutral', 'emoji': '⚠️', 'score_justering': 0,
                           'pris': None, 'sma50': None, 'endring_20d': None}
        try:
            df_usdnok = data.hent_usdnok_data()
            valuta_signal = logic.analyser_usdnok(df_usdnok)
        except Exception:
            valuta_signal = {'trend': 'neutral', 'emoji': '💱⚪',
                             'kurs': None, 'sma20': None, 'sma50': None, 'endring_20d': None}
        
        # Pre-beregn IBD-vektet RS for alle tickers (for ekte persentil-rangering)
        _rs_vekter = {'3mnd': 0.40, '6mnd': 0.20, '9mnd': 0.20, '12mnd': 0.20}
        _rs_perioder = {'3mnd': 63, '6mnd': 126, '9mnd': 189, '12mnd': 252}
        alle_rs_scores = []
        for _t in unike_tickers:
            _df = df_clean[df_clean['Ticker'] == _t]
            if len(_df) >= 63:
                _score = 0.0
                for _lbl, _d in _rs_perioder.items():
                    if len(_df) >= _d:
                        _avk = (_df['Close'].iloc[-1] / _df['Close'].iloc[-_d] - 1) * 100
                    else:
                        _avk = (_df['Close'].iloc[-1] / _df['Close'].iloc[0] - 1) * 100
                        _avk = _avk * (_d / len(_df))
                    _score += _rs_vekter[_lbl] * _avk
                alle_rs_scores.append(_score)
        
        # Pre-hent insider-handler (før parallell skanning — delt data)
        insider_handler = []
        if bruk_insider and insider_tilgjengelig:
            try:
                insider_handler = insider_monitor.hent_innsidehandler(dager=90)
            except Exception as e:
                st.warning(f"Insider-data utilgjengelig: {e}")
                insider_handler = []
        
        # --- PARALLELL ANALYSE ---
        # Hent benchmark-data (delt mellom alle tråder, read-only)
        df_benchmark = st.session_state.get('df_market', pd.DataFrame())
        
        # Pre-split data per ticker (unngå thread-unsafe pandas slice)
        ticker_dataframes = {}
        for ticker in unike_tickers:
            df_t = df_clean[df_clean['Ticker'] == ticker]
            if len(df_t) >= 50:
                # Beregn tekniske indikatorer (bruker cache hvis tilgjengelig)
                if ticker in st.session_state['teknisk_cache']:
                    ticker_dataframes[ticker] = st.session_state['teknisk_cache'][ticker]
                else:
                    df_med_ind = cached_beregn_tekniske_indikatorer(df_t)
                    ticker_dataframes[ticker] = df_med_ind
                    st.session_state['teknisk_cache'][ticker] = df_med_ind
        
        # Pre-beregn backtest win rate per strategi (for strategi-vekting 2.10)
        # Bruker et utvalg tickers for hastighet
        strategi_bt_bonus = {}
        bt_sample_tickers = list(ticker_dataframes.keys())[:15]
        for strat_navn, strat_key in strategier_å_skanne:
            bt_rates = []
            for t in bt_sample_tickers:
                try:
                    df_bt = ticker_dataframes[t]
                    sig_bt = hent_signaler_cached(t, df_bt)
                    bt = hent_backtest_cached(t, df_bt, sig_bt, strat_key, holdingperiode=20)
                    if bt and bt['antall_signaler'] >= 3:
                        bt_rates.append(bt['win_rate'])
                except Exception:
                    continue
            if bt_rates:
                avg_wr = sum(bt_rates) / len(bt_rates)
                # Bonus/straff basert på historisk win rate
                if avg_wr >= 65:
                    strategi_bt_bonus[strat_key] = 8
                elif avg_wr >= 55:
                    strategi_bt_bonus[strat_key] = 4
                elif avg_wr < 40:
                    strategi_bt_bonus[strat_key] = -5
                else:
                    strategi_bt_bonus[strat_key] = 0
            else:
                strategi_bt_bonus[strat_key] = 0
        
        def _analyse_en_ticker(ticker):
            """Analyserer alle strategier for én ticker. Thread-safe (kun lesing)."""
            ticker_resultater = []
            ticker_filtrert = 0
            
            df_t = ticker_dataframes[ticker]
            signaler = hent_signaler_cached(ticker, df_t)
            
            # RS-beregning
            rs_data = logic.beregn_relativ_styrke(df_t, df_benchmark, alle_tickers_avkastning=alle_rs_scores)
            rs_rating = rs_data['rs_rating'] if rs_data else 50
            
            # Multi-timeframe konvergens
            mtf_data = logic.sjekk_multi_timeframe(df_t)
            
            # Exit-signaler
            exit_data = logic.sjekk_exit_signaler(df_t) if vis_exit_signaler else {
                'skal_selge': False, 'grunner': [], 'antall_signaler': 0, 'drawdown_pct': 0.0
            }
            
            # Cache-data for backtest-panel
            cache_data = {
                'df_t': df_t,
                'signaler': signaler,
                'rs_rating': rs_rating,
                'exit_data': exit_data,
                'mtf': mtf_data
            }
            
            for strat_navn, strat_key in strategier_å_skanne:
                signal_info = logic.filtrer_signaler_med_kvalitet(
                    df_t, signaler, strat_key,
                    min_kvalitet=min_kvalitet,
                    max_dager_siden=max_dager,
                    min_volum_ratio=min_volum_ratio_filter,
                    regime=current_regime
                )
                
                if signal_info is None:
                    basic_info = logic.finn_siste_signal_info(df_t, signaler, strat_key)
                    if basic_info['dato'] != 'Ingen signal' and basic_info['dager_siden'] <= max_dager:
                        ticker_filtrert += 1
                    continue

                # Regime-tilpasset signalfiltering (6.1)
                if bruk_regime_filter and current_regime:
                    passerer, grunn = logic.sjekk_regime_filter(
                        signal_info, strat_key, mtf_data, rs_rating, current_regime
                    )
                    if not passerer:
                        ticker_filtrert += 1
                        continue
                
                # Konvergens-bonus
                konvergens = logic.finn_kombinerte_signaler(df_t, signaler)
                konvergens_bonus = konvergens['konvergens_score'] if konvergens['er_sterk_konvergens'] else 0
                mtf_justering = mtf_data.get('score_justering', 0)
                total_score = min(100, max(0, signal_info['kvalitet_score'] + konvergens_bonus + mtf_justering))
                
                # Exit-signal straff: aktive exit-signaler reduserer score
                if exit_data['antall_signaler'] >= 2:
                    total_score = max(0, total_score - 20)
                elif exit_data['antall_signaler'] == 1:
                    total_score = max(0, total_score - 8)
                
                # False breakout-deteksjon: pris < signalets close OG under SMA10
                false_breakout = False
                try:
                    signal_dato_str = signal_info['dato']
                    signal_dato_dt = pd.Timestamp(signal_dato_str)
                    if signal_dato_dt in df_t.index:
                        signal_close = df_t.loc[signal_dato_dt, 'Close']
                    else:
                        signal_close = df_t['Close'].iloc[-1]
                    nåværende_pris = df_t['Close'].iloc[-1]
                    sma10 = df_t['Close'].rolling(10).mean().iloc[-1]
                    if nåværende_pris < signal_close and nåværende_pris < sma10:
                        false_breakout = True
                        total_score = max(0, total_score - 15)
                except Exception:
                    pass
                
                # Filtrer bort false breakouts hvis toggle er aktivert
                if false_breakout and filtrer_false_breakout:
                    ticker_filtrert += 1
                    continue
                
                # Støtte/motstand
                rr_data = logic.beregn_smart_risk_reward(df_t)
                stotte_str = "-"
                motstand_str = "-"
                rr_ratio_val = 0.0
                pris_val = df_t['Close'].iloc[-1]
                
                if rr_data:
                    rr_ratio_val = round(rr_data.get('rr_ratio', 0.0), 1)
                    if rr_data.get('stotte_nivåer'):
                        nærmeste_stotte = [s['pris'] for s in rr_data['stotte_nivåer'] if s['pris'] < pris_val]
                        if nærmeste_stotte:
                            stotte_str = f"{max(nærmeste_stotte):.1f}"
                    if rr_data.get('motstand_nivåer'):
                        nærmeste_motstand = [m['pris'] for m in rr_data['motstand_nivåer'] if m['pris'] > pris_val]
                        if nærmeste_motstand:
                            motstand_str = f"{min(nærmeste_motstand):.1f}"
                
                # R:R-filter (3.3): Filtrer bort aksjer med for lav R:R
                if min_rr_ratio > 0 and rr_ratio_val < min_rr_ratio:
                    ticker_filtrert += 1
                    continue
                
                pris_str = f"{pris_val:.2f}" if pris_val < 100 else f"{pris_val:.1f}" if pris_val < 1000 else f"{pris_val:.0f}"
                utv_val = round(signal_info['utvikling_pst'], 1)
                peak_val = round(signal_info.get('peak_utvikling_pst', utv_val), 1)
                
                # Sektor-momentum
                sek_rs = 50
                if sektor_momentum:
                    sek_rs = sektor_momentum['ticker_sektor_rs'].get(ticker, 50)
                    # Koblet sektor-RS: bonus kun hvis individuell RS også er sterk
                    if sektor_momentum['topp_sektorer'] and logic.hent_sektor(ticker) in sektor_momentum['topp_sektorer']:
                        if rs_rating >= 60:
                            total_score = min(100, total_score + 10)
                        elif rs_rating >= 40:
                            total_score = min(100, total_score + 5)
                        # RS < 40 i topp-sektor → ingen bonus (svak aksje i sterk sektor)
                    elif sektor_momentum['bunn_sektorer'] and logic.hent_sektor(ticker) in sektor_momentum['bunn_sektorer']:
                        if rs_rating < 40:
                            total_score = max(0, total_score - 15)  # Svak aksje i svak sektor → sterkere straff
                        elif rs_rating < 60:
                            total_score = max(0, total_score - 5)
                        # RS >= 60 i bunn-sektor → ingen straff (sterk aksje trosser sektor)
                
                # Makro-justering
                makro_adj = logic.makro_score_justering(ticker, olje_signal, valuta_signal)
                total_score = max(0, min(100, total_score + makro_adj))
                
                # Fundamental score
                fund_score = None
                if vis_fundamental:
                    try:
                        fund_data = fundamental_data.hent_fundamental_data(ticker)
                        if fund_data:
                            fund_score, _ = fundamental_data.beregn_fundamental_score(fund_data)
                            fund_score = round(fund_score)
                            if fund_score > 70:
                                total_score = min(100, total_score + 10)
                            elif fund_score < 30:
                                total_score = max(0, total_score - 10)
                    except Exception:
                        pass
                
                # Smart Money Flow (daglig proxy — rask)
                sm_info = None
                if bruk_smart_money and smart_money_tilgjengelig:
                    try:
                        sm_info = smart_money.beregn_smi_for_scanner(df_t)
                        total_score = max(0, min(100, total_score + sm_info['score_justering']))
                    except Exception:
                        sm_info = None
                
                # Backtest-validert strategi-vekting (2.10)
                bt_bonus = strategi_bt_bonus.get(strat_key, 0)
                if bt_bonus != 0:
                    total_score = max(0, min(100, total_score + bt_bonus))
                
                _row = {
                    "Ticker": ticker,
                    "Sektor": logic.hent_sektor(ticker),
                    "Strategi": strat_navn if valgt_navn == "Alle strategier" else "-",
                    "Kvalitet": f"{signal_info['kvalitet_emoji']} {signal_info['kvalitet_klasse']}",
                    "Score": total_score,
                    "RS": rs_rating,
                    "Sek.RS": sek_rs,
                    "MTF": mtf_data.get('emoji', '⚠️'),
                    "R:R": rr_ratio_val,
                    "Signal": signal_info['dato'],
                    "Dager": signal_info['dager_siden'],
                    "Utv%": utv_val,
                    "Peak%": peak_val,
                    "Pris": pris_str,
                    "Støtte": stotte_str,
                    "Motstand": motstand_str,
                    "Exit⚠️": "⚠️" if exit_data['skal_selge'] else "",
                    "FB": "❌" if false_breakout else "",
                    "Konv.": konvergens['antall']
                }
                if vis_fundamental:
                    _row["Fund."] = fund_score if fund_score is not None else ""
                if bruk_smart_money:
                    _row["SM"] = sm_info['emoji'] if sm_info else '⚪'
                if bruk_insider and insider_tilgjengelig:
                    try:
                        _row["Ins."] = insider_monitor.beregn_insider_for_scanner(ticker, insider_handler)
                    except Exception:
                        _row["Ins."] = "—"
                
                # Mønstergjenkjenning (3.4)
                if mønster_filter != "Ingen mønsterfilter" and pattern_tilgjengelig:
                    try:
                        pat_result = pattern_logic.skann_for_scanner(df_t, ticker)
                        if pat_result['antall'] > 0:
                            _row["Mønster"] = f"{pat_result['emoji']} {pat_result['tekst']}"
                        else:
                            _row["Mønster"] = ""
                    except Exception:
                        _row["Mønster"] = ""
                
                ticker_resultater.append(_row)
            
            return ticker_resultater, ticker_filtrert, ticker, cache_data
        
        def _analyse_mønster_ticker(ticker):
            """Analyserer mønstre for én ticker (mønster-modus). Thread-safe."""
            df_t = ticker_dataframes[ticker]
            
            try:
                # Mønstergjenkjenning
                mønstre = pattern_logic.skann_alle_mønstre(df_t)
                if not mønstre:
                    return [], 0, ticker, {}
                
                # RS-beregning
                rs_data = logic.beregn_relativ_styrke(df_t, df_benchmark, alle_tickers_avkastning=alle_rs_scores)
                rs_rating = rs_data['rs_rating'] if rs_data else 50
                
                # Exit-signaler
                exit_data = logic.sjekk_exit_signaler(df_t) if vis_exit_signaler else {
                    'skal_selge': False, 'grunner': [], 'antall_signaler': 0, 'drawdown_pct': 0.0
                }
                
                ticker_resultater = []
                siste_dato = df_t.index[-1]
                
                for mønster in mønstre:
                    pris_val = df_t['Close'].iloc[-1]
                    pris_str = f"{pris_val:.2f}" if pris_val < 100 else f"{pris_val:.1f}" if pris_val < 1000 else f"{pris_val:.0f}"
                    
                    # Dato og dager siden mønster fullført
                    mønster_dato = mønster.get('dato')
                    if mønster_dato:
                        dato_str = mønster_dato.strftime('%d.%m.%Y')
                        dager_siden = (siste_dato - mønster_dato).days
                    else:
                        dato_str = "-"
                        dager_siden = 0
                    
                    # Filtrer på maks dager (bruk samme max_dager som signal-modus)
                    if dager_siden > max_dager:
                        continue
                    
                    # Beregn kursutvikling siden mønster fullført
                    utv_val = 0.0
                    peak_val = 0.0
                    if mønster_dato and mønster_dato in df_t.index:
                        signal_pris = df_t.loc[mønster_dato, 'Close']
                        if signal_pris > 0:
                            utv_val = round((pris_val - signal_pris) / signal_pris * 100, 1)
                            # Peak siden signal
                            fremtidig_data = df_t.loc[mønster_dato:]
                            if len(fremtidig_data) > 0:
                                peak_pris = fremtidig_data['High'].max()
                                peak_val = round((peak_pris - signal_pris) / signal_pris * 100, 1)
                    
                    # Støtte/motstand
                    rr_data = logic.beregn_smart_risk_reward(df_t)
                    rr_ratio_val = round(rr_data.get('rr_ratio', 0.0), 1) if rr_data else 0.0
                    stotte_str = "-"
                    motstand_str = "-"
                    if rr_data:
                        if rr_data.get('stotte_nivåer'):
                            nærmeste_stotte = [s['pris'] for s in rr_data['stotte_nivåer'] if s['pris'] < pris_val]
                            if nærmeste_stotte:
                                stotte_str = f"{max(nærmeste_stotte):.1f}"
                        if rr_data.get('motstand_nivåer'):
                            nærmeste_motstand = [m['pris'] for m in rr_data['motstand_nivåer'] if m['pris'] > pris_val]
                            if nærmeste_motstand:
                                motstand_str = f"{min(nærmeste_motstand):.1f}"
                    
                    # Mønster-emoji basert på retning
                    retning = mønster.get('retning', 'bullish')
                    if retning == 'bullish':
                        emoji = '🟢'
                    elif retning == 'bearish':
                        emoji = '🔴'
                    else:
                        emoji = '🔵'
                    
                    # Filtrer basert på mønstertype
                    if bruk_mønster_modus:
                        if mønster_type_filter == "Kun bullish" and retning != 'bullish':
                            continue
                        elif mønster_type_filter == "Kun bearish" and retning != 'bearish':
                            continue
                    
                    # Beregn score basert på mønster-styrke
                    styrke = mønster.get('styrke', 50)
                    # Konverter styrke (0-100) til kvalitetsklasse
                    if styrke >= 80:
                        kvalitet_str = "🟢 A"
                    elif styrke >= 60:
                        kvalitet_str = "🟡 B"
                    elif styrke >= 40:
                        kvalitet_str = "🟠 C"
                    else:
                        kvalitet_str = "🔴 D"
                    
                    _row = {
                        "Ticker": ticker,
                        "Sektor": logic.hent_sektor(ticker),
                        "Mønster": f"{emoji} {mønster['mønster']}",
                        "Kvalitet": kvalitet_str,
                        "Score": int(styrke),
                        "RS": rs_rating,
                        "Dato": dato_str,
                        "Dager": dager_siden,
                        "Utv%": utv_val,
                        "Peak%": peak_val,
                        "R:R": rr_ratio_val,
                        "Pris": pris_str,
                        "Støtte": stotte_str,
                        "Motstand": motstand_str,
                        "Exit⚠️": "⚠️" if exit_data['skal_selge'] else "",
                    }
                    ticker_resultater.append(_row)
                
                return ticker_resultater, 0, ticker, {}
            except Exception:
                return [], 0, ticker, {}
        
        # Kjør parallelt med ThreadPoolExecutor
        t_start = time.time()
        scan_ticker_cache = {}
        
        # Velg analysefunksjon basert på modus
        analyse_fn = _analyse_mønster_ticker if bruk_mønster_modus else _analyse_en_ticker
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(analyse_fn, ticker): ticker
                for ticker in ticker_dataframes.keys()
            }
            
            ferdig = 0
            total_tickers = len(futures)
            for future in as_completed(futures):
                ferdig += 1
                pb.progress(
                    ferdig / total_tickers,
                    text=f"Analyserer tickers... ({ferdig}/{total_tickers})"
                )
                try:
                    ticker_res, ticker_filt, ticker, cache_data = future.result()
                    resultater.extend(ticker_res)
                    filtrert_bort += ticker_filt
                    scan_ticker_cache[ticker] = cache_data
                except Exception:
                    pass  # Ignorer feil på enkelt-tickers
        
        t_elapsed = time.time() - t_start
        pb.empty()
        st.caption(f"Skanning fullført på {t_elapsed:.1f}s ({len(ticker_dataframes)} tickers, 8 tråder)")


        # === NY: Sektor-filtrering ===
        if resultater:
            resultater = logic.filtrer_sektor_konsentrasjon(resultater, maks_per_sektor)
        
        # === NY: Mønster-filtrering (3.4) ===
        if resultater and mønster_filter != "Ingen mønsterfilter" and pattern_tilgjengelig:
            if mønster_filter == "Kun bullish mønstre":
                resultater = [r for r in resultater if r.get('Mønster', '') and ('🔵' in r.get('Mønster', '') or '🟢' in r.get('Mønster', ''))]
            elif mønster_filter == "Kun bearish mønstre":
                resultater = [r for r in resultater if r.get('Mønster', '') and '🔴' in r.get('Mønster', '')]
            elif mønster_filter == "Alle mønstre":
                resultater = [r for r in resultater if r.get('Mønster', '')]
        
        # Lagre resultater i session_state
        if resultater:
            st.session_state['scanner_resultater'] = resultater
            st.session_state['scanner_alle_strat'] = (valgt_navn == "Alle strategier")
            st.session_state['scanner_mønster_modus'] = bruk_mønster_modus
            st.session_state['scanner_filtrert_bort'] = filtrert_bort
            st.session_state['scanner_sorter_rr'] = sorter_rr
            st.session_state['scanner_sektor_momentum'] = sektor_momentum
            st.session_state['scanner_olje_signal'] = olje_signal
            st.session_state['scanner_valuta_signal'] = valuta_signal
            st.session_state['scanner_regime_info'] = {
                'active': bruk_regime_filter,
                'regime': current_regime,
                'krav': regime_krav
            } if bruk_regime_filter else None
        else:
            st.session_state['scanner_resultater'] = []
            st.session_state['scanner_filtrert_bort'] = filtrert_bort
            st.session_state['scanner_regime_info'] = {
                'active': bruk_regime_filter,
                'regime': current_regime,
                'krav': regime_krav
            } if bruk_regime_filter else None
    
    # Vis resultater (utenfor button-blokken så de vises ved sorteringsendring)
    _scanner_res = st.session_state.get('scanner_resultater')
    _har_resultater = _scanner_res is not None and (
        (isinstance(_scanner_res, pd.DataFrame) and not _scanner_res.empty) or
        (isinstance(_scanner_res, list) and len(_scanner_res) > 0)
    )
    if _har_resultater:
        resultater = st.session_state['scanner_resultater']
        filtrert_bort = st.session_state.get('scanner_filtrert_bort', 0)
        
        # === MAKRO-STATUS BAR (Olje + Valuta) ===
        _olje = st.session_state.get('scanner_olje_signal', {})
        _valuta = st.session_state.get('scanner_valuta_signal', {})
        _makro_items = []
        if _olje.get('pris'):
            _o_col = "#00c853" if _olje['signal'] == 'bullish' else "#ef5350" if _olje['signal'] == 'bearish' else "#ffa726"
            _makro_items.append(
                f'<span style="margin-right:18px;">{_olje["emoji"]} Brent: '
                f'<b style="color:{_o_col}">${_olje["pris"]}</b> '
                f'(SMA50: ${_olje["sma50"]}, 20d: {_olje["endring_20d"]:+.1f}%)</span>'
            )
        if _valuta.get('kurs'):
            _v_trend = "Svak NOK" if _valuta['trend'] == 'svak_nok' else "Sterk NOK" if _valuta['trend'] == 'sterk_nok' else "Nøytral"
            _v_col = "#ffa726" if _valuta['trend'] == 'svak_nok' else "#42a5f5" if _valuta['trend'] == 'sterk_nok' else "#888"
            _makro_items.append(
                f'<span>{_valuta["emoji"]} USD/NOK: '
                f'<b style="color:{_v_col}">{_valuta["kurs"]}</b> '
                f'({_v_trend}, 20d: {_valuta["endring_20d"]:+.1f}%)</span>'
            )
        if _makro_items:
            st.markdown(
                f'<div style="margin:10px 0 5px 0;padding:8px 14px;border-radius:10px;'
                f'background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);font-size:13px;">'
                f'{"".join(_makro_items)}</div>',
                unsafe_allow_html=True
            )
        
        # === SEKTOR-HEATMAP ===
        _sek_mom = st.session_state.get('scanner_sektor_momentum')
        if _sek_mom and _sek_mom.get('sektor_data'):
            _sek_data = _sek_mom['sektor_data']
            _sortert = sorted(_sek_data.items(), key=lambda x: x[1]['avkastning'], reverse=True)
            _badges = []
            
            # Sikker hex til RGB konvertering
            def hex_til_rgb(hex_color):
                if not hex_color or len(hex_color) != 7 or not hex_color.startswith('#'):
                    return "136, 136, 136"  # Fallback grå
                try:
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    return f"{r}, {g}, {b}"
                except (ValueError, IndexError):
                    return "136, 136, 136"  # Fallback grå
                    
            for _sek_navn, _sek_info in _sortert:
                _avk = _sek_info['avkastning']
                if _avk > 2:
                    _col = "#00c853"
                elif _avk > 0:
                    _col = "#66bb6a"
                elif _avk > -2:
                    _col = "#ffa726"
                else:
                    _col = "#ef5350"
                
                rgb_color = hex_til_rgb(_col)
                _badges.append(
                    f'<span style="display:inline-block;padding:4px 10px;margin:3px;border-radius:8px;'
                    f'background:rgba({rgb_color},0.25);'
                    f'border:1px solid {_col};font-size:13px;">'
                    f'{_sek_navn} <b style="color:{_col}">{_avk:+.1f}%</b></span>'
                )
            st.markdown(
                f'<div style="margin:10px 0 15px 0;"><span style="font-size:13px;color:rgba(255,255,255,0.5);">'
                f'Sektor 20d momentum:</span> {"".join(_badges)}</div>',
                unsafe_allow_html=True
            )
        
        res_df = pd.DataFrame(resultater)
        
        # === LIVE PRISER FRA YAHOO FINANCE ===
        if len(res_df) > 0 and 'Ticker' in res_df.columns:
            result_tickers = res_df['Ticker'].unique().tolist()
            try:
                with st.spinner("Henter live-kurser..."):
                    live_priser = _hent_live_priser(tuple(result_tickers))
                
                if live_priser:
                    # Beregn dagendring: (live - cached_close) / cached_close * 100
                    siste_col = []
                    dag_col = []
                    for _, row in res_df.iterrows():
                        t = row['Ticker']
                        live_p = live_priser.get(t)
                        if live_p is not None:
                            # Formater live pris
                            if live_p < 100:
                                siste_str = f"{live_p:.2f}"
                            elif live_p < 1000:
                                siste_str = f"{live_p:.1f}"
                            else:
                                siste_str = f"{live_p:.0f}"
                            siste_col.append(siste_str)
                            
                            # Beregn % endring fra cached close (Pris-kolonnen)
                            try:
                                cached_p = float(str(row['Pris']).replace(',', '.'))
                                if cached_p > 0:
                                    dag_pct = (live_p - cached_p) / cached_p * 100
                                    dag_col.append(round(dag_pct, 2))
                                else:
                                    dag_col.append(0.0)
                            except (ValueError, TypeError):
                                dag_col.append(0.0)
                        else:
                            siste_col.append(row.get('Pris', '-'))
                            dag_col.append(0.0)
                    
                    # Sett inn kolonner rett etter Pris
                    pris_idx = res_df.columns.get_loc('Pris') + 1
                    res_df.insert(pris_idx, 'Siste', siste_col)
                    res_df.insert(pris_idx + 1, 'Dag%', dag_col)
                    
                    _scanner_logger.info(f"Live priser hentet for {len(live_priser)}/{len(result_tickers)} tickers")
            except Exception as e:
                _scanner_logger.warning(f"Kunne ikke hente live priser: {e}")
        
        # Mønster-modus: vis Mønster i stedet for Strategi
        er_mønster_modus = st.session_state.get('scanner_mønster_modus', False)
        
        if er_mønster_modus:
            # Mønster-modus: fjern Strategi hvis den finnes
            if 'Strategi' in res_df.columns:
                res_df = res_df.drop(columns=['Strategi'])
        else:
            # Signal-modus: fjern Strategi hvis enkelt-strategi valgt
            if not st.session_state.get('scanner_alle_strat', True):
                if 'Strategi' in res_df.columns:
                    res_df = res_df.drop(columns=['Strategi'])
            # Fjern Mønster-kolonnen i signal-modus (hvis den finnes men er tom)
            if 'Mønster' in res_df.columns and res_df['Mønster'].str.strip().eq('').all():
                res_df = res_df.drop(columns=['Mønster'])
        
        # === MODERNE RESULTAT-HEADER ===
        # I mønster-modus har vi ikke Utv%, så sjekk først
        if 'Utv%' in res_df.columns:
            positive = res_df[res_df['Utv%'] > 0]
            positive_pct = len(positive)/len(res_df)*100 if len(res_df) > 0 else 0
        else:
            positive_pct = 0  # Mønster-modus har ikke Utv%
        exit_count = len(res_df[res_df['Exit⚠️'] == '⚠️']) if 'Exit⚠️' in res_df.columns else 0
        snitt_rs = res_df['RS'].mean() if len(res_df) > 0 else 0
        
        # Beregn fordeling (strategi eller mønster)
        if er_mønster_modus and 'Mønster' in res_df.columns:
            mønster_fordeling = res_df['Mønster'].value_counts().to_dict()
            strategi_fordeling = {}
        else:
            strategi_fordeling = {}
            mønster_fordeling = {}
            if st.session_state.get('scanner_alle_strat', True) and 'Strategi' in res_df.columns:
                strategi_fordeling = res_df['Strategi'].value_counts().to_dict()
        
        # Header-label basert på modus
        resultat_label = "Mønstre" if er_mønster_modus else "Signaler"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0, 200, 83, 0.15) 0%, rgba(0, 230, 118, 0.1) 100%); 
                    border-radius: 20px; padding: 24px; margin: 20px 0; 
                    border: 1px solid rgba(0, 200, 83, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div style="text-align: center; flex: 1; min-width: 100px;">
                    <div style="font-size: 36px; font-weight: 700; color: #00c853;">{len(res_df)}</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.6); text-transform: uppercase;">{resultat_label}</div>
                </div>
                <div style="text-align: center; flex: 1; min-width: 100px;">
                    <div style="font-size: 36px; font-weight: 700; color: #667eea;">{positive_pct:.0f}%</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.6); text-transform: uppercase;">Positive</div>
                </div>
                <div style="text-align: center; flex: 1; min-width: 100px;">
                    <div style="font-size: 36px; font-weight: 700; color: #ffd700;">{snitt_rs:.0f}</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.6); text-transform: uppercase;">Snitt RS</div>
                </div>
                <div style="text-align: center; flex: 1; min-width: 100px;">
                    <div style="font-size: 36px; font-weight: 700; color: {'#ff5252' if exit_count > 0 else '#4caf50'};">{exit_count}</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.6); text-transform: uppercase;">Exit-varsler</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Regime-filter oppsummering
        _regime_info = st.session_state.get('scanner_regime_info')
        if _regime_info and _regime_info.get('active') and _regime_info.get('krav'):
            _rk = _regime_info['krav']
            _rn = _regime_info['regime'] or 'Ukjent'
            st.markdown(f"""
            <div style="margin: -8px 0 12px; padding: 10px 16px; border-radius: 10px;
                        background: rgba(102,126,234,0.12); border: 1px solid rgba(102,126,234,0.3);
                        font-size: 13px; display: flex; align-items: center; gap: 12px;">
                <span><b>Regime-filter aktivt</b></span>
                <span style="opacity: 0.8;">|</span>
                <span>{_rk['beskrivelse']}</span>
                <span style="opacity: 0.8;">|</span>
                <span>Filtrert bort: <b style="color: #ffa726;">{filtrert_bort}</b></span>
            </div>
            """, unsafe_allow_html=True)
        
        # === STRATEGI-SAMMENLIGNING (hvis alle strategier) ===
        if st.session_state.get('scanner_alle_strat', True) and strategi_fordeling:
            st.markdown("##### Strategi-sammenligning")
            
            # Beregn statistikk per strategi
            strat_stats = []
            for strat_navn in strategi_fordeling.keys():
                strat_df = res_df[res_df['Strategi'] == strat_navn]
                if len(strat_df) > 0:
                    avg_rs = strat_df['RS'].mean()
                    avg_utv = strat_df['Utv%'].mean()
                    pos_pct = len(strat_df[strat_df['Utv%'] > 0]) / len(strat_df) * 100
                    meta = strategi_metadata.get(strat_navn, {})
                    strat_stats.append({
                        'navn': strat_navn,
                        'antall': len(strat_df),
                        'avg_rs': avg_rs,
                        'avg_utv': avg_utv,
                        'pos_pct': pos_pct,
                        'horisont': meta.get('horisont', 'Medium'),
                        'risiko': meta.get('risiko', 'Medium')
                    })
            
            # Sorter etter antall signaler
            strat_stats.sort(key=lambda x: x['antall'], reverse=True)
            
            # Vis som kompakte kort
            strat_cols = st.columns(min(len(strat_stats), 4))
            for i, stat in enumerate(strat_stats[:4]):
                with strat_cols[i]:
                    # Fargekode basert på ytelse
                    color = '#00c853' if stat['avg_utv'] > 0 else '#ff5252'
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 16px; 
                                border: 1px solid rgba(255,255,255,0.1); text-align: center;">
                        <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px;">{stat['navn'].split('(')[0].strip()}</div>
                        <div style="font-size: 28px; font-weight: 700; color: {color};">{stat['antall']}</div>
                        <div style="font-size: 11px; color: rgba(255,255,255,0.5);">signaler</div>
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);">
                            <span style="font-size: 11px; color: rgba(255,255,255,0.6);">
                                RS {stat['avg_rs']:.0f} • {stat['pos_pct']:.0f}% ↑
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Vis flere strategier hvis det er mer enn 4
            if len(strat_stats) > 4:
                with st.expander(f"Se alle {len(strat_stats)} strategier"):
                    for stat in strat_stats[4:]:
                        st.markdown(f"**{stat['navn']}**: {stat['antall']} signaler (RS: {stat['avg_rs']:.0f}, {stat['pos_pct']:.0f}% positive)")
        
        # === SEKTOR-FORDELING (kompakt) ===
        if 'Sektor' in res_df.columns:
            sektor_counts = res_df['Sektor'].value_counts()
            if len(sektor_counts) > 0:
                with st.expander("Sektor-fordeling", expanded=False):
                    sektor_cols = st.columns(min(len(sektor_counts), 5))
                    for i, (sektor, count) in enumerate(sektor_counts.head(5).items()):
                        with sektor_cols[i]:
                            st.metric(sektor[:15], count)
        
        # === VARSLER (mer kompakt) ===
        if exit_count > 0:
            varsel_cols = st.columns(2)
            
            # Exit-varsler
            with varsel_cols[0]:
                    exit_aksjer = res_df[res_df['Exit⚠️'] == '⚠️']['Ticker'].tolist()
                    st.markdown(f"""
                    <div style="background: rgba(255, 82, 82, 0.1); border-radius: 12px; padding: 16px; 
                                border: 1px solid rgba(255, 82, 82, 0.3);">
                        <div style="font-weight: 600; margin-bottom: 8px;">Exit-varsler</div>
                        <div style="font-size: 13px; color: rgba(255,255,255,0.7);">
                            {exit_count} aksjer med kjøp + salg-signal
                        </div>
                        <div style="font-size: 11px; color: rgba(255,255,255,0.5); margin-top: 4px;">
                            {', '.join(exit_aksjer[:5])}{'...' if len(exit_aksjer) > 5 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === KOMPAKT TABELL-INNSTILLINGER ===
        with st.expander("🔧 Tabell-innstillinger", expanded=False):
            tc1, tc2 = st.columns(2)
            with tc1:
                _sort_options = ["Signal (dato)", "Score", "RS", "Utvikling", "R:R", "Dag%"]
                sorter_etter = st.selectbox("Sorter etter", _sort_options, index=0, key="scanner_sort_by")
                retning_options = ["Nyeste først", "Eldste først"] if sorter_etter == "Signal (dato)" else ["Høyest først", "Lavest først"]
                sorter_retning = st.selectbox("Retning", retning_options, index=0, key="scanner_sort_dir")
            with tc2:
                st.markdown("**Vis kolonner:**")
                vis_sektor = st.checkbox("Sektor", value=True, key="vis_sektor")
                vis_stotte = st.checkbox("Støtte/Motstand", value=False, key="vis_stotte")
                vis_exit = st.checkbox("Exit-varsler", value=True, key="vis_exit")
                vis_konv = st.checkbox("Konvergens", value=False, key="vis_konv")
                if bruk_insider and insider_tilgjengelig:
                    vis_insider = st.checkbox("Insider", value=True, key="vis_insider")
                else:
                    vis_insider = True
        
        # Default for kolonner som ikke er i expander
        vis_motstand = vis_stotte  # Kobler motstand til støtte
        
        # === KOLONNE-FORKLARING (sammenfoldet) ===
        with st.expander("ℹ️ Kolonne-forklaringer", expanded=False):
            st.markdown("""
            | Kolonne | Beskrivelse |
            |---------|-------------|
            | **Pris** | Siste close fra cached data |
            | **Siste** | Live kurs fra Yahoo Finance (oppdateres hvert 5. min) |
            | **Dag%** | Endring fra cached close til live kurs |
            | **RS** | Relativ styrke vs. markedet (1-99, høyere = bedre) |
            | **R:R** | Risk/Reward basert på støtte/motstand |
            | **Utv%** | Kursutvikling siden signalet |
            | **Peak%** | Høyeste kurs etter signal |
            | **Exit⚠️** | Potensielle salgssignaler |
            | **FB** | False Breakout (pris under signal + SMA10) |
            """)
        
        # Anvend sortering (håndter mønster-modus som mangler noen kolonner)
        if sorter_etter == "Signal (dato)" and 'Signal' in res_df.columns:
            res_df['Signal_dato'] = pd.to_datetime(res_df['Signal'], format='%d.%m.%Y', errors='coerce')
            res_df = res_df.sort_values(by='Signal_dato', ascending=(sorter_retning == "Eldste først"))
            res_df = res_df.drop(columns=['Signal_dato'])
        elif sorter_etter == "Score" and 'Score' in res_df.columns:
            res_df = res_df.sort_values(by='Score', ascending=(sorter_retning == "Lavest først"))
        elif sorter_etter == "RS" and 'RS' in res_df.columns:
            res_df = res_df.sort_values(by='RS', ascending=(sorter_retning == "Lavest først"))
        elif sorter_etter == "Utvikling" and 'Utv%' in res_df.columns:
            res_df = res_df.sort_values(by='Utv%', ascending=(sorter_retning == "Lavest først"))
        elif sorter_etter == "R:R" and 'R:R' in res_df.columns:
            res_df = res_df.sort_values(by='R:R', ascending=(sorter_retning == "Lavest først"))
        elif sorter_etter == "Dag%" and 'Dag%' in res_df.columns:
            res_df = res_df.sort_values(by='Dag%', ascending=(sorter_retning == "Lavest først"))
        else:
            # Default: sorter etter Signal dato, nyeste først
            if 'Signal' in res_df.columns:
                res_df['Signal_dato'] = pd.to_datetime(res_df['Signal'], format='%d.%m.%Y', errors='coerce')
                res_df = res_df.sort_values(by='Signal_dato', ascending=False)
                res_df = res_df.drop(columns=['Signal_dato'])
            elif er_mønster_modus:
                res_df = res_df.sort_values(by='Score', ascending=False)
        
        # Fjern kolonner basert på checkboxer
        kolonner_å_fjerne = []
        if not vis_stotte:
            kolonner_å_fjerne.append('Støtte')
        if not vis_motstand:
            kolonner_å_fjerne.append('Motstand')
        if not vis_sektor:
            kolonner_å_fjerne.append('Sektor')
        if not vis_konv:
            kolonner_å_fjerne.append('Konv.')
        if not vis_exit:
            kolonner_å_fjerne.append('Exit⚠️')
        if not vis_insider:
            kolonner_å_fjerne.append('Ins.')
        
        for kol in kolonner_å_fjerne:
            if kol in res_df.columns:
                res_df = res_df.drop(columns=[kol])
        
        # Fargekodet tabell med importerte highlight-funksjoner
        styled_df = res_df.style.map(highlight_kvalitet, subset=['Kvalitet'])
        styled_df = styled_df.map(highlight_rs, subset=['RS'])
        if 'Exit⚠️' in res_df.columns:
            styled_df = styled_df.map(highlight_exit, subset=['Exit⚠️'])
        if 'FB' in res_df.columns:
            styled_df = styled_df.map(highlight_false_breakout, subset=['FB'])
        styled_df = styled_df.map(highlight_utvikling, subset=['Utv%'])
        if 'Peak%' in res_df.columns:
            styled_df = styled_df.map(highlight_utvikling, subset=['Peak%'])
        if 'Dag%' in res_df.columns:
            styled_df = styled_df.map(_highlight_dagendring, subset=['Dag%'])
        if 'R:R' in res_df.columns:
            styled_df = styled_df.map(highlight_rr, subset=['R:R'])
        # Formater Utv% og Peak% med 1 desimal
        fmt = {'Utv%': '{:.1f}'}
        if 'Peak%' in res_df.columns:
            fmt['Peak%'] = '{:.1f}'
        if 'R:R' in res_df.columns:
            fmt['R:R'] = '{:.1f}'
        if 'Dag%' in res_df.columns:
            fmt['Dag%'] = '{:+.2f}'
        styled_df = styled_df.format(fmt)
        
        res_df_med_star = res_df.copy()
        
        # Funksjon for å style DataFrame
        def style_scanner_df(df_to_style):
            styled = df_to_style.style.map(highlight_kvalitet, subset=['Kvalitet'])
            styled = styled.map(highlight_rs, subset=['RS'])
            if 'Exit⚠️' in df_to_style.columns:
                styled = styled.map(highlight_exit, subset=['Exit⚠️'])
            if 'FB' in df_to_style.columns:
                styled = styled.map(highlight_false_breakout, subset=['FB'])
            if 'Utv%' in df_to_style.columns:
                styled = styled.map(highlight_utvikling, subset=['Utv%'])
            if 'Peak%' in df_to_style.columns:
                styled = styled.map(highlight_utvikling, subset=['Peak%'])
            if 'Dag%' in df_to_style.columns:
                styled = styled.map(_highlight_dagendring, subset=['Dag%'])
            if 'R:R' in df_to_style.columns:
                styled = styled.map(highlight_rr, subset=['R:R'])
            fmt = {}
            if 'Utv%' in df_to_style.columns:
                fmt['Utv%'] = '{:.1f}'
            if 'Peak%' in df_to_style.columns:
                fmt['Peak%'] = '{:.1f}'
            if 'Dag%' in df_to_style.columns:
                fmt['Dag%'] = '{:+.2f}'
            if 'R:R' in df_to_style.columns:
                fmt['R:R'] = '{:.1f}'
            if fmt:
                styled = styled.format(fmt)
            return styled
        
        # === RESULTATVISNING ===
        st.markdown(f"**{len(res_df_med_star)} treff**")
        st.caption("Klikk på en rad for å åpne chart")
        event = st.dataframe(
            style_scanner_df(res_df_med_star),
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="scanner_alle"
        )
        
        selected = event.selection.rows if event.selection else []
        if selected and not st.session_state.get('_popup_manually_closed', False):
            valgt_idx = selected[0]
            row = res_df_med_star.iloc[valgt_idx]
            st.session_state['_scanner_popup'] = {
                'ticker': row['Ticker'],
                'row': row.to_dict(),
            }
        
        # === HISTORISK TREFFSIKKERHET ===
        with st.expander("📊 Historisk treffsikkerhet", expanded=False):
            st.caption("Backtest med trailing stop (-1×ATR) + profit target (+2×ATR), maks 20 dager")
            
            backtest_stats = {}
            for strat_navn, strat_key in navn_til_nøkkel.items():
                alle_resultater = []
                for ticker in res_df['Ticker'].unique()[:20]:  # Begrens for hastighet
                    try:
                        df_t = df_clean[df_clean['Ticker'] == ticker]
                        if len(df_t) < 100:
                            continue
                        if ticker not in st.session_state['teknisk_cache']:
                            st.session_state['teknisk_cache'][ticker] = cached_beregn_tekniske_indikatorer(df_t)
                        df_t = st.session_state['teknisk_cache'][ticker]
                        signaler_t = hent_signaler_cached(ticker, df_t)

                        bt = hent_backtest_cached(ticker, df_t, signaler_t, strat_key, holdingperiode=20)
                        if bt:
                            alle_resultater.append(bt)
                    except:
                        continue

                if alle_resultater:
                    snitt_wr = sum(r['win_rate'] for r in alle_resultater) / len(alle_resultater)
                    snitt_avk = sum(r['snitt_avkastning'] for r in alle_resultater) / len(alle_resultater)
                    snitt_dager = sum(r.get('snitt_dager_holdt', 20) for r in alle_resultater) / len(alle_resultater)
                    meta = strategi_metadata.get(strat_navn, {})
                    backtest_stats[strat_navn] = {
                        'emoji': meta.get('emoji', ''),
                        'win_rate': snitt_wr,
                        'snitt_avk': snitt_avk,
                        'snitt_dager': snitt_dager,
                        'signaler': sum(r['antall_signaler'] for r in alle_resultater)
                    }
            
            if backtest_stats:
                # Sorter etter win rate
                sorted_stats = sorted(backtest_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
                
                # Vis som visuelle kort
                bt_cols = st.columns(min(len(sorted_stats), 4))
                for i, (strat_navn, stats) in enumerate(sorted_stats[:4]):
                    with bt_cols[i]:
                        wr_color = '#00c853' if stats['win_rate'] >= 60 else '#ffd700' if stats['win_rate'] >= 50 else '#ff5252'
                        avk_color = '#00c853' if stats['snitt_avk'] > 0 else '#ff5252'
                        
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 16px; 
                                    border: 1px solid rgba(255,255,255,0.1); text-align: center;">
                            <div style="font-size: 20px; margin-bottom: 4px;">{stats['emoji']}</div>
                            <div style="font-size: 12px; font-weight: 500; color: rgba(255,255,255,0.7); margin-bottom: 8px;">
                                {strat_navn.split('(')[0].strip()[:12]}
                            </div>
                            <div style="font-size: 28px; font-weight: 700; color: {wr_color};">{stats['win_rate']:.0f}%</div>
                            <div style="font-size: 10px; color: rgba(255,255,255,0.5); text-transform: uppercase;">Win Rate</div>
                            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);">
                                <span style="font-size: 14px; font-weight: 600; color: {avk_color};">
                                    {'+' if stats['snitt_avk'] > 0 else ''}{stats['snitt_avk']:.1f}%
                                </span>
                                <span style="font-size: 10px; color: rgba(255,255,255,0.5);"> snitt</span>
                            </div>
                            <div style="font-size: 10px; color: rgba(255,255,255,0.4); margin-top: 4px;">
                                {stats['signaler']} signaler · ~{stats['snitt_dager']:.0f}d hold
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Vis full tabell for alle strategier
                if len(sorted_stats) > 4:
                    st.markdown("---")
                    st.caption("Alle strategier:")
                    for strat_navn, stats in sorted_stats[4:]:
                        wr_emoji = '🟢' if stats['win_rate'] >= 60 else '🟡' if stats['win_rate'] >= 50 else '🔴'
                        st.markdown(f"{stats['emoji']} **{strat_navn}**: {wr_emoji} {stats['win_rate']:.0f}% win rate, {stats['snitt_avk']:+.1f}% snitt ({stats['signaler']} signaler)")
                
                # Beste strategi-anbefaling
                best_strat = sorted_stats[0]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(0, 200, 83, 0.1) 0%, rgba(0, 230, 118, 0.05) 100%); 
                            border-radius: 12px; padding: 16px; margin-top: 16px; border: 1px solid rgba(0, 200, 83, 0.2);">
                    <div style="font-size: 13px; color: rgba(255,255,255,0.9);">
                        <strong>Beste strategi akkurat nå:</strong> {best_strat[0]} 
                        med {best_strat[1]['win_rate']:.0f}% treffsikkerhet
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Ikke nok data for backtest")
        
        # Forklaring (mer kompakt)
        with st.expander("Forstå de nye kolonnene"):
            st.markdown("""
            **Nye kolonner:**
            
            | Kolonne | Beskrivelse |
            |---------|-------------|
            | **RS** | Relativ Styrke (1-99). Høyere = outperformer markedet |
            | **Exit⚠️** | Varsel hvis aksjen har salgssignaler |
            | **Konv.** | Konvergens - antall strategier som treffer samtidig |
            | **Sektor** | Bransje for diversifisering |
            | **Støtte** | Nærmeste støttenivå under nåværende kurs |
            
            **RS-rating:**
            - **80+**: Sterk outperformance
            - **60-79**: Moderat styrke
            - **40-59**: Noytral
            - **<40**: Svakere enn markedet
            
            **Exit-signaler trigges ved:**
            - Death Cross (SMA 50 < SMA 200)
            - RSI faller fra overkjøpt
            - Brudd under SMA 50 med volum
            - MACD bearish crossover
            - >7% drawdown fra 20d topp
            """)

        # Popup utenfor tabs — @st.dialog må kalles fra hoved-script-kroppen
        popup_data = st.session_state.get('_scanner_popup')
        if popup_data:
            _vis_scanner_popup()

    elif 'scanner_resultater' in st.session_state:
        # Vist kun hvis vi har kjørt skanning men ikke fant noe
        filtrert_bort = st.session_state.get('scanner_filtrert_bort', 0)
        st.warning(f"Ingen signaler funnet.")
        if filtrert_bort > 0:
            regime_info = st.session_state.get('scanner_regime_info')
            if regime_info and regime_info.get('active'):
                st.info(f"{filtrert_bort} signaler filtrert bort. "
                        f"Regime-filter aktivt: **{regime_info['regime']}** — {regime_info['krav']['kort']}")
            else:
                st.info(f"{filtrert_bort} signaler ble filtrert bort.")
