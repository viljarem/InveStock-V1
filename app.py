import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Importer egne moduler
import data
import logic
import config

# Importer regime-modul hvis tilgjengelig
try:
    import regime_model
    regime_tilgjengelig = True
except ImportError:
    regime_tilgjengelig = False

# --- BETA MODUL INTEGRASJON ---
beta_filstien = "beta_ml.py"
beta_eksisterer = os.path.exists(beta_filstien)
beta_tilgjengelig = False
beta_feilmelding = None
python_sti = sys.executable

if beta_eksisterer:
    try:
        import beta_ml
        beta_tilgjengelig = True
    except Exception as e:
        beta_feilmelding = str(e)

# Konfigurasjon av Streamlit
st.set_page_config(
    page_title="InveStock Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Profesjonell styling (CSS)
st.markdown("""
    <style>
    .stDataFrame { border: 1px solid #e6e9ef; border-radius: 5px; }
    div[data-testid="metric-container"] {
        background-color: rgba(151, 166, 195, 0.1);
        border: 1px solid rgba(151, 166, 195, 0.2);
        padding: 15px;
        border-radius: 5px;
    }
    .info-box {
        background-color: rgba(78, 140, 255, 0.1);
        border-left: 5px solid #4e8cff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: inherit;
    }
    .horisont-label {
        font-weight: bold;
        color: #4e8cff;
        text-transform: uppercase;
        font-size: 0.8rem;
        margin-bottom: 5px;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CACHING FOR DATA ---
@st.cache_data(show_spinner=False)
def cached_hent_data():
    return data.hent_data()

@st.cache_data(show_spinner=False)
def cached_hent_markedsdata_df():
    return data.hent_markedsdata_df()

@st.cache_data(show_spinner=False)
def cached_beregn_tekniske_indikatorer(df):
    return logic.beregn_tekniske_indikatorer(df)

# Initialisering av Session State for datahåndtering
if 'df' not in st.session_state:
    st.session_state['df'] = cached_hent_data()
if 'df_market' not in st.session_state:
    st.session_state['df_market'] = cached_hent_markedsdata_df()
if 'teknisk_cache' not in st.session_state:
    st.session_state['teknisk_cache'] = {}

# --- SIDEBAR NAVIGASJON ---
with st.sidebar:
    st.title("InveStock Pro")
    
    # Dynamisk menyvalg basert på tilgjengelighet av Beta-modul
    meny_opsjoner = ["Hjem", "Markedstemperatur", "Scanner", "Teknisk Analyse"]
    if beta_tilgjengelig:
        meny_opsjoner.append("Beta: AI Scanner")
        
    side_valg = st.radio("Meny", meny_opsjoner)
    
    # Utvidet feilsøking for Beta-modul
    if beta_eksisterer and not beta_tilgjengelig:
        st.sidebar.error(f"⚠️ Beta-fil funnet, men kunne ikke lastes:\n\n`{beta_feilmelding}`")
        with st.sidebar.expander("Diagnostisk verktøy"):
            st.write(f"**Python-sti:** `{python_sti}`")
            st.write(f"**Venv aktiv:** {'Ja' if '.venv' in python_sti else 'Nei'}")
            st.write("**Søkestier (sys.path):**")
            st.write(sys.path)
            
            st.markdown("""
            **Anbefalt løsning:**
            1. Avslutt Streamlit (Ctrl+C i terminalen).
            2. Kjør denne kommandoen for å tvinge reinstallasjon:
            `./.venv/bin/python -m pip install --force-reinstall xgboost`
            3. Start på nytt med:
            `python -m streamlit run app.py`
            """)
    
    st.markdown("---")
    if st.button("Oppdater Data"):
        with st.spinner("Synkroniserer markedet..."):
            data.last_ned_data()
            st.session_state['df'] = cached_hent_data()
            st.session_state['df_market'] = cached_hent_markedsdata_df()
        st.success("Data ble oppdatert")
    
    min_volum = st.number_input("Min. dagsomsetning (NOK)", value=500000, step=100000)

# Last og filtrer data
df_raw = st.session_state['df']
if df_raw.empty:
    st.error("Ingen data funnet i lagring. Vennligst kjør oppdatering fra menyen.")
    st.stop()

# Ekskluderer illikvide aksjer basert på brukerens input
df_clean = data.filtrer_likvide_aksjer(df_raw, min_dagsomsetning=min_volum)
unike_tickers = sorted(df_clean['Ticker'].unique())

# --- SIDE: HJEM ---
if side_valg == "Hjem":
    st.title("InveStock Pro - Teknisk Analyse")
    st.markdown(f"Systemet analyserer for øyeblikket {len(unike_tickers)} aksjer fra Oslo Børs.")
    
    st.markdown("""
    ### Oversikt over algoritmer
    | Horisont | Strategi | Teknisk beskrivelse |
    | :--- | :--- | :--- |
    | **Kort** | RSI Mean Reversion | Identifiserer overreaksjoner ned i sterke trender. |
    | **Kort** | Momentum Burst | Kursutbrudd støttet av uvanlig høyt volum. |
    | **Mellomlang** | Ichimoku Breakout | Pris som bryter gjennom Kumo-skyen. |
    | **Lang** | Golden Cross | Det klassiske krysset mellom 50 og 200 dagers snitt. |
    | **Spesial** | Wyckoff Accumulation | Analyse av 'springs' etter falske brudd ned. |
    | **Spesial** | Bull Race Prep | Identifisering av volatilitetssqueeze (Bollinger). |
    | **Spesial** | VCP (Minervini) | Volatilitetssammentrekning i sterk opptrend (Stage 2). |
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📖 Veiledning til InveStock Pro
    
    Denne appen er et verktøy for teknisk analyse av aksjer på Oslo Børs. Her er en oversikt over hva du finner på hver side:
    
    ---
    
    #### 🌡️ Markedstemperatur
    Gir deg et raskt overblikk over markedets generelle tilstand:
    - **Aksjer i Bull-trend**: Prosentandel av aksjer som handles over 200-dagers glidende snitt (SMA 200). Høy verdi indikerer et sterkt marked.
    - **Positive/Negative aksjer**: Antall aksjer som stiger eller faller i dag sammenlignet med forrige dag.
    
    ---
    
    #### 🔍 Scanner
    Søk etter aksjer som oppfyller kriteriene for en spesifikk handelsstrategi:
    1. Velg en strategi fra nedtrekksmenyen
    2. Klikk "Utfør skanning"
    3. Resultatet viser alle aksjer med nylige signaler, sortert etter hvor ferske de er
    
    **Kolonner i resultatet:**
    - **Siste Signal**: Dato for når strategien sist ga et kjøpssignal
    - **Dager siden**: Antall handelsdager siden signalet
    - **Kursutvikling (%)**: Hvordan kursen har utviklet seg siden signalet
    
    ---
    
    #### 📊 Teknisk Analyse
    Detaljert analyse av én enkelt aksje med interaktivt chart:
    
    **Live Info-boks (øverst):**
    - Siste kurs og daglig endring i kroner og prosent
    - Dagens høyeste og laveste kurs
    - Handelsvolum og forrige dags sluttkurs
    
    **Graf-innstillinger:**
    - **Graftype**: Velg mellom Candlestick, OHLC, Linje eller Area
    - **Tema**: Mørk eller lys bakgrunn
    
    **Indikatorer:**
    - **Trendlinjer**: SMA (10, 20, 50, 100, 150, 200) og EMA (12, 26)
    - **Volatilitet**: Bollinger Bands, Keltner Channel, Donchian Channel, ATR
    - **Oscillatorer**: RSI, Stochastic, MACD, CCI, OBV, Volum
    
    **Signaler:**
    - Vis kjøpssignaler direkte på grafen som trekanter
    - Støtte- og motstandsnivåer vises som stiplede linjer
    
    **Signalhistorikk:**
    - Tabell som viser siste signal fra hver strategi for valgt aksje
    
    **Posisjonskalkulator:**
    - Beregn anbefalt posisjonsstørrelse basert på kapital, risikotoleranse og stop loss
    
    ---
    
    #### ⚙️ Sidebar-innstillinger
    - **Oppdater Data**: Last ned ferske kursdata fra Yahoo Finance
    - **Min. dagsomsetning**: Filtrer bort illikvide aksjer med lav omsetning
    """)

# --- SIDE: MARKEDSTEMPERATUR ---
elif side_valg == "Markedstemperatur":
    st.title("Markedstemperatur")
    
    # --- GMM REGIME ANALYSE ---
    if regime_tilgjengelig:
        st.markdown("""
        <div class='info-box'>
            <span class='horisont-label'>AI-DREVET REGIMEANALYSE</span>
            Bruker Gaussian Mixture Model (GMM) for å identifisere markedsregimer basert på avkastning og volatilitet.
            Inspirert av <a href="https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/" target="_blank">Two Sigma's forskning</a>.
        </div>
        """, unsafe_allow_html=True)
        
        df_market = st.session_state.get('df_market', pd.DataFrame())
        
        if not df_market.empty:
            n_regimes = st.sidebar.slider("Antall regimer", 2, 5, 3, 
                                          help="Flere regimer = mer granulær analyse")
            
            with st.spinner("Analyserer markedsregimer med GMM..."):
                regime_data = regime_model.full_regime_analyse(df_market, n_regimes=n_regimes)
            
            if regime_data:
                current = regime_data['current_info']
                
                # Hovedvisning av nåværende regime
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.1) 100%);
                            border: 3px solid {current['color']}; border-radius: 20px; padding: 35px; 
                            margin-bottom: 24px; text-align: center;
                            box-shadow: 0 0 30px {current['color']}40;">
                    <div style="font-size: 5rem; margin-bottom: 15px;">{current['emoji']}</div>
                    <h1 style="margin: 0; color: {current['color']}; font-size: 2.8rem; font-weight: 800; 
                               text-shadow: 0 0 20px {current['color']}60;">
                        {current['name'].upper()}
                    </h1>
                    <p style="color: #ccd6f6; font-size: 1.15rem; margin-top: 12px; font-style: italic;">
                        {current.get('description', '')}
                    </p>
                    <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.05); 
                                border-radius: 10px; display: inline-block;">
                        <span style="color: #8892b0; font-size: 0.9rem;">ANBEFALT HANDLING:</span>
                        <div style="color: #fff; font-size: 1.1rem; font-weight: 600; margin-top: 5px;">
                            {current.get('action', '')}
                        </div>
                    </div>
                    <div style="margin-top: 20px; color: #8892b0; font-size: 1rem;">
                        Konfidens: <strong style="color: {current['color']};">{current['probability']*100:.1f}%</strong> | 
                        Varighet: <strong>{current['streak_days']} dager</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Regime-sannsynligheter
                st.markdown("### 📊 Regime-sannsynligheter")
                prob_cols = st.columns(len(current['all_probs']))
                
                # Definer farger for hvert regime-navn
                regime_colors = {
                    'Steady Bull': '#00C805',
                    'Volatile Rally': '#90EE90', 
                    'Walking on Ice': '#87CEEB',
                    'Correction': '#FFA500',
                    'Crisis Mode': '#FF5252'
                }
                
                for i, (name, prob) in enumerate(sorted(current['all_probs'].items(), 
                                                        key=lambda x: x[1], reverse=True)):
                    with prob_cols[i]:
                        color = regime_colors.get(name, '#808080')
                        is_current = name == current['name']
                        border_style = f"3px solid {color}" if is_current else f"1px solid {color}40"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 18px; background: {color}15; 
                                    border-radius: 12px; border: {border_style};
                                    {'box-shadow: 0 0 15px ' + color + '40;' if is_current else ''}">
                            <div style="font-size: 2rem; font-weight: 700; color: {color};">
                                {prob*100:.1f}%
                            </div>
                            <div style="color: #ccd6f6; font-size: 0.85rem; margin-top: 5px;">
                                {name}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Metrics
                st.markdown("### 📈 Markedstilstand")
                m1, m2, m3, m4 = st.columns(4)
                
                vol_color = "#FF5252" if current['volatility'] > 0.20 else "#FFA500" if current['volatility'] > 0.12 else "#00C805"
                ret_color = "#00C805" if current['rolling_return'] > 0 else "#FF5252"
                
                m1.metric("Volatilitet (ann.)", f"{current['volatility']*100:.1f}%",
                         delta="Høy" if current['volatility'] > 0.20 else "Normal" if current['volatility'] > 0.12 else "Lav")
                m2.metric("Avkastning (ann.)", f"{current['rolling_return']*100:.1f}%",
                         delta=f"{current['rolling_return']*100:+.1f}%")
                m3.metric("Regime-varighet", f"{current['streak_days']} dager")
                m4.metric("Modell-konfidens", f"{current['probability']*100:.0f}%")
                
                # Visualisering av regime-historie
                st.markdown("### 🕐 Regime-historie")
                
                df_regimes = regime_data['df_regimes']
                
                periode = st.selectbox("Vis periode", ["3 Måneder", "6 Måneder", "1 År", "2 År", "Alt"], index=2)
                dager_map = {"3 Måneder": 63, "6 Måneder": 126, "1 År": 252, "2 År": 504, "Alt": len(df_regimes)}
                df_plot = df_regimes.tail(dager_map[periode])
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.08, row_heights=[0.65, 0.35],
                                   subplot_titles=("Indeks med regime-fargekoding", "Regime-sannsynligheter"))
                
                # Pris med regime-farger
                for regime_name in df_plot['regime_name'].unique():
                    mask = df_plot['regime_name'] == regime_name
                    color = df_plot[mask]['regime_color'].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=df_plot[mask].index,
                        y=df_plot[mask]['Close'],
                        mode='lines',
                        line=dict(color=color, width=2.5),
                        name=regime_name,
                        showlegend=True
                    ), row=1, col=1)
                
                # Regime-sannsynligheter
                for regime_id, label_info in regime_data['regime_labels'].items():
                    name = label_info[0]
                    color = label_info[2]
                    prob_col = f'prob_regime_{regime_id}'
                    
                    if prob_col in df_plot.columns:
                        # Konverter hex til rgba
                        if color == '#00C805':
                            fill_rgba = 'rgba(0, 200, 5, 0.3)'
                        elif color == '#90EE90':
                            fill_rgba = 'rgba(144, 238, 144, 0.3)'
                        elif color == '#87CEEB':
                            fill_rgba = 'rgba(135, 206, 235, 0.3)'
                        elif color == '#FFA500':
                            fill_rgba = 'rgba(255, 165, 0, 0.3)'
                        elif color == '#FF5252':
                            fill_rgba = 'rgba(255, 82, 82, 0.3)'
                        else:
                            fill_rgba = 'rgba(128, 128, 128, 0.3)'
                        
                        fig.add_trace(go.Scatter(
                            x=df_plot.index,
                            y=df_plot[prob_col],
                            mode='lines',
                            fill='tozeroy',
                            line=dict(color=color, width=1.5),
                            fillcolor=fill_rgba,
                            name=f"P({name})",
                            showlegend=True
                        ), row=2, col=1)
                
                fig.update_layout(
                    height=550,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=60, r=20, t=60, b=40),
                    hovermode='x unified'
                )
                fig.update_yaxes(title_text="Indeksverdi", row=1, col=1)
                fig.update_yaxes(title_text="Sannsynlighet", row=2, col=1, range=[0, 1])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Overgangsmatrise
                with st.expander("📊 Regime-overgangsmatrise", expanded=False):
                    st.markdown("""
                    **Hvordan lese tabellen:** Finn raden for nåværende regime, og les kolonnen 
                    for å se sannsynligheten for å gå til det regimet neste periode.
                    
                    *Eksempel: Fra "Walking on Ice" er det X% sjanse for å gå til "Steady Bull".*
                    """)
                    transition_df = regime_data['transition_matrix'] * 100
                    st.dataframe(
                        transition_df.style.format("{:.1f}%").background_gradient(cmap='RdYlGn', axis=1),
                        use_container_width=True
                    )
                
                # Regime-statistikk
                with st.expander("📈 Regime-karakteristikker", expanded=False):
                    st.markdown("**Gjennomsnittlige verdier observert i hvert regime:**")
                    
                    stats_df = regime_data['regime_stats'].copy()
                    
                    # Legg til regime-navn som kolonne (ikke indeks)
                    stats_df = stats_df.reset_index()
                    stats_df['Regime'] = stats_df['regime'].map(
                        lambda x: regime_data['regime_labels'].get(x, ('Ukjent',))[0]
                    )
                    
                    # Velg og rename kolonner
                    stats_df = stats_df[['Regime', 'rolling_return', 'volatility']]
                    stats_df.columns = ['Regime', 'Annualisert Avkastning', 'Annualisert Volatilitet']
                    stats_df['Annualisert Avkastning'] = stats_df['Annualisert Avkastning'] * 100
                    stats_df['Annualisert Volatilitet'] = stats_df['Annualisert Volatilitet'] * 100
                    
                    # Vis uten styling hvis det er problemer, ellers med styling
                    try:
                        st.dataframe(
                            stats_df.style.format({
                                'Annualisert Avkastning': '{:.2f}%',
                                'Annualisert Volatilitet': '{:.2f}%'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    except Exception:
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)

            else:
                st.warning("⚠️ Kunne ikke gjennomføre regimeanalyse. Sjekk at markedsdata er tilgjengelig.")
        else:
            st.warning("Ingen markedsdata tilgjengelig. Klikk 'Oppdater Data' i sidemenyen.")
    else:
        st.info("Regimeanalyse-modul ikke tilgjengelig. Sjekk at `regime_model.py` eksisterer.")
    
    st.markdown("---")
    st.markdown("### 📊 Klassisk Markedsbredde")
    
    sma200_count = 0
    advances = 0
    declines = 0
    
    pb = st.progress(0, text="Beregner markedsbredde...")
    
    # Cache technical indicators per ticker in session state
    if 'teknisk_cache' not in st.session_state:
        st.session_state['teknisk_cache'] = {}

    for i, ticker in enumerate(unike_tickers):
        df_t = df_clean[df_clean['Ticker'] == ticker]
        if len(df_t) > 200:
            if ticker not in st.session_state['teknisk_cache']:
                st.session_state['teknisk_cache'][ticker] = cached_beregn_tekniske_indikatorer(df_t)
            df_t = st.session_state['teknisk_cache'][ticker]
            last_close = df_t['Close'].iloc[-1]
            prev_close = df_t['Close'].iloc[-2]
            sma200 = df_t['SMA_200'].iloc[-1]
            
            if last_close > sma200: sma200_count += 1
            if last_close > prev_close: advances += 1
            elif last_close < prev_close: declines += 1
            
        pb.progress((i+1)/len(unike_tickers))
    pb.empty()
    
    pct_sma200 = (sma200_count / len(unike_tickers)) * 100 if unike_tickers else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Aksjer i Bull-trend (>SMA 200)", f"{pct_sma200:.1f}%")
    c2.metric("Positive aksjer i dag", f"{advances}")
    c3.metric("Negative aksjer i dag", f"{declines}")

# --- SIDE: SCANNER ---
elif side_valg == "Scanner":
    st.title("Algoritmisk Strategi-Scanner")
    
    navn_til_nøkkel = {
        "Kort Sikt (RSI Dip)": "Kort_Sikt_RSI", 
        "Momentum Burst": "Momentum_Burst",
        "Golden Cross": "Golden_Cross", 
        "Ichimoku Breakout": "Ichimoku_Breakout",
        "Wyckoff Spring": "Wyckoff_Spring", 
        "Bull Race Prep": "Bull_Race_Prep",
        "VCP (Minervini)": "VCP_Pattern"
    }
    
    valgt_navn = st.selectbox("Velg teknisk strategi", list(navn_til_nøkkel.keys()))
    strat_key = navn_til_nøkkel[valgt_navn]
    
    detaljer = logic.hent_strategi_detaljer(valgt_navn)
    st.markdown(f"""
    <div class='info-box'>
        <span class='horisont-label'>{detaljer['horisont']}</span>
        {detaljer['beskrivelse']}
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Utfør skanning"):
        resultater = []
        pb = st.progress(0, text="Analyserer tickers...")
        
        for i, ticker in enumerate(unike_tickers):
            df_t = df_clean[df_clean['Ticker'] == ticker]
            if len(df_t) < 50: continue
            # Use cached technical indicators
            if ticker not in st.session_state['teknisk_cache']:
                st.session_state['teknisk_cache'][ticker] = cached_beregn_tekniske_indikatorer(df_t)
            df_t = st.session_state['teknisk_cache'][ticker]
            signaler = logic.sjekk_strategier(df_t)
            info = logic.finn_siste_signal_info(df_t, signaler, strat_key)
            
            resultater.append({
                "Ticker": ticker,
                "Siste Signal": info['dato'],
                "Dager siden": info['dager_siden'],
                "Kursutvikling (%)": info['utvikling_pst'],
                "Nåværende Pris": round(df_t['Close'].iloc[-1], 2)
            })
            pb.progress((i+1)/len(unike_tickers))
        pb.empty()
        
        res_df = pd.DataFrame(resultater).sort_values(by="Dager siden")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

# --- SIDE: TEKNISK ANALYSE ---
elif side_valg == "Teknisk Analyse":
    st.title("Teknisk Analyse Workbench")
    
    c_tick, c_time = st.columns([1, 1])
    ticker = c_tick.selectbox("Velg instrument", unike_tickers)
    horisont = c_time.selectbox("Tidsperspektiv", ["1 Måned", "3 Måneder", "6 Måneder", "1 År", "2 År", "5 År", "Max"], index=3)

    df_full = df_clean[df_clean['Ticker'] == ticker]
    # Use cached technical indicators
    if ticker not in st.session_state['teknisk_cache']:
        st.session_state['teknisk_cache'][ticker] = cached_beregn_tekniske_indikatorer(df_full)
    df_full = st.session_state['teknisk_cache'][ticker]
    signaler = logic.sjekk_strategier(df_full)
    stotte, motstand = logic.finn_stotte_motstand(df_full)

    # --- LIVE INFO BOKS ---
    try:
        import yfinance as yf
        ticker_yf = ticker if ticker.endswith(".OL") else ticker + ".OL"
        data_yf = yf.Ticker(ticker_yf)
        hist = data_yf.history(period="1d", interval="1m")
        if not hist.empty:
            latest_row = hist.iloc[-1]
            latest_price = latest_row['Close']
            latest_time = latest_row.name.strftime("%Y-%m-%d %H:%M")
            day_high = hist['High'].max()
            day_low = hist['Low'].min()
            day_volume = hist['Volume'].sum()
        else:
            latest_price = float(df_full['Close'].iloc[-1])
            latest_time = df_full.index[-1].strftime("%Y-%m-%d")
            day_high = float(df_full['High'].iloc[-1])
            day_low = float(df_full['Low'].iloc[-1])
            day_volume = int(df_full['Volume'].iloc[-1])
    except Exception:
        latest_price = float(df_full['Close'].iloc[-1])
        latest_time = df_full.index[-1].strftime("%Y-%m-%d")
        day_high = float(df_full['High'].iloc[-1])
        day_low = float(df_full['Low'].iloc[-1])
        day_volume = int(df_full['Volume'].iloc[-1])

    # Calculate daily change
    prev_close = float(df_full['Close'].iloc[-2]) if len(df_full) > 1 else latest_price
    daily_change = latest_price - prev_close
    daily_change_pct = (daily_change / prev_close) * 100 if prev_close != 0 else 0
    change_color = "#00C805" if daily_change >= 0 else "#FF5252"
    change_sign = "+" if daily_change >= 0 else ""

    selskap_navn = data.ticker_til_navn(ticker) if hasattr(data, "ticker_til_navn") else ticker.replace(".OL", "")

    # Modern live info box
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                border-radius: 16px; padding: 24px; margin-bottom: 24px; 
                border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
            <div>
                <h1 style="margin: 0; color: #fff; font-size: 2rem; font-weight: 700;">{selskap_navn}</h1>
                <span style="color: #8892b0; font-size: 1rem; font-weight: 500;">{ticker}</span>
            </div>
            <div style="text-align: right;">
                <h1 style="margin: 0; color: #fff; font-size: 2.5rem; font-weight: 800;">{latest_price:.2f}</h1>
                <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
                    <span style="color: {change_color}; font-size: 1.2rem; font-weight: 700;">
                        {change_sign}{daily_change:.2f}
                    </span>
                    <span style="background: {change_color}20; color: {change_color}; padding: 4px 12px; 
                                 border-radius: 20px; font-size: 0.95rem; font-weight: 600;">
                        {change_sign}{daily_change_pct:.2f}%
                    </span>
                </div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 16px; margin-top: 20px; 
                    padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
            <div style="text-align: center;">
                <span style="color: #8892b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Oppdatert</span>
                <div style="color: #fff; font-size: 1rem; font-weight: 600; margin-top: 4px;">{latest_time}</div>
            </div>
            <div style="text-align: center;">
                <span style="color: #8892b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Høy</span>
                <div style="color: #00C805; font-size: 1rem; font-weight: 600; margin-top: 4px;">{day_high:.2f}</div>
            </div>
            <div style="text-align: center;">
                <span style="color: #8892b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Lav</span>
                <div style="color: #FF5252; font-size: 1rem; font-weight: 600; margin-top: 4px;">{day_low:.2f}</div>
            </div>
            <div style="text-align: center;">
                <span style="color: #8892b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Volum</span>
                <div style="color: #fff; font-size: 1rem; font-weight: 600; margin-top: 4px;">{day_volume:,.0f}</div>
            </div>
            <div style="text-align: center;">
                <span style="color: #8892b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Forrige</span>
                <div style="color: #fff; font-size: 1rem; font-weight: 600; margin-top: 4px;">{prev_close:.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    end_date = df_full.index.max()
    offset_dager = {"1 Måned": 30, "3 Måneder": 90, "6 Måneder": 180, "1 År": 365, "2 År": 730, "5 År": 1825, "Max": 3650}
    start_date = end_date - pd.DateOffset(days=offset_dager[horisont])
    df_view = df_full[df_full.index >= start_date]
    
    # Modern chart settings in tabs
    tab_chart, tab_indicators, tab_signals = st.tabs(["📊 Graf", "📈 Indikatorer", "🎯 Signaler"])
    
    with tab_chart:
        col1, col2 = st.columns([1, 1])
        with col1:
            v_chart_type = st.radio("Graftype", ["Candlestick", "OHLC", "Linje", "Area"], horizontal=True)
        with col2:
            v_theme = st.radio("Tema", ["Mørk", "Lys"], horizontal=True)
    
    with tab_indicators:
        st.markdown("##### Trendlinjer")
        col1, col2, col3 = st.columns(3)
        with col1:
            v_sma_10 = st.checkbox("SMA 10", value=False)
            v_sma_20 = st.checkbox("SMA 20", value=False)
            v_sma_50 = st.checkbox("SMA 50", value=True)
        with col2:
            v_sma_100 = st.checkbox("SMA 100", value=False)
            v_sma_150 = st.checkbox("SMA 150", value=False)
            v_sma_200 = st.checkbox("SMA 200", value=True)
        with col3:
            v_ema_12 = st.checkbox("EMA 12", value=False)
            v_ema_26 = st.checkbox("EMA 26", value=False)
            v_vwap = st.checkbox("VWAP", value=False)
        
        st.markdown("##### Volatilitet & Bånd")
        col1, col2 = st.columns(2)
        with col1:
            v_bb = st.checkbox("Bollinger Bands (20, 2)", value=False)
            v_keltner = st.checkbox("Keltner Channel", value=False)
        with col2:
            v_atr = st.checkbox("ATR (14)", value=False)
            v_donchian = st.checkbox("Donchian Channel (20)", value=False)
        
        st.markdown("##### Oscillatorer")
        col1, col2, col3 = st.columns(3)
        with col1:
            v_rsi = st.checkbox("RSI (14)", value=False)
            v_stoch = st.checkbox("Stochastic (14,3)", value=False)
        with col2:
            v_macd = st.checkbox("MACD (12,26,9)", value=False)
            v_cci = st.checkbox("CCI (20)", value=False)
        with col3:
            v_vol = st.checkbox("Volum", value=True)
            v_obv = st.checkbox("OBV", value=False)
    
    with tab_signals:
        # Prepare signal display names for dropdown
        signal_keys = ["Kort_Sikt_RSI", "Momentum_Burst", "Golden_Cross", 
                       "Ichimoku_Breakout", "Wyckoff_Spring", "Bull_Race_Prep", "VCP_Pattern"]
        signal_names = {
            "Kort_Sikt_RSI": "RSI Mean Reversion",
            "Momentum_Burst": "Momentum Burst",
            "Golden_Cross": "Golden Cross",
            "Ichimoku_Breakout": "Ichimoku Breakout",
            "Wyckoff_Spring": "Wyckoff Spring",
            "Bull_Race_Prep": "Bull Race Prep",
            "VCP_Pattern": "VCP (Minervini)"
        }
        
        v_sigs = st.multiselect(
            "Vis kjøpssignaler på chart",
            signal_keys,
            default=["VCP_Pattern"],
            format_func=lambda x: signal_names.get(x, x)
        )
        
        v_support_resistance = st.checkbox("Vis støtte/motstand", value=True)

    # Beregn ekstra indikatorer
    if v_ema_12 or v_ema_26:
        df_view = df_view.copy()
        if v_ema_12:
            df_view['EMA_12'] = df_view['Close'].ewm(span=12, adjust=False).mean()
        if v_ema_26:
            df_view['EMA_26'] = df_view['Close'].ewm(span=26, adjust=False).mean()
    
    if v_sma_10 or v_sma_20 or v_sma_100:
        df_view = df_view.copy() if 'EMA_12' not in df_view.columns else df_view
        if v_sma_10:
            df_view['SMA_10'] = df_view['Close'].rolling(10).mean()
        if v_sma_20:
            df_view['SMA_20'] = df_view['Close'].rolling(20).mean()
        if v_sma_100:
            df_view['SMA_100'] = df_view['Close'].rolling(100).mean()
    
    if v_stoch:
        df_view = df_view.copy() if 'SMA_10' not in df_view.columns else df_view
        low_14 = df_view['Low'].rolling(14).min()
        high_14 = df_view['High'].rolling(14).max()
        df_view['Stoch_K'] = 100 * (df_view['Close'] - low_14) / (high_14 - low_14)
        df_view['Stoch_D'] = df_view['Stoch_K'].rolling(3).mean()
    
    if v_cci:
        df_view = df_view.copy() if 'Stoch_K' not in df_view.columns else df_view
        tp = (df_view['High'] + df_view['Low'] + df_view['Close']) / 3
        df_view['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    
    if v_obv:
        df_view = df_view.copy() if 'CCI' not in df_view.columns else df_view
        obv = [0]
        for i in range(1, len(df_view)):
            if df_view['Close'].iloc[i] > df_view['Close'].iloc[i-1]:
                obv.append(obv[-1] + df_view['Volume'].iloc[i])
            elif df_view['Close'].iloc[i] < df_view['Close'].iloc[i-1]:
                obv.append(obv[-1] - df_view['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df_view['OBV'] = obv
    
    if v_donchian:
        df_view = df_view.copy() if 'OBV' not in df_view.columns else df_view
        df_view['Donchian_High'] = df_view['High'].rolling(20).max()
        df_view['Donchian_Low'] = df_view['Low'].rolling(20).min()
    
    if v_keltner:
        df_view = df_view.copy() if 'Donchian_High' not in df_view.columns else df_view
        df_view['Keltner_Mid'] = df_view['Close'].ewm(span=20, adjust=False).mean()
        df_view['Keltner_Upper'] = df_view['Keltner_Mid'] + 2 * df_view['ATR']
        df_view['Keltner_Lower'] = df_view['Keltner_Mid'] - 2 * df_view['ATR']
    
    if v_vwap:
        df_view = df_view.copy() if 'Keltner_Mid' not in df_view.columns else df_view
        df_view['VWAP'] = (df_view['Volume'] * (df_view['High'] + df_view['Low'] + df_view['Close']) / 3).cumsum() / df_view['Volume'].cumsum()

    # Konstruksjon av plots
    r_count = 1
    heights = [0.55]
    row_titles = ["Pris"]
    
    if v_vol:
        r_count += 1
        heights.append(0.12)
        row_titles.append("Volum")
    if v_rsi:
        r_count += 1
        heights.append(0.11)
        row_titles.append("RSI")
    if v_stoch:
        r_count += 1
        heights.append(0.11)
        row_titles.append("Stochastic")
    if v_macd:
        r_count += 1
        heights.append(0.11)
        row_titles.append("MACD")
    if v_cci:
        r_count += 1
        heights.append(0.11)
        row_titles.append("CCI")
    if v_obv:
        r_count += 1
        heights.append(0.11)
        row_titles.append("OBV")
    if v_atr:
        r_count += 1
        heights.append(0.11)
        row_titles.append("ATR")
    
    norm_heights = [h/sum(heights) for h in heights]
    
    fig = make_subplots(
        rows=r_count, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=norm_heights,
        row_titles=row_titles
    )
    
    # Tema
    if v_theme == "Mørk":
        bg_color = "#0e1117"
        grid_color = "rgba(255,255,255,0.06)"
        text_color = "#fafafa"
        up_color = "#00C805"
        down_color = "#FF5252"
    else:
        bg_color = "#ffffff"
        grid_color = "rgba(0,0,0,0.08)"
        text_color = "#1a1a1a"
        up_color = "#26a69a"
        down_color = "#ef5350"
    
    # Hovedgraf
    if v_chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df_view.index, 
            open=df_view['Open'], 
            high=df_view['High'], 
            low=df_view['Low'], 
            close=df_view['Close'], 
            name="Pris",
            increasing=dict(line=dict(color=up_color), fillcolor=up_color),
            decreasing=dict(line=dict(color=down_color), fillcolor=down_color)
        ), row=1, col=1)
    elif v_chart_type == "OHLC":
        fig.add_trace(go.Ohlc(
            x=df_view.index, 
            open=df_view['Open'], 
            high=df_view['High'], 
            low=df_view['Low'], 
            close=df_view['Close'], 
            name="Pris",
            increasing_line_color=up_color,
            decreasing_line_color=down_color
        ), row=1, col=1)
    elif v_chart_type == "Area":
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view['Close'], 
            mode='lines', 
            fill='tozeroy',
            line=dict(color=up_color, width=2),
            fillcolor=f"rgba(0, 200, 5, 0.1)",
            name="Pris"
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view['Close'], 
            mode='lines', 
            line=dict(color=up_color, width=2), 
            name="Pris"
        ), row=1, col=1)

    # SMAs
    sma_colors = {
        'SMA_10': '#FF6B6B', 'SMA_20': '#4ECDC4', 'SMA_50': '#45B7D1',
        'SMA_100': '#96CEB4', 'SMA_150': '#FFEAA7', 'SMA_200': '#DDA0DD'
    }
    
    if v_sma_10 and 'SMA_10' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_10'], line=dict(color=sma_colors['SMA_10'], width=1), name="SMA 10"), row=1, col=1)
    if v_sma_20 and 'SMA_20' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_20'], line=dict(color=sma_colors['SMA_20'], width=1), name="SMA 20"), row=1, col=1)
    if v_sma_50 and 'SMA_50' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_50'], line=dict(color=sma_colors['SMA_50'], width=1.5), name="SMA 50"), row=1, col=1)
    if v_sma_100 and 'SMA_100' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_100'], line=dict(color=sma_colors['SMA_100'], width=1), name="SMA 100"), row=1, col=1)
    if v_sma_150 and 'SMA_150' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_150'], line=dict(color=sma_colors['SMA_150'], width=1, dash='dot'), name="SMA 150"), row=1, col=1)
    if v_sma_200 and 'SMA_200' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_200'], line=dict(color=sma_colors['SMA_200'], width=1.5), name="SMA 200"), row=1, col=1)
    
    # EMAs
    if v_ema_12 and 'EMA_12' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['EMA_12'], line=dict(color='#FF9F43', width=1, dash='dash'), name="EMA 12"), row=1, col=1)
    if v_ema_26 and 'EMA_26' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['EMA_26'], line=dict(color='#EE5A24', width=1, dash='dash'), name="EMA 26"), row=1, col=1)
    
    # VWAP
    if v_vwap and 'VWAP' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['VWAP'], line=dict(color='#9B59B6', width=1.5, dash='dot'), name="VWAP"), row=1, col=1)
    
    # Bollinger Bands
    if v_bb and 'BB_Upper' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['BB_Upper'], line=dict(color='rgba(174, 214, 241, 0.7)', width=1), name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['BB_Lower'], line=dict(color='rgba(174, 214, 241, 0.7)', width=1), name="BB Lower", fill='tonexty', fillcolor='rgba(174, 214, 241, 0.15)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['BB_Middle'], line=dict(color='rgba(174, 214, 241, 0.9)', width=1, dash='dash'), name="BB Mid"), row=1, col=1)
    
    # Keltner Channel
    if v_keltner and 'Keltner_Upper' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Keltner_Upper'], line=dict(color='rgba(243, 156, 18, 0.6)', width=1), name="Keltner Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Keltner_Lower'], line=dict(color='rgba(243, 156, 18, 0.6)', width=1), name="Keltner Lower", fill='tonexty', fillcolor='rgba(243, 156, 18, 0.1)'), row=1, col=1)
    
    # Donchian Channel
    if v_donchian and 'Donchian_High' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Donchian_High'], line=dict(color='rgba(46, 204, 113, 0.6)', width=1), name="Donchian High", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Donchian_Low'], line=dict(color='rgba(46, 204, 113, 0.6)', width=1), name="Donchian Low", fill='tonexty', fillcolor='rgba(46, 204, 113, 0.1)'), row=1, col=1)

    # Signaler (Markører)
    farger = {
        "Kort_Sikt_RSI": "#2ecc71", "Momentum_Burst": "#9b59b6", "Golden_Cross": "#f1c40f", 
        "Ichimoku_Breakout": "#00cec9", "Wyckoff_Spring": "#3498db", "Bull_Race_Prep": "#e91e63",
        "VCP_Pattern": "#ff9f43"
    }
    
    for s in v_sigs:
        mask = signaler[s][signaler[s].index >= start_date]
        sigs = df_view[mask]
        if not sigs.empty:
            fig.add_trace(go.Scatter(
                x=sigs.index, 
                y=sigs['Low'] * 0.97, 
                mode='markers', 
                marker=dict(symbol='triangle-up', size=14, color=farger[s], line=dict(width=1, color='white')),
                name=signal_names.get(s, s),
                hovertemplate=f"<b>{signal_names.get(s, s)}</b><br>Dato: %{{x}}<br>Pris: %{{y:.2f}}<extra></extra>"
            ), row=1, col=1)

    # Støtte og Motstandslinjer
    if v_support_resistance:
        for val in stotte:
            fig.add_hline(y=val, line_dash="dash", line_color="#2ecc71", opacity=0.5, row=1, col=1, 
                         annotation_text=f"S: {val:.2f}", annotation_position="right")
        for val in motstand:
            fig.add_hline(y=val, line_dash="dash", line_color="#e74c3c", opacity=0.5, row=1, col=1,
                         annotation_text=f"R: {val:.2f}", annotation_position="right")

    # Sekundære grafer
    curr = 2
    
    if v_vol:
        colors = [up_color if df_view['Close'].iloc[i] >= df_view['Open'].iloc[i] else down_color for i in range(len(df_view))]
        fig.add_trace(go.Bar(x=df_view.index, y=df_view['Volume'], name="Volum", marker_color=colors, opacity=0.7), row=curr, col=1)
        curr += 1
    
    if v_rsi:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['RSI'], name="RSI", line=dict(color='#a29bfe', width=1.5)), row=curr, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#e74c3c", opacity=0.5, row=curr, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#2ecc71", opacity=0.5, row=curr, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(162, 155, 254, 0.1)", line_width=0, row=curr, col=1)
        curr += 1
    
    if v_stoch and 'Stoch_K' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Stoch_K'], name="%K", line=dict(color='#00cec9', width=1.5)), row=curr, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Stoch_D'], name="%D", line=dict(color='#fdcb6e', width=1.5)), row=curr, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="#e74c3c", opacity=0.5, row=curr, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="#2ecc71", opacity=0.5, row=curr, col=1)
        curr += 1
    
    if v_macd and 'MACD' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MACD'], name="MACD", line=dict(color='#3498db', width=1.5)), row=curr, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MACD_Signal'], name="Signal", line=dict(color='#e74c3c', width=1.5)), row=curr, col=1)
        colors_macd = [up_color if v >= 0 else down_color for v in df_view['MACD_Hist']]
        fig.add_trace(go.Bar(x=df_view.index, y=df_view['MACD_Hist'], name="Histogram", marker_color=colors_macd, opacity=0.6), row=curr, col=1)
        curr += 1
    
    if v_cci and 'CCI' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['CCI'], name="CCI", line=dict(color='#e67e22', width=1.5)), row=curr, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="#e74c3c", opacity=0.5, row=curr, col=1)
        fig.add_hline(y=-100, line_dash="dash", line_color="#2ecc71", opacity=0.5, row=curr, col=1)
        curr += 1
    
    if v_obv and 'OBV' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['OBV'], name="OBV", line=dict(color='#1abc9c', width=1.5), fill='tozeroy', fillcolor='rgba(26, 188, 156, 0.1)'), row=curr, col=1)
        curr += 1
    
    if v_atr and 'ATR' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['ATR'], name="ATR", line=dict(color='#9b59b6', width=1.5), fill='tozeroy', fillcolor='rgba(155, 89, 182, 0.1)'), row=curr, col=1)

    fig.update_layout(
        height=700 + (r_count - 1) * 80,
        template="plotly_dark" if v_theme == "Mørk" else "plotly_white",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10)
        ),
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            gridcolor=grid_color,
            showgrid=True
        ),
        yaxis=dict(gridcolor=grid_color, showgrid=True),
        hovermode='x unified'
    )
    
    # Update all y-axes
    for i in range(1, r_count + 1):
        fig.update_yaxes(gridcolor=grid_color, showgrid=True, row=i, col=1)
    
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {'format': 'png', 'filename': f'{ticker}_chart', 'height': 800, 'width': 1600, 'scale': 2}
    })
    
    # --- Signalhistorikk tabell ---
    st.markdown("### Siste signaler og kursutvikling for alle strategier")
    all_signal_keys = ["Kort_Sikt_RSI", "Momentum_Burst", "Golden_Cross", 
                      "Ichimoku_Breakout", "Wyckoff_Spring", "Bull_Race_Prep", "VCP_Pattern"]
    signal_display = {k: k.replace("_", " ").title() for k in all_signal_keys}
    signal_rows = []
    for s in all_signal_keys:
        info = logic.finn_siste_signal_info(df_full, signaler, s)
        signal_rows.append({
            "Strategi": signal_display[s],
            "Siste Signal": info['dato'],
            "Dager siden": info['dager_siden'],
            "Kursutvikling (%)": info['utvikling_pst']
        })
    signal_df = pd.DataFrame(signal_rows)
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    # Kalkulator
    st.markdown("---")
    st.subheader("Posisjonskalkulator")
    k1, k2, k3, k4 = st.columns(4)
    kap = k1.number_input("Kapital (NOK)", value=config.DEFAULT_KAPITAL)
    ris = k2.number_input("Risiko per handel (%)", value=config.DEFAULT_RISIKO_PROSENT)
    ent = k3.number_input("Inngangspris", value=float(df_full['Close'].iloc[-1]))
    stp = k4.number_input("Stop Loss", value=float(df_full['Close'].iloc[-1]*0.95))
    
    if ent > stp:
        res = logic.beregn_risk_reward(ent, stp, kap, ris)
        if res:
            st.info(f"Anbefalt posisjonsstørrelse for {ticker}: **{res['antall']} aksjer**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Eksponering", f"{res['total_investering']:,.0f} kr")
            c2.metric("Risiko i kroner", f"{res['risiko_kr']:.0f} kr")
            c3.metric("Mål (2R)", f"{res['target_2r']:.2f}")

# --- SIDE: BETA AI SCANNER ---
elif side_valg == "Beta: AI Scanner" and beta_tilgjengelig:
    beta_ml.vis_beta_side(df_clean, unike_tickers)