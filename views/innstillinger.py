"""
Innstillinger — Brukerpreferanser og standardverdier.

Lar brukeren justere:
- Scanner-standarder (kvalitet, dager, volum, R:R)
- Chart-preferanser (indikatorer, tema)
- Portefølje-innstillinger (kurtasje, spread)
- Generelle innstillinger
"""
import streamlit as st
import user_settings


def render():
    """Rendrer innstillingssiden."""
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 8px;">⚙️ Innstillinger</h1>
        <p style="color: rgba(255,255,255,0.6); font-size: 1.1rem;">
            Tilpass standardverdier og preferanser
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Last gjeldende innstillinger
    settings = user_settings.load_settings()
    
    # Track endringer
    changed = False
    
    # === SCANNER-INNSTILLINGER ===
    st.markdown("### 📊 Scanner")
    st.markdown("Standardfiltre for aksje-scanneren.")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            kvalitet_valg = ["A", "B", "C", "D"]
            current_kval = settings['scanner'].get('min_kvalitet', 'C')
            idx = kvalitet_valg.index(current_kval) if current_kval in kvalitet_valg else 2
            ny_min_kvalitet = st.selectbox(
                "Minimum kvalitet",
                kvalitet_valg,
                index=idx,
                help="Laveste kvalitetsklasse som vises i scanner. A=beste, D=laveste."
            )
            if ny_min_kvalitet != settings['scanner']['min_kvalitet']:
                settings['scanner']['min_kvalitet'] = ny_min_kvalitet
                changed = True
        
        with col2:
            ny_max_dager = st.number_input(
                "Maks dager siden signal",
                min_value=1,
                max_value=90,
                value=settings['scanner'].get('max_dager', 30),
                help="Signaler eldre enn dette filtreres bort."
            )
            if ny_max_dager != settings['scanner']['max_dager']:
                settings['scanner']['max_dager'] = ny_max_dager
                changed = True
        
        with col3:
            ny_min_volum = st.number_input(
                "Min volum-ratio",
                min_value=0.0,
                max_value=5.0,
                value=float(settings['scanner'].get('min_volum_ratio', 0.8)),
                step=0.1,
                help="Minimum volum relativt til gjennomsnitt. 1.0 = gjennomsnittlig volum."
            )
            if ny_min_volum != settings['scanner']['min_volum_ratio']:
                settings['scanner']['min_volum_ratio'] = ny_min_volum
                changed = True
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        ny_min_rr = st.number_input(
            "Minimum R:R ratio",
            min_value=0.0,
            max_value=10.0,
            value=float(settings['scanner'].get('min_rr', 0.0)),
            step=0.5,
            help="Minimum risk/reward ratio. 0 = ingen filter."
        )
        if ny_min_rr != settings['scanner'].get('min_rr', 0.0):
            settings['scanner']['min_rr'] = ny_min_rr
            changed = True
    
    with col5:
        ny_filtrer_fb = st.checkbox(
            "Filtrer false breakouts",
            value=settings['scanner'].get('filtrer_false_breakout', False),
            help="Skjul signaler der prisen har falt under signalnivå OG SMA10."
        )
        if ny_filtrer_fb != settings['scanner'].get('filtrer_false_breakout', False):
            settings['scanner']['filtrer_false_breakout'] = ny_filtrer_fb
            changed = True
    
    with col6:
        ny_vis_exit = st.checkbox(
            "Vis exit-signaler",
            value=settings['scanner'].get('vis_exit', True),
            help="Marker aksjer med aktive selg-signaler."
        )
        if ny_vis_exit != settings['scanner'].get('vis_exit', True):
            settings['scanner']['vis_exit'] = ny_vis_exit
            changed = True
    
    st.markdown("---")
    
    # === CHART-INNSTILLINGER ===
    st.markdown("### 📈 Chart")
    st.markdown("Standardinnstillinger for prisdiagrammer.")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        chart_typer = ["candlestick", "linje"]
        current_chart = settings['chart'].get('chart_type', 'candlestick')
        idx_chart = chart_typer.index(current_chart) if current_chart in chart_typer else 0
        ny_chart_type = st.selectbox(
            "Chart-type",
            chart_typer,
            index=idx_chart,
            format_func=lambda x: "Candlestick 🕯️" if x == "candlestick" else "Linje 📉"
        )
        if ny_chart_type != settings['chart']['chart_type']:
            settings['chart']['chart_type'] = ny_chart_type
            changed = True
    
    with col_c2:
        perioder = ["3M", "6M", "1Y", "2Y", "5Y", "Max"]
        current_periode = settings['chart'].get('tidsperiode', '1Y')
        idx_periode = perioder.index(current_periode) if current_periode in perioder else 2
        ny_periode = st.selectbox(
            "Standard tidsperiode",
            perioder,
            index=idx_periode,
            help="Hvor langt tilbake chartet viser som standard."
        )
        if ny_periode != settings['chart']['tidsperiode']:
            settings['chart']['tidsperiode'] = ny_periode
            changed = True
    
    st.markdown("**Standard indikatorer:**")
    
    indikatorer = settings['chart'].get('indikatorer', {})
    
    ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
    
    with ind_col1:
        ny_sma50 = st.checkbox("SMA 50", value=indikatorer.get('sma_50', True))
        if ny_sma50 != indikatorer.get('sma_50', True):
            settings['chart']['indikatorer']['sma_50'] = ny_sma50
            changed = True
        
        ny_sma200 = st.checkbox("SMA 200", value=indikatorer.get('sma_200', True))
        if ny_sma200 != indikatorer.get('sma_200', True):
            settings['chart']['indikatorer']['sma_200'] = ny_sma200
            changed = True
    
    with ind_col2:
        ny_sma20 = st.checkbox("SMA 20", value=indikatorer.get('sma_20', False))
        if ny_sma20 != indikatorer.get('sma_20', False):
            settings['chart']['indikatorer']['sma_20'] = ny_sma20
            changed = True
        
        ny_bollinger = st.checkbox("Bollinger Bands", value=indikatorer.get('bollinger', False))
        if ny_bollinger != indikatorer.get('bollinger', False):
            settings['chart']['indikatorer']['bollinger'] = ny_bollinger
            changed = True
    
    with ind_col3:
        ny_rsi = st.checkbox("RSI", value=indikatorer.get('rsi', False))
        if ny_rsi != indikatorer.get('rsi', False):
            settings['chart']['indikatorer']['rsi'] = ny_rsi
            changed = True
        
        ny_macd = st.checkbox("MACD", value=indikatorer.get('macd', False))
        if ny_macd != indikatorer.get('macd', False):
            settings['chart']['indikatorer']['macd'] = ny_macd
            changed = True
    
    with ind_col4:
        ny_volum = st.checkbox("Volum", value=indikatorer.get('volum', True))
        if ny_volum != indikatorer.get('volum', True):
            settings['chart']['indikatorer']['volum'] = ny_volum
            changed = True
        
        ny_ichimoku = st.checkbox("Ichimoku Cloud", value=indikatorer.get('ichimoku', False))
        if ny_ichimoku != indikatorer.get('ichimoku', False):
            settings['chart']['indikatorer']['ichimoku'] = ny_ichimoku
            changed = True
    
    st.markdown("---")
    
    # === ML-INNSTILLINGER ===
    st.markdown("### 🧠 ML / AI")
    st.markdown("Innstillinger relatert til maskinlæringsanalyse og subchart-historikk.")
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        ml_options = [30, 60, 120, 180, 365]
        current_ml = settings.get('teknisk_analyse', {}).get('ml_hist_days', 120)
        if current_ml in ml_options:
            idx_ml = ml_options.index(current_ml)
        else:
            idx_ml = 2
        ny_ml_hist = st.selectbox("ML-historikk (dager)", ml_options, index=idx_ml, help="Hvor mange dager historisk ML-score skal beregnes for subchart")
        if ny_ml_hist != settings['teknisk_analyse'].get('ml_hist_days', 120):
            settings['teknisk_analyse']['ml_hist_days'] = int(ny_ml_hist)
            changed = True
    with ml_col2:
        st.markdown("\n")

    st.markdown("---")
    
    # === PORTEFØLJE-INNSTILLINGER ===
    st.markdown("### 💼 Portefølje")
    st.markdown("Transaksjonskostnader og valuta.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        ny_kurtasje = st.number_input(
            "Kurtasje (%)",
            min_value=0.0,
            max_value=1.0,
            value=float(settings['portefolje'].get('kurtasje_pct', 0.05)),
            step=0.01,
            format="%.2f",
            help="Kurtasje per transaksjon i prosent."
        )
        if ny_kurtasje != settings['portefolje'].get('kurtasje_pct', 0.05):
            settings['portefolje']['kurtasje_pct'] = ny_kurtasje
            changed = True
    
    with col_p2:
        ny_spread = st.number_input(
            "Spread/slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=float(settings['portefolje'].get('spread_pct', 0.10)),
            step=0.01,
            format="%.2f",
            help="Estimert spread og slippage i prosent."
        )
        if ny_spread != settings['portefolje'].get('spread_pct', 0.10):
            settings['portefolje']['spread_pct'] = ny_spread
            changed = True
    
    with col_p3:
        valuta_valg = ["NOK", "USD", "EUR"]
        current_valuta = settings['portefolje'].get('valuta', 'NOK')
        idx_valuta = valuta_valg.index(current_valuta) if current_valuta in valuta_valg else 0
        ny_valuta = st.selectbox(
            "Valuta",
            valuta_valg,
            index=idx_valuta
        )
        if ny_valuta != settings['portefolje']['valuta']:
            settings['portefolje']['valuta'] = ny_valuta
            changed = True
    
    st.markdown("---")
    
    # === GENERELLE INNSTILLINGER ===
    st.markdown("### 🔧 Generelt")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        ny_min_omsetning = st.number_input(
            "Min. dagsomsetning (NOK)",
            min_value=0,
            max_value=10_000_000,
            value=int(settings['generelt'].get('min_dagsomsetning', 500000)),
            step=100000,
            help="Aksjer med lavere gjennomsnittlig dagsomsetning filtreres bort."
        )
        if ny_min_omsetning != settings['generelt'].get('min_dagsomsetning', 500000):
            settings['generelt']['min_dagsomsetning'] = ny_min_omsetning
            changed = True
    
    st.markdown("---")
    
    # === LAGRE / TILBAKESTILL ===
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("💾 Lagre innstillinger", type="primary", disabled=not changed):
            if user_settings.save_settings(settings):
                st.success("✅ Innstillinger lagret!")
                st.rerun()
            else:
                st.error("Kunne ikke lagre innstillinger.")
    
    with col_btn2:
        if st.button("🔄 Tilbakestill alt", type="secondary"):
            if user_settings.reset_settings():
                st.success("Innstillinger tilbakestilt til standard.")
                st.rerun()
            else:
                st.error("Kunne ikke tilbakestille innstillinger.")
    
    # Vis endringsstatus
    if changed:
        st.info("💡 Du har ulagrede endringer. Klikk 'Lagre innstillinger' for å beholde dem.")
    
    # === INFO-BOKS ===
    with st.expander("ℹ️ Om innstillinger"):
        st.markdown("""
        **Innstillinger lagres lokalt** i `data_storage/user_settings.json`.
        
        Disse innstillingene brukes som standardverdier når du:
        - Åpner scanneren
        - Viser charts i Teknisk Analyse
        - Beregner porteføljeavkastning
        
        Du kan fortsatt overstyre innstillingene midlertidig i hver visning.
        
        **Tips:** Klikk '💾 Lagre filter' i scanneren for å lagre gjeldende 
        filterinnstillinger direkte derfra.
        """)
