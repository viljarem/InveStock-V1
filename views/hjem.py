# views/hjem.py
"""Hjemmeside for InveStock Pro."""

import streamlit as st


def render():
    """Renderer Hjem-siden."""
    unike_tickers = st.session_state.get('unike_tickers', [])

    st.title("InveStock Pro")
    st.markdown(f"*Avansert teknisk analyse for Oslo Børs — {len(unike_tickers)} aksjer overvåkes*")

    st.markdown("---")

    # === INTRODUKSJON ===
    st.markdown("""
    ## Velkommen

    InveStock Pro er et profesjonelt analyseverktøy for teknisk og fundamental analyse av Oslo Børs.
    Appen kombinerer 9 tekniske strategier med avanserte analysemetoder som markedsregime-analyse,
    Smart Money Flow, mønstergjenkjenning og realistisk backtesting med transaksjonskostnader.

    **Nylig strømlinjeformet** — fjernet eksperimentelle moduler for å fokusere på kjernefunksjonalitet.

    **NB:** Dette er et analyseverktøy, ikke investeringsråd. All handel innebærer risiko.
    """)

    # === APPENS SIDER ===
    st.markdown("---")
    st.markdown("## Appens sider")

    tab_info = st.tabs([
        "Markedstemperatur", "Scanner", "Teknisk Analyse"
    ])

    with tab_info[0]:
        st.markdown("""
        ### Markedstemperatur

        Overordnet markedsbilde med regime-analyse og bredde-indikatorer.

        **Regime-analyse (Hidden Markov Model):**
        - Klassifiserer markedet i regimer (Bull, Bear, Volatilt, Stabilt, Korreksjon)
        - Sanntids-sannsynligheter for hvert regime med handlingsanbefaling
        - Historisk regime-graf og overgangsmatrise

        **Markedsbredde:**
        - Andel aksjer over SMA 200 og SMA 50
        - Nye 52-ukers høy vs. nye 52-ukers lav
        - Advance/Decline-ratio og kumulativ A/D-linje
        - McClellan Oscillator (bredde-momentum)
        - Historisk bredde-chart (siste 6 måneder, 3 paneler)

        **Makroindikatorer:**
        - Brent crude-korrelasjon med OSEBX
        - USD/NOK-paring og dens påvirkning på markedet
        """)

    with tab_info[1]:
        st.markdown("""
        ### Scanner

        Skanner Oslo Børs for aksjer som oppfyller kriteriene til 9 tekniske strategier.
        Nå med **realistisk backtesting** inkludert transaksjonskostnader og likviditetsfilter.

        **Slik bruker du scanneren:**
        1. Velg en strategi fra nedtrekksmenyen (eller "Alle strategier")
        2. Klikk "Utfør skanning"  
        3. Se resultater sortert etter dato, score, RS-rating eller kursutvikling

        **Resultat-kolonner:**

        | Kolonne | Beskrivelse |
        |---------|-------------|
        | Ticker | Aksjens ticker-symbol |
        | Signal | Dato for siste kjøpssignal |
        | Dager | Handelsdager siden signalet |
        | Pris | Siste sluttkurs |
        | Utv% | Kursutvikling siden signalet (inkl. peak%) |
        | Score | Signalkvalitet (A-D) basert på realistisk backtesting |
        | RS | IBD Relativ Styrke vs. OSEBX |
        | MTF | Multi-timeframe konvergens |
        | R:R | Risk/Reward-ratio basert på støtte/motstand |
        | Mønster | Detekterte tekniske mønstre (bullish/bearish) |
        | Exit | Markeres hvis exit-signal er aktivt |

        **Valgfrie tillegg (checkboxer):**
        - Fundamentalanalyse (P/E, gjeld, inntjening)
        - Smart Money Flow (institusjonell akkumulering/distribusjon)
        - Innsidehandel (kjøp/salg fra innsidere)
        - Mønsterfilter (6 tekniske mønstre: H&S, trekanter, kopp&hank, m.fl.)
        - Risk/Reward-filter (min. R:R-ratio basert på støtte/motstand)
        - Regime-filter (tilpasser signalkrav til markedsregimet)
        - Sektor-konsentrasjonsfilter

        **Resultat-faner:**
        - Alle treff med klikkbar navigering til Teknisk Analyse
        - Watchlist med P/L-oversikt, gjennomsnittlig avkastning og metadata
        
        **Backtest-statistikk per strategi:**
        - Realistisk win rate med +1 dag entry timing
        - Profit factor med trailing stops og profit targets
        - Transaksjonskostnader (0.15% roundtrip) inkludert
        - Likviditetsfilter (min. 5M NOK dagsomsetning)
        """)

    with tab_info[2]:
        st.markdown("""
        ### Teknisk Analyse

        Detaljert analyse av enkeltticker med TradingView-inspirerte charts.

        **Chart-funksjoner:**
        - Graftyper: Candlestick, OHLC, Linje, Area
        - Tidsperioder: 3 måneder til 5 år
        - Mørkt og lyst tema
        - Forhåndsdefinerte indikatoroppsett (Swing, Trend, Intraday m.fl.)

        **Tilgjengelige indikatorer:**

        | Kategori | Indikatorer |
        |----------|-------------|
        | Glidende snitt | SMA (10, 20, 50, 100, 150, 200), EMA (9, 12, 21, 26) |
        | Volatilitet | Bollinger Bands, Keltner Channel, Donchian Channel |
        | Oscillatorer | RSI, Stochastic, MACD, CCI, OBV |
        | Volum | Volumsøyler med glidende snitt |
        | Andre | VWAP, ATR, støtte/motstand-linjer |

        Kjøps- og exit-markører vises direkte på chart (valgfritt).

        **Utvidede analysemoduler (ekspanderbare):**
        - Smart Money Flow — CMF, OBV, divergensanalyse
        - Innsidehandel — siste handler, roller, beløp, score med markører på chart
        - Mønstergjenkjenning — 6 tekniske mønstre basert på prisdata

        **Posisjonskalkulator:**
        - Beregn posisjonsstørrelse basert på kapital og risikotoleranse
        - Automatisk stop-loss med ATR
        - Kelly Criterion for optimal innsats
        """)




    # === HANDELSSTRATEGIER ===
    st.markdown("---")
    st.markdown("## Handelsstrategier")

    st.markdown("""
    InveStock Pro bruker 7 tekniske strategier for å identifisere kjøpssignaler.
    Hver strategi har spesifikke kriterier og egner seg for ulike tidshorisonter.
    
    **Nytt:** Strength Pullback-strategien med optimaliserte trailing stops (1.5×ATR).
    """)

    st.markdown("""
    | Horisont | Strategi | Beskrivelse |
    | :--- | :--- | :--- |
    | Kort (1-5 dager) | RSI Mean Reversion | Pris over SMA 200, RSI under 30, var over 50 siste 10 dager. Fanger overreaksjoner i sterke trender. |
    | Kort (1-10 dager) | Momentum Burst | Breakout over 20-dagers høy med volum 2x+ snitt. Identifiserer eksplosive utbrudd. |
    | Kort-mellom (3-20 dager) | Strength Pullback | Opptrendaksje som trekker tilbake på tørt volum. Ønsker akkumulering før neste fase. |
    | Kort-mellom (3-20 dager) | Wyckoff Spring | Falskt brudd under støtte som lukker over. Klassisk akkumulasjonsmønster. |
    | Kort-mellom (5-20 dager) | Pocket Pivot | Volum på opp-dag overstiger høyeste volum på ned-dag. Tegn på institusjonell akkumulering. |
    | Mellom (2-8 uker) | VCP (Minervini) | Stage 2 opptrend med krympende volatilitet. Mark Minervinis signaturmønster. |
    | Lang (3-12 mnd) | Golden Cross | SMA 50 krysser over SMA 200 med stigende snitt. Klassisk langsiktig trendsignal. |
    """)

    # === EXIT-SIGNALER ===
    st.markdown("---")
    st.markdown("## Exit-signaler")

    st.markdown("""
    Exit-signaler hjelper med å identifisere når det kan være tid for å ta gevinst eller kutte tap.
    Et exit-signal utløses når **2 eller flere** av følgende kriterier er oppfylt samtidig:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Tekniske kriterier:**
        - **Death Cross** — SMA 50 krysser under SMA 200
        - **RSI reversal** — RSI faller fra overkjøpt nivå (>70)
        - **SMA 50 break** — Pris bryter under SMA 50 med høyt volum
        """)
    with col2:
        st.markdown("""
        **Risiko-kriterier:**
        - **MACD bearish** — MACD krysser under signallinjen
        - **Drawdown** — Ned mer enn 7 % fra 20-dagers topp
        - **Bollinger break** — Pris under nedre Bollinger Band
        """)

    st.info("Aktiver \"Vis exit-signaler\" i Teknisk Analyse for å se historiske exit-signaler på chart.")

    # === SIGNALKVALITET ===
    st.markdown("---")
    st.markdown("## Signalkvalitet (Score)")

    st.markdown("""
    Hver signal får en kvalitetscore basert på **realistisk backtesting**.
    Inkluderer +1 dag entry timing, transaksjonskostnader og likviditetsfilter.
    Scoren kombinerer win rate, profit factor og konsistens over tid.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### A")
        st.markdown("Score 60+\n\n**Høy kvalitet**\n\nHistorisk god treffsikkerhet")
    with col2:
        st.markdown("### B")
        st.markdown("Score 45-59\n\n**Middels-høy**\n\nGod, men noe variabel")
    with col3:
        st.markdown("### C")
        st.markdown("Score 35-44\n\n**Lav kvalitet**\n\nVær forsiktig")
    with col4:
        st.markdown("### D")
        st.markdown("Score <35\n\n**Unngå**\n\nHistorisk svak")

    # === INNSTILLINGER ===
    st.markdown("---")
    st.markdown("## Innstillinger")

    st.markdown("""
    **Tilgjengelige innstillinger i sidemenyen:**

    - **Oppdater Data** — Last ned ferske kursdata fra Yahoo Finance
    - **Min. dagsomsetning** — Filtrer bort illikvide aksjer (standard: 500 000 NOK)

    **Egendefinerte aksjer:**

    Opprett filen `mine_tickere.txt` i app-mappen med én ticker per linje for kun å analysere
    utvalgte aksjer. Standard-universet hentes fra `data_storage/tickers.txt`.
    ```
    EQNR.OL
    DNB.OL
    NHY.OL
    ```
    """)

    # === FOOTER ===
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>InveStock Pro v2.1 — Strømlinjeformet</p>
        <p><strong>Disclaimer:</strong> Dette er ikke investeringsråd. All handel innebærer risiko.
        Gjør alltid egen research og vurder å konsultere en finansiell rådgiver.</p>
    </div>
    """, unsafe_allow_html=True)
