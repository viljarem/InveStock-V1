"""
Beta: Insider Signal — Finn bullish cases basert på meldepliktige handler.

Konsept:
  Insider-kjøp (spesielt fra CEO/CFO) er et av de sterkeste bullish-signalene
  fordi innsidere har best informasjon om selskapets fremtid. Denne modulen
  kombinerer insider-aktivitet med teknisk analyse for å finne nye bull-cases.

Visualisering:
  1. Live Feed        — Nyeste insider-handler med scoring
  2. Bull Case Radar  — Ranker aksjer etter insider+teknisk styrke
  3. Case Deep-Dive   — Detaljert analyse med prisutvikling etter kjøp
  4. Historisk track   — Hvor ofte har insider-kjøp ført til oppgang?
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logic
import data
import config

try:
    import insider_monitor
    INSIDER_OK = True
except ImportError:
    INSIDER_OK = False

try:
    from chart_utils import render_modern_chart, LWC_INSTALLED
    CHART_OK = LWC_INSTALLED
except ImportError:
    CHART_OK = False


# =========================================================================
# HJELPEFUNKSJONER
# =========================================================================

def _beregn_prisutvikling_etter_kjøp(df: pd.DataFrame, kjøpsdato: str, dager: list[int] = None) -> dict:
    """
    Beregner prisutvikling X dager etter en insider-kjøpsdato.
    
    Returns:
        dict med {5d, 10d, 20d, 40d, 60d} prisendring i prosent, eller None.
    """
    if dager is None:
        dager = [5, 10, 20, 40, 60]
    
    try:
        kjøps_ts = pd.Timestamp(kjøpsdato)
        if kjøps_ts not in df.index:
            # Finn nærmeste handelsdag
            mask = df.index >= kjøps_ts
            if not mask.any():
                return None
            kjøps_ts = df.index[mask][0]
        
        kjøps_idx = df.index.get_loc(kjøps_ts)
        kjøpspris = float(df.iloc[kjøps_idx]['Close'])
        
        result = {'kjøpspris': kjøpspris, 'kjøpsdato': kjøps_ts.strftime('%Y-%m-%d')}
        
        for d in dager:
            target_idx = kjøps_idx + d
            if target_idx < len(df):
                target_pris = float(df.iloc[target_idx]['Close'])
                endring = (target_pris - kjøpspris) / kjøpspris * 100
                result[f'{d}d'] = round(endring, 1)
            else:
                result[f'{d}d'] = None
        
        # Nåværende pris og endring
        nåpris = float(df.iloc[-1]['Close'])
        result['nåpris'] = nåpris
        result['total_endring'] = round((nåpris - kjøpspris) / kjøpspris * 100, 1)
        result['dager_siden'] = (df.index[-1] - kjøps_ts).days
        
        # Maks drawdown etter kjøp
        future_lows = df.iloc[kjøps_idx:]['Low']
        if len(future_lows) > 1:
            min_pris = float(future_lows.min())
            result['maks_drawdown'] = round((min_pris - kjøpspris) / kjøpspris * 100, 1)
        else:
            result['maks_drawdown'] = 0
        
        return result
    except Exception:
        return None


def _beregn_bull_case_score(insider_data: dict, teknisk_data: dict,
                            pris_etter_kjøp: dict = None) -> dict:
    """
    Scoring-modell som finner UBRUKTE muligheter der innsidere har kjøpt.

    Kjerneidé:
      Aksjer som IKKE har steget etter insider-kjøp scorer høyest — vinduet er åpent.
      Aksjer som allerede har steget kraftig scorer lavt — toget har gått.

    Scoring (0-100):
      - Insider-kvalitet:      35p  (hvem, hvor mye, klynge-kjøp)
      - Ubrukt potensial:      30p  (lite prisbevegelse etter kjøp = bra)
      - Teknisk inngangspunkt: 25p  (RSI, SMA, volum)
      - Risiko/Reward:         10p  (avstand fra 52w high)
    """
    score = 0
    komponenter = {}

    # --- 1. INSIDER-KVALITET (maks 35p) ---
    insider_score_raw = insider_data.get('score', 0)
    antall_kjøp = insider_data.get('antall_kjøp', 0)
    antall_salg = insider_data.get('antall_salg', 0)

    # Netto kjøp-signal (0-20p)
    if insider_score_raw > 0:
        insider_p = min(20, insider_score_raw / 100 * 20)
    else:
        insider_p = max(-10, insider_score_raw / 100 * 10)

    # Klynge-bonus: flere insider-kjøp = sterkere signal (0-10p)
    klynge_p = min(10, antall_kjøp * 3)
    if antall_salg > 0:
        klynge_p -= antall_salg * 5

    # Ferskhet: nylig kjøp gir bonus (0-5p)
    siste = insider_data.get('siste_handel')
    if siste:
        try:
            dager_siden_siste = (datetime.now() - datetime.strptime(siste, '%Y-%m-%d')).days
            ferskhet_p = max(0, 5 - dager_siden_siste / 6)
        except (ValueError, TypeError):
            ferskhet_p = 0
    else:
        ferskhet_p = 0

    insider_total = max(0, insider_p + klynge_p + ferskhet_p)
    komponenter['insider'] = round(min(35, insider_total), 1)
    score += komponenter['insider']

    # --- 2. UBRUKT POTENSIAL (maks 30p) ---
    # Nøkkelidé: jo MINDRE aksjen har steget siden insider-kjøp, desto BEDRE
    bevegelse = 0
    dager_siden_kjøp = 999

    if pris_etter_kjøp:
        bevegelse = pris_etter_kjøp.get('total_endring', 0) or 0
        dager_siden_kjøp = pris_etter_kjøp.get('dager_siden', 999) or 999

    # Bevegelsesscoring: PENALISER oppgang, BELØNN flat/negativ
    if bevegelse <= -5:
        potensial_p = 28      # Falt etter kjøp = ekstra billig inngang
    elif bevegelse <= 0:
        potensial_p = 30      # Flat/svak = vinduet er helt åpent
    elif bevegelse <= 3:
        potensial_p = 25      # Marginal oppgang, fortsatt godt vindu
    elif bevegelse <= 8:
        potensial_p = 18      # Noe bevegelse, OK timing
    elif bevegelse <= 15:
        potensial_p = 8       # Begynner å bli sent
    elif bevegelse <= 25:
        potensial_p = 3       # Mesteparten av bevegelsen er tatt
    else:
        potensial_p = 0       # >25% — toget har gått

    # Ferskhets-multiplikator
    if dager_siden_kjøp <= 5 and bevegelse <= 3:
        potensial_p = min(30, potensial_p + 5)  # Veldig ferskt og ubrukt
    elif dager_siden_kjøp > 60 and bevegelse <= 3:
        potensial_p = min(30, potensial_p + 3)  # Gammelt kjøp, markedet ignorerer signalet

    komponenter['potensial'] = round(potensial_p, 1)
    score += komponenter['potensial']

    # --- 3. TEKNISK INNGANGSPUNKT (maks 25p) ---
    rsi = teknisk_data.get('rsi', 50)
    pris_over_sma50 = teknisk_data.get('over_sma50', False)
    pris_over_sma200 = teknisk_data.get('over_sma200', False)
    volum_ratio = teknisk_data.get('volum_ratio', 1.0)

    # RSI: oversold = ideelt kjøpspunkt (0-10p)
    if 30 <= rsi <= 45:
        rsi_p = 10      # Sweet spot — oversolgt men ikke i fritt fall
    elif 25 <= rsi < 30:
        rsi_p = 8       # Veldig oversolgt, høy risiko men stort potensial
    elif 45 < rsi <= 55:
        rsi_p = 7       # Nøytral, OK inngang
    elif rsi < 25:
        rsi_p = 5       # Ekstremt oversolgt, kan signalisere problemer
    elif 55 < rsi <= 65:
        rsi_p = 4       # Moderat overkjøpt
    else:
        rsi_p = 1       # RSI > 65, allerede overkjøpt

    # Trend-kontekst (0-10p)
    trend_p = 0
    if pris_over_sma200:
        trend_p += 5
    if pris_over_sma50:
        trend_p += 5

    # Volum (0-5p)
    vol_p = min(5, max(0, (volum_ratio - 0.8) * 6))

    komponenter['teknisk'] = round(min(25, rsi_p + trend_p + vol_p), 1)
    score += komponenter['teknisk']

    # --- 4. RISIKO/REWARD (maks 10p) ---
    dist_52w_high = teknisk_data.get('dist_52w_high', 0)

    if dist_52w_high < -25:
        rr_p = 10       # Langt under 52w high, mye oppside
    elif dist_52w_high < -15:
        rr_p = 8
    elif dist_52w_high < -8:
        rr_p = 5
    elif dist_52w_high < -3:
        rr_p = 3
    else:
        rr_p = 1        # Nær/over 52w high, begrenset oppside

    komponenter['risiko_reward'] = round(rr_p, 1)
    score += komponenter['risiko_reward']

    # --- TOTAL SCORE & STATUS ---
    total = round(min(100, max(0, score)), 1)

    # Status: kombinerer score OG faktisk bevegelse for tydelig anbefaling
    if bevegelse > 20:
        status = "FOR SENT"
        status_tekst = f"Allerede +{bevegelse:.0f}% — toget har gått"
        farge = "#f44336"
    elif bevegelse > 12:
        status = "SEN INNGANG"
        status_tekst = f"Opp {bevegelse:.0f}% — mye er priset inn"
        farge = "#ff9800"
    elif total >= 60 and bevegelse <= 5:
        status = "KJØP"
        status_tekst = "Sterkt insider-signal, vinduet er åpent"
        farge = "#00c853"
    elif total >= 45 and bevegelse <= 8:
        status = "INTERESSANT"
        status_tekst = "Godt signal, akseptabel timing"
        farge = "#26a69a"
    elif total >= 30:
        status = "OVERVÅK"
        status_tekst = "Signal registrert, vent på bedre inngang"
        farge = "#ffd600"
    else:
        status = "SVAK"
        status_tekst = "Svakt signal eller dårlig timing"
        farge = "#888888"

    return {
        'score': total,
        'komponenter': komponenter,
        'status': status,
        'status_tekst': status_tekst,
        'farge': farge,
        'bevegelse_siden_kjøp': round(bevegelse, 1),
        'dager_siden_kjøp': dager_siden_kjøp,
    }


def _hent_teknisk_snapshot(df: pd.DataFrame) -> dict:
    """Henter teknisk snapshot for en ticker."""
    if df.empty or len(df) < 50:
        return {}
    
    try:
        close = df['Close']
        nåpris = float(close.iloc[-1])
        
        snapshot = {
            'nåpris': nåpris,
            'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else 50,
            'over_sma50': nåpris > float(df['SMA_50'].iloc[-1]) if 'SMA_50' in df.columns else False,
            'over_sma200': nåpris > float(df['SMA_200'].iloc[-1]) if 'SMA_200' in df.columns else False,
            'volum_ratio': float(df['Volume'].iloc[-5:].mean() / df['Volume'].iloc[-30:].mean()) if len(df) >= 30 else 1.0,
            'endring_5d': float(close.pct_change(5).iloc[-1] * 100) if len(df) > 5 else 0,
            'endring_20d': float(close.pct_change(20).iloc[-1] * 100) if len(df) > 20 else 0,
        }
        
        # 52w high/low
        if len(df) >= 252:
            high_52w = float(df['High'].iloc[-252:].max())
            snapshot['dist_52w_high'] = (nåpris - high_52w) / high_52w * 100
        else:
            snapshot['dist_52w_high'] = 0
        
        return snapshot
    except Exception:
        return {}


# =========================================================================
# POPUP DIALOG
# =========================================================================

@st.dialog("Insider Case", width="large")
def _vis_case_popup():
    """Viser kursdiagram med insider-markeringer i en popup."""
    case = st.session_state.get('_insider_popup')
    ticker_handler = st.session_state.get('_insider_popup_handler', [])

    if not case:
        st.warning("Ingen case valgt.")
        return

    ticker = case['Ticker']
    bc = case['_bull_case']
    farge = case['_farge']

    # Header med status
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="background: {farge}; color: #000; padding: 4px 14px;
                     border-radius: 8px; font-size: 14px; font-weight: 800;">{bc['status']}</span>
        <span style="font-size: 22px; font-weight: 700; color: #fff;">{case['Selskap']}</span>
        <span style="color: #888;">{ticker}</span>
        <span style="margin-left: auto; font-size: 20px; font-weight: 700; color: {farge};">
            {bc['score']:.0f}<span style="font-size: 12px; color: #888;">/100</span>
        </span>
    </div>
    <div style="color: #aaa; font-size: 13px; margin-bottom: 16px;">{case['_status_tekst']}</div>
    """, unsafe_allow_html=True)

    # Chart
    df_clean = st.session_state.get('df_clean', pd.DataFrame())
    df_valgt = df_clean[df_clean['Ticker'] == ticker].copy()

    if not df_valgt.empty and len(df_valgt) >= 30 and CHART_OK:
        df_chart = df_valgt.iloc[-130:].copy()

        if 'SMA_50' not in df_chart.columns and len(df_valgt) >= 50:
            df_chart['SMA_50'] = df_valgt['Close'].rolling(50).mean().iloc[-130:]

        indikatorer = {}
        if 'SMA_50' in df_chart.columns:
            sma_data = df_chart['SMA_50'].dropna()
            if not sma_data.empty:
                indikatorer['SMA 50'] = {
                    'name': 'SMA 50',
                    'data': sma_data,
                    'color': '#ffd600',
                }

        render_modern_chart(
            df_chart,
            indicators=indikatorer,
            chart_height=350,
            insider_trades=ticker_handler,
        )
    elif not CHART_OK:
        st.caption("Chart ikke tilgjengelig (mangler streamlit-lightweight-charts)")

    # Handler-liste
    if ticker_handler:
        st.markdown("**Insider-handler:**")
        for h in ticker_handler:
            type_ikon = "🟢" if h.get('type') == 'kjøp' else "🔴" if h.get('type') == 'salg' else "⚪"
            st.markdown(f"- {type_ikon} **{h.get('dato', '')}** — {h.get('tittel', '')}")

    # Nøkkeltall
    komp = bc['komponenter']
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Insider", f"{komp.get('insider', 0):.0f}/35")
    k2.metric("Ubrukt potensial", f"{komp.get('potensial', 0):.0f}/30")
    k3.metric("Teknisk", f"{komp.get('teknisk', 0):.0f}/25")
    k4.metric("R/R", f"{komp.get('risiko_reward', 0):.0f}/10")

    # Knapp for TA
    if st.button("Åpne i Teknisk Analyse", type="primary", key="popup_ta_btn"):
        st.session_state['valgt_ticker'] = ticker
        st.session_state['navigate_to'] = "Teknisk Analyse"
        st.rerun()


# =========================================================================
# RENDER
# =========================================================================

def render():
    df_clean = st.session_state.get('df_clean', pd.DataFrame())
    unike_tickers = st.session_state.get('unike_tickers', [])
    
    if not INSIDER_OK:
        st.error("Insider Monitor er ikke tilgjengelig. Sjekk at insider_monitor.py finnes.")
        return
    
    # === HEADER ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #0a2647 50%, #144272 100%);
                border-radius: 20px; padding: 30px; margin-bottom: 24px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <div style="font-size: 28px; font-weight: 700; color: #ffffff;
                    margin-bottom: 8px; letter-spacing: -0.5px;">
            Insider Signal
        </div>
        <div style="font-size: 15px; color: rgba(255,255,255,0.7);">
            Finner aksjer der innsidere har kjøpt, men prisen ennå ikke har reagert.
            Jo mindre bevegelse etter insider-kjøp, desto bedre — da er vinduet fortsatt åpent.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === HENT DATA ===
    with st.spinner("Henter insider-handler fra Newsweb..."):
        handler = insider_monitor.hent_innsidehandler(dager=90)
    
    if not handler:
        st.warning("Ingen insider-handler funnet. API kan være midlertidig utilgjengelig.")
        return
    
    # Filtrer til kun kjøp
    kjøp_handler = [h for h in handler if h.get('type') == 'kjøp']
    salg_handler = [h for h in handler if h.get('type') == 'salg']
    
    # Finn unike tickers med aktivitet
    aktive_tickers = sorted(set(h['ticker'] for h in handler if h.get('ticker')))
    
    # === OPPSUMMERING ===
    kol1, kol2, kol3, kol4 = st.columns(4)
    kol1.metric("Handler totalt", len(handler))
    kol2.metric("Kjøp", len(kjøp_handler))
    kol3.metric("Salg", len(salg_handler))
    kol4.metric("Selskaper", len(aktive_tickers))
    
    # === TABS ===
    tab_radar, tab_feed, tab_deep_dive, tab_historikk = st.tabs([
        "Bull Case Radar", "Live Feed", "Case Deep-Dive", "Historisk Track Record"
    ])
    
    # =====================================================================
    # TAB 1: BULL CASE RADAR — Hovedvisningen
    # =====================================================================
    with tab_radar:
        st.markdown("##### Aksjer med insider-kjøp — rangert etter ubrukt potensial")
        
        # Bygg bull-case tabell
        radar_rows = []
        progress = st.progress(0, text="Analyserer insider-cases...")
        
        for i, ticker in enumerate(aktive_tickers):
            progress.progress((i + 1) / len(aktive_tickers), text=f"Analyserer {ticker}...")
            
            # Insider-scoring
            insider_data = insider_monitor.beregn_insider_score(ticker, handler)
            
            # Hopp over tickers uten kjøp
            if insider_data['antall_kjøp'] == 0:
                continue
            
            # Teknisk data
            df_ticker = df_clean[df_clean['Ticker'] == ticker].copy()
            if df_ticker.empty or len(df_ticker) < 50:
                continue
            
            # Beregn indikatorer hvis ikke allerede gjort
            if 'RSI' not in df_ticker.columns:
                try:
                    df_ticker = logic.beregn_tekniske_indikatorer(df_ticker)
                except Exception:
                    continue
            
            teknisk = _hent_teknisk_snapshot(df_ticker)
            if not teknisk:
                continue
            
            # Prisutvikling etter siste kjøp
            siste_kjøp = insider_data.get('siste_handel')
            pris_etter = None
            if siste_kjøp:
                pris_etter = _beregn_prisutvikling_etter_kjøp(df_ticker, siste_kjøp)
            
            # Bull case score
            bull_case = _beregn_bull_case_score(insider_data, teknisk, pris_etter)
            
            # Finn handler-detaljer
            ticker_kjøp = [h for h in handler if h['ticker'] == ticker and h['type'] == 'kjøp']
            
            # Selskap-navn
            selskap = data.ticker_til_navn(ticker) if hasattr(data, 'ticker_til_navn') else ticker.replace('.OL', '')
            
            bev = bull_case['bevegelse_siden_kjøp']
            dager_s = bull_case['dager_siden_kjøp']

            row = {
                'Ticker': ticker,
                'Selskap': selskap,
                'Status': bull_case['status'],
                'Score': bull_case['score'],
                'Bevegelse': f"{bev:+.1f}%",
                'Dager': dager_s if dager_s < 900 else '—',
                'Kjøp': insider_data['antall_kjøp'],
                'Salg': insider_data['antall_salg'],
                'Siste kjøp': siste_kjøp or '—',
                'Pris': f"{teknisk.get('nåpris', 0):.2f}",
                'RSI': f"{teknisk.get('rsi', 50):.0f}",
                '_farge': bull_case['farge'],
                '_status_tekst': bull_case['status_tekst'],
                '_bull_case': bull_case,
                '_insider_data': insider_data,
                '_teknisk': teknisk,
                '_pris_etter': pris_etter,
            }
            radar_rows.append(row)
        
        progress.empty()
        
        if not radar_rows:
            st.info("Ingen insider-kjøp funnet i analyseperioden.")
        else:
            # Sorter etter score
            radar_rows.sort(key=lambda x: x['Score'], reverse=True)
            
            # === KJØP-ANBEFALINGER (tydelig status) ===
            kjøp_cases = [r for r in radar_rows if r['Status'] in ('KJØP', 'INTERESSANT')]
            for_sent_cases = [r for r in radar_rows if r['Status'] in ('FOR SENT', 'SEN INNGANG')]

            if kjøp_cases:
                st.markdown(f"##### {len(kjøp_cases)} aktive muligheter")

                for case in kjøp_cases[:5]:
                    farge = case['_farge']
                    bc = case['_bull_case']
                    komp = bc['komponenter']
                    status = case['Status']
                    bev_farge = '#00c853' if bc['bevegelse_siden_kjøp'] <= 3 else '#ffd600' if bc['bevegelse_siden_kjøp'] <= 8 else '#ff9800'

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {farge}15 0%, {farge}05 100%);
                                border-left: 4px solid {farge}; border-radius: 12px;
                                padding: 20px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <span style="background: {farge}; color: #000; padding: 4px 14px;
                                             border-radius: 8px; font-size: 13px; font-weight: 800;
                                             letter-spacing: 0.5px;">{status}</span>
                                <span style="font-size: 20px; font-weight: 700; color: #fff;">
                                    {case['Selskap']}
                                </span>
                                <span style="color: #888;">{case['Ticker']}</span>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 24px; font-weight: 700; color: {farge};">
                                    {case['Score']:.0f}
                                </span>
                                <span style="font-size: 11px; color: #888;">/100</span>
                            </div>
                        </div>
                        <div style="margin-top: 8px; color: #aaa; font-size: 13px; font-style: italic;">
                            {case['_status_tekst']}
                        </div>
                        <div style="margin-top: 14px; display: flex; gap: 28px; flex-wrap: wrap;">
                            <div>
                                <div style="font-size: 10px; color: #666; text-transform: uppercase;
                                            letter-spacing: 0.5px;">Bevegelse siden kjøp</div>
                                <div style="font-size: 18px; font-weight: 700; color: {bev_farge};">
                                    {case['Bevegelse']}
                                </div>
                            </div>
                            <div>
                                <div style="font-size: 10px; color: #666; text-transform: uppercase;
                                            letter-spacing: 0.5px;">Dager siden kjøp</div>
                                <div style="font-size: 18px; font-weight: 700; color: #fff;">
                                    {case['Dager']}
                                </div>
                            </div>
                            <div>
                                <div style="font-size: 10px; color: #666; text-transform: uppercase;
                                            letter-spacing: 0.5px;">Insider-kjøp</div>
                                <div style="font-size: 18px; font-weight: 700; color: #fff;">
                                    {case['Kjøp']}
                                </div>
                            </div>
                            <div>
                                <div style="font-size: 10px; color: #666; text-transform: uppercase;
                                            letter-spacing: 0.5px;">RSI</div>
                                <div style="font-size: 18px; font-weight: 700; color: #fff;">
                                    {case['RSI']}
                                </div>
                            </div>
                            <div>
                                <div style="font-size: 10px; color: #666; text-transform: uppercase;
                                            letter-spacing: 0.5px;">Pris</div>
                                <div style="font-size: 18px; font-weight: 700; color: #fff;">
                                    {case['Pris']}
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 10px; display: flex; gap: 8px;">
                            <span style="background: rgba(255,255,255,0.08); padding: 3px 10px;
                                         border-radius: 12px; font-size: 11px; color: #aaa;">
                                Insider: {komp.get('insider', 0):.0f}/35
                            </span>
                            <span style="background: rgba(255,255,255,0.08); padding: 3px 10px;
                                         border-radius: 12px; font-size: 11px; color: #aaa;">
                                Potensial: {komp.get('potensial', 0):.0f}/30
                            </span>
                            <span style="background: rgba(255,255,255,0.08); padding: 3px 10px;
                                         border-radius: 12px; font-size: 11px; color: #aaa;">
                                Teknisk: {komp.get('teknisk', 0):.0f}/25
                            </span>
                            <span style="background: rgba(255,255,255,0.08); padding: 3px 10px;
                                         border-radius: 12px; font-size: 11px; color: #aaa;">
                                R/R: {komp.get('risiko_reward', 0):.0f}/10
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Ingen tydelige kjøpsmuligheter akkurat nå. Alle insider-kjøp har allerede blitt priset inn.")

            # Advarsel om aksjer der toget har gått
            if for_sent_cases:
                with st.expander(f"Toget har gått ({len(for_sent_cases)} aksjer som allerede har steget kraftig)", expanded=False):
                    for case in for_sent_cases:
                        st.markdown(f"- **{case['Selskap']}** ({case['Ticker']}): {case['Bevegelse']} siden insider-kjøp — {case['_status_tekst']}")
            
            # === FULL TABELL ===
            st.markdown("##### Alle cases")
            vis_kolonner = ['Status', 'Ticker', 'Selskap', 'Score', 'Bevegelse',
                           'Dager', 'Kjøp', 'Siste kjøp', 'Pris', 'RSI']
            
            tabell_df = pd.DataFrame(radar_rows)[vis_kolonner]
            
            def _farge_score(val):
                try:
                    v = float(val)
                    if v >= 60: return 'color: #00c853; font-weight: 700'
                    if v >= 45: return 'color: #26a69a; font-weight: 600'
                    if v >= 30: return 'color: #ffd600'
                    return 'color: #888'
                except (ValueError, TypeError):
                    return ''
            
            def _farge_endring(val):
                try:
                    if val == '—': return 'color: #666'
                    v = float(val.replace('%', '').replace('+', ''))
                    if v > 0: return 'color: #00c853'
                    if v < 0: return 'color: #ff5252'
                    return ''
                except (ValueError, TypeError):
                    return ''

            def _farge_status(val):
                farger = {
                    'KJØP': 'color: #00c853; font-weight: 800',
                    'INTERESSANT': 'color: #26a69a; font-weight: 700',
                    'OVERVÅK': 'color: #ffd600; font-weight: 600',
                    'SEN INNGANG': 'color: #ff9800; font-weight: 600',
                    'FOR SENT': 'color: #f44336; font-weight: 700',
                    'SVAK': 'color: #888',
                }
                return farger.get(val, '')
            
            styled = tabell_df.style \
                .map(_farge_status, subset=['Status']) \
                .map(_farge_score, subset=['Score']) \
                .map(_farge_endring, subset=['Bevegelse'])
            
            event = st.dataframe(
                styled,
                hide_index=True,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="insider_radar_table"
            )
            
            # Detaljer ved valg — åpne popup
            selected = event.selection.rows if event.selection else []
            if selected:
                valgt = radar_rows[selected[0]]
                # Lagre valgt case i session_state og trigger dialog
                st.session_state['_insider_popup'] = valgt
                st.session_state['_insider_popup_handler'] = [h for h in handler if h['ticker'] == valgt['Ticker']]
    
    # =====================================================================
    # TAB 2: LIVE FEED — Kronologisk visning
    # =====================================================================
    with tab_feed:
        st.markdown("##### Nyeste meldepliktige handler")
        
        # Filter
        feed_col1, feed_col2 = st.columns([1, 1])
        with feed_col1:
            vis_type = st.selectbox("Type", ["Alle", "Kun kjøp", "Kun salg"], key="feed_type")
        with feed_col2:
            feed_dager = st.slider("Periode (dager)", 7, 90, 30, key="feed_dager")
        
        # Filtrer
        cutoff = (datetime.now() - timedelta(days=feed_dager)).strftime('%Y-%m-%d')
        filtrert = [h for h in handler if h.get('dato', '') >= cutoff]
        
        if vis_type == "Kun kjøp":
            filtrert = [h for h in filtrert if h.get('type') == 'kjøp']
        elif vis_type == "Kun salg":
            filtrert = [h for h in filtrert if h.get('type') == 'salg']
        
        # Sorter nyeste først
        filtrert.sort(key=lambda x: x.get('dato_tid', ''), reverse=True)
        
        if not filtrert:
            st.info("Ingen handler i valgt periode.")
        else:
            # Vis som tidslinje
            for h in filtrert:
                type_farge = '#00c853' if h.get('type') == 'kjøp' else '#ff5252' if h.get('type') == 'salg' else '#888'
                type_label = h.get('type', 'ukjent').upper()
                
                st.markdown(f"""
                <div style="display: flex; align-items: flex-start; gap: 16px;
                            padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <div style="min-width: 80px; text-align: right;">
                        <div style="font-size: 13px; font-weight: 600; color: #ccc;">{h.get('dato', '')}</div>
                    </div>
                    <div style="width: 3px; background: {type_farge}; border-radius: 2px; min-height: 40px;"></div>
                    <div style="flex: 1;">
                        <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 4px;">
                            <span style="background: {type_farge}20; color: {type_farge}; 
                                         padding: 2px 8px; border-radius: 6px; font-size: 11px;
                                         font-weight: 600;">{type_label}</span>
                            <span style="font-weight: 600; color: #fff;">{h.get('issuer_name', h.get('ticker', ''))}</span>
                            <span style="color: #666; font-size: 12px;">{h.get('ticker', '')}</span>
                        </div>
                        <div style="color: #aaa; font-size: 13px;">{h.get('tittel', '')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # =====================================================================
    # TAB 3: CASE DEEP-DIVE — Detaljert analyse av én case
    # =====================================================================
    with tab_deep_dive:
        st.markdown("##### Velg en aksje for detaljert insider-analyse")
        
        # Filtrer til tickers med kjøp
        kjøp_tickers = sorted(set(h['ticker'] for h in kjøp_handler if h.get('ticker')))
        
        if not kjøp_tickers:
            st.info("Ingen insider-kjøp funnet i perioden.")
        else:
            valgt_deep = st.selectbox("Aksje", kjøp_tickers, key="deep_dive_ticker")
            
            if valgt_deep:
                # Hent data
                df_ticker = df_clean[df_clean['Ticker'] == valgt_deep].copy()
                insider_data = insider_monitor.beregn_insider_score(valgt_deep, handler)
                ticker_handler = [h for h in handler if h['ticker'] == valgt_deep]
                
                if df_ticker.empty or len(df_ticker) < 50:
                    st.warning(f"Ikke nok prisdata for {valgt_deep}")
                else:
                    if 'RSI' not in df_ticker.columns:
                        try:
                            df_ticker = logic.beregn_tekniske_indikatorer(df_ticker)
                        except Exception:
                            pass
                    
                    teknisk = _hent_teknisk_snapshot(df_ticker)

                    siste_kjøp_deep = insider_data.get('siste_handel')
                    pris_etter_deep = None
                    if siste_kjøp_deep:
                        pris_etter_deep = _beregn_prisutvikling_etter_kjøp(df_ticker, siste_kjøp_deep)

                    bull_case = _beregn_bull_case_score(insider_data, teknisk, pris_etter_deep)
                    
                    # Score + status-visning
                    farge = bull_case['farge']
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
                        <div style="font-size: 48px; font-weight: 700; color: {farge};">{bull_case['score']:.0f}</div>
                        <div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="background: {farge}; color: #000; padding: 3px 12px;
                                             border-radius: 6px; font-size: 13px; font-weight: 800;">
                                    {bull_case['status']}
                                </span>
                            </div>
                            <div style="color: #aaa; font-size: 13px; margin-top: 4px;">{bull_case['status_tekst']}</div>
                            <div style="color: #666; font-size: 12px;">Bevegelse siden kjøp: {bull_case['bevegelse_siden_kjøp']:+.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Score-komponent bar
                    komp = bull_case['komponenter']
                    total = bull_case['score']
                    
                    bar_items = [
                        ('Insider', komp.get('insider', 0), 35, '#667eea'),
                        ('Ubrukt potensial', komp.get('potensial', 0), 30, '#00c853'),
                        ('Teknisk', komp.get('teknisk', 0), 25, '#26a69a'),
                        ('R/R', komp.get('risiko_reward', 0), 10, '#e040fb'),
                    ]
                    
                    for navn, verdi, maks, bar_farge in bar_items:
                        pct = (verdi / maks * 100) if maks > 0 else 0
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                            <div style="min-width: 70px; font-size: 12px; color: #aaa; text-align: right;">{navn}</div>
                            <div style="flex: 1; background: rgba(255,255,255,0.05); border-radius: 4px; height: 20px; overflow: hidden;">
                                <div style="width: {pct:.0f}%; background: {bar_farge}; height: 100%; border-radius: 4px;
                                            display: flex; align-items: center; justify-content: flex-end; padding-right: 6px;">
                                    <span style="font-size: 10px; color: #fff; font-weight: 600;">{verdi:.0f}/{maks}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Handler-tidslinje
                    st.markdown("**Insider-handler for denne aksjen:**")
                    for h in ticker_handler:
                        type_farge = '#00c853' if h['type'] == 'kjøp' else '#ff5252'
                        type_label = h['type'].upper()
                        
                        # Prisutvikling etter denne handelen
                        pris_etter = _beregn_prisutvikling_etter_kjøp(df_ticker, h['dato'])
                        endring_str = ""
                        if pris_etter and pris_etter.get('total_endring') is not None:
                            e = pris_etter['total_endring']
                            e_farge = '#00c853' if e >= 0 else '#ff5252'
                            endring_str = f'<span style="color: {e_farge}; font-weight: 600; margin-left: 12px;">{e:+.1f}% siden</span>'
                        
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 12px;
                                    padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <span style="background: {type_farge}20; color: {type_farge}; 
                                         padding: 2px 8px; border-radius: 6px; font-size: 11px;
                                         font-weight: 600; min-width: 50px; text-align: center;">{type_label}</span>
                            <span style="color: #ccc; min-width: 80px;">{h['dato']}</span>
                            <span style="color: #aaa; flex: 1; font-size: 13px;">{h['tittel']}</span>
                            {endring_str}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Navigering
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Åpne i Teknisk Analyse", type="primary", key="deep_ta"):
                        st.session_state['valgt_ticker'] = valgt_deep
                        st.session_state['navigate_to'] = "Teknisk Analyse"
                        st.rerun()
    
    # =====================================================================
    # TAB 4: HISTORISK TRACK RECORD
    # =====================================================================
    with tab_historikk:
        st.markdown("##### Historisk treffsikkerhet for insider-kjøp")
        st.caption("Måler prisutvikling etter alle insider-kjøp i dataperioden.")
        
        # Analyser alle kjøp-handler
        track_rows = []
        for h in kjøp_handler:
            ticker = h.get('ticker', '')
            if not ticker:
                continue
            
            df_ticker = df_clean[df_clean['Ticker'] == ticker].copy()
            if df_ticker.empty or len(df_ticker) < 50:
                continue
            
            pris_etter = _beregn_prisutvikling_etter_kjøp(df_ticker, h['dato'])
            if pris_etter is None:
                continue
            
            selskap = data.ticker_til_navn(ticker) if hasattr(data, 'ticker_til_navn') else ticker.replace('.OL', '')
            
            track_rows.append({
                'Dato': h['dato'],
                'Selskap': selskap,
                'Ticker': ticker,
                'Melding': h.get('tittel', '')[:60],
                '5d': pris_etter.get('5d'),
                '10d': pris_etter.get('10d'),
                '20d': pris_etter.get('20d'),
                '40d': pris_etter.get('40d'),
                'Drawdown': pris_etter.get('maks_drawdown'),
            })
        
        if not track_rows:
            st.info("Ikke nok data for historisk analyse.")
        else:
            track_df = pd.DataFrame(track_rows)
            
            # Statistikk-oppsummering
            for periode in ['5d', '10d', '20d', '40d']:
                verdier = track_df[periode].dropna()
                if len(verdier) == 0:
                    continue
            
            stat_cols = st.columns(4)
            for i, periode in enumerate(['5d', '10d', '20d', '40d']):
                verdier = track_df[periode].dropna()
                if len(verdier) == 0:
                    continue
                positiv_pct = (verdier > 0).mean() * 100
                snitt = verdier.mean()
                
                farge = '#00c853' if positiv_pct >= 55 else '#ff9800' if positiv_pct >= 45 else '#ff5252'
                
                with stat_cols[i]:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border-radius: 12px;
                                padding: 16px; text-align: center; border: 1px solid rgba(255,255,255,0.08);">
                        <div style="font-size: 12px; color: #888; text-transform: uppercase;">Etter {periode.replace('d', ' dager')}</div>
                        <div style="font-size: 28px; font-weight: 700; color: {farge};">{positiv_pct:.0f}%</div>
                        <div style="font-size: 11px; color: #aaa;">positive ({len(verdier)} cases)</div>
                        <div style="font-size: 14px; color: {'#00c853' if snitt > 0 else '#ff5252'}; margin-top: 4px;">
                            Snitt: {snitt:+.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Full tabell
            st.markdown("<br>", unsafe_allow_html=True)
            
            def _farge_pct(val):
                try:
                    if val is None or pd.isna(val):
                        return 'color: #555'
                    if float(val) > 0:
                        return 'color: #00c853'
                    return 'color: #ff5252'
                except (ValueError, TypeError):
                    return ''
            
            vis_kol = ['Dato', 'Selskap', 'Ticker', '5d', '10d', '20d', '40d', 'Drawdown']
            vis_df = track_df[vis_kol].copy()
            
            # Formater tall
            for k in ['5d', '10d', '20d', '40d', 'Drawdown']:
                vis_df[k] = vis_df[k].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else '—')
            
            st.dataframe(
                vis_df.style.map(_farge_pct, subset=['5d', '10d', '20d', '40d', 'Drawdown']),
                hide_index=True,
                use_container_width=True,
            )

    # Popup utenfor tabs — @st.dialog må kalles fra hoved-script-kroppen
    if st.session_state.get('_insider_popup'):
        _vis_case_popup()
