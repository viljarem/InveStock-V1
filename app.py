# Deaktiver MPS for PyTorch (forhindrer segfault med YOLO på Apple Silicon)
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_MPS_DISABLE'] = '1'

# Undertrykk Streamlit threading warning (harmløs)
import logging
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import sys
from PIL import Image, ImageDraw
import pathlib

# === SETT PAGE CONFIG - MÅ VÆRE FØRST ===
# Prøv å laste lagret ikon, ellers bruk emoji
icon_path = pathlib.Path(__file__).parent / ".streamlit" / "favicon.png"
page_icon = "📈"  # Fallback emoji

if icon_path.exists():
    try:
        page_icon = Image.open(icon_path)
    except:
        pass

st.set_page_config(
    page_title="InveStock",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Generer og lagre favicon hvis det ikke finnes
def _create_and_save_favicon():
    """Generer og lagre InveStock app-ikon"""
    icon_dir = pathlib.Path(__file__).parent / ".streamlit"
    icon_dir.mkdir(exist_ok=True)
    icon_path = icon_dir / "favicon.png"
    
    if icon_path.exists():
        return
    
    size = 64
    img = Image.new('RGBA', (size, size), (14, 17, 23, 255))
    draw = ImageDraw.Draw(img)
    
    # Ramme
    draw.rectangle([(0, 0), (size-1, size-1)], fill=(14, 17, 23), outline=(38, 166, 154, 200), width=2)
    
    # Stilisert chart (grønn opptrend)
    chart_data = [(10, 45), (18, 32), (26, 22), (34, 26), (42, 14), (50, 8)]
    draw.line(chart_data, fill=(38, 166, 154, 255), width=3)
    
    for x, y in chart_data:
        draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(38, 166, 154, 255))
    
    img.save(icon_path, 'PNG')

_create_and_save_favicon()

# Importer egne moduler
import data
import config
import logic
from styles import get_base_css, get_liquid_glass_css
from shared_cache import cached_hent_data, cached_hent_markedsdata_df

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

# --- PORTEFØLJE MODUL INTEGRASJON ---
portfolio_tilgjengelig = False
try:
    import portfolio
    portfolio_tilgjengelig = True
except ImportError:
    pass

# === Sidefiler ===
from views import hjem, markedstemperatur, scanner, teknisk_analyse
from views import beta_ai_scanner, portefolje, innstillinger

# Beta-moduler som krever ekstra avhengigheter
try:
    from views import beta_insider_signal
    insider_signal_tilgjengelig = True
except ImportError:
    insider_signal_tilgjengelig = False


# Profesjonell styling (CSS)
st.markdown(get_base_css(), unsafe_allow_html=True)
st.markdown(get_liquid_glass_css(), unsafe_allow_html=True)


# ===================================================================
# Initialisering av Session State
# ===================================================================
if 'df' not in st.session_state:
    st.session_state['df'] = cached_hent_data()
if 'df_market' not in st.session_state:
    st.session_state['df_market'] = cached_hent_markedsdata_df()
if 'teknisk_cache' not in st.session_state:
    st.session_state['teknisk_cache'] = {}
if 'signal_cache' not in st.session_state:
    st.session_state['signal_cache'] = {}
if 'backtest_cache' not in st.session_state:
    st.session_state['backtest_cache'] = {}


# ===================================================================
# SIDEBAR NAVIGASJON
# ===================================================================
with st.sidebar:
    st.title("InveStock Pro")

    # Dynamisk menyvalg basert på tilgjengelighet av moduler
    meny_opsjoner = ["Hjem", "Markedstemperatur", "Scanner", "Teknisk Analyse"]
    if portfolio_tilgjengelig and getattr(config, "ENABLE_PORTFOLIO_PAGE", True):
        meny_opsjoner.insert(3, "Portefølje")
    if beta_tilgjengelig:
        meny_opsjoner.append("Beta: AI Scanner")
    if insider_signal_tilgjengelig:
        meny_opsjoner.append("Beta: Insider Signal")
    meny_opsjoner.append("Innstillinger")

    # Håndter navigasjon fra andre sider
    if 'navigate_to' in st.session_state:
        nav_target = st.session_state.pop('navigate_to')
        if nav_target in meny_opsjoner:
            st.session_state['side_valg_radio'] = nav_target

    # Sett default hvis ikke satt
    if 'side_valg_radio' not in st.session_state:
        st.session_state['side_valg_radio'] = meny_opsjoner[0]

    side_valg = st.radio("Meny", meny_opsjoner, key="side_valg_radio")

    # Utvidet feilsøking for Beta-modul
    if beta_eksisterer and not beta_tilgjengelig:
        st.sidebar.error(f"Beta-fil funnet, men kunne ikke lastes:\n\n`{beta_feilmelding}`")
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

    # Vis sist oppdateringstid
    last_update = st.session_state.get('last_update') or data.hent_siste_oppdateringstid()
    if last_update:
        st.caption(f"Sist oppdatert: {last_update.strftime('%Y-%m-%d %H:%M')}")

    st.markdown("---")
    if st.button("Oppdater Data"):
        progress = st.progress(0, text="Starter oppdatering...")
        with st.spinner("Synkroniserer markedet..."):
            progress.progress(0.2, text="Laster ned aksjedata...")
            data.last_ned_data()

            progress.progress(0.5, text="Oppdaterer markedsdata...")
            data.hent_markedsdata_df(force_refresh=True)

            progress.progress(0.7, text="Oppdaterer sektor-mapping...")
            logic.oppdater_sektor_cache()

            progress.progress(0.8, text="Rydder cache...")
            st.cache_data.clear()
            st.session_state['df'] = data.hent_data()
            st.session_state['df_market'] = data.hent_markedsdata_df()
            st.session_state['teknisk_cache'] = {}
            st.session_state['signal_cache'] = {}
            st.session_state['backtest_cache'] = {}

            # Oppdater timestamp
            st.session_state['last_update'] = data.hent_siste_oppdateringstid() or pd.Timestamp.now()
            progress.progress(1.0, text="Fullført.")
        st.success("Data ble oppdatert")

    # AI TEST CHAT
    st.markdown("---")
    with st.expander("🤖 AI Test Chat"):
        try:
            from gemini_analyzer import _setup_gemini, GEMINI_MODEL, samle_markedskontekst
            import google.generativeai as genai
            import json
            
            # Initialiser chat historie hvis den ikke eksisterer
            if 'chat_historie' not in st.session_state:
                st.session_state.chat_historie = []
            
            # Input felt for chat
            user_input = st.text_input("Test AI:", placeholder="Spør om markedet eller test AI...")
            inklude_kontekst = st.checkbox("Inkluder markedskontekst", value=True)
            
            if st.button("Send", key="ai_test_send") and user_input:
                with st.spinner("AI tenker..."):
                    try:
                        # Setup Gemini - sjekk at det fungerer
                        success, error_msg = _setup_gemini()
                        
                        if not success:
                            st.error(f"Kunne ikke koble til AI: {error_msg}")
                        else:
                            # Hent kontekst hvis valgt
                            marked_data = ""
                            if inklude_kontekst:
                                kontekst = samle_markedskontekst()
                                
                                # Legg til scanner-resultater hvis tilgjengelig
                                scanner_df = st.session_state.get('scanner_resultater', pd.DataFrame())
                                if not scanner_df.empty:
                                    # Ta topp 10 aksjer fra scanneren
                                    top_aksjer = []
                                    for _, row in scanner_df.head(10).iterrows():
                                        aksje = {
                                            "ticker": str(row.get('Ticker', '')),
                                            "selskap": str(row.get('Selskap', '')),
                                            "strategi": str(row.get('Strategi', '')),
                                            "total_score": float(row.get('Total', 0)) if pd.notna(row.get('Total')) else 0,
                                            "kvalitet": str(row.get('Kvalitet', '')),
                                            "rs_rating": float(row.get('RS', 0)) if pd.notna(row.get('RS')) else 0,
                                            "pris": str(row.get('Pris', '')),
                                            "sektor": str(row.get('Sektor', '')),
                                            "utvikling_pct": float(row.get('Utv%', 0)) if pd.notna(row.get('Utv%')) else 0,
                                        }
                                        top_aksjer.append(aksje)
                                    kontekst["scanner_topp_aksjer"] = top_aksjer
                                
                                # Sjekk om vi kan hente metadata for en spesifikk ticker hvis nevnt
                                import re
                                ticker_match = re.search(r'\b([A-Z]{2,5})\b', user_input.upper())
                                if ticker_match:
                                    ticker = ticker_match.group(1)
                                    df = st.session_state.get('df_clean', st.session_state.get('df', pd.DataFrame()))
                                    if not df.empty and 'Ticker' in df.columns:
                                        ticker_data = df[df['Ticker'] == ticker]
                                        if not ticker_data.empty:
                                            # Beregn tekniske indikatorer for denne tickeren
                                            try:
                                                df_t = ticker_data.copy().sort_index()
                                                if len(df_t) >= 50:
                                                    last_close = df_t['Close'].iloc[-1]
                                                    sma50 = df_t['Close'].rolling(50).mean().iloc[-1]
                                                    sma200 = df_t['Close'].rolling(200).mean().iloc[-1] if len(df_t) >= 200 else None
                                                    rsi = None
                                                    
                                                    # Beregn RSI
                                                    delta = df_t['Close'].diff()
                                                    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
                                                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
                                                    if loss != 0:
                                                        rsi = 100 - (100 / (1 + gain/loss))
                                                    
                                                    kontekst[f"ticker_{ticker}"] = {
                                                        "pris": round(last_close, 2),
                                                        "sma50": round(sma50, 2) if pd.notna(sma50) else None,
                                                        "sma200": round(sma200, 2) if sma200 and pd.notna(sma200) else None,
                                                        "rsi": round(rsi, 1) if rsi and pd.notna(rsi) else None,
                                                        "over_sma50": last_close > sma50 if pd.notna(sma50) else None,
                                                        "over_sma200": last_close > sma200 if sma200 and pd.notna(sma200) else None,
                                                        "volum_siste": int(df_t['Volume'].iloc[-1]),
                                                        "volum_snitt": int(df_t['Volume'].rolling(20).mean().iloc[-1]),
                                                    }
                                            except Exception as e:
                                                pass

                                if kontekst:
                                    marked_data = f"\n\nNåværende markedskontekst og teknisk metadata:\n{json.dumps(kontekst, indent=2, ensure_ascii=False, default=str)}"
                            
                            # Nå kan vi bruke API-en
                            model = genai.GenerativeModel(GEMINI_MODEL)
                            
                            # Enkel prompt for testing
                            prompt = f"""Du er en norsk finansassistent integrert i InveStock-appen. 
                            Svar på brukerens spørsmål basert på tilgjengelig data hvis relevant.
                            {marked_data}
                            
                            Brukers spørsmål: {user_input}"""
                        
                            # Få svar från AI
                            response = model.generate_content(prompt)
                            
                            if response and response.text:
                                respons_tekst = response.text
                            else:
                                respons_tekst = "Ingen respons mottatt från AI."
                        
                            # Legg til i chat historie
                            st.session_state.chat_historie.append({"user": user_input, "ai": respons_tekst})
                            
                            # Success melding
                            st.success(f"AI svarte! (Model: {GEMINI_MODEL})")
                            
                            # Tøm input og refresh
                            st.rerun()
                        
                    except Exception as e:
                        st.error(f"AI-feil: {str(e)}")
                        st.info("Sjekk at API-nøkkelen er riktig satt i .streamlit/secrets.toml")
            
            # Vis chat historie (siste 3 meldinger)
            if st.session_state.chat_historie:
                st.markdown("**Siste samtaler:**")
                for i, chat in enumerate(st.session_state.chat_historie[-3:]):
                    st.markdown(f"**Du:** {chat['user']}")
                    st.markdown(f"**AI:** {chat['ai']}")
                    if i < len(st.session_state.chat_historie[-3:]) - 1:
                        st.markdown("---")
                
                if st.button("Tøm chat", key="clear_chat"):
                    st.session_state.chat_historie = []
                    st.rerun()
        
        except ImportError as e:
            st.error("Gemini AI er ikke tilgjengelig. Sjekk at alle avhengigheter er installert.")
            st.code(f"ImportError: {e}")
        except Exception as e:
            st.error(f"Feil ved oppsett av AI chat: {e}")
    
    
    
    st.markdown("---")
    min_volum = st.number_input("Min. dagsomsetning (NOK)", value=500000, step=100000)


# ===================================================================
# DATA LASTING & FILTRERING
# ===================================================================
df_raw = st.session_state['df']
if df_raw.empty:
    st.error("Ingen data funnet i lagring. Vennligst kjør oppdatering fra menyen.")
    st.stop()

# Ekskluderer illikvide aksjer basert på brukerens input
df_clean = data.filtrer_likvide_aksjer(df_raw, min_dagsomsetning=min_volum)
unike_tickers = sorted(df_clean['Ticker'].unique())

if len(unike_tickers) == 0:
    st.warning("Ingen aksjer matcher valgt minimumsomsetning. Senk filteret i sidebar.")
    st.stop()

# Lagre i session_state for sidefiler
st.session_state['df_clean'] = df_clean
st.session_state['unike_tickers'] = unike_tickers


# ===================================================================
# SIDE-DISPATCH
# ===================================================================
if side_valg == "Hjem":
    hjem.render()

elif side_valg == "Markedstemperatur":
    markedstemperatur.render()

elif side_valg == "Scanner":
    scanner.render()

elif side_valg == "Teknisk Analyse":
    teknisk_analyse.render()

elif side_valg == "Portefølje":
    portefolje.render()

elif side_valg == "Beta: AI Scanner":
    beta_ai_scanner.render()

elif side_valg == "Beta: Insider Signal":
    beta_insider_signal.render()

elif side_valg == "Innstillinger":
    innstillinger.render()
