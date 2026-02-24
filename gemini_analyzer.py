# gemini_analyzer.py
"""
Gemini Flash AI Analyzer for InveStock
Integrerer Google Gemini Flash for dyp markedsanalyse

NOTE: Using google.generativeai (deprecated) package.
Future versions should migrate to google.genai package.
"""

import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai

from log_config import get_logger

logger = get_logger(__name__)

# Gemini konfigurasjon
GEMINI_MODEL = "gemini-2.5-flash"

def _setup_gemini() -> tuple:
    """Setter opp Gemini API. Returns (success: bool, error_msg: str)."""
    try:
        api_key = None
        
        # Prøv Streamlit secrets først
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                logger.info("API-nøkkel funnet i Streamlit secrets")
        except Exception as e:
            logger.debug(f"Kunne ikke lese Streamlit secrets: {e}")
            
        # Falle tilbake til miljøvariabel
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                logger.info("API-nøkkel funnet i miljøvariabel")
            
        if not api_key:
            return False, "Ingen Gemini API-nøkkel funnet. Sjekk .streamlit/secrets.toml"
            
        # Konfigurer gamle API
        genai.configure(api_key=api_key)
        
        # Prøv først den foretrukne modellen
        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash", 
            "gemini-flash-latest",
            "gemini-pro-latest",
            "gemini-2.5-pro"
        ]
        
        for model_name in models_to_try:
            try:
                # Test API med enkelt spørsmål
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hei")
                
                if response and response.text:
                    # Oppdater global modell-variabel
                    global GEMINI_MODEL
                    GEMINI_MODEL = model_name
                    
                    logger.info(f"Gemini API OK med modell: {model_name}")
                    return True, None
                    
            except Exception as e:
                logger.debug(f"Modell {model_name} feilet: {e}")
                continue
        
        return False, "Ingen tilgjengelige Gemini-modeller funnet"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Feil ved oppsett av Gemini: {error_msg}")
        return False, f"API-feil: {error_msg}"

def samle_markedskontekst() -> Dict:
    """Samler all tilgjengelig markedsdata for AI-analyse."""
    try:
        kontekst = {
            "timestamp": datetime.now().isoformat(),
            "markedsregime": None,
            "markedsbredde": None,
            "makro_data": {},
            "sektor_styrke": {},
            "insider_aktivitet": {},
            "teknisk_status": {}
        }
        
        # Markedsregime (hvis tilgjengelig)
        try:
            import regime_model
            df_market = st.session_state.get('df_market', pd.DataFrame())
            if not df_market.empty:
                regime_data = regime_model.full_regime_analyse(df_market, n_regimes=3)
                if regime_data:
                    current = regime_data['current_info']
                    kontekst["markedsregime"] = {
                        "navn": current['name'],
                        "confidence": current.get('confidence', 'medium'),
                        "emoji": current['emoji'],
                        "beskrivelse": current.get('description', ''),
                        "anbefaling": current.get('action', '')
                    }
        except Exception as e:
            logger.debug(f"Kunne ikke hente regime-data: {e}")
            
        # Markedsbredde (Kritisk fiks for "0%"-feilen)
        # MERK: df_clean har IKKE SMA-kolonner! De må beregnes per ticker.
        try:
            df_clean = st.session_state.get('df_clean', pd.DataFrame())
            
            if not df_clean.empty and 'Ticker' in df_clean.columns:
                tickers = df_clean['Ticker'].unique()
                total = 0
                over_sma200 = 0
                over_sma50 = 0
                
                for ticker in tickers:
                    try:
                        df_t = df_clean[df_clean['Ticker'] == ticker].copy()
                        if len(df_t) < 200:
                            continue
                            
                        # Beregn SMA manuelt
                        df_t = df_t.sort_index()
                        sma_200 = df_t['Close'].rolling(window=200).mean().iloc[-1]
                        sma_50 = df_t['Close'].rolling(window=50).mean().iloc[-1]
                        last_close = df_t['Close'].iloc[-1]
                        
                        if pd.notna(sma_200):
                            total += 1
                            if last_close > sma_200:
                                over_sma200 += 1
                            if pd.notna(sma_50) and last_close > sma_50:
                                over_sma50 += 1
                    except Exception:
                        continue
                
                if total > 0:
                    sma200_pct = (over_sma200 / total) * 100
                    sma50_pct = (over_sma50 / total) * 100
                    kontekst["markedsbredde"] = {
                        "sma200_pct": round(sma200_pct, 1),
                        "sma50_pct": round(sma50_pct, 1),
                        "total_aksjer": total,
                        "trend_status": "Bullish" if sma200_pct > 60 else 
                                      "Bearish" if sma200_pct < 40 else "Nøytral"
                    }
                    logger.info(f"Bredde beregnet: {sma200_pct:.1f}% over SMA200 ({over_sma200}/{total} aksjer)")
                else:
                    logger.warning("Ingen aksjer med nok historikk for bredde-beregning")
        except Exception as e:
            logger.debug(f"Kunne ikke hente markedsbredde: {e}")
            
        # Makro-data (olje, valuta)
        try:
            import data
            # Brent olje
            brent_data = data.hent_brent_data()
            if not brent_data.empty:
                siste_brent = brent_data['Close'].iloc[-1]
                endring_1d = ((siste_brent / brent_data['Close'].iloc[-2]) - 1) * 100
                kontekst["makro_data"]["brent_oil"] = {
                    "pris": round(siste_brent, 2),
                    "endring_1d_pct": round(endring_1d, 2)
                }
                
            # USD/NOK
            usdnok_data = data.hent_usdnok_data()
            if not usdnok_data.empty:
                siste_nok = usdnok_data['Close'].iloc[-1]
                endring_1d = ((siste_nok / usdnok_data['Close'].iloc[-2]) - 1) * 100
                kontekst["makro_data"]["usd_nok"] = {
                    "kurs": round(siste_nok, 3),
                    "endring_1d_pct": round(endring_1d, 2)
                }
        except Exception as e:
            logger.debug(f"Kunne ikke hente makro-data: {e}")
            
        # Sektor-styrke (basert på scanner-data)
        try:
            sektor_performance = st.session_state.get('sektor_performance', {})
            if sektor_performance:
                kontekst["sektor_styrke"] = sektor_performance
        except Exception:
            pass
            
        return kontekst
        
    except Exception as e:
        logger.error(f"Feil ved samling av markedskontekst: {e}")
        return {}

def samle_scanner_resultater(scanner_df: pd.DataFrame, max_aksjer: int = 15) -> List[Dict]:
    """Konverterer scanner-resultater til AI-vennlig format."""
    try:
        if scanner_df.empty:
            return []
            
        # Ta de beste resultatene
        top_aksjer = scanner_df.head(max_aksjer)
        
        resultater = []
        for _, row in top_aksjer.iterrows():
            aksje_data = {
                "ticker": row.get('Ticker', ''),
                "selskap": row.get('Selskap', ''),
                "strategi": row.get('Strategi', ''),
                "total_score": row.get('Total', 0),
                "kvalitet": row.get('Kvalitet', ''),
                "rs_rating": row.get('RS', 0),
                "signal_dager": row.get('Dager', 0),
                "utvikling_pct": row.get('Utv%', 0),
                "peak_pct": row.get('Peak%', 0),
                "pris": row.get('Pris', ''),
                "sektor": row.get('Sektor', ''),
                "volum": row.get('Vol', ''),
                "fundamental_score": row.get('Fund.', ''),
                "smart_money": row.get('SM', ''),
                "insider": row.get('Ins.', ''),
                "exit_warning": "⚠️" in str(row.get('Exit⚠️', '')),
                "false_breakout": "❌" in str(row.get('FB', '')),
                "konvergens": row.get('Konv.', 0)
            }
            resultater.append(aksje_data)
            
        return resultater
        
    except Exception as e:
        logger.error(f"Feil ved konvertering av scanner-resultater: {e}")
        return []

def analyser_med_gemini(kontekst: Dict, scanner_resultater: List[Dict], 
                       analyse_type: str = "anbefalinger") -> Optional[str]:
    """Sender data til Gemini Flash for analyse."""
    try:
        success, _ = _setup_gemini()
        if not success:
            return None
            
        # Bygg omfattende prompt
        if kontekst is None:
            kontekst = {}
            
        regime_info = kontekst.get('markedsregime', {}) or {}
        bredde_info = kontekst.get('markedsbredde', {}) or {}
        makro_info = kontekst.get('makro_data', {}) or {}
        
        # Base system prompt
        system_prompt = """Du er en ekspert aksjeanalytiker som analyserer Oslo Børs. 
        Du får tilgang til omfattende markedsdata og skal gi konkrete, handlingsrettede anbefalinger.
        
        Svar alltid på NORSK og bruk emojis for å gjøre det engasjerende.
        Vær spesifikk og gi konkrete begrunnelser for dine anbefalinger.
        """
        
        if analyse_type == "anbefalinger":
            # Sikre verdier for bredde
            sma200 = bredde_info.get('sma200_pct', 0) if bredde_info else 0
            sma50 = bredde_info.get('sma50_pct', 0) if bredde_info else 0
            trend = bredde_info.get('trend_status', 'Ukjent') if bredde_info else 'Ukjent'
            
            prompt = f"""
            {system_prompt}
            
            ## MARKEDSSITUASJON:
            
            **Markedsregime:** {regime_info.get('navn', 'Ukjent') if regime_info else 'Ukjent'} {regime_info.get('emoji', '') if regime_info else ''}
            - Confidence: {regime_info.get('confidence', 'medium') if regime_info else 'medium'}
            - Anbefaling: {regime_info.get('anbefaling', '') if regime_info else ''}
            
            **Markedsbredde:**
            - {sma200:.1f}% av aksjer over SMA 200
            - {sma50:.1f}% av aksjer over SMA 50  
            - Trend: {trend}
            
            **Makroøkonomi:**
            """
            
            if makro_info and makro_info.get('brent_oil'):
                brent = makro_info['brent_oil']
                prompt += f"- Brent olje: ${brent['pris']} ({brent['endring_1d_pct']:+.1f}%)\n"
                
            if makro_info and makro_info.get('usd_nok'):
                nok = makro_info['usd_nok'] 
                prompt += f"- USD/NOK: {nok['kurs']} ({nok['endring_1d_pct']:+.1f}%)\n"
                
            prompt += f"""
            
            ## TOP SCANNER-RESULTATER:
            
            Her er de {len(scanner_resultater)} beste aksjene fra teknisk scanner:
            
            """
            
            for i, aksje in enumerate(scanner_resultater[:10], 1):
                prompt += f"""
                **{i}. {aksje['selskap']} ({aksje['ticker']})**
                - Strategi: {aksje['strategi']} 
                - Total Score: {aksje['total_score']}/100
                - Kvalitet: {aksje['kvalitet']} | RS Rating: {aksje['rs_rating']}/100
                - Signal: {aksje['signal_dager']} dager siden
                - Utvikling: {aksje['utvikling_pct']}% | Peak: {aksje['peak_pct']}%
                - Pris: {aksje['pris']} | Sektor: {aksje['sektor']}
                """
                
                if aksje['fundamental_score']:
                    prompt += f"- Fundamental Score: {aksje['fundamental_score']}/100\n"
                if aksje['smart_money']:
                    prompt += f"- Smart Money: {aksje['smart_money']}\n"
                if aksje['insider']:
                    prompt += f"- Insider Activity: {aksje['insider']}\n"
                if aksje['exit_warning']:
                    prompt += f"- ⚠️ EXIT WARNING\n"
                if aksje['false_breakout']:
                    prompt += f"- ❌ False Breakout Risk\n"
                    
                prompt += "\n"
                
            prompt += """
            
            ## OPPGAVE:
            
            Analyser denne situasjonen og gi meg:
            
            1. **🎯 TOP 3 KJØPSANBEFALINGER** 
               - Velg de 3 beste aksjene og forklar hvorfor
               - Gi konkrete inngangs- og stoppnivåer hvis mulig
               
            2. **⚠️ RISIKOFAKTORER**
               - Hva bør jeg være forsiktig med nå?
               - Sektorrisiko? Markedsrisiko?
               
            3. **📈 MARKEDSUTSIKTER** 
               - Hvordan ser det generelle markedsbildet ut?
               - Bullish/bearish bias for neste periode?
               
            4. **💡 HANDELSSTRATEGI**
               - Bør jeg være aggressiv eller konservativ nå?
               - Timing-råd basert på regime og bredde?
            
            Vær spesifikk, konkret og handlingsrettet!
            """
            
        # Bruk gamle API
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        if response and response.text:
            logger.info("Gemini-analyse fullført")
            return response.text
        else:
            st.session_state['gemini_error'] = "Tom respons fra Gemini API"
            logger.warning("Tom respons fra Gemini")
            return None
            
    except Exception as e:
        error_str = str(e)
        st.session_state['gemini_error'] = f"API-feil: {error_str}"
        logger.error(f"Feil ved Gemini-analyse: {e}")
        return None

def vis_ai_analyse_ui():
    """UI-komponent for AI-analyse i scanner."""
    st.markdown("---")
    
    # Header med info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 🤖 AI Markedsanalyst (Gemini Flash)")
        st.caption("Dyp AI-analyse av alle tilgjengelige markedsdata")
    
    with col2:
        if st.button("🚀 Analyser nå", type="primary"):
            st.session_state['run_ai_analysis'] = True
            st.rerun()
    
    # Kjør analyse hvis forespurt
    if st.session_state.get('run_ai_analysis', False):
        st.session_state['run_ai_analysis'] = False
        
        # Test API-tilgang først
        success, error_msg = _setup_gemini()
        if not success:
            st.error(f"❌ Gemini API-feil: {error_msg}")
            return
        
        with st.spinner("🧠 AI analyserer markedet... (dette kan ta 10-30 sekunder)"):
            # Samle kontekst
            kontekst = samle_markedskontekst()
            
            # Hent scanner-resultater
            scanner_df = st.session_state.get('scanner_resultater', pd.DataFrame())
            scanner_resultater = samle_scanner_resultater(scanner_df)
            
            if not scanner_resultater:
                st.warning("⚠️ Kjør scanner først for å få AI-analyse")
                return
                
            # Analyser med Gemini
            try:
                analyse = analyser_med_gemini(kontekst, scanner_resultater)
            except Exception as e:
                st.error(f"❌ Teknisk feil under analyse: {str(e)}")
                analyse = None
            
            if analyse:
                st.session_state['ai_analyse_resultat'] = analyse
                st.success("✅ AI-analyse fullført!")
            else:
                # Hvis ingen spesifikk feil ble vist over
                if 'gemini_error' in st.session_state:
                    st.error(f"❌ {st.session_state.gemini_error}")
                else:
                    st.error("❌ Kunne ikke fullføre AI-analyse. Sjekk om scanneren har resultater.")
    
    # Vis resultat
    if st.session_state.get('ai_analyse_resultat'):
        st.markdown("### 🎯 AI Analyse & Anbefalinger")
        
        # Vis analyse i expandable container
        with st.expander("📊 **Komplett AI-analyse**", expanded=True):
            st.markdown(st.session_state['ai_analyse_resultat'])
            
        # Clear knapp
        if st.button("🗑️ Fjern analyse"):
            if 'ai_analyse_resultat' in st.session_state:
                del st.session_state['ai_analyse_resultat']
            st.rerun()

# Test-funksjon
def test_gemini_setup():
    """Tester Gemini-oppsettet."""
    print("Tester Gemini Flash setup...")
    
    if _setup_gemini():
        print("✅ Gemini API OK")
        
        # Test enkelt spørsmål
        try:
            client = genai._client
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents="Hva er hovedstaden i Norge?"
            )
            print(f"Test-respons: {response.text}")
            print("✅ Gemini Flash fungerer!")
        except Exception as e:
            print(f"❌ Feil ved test: {e}")
    else:
        print("❌ Gemini API setup feilet")

if __name__ == "__main__":
    test_gemini_setup()